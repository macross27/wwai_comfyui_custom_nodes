"""WWAI's Ripple charge against WepeNerd's real H3 Ripple From Frames (plan 2431).

WWAI charges a Ripple job ``window_bytes`` (backend/core/ripple_job.py): the window
WWAIExposeRippleJob decodes (``window_decoded_bytes``) plus what WN_RippleFromFrames keeps
beside it. This drives the real chain on a real movie for every ``gpu_check`` row of
ripple_cases.json: WWAIExposeRippleJob.expose, then WepeNerd's execute with exactly its
outputs, and checks the tensors From Frames holds (the padded canvas window, the marked
frames, the sound clone) come to ``bytes - window_decoded_bytes``.

WepeNerd is importable only beside a real ComfyUI, so this skips anywhere else. On the
GPU machine:

    PYTHONPATH=<ComfyUI dir> WEPENERD_DIR=<ComfyUI dir>/custom_nodes/ComfyUI-WepeNerd \\
        <ComfyUI venv>/bin/python -m unittest test_wepenerd_ripple_parity -v

Nothing here is installed or written outside a temp directory.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

from test_nodes_contract import _movie_stamps, _ripple_movie

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEPENERD = os.environ.get("WEPENERD_DIR", os.path.join(os.path.dirname(_HERE), "ComfyUI-WepeNerd"))
_IN = None


def _annotated(name):
    return os.path.join(_IN, name)


def _load():
    """(the pack's nodes module, WepeNerd's ripple_node) with ComfyUI's model and server
    code replaced by stand-ins, as WepeNerd's own tests/test_ripple_node.py loads it;
    sys.modules is left as it was found, so test_nodes_contract's stubbed import still runs."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_annotated_filepath = _annotated
    folder_paths.get_input_directory = lambda: _IN
    stand_ins = {
        "folder_paths": folder_paths,
        "comfy.model_management": types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None),
        # None makes each import an ImportError: the pack's optional imports fall back.
        "server": None,
        "nodes": None,
        "comfy_extras.nodes_sam3": None,
        "comfy_extras.nodes_save_3d": None,
        "comfy_extras.nodes_images": None,
    }
    package = os.path.basename(_HERE)
    saved = {name: sys.modules.get(name) for name in (*stand_ins, "ripple_core", "load_video_node")}
    sys.modules.update(stand_ins)
    try:
        with mock.patch.object(sys, "path", [_WEPENERD, os.path.dirname(_HERE), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "wepenerd_ripple_parity", os.path.join(_WEPENERD, "ripple_node.py")
            )
            ripple = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ripple)
            nodes = importlib.import_module(f"{package}.nodes")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
            del sys.modules[name]
    return nodes, ripple


try:
    nodes, ripple = _load()
    _SKIP = None
except (ImportError, FileNotFoundError) as exc:  # no ComfyUI / WepeNerd on this machine
    nodes = ripple = None
    _SKIP = f"WepeNerd's ripple_node cannot be imported from {_WEPENERD} ({exc})"


def setUpModule():
    global _IN
    _IN = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(_IN, ignore_errors=True)


def _stage(write, ext):
    scratch = os.path.join(_IN, "scratch." + ext)
    write(scratch)
    with open(scratch, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    os.makedirs(os.path.join(_IN, "wwai"), exist_ok=True)
    name = f"wwai/wwai-{digest}.{ext}"
    os.replace(scratch, os.path.join(_IN, name))
    return name


@unittest.skipUnless(_SKIP is None, _SKIP or "")
class WepeNerdHoldsWhatWwaiChargesTest(unittest.TestCase):
    def test_from_frames_holds_the_charge_beyond_the_decoded_window(self):
        with open(os.path.join(_HERE, "ripple_cases.json"), encoding="utf-8") as handle:
            rows = [row for row in json.load(handle)["window_bytes"] if row["gpu_check"]]
        self.assertTrue(any(row["audio_rate"] and row["edits"] >= 2 for row in rows))
        for row in rows:
            with self.subTest(row["name"]):
                self._check(row)

    def _check(self, row):
        self.assertIn(row["audio_rate"], (0, 8000))  # _ripple_movie writes 8 kHz mono
        video = _stage(lambda path: _ripple_movie(path, audio_from=0.1 if row["audio_rate"] else None), "mov")
        time_base, pts, _end, _first = _movie_stamps(os.path.join(_IN, video))
        edit = _stage(lambda path: Image.new("RGB", (4, 4), (9, 9, 9)).save(path), "png")
        start = 3  # a 24 fps movie: window frame k is grid frame 3 + k, the packet at index 3 + k
        job = {
            "version": 2, "video": video, "time_base": f"{time_base.numerator}/{time_base.denominator}",
            "window_start": start, "length": row["length"], "decode_width": row["decode_width"],
            "decode_height": row["decode_height"], "megapixels": row["megapixels"],
            "allow_upscale": row["allow_upscale"], "resize_method": "lanczos",
            "output_width": row["output_width"], "output_height": row["output_height"],
            "markers": [{"frame": k, "pts": pts[start + k], "edited": edit} for k in range(row["edits"])],
        }
        images, audio, rate, duration, mp, window_start, markers, upscale, method, width, height = (
            nodes.WWAIExposeRippleJob().expose(value=json.dumps(job), name="job", description="", required=True)
        )
        decoded = images.nbytes + (audio["waveform"].nbytes if audio is not None else 0)
        self.assertEqual(decoded, row["window_decoded_bytes"])

        out = ripple.WN_RippleFromFrames.execute(
            images, rate, duration, mp, window_start, markers, audio=audio, allow_upscale=upscale,
            resize_method=method, output_width=width, output_height=height,
        )
        selected, context, window, sound = out.result[:4]
        self.assertIs(context["window"], window)
        self.assertEqual(len(window), row["length"])
        self.assertEqual((window.shape[1] % 32, window.shape[2] % 32), (0, 0))  # the padded canvas
        self.assertEqual(len(selected), row["edits"])
        # Each is its own memory, not a view of the input or of the window.
        self.assertNotEqual(selected.untyped_storage().data_ptr(), window.untyped_storage().data_ptr())
        held = window.nbytes + selected.nbytes
        if row["audio_rate"]:
            self.assertNotEqual(
                sound["waveform"].untyped_storage().data_ptr(), audio["waveform"].untyped_storage().data_ptr()
            )
            held += sound["waveform"].nbytes
        else:
            self.assertIsNone(sound)
        self.assertEqual(held, row["bytes"] - row["window_decoded_bytes"])


if __name__ == "__main__":
    unittest.main()
