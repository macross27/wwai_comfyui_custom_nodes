"""Contract tests for the marker nodes, run without ComfyUI installed.

ComfyUI is not importable in CI or on a dev box, so this module installs
minimal stand-ins for the four things `nodes.py` imports from its host
(`torch`, `folder_paths`, `comfy_api.latest`, `comfy_extras.nodes_save_3d`)
and then exercises the real node code against them.

What it guards is the class of mistake that shipped in the first version of
this pack and was only caught by review: a socket declared with the wrong type
string, an output marker that ComfyUI prunes because it used the newer
`is_output_node` spelling, and a marker that silently accepts a nameless or
multi-artifact result. None of those fail at import — they fail quietly, in a
running ComfyUI, as a plug that will not connect or a result that never
arrives.

Real-ComfyUI verification is still required for registration and socket
compatibility; this is the regression net underneath it.
"""

import inspect
import os
import shutil
import sys
import tempfile
import types
import unittest
from enum import Enum
from fractions import Fraction

import numpy as np
from PIL import Image

_TMP = None
_IN = None
_OUT = None
_REAL_MODULES = {}


class _Tensor(np.ndarray):
    """The slice of the torch tensor surface nodes.py actually touches."""

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def unsqueeze(self, axis):
        return _Tensor(np.expand_dims(self, axis))

    def clamp(self, lo, hi):
        return _Tensor(np.clip(self, lo, hi))


def _torch_stub():
    mod = types.ModuleType("torch")
    mod.float32 = np.float32
    mod.from_numpy = _Tensor
    mod.zeros = lambda shape, dtype=None: _Tensor(np.zeros(shape, dtype=dtype or np.float32))
    return mod


def _folder_paths_stub():
    mod = types.ModuleType("folder_paths")
    image_ext = (".png", ".jpg", ".jpeg", ".webp")
    video_ext = (".mp4", ".mov", ".mkv", ".webm")
    counter = {"n": 1}

    def filter_files_content_types(files, content_types):
        exts = ()
        if "image" in content_types:
            exts += image_ext
        if "video" in content_types:
            exts += video_ext
        return [f for f in files if f.lower().endswith(exts)]

    def get_save_image_path(filename_prefix, output_dir, width=0, height=0):
        subfolder = os.path.dirname(filename_prefix)
        full = os.path.join(output_dir, subfolder)
        os.makedirs(full, exist_ok=True)
        n = counter["n"]
        counter["n"] += 1
        return full, os.path.basename(filename_prefix), n, subfolder, filename_prefix

    mod.get_input_directory = lambda: _IN
    mod.get_output_directory = lambda: _OUT
    mod.filter_files_content_types = filter_files_content_types
    def filter_files_content_types_with_audio(files, content_types):
        exts = ()
        if "image" in content_types:
            exts += image_ext
        if "video" in content_types:
            exts += video_ext
        if "audio" in content_types:
            exts += (".flac", ".mp3", ".wav", ".opus")
        return [f for f in files if f.lower().endswith(exts)]

    mod.filter_files_content_types = filter_files_content_types_with_audio
    mod.get_annotated_filepath = lambda name: os.path.join(_IN, name)
    mod.exists_annotated_filepath = lambda name: os.path.isfile(os.path.join(_IN, name))
    mod.get_save_image_path = get_save_image_path
    return mod


class _VideoCodec(str, Enum):
    AUTO = "auto"


class _VideoContainer(str, Enum):
    AUTO = "auto"
    MP4 = "mp4"

    @classmethod
    def get_extension(cls, value):
        return "mp4" if cls(value) in (cls.MP4, cls.AUTO) else ""


class _File3D:
    """Mirrors comfy_api/latest/_util/geometry_types.py:56."""

    def __init__(self, source, file_format=""):
        self._source = source
        self._format = file_format or (
            os.path.splitext(source)[1].lstrip(".").lower() if isinstance(source, str) else ""
        )

    @property
    def format(self):
        return self._format

    def get_source(self):
        return self._source

    def get_bytes(self):
        with open(self._source, "rb") as handle:
            return handle.read()


class _VideoFromFile:
    def __init__(self, path):
        self.path = path

    def get_components(self):
        return _VideoComponents(
            images=_Tensor(np.zeros((5, 8, 8, 3), dtype=np.float32)),
            audio={"waveform": _Tensor(np.zeros((1, 2, 16), dtype=np.float32)), "sample_rate": 44100},
            frame_rate=Fraction(24, 1),
        )

    def save_to(self, path, format=None, codec=None):
        with open(path, "wb") as handle:
            handle.write(b"video-bytes")
        return path


class _VideoComponents:
    def __init__(self, images, audio, frame_rate):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate


class _VideoFromComponents(_VideoFromFile):
    def __init__(self, components, bit_depth=8):
        self.components = components
        self.path = None


class _FolderType(str, Enum):
    output = "output"


class _AudioSaveHelper:
    """Mirrors comfy_api/latest/_ui.py:260 — one saved file per batch item."""

    @staticmethod
    def save_audio(audio, filename_prefix, folder_type, cls=None, format="flac", quality="128k"):
        subfolder = os.path.dirname(filename_prefix)
        full = os.path.join(_OUT, subfolder)
        os.makedirs(full, exist_ok=True)
        results = []
        for index in range(audio["waveform"].shape[0]):
            file = f"{os.path.basename(filename_prefix)}_{index:05}.{format}"
            with open(os.path.join(full, file), "wb") as handle:
                handle.write(b"fLaC-bytes")
            results.append({"filename": file, "subfolder": subfolder, "type": folder_type.value})
        return results


def _comfy_api_stub():
    latest = types.ModuleType("comfy_api.latest")
    latest.Types = type("Types", (), {
        "File3D": _File3D, "VideoCodec": _VideoCodec, "VideoContainer": _VideoContainer,
        "VideoComponents": _VideoComponents,
    })
    latest.InputImpl = type("InputImpl", (), {
        "VideoFromFile": _VideoFromFile, "VideoFromComponents": _VideoFromComponents,
    })
    latest.IO = type("IO", (), {"FolderType": _FolderType})
    latest.UI = type("UI", (), {"AudioSaveHelper": _AudioSaveHelper})
    parent = types.ModuleType("comfy_api")
    parent.latest = latest
    return parent, latest


def _torchaudio_stub():
    mod = types.ModuleType("torchaudio")
    mod.load = lambda path: (_Tensor(np.zeros((2, 16), dtype=np.float32)), 44100)
    return mod


def _save_3d_stub():
    mod = types.ModuleType("comfy_extras.nodes_save_3d")
    mod.get_mesh_batch_item = lambda mesh, index: ("v", "f", "c", "uv")

    def save_glb(vertices, faces, filepath, metadata=None, **kwargs):
        with open(filepath, "wb") as handle:
            handle.write(b"glTF\x02\x00\x00\x00")
        return filepath

    mod.save_glb = save_glb
    parent = types.ModuleType("comfy_extras")
    parent.nodes_save_3d = mod
    return parent, mod


def setUpModule():
    global _TMP, _IN, _OUT, nodes, MarkerContractError
    _TMP = tempfile.mkdtemp()
    _IN = os.path.join(_TMP, "input")
    _OUT = os.path.join(_TMP, "output")
    os.makedirs(_IN)
    os.makedirs(_OUT)

    comfy_api, comfy_api_latest = _comfy_api_stub()
    comfy_extras, save_3d = _save_3d_stub()
    stubs = {
        "torch": _torch_stub(),
        "torchaudio": _torchaudio_stub(),
        "folder_paths": _folder_paths_stub(),
        "comfy_api": comfy_api,
        "comfy_api.latest": comfy_api_latest,
        "comfy_extras": comfy_extras,
        "comfy_extras.nodes_save_3d": save_3d,
    }
    for name, module in stubs.items():
        _REAL_MODULES[name] = sys.modules.get(name)
        sys.modules[name] = module

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import importlib

    package = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    nodes = importlib.import_module(f"{package}.nodes")
    MarkerContractError = importlib.import_module(
        f"{package}.wwai_markers_core"
    ).MarkerContractError

    Image.new("RGBA", (8, 6), (10, 20, 30, 255)).save(os.path.join(_IN, "shot.png"))
    with open(os.path.join(_IN, "clip.mp4"), "wb") as handle:
        handle.write(b"fake")
    with open(os.path.join(_IN, "track.flac"), "wb") as handle:
        handle.write(b"fLaC")
    with open(os.path.join(_IN, "splat.ply"), "wb") as handle:
        handle.write(_splat_ply())


def tearDownModule():
    for name, module in _REAL_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    shutil.rmtree(_TMP, ignore_errors=True)


def _splat_ply():
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float f_dc_0\nproperty float opacity\n"
        b"property float scale_0\nproperty float rot_0\nend_header\n"
    )
    return header + bytes(range(256)) * 2 + b"\x00\xff\xfe\x80"


def _written(entry):
    return os.path.join(_OUT, entry["subfolder"], entry["filename"])


ALL_NODES = {
    "WWAIExposeText", "WWAIExposeInt", "WWAIExposeFloat", "WWAIExposeBool", "WWAIExposeImage",
    "WWAIExposeAudio", "WWAIExposeVideo", "WWAIExposeMesh",
    "WWAIExposeOutputImage", "WWAIExposeOutputAudio",
    "WWAIExposeOutputVideo", "WWAIExposeOutputMesh",
}
OUTPUT_NODES = {
    "WWAIExposeOutputImage", "WWAIExposeOutputAudio",
    "WWAIExposeOutputVideo", "WWAIExposeOutputMesh",
}


def _audio(channels=2, batch=1):
    return {"waveform": _Tensor(np.zeros((batch, channels, 16), dtype=np.float32)), "sample_rate": 44100}


class RegistrationTest(unittest.TestCase):
    def test_every_node_registers_with_a_display_name(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), ALL_NODES)
        self.assertEqual(set(nodes.NODE_DISPLAY_NAME_MAPPINGS), ALL_NODES)

    def test_every_node_declares_name_and_description_widgets(self):
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            required = cls.INPUT_TYPES()["required"]
            self.assertIn("name", required, name)
            self.assertIn("description", required, name)

    def test_every_input_marker_declares_a_required_toggle(self):
        for name in ALL_NODES - OUTPUT_NODES:
            widget = nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"]["required"]
            self.assertEqual(widget[0], "BOOLEAN", name)
            self.assertIs(widget[1]["default"], True, name)

    def test_no_output_marker_declares_a_required_toggle(self):
        for name in OUTPUT_NODES:
            self.assertNotIn("required", nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"])

    def test_output_markers_use_the_legacy_output_node_flag(self):
        # `is_output_node=True` belongs to the newer schema API and has NO
        # effect here — a terminal marker spelled that way is pruned from the
        # prompt and never reaches /history.
        for name in OUTPUT_NODES:
            cls = nodes.NODE_CLASS_MAPPINGS[name]
            self.assertIs(getattr(cls, "OUTPUT_NODE", False), True, name)
            self.assertFalse(hasattr(cls, "is_output_node"), name)


class SignatureContractTest(unittest.TestCase):
    """A widget added to INPUT_TYPES()["required"] must reach every hook
    ComfyUI calls with the node's declared inputs -- this is the test that
    plan 1961 needed and did not have, when `required_widget()` shipped
    without updating IS_CHANGED/VALIDATE_INPUTS on FileMarkerMixin.
    """

    def test_is_changed_accepts_every_required_widget(self):
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            if not hasattr(cls, "IS_CHANGED"):
                continue
            accepted = set(inspect.signature(cls.IS_CHANGED).parameters)
            required = set(cls.INPUT_TYPES()["required"])
            missing = required - accepted
            self.assertFalse(missing, f"{name}.IS_CHANGED does not accept {missing}")

    def test_expose_accepts_every_required_widget(self):
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            accepted = set(inspect.signature(cls.expose).parameters) - {"self"}
            required = set(cls.INPUT_TYPES()["required"])
            missing = required - accepted
            self.assertFalse(missing, f"{name}.expose does not accept {missing}")


class SocketTypeTest(unittest.TestCase):
    def test_the_input_key_table_wwai_rewrites_against(self):
        for name in ("WWAIExposeImage", "WWAIExposeAudio", "WWAIExposeVideo", "WWAIExposeMesh"):
            self.assertIn("file", nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"], name)
        for name in ("WWAIExposeText", "WWAIExposeInt", "WWAIExposeFloat", "WWAIExposeBool"):
            self.assertIn("value", nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"], name)

    def test_video_input_serves_both_video_worlds(self):
        # VHS_LoadVideo returns (IMAGE, frame_count, AUDIO, VHS_VIDEOINFO) --
        # there is no VIDEO type in that world. Emitting only VIDEO would make
        # this marker unusable as the entry point of a VHS workflow.
        self.assertEqual(nodes.WWAIExposeVideo.RETURN_TYPES, ("VIDEO", "IMAGE", "AUDIO", "FLOAT"))
        self.assertEqual(nodes.WWAIExposeVideo.RETURN_NAMES, ("video", "images", "audio", "fps"))

    def test_video_output_accepts_a_finished_video(self):
        self.assertEqual(
            nodes.WWAIExposeOutputVideo.INPUT_TYPES()["optional"]["video"][0], "VIDEO"
        )

    def test_video_output_also_serves_the_video_helper_suite_world(self):
        # VHS has no VIDEO type: it carries IMAGE frames plus a separate AUDIO
        # track. Without these ports this marker cannot end a VHS workflow.
        optional = nodes.WWAIExposeOutputVideo.INPUT_TYPES()["optional"]
        self.assertEqual(optional["images"][0], "IMAGE")
        self.assertEqual(optional["audio"][0], "AUDIO")
        self.assertEqual(optional["fps"][1]["default"], 30.0)

    def test_audio_sockets_are_comfyui_native_audio(self):
        self.assertEqual(nodes.WWAIExposeAudio.RETURN_TYPES, ("AUDIO",))
        self.assertEqual(nodes.WWAIExposeOutputAudio.INPUT_TYPES()["required"]["value"][0], "AUDIO")

    def test_mesh_input_emits_what_load3d_emits(self):
        self.assertEqual(nodes.WWAIExposeMesh.RETURN_TYPES, ("FILE_3D",))

    def test_mesh_output_accepts_every_3d_type_comfyui_has(self):
        # ComfyUI matches sockets by exact string and has fourteen 3D types.
        # Declaring one would refuse most real 3D producers.
        declared = nodes.WWAIExposeOutputMesh.INPUT_TYPES()["required"]["value"][0].split(",")
        self.assertEqual(len(declared), 14)
        for needed in ("MESH", "FILE_3D", "FILE_3D_GLB", "FILE_3D_PLY", "FILE_3D_SPLAT_ANY"):
            self.assertIn(needed, declared)


class InputMarkerTest(unittest.TestCase):
    def test_image_marker_loads_real_pixels(self):
        image, mask = nodes.WWAIExposeImage().expose(file="shot.png", name="ref", description="", required=True)
        self.assertEqual(image.shape, (1, 6, 8, 3))
        self.assertEqual(mask.shape, (1, 6, 8))

    def test_bool_marker_returns_the_value_unchanged(self):
        self.assertEqual(
            nodes.WWAIExposeBool().expose(value=True, name="flag", description="", required=False),
            (True,),
        )

    def test_video_marker_opens_the_file_and_splits_it(self):
        video, images, audio, fps = nodes.WWAIExposeVideo().expose(
            file="clip.mp4", name="src", description="", required=True
        )
        self.assertTrue(video.path.endswith("clip.mp4"))
        self.assertEqual(images.shape, (5, 8, 8, 3))
        self.assertEqual(audio["sample_rate"], 44100)
        self.assertEqual(fps, 24.0)
        self.assertIsInstance(fps, float)

    def test_video_marker_round_trips_into_the_frames_output_path(self):
        # The whole point of splitting: a VHS-shaped workflow can take the
        # artist's video in and put a finished one back out with sound.
        _, images, audio, fps = nodes.WWAIExposeVideo().expose(
            file="clip.mp4", name="src", description="", required=True
        )
        result = nodes.WWAIExposeOutputVideo().expose(
            name="out", description="", images=images, audio=audio, fps=fps
        )
        self.assertTrue(os.path.isfile(_written(result["ui"]["wwai_result"][0])))

    def test_mesh_marker_keeps_the_splat_container(self):
        (payload,) = nodes.WWAIExposeMesh().expose(file="splat.ply", name="geo", description="", required=True)
        self.assertEqual(payload.format, "ply")

    def test_a_file_staged_after_the_listing_still_validates(self):
        # WWAI uploads and submits in one breath, so the file can post-date the
        # directory listing INPUT_TYPES was built from.
        late = "staged_by_wwai.png"
        Image.new("RGB", (2, 2)).save(os.path.join(_IN, late))
        self.assertIs(nodes.WWAIExposeImage.VALIDATE_INPUTS(late, "n", "", True), True)
        self.assertIn("not found", nodes.WWAIExposeImage.VALIDATE_INPUTS("nope.png", "n", "", True))

    def test_audio_marker_loads_a_waveform(self):
        (payload,) = nodes.WWAIExposeAudio().expose(file="track.flac", name="bgm", description="", required=True)
        self.assertEqual(payload["sample_rate"], 44100)
        self.assertEqual(payload["waveform"].shape, (1, 2, 16))

    def test_no_file_is_the_default_and_first_choice(self):
        for name in ("WWAIExposeImage", "WWAIExposeAudio", "WWAIExposeVideo", "WWAIExposeMesh"):
            options = nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"]["file"][0]
            self.assertEqual(options[0], nodes.NO_FILE, name)

    def test_image_marker_leaves_the_reference_empty_when_no_file_is_chosen(self):
        self.assertEqual(
            nodes.WWAIExposeImage().expose(file=nodes.NO_FILE, name="ref", description="", required=False),
            (None, None),
        )

    def test_video_marker_leaves_the_reference_empty_when_no_file_is_chosen(self):
        self.assertEqual(
            nodes.WWAIExposeVideo().expose(file=nodes.NO_FILE, name="src", description="", required=False),
            (None, None, None, 0.0),
        )

    def test_audio_marker_leaves_the_reference_empty_when_no_file_is_chosen(self):
        self.assertEqual(
            nodes.WWAIExposeAudio().expose(file=nodes.NO_FILE, name="bgm", description="", required=False),
            (None,),
        )

    def test_mesh_marker_leaves_the_reference_empty_when_no_file_is_chosen(self):
        self.assertEqual(
            nodes.WWAIExposeMesh().expose(file=nodes.NO_FILE, name="geo", description="", required=False),
            (None,),
        )

    def test_no_file_still_requires_a_marker_name(self):
        for name, kwargs in (
            ("WWAIExposeImage", {}),
            ("WWAIExposeAudio", {}),
            ("WWAIExposeVideo", {}),
            ("WWAIExposeMesh", {}),
        ):
            with self.assertRaises(MarkerContractError, msg=name):
                nodes.NODE_CLASS_MAPPINGS[name]().expose(
                    file=nodes.NO_FILE, name="  ", description="", required=True, **kwargs
                )

    def test_no_file_passes_validation_without_touching_disk(self):
        for name in ("WWAIExposeImage", "WWAIExposeAudio", "WWAIExposeVideo", "WWAIExposeMesh"):
            self.assertIs(nodes.NODE_CLASS_MAPPINGS[name].VALIDATE_INPUTS(nodes.NO_FILE, "n", "", True), True, name)
            self.assertIs(nodes.NODE_CLASS_MAPPINGS[name].VALIDATE_INPUTS("", "n", "", True), True, name)

    def test_every_input_marker_refuses_a_blank_name(self):
        cases = [
            ("WWAIExposeText", {"value": "hi"}),
            ("WWAIExposeInt", {"value": 1}),
            ("WWAIExposeFloat", {"value": 1.0}),
            ("WWAIExposeBool", {"value": True}),
            ("WWAIExposeImage", {"file": "shot.png"}),
            ("WWAIExposeAudio", {"file": "track.flac"}),
            ("WWAIExposeVideo", {"file": "clip.mp4"}),
            ("WWAIExposeMesh", {"file": "splat.ply"}),
        ]
        for name, kwargs in cases:
            with self.assertRaises(MarkerContractError, msg=name):
                nodes.NODE_CLASS_MAPPINGS[name]().expose(name="  ", description="", required=True, **kwargs)


class OutputMarkerTest(unittest.TestCase):
    def test_image_marker_writes_one_png_and_reports_it(self):
        arr = np.zeros((1, 4, 4, 3), dtype=np.float32)
        result = nodes.WWAIExposeOutputImage().expose(
            value=_Tensor(arr), name="result", description=""
        )
        entries = result["ui"]["wwai_result"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "output")
        self.assertTrue(entries[0]["filename"].endswith(".png"))
        self.assertTrue(os.path.isfile(_written(entries[0])))

    def test_image_marker_refuses_a_batch(self):
        # One marker means one published asset; taking the first of several
        # would publish an arbitrary one.
        arr = np.zeros((2, 4, 4, 3), dtype=np.float32)
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIExposeOutputImage().expose(
                value=_Tensor(arr), name="result", description=""
            )
        self.assertIn("got 2", str(caught.exception))

    def test_video_marker_writes_and_reports_an_mp4_from_a_finished_video(self):
        video = _VideoFromFile(os.path.join(_IN, "clip.mp4"))
        result = nodes.WWAIExposeOutputVideo().expose(name="out", description="", video=video)
        entry = result["ui"]["wwai_result"][0]
        self.assertTrue(entry["filename"].endswith(".mp4"))
        self.assertTrue(os.path.isfile(_written(entry)))

    def test_video_marker_encodes_frames_plus_audio_like_video_combine(self):
        frames = _Tensor(np.zeros((4, 8, 8, 3), dtype=np.float32))
        result = nodes.WWAIExposeOutputVideo().expose(
            name="out", description="", images=frames, audio=_audio(), fps=24.0
        )
        entry = result["ui"]["wwai_result"][0]
        self.assertTrue(entry["filename"].endswith(".mp4"))
        self.assertTrue(os.path.isfile(_written(entry)))

    def test_video_marker_refuses_both_or_neither_source(self):
        video = _VideoFromFile(os.path.join(_IN, "clip.mp4"))
        frames = _Tensor(np.zeros((2, 8, 8, 3), dtype=np.float32))
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeOutputVideo().expose(name="o", description="")
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeOutputVideo().expose(
                name="o", description="", video=video, images=frames
            )

    def test_video_marker_refuses_audio_alongside_a_finished_video(self):
        # Muxing would mean decode + re-encode; dropping it would lose the
        # artist's audio silently. Neither is acceptable, so it refuses.
        video = _VideoFromFile(os.path.join(_IN, "clip.mp4"))
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIExposeOutputVideo().expose(
                name="o", description="", video=video, audio=_audio()
            )
        self.assertIn("already carries its own sound", str(caught.exception))

    def test_audio_marker_writes_and_reports_one_file(self):
        result = nodes.WWAIExposeOutputAudio().expose(
            value=_audio(), format="flac", name="track", description=""
        )
        entries = result["ui"]["wwai_result"]
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["filename"].endswith(".flac"))
        self.assertEqual(entries[0]["type"], "output")
        self.assertTrue(os.path.isfile(_written(entries[0])))

    def test_audio_marker_refuses_a_batch(self):
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIExposeOutputAudio().expose(
                value=_audio(batch=3), format="flac", name="track", description=""
            )
        self.assertIn("got 3", str(caught.exception))

    def test_mesh_marker_copies_a_splat_ply_byte_for_byte(self):
        source = os.path.join(_IN, "splat.ply")
        result = nodes.WWAIExposeOutputMesh().expose(
            value=_File3D(source), name="geo", description=""
        )
        entry = result["ui"]["wwai_result"][0]
        self.assertTrue(entry["filename"].endswith(".ply"))
        with open(_written(entry), "rb") as written, open(source, "rb") as original:
            self.assertEqual(written.read(), original.read())

    def test_mesh_marker_writes_glb_for_tensor_geometry(self):
        mesh = type("Mesh", (), {"vertices": np.zeros((1, 3, 3), dtype=np.float32)})()
        result = nodes.WWAIExposeOutputMesh().expose(value=mesh, name="geo", description="")
        self.assertTrue(result["ui"]["wwai_result"][0]["filename"].endswith(".glb"))

    def test_every_output_marker_refuses_a_blank_name(self):
        arr = _Tensor(np.zeros((1, 2, 2, 3), dtype=np.float32))
        cases = [
            ("WWAIExposeOutputImage", {"value": arr}),
            ("WWAIExposeOutputAudio", {"value": _audio(), "format": "flac"}),
            ("WWAIExposeOutputVideo", {"video": _VideoFromFile(os.path.join(_IN, "clip.mp4"))}),
            ("WWAIExposeOutputMesh", {"value": _File3D(os.path.join(_IN, "splat.ply"))}),
        ]
        for name, kwargs in cases:
            with self.assertRaises(MarkerContractError, msg=name):
                nodes.NODE_CLASS_MAPPINGS[name]().expose(name="", description="", **kwargs)


if __name__ == "__main__":
    unittest.main()
