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

import os
import shutil
import sys
import tempfile
import types
import unittest
from enum import Enum

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

    def save_to(self, path, format=None, codec=None):
        with open(path, "wb") as handle:
            handle.write(b"video-bytes")
        return path


def _comfy_api_stub():
    latest = types.ModuleType("comfy_api.latest")
    latest.Types = type("Types", (), {
        "File3D": _File3D, "VideoCodec": _VideoCodec, "VideoContainer": _VideoContainer,
    })
    latest.InputImpl = type("InputImpl", (), {"VideoFromFile": _VideoFromFile})
    parent = types.ModuleType("comfy_api")
    parent.latest = latest
    return parent, latest


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
    "WWAIExposeText", "WWAIExposeInt", "WWAIExposeFloat", "WWAIExposeImage",
    "WWAIExposeVideo", "WWAIExposeMesh", "WWAIExposeOutputImage",
    "WWAIExposeOutputVideo", "WWAIExposeOutputMesh",
}
OUTPUT_NODES = {"WWAIExposeOutputImage", "WWAIExposeOutputVideo", "WWAIExposeOutputMesh"}


class RegistrationTest(unittest.TestCase):
    def test_every_node_registers_with_a_display_name(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), ALL_NODES)
        self.assertEqual(set(nodes.NODE_DISPLAY_NAME_MAPPINGS), ALL_NODES)

    def test_every_node_declares_name_and_description_widgets(self):
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            required = cls.INPUT_TYPES()["required"]
            self.assertIn("name", required, name)
            self.assertIn("description", required, name)

    def test_output_markers_use_the_legacy_output_node_flag(self):
        # `is_output_node=True` belongs to the newer schema API and has NO
        # effect here — a terminal marker spelled that way is pruned from the
        # prompt and never reaches /history.
        for name in OUTPUT_NODES:
            cls = nodes.NODE_CLASS_MAPPINGS[name]
            self.assertIs(getattr(cls, "OUTPUT_NODE", False), True, name)
            self.assertFalse(hasattr(cls, "is_output_node"), name)


class SocketTypeTest(unittest.TestCase):
    def test_the_input_key_table_wwai_rewrites_against(self):
        for name in ("WWAIExposeImage", "WWAIExposeVideo", "WWAIExposeMesh"):
            self.assertIn("file", nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"], name)
        for name in ("WWAIExposeText", "WWAIExposeInt", "WWAIExposeFloat"):
            self.assertIn("value", nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"], name)

    def test_video_sockets_are_comfyui_native_video(self):
        self.assertEqual(nodes.WWAIExposeVideo.RETURN_TYPES, ("VIDEO",))
        self.assertEqual(
            nodes.WWAIExposeOutputVideo.INPUT_TYPES()["required"]["value"][0], "VIDEO"
        )

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
        image, mask = nodes.WWAIExposeImage().expose(file="shot.png", name="ref", description="")
        self.assertEqual(image.shape, (1, 6, 8, 3))
        self.assertEqual(mask.shape, (1, 6, 8))

    def test_video_marker_opens_the_file(self):
        (video,) = nodes.WWAIExposeVideo().expose(file="clip.mp4", name="src", description="")
        self.assertTrue(video.path.endswith("clip.mp4"))

    def test_mesh_marker_keeps_the_splat_container(self):
        (payload,) = nodes.WWAIExposeMesh().expose(file="splat.ply", name="geo", description="")
        self.assertEqual(payload.format, "ply")

    def test_a_file_staged_after_the_listing_still_validates(self):
        # WWAI uploads and submits in one breath, so the file can post-date the
        # directory listing INPUT_TYPES was built from.
        late = "staged_by_wwai.png"
        Image.new("RGB", (2, 2)).save(os.path.join(_IN, late))
        self.assertIs(nodes.WWAIExposeImage.VALIDATE_INPUTS(late, "n", ""), True)
        self.assertIn("not found", nodes.WWAIExposeImage.VALIDATE_INPUTS("nope.png", "n", ""))

    def test_every_input_marker_refuses_a_blank_name(self):
        cases = [
            ("WWAIExposeText", {"value": "hi"}),
            ("WWAIExposeInt", {"value": 1}),
            ("WWAIExposeFloat", {"value": 1.0}),
            ("WWAIExposeImage", {"file": "shot.png"}),
            ("WWAIExposeVideo", {"file": "clip.mp4"}),
            ("WWAIExposeMesh", {"file": "splat.ply"}),
        ]
        for name, kwargs in cases:
            with self.assertRaises(MarkerContractError, msg=name):
                nodes.NODE_CLASS_MAPPINGS[name]().expose(name="  ", description="", **kwargs)


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

    def test_video_marker_writes_and_reports_an_mp4(self):
        video = _VideoFromFile(os.path.join(_IN, "clip.mp4"))
        result = nodes.WWAIExposeOutputVideo().expose(value=video, name="out", description="")
        entry = result["ui"]["wwai_result"][0]
        self.assertTrue(entry["filename"].endswith(".mp4"))
        self.assertTrue(os.path.isfile(_written(entry)))

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
            ("WWAIExposeOutputVideo", {"value": _VideoFromFile(os.path.join(_IN, "clip.mp4"))}),
            ("WWAIExposeOutputMesh", {"value": _File3D(os.path.join(_IN, "splat.ply"))}),
        ]
        for name, kwargs in cases:
            with self.assertRaises(MarkerContractError, msg=name):
                nodes.NODE_CLASS_MAPPINGS[name]().expose(name="", description="", **kwargs)


if __name__ == "__main__":
    unittest.main()
