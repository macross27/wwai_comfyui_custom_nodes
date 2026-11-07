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

import av
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

    def numel(self):
        return self.size


def _torch_stub():
    mod = types.ModuleType("torch")
    mod.float32 = np.float32
    mod.from_numpy = _Tensor
    mod.zeros = lambda shape, dtype=None: _Tensor(np.zeros(shape, dtype=dtype or np.float32))
    mod.flip = lambda tensor, dims: _Tensor(np.flip(tensor, axis=dims))
    mod.maximum = lambda a, b: _Tensor(np.maximum(a, b))
    mod.stack = lambda tensors, dim=0: _Tensor(np.stack(tensors, axis=dim))
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


class _Sam3:
    """Stand-in for comfy_extras.nodes_sam3 (ComfyUI b53f77a7), recording every call.

    A frame's pixels hold its clip index, so each call knows which frames it got.
    Detect paints each include point with (keyframe index + 1); a track hands its
    initial mask to every frame unless a test sets `track`.
    """

    calls = []
    track = None  # (clip frame, track_data) -> (H, W) mask

    @staticmethod
    def frames(images):
        return [int(images[i, 0, 0, 0]) for i in range(images.shape[0])]


def _sam3_stub():
    import json

    def node(name, body):
        @classmethod
        def execute(cls, **kwargs):
            _Sam3.calls.append((name, kwargs))
            return types.SimpleNamespace(args=body(**kwargs))

        return type(name, (), {"execute": execute})

    def detect(model, image, conditioning=None, bboxes=None, positive_coords=None,
               negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False):
        (frame,) = _Sam3.frames(image)
        mask = np.zeros((1,) + image.shape[1:3], dtype=np.float32)
        for point in json.loads(positive_coords or "[]"):
            mask[0, point["y"], point["x"]] = frame + 1
        return (_Tensor(mask), [])

    def video_track(images, model, initial_mask=None, conditioning=None,
                    detection_threshold=0.5, max_objects=0, detect_interval=1):
        return ({"frames": _Sam3.frames(images), "init": initial_mask, "cond": conditioning,
                 "size": images.shape[1:3]},)

    def track_to_mask(track_data, object_indices=""):
        def default(frame, data):
            return data["init"][0] if data["init"] is not None else np.zeros(data["size"], np.float32)

        rule = _Sam3.track or default
        return (_Tensor(np.stack([rule(f, track_data) for f in track_data["frames"]])),)

    mod = types.ModuleType("comfy_extras.nodes_sam3")
    mod.SAM3_Detect = node("SAM3_Detect", detect)
    mod.SAM3_VideoTrack = node("SAM3_VideoTrack", video_track)
    mod.SAM3_TrackToMask = node("SAM3_TrackToMask", track_to_mask)
    return mod


def _comfy_nodes_stub():
    """ComfyUI's top-level `nodes` module: only CLIPTextEncode is used."""

    class CLIPTextEncode:
        def encode(self, clip, text):
            _Sam3.calls.append(("CLIPTextEncode", {"clip": clip, "text": text}))
            return ([["cond", {"text": text}]],)

    mod = types.ModuleType("nodes")
    mod.CLIPTextEncode = CLIPTextEncode
    return mod


class _Routes:
    def __init__(self):
        self.handlers = {}

    def post(self, path):
        def register(handler):
            self.handlers[path] = handler
            return handler

        return register


class _PromptQueue:
    """Upstream ``PromptQueue.get_current_queue()``: ``(running, pending)``."""

    def __init__(self):
        self.current = ([], [])
        self.error = None

    def get_current_queue(self):
        if self.error is not None:
            raise self.error
        return self.current


def _server_stub():
    mod = types.ModuleType("server")
    mod.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_Routes(), prompt_queue=_PromptQueue())
    )
    return mod


def _aiohttp_stub():
    web = types.ModuleType("aiohttp.web")
    web.json_response = lambda body, status=200: (status, body)
    parent = types.ModuleType("aiohttp")
    parent.web = web
    return parent, web


def setUpModule():
    global _TMP, _IN, _OUT, nodes, MarkerContractError
    _TMP = tempfile.mkdtemp()
    _IN = os.path.join(_TMP, "input")
    _OUT = os.path.join(_TMP, "output")
    os.makedirs(_IN)
    os.makedirs(_OUT)

    comfy_api, comfy_api_latest = _comfy_api_stub()
    comfy_extras, save_3d = _save_3d_stub()
    sam3 = _sam3_stub()
    comfy_extras.nodes_sam3 = sam3
    aiohttp, aiohttp_web = _aiohttp_stub()
    stubs = {
        "server": _server_stub(),
        "aiohttp": aiohttp,
        "aiohttp.web": aiohttp_web,
        "torch": _torch_stub(),
        "torchaudio": _torchaudio_stub(),
        "folder_paths": _folder_paths_stub(),
        "comfy_api": comfy_api,
        "comfy_api.latest": comfy_api_latest,
        "comfy_extras": comfy_extras,
        "comfy_extras.nodes_save_3d": save_3d,
        "comfy_extras.nodes_sam3": sam3,
        "nodes": _comfy_nodes_stub(),
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
    "WWAIExposeText", "WWAIExposeClicks", "WWAIExposeInt", "WWAIExposeFloat", "WWAIExposeBool", "WWAIExposeCombo", "WWAIExposeImage",
    "WWAIExposeAudio", "WWAIExposeVideo", "WWAIExposeMesh",
    "WWAIExposeOutputImage", "WWAIExposeOutputAudio",
    "WWAIExposeOutputVideo", "WWAIExposeOutputMesh",
    "WWAIRefuseBlankMask",
    "WWAIExposeKeyframes", "WWAIExposeImageOrVideo", "WWAIExposeOutputImageOrVideo", "WWAIRotoTrack",
}
# Utility nodes WWAI does not treat as markers (no WWAIExpose prefix): they
# carry no name/description/required widgets.
NON_MARKER_NODES = {"WWAIRefuseBlankMask", "WWAIRotoTrack"}
OUTPUT_NODES = {
    "WWAIExposeOutputImage", "WWAIExposeOutputAudio",
    "WWAIExposeOutputVideo", "WWAIExposeOutputMesh", "WWAIExposeOutputImageOrVideo",
}


def _audio(channels=2, batch=1):
    return {"waveform": _Tensor(np.zeros((batch, channels, 16), dtype=np.float32)), "sample_rate": 44100}


class RegistrationTest(unittest.TestCase):
    def test_clicks_node_shape(self):
        value = nodes.WWAIExposeClicks.INPUT_TYPES()["required"]["value"]
        self.assertEqual(value[0], "STRING")
        self.assertEqual(value[1]["default"], "")
        self.assertEqual(
            nodes.WWAIExposeClicks.RETURN_TYPES, ("STRING", "STRING", "BOUNDING_BOX", "BOUNDING_BOX", "INT")
        )

    def test_every_node_registers_with_a_display_name(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), ALL_NODES)
        self.assertEqual(set(nodes.NODE_DISPLAY_NAME_MAPPINGS), ALL_NODES)

    def test_every_node_declares_name_and_description_widgets(self):
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            if name in NON_MARKER_NODES:
                continue
            required = cls.INPUT_TYPES()["required"]
            self.assertIn("name", required, name)
            self.assertIn("description", required, name)

    def test_every_input_marker_declares_a_required_toggle(self):
        for name in ALL_NODES - OUTPUT_NODES - NON_MARKER_NODES:
            widget = nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["required"]["required"]
            self.assertEqual(widget[0], "BOOLEAN", name)
            self.assertIs(widget[1]["default"], True, name)

    def test_non_markers_do_not_use_the_marker_prefix(self):
        display = {"WWAIRefuseBlankMask": "WWAI Refuse Blank Mask", "WWAIRotoTrack": "WWAI Roto Track"}
        self.assertEqual(set(display), NON_MARKER_NODES)
        for name in NON_MARKER_NODES:
            self.assertFalse(name.startswith("WWAIExpose"), name)
            self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS[name], display[name])

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
            params = inspect.signature(cls.IS_CHANGED).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue  # **kwargs takes every widget
            accepted = set(params)
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

    def test_image_marker_loads_an_exr_as_unclamped_linear(self):
        # PIL cannot open EXR; an HDR value above 1.0 must reach the graph as is.
        pixels = np.full((6, 8, 3), 4.0, dtype=np.float32)
        pixels[0, 0] = (0.25, 0.5, 2.0)
        codec = av.CodecContext.create("exr", "w")
        codec.width, codec.height, codec.pix_fmt = 8, 6, "gbrpf32le"
        codec.time_base = Fraction(1, 1)
        frame = av.VideoFrame.from_ndarray(pixels, format="gbrpf32le")
        frame.pts = 0
        with open(os.path.join(_IN, "plate.exr"), "wb") as handle:
            handle.write(b"".join(bytes(p) for p in list(codec.encode(frame)) + list(codec.encode(None))))
        image, mask = nodes.WWAIExposeImage().expose(file="plate.exr", name="ref", description="", required=True)
        self.assertEqual(image.shape, (1, 6, 8, 3))
        np.testing.assert_allclose(np.asarray(image)[0], pixels, rtol=1e-3)
        self.assertEqual(mask.shape, (1, 64, 64))

    def test_bool_marker_returns_the_value_unchanged(self):
        self.assertEqual(
            nodes.WWAIExposeBool().expose(value=True, name="flag", description="", required=False),
            (True,),
        )

    def test_combo_marker_returns_a_listed_choice(self):
        self.assertEqual(
            nodes.WWAIExposeCombo().expose(value="base", choices="turbo\nbase\nv1\n", name="mode", description="", required=True),
            ("base",),
        )

    def test_combo_marker_refuses_a_value_not_in_its_choices(self):
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeCombo().expose(value="v2", choices="turbo\nbase\nv1", name="mode", description="", required=True)
        self.assertIsInstance(
            nodes.WWAIExposeCombo.VALIDATE_INPUTS(value="v2", choices="turbo\nbase", name="mode", description="", required=True),
            str,
        )

    def test_combo_marker_hands_a_number_dropdown_a_number(self):
        expose = nodes.WWAIExposeCombo().expose
        self.assertEqual(expose(value="2", choices="4\n2\n1", name="tc", description="", required=True), (2,))
        self.assertIsInstance(expose(value="2", choices="4\n2\n1", name="tc", description="", required=True)[0], int)
        self.assertEqual(expose(value="0.5", choices="1\n0.5", name="s", description="", required=True), (0.5,))
        self.assertEqual(
            expose(value="2", choices="4\n2\n1", name="tc", description="", required=True, value_type="text"),
            ("2",),
        )

    def test_combo_marker_output_links_into_a_legacy_dropdown(self):
        self.assertFalse(nodes.WWAIExposeCombo.RETURN_TYPES[0] != ["turbo", "base"])

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
            ("WWAIExposeCombo", {"value": "turbo", "choices": "turbo\nbase"}),
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

    def test_image_format_widgets_are_optional(self):
        # API prompts exported before these widgets existed carry no such keys;
        # a required input would make ComfyUI refuse every one of them.
        spec = nodes.WWAIExposeOutputImage.INPUT_TYPES()
        self.assertEqual(set(spec["required"]), {"value", "name", "description"})
        self.assertEqual(set(spec["optional"]), {"format", "bit_depth", "input_color_space"})

    def test_image_marker_refuses_a_format_pairing_core_cannot_write(self):
        validate = nodes.WWAIExposeOutputImage.VALIDATE_INPUTS
        self.assertIs(validate(), True)
        self.assertIs(validate("exr", "32-bit float", "linear"), True)
        self.assertIs(validate("avif", "auto", "HDR PQ"), True)
        self.assertIn("16-bit", validate("png", "32-bit float", "sRGB"))
        self.assertIn("HDR PQ", validate("exr", "auto", "HDR PQ"))
        self.assertIn("linear", validate("png", "auto", "linear"))
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeOutputImage().expose(
                value=_Tensor(np.zeros((1, 4, 4, 3), dtype=np.float32)), name="result", description="",
                format="avif", bit_depth="16-bit", input_color_space="sRGB",
            )

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



class RefuseBlankMaskTest(unittest.TestCase):
    def test_passes_a_batch_with_one_lit_pixel_in_one_frame(self):
        frames = _Tensor(np.zeros((3, 4, 4, 3), dtype=np.float32))
        frames[2, 1, 1, 0] = 1.0
        (out,) = nodes.WWAIRefuseBlankMask().expose(frames)
        self.assertIs(out, frames)

    def test_refuses_an_all_zero_batch_with_the_exact_message(self):
        frames = _Tensor(np.zeros((2, 4, 4, 3), dtype=np.float32))
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIRefuseBlankMask().expose(frames)
        self.assertEqual(str(caught.exception), NOTHING_FOUND)

    def test_an_empty_batch_is_nothing_found_not_a_max_error(self):
        frames = _Tensor(np.zeros((0, 4, 4, 3), dtype=np.float32))
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIRefuseBlankMask().expose(frames)
        self.assertEqual(str(caught.exception), NOTHING_FOUND)

    def test_only_this_node_raises_the_token(self):
        import ast

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        (cls,) = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "WWAIRefuseBlankMask"]
        lines = [i for i, line in enumerate(source.splitlines(), 1) if "WWAI_NOTHING_FOUND" in line]
        self.assertTrue(lines)
        for line in lines:
            self.assertTrue(cls.lineno <= line <= cls.end_lineno, f"WWAI_NOTHING_FOUND outside WWAIRefuseBlankMask at line {line}")

    def test_is_a_plain_non_output_image_node(self):
        cls = nodes.WWAIRefuseBlankMask
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(cls.RETURN_NAMES, ("images",))
        self.assertEqual(cls.INPUT_TYPES(), {"required": {"images": ("IMAGE",)}})
        self.assertFalse(getattr(cls, "OUTPUT_NODE", False))


NOTHING_FOUND = "WWAI_NOTHING_FOUND: Nothing was found, so no mask was made."

SETTINGS = {
    "prompt_threshold": 0.11, "prompt_max_objects": 7, "prompt_detect_interval": 3,
    "click_threshold": 0.22, "click_max_objects": 5, "click_detect_interval": 2,
    "detect_threshold": 0.33, "refine_iterations": 4,
}


def _clip(frame_count, size=4):
    """A clip whose every pixel holds its frame index (see _Sam3)."""
    frames = np.zeros((frame_count, size, size, 3), dtype=np.float32)
    for i in range(frame_count):
        frames[i] = i
    return _Tensor(frames)


def _keys(keys, frame_range=None):
    import json

    point = {1: (1, 1), 2: (2, 2), 0: (3, 3)}
    state = {"source": "a" * 64, "keyframes": [
        {"frame": f, "object": o, "items": [] if empty else [
            {"kind": "point", "mode": "include", "x": point[o][0], "y": point[o][1]}]}
        for f, o, *empty in keys
    ]}
    if frame_range:
        state["range"] = {"in": frame_range[0], "out": frame_range[1]}
    return json.dumps(state)


def _track(frames, keyframes, prompt="", **settings):
    _Sam3.calls.clear()
    (mask,) = nodes.WWAIRotoTrack().expose(
        frames=frames, model="sam3-model", clip="sam3-clip", keyframes=keyframes, prompt=prompt,
        **{**SETTINGS, **settings},
    )
    return mask


def _calls(name):
    return [kwargs for call, kwargs in _Sam3.calls if call == name]


class RotoTrackTest(unittest.TestCase):
    def tearDown(self):
        _Sam3.track = None

    def test_shape(self):
        cls = nodes.WWAIRotoTrack
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(cls.RETURN_NAMES, ("mask",))
        required = cls.INPUT_TYPES()["required"]
        self.assertEqual(
            [(k, v[0]) for k, v in list(required.items())[:5]],
            [("frames", "IMAGE"), ("model", "MODEL"), ("clip", "CLIP"), ("keyframes", "STRING"), ("prompt", "STRING")],
        )
        self.assertEqual(list(required)[5:], list(SETTINGS))
        for name in SETTINGS:
            self.assertIn(required[name][0], ("FLOAT", "INT"), name)
            self.assertNotIn("default", required[name][1], name)
        self.assertNotIn("optional", cls.INPUT_TYPES())

    def test_staggered_keys_each_object_takes_its_nearest_keyframe(self):
        mask = _track(_clip(150), _keys([(0, 1), (100, 1), (50, 2)], (40, 120)))
        self.assertEqual(mask.shape, (81, 4, 4, 3))
        frame_60 = mask[60 - 40]
        self.assertEqual(frame_60[1, 1, 0], 101)  # object 1 from its key at 100
        self.assertEqual(frame_60[2, 2, 0], 51)  # object 2 from its key at 50
        self.assertEqual(mask[50 - 40][1, 1, 0], 1)  # a tie goes to the earlier key (0)
        self.assertEqual(mask[51 - 40][1, 1, 0], 101)

    def test_steps_match_every_tracked_frames_row(self):
        import json

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "click_cases.json")
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)["tracked_frames"]
        for row in rows:
            with self.subTest(row["name"]):
                _track(_clip(row["frame_count"]), row["state"], "person" if row["prompt_set"] else "")
                stepped = sum(call["images"].shape[0] for call in _calls("SAM3_VideoTrack"))
                self.assertEqual(stepped, row["tracked"])

    def test_backward_runs_get_the_frames_reversed(self):
        _track(_clip(10), _keys([(5, 1)], (2, 5)))
        (call,) = _calls("SAM3_VideoTrack")
        self.assertEqual(_Sam3.frames(call["images"]), [5, 4, 3, 2])

    def test_a_blank_prompt_is_never_encoded(self):
        _track(_clip(10), _keys([(0, 1)]), prompt="   ")
        self.assertEqual(_calls("CLIPTextEncode"), [])
        self.assertTrue(all(call.get("conditioning") is None for call in _calls("SAM3_VideoTrack")))

    def test_a_prompt_with_no_object_0_key_tracks_from_frame_0(self):
        _track(_clip(100), _keys([], (50, 60)), prompt="person")
        (encode,) = _calls("CLIPTextEncode")
        self.assertEqual((encode["clip"], encode["text"]), ("sam3-clip", "person"))
        self.assertEqual(_calls("SAM3_Detect"), [])
        (call,) = _calls("SAM3_VideoTrack")
        self.assertEqual(_Sam3.frames(call["images"]), list(range(0, 61)))
        self.assertIsNone(call["initial_mask"])
        self.assertIsNotNone(call["conditioning"])

    def test_a_prompt_keyframe_detects_with_the_prompt_and_its_clicks(self):
        _track(_clip(10), _keys([(4, 0)], (4, 9)), prompt="person")
        (detect,) = _calls("SAM3_Detect")
        self.assertEqual(_Sam3.frames(detect["image"]), [4])
        self.assertIsNotNone(detect["conditioning"])
        self.assertEqual(detect["positive_coords"], '[{"x": 3, "y": 3}]')
        (call,) = _calls("SAM3_VideoTrack")
        self.assertEqual(call["initial_mask"][0][3, 3], 5)  # the detect mask at frame 4
        self.assertEqual(_Sam3.frames(call["images"]), list(range(4, 10)))

    def test_each_setting_reaches_its_sam3_call(self):
        _track(_clip(10), _keys([(0, 0, "empty"), (0, 1)]), prompt="person")
        detects = _calls("SAM3_Detect")
        self.assertEqual(len(detects), 2)
        for call in detects:
            self.assertEqual((call["threshold"], call["refine_iterations"]), (0.33, 4))
            self.assertIs(call["individual_masks"], False)
        prompt_track, click_track = _calls("SAM3_VideoTrack")
        self.assertIsNotNone(prompt_track["conditioning"])
        self.assertEqual(
            (prompt_track["detection_threshold"], prompt_track["max_objects"], prompt_track["detect_interval"]),
            (0.11, 7, 3),
        )
        self.assertIsNone(click_track.get("conditioning"))
        self.assertEqual(
            (click_track["detection_threshold"], click_track["max_objects"], click_track["detect_interval"]),
            (0.22, 5, 2),
        )
        for call in _calls("SAM3_VideoTrack") + detects:
            self.assertEqual(call["model"], "sam3-model")

    def test_a_grammar_refusal_comes_before_any_sam3_work(self):
        with self.assertRaises(MarkerContractError) as caught:
            _track(_clip(100), _keys([(55, 1)], (60, 50)))
        self.assertIn("COMFYUI_CLICKS_INVALID: in_after_out:", str(caught.exception))
        self.assertEqual(_Sam3.calls, [])

    def test_twenty_objects_and_three_hundred_items_pass(self):
        import json

        many = json.dumps({"source": "a" * 64, "keyframes": [
            {"frame": 0, "object": o, "items": [{"kind": "point", "mode": "include", "x": 1, "y": 1}] * (300 if o == 1 else 1)}
            for o in range(1, 21)
        ]})
        mask = _track(_clip(3), many)
        self.assertEqual(mask.shape[0], 3)

    def test_the_layers_are_unioned(self):
        mask = _track(_clip(5), _keys([(0, 1), (0, 2)]))
        self.assertEqual((mask[3][1, 1, 0], mask[3][2, 2, 0]), (1, 1))

    # Nothing found: the tracker never raises it; WWAIRefuseBlankMask after it does.

    def test_a_prompt_that_matches_from_frame_30_passes(self):
        _Sam3.track = lambda f, data: np.full(data["size"], 1.0 if f >= 30 else 0.0, np.float32)
        mask = _track(_clip(50), _keys([]), prompt="person")
        self.assertEqual(float(mask[0].max()), 0.0)
        (out,) = nodes.WWAIRefuseBlankMask().expose(mask)
        self.assertIs(out, mask)

    def test_a_click_track_lost_before_in_is_blank_and_then_refused(self):
        _Sam3.track = lambda f, data: data["init"][0] * (f < 40)
        mask = _track(_clip(100), _keys([(0, 1)], (50, 60)))
        self.assertEqual(float(mask.max()), 0.0)
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIRefuseBlankMask().expose(mask)
        self.assertEqual(str(caught.exception), NOTHING_FOUND)

    def test_one_masked_frame_in_the_range_passes(self):
        _Sam3.track = lambda f, data: data["init"][0] * (f == 55)
        mask = _track(_clip(100), _keys([(0, 1)], (50, 60)))
        nodes.WWAIRefuseBlankMask().expose(mask)


class ExposeKeyframesTest(unittest.TestCase):
    def test_shape(self):
        cls = nodes.WWAIExposeKeyframes
        self.assertEqual(cls.RETURN_TYPES, ("STRING",))
        self.assertEqual(cls.RETURN_NAMES, ("keyframes",))
        self.assertEqual(list(cls.INPUT_TYPES()["required"]), ["value", "name", "description", "required"])

    def test_passes_the_text_and_refuses_malformed_text(self):
        text = _keys([(0, 1)])
        self.assertEqual(nodes.WWAIExposeKeyframes().expose(text, "clicks", "", True), (text,))
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIExposeKeyframes().expose("{", "clicks", "", True)
        self.assertIn("COMFYUI_CLICKS_INVALID: malformed:", str(caught.exception))
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeKeyframes().expose(text, " ", "", True)


class ImageOrVideoTest(unittest.TestCase):
    def test_shape(self):
        cls = nodes.WWAIExposeImageOrVideo
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "WWAI_MEDIA_INFO"))
        self.assertEqual(cls.RETURN_NAMES, ("images", "media_info"))
        self.assertIn("file", cls.INPUT_TYPES()["required"])
        out = nodes.WWAIExposeOutputImageOrVideo.INPUT_TYPES()
        self.assertEqual(list(out["required"]), ["images", "media_info", "name", "description"])
        self.assertEqual(out["required"]["media_info"], ("WWAI_MEDIA_INFO",))

    def test_a_picture_gives_what_expose_image_gives(self):
        images, info = nodes.WWAIExposeImageOrVideo().expose("shot.png", "src", "", True)
        image, _ = nodes.WWAIExposeImage().expose("shot.png", "src", "", True)
        np.testing.assert_array_equal(images, image)
        self.assertEqual(info, {"kind": "image", "fps": None})

    def test_a_movie_gives_what_expose_video_gives(self):
        images, info = nodes.WWAIExposeImageOrVideo().expose("clip.mp4", "src", "", True)
        _, frames, _, fps = nodes.WWAIExposeVideo().expose("clip.mp4", "src", "", True)
        np.testing.assert_array_equal(images, frames)
        self.assertEqual(info, {"kind": "video", "fps": Fraction(24, 1)})
        self.assertEqual(float(info["fps"]), fps)

    def test_a_file_that_is_neither_is_refused(self):
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeImageOrVideo().expose("track.flac", "src", "", True)

    def test_staged_file_refusal_is_shared(self):
        import hashlib

        os.makedirs(os.path.join(_IN, "wwai"), exist_ok=True)
        rel = f"wwai/wwai-{hashlib.sha256(b'abc').hexdigest()}.mp4"
        with open(os.path.join(_IN, rel), "wb") as handle:
            handle.write(b"not-abc")
        with self.assertRaises(MarkerContractError) as caught:
            nodes.WWAIExposeImageOrVideo().expose(rel, "src", "", True)
        self.assertIn("WWAI_STAGED_FILE_UNUSABLE", str(caught.exception))

    def test_output_writes_a_png_for_a_picture(self):
        _, info = nodes.WWAIExposeImageOrVideo().expose("shot.png", "src", "", True)
        result = nodes.WWAIExposeOutputImageOrVideo().expose(
            images=_Tensor(np.zeros((1, 4, 4, 3), dtype=np.float32)), media_info=info, name="mask", description=""
        )
        entry = result["ui"]["wwai_result"][0]
        self.assertTrue(entry["filename"].endswith(".png"))
        with Image.open(_written(entry)) as img:
            self.assertEqual(img.format, "PNG")

    def test_output_writes_an_mp4_at_the_input_fps(self):
        info = {"kind": "video", "fps": Fraction(15, 1)}
        captured = {}
        real = nodes.InputImpl.VideoFromComponents

        def spy(components, **kwargs):
            captured["fps"] = components.frame_rate
            return real(components, **kwargs)

        nodes.InputImpl.VideoFromComponents = spy
        try:
            result = nodes.WWAIExposeOutputImageOrVideo().expose(
                images=_Tensor(np.zeros((3, 4, 4, 3), dtype=np.float32)), media_info=info, name="mask", description=""
            )
        finally:
            nodes.InputImpl.VideoFromComponents = real
        self.assertTrue(result["ui"]["wwai_result"][0]["filename"].endswith(".mp4"))
        self.assertEqual(captured["fps"], Fraction(15, 1))

    def test_output_refuses_a_blank_name(self):
        with self.assertRaises(MarkerContractError):
            nodes.WWAIExposeOutputImageOrVideo().expose(
                images=_Tensor(np.zeros((1, 4, 4, 3), dtype=np.float32)),
                media_info={"kind": "image", "fps": None}, name="", description="",
            )


class StagedFileUseTest(unittest.TestCase):
    """A staged name whose bytes do not match is refused at use (D3)."""

    def _stage(self, ext, data=b"abc"):
        import hashlib

        os.makedirs(os.path.join(_IN, "wwai"), exist_ok=True)
        hexd = hashlib.sha256(b"abc").hexdigest()
        rel = f"wwai/wwai-{hexd}.{ext}"
        with open(os.path.join(_IN, rel), "wb") as handle:
            handle.write(data)
        return rel

    def test_each_file_marker_refuses_mismatched_bytes(self):
        for cls, ext in (
            (nodes.WWAIExposeImage, "png"), (nodes.WWAIExposeVideo, "mp4"),
            (nodes.WWAIExposeAudio, "flac"), (nodes.WWAIExposeMesh, "glb"),
        ):
            rel = self._stage(ext, b"not-abc")
            with self.assertRaises(MarkerContractError) as ctx:
                cls().expose(file=rel, name="n", description="", required=True)
            self.assertIn("WWAI_STAGED_FILE_UNUSABLE", str(ctx.exception), cls.__name__)
            self.assertFalse(os.path.exists(os.path.join(_IN, rel)))

    def test_validate_inputs_passes_a_missing_staged_name(self):
        rel = "wwai/wwai-" + "a" * 64 + ".png"
        self.assertIs(nodes.WWAIExposeImage.VALIDATE_INPUTS(rel, "n", "", True), True)

    def test_validate_inputs_still_refuses_a_missing_author_file(self):
        self.assertIn("input file not found", nodes.WWAIExposeImage.VALIDATE_INPUTS("nope.png", "n", "", True))


class OutputMarkerCachingTest(unittest.TestCase):
    """An output marker must never be served from ComfyUI's cache: a cached one
    reports an old output file the sweeper may have deleted."""

    def test_every_output_marker_always_reruns(self):
        import math

        outputs = [cls for cls in nodes.NODE_CLASS_MAPPINGS.values() if getattr(cls, "OUTPUT_NODE", False)]
        self.assertEqual(len(outputs), 5)
        for cls in outputs:
            first = cls.IS_CHANGED(value=None, name="n", description="")
            self.assertTrue(math.isnan(first), cls.__name__)
            self.assertNotEqual(first, cls.IS_CHANGED(value=None, name="n", description=""), cls.__name__)


class StagedFilesRouteTest(unittest.TestCase):
    """The delete door reads upstream's queue shape and refuses when it cannot."""

    HEX = "a" * 64

    def setUp(self):
        import importlib

        package = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        self.route = importlib.import_module(f"{package}.staged_files_route")
        self.queue = self.route.PromptServer.instance.prompt_queue
        self.addCleanup(setattr, self.queue, "current", ([], []))
        self.addCleanup(setattr, self.queue, "error", None)
        secret_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, secret_dir)
        self.secret = "s" * 40
        secret_path = os.path.join(secret_dir, "secret")
        with open(secret_path, "w") as handle:
            handle.write(self.secret)
        original = self.route.SECRET_PATH
        self.route.SECRET_PATH = secret_path
        self.addCleanup(setattr, self.route, "SECRET_PATH", original)

    def _item(self, number, value):
        prompt = {"1": {"class_type": "WWAIExposeImage", "inputs": {"file": value, "name": "x"}}}
        return (number, f"p{number}", prompt, {}, ["1"], {})

    def _post(self, filename):
        import asyncio

        async def body():
            return {"filename": filename}

        request = types.SimpleNamespace(
            json=body, headers={"X-WWAI-Staged-File-Secret": self.secret}
        )
        handler = self.route.PromptServer.instance.routes.handlers["/wwai/staged-file/delete"]
        self.assertIs(handler, self.route.delete_staged_file)
        return asyncio.run(handler(request))

    def test_running_and_pending_values_are_read(self):
        running = f"wwai/wwai-{self.HEX}.png"
        pending = f"wwai/wwai-{'b' * 64}.png"
        self.queue.current = ([self._item(1, running)], [self._item(2, pending)])
        values = self.route.queued_input_values()
        self.assertIn(running, values)
        self.assertIn(pending, values)
        self.assertIn("x", values)

    def test_annotated_pending_value_is_in_use(self):
        self.queue.current = ([], [self._item(2, f"wwai/wwai-{self.HEX}.png [input]")])
        self.assertEqual(self._post(f"wwai-{self.HEX}.png"), (409, {"wwai_staged_file": "in_use"}))

    def test_unreadable_queue_is_a_typed_refusal(self):
        self.queue.error = RuntimeError("queue locked")
        self.assertEqual(
            self._post(f"wwai-{self.HEX}.png"),
            (500, {"wwai_staged_file": "failed", "detail": "cannot read the prompt queue: queue locked"}),
        )

if __name__ == "__main__":
    unittest.main()
