"""WWAI expose/marker nodes for ComfyUI.

A workflow author drops these into a ComfyUI graph to say "this value is
controllable from outside" and "this is the finished result", typing a name
and a short description on the node. WWAI reads those markers back out of the
saved workflow JSON, so no admin ever hand-types a parameter list.

The nodes are not labels. An input marker LOADS a real file and hands out the
concrete payload its consumer expects (an IMAGE tensor, ComfyUI's native VIDEO
object, a File3D). An output marker WRITES the finished file and reports it
under `wwai_result` — a key only these nodes emit, so WWAI can pick this job's
result out of `/history/{prompt_id}` without scanning folders or guessing at
the shared `images`/`video`/`3d` keys ComfyUI's own save nodes also write.

Written against the legacy INPUT_TYPES/RETURN_TYPES node API on purpose. Every
socket type below is a plain string in both APIs, and OUTPUT_NODE (the legacy
spelling) is what actually keeps a terminal marker from being pruned out of
the prompt — the newer `is_output_node=True` has no effect on a class like
these.
"""

import json
import logging
import os
from fractions import Fraction

import av
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from .wwai_markers_core import (
    MarkerContractError,
    copy_file_verbatim,
    extension_of,
    require_marker_name,
    require_single,
)

logger = logging.getLogger(__name__)

# Payload classes for the video and 3D markers. Text/Int/Float/Image need
# none of this, so a ComfyUI too old to provide them still gets those five
# nodes — with a warning naming exactly what is missing, never a quiet
# degrade to a wildcard socket.
try:
    from comfy_api.latest import IO, UI, InputImpl, Types

    MEDIA_PAYLOADS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the ComfyUI host
    IO = UI = InputImpl = Types = None
    MEDIA_PAYLOADS_IMPORT_ERROR = exc

# Only the audio INPUT marker decodes a file itself; everything else about
# audio goes through ComfyUI's own encoder.
try:
    import torchaudio

    AUDIO_LOADER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the ComfyUI host
    torchaudio = None
    AUDIO_LOADER_IMPORT_ERROR = exc

# ComfyUI's own GLB writer. Reusing it is deliberate: hand-rolling a glTF
# serializer here would duplicate several hundred lines of UV/vertex-colour/
# texture handling that already exists and is already exercised upstream.
try:
    from comfy_extras.nodes_save_3d import get_mesh_batch_item, save_glb

    GLB_WRITER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the ComfyUI host
    get_mesh_batch_item = save_glb = None
    GLB_WRITER_IMPORT_ERROR = exc

# ComfyUI's own Save Image (Advanced) encoders, for the same reason: the
# sRGB/HLG -> scene-linear EXR conversion and the AVIF colour tagging are
# colour-science that must match what core writes, not a second opinion.
try:
    from comfy_extras.nodes_images import _encode_image, _save_avif, inject_exr_metadata, inject_png_metadata

    IMAGE_ENCODER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the ComfyUI host
    _encode_image = _save_avif = inject_exr_metadata = inject_png_metadata = None
    IMAGE_ENCODER_IMPORT_ERROR = exc


# --- socket types -----------------------------------------------------------
#
# ComfyUI matches sockets by exact type string. There is no single 3D type —
# there are fourteen — which is why core's SaveGLB declares a comma-joined set
# rather than one name. Declaring one here would silently refuse most real 3D
# producers.

IMAGE_TYPE = "IMAGE"
AUDIO_TYPE = "AUDIO"
VIDEO_TYPE = "VIDEO"
FILE_3D_TYPE = "FILE_3D"
MESH_OUTPUT_TYPE = ",".join(
    (
        "MESH",
        "FILE_3D",
        "FILE_3D_GLB",
        "FILE_3D_GLTF",
        "FILE_3D_OBJ",
        "FILE_3D_FBX",
        "FILE_3D_STL",
        "FILE_3D_USDZ",
        "FILE_3D_PLY",
        "FILE_3D_SPLAT",
        "FILE_3D_SPZ",
        "FILE_3D_KSPLAT",
        "FILE_3D_SPLAT_ANY",
        "FILE_3D_POINT_CLOUD_ANY",
    )
)

MESH_EXTENSIONS = (".gltf", ".glb", ".obj", ".fbx", ".stl", ".spz", ".splat", ".ply", ".ksplat")

# What core's Save Image (Advanced) can write, per format: (bit depths, input
# colour spaces). The first bit depth is what "auto" picks. Flat combos rather
# than core's DynamicCombo because these markers use the legacy node API, so a
# mismatched pair is refused by name instead of being hidden by the UI.
IMAGE_FORMATS = {
    "png": (("8-bit", "16-bit"), ("sRGB",)),
    "exr": (("16-bit float", "32-bit float"), ("sRGB", "HDR", "linear")),
    "avif": (("auto", "8-bit YUV420", "10-bit YUV420"), ("sRGB", "HDR", "HDR PQ")),
}
IMAGE_BIT_DEPTHS = ("auto", "8-bit", "16-bit", "16-bit float", "32-bit float", "8-bit YUV420", "10-bit YUV420")
IMAGE_COLOR_SPACES = ("sRGB", "HDR", "HDR PQ", "linear")
# AVIF keeps core's default quality; there is no single-image reason to differ.
AVIF_CRF = 18

# Where output markers write. WWAI locates the result by node id out of the
# history response, so the filename only has to be unique, not meaningful.
FILENAME_PREFIX = "wwai/WWAI"

INPUT_CATEGORY = "WWAI/Expose"
OUTPUT_CATEGORY = "WWAI/Expose Output"

# Sentinel choice on every file-based input marker's dropdown that means "this
# optional reference has no data for this run." It is always the first entry
# (so a freshly dropped node defaults to skipped, not to an arbitrary real
# file), and the marker's own extension/content-type filtering means no real
# upload can ever collide with it.
NO_FILE = "None"


def _is_no_file(file):
    return not file or file == NO_FILE


def metadata_widgets():
    """The two widgets every marker carries, in one place.

    WWAI reads these back off the saved workflow JSON. Keep them widgets — if
    an author converts one into a wired input its API value becomes a link
    array rather than a string, which WWAI's upload gate rejects.
    """
    return {
        "name": ("STRING", {"default": "", "tooltip": "Unique parameter name WWAI shows to the artist."}),
        "description": (
            "STRING",
            {"multiline": True, "default": "", "tooltip": "One line telling the artist what this is for."},
        ),
    }


def required_widget():
    """The tick box that says whether an artist MUST fill this input.

    Plan 1961 (CFY6). WWAI used to INFER requiredness -- a file marker left on
    NO_FILE was optional, and a scalar marker was always optional, so a text or
    number input could never be made mandatory and clearing a file silently
    turned a needed input into an optional one. The author declares it here
    instead, and WWAI reads exactly this widget off the export.

    Deliberately NOT part of ``metadata_widgets()``: the four OUTPUT markers
    share that helper, and a workflow's result is never optional, so a tick box
    on an output node would offer a choice that does not exist.

    Spread LAST, after ``**metadata_widgets()``, on every input marker. A widget
    inserted before ``name``/``description`` would shift their positions in the
    ``widgets_values`` array of every workflow JSON already saved from ComfyUI's
    own UI, loading the saved name into this toggle.
    """
    return {
        "required": (
            "BOOLEAN",
            {
                "default": True,
                "label_on": "artist must fill this",
                "label_off": "artist may leave this empty",
                "tooltip": (
                    "On: WWAI refuses to run until the artist supplies a value. "
                    "Off: WWAI runs with the value saved in this node."
                ),
            },
        ),
    }


def input_root_files(content_types=None, extensions=None):
    """Files directly in ComfyUI's input directory.

    The root, never a subfolder: WWAI stages an artist's file by POSTing it to
    `/upload/image` with no `subfolder`, so anything nested here would be
    unreachable to it.
    """
    input_dir = folder_paths.get_input_directory()
    names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if content_types is not None:
        names = folder_paths.filter_files_content_types(names, content_types)
    if extensions is not None:
        names = [f for f in names if f.lower().endswith(extensions)]
    # NO_FILE goes first: it is the "skip this optional reference" choice, and
    # a freshly dropped node should default to skipped, not to some arbitrary
    # real file that happens to sort first.
    return [NO_FILE] + sorted(f for f in names if f != NO_FILE)


def save_path(class_name, name, extension):
    """Reserve the next free output path, and report it the way WWAI reads it."""
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        FILENAME_PREFIX, folder_paths.get_output_directory()
    )
    file = f"{filename}_{counter:05}_.{extension}"
    return os.path.join(full_folder, file), {
        "filename": file,
        "subfolder": subfolder,
        "type": "output",
    }


def wwai_result(entry):
    """The one shape every output marker returns."""
    return {"ui": {"wwai_result": [entry]}}


def load_exr(path):
    """An EXR as an IMAGE + MASK pair, pixels left exactly as stored.

    PIL cannot read EXR; PyAV (already a ComfyUI dependency) can. EXR is
    scene-linear and may exceed 1.0, so nothing is clamped or re-encoded to
    sRGB here — the same linear the WWAI output marker and core's Save Image
    (Advanced) write. A workflow that wants sRGB puts core's Convert Image
    Color Space (linear -> sRGB) after this node.
    """
    with av.open(path) as container:
        frame = next(container.decode(video=0))
        # Grey+alpha decodes to 2 components, RGBA to 4.
        components = len(frame.format.components)
        has_alpha = components in (2, 4)
        if components == 1:
            # Read grey as grey: swscale's grey->RGB path range-maps and shifts values.
            pixels = np.repeat(frame.to_ndarray(format="grayf32le")[..., None], 3, axis=-1)
        else:
            pixels = frame.to_ndarray(format="gbrapf32le" if has_alpha else "gbrpf32le")
    image = torch.from_numpy(np.ascontiguousarray(pixels[..., :3]))[None,]
    if has_alpha:
        mask = 1.0 - torch.from_numpy(np.ascontiguousarray(pixels[..., 3]))
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32)
    return (image, mask.unsqueeze(0))


class FileMarkerMixin:
    """Shared validation for the markers that open a file.

    WWAI uploads the artist's file and submits in the same breath, so the file
    can post-date the directory listing INPUT_TYPES was built from. These two
    hooks are what let it validate anyway — the same pair core's LoadImage
    carries, for the same reason.
    """

    # `required` (plan 1961's tick box) is accepted here on both hooks though
    # neither body reads it. VALIDATE_INPUTS is argspec-filtered by ComfyUI
    # before the call, so it would tolerate a missing parameter -- but
    # IS_CHANGED is NOT filtered, it is always called with every key declared
    # in INPUT_TYPES()["required"], so adding a widget there is a signature
    # change for IS_CHANGED whether the hook cares about the value or not.
    # No default: ComfyUI always supplies it, so a default would silently
    # hide a marker subclass whose widget failed to register.
    @classmethod
    def IS_CHANGED(cls, file, name, description, required):
        if _is_no_file(file):
            return NO_FILE
        path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(path), os.path.getsize(path)

    @classmethod
    def VALIDATE_INPUTS(cls, file, name, description, required):
        if _is_no_file(file):
            return True
        if not folder_paths.exists_annotated_filepath(file):
            return f"{cls.__name__}: input file not found: {file}"
        return True


# --- input markers ----------------------------------------------------------


class WWAIExposeText:
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    DESCRIPTION = "Marks a text value as controllable from WWAI."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"multiline": True, "default": ""}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, value, name, description, required):
        require_marker_name(name, type(self).__name__)
        return (value,)


class WWAIExposeInt:
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    DESCRIPTION = "Marks a whole number as controllable from WWAI."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0x7FFFFFFF, "max": 0x7FFFFFFF}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, value, name, description, required):
        require_marker_name(name, type(self).__name__)
        return (value,)


class WWAIExposeFloat:
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    DESCRIPTION = "Marks a decimal number as controllable from WWAI."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "step": 0.01}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, value, name, description, required):
        require_marker_name(name, type(self).__name__)
        return (value,)


class WWAIExposeBool:
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("value",)
    DESCRIPTION = "Marks an on/off switch as controllable from WWAI."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, value, name, description, required):
        require_marker_name(name, type(self).__name__)
        return (value,)


class AnyType(str):
    """ComfyUI's legacy wildcard socket type.

    A dropdown on a legacy node is declared as a bare list of options, and core
    only accepts a link into it from a "*" output (validate_node_input). It also
    skips the value-in-list check for a linked input, so WWAIExposeCombo checks
    the value against its own choices instead.
    """

    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


def parse_choices(choices, class_name):
    """The author's choices, one per line, blank lines dropped."""
    parsed = [line.strip() for line in (choices or "").splitlines() if line.strip()]
    if not parsed:
        raise MarkerContractError(
            f"{class_name}: the 'choices' field is empty. Type the dropdown's "
            f"options, one per line, exactly as the target node lists them."
        )
    duplicates = sorted({c for c in parsed if parsed.count(c) > 1})
    if duplicates:
        raise MarkerContractError(f"{class_name}: duplicate choices {duplicates}.")
    return parsed


class WWAIExposeCombo:
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    DESCRIPTION = (
        "Marks a dropdown choice (e.g. mode: turbo / base / v1) as controllable "
        "from WWAI. Convert the target node's dropdown to an input and wire this into it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "tooltip": "The choice used when WWAI supplies none."}),
                "choices": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "The dropdown's options, one per line, spelled exactly as the target node lists them.",
                    },
                ),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, value, choices, name, description, required):
        try:
            options = parse_choices(choices, cls.__name__)
        except MarkerContractError as exc:
            return str(exc)
        if value not in options:
            return f"{cls.__name__}: {value!r} is not one of the choices {options}."
        return True

    def expose(self, value, choices, name, description, required):
        class_name = type(self).__name__
        require_marker_name(name, class_name)
        options = parse_choices(choices, class_name)
        if value not in options:
            raise MarkerContractError(f"{class_name}: {value!r} is not one of the choices {options}.")
        return (value,)


class WWAIExposeImage(FileMarkerMixin):
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = (IMAGE_TYPE, "MASK")
    RETURN_NAMES = ("value", "mask")
    DESCRIPTION = (
        "Opens an image and marks it as an input WWAI supplies. Leave the "
        f"file as \"{NO_FILE}\" to skip this reference for a run — the node "
        "outputs None and any optional downstream input it feeds is ignored."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (input_root_files(content_types=["image"]), {"image_upload": True}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, file, name, description, required):
        require_marker_name(name, type(self).__name__)
        if _is_no_file(file):
            return (None, None)
        path = folder_paths.get_annotated_filepath(file)
        if extension_of(path, "") == "exr":
            return load_exr(path)
        img = ImageOps.exif_transpose(Image.open(path))
        rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        image = torch.from_numpy(rgb)[None,]
        if "A" in img.getbands():
            alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)
        return (image, mask.unsqueeze(0))


class WWAIExposeVideo(FileMarkerMixin):
    """Opens a video for either of ComfyUI's two video worlds.

    Core consumers take a whole `VIDEO` object; Video Helper Suite has no such
    type and works in frames plus a separate audio track — its own
    `VHS_LoadVideo` returns `(IMAGE, frame_count, AUDIO, VHS_VIDEOINFO)`. So
    this marker emits both shapes from one node, the way core's
    `GetVideoComponents` splits a video, and the author wires whichever port
    their graph speaks.

    Splitting costs one decode of the file into frames. That is the same cost
    `VHS_LoadVideo` already pays in any workflow that uses it, and ComfyUI
    caches it per unique input.
    """

    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = (VIDEO_TYPE, IMAGE_TYPE, AUDIO_TYPE, "FLOAT")
    RETURN_NAMES = ("video", "images", "audio", "fps")
    DESCRIPTION = (
        "Opens a video and marks it as an input WWAI supplies. Leave the "
        f"file as \"{NO_FILE}\" to skip this reference for a run — every "
        "output is None (fps is 0.0) and any optional downstream input it "
        "feeds is ignored."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (input_root_files(content_types=["video"]), {"video_upload": True}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, file, name, description, required):
        require_marker_name(name, type(self).__name__)
        if _is_no_file(file):
            return (None, None, None, 0.0)
        path = folder_paths.get_annotated_filepath(file)
        video = InputImpl.VideoFromFile(path)
        components = video.get_components()
        # audio is None when the file carries no sound track — the same thing
        # GetVideoComponents passes on, and downstream audio ports are optional.
        return (video, components.images, components.audio, float(components.frame_rate))


class WWAIExposeAudio(FileMarkerMixin):
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("value",)
    DESCRIPTION = (
        "Opens an audio file and marks it as an input WWAI supplies. Leave "
        f"the file as \"{NO_FILE}\" to skip this reference for a run — the "
        "node outputs None and any optional downstream input it feeds is "
        "ignored."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (input_root_files(content_types=["audio", "video"]), {"audio_upload": True}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, file, name, description, required):
        require_marker_name(name, type(self).__name__)
        if _is_no_file(file):
            return (None,)
        path = folder_paths.get_annotated_filepath(file)
        waveform, sample_rate = torchaudio.load(path)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


class WWAIExposeMesh(FileMarkerMixin):
    CATEGORY = INPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = (FILE_3D_TYPE,)
    RETURN_NAMES = ("value",)
    DESCRIPTION = (
        "Opens a 3D file and marks it as an input WWAI supplies. Leave the "
        f"file as \"{NO_FILE}\" to skip this reference for a run — the node "
        "outputs None and any optional downstream input it feeds is ignored."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (input_root_files(extensions=MESH_EXTENSIONS), {"file_upload": True}),
                **metadata_widgets(),
                **required_widget(),
            }
        }

    def expose(self, file, name, description, required):
        require_marker_name(name, type(self).__name__)
        if _is_no_file(file):
            return (None,)
        path = folder_paths.get_annotated_filepath(file)
        return (Types.File3D(path),)


# --- output markers ---------------------------------------------------------


class WWAIExposeOutputImage:
    CATEGORY = OUTPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Saves the finished image and marks it as WWAI's result. Writes PNG, "
        "or HDR-capable EXR / AVIF with the same encoders as Save Image (Advanced)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        # Optional, and after name/description: API prompts exported before
        # these existed carry no such keys and must still validate, and the
        # saved widgets_values positions of name/description stay put.
        formats = list(IMAGE_FORMATS) if IMAGE_ENCODER_IMPORT_ERROR is None else ["png"]
        return {
            "required": {"value": (IMAGE_TYPE,), **metadata_widgets()},
            "optional": {
                "format": (formats, {"default": "png"}),
                "bit_depth": (
                    list(IMAGE_BIT_DEPTHS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: 8-bit for PNG, 16-bit float for EXR, and for AVIF 8-bit "
                            "on sRGB / 10-bit on HDR.\n"
                            "png: 8-bit, 16-bit. exr: 16-bit float, 32-bit float. "
                            "avif: 8-bit YUV420, 10-bit YUV420."
                        ),
                    },
                ),
                "input_color_space": (
                    list(IMAGE_COLOR_SPACES),
                    {
                        "default": "sRGB",
                        "tooltip": (
                            "What the incoming image is, not what the file becomes.\n"
                            "png: sRGB.\n"
                            "exr: sRGB, HDR (HLG Rec.2020) or linear; always written scene-linear.\n"
                            "avif: sRGB, HDR (BT.2020/HLG) or HDR PQ (BT.2020/PQ)."
                        ),
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, format="png", bit_depth="auto", input_color_space="sRGB"):
        # Checked before the queue runs, so a wrong pairing fails in seconds
        # rather than after the whole workflow has rendered.
        if format not in IMAGE_FORMATS:
            return f"{cls.__name__}: unknown format {format!r}."
        depths, spaces = IMAGE_FORMATS[format]
        if bit_depth != "auto" and bit_depth not in depths:
            return f"{cls.__name__}: {format} cannot be written at {bit_depth!r}; choose one of {', '.join(depths)}."
        if input_color_space not in spaces:
            return f"{cls.__name__}: {format} does not take {input_color_space!r} input; choose one of {', '.join(spaces)}."
        return True

    def expose(self, value, name, description, format="png", bit_depth="auto", input_color_space="sRGB", prompt=None, extra_pnginfo=None):
        class_name = type(self).__name__
        require_marker_name(name, class_name)
        image = require_single(list(value), class_name, "image")

        verdict = self.VALIDATE_INPUTS(format, bit_depth, input_color_space)
        if verdict is not True:
            raise MarkerContractError(verdict)
        if bit_depth == "auto":
            bit_depth = IMAGE_FORMATS[format][0][0]

        if format != "png" or bit_depth != "8-bit":
            if IMAGE_ENCODER_IMPORT_ERROR is not None:
                raise RuntimeError(
                    f"{class_name}: this ComfyUI has no Save Image (Advanced) encoders "
                    f"(comfy_extras.nodes_images): {IMAGE_ENCODER_IMPORT_ERROR}"
                )
            path, entry = save_path(class_name, name, format)
            if format == "avif":
                metadata = dict(extra_pnginfo or {})
                if prompt is not None:
                    metadata["prompt"] = prompt
                _save_avif(image.unsqueeze(0), path, bit_depth, input_color_space, AVIF_CRF, metadata=metadata)
                return wwai_result(entry)
            encoded = _encode_image(image, format, bit_depth, input_color_space)
            if format == "exr":
                encoded = inject_exr_metadata(encoded, prompt, extra_pnginfo, input_color_space)
            else:
                encoded = inject_png_metadata(encoded, prompt, extra_pnginfo)
            with open(path, "wb") as handle:
                handle.write(encoded)
            return wwai_result(entry)

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        for key, entry in (extra_pnginfo or {}).items():
            metadata.add_text(key, json.dumps(entry))

        path, entry = save_path(class_name, name, "png")
        array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
        Image.fromarray(array).save(path, pnginfo=metadata, compress_level=4)
        return wwai_result(entry)


class WWAIExposeOutputVideo:
    """Saves the finished video, from either of ComfyUI's two video worlds.

    ComfyUI core passes a finished VIDEO object around (MiniMax and every other
    API video node emit one). Video Helper Suite — which most real workflows
    end in — has no VIDEO type at all: it carries IMAGE frames plus a separate
    AUDIO track, and its own VHS_VideoCombine both encodes and saves them.

    Accepting both is what lets this marker sit at the end of either kind of
    workflow. On the frames path it replaces VHS_VideoCombine outright rather
    than reading its VHS_FILENAMES list, which is a third-party structure whose
    "the real video is the last entry" ordering WWAI would have to guess at.
    """

    CATEGORY = OUTPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the finished video (or frames + audio) and marks it as WWAI's result."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": metadata_widgets(),
            "optional": {
                "video": (VIDEO_TYPE, {"tooltip": "A finished video, e.g. from MiniMax or Create Video."}),
                "images": (IMAGE_TYPE, {"tooltip": "Frames to encode, e.g. what Video Combine takes."}),
                "audio": (AUDIO_TYPE, {"tooltip": "Sound track. Only used with frames."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
        }

    def expose(self, name, description, video=None, images=None, audio=None, fps=30.0):
        class_name = type(self).__name__
        require_marker_name(name, class_name)

        if (video is None) == (images is None):
            raise MarkerContractError(
                f"{class_name}: connect EITHER 'video' (a finished video) OR "
                f"'images' (frames to encode) — not both, and not neither."
            )

        if video is None:
            video = InputImpl.VideoFromComponents(
                Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
            )
        elif audio is not None:
            # Muxing a separate track into a finished video means decoding and
            # re-encoding it. Refuse rather than quietly dropping the audio or
            # quietly degrading the video.
            raise MarkerContractError(
                f"{class_name}: 'audio' cannot be combined with a finished "
                f"'video' — a video already carries its own sound. Connect "
                f"frames to 'images' instead, or mux upstream."
            )

        container = Types.VideoContainer.AUTO
        path, entry = save_path(class_name, name, Types.VideoContainer.get_extension(container))
        video.save_to(path, format=container, codec=Types.VideoCodec.AUTO)
        return wwai_result(entry)


class WWAIExposeOutputAudio:
    CATEGORY = OUTPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the finished audio and marks it as WWAI's result."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (AUDIO_TYPE,),
                "format": (["flac", "mp3", "opus"], {"default": "flac"}),
                **metadata_widgets(),
            }
        }

    def expose(self, value, format, name, description):
        class_name = type(self).__name__
        require_marker_name(name, class_name)
        # ComfyUI's own encoder, so container and codec handling stay identical
        # to what its Save Audio nodes produce.
        results = UI.AudioSaveHelper.save_audio(
            value, FILENAME_PREFIX, IO.FolderType.output, cls=None, format=format
        )
        entry = require_single(results, class_name, "audio file")
        return wwai_result(dict(entry))


class WWAIExposeOutputMesh:
    CATEGORY = OUTPUT_CATEGORY
    FUNCTION = "expose"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the finished 3D model and marks it as WWAI's result."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (MESH_OUTPUT_TYPE,), **metadata_widgets()}}

    def expose(self, value, name, description):
        class_name = type(self).__name__
        require_marker_name(name, class_name)

        if isinstance(value, Types.File3D):
            # A splat or point-cloud .ply is not geometry. Copy the bytes and
            # keep the container; parsing it as a mesh would keep the
            # extension and throw the payload away.
            source = value.get_source()
            extension = value.format or extension_of(source if isinstance(source, str) else "", "glb")
            path, entry = save_path(class_name, name, extension)
            if isinstance(source, str):
                copy_file_verbatim(source, path)
            else:
                with open(path, "wb") as handle:
                    handle.write(value.get_bytes())
            return wwai_result(entry)

        if save_glb is None:
            raise RuntimeError(
                f"{class_name}: this ComfyUI has no GLB writer "
                f"(comfy_extras.nodes_save_3d): {GLB_WRITER_IMPORT_ERROR}"
            )

        index = require_single(list(range(value.vertices.shape[0])), class_name, "mesh")
        vertices, faces, vertex_colors, uvs = get_mesh_batch_item(value, index)
        texture = getattr(value, "texture", None)
        texture_image = None
        if texture is not None:
            array = (texture[index].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            texture_image = Image.fromarray(array, mode="RGB")

        path, entry = save_path(class_name, name, "glb")
        save_glb(
            vertices,
            faces,
            path,
            {},
            uvs=uvs,
            vertex_colors=vertex_colors,
            texture_image=texture_image,
            unlit=getattr(value, "unlit", False),
        )
        return wwai_result(entry)


# --- registration -----------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "WWAIExposeText": WWAIExposeText,
    "WWAIExposeInt": WWAIExposeInt,
    "WWAIExposeFloat": WWAIExposeFloat,
    "WWAIExposeBool": WWAIExposeBool,
    "WWAIExposeCombo": WWAIExposeCombo,
    "WWAIExposeImage": WWAIExposeImage,
    "WWAIExposeOutputImage": WWAIExposeOutputImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WWAIExposeText": "WWAI Expose Text",
    "WWAIExposeInt": "WWAI Expose Int",
    "WWAIExposeFloat": "WWAI Expose Float",
    "WWAIExposeBool": "WWAI Expose Bool",
    "WWAIExposeCombo": "WWAI Expose Combo",
    "WWAIExposeImage": "WWAI Expose Image",
    "WWAIExposeOutputImage": "WWAI Expose Output Image",
}

# The video and 3D markers need payload classes an older ComfyUI does not
# ship. Registering them anyway would put sockets in the picker that fail the
# moment anyone runs them; leaving them out with a named reason is the honest
# failure.
if MEDIA_PAYLOADS_IMPORT_ERROR is None:
    NODE_CLASS_MAPPINGS.update(
        {
            "WWAIExposeVideo": WWAIExposeVideo,
            "WWAIExposeMesh": WWAIExposeMesh,
            "WWAIExposeOutputVideo": WWAIExposeOutputVideo,
            "WWAIExposeOutputAudio": WWAIExposeOutputAudio,
            "WWAIExposeOutputMesh": WWAIExposeOutputMesh,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "WWAIExposeVideo": "WWAI Expose Video",
            "WWAIExposeMesh": "WWAI Expose Mesh",
            "WWAIExposeOutputVideo": "WWAI Expose Output Video",
            "WWAIExposeOutputAudio": "WWAI Expose Output Audio",
            "WWAIExposeOutputMesh": "WWAI Expose Output Mesh",
        }
    )
else:
    logger.warning(
        "WWAI markers: the video, audio-output and 3D nodes are unavailable "
        "because comfy_api.latest could not be imported (%s). Update ComfyUI "
        "to a build that ships comfy_api.latest (IO, UI, InputImpl, Types).",
        MEDIA_PAYLOADS_IMPORT_ERROR,
    )

if AUDIO_LOADER_IMPORT_ERROR is None:
    NODE_CLASS_MAPPINGS["WWAIExposeAudio"] = WWAIExposeAudio
    NODE_DISPLAY_NAME_MAPPINGS["WWAIExposeAudio"] = "WWAI Expose Audio"
else:
    logger.warning(
        "WWAI markers: the audio input node is unavailable because torchaudio "
        "could not be imported (%s) — the same dependency ComfyUI's own Load "
        "Audio node needs.",
        AUDIO_LOADER_IMPORT_ERROR,
    )
