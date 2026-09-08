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


class FileMarkerMixin:
    """Shared validation for the markers that open a file.

    WWAI uploads the artist's file and submits in the same breath, so the file
    can post-date the directory listing INPUT_TYPES was built from. These two
    hooks are what let it validate anyway — the same pair core's LoadImage
    carries, for the same reason.
    """

    @classmethod
    def IS_CHANGED(cls, file, name, description):
        if _is_no_file(file):
            return NO_FILE
        path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(path), os.path.getsize(path)

    @classmethod
    def VALIDATE_INPUTS(cls, file, name, description):
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
        return {"required": {"value": ("STRING", {"multiline": True, "default": ""}), **metadata_widgets()}}

    def expose(self, value, name, description):
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
            }
        }

    def expose(self, value, name, description):
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
        return {"required": {"value": ("FLOAT", {"default": 0.0, "step": 0.01}), **metadata_widgets()}}

    def expose(self, value, name, description):
        require_marker_name(name, type(self).__name__)
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
            }
        }

    def expose(self, file, name, description):
        require_marker_name(name, type(self).__name__)
        if _is_no_file(file):
            return (None, None)
        path = folder_paths.get_annotated_filepath(file)
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
            }
        }

    def expose(self, file, name, description):
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
            }
        }

    def expose(self, file, name, description):
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
            }
        }

    def expose(self, file, name, description):
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
    DESCRIPTION = "Saves the finished image and marks it as WWAI's result."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (IMAGE_TYPE,), **metadata_widgets()},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def expose(self, value, name, description, prompt=None, extra_pnginfo=None):
        class_name = type(self).__name__
        require_marker_name(name, class_name)
        image = require_single(list(value), class_name, "image")

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
    "WWAIExposeImage": WWAIExposeImage,
    "WWAIExposeOutputImage": WWAIExposeOutputImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WWAIExposeText": "WWAI Expose Text",
    "WWAIExposeInt": "WWAI Expose Int",
    "WWAIExposeFloat": "WWAI Expose Float",
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
