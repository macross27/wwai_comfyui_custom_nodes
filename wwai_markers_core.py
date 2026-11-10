"""Stdlib-only rules shared by the WWAI marker nodes.

This module deliberately imports nothing from torch, PIL, numpy,
``folder_paths`` or ``comfy_api``. The contracts that are easiest to break and
most expensive to break — copying a Gaussian Splat file byte-for-byte, and
refusing a marker that would produce an unnamed or ambiguous parameter — are
therefore unit-testable on a bare Python, with no ComfyUI installed.
"""

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
from collections import namedtuple

# Containers that carry Gaussian Splat / point-cloud attributes rather than
# triangle geometry. Round-tripping one of these through a mesh serializer
# keeps the extension but silently discards the payload, so these files are
# only ever copied verbatim.
SPLAT_CONTAINERS = frozenset({"ply", "splat", "spz", "ksplat"})


class MarkerContractError(ValueError):
    """A WWAI marker node was used in a way its contract forbids.

    Raised — never swallowed, never defaulted around — so ComfyUI surfaces the
    reason in the UI and the prompt fails instead of producing a result WWAI
    would go on to publish as if it were correct.
    """


def require_marker_name(name, class_name):
    """Return the marker's trimmed name, or refuse the run.

    A blank name reaches WWAI as an unnamed parameter, which no admin can map
    and no artist can fill. It is rejected here, at the point the name is
    supplied, rather than downstream where the cause is invisible.
    """
    trimmed = (name or "").strip()
    if not trimmed:
        raise MarkerContractError(
            f"{class_name}: the 'name' field is empty. Type a short unique "
            f"name — WWAI uses it as the parameter's label."
        )
    return trimmed


def require_single(items, class_name, what):
    """Return the one element of ``items``, or refuse the run.

    An output marker stands for exactly one finished result: WWAI's node has a
    single output port and publishes a single asset. Silently taking the first
    of several would publish an arbitrary one of them.
    """
    count = len(items)
    if count != 1:
        raise MarkerContractError(
            f"{class_name}: expected exactly 1 {what}, got {count}. An output "
            f"marker stands for one finished result — split a batch before it "
            f"reaches the marker."
        )
    return items[0]


def extension_of(path, fallback):
    """Return the lowercase extension of ``path`` without its dot."""
    suffix = os.path.splitext(str(path))[1].lstrip(".").lower()
    return suffix or fallback


def copy_file_verbatim(src_path, dest_path):
    """Copy ``src_path`` to ``dest_path`` byte-for-byte.

    Never parses, decodes or re-encodes. This is what keeps a Gaussian Splat
    ``.ply`` a valid splat instead of a stripped triangle soup that still ends
    in ``.ply``.
    """
    if os.path.abspath(src_path) == os.path.abspath(dest_path):
        return dest_path
    parent = os.path.dirname(os.path.abspath(dest_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return dest_path


# Name grammars shared with WWAI's backend (a sibling-checkout test compares
# these two literals with WWAI's own). Keep them as plain string literals.
STAGED_NAME_PATTERN = r"wwai/wwai-[0-9a-f]{64}\.[a-z0-9]{1,16}"
STAGED_DOOR_NAME_PATTERN = r"wwai-[0-9a-f]{64}( \([1-9][0-9]{0,5}\))?\.[a-z0-9]{1,16}"

_STAGED_NAME = re.compile(STAGED_NAME_PATTERN)
_DOOR_NAME = re.compile(STAGED_DOOR_NAME_PATTERN)
_MIN_SECRET_CHARS = 32


def is_staged_name(name):
    """True for a name WWAI staged (``wwai/wwai-<sha256>.<ext>``)."""
    return isinstance(name, str) and _STAGED_NAME.fullmatch(name) is not None


def check_staged_file(path, name):
    """Refuse a staged file that is missing or whose bytes do not match its name.

    A staged name carries the sha256 of its bytes. A truncated upload, or any
    other writer to ComfyUI's unauthenticated upload route, can leave a file
    under that name with other bytes: remove it and raise, so WWAI stages a
    fresh copy. Author files (any other name) are not checked.
    """
    if not is_staged_name(name):
        return
    expected = name.split("/", 1)[1][len("wwai-") :].split(".", 1)[0]
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError:
        reason = "missing"
    else:
        if digest.hexdigest() == expected:
            return
        reason = "mismatch"
        try:
            os.unlink(path)
        except OSError as exc:
            raise MarkerContractError(
                f"WWAI_STAGED_FILE_UNUSABLE: {name} ({reason})"
            ) from exc
    raise MarkerContractError(f"WWAI_STAGED_FILE_UNUSABLE: {name} ({reason})")


# POSIX can pin the folder with an fd and act inside it; Windows cannot.
_DIR_FD = os.stat in os.supports_dir_fd and os.unlink in os.supports_dir_fd


def read_secret(secret_path):
    """The door's shared secret, or "" when unreadable. Windows editors and
    PowerShell's ``>`` write UTF-8 with a BOM or UTF-16, so both are read."""
    try:
        with open(secret_path, "rb") as handle:
            raw = handle.read()
        encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        return raw.decode(encoding).strip()
    except (OSError, ValueError):
        return ""


# Windows junctions and symlinks are reparse points; lstat reports a junction
# as a plain folder, so the attribute is the only tell. Other reparse points
# (OneDrive and dedup placeholders) are ordinary files, so only the link tags count.
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_LINK_REPARSE_TAGS = (
    getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
    getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
)


def _is_reparse_point(st):
    """True only for a link (symlink or junction), never a cloud or dedup placeholder."""
    return bool(getattr(st, "st_file_attributes", 0) & _REPARSE_POINT) and (
        getattr(st, "st_reparse_tag", 0) in _LINK_REPARSE_TAGS
    )


# ComfyUI's folder_paths.annotated_filepath: a value ending in "[input]",
# "[output]" or "[temp]" names the file without that tag and the one character
# (the space) before it.
_ANNOTATIONS = ("[input]", "[output]", "[temp]")


def unannotated(value):
    """*value* as the file name ComfyUI resolves it to, annotation removed."""
    for tag in _ANNOTATIONS:
        if value.endswith(tag):
            return value[: -len(tag) - 1]
    return value


def _delete_entry(answer, path, **dir_fd):
    try:
        st = os.lstat(path, **dir_fd)
    except FileNotFoundError:
        return answer(404, "absent")
    except OSError as exc:
        return answer(500, "failed", detail=str(exc))
    if not stat.S_ISREG(st.st_mode) or _is_reparse_point(st):
        return answer(400, "not_regular")
    try:
        os.unlink(path, **dir_fd)
    except FileNotFoundError:
        return answer(404, "absent")
    except OSError as exc:
        return answer(500, "failed", detail=str(exc))
    return answer(200, "deleted")


def staged_file_delete(
    input_dir, secret_path, header_secret, filename, queued_input_values
):
    """Delete one staged file; return ``(http_status, body)``.

    The value of ``wwai_staged_file`` is the outcome word, so WWAI can tell this door's typed
    404 from a ComfyUI that has no such route. ``queued_input_values()`` is called
    after the secret and name checks and returns every string input value of
    the queued and running prompts: a file one of them names (annotated or
    not) is refused as ``in_use``, so a late delete cannot pull a file out from
    under a prompt that already passed its marker's hash check. A queue that
    cannot be read is a 500 ``failed``: nothing is deleted blind.
    """

    def answer(status, word, **extra):
        return status, {"wwai_staged_file": word, **extra}

    secret = read_secret(secret_path)
    if len(secret) < _MIN_SECRET_CHARS:
        return answer(503, "not_configured")
    if not isinstance(header_secret, str):
        return answer(403, "forbidden")
    try:
        # A header with undecodable bytes arrives surrogate-escaped and cannot encode.
        header_bytes = header_secret.encode("utf-8")
    except UnicodeEncodeError:
        return answer(403, "forbidden")
    if not hmac.compare_digest(header_bytes, secret.encode("utf-8")):
        return answer(403, "forbidden")
    if not isinstance(filename, str) or _DOOR_NAME.fullmatch(filename) is None:
        return answer(400, "refused")
    try:
        values = queued_input_values()
        # A Ripple job names its files inside one string value: count those too.
        queued = {unannotated(value) for value in values}.union(*map(ripple_manifest_names, values))
    except Exception as exc:  # the host's queue failed: refuse, typed and loud
        return answer(500, "failed", detail=f"cannot read the prompt queue: {exc}")
    if f"wwai/{filename}" in queued:
        return answer(409, "in_use")
    folder = os.path.join(input_dir, "wwai")
    try:
        folder_st = os.lstat(folder)
    except FileNotFoundError:
        return answer(404, "absent")
    except OSError as exc:
        return answer(500, "failed", detail=str(exc))
    if not stat.S_ISDIR(folder_st.st_mode) or _is_reparse_point(folder_st):
        return answer(400, "not_regular", detail=f"{folder} is a link or not a folder")
    if not _DIR_FD:
        # Deliberate limit: without dir_fd (Windows) the folder check and the
        # unlink are two steps; swapping the folder between them needs write
        # access to the machine's disk, which can already delete these files.
        return _delete_entry(answer, os.path.join(folder, filename))
    # O_NOFOLLOW closes the gap between the lstat above and this open.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(folder, flags)
    except OSError as exc:
        return answer(500, "failed", detail=str(exc))
    try:
        return _delete_entry(answer, filename, dir_fd=fd)
    finally:
        os.close(fd)


# --- clicks -----------------------------------------------------------------

CLICKS_INVALID = "COMFYUI_CLICKS_INVALID"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ITEM_KEYS = {
    "point": {"kind", "mode", "x", "y"},
    "box": {"kind", "mode", "x", "y", "width", "height"},
}


def _click_int(value, what, minimum=None):
    """An int, or an integer-valued float (as WWAI's _coerce_int accepts)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarkerContractError(f"{CLICKS_INVALID}: {what} must be a whole number")
    if isinstance(value, float):
        if not value.is_integer():
            raise MarkerContractError(f"{CLICKS_INVALID}: {what} must be a whole number")
        value = int(value)
    if minimum is not None and value < minimum:
        raise MarkerContractError(f"{CLICKS_INVALID}: {what} must be at least {minimum}")
    return value


def split_items(items):
    """Validate click items; return (include_points, exclude_points, include_boxes,
    exclude_boxes): points as JSON text of [{"x","y"}], boxes as lists of
    {x, y, width, height}, each kind in the artist's order."""
    points = {"include": [], "exclude": []}
    boxes = {"include": [], "exclude": []}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str) or item["kind"] not in _ITEM_KEYS:
            raise MarkerContractError(f"{CLICKS_INVALID}: each item must have kind point or box")
        if set(item) != _ITEM_KEYS[item["kind"]]:
            raise MarkerContractError(f"{CLICKS_INVALID}: a {item['kind']} item has the wrong keys")
        if not isinstance(item["mode"], str) or item["mode"] not in points:
            raise MarkerContractError(f"{CLICKS_INVALID}: mode must be include or exclude")
        x = _click_int(item["x"], "x")
        y = _click_int(item["y"], "y")
        if item["kind"] == "point":
            points[item["mode"]].append({"x": x, "y": y})
        else:
            boxes[item["mode"]].append({
                "x": x,
                "y": y,
                "width": _click_int(item["width"], "width", 1),
                "height": _click_int(item["height"], "height", 1),
            })
    return (
        json.dumps(points["include"]),
        json.dumps(points["exclude"]),
        boxes["include"],
        boxes["exclude"],
    )


# --- keyframes --------------------------------------------------------------
#
# The keyframes text: {source, keyframes: [{frame, object, items}], range?: {in, out}}.
# Object 0 is the prompt's layer (its items refine the prompt and may be empty);
# objects 1..N are clicks only. range is the frames a run outputs, In <= Out;
# absent, the whole clip. Grammar only: no count limits (WWAI caps size).

Keyframe = namedtuple("Keyframe", "frame object items")
# One keyframe's share of a layer: output frames first..last take their mask
# from the track started at key; calls are the (start, end) SAM3_VideoTrack
# runs, end < start meaning backward over the reversed frames.
Span = namedtuple("Span", "key items first last calls")


def _keyframes_invalid(reason, detail):
    return MarkerContractError(f"{CLICKS_INVALID}: {reason}: {detail}")


def _whole(value, what):
    try:
        return _click_int(value, what, 0)
    except MarkerContractError as exc:
        raise _keyframes_invalid("malformed", str(exc).split(": ", 1)[1]) from None


def parse_keyframes(text):
    """The keyframes text as ([Keyframe], (in, out) or None). Refuses malformed
    text, In after Out and two keyframes with the same frame and object."""
    if not isinstance(text, str):
        raise _keyframes_invalid("malformed", "the keyframes state must be text")
    try:
        state = json.loads(text)
    except (ValueError, RecursionError):
        raise _keyframes_invalid("malformed", "the keyframes state is not valid JSON") from None
    if not isinstance(state, dict) or not {"source", "keyframes"} <= set(state) <= {"source", "keyframes", "range"}:
        raise _keyframes_invalid("malformed", "the state must have source, keyframes and an optional range")
    if not isinstance(state["source"], str) or not _HEX64.fullmatch(state["source"]):
        raise _keyframes_invalid("malformed", "source must be 64 lower-case hex characters")
    if not isinstance(state["keyframes"], list):
        raise _keyframes_invalid("malformed", "keyframes must be a list")
    keyframes, seen = [], set()
    for entry in state["keyframes"]:
        if not isinstance(entry, dict) or set(entry) != {"frame", "object", "items"}:
            raise _keyframes_invalid("malformed", "each keyframe has exactly frame, object and items")
        frame, obj = _whole(entry["frame"], "frame"), _whole(entry["object"], "object")
        if not isinstance(entry["items"], list):
            raise _keyframes_invalid("malformed", "items must be a list")
        try:
            split_items(entry["items"])
        except MarkerContractError as exc:
            raise _keyframes_invalid("malformed", str(exc).split(": ", 1)[1]) from None
        if any(item["kind"] == "box" and item["mode"] == "exclude" for item in entry["items"]):
            # SAM3_Detect takes no exclude box; refuse rather than drop it.
            raise _keyframes_invalid("exclude_box_unsupported", f"object {obj} at frame {frame} has an exclude box")
        if (frame, obj) in seen:
            raise _keyframes_invalid("repeated_keyframe", f"object {obj} has two keyframes at frame {frame}")
        seen.add((frame, obj))
        keyframes.append(Keyframe(frame, obj, entry["items"]))
    frame_range = None
    if "range" in state:
        rng = state["range"]
        if not isinstance(rng, dict) or set(rng) != {"in", "out"}:
            raise _keyframes_invalid("malformed", "range has exactly in and out")
        frame_range = (_whole(rng["in"], "In"), _whole(rng["out"], "Out"))
        if frame_range[0] > frame_range[1]:
            raise _keyframes_invalid("in_after_out", f"In {frame_range[0]} is after Out {frame_range[1]}")
    return keyframes, frame_range


# The Ripple job text a WWAIExposeRippleJob reads (WWAI HRP0 schema v1). The
# resize methods and bounds are From Frames' own (WepeNerd ripple_core.py /
# ripple_node.py); a value outside them is refused here, never clamped.
RIPPLE_JOB_INVALID = "WWAI_RIPPLE_JOB_INVALID"
RIPPLE_MAX_TEXT_BYTES = 16 * 1024
RIPPLE_MAX_EDITS = 32
RIPPLE_RESIZE_METHODS = ("lanczos", "bicubic", "bilinear", "nearest-exact", "area")
_RIPPLE_KEYS = {"version", "clip", "length", "megapixels", "allow_upscale", "resize_method", "output_width", "output_height", "markers"}


def _ripple_invalid(detail):
    return MarkerContractError(f"{RIPPLE_JOB_INVALID}: {detail}")


def _ripple_int(value, what, low, high):
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise _ripple_invalid(f"{what} must be a whole number from {low} to {high}")
    return value


def parse_ripple_manifest(text):
    """The Ripple job text as a dict (schema in plans/2419 HRP0). Refuses text over
    16 KiB before parsing, then anything outside the schema; nothing is clamped."""
    if not isinstance(text, str):
        raise _ripple_invalid("the job must be text")
    if len(text.encode("utf-8")) > RIPPLE_MAX_TEXT_BYTES:
        raise _ripple_invalid(f"the job text is over {RIPPLE_MAX_TEXT_BYTES} bytes")
    try:
        job = json.loads(text)
    except (ValueError, RecursionError):
        raise _ripple_invalid("the job is not valid JSON") from None
    if not isinstance(job, dict) or set(job) != _RIPPLE_KEYS:
        raise _ripple_invalid("the job must have exactly " + ", ".join(sorted(_RIPPLE_KEYS)))
    if job["version"] != 1 or isinstance(job["version"], bool):
        raise _ripple_invalid("version must be 1")
    if not is_staged_name(job["clip"]):
        raise _ripple_invalid("clip must be a staged file name")
    length = _ripple_int(job["length"], "length", 5, 3592)
    if (length - 5) % 17:
        raise _ripple_invalid("length must be 17k + 5 frames")
    megapixels = job["megapixels"]
    if isinstance(megapixels, bool) or not isinstance(megapixels, (int, float)) or not 0.001 <= megapixels <= 268.435456:
        raise _ripple_invalid("megapixels must be a number from 0.001 to 268.435456")
    if not isinstance(job["allow_upscale"], bool):
        raise _ripple_invalid("allow_upscale must be true or false")
    if job["resize_method"] not in RIPPLE_RESIZE_METHODS:
        raise _ripple_invalid("resize_method must be one of " + ", ".join(RIPPLE_RESIZE_METHODS))
    width = _ripple_int(job["output_width"], "output_width", 0, 16384)
    height = _ripple_int(job["output_height"], "output_height", 0, 16384)
    if (width == 0) != (height == 0):
        raise _ripple_invalid("output_width and output_height must both be set or both be 0")
    markers = job["markers"]
    if not isinstance(markers, list) or not 1 <= len(markers) <= RIPPLE_MAX_EDITS:
        raise _ripple_invalid(f"markers must be a list of 1 to {RIPPLE_MAX_EDITS} edits")
    frames = set()
    for marker in markers:
        if not isinstance(marker, dict) or set(marker) != {"frame", "edited"}:
            raise _ripple_invalid("each marker has exactly frame and edited")
        frame = _ripple_int(marker["frame"], "frame", 0, length - 1)
        if frame in frames:
            raise _ripple_invalid(f"two edits are for frame {frame}")
        frames.add(frame)
        if not is_staged_name(marker["edited"]):
            raise _ripple_invalid("edited must be a staged file name")
    return job


def ripple_manifest_names(text):
    """The staged names a queued value holds if it is a Ripple job, else an empty set."""
    try:
        job = parse_ripple_manifest(text)
    except MarkerContractError:
        return set()
    return {job["clip"], *(marker["edited"] for marker in job["markers"])}


def check_keyframes(parsed, frame_count, prompt_set):
    """Refuse what parsed keyframes cannot mean on a clip of frame_count frames
    (a picture: 1) with the prompt set or blank; return (first, last) output frames."""
    keyframes, frame_range = parsed
    if frame_count < 1:
        raise _keyframes_invalid("empty_clip", "the picture or movie has no frames")
    for kf in keyframes:
        if kf.frame >= frame_count:
            if frame_count == 1:
                raise _keyframes_invalid("picture_frame_not_zero", f"a picture has only frame 0, not {kf.frame}")
            raise _keyframes_invalid("frame_outside_clip", f"frame {kf.frame} is past the last frame {frame_count - 1}")
        if kf.object == 0 and not prompt_set:
            raise _keyframes_invalid("prompt_keyframe_without_prompt", f"object 0 at frame {kf.frame} refines a prompt, and the prompt is blank")
        if kf.object > 0 and not kf.items:
            raise _keyframes_invalid("click_object_without_clicks", f"object {kf.object} at frame {kf.frame} has no clicks")
    if not prompt_set and not keyframes:
        raise _keyframes_invalid("nothing_to_track", "there is no prompt and no click")
    if frame_range is None:
        return 0, frame_count - 1
    if frame_range[1] >= frame_count:
        raise _keyframes_invalid("range_outside_clip", f"Out {frame_range[1]} is past the last frame {frame_count - 1}")
    return frame_range


def track_calls(key, first, last):
    """The SAM3_VideoTrack runs that give frames first..last from a track keyed
    at key: forward over key..last when last > key, backward over key..first
    when first < key, one one-frame run when first == last == key."""
    calls = []
    if last > key:
        calls.append((key, last))
    if first < key:
        calls.append((key, first))
    return tuple(calls) or ((key, key),)


def _spans(keys, first, last):
    """Each output frame takes its nearest key (a tie goes to the earlier one);
    a key that no output frame takes gets no span."""
    keys = sorted(keys, key=lambda kf: kf[0])
    owned = {}
    for frame in range(first, last + 1):
        best = min(keys, key=lambda kf: (abs(frame - kf[0]), kf[0]))
        owned.setdefault(best[0], [best[1], frame, frame])[2] = frame
    return [Span(key, items, lo, hi, track_calls(key, lo, hi)) for key, (items, lo, hi) in sorted(owned.items())]


def plan_tracking(text, frame_count, prompt_set):
    """((first, last), [(object, [Span])]): the layers a run tracks, object 0
    (the prompt) first when the prompt is set. A prompt with no object-0
    keyframe is keyed at frame 0 with no clicks (items None)."""
    parsed = parse_keyframes(text)
    first, last = check_keyframes(parsed, frame_count, prompt_set)
    by_object = {}
    for kf in parsed[0]:
        by_object.setdefault(kf.object, []).append((kf.frame, kf.items))
    if prompt_set:
        by_object.setdefault(0, [(0, None)])
    layers = [(obj, _spans(keys, first, last)) for obj, keys in sorted(by_object.items())]
    return (first, last), layers


def tracked_frames(text, frame_count, prompt_set):
    """Frames SAM3_VideoTrack steps through, summed over every run of every layer."""
    _, layers = plan_tracking(text, frame_count, prompt_set)
    return sum(abs(end - start) + 1 for _, spans in layers for span in spans for start, end in span.calls)
