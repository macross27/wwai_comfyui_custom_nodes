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
        queued = {unannotated(value) for value in queued_input_values()}
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


def split_clicks(text):
    """Validate a WWAI click state (JSON text) and split it into five outputs.

    Returns (include_points, exclude_points, include_boxes, exclude_boxes,
    frame): points as JSON text of [{"x","y"}], boxes as lists of
    {x, y, width, height}, each kind in the artist's order. Same grammar as
    WWAI's _coerce_clicks, without its item-count cap.
    """
    if not isinstance(text, str):
        raise MarkerContractError(f"{CLICKS_INVALID}: the click state must be text")
    try:
        state = json.loads(text)
    except (ValueError, RecursionError):
        raise MarkerContractError(f"{CLICKS_INVALID}: the click state is not valid JSON") from None
    if not isinstance(state, dict) or set(state) != {"source", "frame", "items"}:
        raise MarkerContractError(
            f"{CLICKS_INVALID}: the click state must be an object with exactly source, frame and items"
        )
    if not isinstance(state["source"], str) or not _HEX64.fullmatch(state["source"]):
        raise MarkerContractError(f"{CLICKS_INVALID}: source must be 64 lower-case hex characters")
    frame = _click_int(state["frame"], "frame", 0)
    items = state["items"]
    if not isinstance(items, list) or not items:
        raise MarkerContractError(f"{CLICKS_INVALID}: items must be a non-empty list")
    points = {"include": [], "exclude": []}
    boxes = {"include": [], "exclude": []}
    for item in items:
        if not isinstance(item, dict) or item.get("kind") not in _ITEM_KEYS:
            raise MarkerContractError(f"{CLICKS_INVALID}: each item must have kind point or box")
        if set(item) != _ITEM_KEYS[item["kind"]]:
            raise MarkerContractError(f"{CLICKS_INVALID}: a {item['kind']} item has the wrong keys")
        if item["mode"] not in points:
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
        frame,
    )
