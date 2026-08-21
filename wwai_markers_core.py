"""Stdlib-only rules shared by the WWAI marker nodes.

This module deliberately imports nothing from torch, PIL, numpy,
``folder_paths`` or ``comfy_api``. The contracts that are easiest to break and
most expensive to break — copying a Gaussian Splat file byte-for-byte, and
refusing a marker that would produce an unnamed or ambiguous parameter — are
therefore unit-testable on a bare Python, with no ComfyUI installed.
"""

import os
import shutil

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
