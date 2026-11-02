from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# The delete door needs ComfyUI's server; it is absent in unit tests (no
# ComfyUI), where only this import is skipped.
try:
    from server import PromptServer  # noqa: F401
except ImportError:
    PromptServer = None
if PromptServer is not None:
    from . import staged_files_route  # noqa: F401  (registers the route)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
