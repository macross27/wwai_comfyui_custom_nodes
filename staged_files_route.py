"""The delete door WWAI's staged-file cleaner calls (all rules live in
``wwai_markers_core.staged_file_delete``). Registered from ``__init__.py``."""

import os

import folder_paths
from aiohttp import web
from server import PromptServer

from .wwai_markers_core import staged_file_delete

SECRET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".wwai_staged_file_secret")


def queued_input_values():
    """Every string input value of the running and pending prompts.

    ``PromptQueue.get_current_queue()`` returns ``(running, pending)``; each
    item is a tuple whose index 2 is the prompt ``{node_id: {"inputs": ...}}``.
    """
    running, pending = PromptServer.instance.prompt_queue.get_current_queue()
    return {
        value
        for item in (*running, *pending)
        for node in item[2].values()
        for value in node.get("inputs", {}).values()
        if isinstance(value, str)
    }


@PromptServer.instance.routes.post("/wwai/staged-file/delete")
async def delete_staged_file(request):
    try:
        payload = await request.json()
    except ValueError:
        payload = None
    filename = payload.get("filename") if isinstance(payload, dict) else None
    status, body = staged_file_delete(
        folder_paths.get_input_directory(),
        SECRET_PATH,
        request.headers.get("X-WWAI-Staged-File-Secret"),
        filename,
        queued_input_values,
    )
    return web.json_response(body, status=status)
