# wwai_comfyui_custom_nodes

ComfyUI custom nodes that let a workflow author mark, inside the ComfyUI
graph itself, which sockets should be controllable from outside ComfyUI and
which socket holds the finished result. Each node is a plain pass-through —
it forwards whatever value is wired into it — plus a `name` and
`description` text field an external tool can read back off the saved
workflow JSON.

## Install

Clone this repo into your ComfyUI installation's `custom_nodes` folder, then
restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/macross27/wwai_comfyui_custom_nodes.git
```

No extra Python dependencies.

## Nodes

Inputs (category `WWAI/Expose`):

| Node | Type |
|---|---|
| WWAI Expose Text | STRING |
| WWAI Expose Int | INT |
| WWAI Expose Float | FLOAT |
| WWAI Expose Image | IMAGE |
| WWAI Expose Video | any |
| WWAI Expose Mesh | any |

Outputs (category `WWAI/Expose Output`):

| Node | Type |
|---|---|
| WWAI Expose Output Image | IMAGE |
| WWAI Expose Output Video | any |
| WWAI Expose Output Mesh | any |

Every node has a `name` and `description` widget. Type a name and a short
note, then wire the node's output into wherever that value belongs in your
workflow.
