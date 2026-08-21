# wwai_comfyui_custom_nodes

ComfyUI custom nodes that let a workflow author mark, inside the ComfyUI graph
itself, which values should be controllable from outside ComfyUI and which one
holds the finished result. WWAI reads those marks back out of the exported
workflow JSON, so nobody has to hand-type a parameter list.

These are not labels. An **input marker** opens a real file and hands out the
payload its consumer expects — an `IMAGE` tensor, ComfyUI's native `VIDEO`
object, a `FILE_3D`. An **output marker** writes the finished file and reports
it under `wwai_result`, a key only these nodes emit, so WWAI can pick this
job's result out of `/history/{prompt_id}` without scanning folders.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/macross27/wwai_comfyui_custom_nodes.git
```

Restart ComfyUI. No extra Python dependencies.

**Requires** a ComfyUI that ships `comfy_api.latest` (`IO`, `UI`, `InputImpl`,
`Types`) — the same build that provides `Load3D`, `SaveGLB` and the native
`VIDEO` type. The audio input node additionally needs `torchaudio`, the same
dependency ComfyUI's own Load Audio node has. On a build missing either, the
affected nodes are skipped with a warning naming the reason and the rest still
load; nothing is ever registered in a state where it would fail at run time.

## Nodes

Inputs — category `WWAI/Expose`. The file/value slot is named `file` on the
media markers and `value` on the scalar ones; WWAI rewrites exactly that key.

| Node | Emits |
|---|---|
| WWAI Expose Text | `STRING` |
| WWAI Expose Int | `INT` |
| WWAI Expose Float | `FLOAT` |
| WWAI Expose Image | `IMAGE` + `MASK` |
| WWAI Expose Audio | `AUDIO` |
| WWAI Expose Video | `video`, `images`, `audio`, `fps` — see below |
| WWAI Expose Mesh | `FILE_3D` |

Outputs — category `WWAI/Expose Output`. Each is a terminal node that saves
one file.

| Node | Accepts | Writes |
|---|---|---|
| WWAI Expose Output Image | `IMAGE` | `.png` |
| WWAI Expose Output Audio | `AUDIO` | `.flac`, `.mp3` or `.opus` |
| WWAI Expose Output Video | `video`, **or** `images` + `audio` + `fps` | `.mp4` |
| WWAI Expose Output Mesh | `MESH` or any of ComfyUI's 13 `FILE_3D_*` types | `.glb` from geometry; the source file copied verbatim otherwise |

### The two video worlds

ComfyUI core passes a finished `VIDEO` object around — MiniMax and the other
API video nodes emit one. [Video Helper
Suite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite), which most
real workflows use, has no `VIDEO` type at all. It works in `IMAGE` frames
plus a separate `AUDIO` track at both ends:

- `VHS_LoadVideo` returns `(IMAGE, frame_count, AUDIO, VHS_VIDEOINFO)`
- `VHS_VideoCombine` takes `images` + optional `audio` + `frame_rate`

**Both WWAI video markers speak both worlds**, so they can start and end
either kind of workflow.

**Expose Video** splits the file the way core's `GetVideoComponents` does, and
you wire whichever port your graph speaks:

```
                        ┌─► video   ─► MiniMax, Create Video
WWAI Expose Video ──────┼─► images  ─┐
                        ├─► audio   ─┼─► anything VHS-shaped
                        └─► fps     ─┘
```

Splitting costs one decode of the file into frames — the same cost
`VHS_LoadVideo` already pays. ComfyUI caches it per unique input. `audio` is
`None` when the file has no sound track.

**Expose Output Video** accepts either shape:

```
MiniMax ──────────────────────► video ─┐
                                       ├─► WWAI Expose Output Video ─► .mp4
VAE Decode ──► images ─┐               │
Expose Audio ──► audio ─┴─► fps ───────┘
```

Connect **either** `video` **or** `images`, never both. `audio` applies only
to the frames path — a finished video already carries its own sound, so
passing both is refused rather than silently dropped or re-encoded.

On the frames path this node **replaces** `VHS_VideoCombine`, rather than
sitting after it. Video Combine returns a `VHS_FILENAMES` list holding a
metadata PNG, a silent video and an "-audio" mux, where the real result is
whichever entry happens to be last — a third-party ordering convention WWAI
would have to guess at. Writing the file here removes the guess.


Every node has a **name** and **description** field. The name must not be
blank and must be unique in the workflow — a blank one fails the run rather
than reaching WWAI as an unnamed parameter. Leave both as widgets; converting
one into a wired input replaces its text with a link, and WWAI's upload check
rejects that.

### Gaussian Splats and point clouds

A splat `.ply` is not geometry. Feed one through **Expose Mesh** → **Expose
Output Mesh** and it is copied byte-for-byte with its container intact —
never parsed as a mesh, which would keep the `.ply` extension and throw the
splat attributes away. The same holds for `.splat`, `.spz`, `.ksplat` and
point-cloud `.ply`.

## Tests

```
python -m unittest
```

Runs anywhere — no ComfyUI, no torch. `test_wwai_markers_core.py` covers the
byte-for-byte copy and the name/cardinality rules on their own;
`test_nodes_contract.py` stubs ComfyUI's four host modules and exercises the
real node code, pinning the socket type strings and the terminal-node flag.
Verification in a real ComfyUI is still required for registration and socket
compatibility.
