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
| WWAI Expose Bool | `BOOLEAN` |
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


### Required, and leaving a reference empty

Every input marker carries a **required** tick box (default on). This is the
one and only thing that decides whether WWAI makes an artist fill the input
before a run: ticked, WWAI refuses to run until a value is supplied; unticked,
WWAI runs with whatever value is saved in the node.

This is a change from how WWAI used to behave: it used to *infer*
requiredness from the `file` dropdown — a file marker left on `None` was
optional, a scalar marker (text, number) was always optional and could never
be made mandatory, and simply clearing a file silently turned a needed input
into an optional one. The tick box replaces all of that guessing with an
explicit author choice, for every input marker alike.

`None` still means what it always meant on a file-based input marker's `file`
dropdown (**Expose Image**, **Expose Audio**, **Expose Video**, **Expose
Mesh**): it lists `None` as the first choice, a fresh node defaults to it, and
picking it skips the file entirely inside the ComfyUI graph — the node
returns `None` on every output (`fps` returns `0.0` on Expose Video, since it
is a plain number) rather than a real payload, and WWAI writes `None` into
that slot when an artist leaves an *unticked* file input empty. It is no
longer what makes the input optional — the tick box is.

This is what lets one workflow wire the *full* set of optional reference
slots a downstream node exposes (e.g. MiniMax H3 Reference to Video's up to 9
image / 3 video / 3 audio references) and skip whichever ones a given run has
no data for, just by leaving those markers on `None` — no rewiring, no
deleting nodes from the exported API JSON. It only works because the
downstream socket is declared optional and its node checks for `None`;
connecting a marker left on `None` into a *required* input still fails, as it
always did.

Every node has a **name** and **description** field, and every input marker
adds the **required** tick box above as a third. The name must not be
blank and must be unique in the workflow — a blank one fails the run rather
than reaching WWAI as an unnamed parameter. Leave all three as widgets;
converting one into a wired input replaces its value with a link, and WWAI's
upload check rejects that.

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
real node code, pinning the socket type strings, the terminal-node flag, and
the required-toggle contract — every input marker declares it (default on),
no output marker does.
Verification in a real ComfyUI is still required for registration and socket
compatibility.
