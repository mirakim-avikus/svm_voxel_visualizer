# NPZ Semantic Occupancy Viewer

A Three.js-based browser viewer for semantic occupancy NPZ files. Preprocesses
`(200, 200, 16)` uint8 `semantics` voxel grids into lightweight primitives
(cuboids, prisms, cylinders, ellipsoids, voxel point clouds) and serves them
as an interactive web app.

## Contents

| Path | Description |
| ---- | ----------- |
| `vis_npz_threejs.py` | Preprocessor + local server. Reads NPZ frames, extracts per-class primitives, writes `viewer_build/`, and serves it. |
| `occupancy_pred/` | Sample input NPZ frames (`{timestamp}.npz`; `_sym.npz` companions are skipped). |
| `3dboat.glb` | GLB asset used as the ego boat model and (decimated) detected-boat clones. |
| `viewer_build/` | Generated artifacts: `index.html`, `frames.json`, `voxels/*.bin`, vendored Three.js modules. |
| `requirements.txt` | Python dependency note (`opencv-python`). |

## Semantic class IDs

`1` poles · `2` docks · `3` persons · `4` boats · `5` buildings · `6` water.

## Setup

```bash
pip install numpy scipy opencv-python
```

The script also vendors Three.js `0.162.0` modules into `viewer_build/vendor/`
on first run (requires network access).

## Usage

Build and serve from a directory of NPZ files:

```bash
python vis_npz_threejs.py occupancy_pred/
```

Opens `http://localhost:8765/` in your browser.

Useful flags:

- `--out <dir>` — output directory (default `viewer_build`).
- `--port <n>` — HTTP port (default `8765`).
- `--no-browser` — don't auto-open the browser.
- `--no-serve` — build only.
- `--workers <n>` — parallel extract workers (default `cpu_count - 1`).
- `--html-only` — rewrite `index.html` only; reuse existing `frames.json` and
  `voxels/`. Use after JS-only edits.

## Viewer controls

- **Play/Pause, « / »** — playback at 10 FPS, or step a single frame.
- **BEV / Perspective** — top-down vs. tilted camera.
- **Boats: Model / Voxels / Cuboid / Ellipsoid** — cycle detected-boat
  rendering. *Model* uses a palette-coloured GLB clone, *Ellipsoid* shows a
  PCA-fit ellipsoid with a bow arrow.
- **Mask: Off / On** — toggle camera-mask filtering (frames with a
  `mask_camera` array only).
- **Scrub bar** — jump to any frame.
- **Numpad 4/6, 8/2** — Blender-style orbit (yaw / pitch).
- **Numpad +/-** — zoom in / out.
- **Numpad 5** — reset camera up to world +Z.

Docks render with a red→orange→white BEV gradient based on horizontal
distance from the ego (visual proximity cue).
