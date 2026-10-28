# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Part of DeMoSearch (class project generating personalized bouldering beta). `extract_holds.py` turns a Blender-exported GLB (Polycam LiDAR wall scan + one route's holds as separate objects) into per-hold features (JSON + CSV) for a physics-based planner.

## Commands

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python extract_holds.py route.glb --out out/route_holds.json [--plot | --plot-out out/view.png]
.venv/Scripts/python -m pytest tests -q                       # all tests
.venv/Scripts/python -m pytest tests -q -k z_up               # one test
.venv/Scripts/python tests/synthetic_wall.py                  # write tests/synthetic_wall.glb to inspect
```

## How extract_holds.py works

- Every `Params` dataclass field is automatically a `--kebab-case` CLI flag (`build_parser`); add new thresholds there and document them in the module docstring.
- Wall = `--wall`, else name contains "wall", else largest-area mesh. All other meshes are holds.
- Per hold: weld vertices (scans store 3 verts/face) -> make winding consistent per connected component -> RANSAC a local wall plane on wall face centers near the hold (excluding wall faces within `--exclude-margin` of any route hold, since Blender-duplicated holds remain inside the wall mesh) -> features in the plane's frame.
- Plane orientation is the subtle part: RANSAC candidates are oriented by the scan's face normals (they face the scanner = room) and rejected if the hold sits behind them (handles holds near panel folds). If most holds conflict with scan normals, the wall winding is treated as globally inverted and everything reruns.
- `tests/synthetic_wall.py` builds a GLB with known answers (−10°/−15° overhang, floor, other routes' holds, duplicated holds, node transforms incl. rotation, flipped faces, texture/vertex/no color). Tests compare against those answers; extend it when adding features.

## Data notes (wallC_overhang_right_marked.glb)

- glTF Y-up (Blender Z -> glTF Y); wall bottom ≈ 0 on Y (0.5th percentile −0.022 m, raw min −0.058 m from scan noise). Wall object is `Mesh_0`; holds `blue_v5_holds.001–.009`; room side is roughly −Z/−X.
- The scan has almost no upward-facing surface on overhang holds (phone was below them), so most holds get `no_grip_surface`.
