# Coordinate frames: projecting an export back into the photographs

Read this before any work that joins an exported splat to per-photograph data:
masks, instance labels, clicked points, or anything else drawn on the original
images. The tools under `scripts/organ-analysis/` already do this internally;
this page is for checking their output or writing your own projection.

Checked 2026-11-01 against the pinned nerfstudio 1.1.5 source
(`nerfstudio_dataparser.py`, `process_data_utils.py`, `colmap_utils.py` and
`scripts/exporter.py` in the installed package).

## Which frame each file is in

| File | Frame |
|---|---|
| `colmap/.../{run-id}/sparse/0/` | COLMAP's own frame |
| `colmap/.../{run-id}/transforms.json` (camera poses) and `sparse_pc.ply` | COLMAP's frame after `applied_transform`, the fixed axis swap `ns-process-data` writes into `transforms.json`. Poses are camera-to-world in OpenGL convention |
| the exported `.ply` | nerfstudio's dataparser frame: the trained gaussian centers as `ns-export gaussian-splat` writes them, from `model.means` with no transform |
| `outputs/.../splatfacto/{timestamp}/dataparser_transforms.json` | the `transform` and `scale` that map into the dataparser frame |

## Bringing a camera into the export's frame

The saved `transform` is **not** the matrix that was applied to the poses in
`transforms.json`. nerfstudio stores `transform @ applied_transform`, so that the
saved matrix maps COLMAP-frame points (`nerfstudio_dataparser.py`, where
`dataparser_transform_matrix` is built). To bring one `transforms.json` pose
`c2w` into the export's frame:

```python
import json
import numpy as np

tj = json.load(open("transforms.json"))
dp = json.load(open("dataparser_transforms.json"))

applied = np.eye(4)
applied[:3, :] = np.array(tj["applied_transform"])
t_saved = np.eye(4)
t_saved[:3, :] = np.array(dp["transform"])
to_export = t_saved @ np.linalg.inv(applied)       # the matrix the poses actually got

cameras = {}
for frame in tj["frames"]:
    c2w = to_export @ np.array(frame["transform_matrix"])
    c2w[:3, 3] *= dp["scale"]                      # scale the translation only
    c2w[:3, 1:3] *= -1                             # OpenGL to OpenCV camera axes
    cameras[frame["file_path"]] = c2w
```

Using `t_saved` directly on a pose puts every projected point outside every
photograph. The intrinsics (`fl_x`, `fl_y`, `cx`, `cy`) and distortion
coefficients (`k1`, `k2`, `p1`, `p2`) are at the top of `transforms.json`, in the
pixel units of the full-size images (`w`, `h`).

## Which photograph a frame is

`ns-process-data` copies the photographs into the workspace's `images/` and
renames them `frame_00001` onward in **sorted path order** of the directory it
was given (`list_images`, then `copy_images`). Frame N is the Nth photograph in
that order, and the `frames` array in `transforms.json` is not necessarily in
that order, so match frames by `file_path`, never by array position. For a
masked run the directory it was given holds the composites, which keep the
photographs' relative paths and file stems (as `.png`), so the order is the
same. `scripts/organ-analysis/` takes an
explicit `frame_map.json` instead of relying on this rule; build that map from
the sorted list and check it.

## Check the chain before using it

Project the gaussian centers into every photograph and count the share that
falls inside that photograph's subject mask. On one 24-photograph plant capture
(2026-10-14) the median was 97%. A photograph far below the median has a bad
mask or a bad pose; find out which before using its data. Use the masks the
reconstruction was actually run with, not an older or test copy.

## Other traps

- **`sparse_pc.ply` is ASCII; the splat export is binary** (little-endian). A
  reader that assumes one format fails on the other.
- **For visibility, splat a depth map; do not z-buffer the centers.** Centers
  are sparse, so points behind a surface show through it. Draw each gaussian as
  a disc of its projected middle scale.
- **Masks lifted across photographs cannot tell "same object" from "behind that
  object".** An occluded part projects into the mask of the part in front of it.
  Check contact in 3D before merging labels from several views.
