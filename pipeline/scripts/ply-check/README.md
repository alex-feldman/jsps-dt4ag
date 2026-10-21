# ply-check

A read-only integrity check for exported Gaussian-splat PLY files, run after the
`export` stage or before you hand a file to someone. One file, `ply-check.py`.
It needs numpy; `--preview` also needs matplotlib. Both are in the pipeline
environment.

```bash
uv run python pipeline/scripts/ply-check/ply-check.py {data_root}/exports/{name}.ply
uv run python pipeline/scripts/ply-check/ply-check.py {ply} {ply} --json
uv run python pipeline/scripts/ply-check/ply-check.py {ply} --preview {dir}/centers.png
```

It never writes next to the PLY. The only file it can write is the PNG you name
with `--preview`.

## What it reports

For each file:

- **vertices, bytes, sha256.** The sha256 identifies the exact file, so a number
  quoted in a report can be matched to it later. `--no-hash` skips it.
- **non-finite values.** How many NaN or infinite values each float property
  holds. Any at all is a failure: a viewer or a later tool can choke on them.
- **opacity.** The median and the share above 0.5, after the sigmoid that turns
  the stored logit into an opacity.
- **scale.** The median and 99th percentile of each Gaussian's largest axis,
  after `exp`.
- **xyz 1-99 pct extent.** The size of the box that holds the middle 98 percent
  of the Gaussian centers, per axis, in scene units. Strays far from the subject
  do not inflate it.

The opacity, scale and extent lines appear only when the file has those
properties (the 3DGS names `opacity`, `scale_0..2`, `x`, `y`, `z`).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | every file is sound |
| `1` | at least one file has non-finite values, zero vertices, fewer vertex records than its header declares (truncated), or cannot be read (not a PLY, not binary little-endian, missing) |
| `2` | usage error |

With several files all are checked and the exit code is 1 if any one fails.

## JSON

`--json` prints one list with one object per file: `path`, `name`, `ok`,
`problems`, `bytes`, `vertices`, `properties`, `sha256`, `non_finite` (property to
count), `opacity_median`, `opacity_share_above_half`, `scale_median`, `scale_p99`
and `xyz_extent_1_99`. Values that are not finite numbers are written as null,
so the output is always valid JSON.

## The preview

`--preview {png}` (one PLY only) draws the Gaussian centers in three views,
X-Y, X-Z and Y-Z, colored from the spherical-harmonics base color. It keeps
Gaussians with opacity above 0.5, the central region only (the 5-95 percentile
box widened by 10 percent), and at most 250,000 of them, chosen with a fixed
seed so the same file gives the same picture.

**It is not a render.** It shows where Gaussians sit, not how the scene looks
from a camera: no size, no blending, no view-dependent color. Use it to spot a
file that is empty, flat, exploded or far off scale. Open the PLY in a splat
viewer to judge the look.

## Limits

- It checks that a file is sound, not that it is good. A file can pass and still
  be a poor reconstruction.
- Only `binary_little_endian` PLY is read, with the vertex element first and no
  list properties on it. That is what splat exporters write.
- The whole vertex block is read into memory (about the file size).

Tests: `pipeline/tests/test_ply_check.py` (synthetic PLYs written by the test;
skipped where numpy is not installed, so run the suite with the pipeline
environment's Python to include them).
