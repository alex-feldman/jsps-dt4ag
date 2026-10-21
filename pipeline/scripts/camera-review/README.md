# camera-review

A look at one COLMAP camera solve, to run after the `colmap` and `process`
stages and **before training**. It reports; it does not accept or reject. One
file, `camera-review.py`. It needs the pipeline environment (nerfstudio, numpy,
matplotlib) because it reads COLMAP's binary model with nerfstudio's parser.

```bash
uv run python pipeline/scripts/camera-review/camera-review.py \
    --capture {data_root}/datasets/{capture} \
    --workspace {data_root}/colmap/{capture}/{run-id} \
    --out {data_root}/evals/{capture}/camera-review
```

It reads the capture and the workspace and writes only into `--out`.

| Option | Meaning |
|---|---|
| `--capture` | The capture directory, the one holding `images/`. |
| `--workspace` | The run's COLMAP folder, the one that holds `sparse/0`. |
| `--out` | Where `camera-checks.json` and `camera-review.png` go. Created if needed. |
| `--label` | Title for the plot. Default: the workspace folder's name. |
| `--model` | Which folder under `sparse/` to review. Default `0`, the largest model COLMAP made. |

## Frame times

If the capture has `provenance/frames.csv` (written by `video-frames.py
extract`), the expected frame list comes from it and the plot is colored by clip
time, using `source_seconds_approx` or else `output_grid_seconds`. Otherwise the
expected frames are the images under `images/`, in sorted-name order, and the
plot is colored by that order. The JSON says which in `time_axis`.

## What it writes

`camera-checks.json`:

- `registered` and `expected`: frames COLMAP placed, and frames in the capture.
- `missing`, `unexpected`: expected frames that did not register, and registered
  names the capture does not list.
- `sparse_models`: the model folders under `sparse/`. More than one means COLMAP
  split the solve into disconnected pieces.
- `reviewed_model`, `points`, `mean_reprojection_error_px`, `mean_track_length`.
- `cameras`: model, size and parameters of each camera.
- `time_axis`, `registered_span`, and
  `largest_step_between_adjacent_registered_cameras` (scene units, in name order).
- `limits`: the reminder below.

`camera-review.png`: three views (X-Y, X-Z, Y-Z) of the sparse points with the
camera path over them, colored by time or order, plus the step between adjacent
cameras along that axis.

The same numbers are printed to the terminal, without the long `missing`,
`unexpected` and `cameras` fields apart from a one-line `missing:` list.

## How to read it

You want one continuous camera path around the subject, no sudden jumps or
disconnected clusters, and sparse points that form the subject and its
surroundings. A spike in the step plot is a jump. **The geometry is scale-free,
and a high registration count does not prove the poses are right or that the
whole subject is covered: judge the plot.** If it looks wrong, change the
interval, the range or `data_type` and make a new capture or a new run. Do not
train on a bad solve.

## Exit

It exits with a message, and writes nothing, when there is no model at
`sparse/{model}/images.bin` or the model has no images or no points.

The workflow context is `pipeline/VIDEO-WORKFLOW.md`, step 6.

Tests: `pipeline/tests/test_camera_review.py` covers the helpers that need no
COLMAP data (frame times, missing and unexpected frames, model folders, the
arguments). The plot itself is checked only by a real run.
