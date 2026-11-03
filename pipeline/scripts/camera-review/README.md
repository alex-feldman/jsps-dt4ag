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
plot is colored by that order. The JSON says which in `time_axis`. For image captures, relative paths are kept so camera folders remain distinct. The review matches capture files and COLMAP names by relative stem, allowing JPEG originals to correspond to PNG mask composites; duplicate stems are refused rather than guessed.

Ordering is lexical by relative image name within each folder, an assumption that
may not match capture chronology. Only pairs consecutive in the expected list
and registered in the same folder contribute to steps or connecting lines.
Missing frames and folder boundaries break the path. Unexpected registered
frames are shown as points and excluded from steps. Unexpected names between two expected registered names do not break their link.
For a single video folder with every frame registered, this preserves the
original sequence.

## What it writes

`camera-checks.json`:

- `registered` and `expected`: frames COLMAP placed, and frames in the capture.
- `missing`, `unexpected`: expected frames that did not register, and registered
  names the capture does not list.
- `sparse_models`: the model folders under `sparse/`. More than one means COLMAP
  split the solve into disconnected pieces.
- `reviewed_model`, `points`, `mean_reprojection_error_px`, `mean_track_length`.
- `cameras`: model, size and parameters of each camera.
- `per_image_track_support`: retained 3D observation counts and image-cell occupancy for every registered image, including unexpected images, keyed by full COLMAP image name. Each image uses its own camera dimensions. `image_track_support()` owns the count and boundary rules; `track_support_grid` records the grid shape and semantics in the JSON. The grid is 4 by 4, with counts stored as top-to-bottom rows and left-to-right columns. Only nonnegative point IDs present in the reviewed sparse model count. Nonfinite or outside-image coordinates contribute to the retained count but occupy no cell. Zero retained observations give zero occupancy; invalid or missing dimensions give zero cell counts and a null coverage fraction. These are observation entries, not unique point IDs.
- `time_axis` and `registered_span`.
- `ordering_assumption`, `step_policy` and `adjacent_step_count` describe the ordering and included pairs. `folder_step_summaries` gives each expected folder its own count, median, maximum, maximum/median ratio and warning using the same eligible edges. Existing step fields retain their names but now summarize only eligible pairs; they are null when no pair is eligible.
- `median_step_between_adjacent_registered_cameras`,
  `largest_step_between_adjacent_registered_cameras`, its frame pair and
  time interval, and the largest/median ratio.
- `camera_path_warnings`: advisory flags produced by `summarize_camera_steps()`
  using `CAMERA_STEP_WARNING_RATIO` in `camera-review.py`. This includes pooled and folder-specific alerts. `flagged_camera_steps` names each flagged peak and its scopes.
- `limits`: the reminder below.

`camera-review.png`: three views (X-Y, X-Z, Y-Z) of the sparse points with the
camera path over them, colored by time or order, plus the eligible steps along that axis. Each flagged pooled or folder peak is marked with a red X. A dashed reference line appears for a pooled alert only, at its pooled threshold; folder thresholds differ and are described in the JSON.

Image-cell occupancy describes where retained sparse observations fall on an image plane. It does not measure subject coverage or establish pose accuracy, and supplies no quality threshold or accept/reject decision.

The terminal prints a compact summary through `terminal_summary()`, omitting
the long `missing`, `unexpected`, `cameras` and `per_image_track_support` fields.
The complete per-image support grids remain in `camera-checks.json`;
`track_support_grid` semantics remain in the terminal summary. A separate one-line
`missing:` list is printed as before.

## How to read it

You want one continuous camera path around the subject, no sudden jumps or
disconnected clusters, and sparse points that form the subject and its
surroundings. A spike in the step plot is a jump. **The geometry is scale-free,
and a high registration count does not prove the poses are right or that the
whole subject is covered: judge the plot. A flagged step is a review alert,
not an automatic rejection; a real rapid camera move can also trigger it.
Inspect the named frame transition and surrounding images before training. If
the pose jump is not physically credible, change the interval, the range or
`data_type` and make a new capture or a new run.

## Exit

It exits with a message, and writes nothing, when there is no model at
`sparse/{model}/images.bin` or the model has no images or no points.

The workflow context is `pipeline/VIDEO-WORKFLOW.md`, step 6.

Tests: `pipeline/tests/test_camera_review.py` covers the helpers that need no
COLMAP data (frame times, missing and unexpected frames, model folders, the
arguments). The plot itself is checked only by a real run.
