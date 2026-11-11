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
- `time_axis` and `registered_span`.
- `median_step_between_adjacent_registered_cameras`,
  `largest_step_between_adjacent_registered_cameras`, its frame pair and
  time interval, and the largest/median ratio.
- `camera_path_warnings`: advisory flags produced by `summarize_camera_steps()`
  using `CAMERA_STEP_WARNING_RATIO` in `camera-review.py`, plus one summary line from
  `flag_camera_steps()` when any step or turn is flagged.
- `flagged_steps` and `flagged_step_count`: **every** adjacent-camera step at least
  `CAMERA_STEP_WARNING_RATIO` times the median, not only the largest. Each is divided by the time
  or frame count between its two cameras first (so a stretch of unregistered frames is not read as
  a jump), and carries the turn between the two cameras. `median_step_rate` is the baseline.
- `large_turns`, `large_turn_count`, `median_turn_degrees`: turns of at least
  `CAMERA_TURN_WARNING_DEGREES` (90) between adjacent registered cameras, and the median turn.
- `out_and_back`: cameras where the path jumps away and straight back, meaning one frame was placed
  in the wrong spot.
- `median_points_per_image` and `low_point_frames`: frames that see under a tenth of the median
  number of 3D points. A frame the model barely sees is usually a frame it placed by luck.
- `distant_pairs`: from `database.db` when it is in the workspace. How many verified image pairs join
  frames more than 3 apart in order. Context only; see "Signs of a bad solve" for why it is not a test.
- `limits`: the reminder below.

`camera-review.png`: three views (X-Y, X-Z, Y-Z) of the sparse points with the
camera path over them, colored by time or order, plus the step between adjacent
cameras along that axis. Every flagged step is marked with a red X, with a dashed
reference line at the threshold defined by `CAMERA_STEP_WARNING_RATIO` in
`camera-review.py`. Frames are not labeled on the plot; the frame names are in
`flagged_steps` and `out_and_back` in the JSON.

The same numbers are printed to the terminal, without the long `missing`,
`unexpected` and `cameras` fields apart from a one-line `missing:` list.

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

## Signs of a bad solve

The camera path is whatever the person walked, so there is no expected shape and no loop to close. Judge
whether the solve is physically plausible and consistent with itself. Most useful first. The baselines are two
walking solves of UA CEAC facilities, measured with this tool: a teaching-greenhouse solve that was reviewed
and accepted (160 of 161 registered, 1.27 px) and a vertical-farm solve that was wrong (276 of 292
registered, 1.18 px). Two solves are a small sample; treat the numbers as examples, not thresholds.

| Sign | Why it means a wrong pose | Accepted greenhouse | Bad vertical farm |
|---|---|---|---|
| Turns near 180 degrees between adjacent frames (`large_turns`) | A walker does not turn around in half a second. The camera was put facing backward, which happens in look-alike places. | 0 turns of 90 or more, median 3.7 degrees | 2 (175.6 and 164.1 degrees), median 6.7 degrees |
| Out-and-back frames (`out_and_back`) | The path jumps away and returns, so one frame is in the wrong place. | none | 3 (frames 75, 82, 132) |
| Many flagged steps, not one (`flagged_step_count`) | One jump can be a fast move or a clip cut. Ten cannot. The tool used to report only the largest. | 1 (19x, with a 44 degree turn) | 10 |
| Registered frames that lie between missing ones | A lone registered frame beside a missing run was found on very little evidence (frame 252, 50 points, beside 247 to 262 missing). | frame 59 beside missing 60 | frame 252 |
| Point cloud is a flat slab with repeated bands, not the room | Duplicated structure from matching look-alike places. Judged by eye on the plot or in COLMAP's viewer; it is not a test. | | |

A flagged step alone is not a rejection: the accepted greenhouse solve has one at 19x and was accepted after the
frames were reviewed. Turns near 180 degrees and out-and-back frames are what separated the two solves, and
the count of flagged steps.

What does **not** tell you the solve is right or wrong:

- **Mean reprojection error.** It measures how well the points fit whatever poses COLMAP chose, so a folded
  solve can score well. The bad solve above was 1.18 px, better than the good one.
- **Registered count.** 276 of 292 looks fine and was not.
- **Mean track length, the distant-pair share and the low-point frame count.** The accepted greenhouse solve
  had a mean track length of 2.5, 76 percent distant verified pairs and 4 frames under a tenth of the median
  3D points, against 3.1, 85 percent and 5 for the bad one. None separates them on these two solves, and a
  walk through repeated structure puts distant pairs in a good solve too. They are printed for context. Do
  not use any of them as a gate.

A scene full of repeated structure (racks, trays, lights, rows of identical plants) is a risk before you
start, and exhaustive matching makes it worse because it tries every pair. Prefer sequential matching with
loop detection for such a walk, and expect it to need review.

A flag is an alert, not a verdict. Clip boundaries and genuinely fast moves also flag, and the turn limit
assumes a half-second sampling interval; at slower sampling a real turn is larger. The step test divides by
the time between cameras, and uses the median of the whole path as its baseline, so a capture that is
fast for long stretches and slow for others can flag its fast stretches. Open the named frames and compare
them before deciding.

## Exit

It exits with a message, and writes nothing, when there is no model at
`sparse/{model}/images.bin` or the model has no images or no points.

The workflow context is `pipeline/VIDEO-WORKFLOW.md`, step 6.

Tests: `pipeline/tests/test_camera_review.py` covers the helpers that need no
COLMAP data (frame times, missing and unexpected frames, model folders, the
arguments). The plot itself is checked only by a real run.
