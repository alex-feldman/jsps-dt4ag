# From a video to a Gaussian splat

The whole path, start to finish, as plain commands you can paste into a terminal. Every step is a
Python script or the pipeline runner with arguments; nothing needs an agent. An agent runs the same
lines on your behalf.

```text
video file
  1  review      one script makes a page of labeled thumbnails      video-frames.py review
  2  choose      you pick the sampling interval (and range) there   the page prints the command
  3  extract     frames + provenance into a new capture            video-frames.py extract
     (subset)    optional: sparser captures to compare densities    video-frames.py subset
  4  config      copy and edit one INI file                         by hand
  5  solve       COLMAP cameras, then image preparation             run_pipeline.py --stage colmap,process
  6  look        camera plot: is the camera path believable?       camera-review.py
  7  train       Gaussian splat, then PLY export                    run_pipeline.py --stage train,export
  8  judge       open the PLY and decide whether it is useful       you
```

Install and machine setup (CUDA GPU, COLMAP, ffmpeg, `uv`) are in [`QUICKSTART.md`](QUICKSTART.md) and
are assumed done. Run everything from the repository root. Set these once per terminal:

```bash
DATA=/path/to/data                        # your data_root: datasets/, colmap/, outputs/, exports/ live under it
NAME=my-subject                           # a short name for this video
FFPREFIX=$HOME/opt/colmap-prefix/bin      # the folder holding ffmpeg and ffprobe (QUICKSTART section 3)
export PATH="$FFPREFIX:$HOME/.local/bin:$PATH"
```

## 1. Review the video

```bash
python3 pipeline/scripts/video-frames/video-frames.py review \
    --source /path/to/video.mp4 --review-root "$DATA/evals/$NAME" --ffmpeg-prefix "$FFPREFIX"
```

It decodes the clip once and prints the path of an `index.html`. Open that page in a browser. It shows
the whole clip as thumbnails, and the first 30 seconds (`--preview-seconds 60` for a minute) at several
spacings, every tile labeled with its time. It writes nothing outside its own new folder and never
touches the video. Details and every file it writes: [`scripts/video-frames/README.md`](scripts/video-frames/README.md).

## 2. Choose the interval

On the page, pick the spacing that is the smallest change between neighboring tiles that still shows the
camera moving. Too close wastes disk and time; too far may leave no overlap for COLMAP to match. This is
your decision: the tool does not measure overlap, and the camera review in step 6 is what tells you if it
worked. If the camera barely moves, or the subject does, sampling cannot fix that.

The page prints the exact `extract` command for the spacing you chose. Copy it.

## 3. Extract the frames

The copied command ends in `--dry-run`. Run it as is first; it checks the video, its hash and the range and
prints what it would do. Then run it again without `--dry-run`. It writes a **new** capture and refuses to
touch an existing one:

```text
$DATA/datasets/{capture}/
  source/        a hash-checked copy of the video
  images/        frame_000001.png ...      <- the pipeline reads this
  provenance/    extraction.json, ffprobe.json, ffmpeg.log, frames.csv
```

Look at a few frames: the right way up (portrait phone video can come out rotated), not blurred, one image
size throughout (the tool stops if there are several). A different interval or range is a **new capture**,
never an edit of this one.

**To compare sampling densities of one video** without copying any images, extract at the finest spacing
once and then make sparser captures from it (symlinks, with provenance):

```bash
python3 pipeline/scripts/video-frames/video-frames.py subset \
    --from "$DATA/datasets/{capture}" --capture "$DATA/datasets/{capture}-1000ms" --every 2
```

## 4. Write the config

```bash
mkdir -p "$DATA/configs" && cp pipeline/configs/example.ini "$DATA/configs/$NAME.ini"
```

Edit these keys (`configs/README.md` documents every key; `QUICKSTART.md` section 4 explains them):

| Key | Set it to |
|---|---|
| `[paths] data_root` | your `$DATA` |
| `[dataset] images_subpath` | `{capture}/images` (relative to `$DATA/datasets`) |
| `[dataset] image_extensions` | `.png` |
| `[dataset] use_masks` | `false` for a first, unmasked run |
| `[run] id_prefix` | a short prefix for this video |
| `[colmap] data_type` | `individual` (exhaustive matching, robust when frames are far apart) or `video` (sequential matching, for closely spaced frames). The pipeline cannot infer `video` from the folder name, so say it explicitly |
| `[colmap] extra_args` | on WSL2, `--gpu_index 0`; add `--num_threads 2` if Windows RAM is tight (QUICKSTART, "COLMAP's GPU matcher on WSL2") |
| `[nerfstudio] scene_type` | `images`, with `video_path` left empty. The pipeline does not decode video |
| `[train] max_num_iterations` | `30000` |
| `[train] quit_on_train_completion` | `true` (required for any non-interactive run) |
| `[train] eval_mode` | `all` to train on every frame (no held-out frames, so no generalization claim), or the default `fraction` to hold 10% out |

Then check it resolves, with nothing executed:

```bash
uv run python pipeline/run_pipeline.py --config "$DATA/configs/$NAME.ini" --dry-run
```

## 5. Solve the cameras

```bash
uv run python pipeline/run_pipeline.py --config "$DATA/configs/$NAME.ini" --stage colmap,process
```

Note the **run id** it prints near the top (for example `my-subject_261020-01-3120`); the next steps need
it. This runs COLMAP and prepares the images and stops before training, which is deliberate: training is
the expensive stage and should not start on a bad solve.

## 6. Look at the cameras before training

```bash
uv run python pipeline/scripts/camera-review/camera-review.py \
    --capture "$DATA/datasets/{capture}" --workspace "$DATA/colmap/{capture}/{run-id}" \
    --out "$DATA/evals/$NAME/camera-review"
```

Open `camera-review.png`. You want one continuous camera path around the subject, no sudden jumps or
disconnected clusters, and sparse points that form the subject and its surroundings. The numbers (frames
registered of frames given, which are missing, reprojection error) are in `camera-checks.json`. A high
registration count does not prove the poses are right; the plot is the check. If it looks wrong, change
the interval, the range or `data_type` and make a new capture or a new run. Do not train.

## 7. Train and export

```bash
uv run python pipeline/run_pipeline.py --config "$DATA/configs/$NAME.ini" \
    --stage train,export --run-id {run-id}
```

The PLY files land in `$DATA/exports/`. One `run-log.csv` row and a frozen copy of the config under
`$DATA/configs/runs/` record what ran.

## 8. Judge the result

Open the PLY in a splat viewer such as SuperSplat. A valid file means the export worked; it does not mean
the model is useful. Check that the whole subject is there, that the arrangement is right and that
nothing important is missing or smeared. The gaussian count is model size, not quality. With
`eval_mode = all` any score measures fit, not generalization; held-out scores, and how to compare runs, are
in `QUICKSTART.md` section 6.

## When something stops

- **`CUDA driver version is insufficient` from COLMAP, or `nvidia-smi` fails inside WSL:** WSL has lost the
  GPU. Check `nvidia-smi` *inside* WSL, not on Windows. A laptop in an "eco" or hybrid power mode switches
  the NVIDIA GPU off; set it to a mode that keeps it on. After a driver update or sleep, close other WSL work
  and run `wsl --shutdown` in PowerShell.
- **WSL dies or runs out of memory during COLMAP or masking:** free Windows RAM, lower `--num_threads`, and
  check free space on the Windows drive that holds the WSL disk (QUICKSTART, WSL2 section).
- **Unattended runs:** `uv` and the COLMAP prefix must be on `PATH` (the first block above), and one GPU job
  at a time.

## What this workflow does not do

It does not decode video inside the pipeline (`[nerfstudio] scene_type = video` is unverified), choose the
interval for you, measure overlap, make masks (a separate tool), or tell you whether the reconstruction is
metrically accurate.
