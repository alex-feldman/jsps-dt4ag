# video-frames

Preparation for a movie that will become a pipeline capture. The pipeline starts
from an image sequence and never decodes a movie (`pipeline/QUICKSTART.md`,
section 7); this tool is the step in front of it. It never launches a pipeline
stage.

One file, `video-frames.py`, standard library only. It needs `ffmpeg` and
`ffprobe`: on `PATH`, or pass `--ffmpeg-prefix {directory}`. The FFmpeg that
comes with the COLMAP recipe in QUICKSTART section 3 is enough.

## The steps, and the decision between them

1. **`review`**: look at the footage. Read-only on the movie. Open the
   `index.html` it writes.
2. **You choose** the time range and the sampling interval from what you see.
   The tool never does. The page builds the `extract` command for you.
3. **`extract`**: write a frozen frame set into a new capture.
4. Optionally, **`subset`**: make sparser captures (every 2nd, every 4th frame)
   from that extraction, with no photograph copied, to compare sampling
   densities of one video.
5. Review the extracted frames, then solve and inspect the cameras as the
   QUICKSTART describes, before any training.

### review

```bash
python pipeline/scripts/video-frames/video-frames.py review \
    --source /path/to/movie.mp4 \
    --review-root {data_root}/evals/{subject}-video
```

Creates a new `sampling-review-{timestamp}/` folder under the review root and
writes only there:

| File | What it is |
|---|---|
| `source.sha256` | the movie's hash; pass it to `extract` |
| `ffprobe.json`, `summary.json` | stream metadata, and the facts that matter for sampling (size, frame rate, duration, rotation, codec) plus an approximate frame count per interval |
| `index.html` | **open this first.** The sheets with a time printed on every tile, a "use this interval" button per candidate, and a form that builds the exact `extract` command (it refuses to enable Copy while the capture path still holds a placeholder). Offline and static: no server, no network |
| `contact-NN.jpg` | coarse overview of the whole clip, one frame every 5 s (`--coarse-seconds`), 20 per sheet |
| `candidate-{interval}-NN.jpg` | the **first 30 seconds** at each candidate interval (`--candidate-intervals`, default 2, 1 and 0.5 s), 20 frames per sheet, so 30 s at 0.5 s is three sheets. `--preview-seconds 60` looks at a full minute; `--candidate-start` moves the window along the clip |
| `review-notes.json` | a record for your decisions. Every human field starts empty |
| `ffmpeg-sheets.log` | the exact commands that made the sheets |

The sheets are for choosing, not for training. They are decoded on the CPU with
two threads and default pixel autorotation, and a long 4K clip takes a while
(a 54 s 4K HEVC clip with the default 30 s preview took 152 seconds on one
machine; `--preview-seconds 60` takes longer).

### extract

```bash
python pipeline/scripts/video-frames/video-frames.py extract \
    --source /path/to/movie.mp4 --sha256 {hash from source.sha256} \
    --capture {data_root}/datasets/{new-capture-id} \
    --interval 0.5 [--start 0 --end 90] \
    --identity "what the footage shows, in your words" --dry-run
```

Drop `--dry-run` to run it. The dry run checks the source and its hash and
prints the exact command and the approximate frame count, and writes nothing.

The capture directory must not exist. A capture is frozen once extracted, so a
different range or interval is a new capture. A name that records the movie and
the interval keeps them apart, for example
`{subject}-video-{yyyymmdd}-{clip}-0500ms-v1`.

It produces the layout the pipeline expects (`pipeline/LAYOUT.md`):

```text
{capture}/
  source/{movie}                 hash-checked copy, so decoding reads the fast filesystem
  images/frame_000001.png ...    lossless frames
  provenance/extraction.json     source, hash, range, interval, exact command, FFmpeg version, status
  provenance/ffprobe.json
  provenance/ffmpeg.log
  provenance/frames.csv          filename, output_grid_seconds, source_seconds_approx, width, height
```

What it checks, and refuses on:

- an existing capture directory (nothing is touched);
- a source whose hash is not the one you passed;
- an interval or range that makes no sense;
- after extraction, **no frames, or frames of more than one shape**. The record
  is kept with a status that says not to train, and the exit code is non-zero.
  A single COLMAP camera needs one image shape.

Exit status is 0 on success and 2 on any refusal or failure, with the reason on
standard error. A failed extraction is preserved, never cleaned up.

### subset

```bash
python pipeline/scripts/video-frames/video-frames.py subset \
    --from {data_root}/datasets/{extracted-capture} \
    --capture {data_root}/datasets/{new-capture-id} --every 2 [--offset 0] [--dry-run]
```

Keeps every Nth frame of an extracted capture as a symlink in a new capture's
`images/`, so `--every 2` doubles the interval and `--every 4` quadruples it.
**No photograph is copied**, and the tool refuses rather than copy if links are
impossible (exFAT, or Windows without link rights). The pipeline already stages
photographs by link, so such a capture runs like any other. Each subset writes
`provenance/subset.json` (parent, parent hash, rule, the exact frame list) and a
filtered `provenance/frames.csv`.

It refuses a parent that is not an `extract` capture, whose status says not to
train, or whose `images/` no longer matches its `frames.csv`; it also refuses an
existing capture, an `--every` below 2, an `--offset` outside `0..N-1`, and a
result under 3 frames.

Things to know:

- The links point at the parent's `images/`. Moving or deleting the parent breaks
  them. The pipeline still writes its own per-run copies and downscaled
  pyramids (nerfstudio's `ns-process-data`), which no capture layout avoids.
- Subsets of one video share frames. They show what sampling density costs and
  buys; they are not independent replicates.
- Every subset needs its own COLMAP solve (a different image set). COLMAP is not
  bit-reproducible, so differences between runs include solver noise.
- Image-space scores (PSNR, SSIM, LPIPS) from runs on different image sets are
  not like for like: the runs trained on, and would hold out, different
  photographs. Compare sampling densities by the camera solve, the model size and
  a visual review, and say which photographs each score was computed on.

## What this tool does not establish

- **It does not choose the interval and does not measure overlap.** The
  interval is your decision, recorded as such. A reviewed interval is not proof
  that neighboring frames overlap; the camera review before training is.
- **The recorded times are not source timestamps.** `output_grid_seconds` is the
  `fps` filter's output grid. `source_seconds_approx` adds your range start to
  it. If exact source-frame timestamps matter, build and check a separate PTS
  mapping.
- **Frames are extracted with FFmpeg's default pixel autorotation.** The source
  rotation tag is recorded in `extraction.json`. Check the extracted frames'
  orientation before continuing.
- **A movie with moving subjects, or a camera that does not move, is not made
  reconstructable by sampling.** The review is where to notice that.
- It extracts PNG only, and decodes on the CPU (`-hwaccel none`).

How to choose an interval from camera motion and blur, and the resource limits
that apply to long jobs on a Windows host, are the workspace's procedure, not
repeated here.

## Tests

`pipeline/tests/test_video_frames.py`, in the stdlib suite. ffmpeg and ffprobe
are replaced by fakes, so the suite needs neither. What the real tools do with a
real movie is checked only by a real run.
