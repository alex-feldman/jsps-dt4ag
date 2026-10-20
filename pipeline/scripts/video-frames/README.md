# video-frames

Preparation for a movie that will become a pipeline capture. The pipeline starts
from an image sequence and never decodes a movie (`pipeline/QUICKSTART.md`,
section 7); this tool is the step in front of it. It never launches a pipeline
stage.

One file, `video-frames.py`, standard library only. It needs `ffmpeg` and
`ffprobe`: on `PATH`, or pass `--ffmpeg-prefix {directory}`. The FFmpeg that
comes with the COLMAP recipe in QUICKSTART section 3 is enough.

## The two steps, and the decision between them

1. **`review`**: look at the footage. Read-only on the movie.
2. **You choose** the time range and the sampling interval from what you see.
   The tool never does.
3. **`extract`**: write a frozen frame set into a new capture.
4. Review the extracted frames, then solve and inspect the cameras as the
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
| `contact-NN.jpg` | coarse overview, one frame every 5 s (`--coarse-seconds`), 20 per sheet |
| `candidate-{interval}.jpg` | the first 20 frames at each candidate interval (`--candidate-intervals`, default 2, 1 and 0.5 s), from `--candidate-start`, so you can see how much the view changes between neighbors |
| `review-notes.json` | a record for your decisions. Every human field starts empty |
| `ffmpeg-sheets.log` | the exact commands that made the sheets |

The sheets are for choosing, not for training. They are decoded on the CPU with
two threads and default pixel autorotation, and a long 4K clip takes a while.

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
