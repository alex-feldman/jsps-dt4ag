# frame-selection

Choose which frames of an existing capture go into a camera solve, then build a
new capture from that choice. It works on any capture with an `images/` folder,
not only one made by `video-frames.py extract`, and before or after a COLMAP
solve. It never launches a pipeline stage and never modifies or moves a source
image.

One file, `frame-selection.py`, standard library only, plus the page template
`frame-selection.template.html` beside it. `preview` needs `ffmpeg` on `PATH`,
or pass `--ffmpeg-prefix {directory}` (the FFmpeg from the COLMAP recipe in
QUICKSTART section 3 is enough).

## The steps, and the decision between them

1. **`preview`**: inventory the capture and write an offline review page.
2. **You choose** on the page. Every frame starts kept. Per clip you exclude
   ranges (as many as you like, also in the middle) and may thin the clip to
   every Nth frame. You download a selection JSON. The page never decides.
3. **`build`**: validate the selection against the source as it is now and
   write a NEW capture. Run it with `--dry-run` first.
4. Review the new capture, then solve and inspect the cameras as
   `pipeline/VIDEO-WORKFLOW.md` describes. A new selection is a new image set and
   needs its own solve.

### preview

```bash
python3 pipeline/scripts/frame-selection/frame-selection.py preview \
    --capture {data_root}/datasets/{capture} \
    --out {data_root}/evals/{subject}/frame-selection-preview-{yyyymmdd-hhmm} \
    [--clips {clips.csv}] [--annotations {camera-checks.json}] [--title "..."] \
    [--ffmpeg-prefix {directory}]
```

`--out` must not exist and must not be inside the capture. Everything that can
be refused (clips, annotations, ffmpeg) is checked before it is created.

**Inventory.** The top-level image files of `{capture}/images/` (`.png`,
`.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, any case), sorted by filename.
Subdirectories and other files are skipped and reported. A link that does not
resolve is refused.

**Clips.** Where each frame's clip and time come from, first match wins:

1. `--clips {csv}` with columns `filename,clip` and an optional `seconds`. It
   must list every inventory file exactly once; a missing, duplicated or unknown
   filename is refused.
2. `{capture}/provenance/frames.csv` (written by `extract` or `subset`): one
   clip, times from `source_seconds_approx` (or `output_grid_seconds`).
3. Otherwise one clip, named after the capture, with no times.

Within a clip, the base order is filename order. Clips appear in the order of
their first file.

| File | What it is |
|---|---|
| `index.html` | **open this.** Offline and self-contained (relative files, no network). Per clip: a grid of frames with filename, position and time; click to enlarge (arrow keys move, Esc closes); `exclude from` then `to here` marks a range (`exclude only this` for one frame); a list of excluded ranges, each removable; `every` and `offset` controls; the kept ranges. A sticky bar keeps `Download selection JSON`, `Download inventory CSV` (with `sampled`, `excluded` and `kept` columns), the kept and total counts and the autosave status visible. Choices autosave in this browser's storage; the download is the handoff. A downloaded selection can be loaded back |
| `inventory.csv` | `index, filename, clip, clip_index, seconds, width, height, size_bytes, mtime_ns`. `index` and `clip_index` count from 0. `width` and `height` are read from PNG headers and left empty for other formats; `seconds` is empty when unknown |
| `inventory.sha256` | SHA-256 of `inventory.csv`, in `sha256sum` format |
| `preview.json` | the record: capture, inventory hash, clip source (with the clips CSV's hash), annotations record, thumbnail sizes, FFmpeg version, tool version |
| `thumbs/`, `large/` | JPEG views, 320 and 1280 pixels on the longest edge, named by inventory index. For choosing only, never pipeline input |
| `ffmpeg-thumbs.log` | every FFmpeg command run |

**Annotations (optional).** `--annotations` takes the `camera-checks.json` that
`camera-review.py` writes for a prior solve. The page shows its tags beside the
frames, labeled "prior solve" and with the file's path and hash: `missing`
becomes "not registered", and when `camera_path_warnings` is not empty,
`largest_step_from` and `largest_step_to` mark the largest camera step. The
extended fields `low_point_frames`, `flagged_steps`, `large_turns` and
`out_and_back` are read when present (the current `camera-review.py` does not
write them). Unknown fields and absent fields are reported, and frame names that
are not in this capture are reported, not fatal. The annotations never change
what is kept.

### The selection rules

For each clip, with base positions `p = 0, 1, 2, ...` in filename order:

- a frame is **sampled** when `p >= offset` and `(p - offset) % every == 0`
  (`every` a whole number, default 1; `offset` from 0 to `every - 1`, default 0
  = the first frame);
- a frame is **excluded** when it lies inside any excluded range (inclusive
  filename pairs; ranges may overlap);
- a frame is **kept** when it is sampled and not excluded.

Sampling is fixed on the original sequence before exclusions, so excluding the
first frames of a clip does not shift which later frames are sampled.

### The selection JSON

`"format": "dt4ag-frame-selection/1"`. The fields `build` reads:

```json
{
  "format": "dt4ag-frame-selection/1",
  "source": {"capture": "/abs/path/to/capture", "inventory_sha256": "..."},
  "clips": [
    {"name": "vf-1", "frames": ["frame_000001.png", "..."], "seconds": [0.0, "..."],
     "every": 1, "offset": 0,
     "exclude": [["frame_000001.png", "frame_000005.png"], ["frame_000120.png", "frame_000140.png"]]}
  ],
  "kept": ["frame_000006.png", "..."]
}
```

The page also writes `created`, `title`, `preview`, `annotations` (path and
hash), `rules`, per-clip `kept_count` and `kept_ranges`, `kept_count`,
`total_count` and `limits`, for the record; `build` does not rely on them.

`inventory_sha256` is the SHA-256 of one line per image in `images/`, in name
order: filename, size in bytes and modification time in nanoseconds,
tab-separated. It is computed from the files, not from `inventory.csv`.

### build

```bash
python3 pipeline/scripts/frame-selection/frame-selection.py build \
    --selection {downloaded selection JSON} \
    --capture {data_root}/datasets/{new-capture-id} \
    [--source {capture}] [--materialize] --dry-run
```

Drop `--dry-run` to build. The dry run validates everything and prints, per
clip, the base, sampled and kept counts and the kept ranges, then the mode
(links, or copies with the estimated bytes) and writes nothing.
`--source` overrides the capture path recorded in the selection, for a source
that moved; its images must still match the inventory hash.

It refuses, before creating anything:

- a format other than `dt4ag-frame-selection/1`;
- a source that does not exist, or whose current inventory hash differs from
  the selection's (the source changed: make a new preview);
- clips that do not cover the source images exactly once, or whose frames are
  not in filename order;
- a filename with a path separator, `..` or a control character;
- an `every` below 1 or not a whole number, an `offset` outside `0..every-1`;
- an excluded range that is not a pair, names a frame outside its clip, or
  starts after it ends;
- a `kept` list with duplicates, or that differs from what the rules keep
  (including order);
- fewer than 3 kept frames;
- an existing destination (nothing in it is touched);
- with `--materialize`, less free space than the bytes to copy.

It writes:

```text
{capture}/
  images/{filename}                 relative symlinks to the source images (or copies with --materialize)
  provenance/selection.json         the input selection, byte for byte
  provenance/frame-selection.json   tool and version, source path, inventory hash, selection hash,
                                    mode, timestamp, per-clip counts and kept ranges, status
  provenance/frames.csv             filename, clip, clip_base_index, source_filename[, source_seconds_approx]
```

Filenames are kept as they are in the source. `frames.csv` lists the kept
frames in filename order and carries `source_seconds_approx` only when every
kept frame has a time; `camera-review.py` then colors by clip time, otherwise by
frame order. If links cannot be made it stops and removes what it created,
rather than silently copying; `--materialize` is the explicit way to copy. If
the source has `masks/` or `masks_{variant}/`, the build says so: **masks are
not copied or linked**, so a new capture from this tool is for unmasked runs
until masks are made for it.

Exit status is 0 on success and 2 on any refusal, with the reason on standard
error.

## What this tool does not establish

- **It does not measure overlap** or judge whether a selection will solve. The
  camera review after a new solve is what tells you.
- **A changed selection needs a new solve.** COLMAP results, scores and
  annotations from one image set do not carry over to another.
- **Annotations describe only that prior solve and the image set it was run
  on.** They are context for choosing, not a verdict on a frame.
- **The inventory hash does not prove the bytes are unchanged.** It covers
  names, sizes and modification times; a file rewritten with the same size and
  a restored modification time would pass.
- **Masks are not copied** in this version.
- **Links need the source and the new capture to travel together.** The links
  are relative, so moving the whole `datasets/` tree keeps them valid; moving or
  deleting the source alone breaks them. Use `--materialize` for a capture that
  must travel on its own.
- The thumbnails are FFmpeg's reading of each file, without EXIF orientation
  handling for JPEGs; they are for choosing.
- Facility imagery from the page, thumbnails and captures stays outside git.

## Tests

`pipeline/tests/test_frame_selection.py`, in the stdlib suite. FFmpeg is
replaced by a fake. A test that runs the page's `resolveClip()` under Node and
compares it with the Python rules is skipped when `node` is not on `PATH`. How
the page behaves in a browser is checked only by opening it.
