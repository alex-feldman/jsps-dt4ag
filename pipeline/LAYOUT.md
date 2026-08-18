# Data layout

The directory structure the pipeline expects, and the reasoning behind it.
Specified 2026-08-18. Everything produced before that date predates this spec and
is test output.

## The one concept: a capture

A **capture** is the atomic unit that reconstructs together: one object, at one
point in time, photographed from however many cameras. One capture in, one
reconstruction out. There is no smaller unit the pipeline can act on and no
larger one it understands.

Everything *above* a capture is your organisational choice and the pipeline
treats it as an opaque relative path. This is deliberate. Real collections
disagree about what the upper levels mean:

- `<date+location+species>/<object-id>` — several objects imaged on one date
- `<experiment>/<date>` — one object imaged on several dates

Both are legitimate, and they are inverted with respect to each other. Any spec
that assigns fixed meanings to level 1 and level 2 has to lie about one of them,
so this one assigns meaning only to the capture and passes the rest through.

**Semantics belong in metadata, not in directory names.** A name has to serve
browsing, sorting and uniqueness simultaneously, which is why names always end up
encoding three facts and satisfying none of them. `capture.ini` is where they go
instead. See "Capture metadata" below.

## Structure

```
data_root/
  datasets/                      INPUT. Pipeline reads, never writes.
    <collection path…>/
      <capture>/
        images/<camera>/*.jpg    the photographs
        masks/<camera>/*.png     optional, one per photograph, same relative path
        capture.ini              optional provenance (see below)

  derived/                       pipeline-owned, disposable, rebuildable
    masked/<collection path…>/<capture>/<camera>/*.png

  colmap/<collection path…>/<capture>/<run-id>/
  outputs/<collection path…>/<capture>/<run-id>/
  exports/<capture>_<run-id>_*.ply        FLAT, see below
  run-log.csv
```

`<collection path…>` is the same relative path in the `datasets/`, `derived/`,
`colmap/` and `outputs/` trees, so a capture's inputs, intermediates and
training outputs are all findable from any one of them.

**`exports/` is deliberately flat** and is the one exception. The export
filename already carries the capture, the run id, the iteration count and the
resolution, so it cannot collide, and a single directory gives one place to find
every export rather than a tree to walk. That decision predates this spec
(2026-08-07) and the reasoning is at the `export_dir` assignment in
`run_pipeline.py`. An earlier draft of this file specified a nested exports tree
that the code has never written.

## Capture metadata

`capture.ini` is an optional provenance file describing the physical capture:
what was imaged, when, by whom, with what. It sits **inside the capture
directory, beside `images/` and `masks/`**.

That placement is the point. A capture is meant to be one directory that can be
copied, archived or handed to a collaborator as a unit, and metadata that lives
anywhere else stops travelling with the thing it describes. A central registry
listing every capture was the alternative and is worse: it splits a capture
across two locations and goes stale the first time a directory is moved.

**It holds no file paths.** The filesystem already expresses where things are and
the `images/` rule already answers where the photographs are. A path recorded here
would be a second source of truth that silently rots the first time anything
moves.

**It is INI, not TOML.** The pipeline environment is Python 3.10, where `tomllib`
is not in the standard library, and this repository carries no third-party
configuration dependency. That is the same reasoning that made the run config INI
(`dt4ag_config.py` module docstring). An earlier draft of this file said
`capture.toml`, which nothing here could have parsed.

```ini
[capture]
object_id     = LD1          ; defaults to the capture directory name if omitted
imaging_date  = 2024-10-01   ; ISO, when the photographs were taken
species       =              ; free text, optional
location      =              ; free text, optional
operator      =              ; who ran the capture
notes         =

[equipment]
camera        =
lens          =
rig           =

[masks]
source        =              ; what produced the masks, e.g. model name + version
created       =              ; ISO date
```

**Everything about it is optional and nothing breaks without it.** The pipeline
reads it when present, records `object_id` and `imaging_date` in the run log and
the run banner, and continues unchanged when it is absent. A file that is present
but unparseable IS an error, because a provenance record that silently fails to
load is worse than none.

**It never affects behaviour.** No paths, no stages, no parameters. It cannot
contradict the run config because it has no say in anything the run config
decides. A FAIR release renders it to JSON or DataCite at publish time; that is a
transformation, not a reason to store it as JSON now.

## The discovery rule

**A directory containing an `images/` subdirectory is a capture.**

That single rule is what lets the pipeline be pointed at any level of a
collection and process every capture beneath it, which is the batch capability
`SEQUENTIAL-RUNS.md` records as missing. It is self-describing: no manifest, no
config, no naming convention to remember.

It has to be strict to be useful. "A directory containing image files, possibly
nested" would make every ancestor of a capture also look like a capture, and
discovery would be ambiguous at every level.

**A capture must not itself be named `images`.** A directory named `images` is
read as the photographs directory INSIDE a capture, so a capture called that
would be mistaken for its own contents: the tree above it would be keyed as the
capture in `colmap/`, `outputs/` and `derived/`, and a `masks/` directory
belonging to the collection would attach to every capture beneath it.

**Containing the word is fine.** The rule is exact equality, not a substring
test, so `raw-images`, `images-2024` and `plant_images` are all legal capture
names. They simply are not canonical, and are treated as the pre-2026-08-18
layout.

One nearby rule IS a substring test and is easy to confuse with this one: the
`[colmap] data_type = auto` heuristic treats a directory whose name contains
`frames` as video. It inspects only the last component of `images_subpath`,
which under this layout is always `images`, so **`auto` can never infer video
for a canonical capture**. Video input is not supported in v0.2.0; it is a v1.0
item (`ROADMAP.md`). Set `data_type` explicitly to experiment.

**An explicitly configured `images_subpath` bypasses discovery entirely** and
works against any layout, with or without an `images/` level. So the convention is
required only for automatic multi-capture processing, never for running one
capture by hand. That is the compatibility path for existing data.

## Input versus derived

The split that matters operationally:

- **`datasets/` is input.** Produced by cameras and by whatever tool made the
  masks. The pipeline never writes here. Back it up.
- **`derived/`, `colmap/`, `outputs/`, `exports/` are the pipeline's.** Every one
  of them can be rebuilt from `datasets/` plus a config file. `derived/` in
  particular can be deleted at any time to reclaim space; composited captures run
  to gigabytes each.

Composited masked images were briefly written into `datasets/` (2026-08-17,
during the first implementation of mask compositing). That was wrong: it put
derived, rebuildable, multi-gigabyte data inside the one tree you most want to
preserve and back up. `derived/masked/` exists to correct it.

## Migrating existing data

No collection predating this spec has an `images/` level; camera directories sit
directly under the capture. **There are two source shapes and they are not the
same migration**, which is the thing to establish before writing any `mv`:

**Masks in a parallel tree.** One `mv` per camera directory, plus one for the
masks:

```bash
# inside a capture directory
mkdir images && mv <camera-dirs> images/
mv <mask-tree>/<capture> masks
```

**Masks beside the photographs**, the `.jpg` and its `.png` in one directory.
This is a per-file split by extension, and no directory move expresses it:

```bash
# inside a capture directory, per camera directory
mkdir -p images/<camera> masks/<camera>
mv <camera>/*.jpg images/<camera>/ && mv <camera>/*.png masks/<camera>/
rmdir <camera>
```

Both were migrated on 2026-08-18: five captures of the first shape and six of
the second, 1,160 photographs and 1,160 masks, counts verified per capture
before and after. The second shape is also the one that caused the 2026-08-17
downscale-pyramid failure, since ns-process-data interleaves a mixed-extension
directory into one ffmpeg sequence, so migrating it is worth more than tidiness.

**Count the files before and after, per capture, and refuse to continue on a
mismatch.** A migration that half-succeeds leaves a capture whose masks no
longer pair with its photographs, and the pipeline cannot tell that from a
capture that legitimately has fewer masks.

Until a collection is migrated, run it by naming `images_subpath` explicitly.
Discovery will not see it, which is the intended failure: silent partial discovery
would be worse than none. A non-canonical `images_subpath` also keeps masks
looked for beside the photographs, since the `masks/` sibling is only consulted
when the configured directory is actually named `images`.
