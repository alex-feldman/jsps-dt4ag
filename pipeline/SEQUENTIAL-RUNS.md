# Running several reconstructions in sequence

`run_pipeline.py` reconstructs **one capture per invocation**. That is its whole
contract. `run_collection.py` sits above it: it finds every capture beneath a
collection, derives each capture's config from one base config, and runs
`run_pipeline.py` once per capture, one at a time, skipping captures whose
catalog record shows they already completed. This file is the operator
documentation for that, and the record of what is still missing. First written
2026-08-17, after a batch of four dates was driven by a shell loop; the
collection runner was added 2026-10-31.

## Collection runs

```bash
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"
uv run python pipeline/run_collection.py --config pipeline/configs/base.ini \
    --collection {collection path} \
    --catalog-root "$HOME/data" --catalog-root-id wsl-data --dry-run
```

`--collection` is relative to the base config's datasets directory (or an
absolute path inside it). `--dry-run` lists every capture found, in the order
it would run, with `run`, `skip` or `invalid` and the reason, and writes and
runs nothing. Drop it to execute. `--allow-full-resolution` and
`--catalog-hash-artifacts` are passed to every capture's `run_pipeline.py`. The
catalog flags fall back to `PLANT_CATALOG_ROOT` and `PLANT_CATALOG_ROOT_ID`; the
data must lie beneath the catalog root ([CATALOG-RECORDS.md](CATALOG-RECORDS.md)).

What it does, by function in `pipeline/run_collection.py`:

- **Discovery** (`discover_captures`): a directory containing `images/` is a
  capture ([LAYOUT.md](LAYOUT.md#the-discovery-rule)), ordered by path.
  Hidden directories are not searched. Symbolic links are never followed and
  each one passed is listed as ignored. A capture inside another capture,
  including a directory named `images` inside a capture's `images/`, refuses
  the whole collection, since which one is the capture would be a guess.
- **One base config** (`derive_config_text`): each capture's config is the base
  with `[dataset] images_subpath` set to `{capture path}/images` and
  `[paths] data_root` made absolute. Nothing else changes. It is written to
  `derived/collection-configs/{collection path}/{capture}/{hash}.ini`, named by
  its content hash, so an existing file is never rewritten
  (`write_derived_config`). Edit the base config, not these files.
- **Validation before anything runs** (`prepare`): every derived config is
  loaded with `dt4ag_config.load_config`. A capture is `invalid`, and is not
  run, when its config does not load, `[run] run_count` is pinned, `[train]
  quit_on_train_completion` is false, it has no photographs, its data is not
  beneath the catalog root, or its catalog capture ID already names a
  different capture. Invalid
  captures do not stop the valid ones.
- **Sequential execution** (`execute`, `run_child`): each capture runs
  `run_pipeline.py` in a fresh process with catalog emission on, under the
  capture ID from `capture_id_for`. A failing capture does not stop the rest.
  Ctrl-C or SIGTERM stops the batch: the running capture finalizes its own job
  record and the remaining captures are reported as not started.
- **Summary** (`record_outcome`, `print_summary`): every capture ends as
  `completed`, `skipped`, `failed`, `invalid`, `interrupted` or `not started`,
  with elapsed time, job ID and, for a failure, the exit code, failure cause,
  stage, furthest milestone and first line of the error, all read from the job
  record. Exit 0 counts as `completed` only when that record verifies. The
  catalog job's evidence log holds each capture's tool output.

Exit codes: `0` every capture completed or was skipped, `1` at least one
failed or was invalid (also `--dry-run` with an invalid capture), `2` the
collection itself could not start (an unreadable base config or collection, no
capture found, an ambiguous nested capture, two captures mapping to one capture
ID, a missing catalog flag),
`130` or `143` the batch was interrupted or terminated.

### When a capture counts as completed

A capture is skipped only when a catalog job record for its capture ID passes
every check in `why_incomplete`: status `completed` with exit 0, a progress
`verified` milestone, every export artifact (`EXPORT_ROLES`) still present,
at least `run_pipeline.MIN_EXPORT_BYTES`, the size its record states and a
nonzero vertex count, the same derived config sections (`config.ini`) and the
same photograph list (`selected_images`), and no input image modified after the
job started. Any newer or older job passing those checks is enough; otherwise
the capture runs again and the plan says why, citing the newest job. Nothing is
deleted or overwritten: a rerun gets a new run ID beside the earlier outputs,
and the earlier job records stay.

### Decisions (Alex, 2026-11-01)

Each choice below was the conservative default, implemented and tested, and
was accepted as implemented on 2026-11-01. Each is cheap to reverse.

- **Changed settings rerun.** Any difference in the derived config sections
  since the completed job (including `[host]`, labels and log settings, not
  only reconstruction parameters) makes the capture run again. There is no
  flag to skip anyway. Comments and whitespace do not count.
- **Changed inputs rerun.** An added, removed or renamed photograph, or any
  image under `images/` (and the selected mask set, with `use_masks`) modified
  after the job started, makes the capture run again. Content changes that
  keep the old modification time (`cp -p`, `rsync -a` of new content) are not
  detected; photographs are not hashed.
- **Old records do not prove completion.** A job without progress milestones
  (progress schema 0) or without recorded `config.ini` is treated as not
  completed. Progress schema 1 records are accepted; their export size is not
  compared because schema 1 did not record it.
- **Capture IDs come from the path.** `capture_id_for` lowercases the capture's
  path under `datasets/` and joins it with hyphens; the collection runner is
  the caller that supplies the ID ([CATALOG-RECORDS.md](CATALOG-RECORDS.md)).
  An explicit ID per capture, for example a key in `capture.ini`, was
  considered and not chosen. Consequences: moving or renaming a capture makes it a
  new capture that runs again, and a single-capture run made by hand under
  another ID is not recognized. Two captures mapping to one ID refuse the
  collection; an ID whose existing capture record points elsewhere makes that
  capture invalid.
- **A pinned `[run] run_count` is refused.** With it, a rerun would reuse the
  earlier run's COLMAP workspace, output directory and export filename, and
  could overwrite them. Leave it empty in a base config; `[run] date` may
  stay pinned.
- **Invalid captures are skipped, not fatal.** A capture whose derived config
  fails validation is reported and the others run. The alternative is to
  refuse the whole collection until every capture validates.
- **A job still marked running is not completed.** It may be live in another
  invocation or left by a hard kill; either way the capture runs again. There
  is no lock (see "Concurrency policy").
- **Symbolic links are never followed.** A capture whose `images` is a symlink
  is listed as ignored rather than run.
- **Interrupt stops the batch.** Ctrl-C, SIGTERM, or a child exiting 130 or 143
  stops the remaining captures. A child that does not exit within
  `CHILD_GRACE_SECONDS` of Ctrl-C is sent SIGTERM.

### Not covered

- No real reconstruction has been run through `run_collection.py`; tests use
  fake children and `run_pipeline.main` with mocked tools.
- Children are not routed through `wsl-guard`, so on WSL2 the per-run cache
  release described below does not happen between captures.
- No batch provenance (section 3), no rebuildable index, and no optional stop
  after camera solves for review; those are later release-plan steps.

## The shell loop it replaces

Before `run_collection.py`, a batch was a driver script, one config per
capture, and a plain loop. It still works, and it is still the way to run each
capture under `wsl-guard`. Recorded here because its deliberate properties are
the ones the collection runner keeps.

```bash
#!/usr/bin/env bash
set -u
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"   # colmap + ffmpeg, not in the venv
cd "$HOME/PycharmProjects/jsps-dt4ag" || exit 1

declare -A STATUS ELAPSED GAUSS
for c in <capture-ids>; do
  start=$(date +%s)
  uv run python pipeline/run_pipeline.py --config "pipeline/configs/<prefix>-$c.ini" \
      > "logs/$c.log" 2>&1
  rc=$?
  ELAPSED[$c]=$(( ($(date +%s) - start) / 60 ))
  STATUS[$c]=$([ $rc -eq 0 ] && echo OK || echo "FAILED($rc)")
  GAUSS[$c]=$(grep -oE "vertices *: [0-9]+" "logs/$c.log" | tail -1 | grep -oE "[0-9]+$")
done
# then print a summary table from the three arrays
```

Four properties of this shape are deliberate:

- **Sequential by construction.** No `&`, no job control. One GPU; two
  concurrent `ns-train` processes contend for it and the symptom is a CUDA OOM
  in whichever next crosses splatfacto's resolution jump, which reads like a
  data problem rather than contention.
- **A failing capture does not stop the others.** Captures are independent, and
  aborting a batch on one bad input wastes the remaining runs. Status is
  collected and reported at the end instead.
- **Per-capture logs**, because interleaved output from a long batch is
  unreadable and the failure you care about is always in the middle.
- **The `PATH` export lives in the driver**, not in the operator's memory. It is
  the single step whose omission caused the 2026-08-17 failure.

**On WSL2, sequence through the guard.** Each run leaves the file cache it
filled inside the WSL VM, which Windows cannot use, so a batch on WSL2 starts
each run with less Windows RAM than the one before (9.6 GiB of leftover cache
was measured after a single job). Replace the `uv run python
pipeline/run_pipeline.py ...` line with
`uv run python pipeline/scripts/wsl-guard/wsl-guard.py --log "logs/$c.log" -- --config ...`
and set `[host] memory_high` in the configs. The guard releases leftover cache
before each start (with passwordless sudo) and waits for Windows to see it,
refuses with exit 76 when there is still no headroom, and stops a run that
would empty Windows' RAM with exit 75. Treat both as "not done, rerun later",
not as a failed capture. It writes the log itself, so drop the `> logs/$c.log`
redirect. See `scripts/wsl-guard/README.md`.

What the driver captures per capture: exit status, wall-clock minutes, and the
exported gaussian count, which is the cheapest signal that masking worked. On
subject-only captures a count in the tens of thousands means the background was
not removed (see `MASKING.md`).

**The cost of this shape** is one config file per capture that differs from its
siblings in about four lines, and no resume: re-running the loop re-runs
everything, deriving new run ids, because nothing knows what already completed.
`run_collection.py` removes both.

## 1. Concurrency policy (decided 2026-10-20)

A collection runs one capture at a time within its own batch. Independent
invocations remain permitted; there is no mandatory global GPU lock. This allows
deliberate concurrent work, including jobs on different GPUs. Each invocation
uses its configured device settings; this policy adds no device scheduler.

Two jobs sharing a GPU can exceed available VRAM, slow each other down, or fail
with a CUDA out-of-memory error. Free memory at startup is not a reservation:
training demand can grow later. Coordinate GPU-heavy work manually and inspect
failure logs before treating an OOM as a capture or configuration defect.

Optional coordination per GPU may be considered later. It is not a v0.3.0 gate.
The batch itself must still remain sequential and preserve failure evidence.

## 2. Resume: closed for collection runs

`run_collection.py` skips a capture only when its catalog record proves
completion ("When a capture counts as completed" above). The rest of this
section is why that was needed, and it still describes the shell loop and any
hand-run `run_pipeline.py` invocation.

Run ids auto-increment when `[run] date` and `run_count` are blank. So if a
batch fails on its third dataset and the loop is simply re-run, the first two
execute **again** under new ids. Nothing is overwritten and nothing warns: you
get a second full copy of the COLMAP workspace, the training output and the
export, several GB apiece, and two sets of results that differ only by id.

The existing escape is manual: pass `--run-id <existing>` with `--from-stage` to
resume one dataset. That does not compose with a loop, because the id is derived
per dataset and the operator would have to know each one.

The collection runner does not resume a failed capture from its last good
stage: a capture that did not complete runs again from the first stage under a
new run ID, beside the failed attempt's outputs.

## 3. A batch has no provenance as a batch

`run-log.csv` gains one row per run with no field tying rows together. That the
rows form one time series exists only in the operator's memory and in the
similarity of the config filenames.

Worse, the log's columns are fixed at file creation: a log created before a
column existed keeps its original header forever, and later rows are written
under it with the newer fields dropped (deliberately, since a row wider than its
header is a corrupt CSV). The first batch therefore has **no recorded
downscale factor**, which is precisely the parameter a reader would want when
comparing datasets in the series.

Fix: a `series` or `batch_id` column, and a documented way to start a fresh log
when the columns change (rename the old file; the runner creates a new one with
current columns). `run_collection.py` does not add one: its captures' jobs are
linked by capture ID only, and catalog batch provenance is release-plan step 6.

## 4. The training resolution is recorded in the artefact, not in the run log

**Mostly closed since this was written.** The effective factor is now resolved
before training, logged to the console, and carried in the export filename as
`dsN`, so the resolution is a property of the artefact and two resolutions of
one capture can no longer overwrite each other.

What remains: the run log's `downscale_factor` column holds the CONFIGURED
value, because the row is appended before any stage runs and therefore states
intent rather than outcome. With `[train] downscale_factor = 0` it records the
literal `auto`, and nerfstudio's own choice, made by probing the pyramid on
disk, still reaches no CSV column. `config.yml` records the *request*
(`downscale_factor: null`), not the decision.

That gap belongs to v0.3.0's per-run records rather than to this file's batch
concerns: the fix is a record written incrementally as a run progresses, which
is exactly what a row written up front cannot be. With catalog emission enabled,
the job record's `progress.resolved.effective_downscale` now holds the decision
([CATALOG-RECORDS.md](CATALOG-RECORDS.md)).

## 5. The `PATH` export is operator memory

`colmap` and `ffmpeg` live in a prefix that is not on the default `PATH`, by
design (`QUICKSTART.md`). Omitting the export is what caused the 2026-08-17
failure: `ns-process-data` ran with no `ffmpeg`, built no image pyramid, and
training silently fell back to native resolution and died mid-run. The
prerequisite check added the same day now catches it before COLMAP starts, but
the batch driver still has to remember the line. `run_collection.py` passes its
own environment to every capture, so export it before starting a collection.

Fix: none needed in the runner. A driver script should export it itself and log
which binaries resolved, so the log answers the question later.

## What to change next

Follow [the ordered release plan](../docs/v0.3.0-release-plan.md). It sequences
catalog records, documented concurrency behavior, collection execution, safe
resume and record-derived summaries. ROADMAP owns the release acceptance criteria.

Do not change runner code beneath an active batch: every capture starts a fresh
process and could otherwise run a different implementation.
