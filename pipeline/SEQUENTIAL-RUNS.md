# Running several reconstructions in sequence

Status: **not a supported feature.** This document records what is missing and
why each gap matters, so the next person to batch a set of runs knows what the
runner does and does not do for them. Written 2026-08-17, after the first batch
of four dates was driven by a shell loop.

`run_pipeline.py` reconstructs **one dataset per invocation**. That is the whole
contract. There is no batch mode, no queue, no scheduler, and no notion that
several runs belong to one series. Sequencing is currently an external shell
loop:

```bash
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"
for d in <session-ids>; do
  uv run python pipeline/run_pipeline.py --config pipeline/configs/<prefix>-$d.ini
done
```

A loop is sequential by construction, which is the one property that genuinely
matters here, so this works. Everything below is what it does not give you.

## How it is actually done today

A driver script, one config per capture, a plain loop. This is the whole
mechanism; there is nothing else. Recorded here because it works and because
whoever automates this properly should start from what is already true.

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

What the driver captures per capture: exit status, wall-clock minutes, and the
exported gaussian count, which is the cheapest signal that masking worked. On
subject-only captures a count in the tens of thousands means the background was
not removed (see `MASKING.md`).

**The cost of this shape** is one config file per capture that differs from its
siblings in about four lines, and no resume: re-running the loop re-runs
everything, deriving new run ids, because nothing knows what already completed.
Both are addressed in the gaps below.

## 1. Nothing enforces that only one run happens at a time

**This is the important one.** The machine has a single GPU. Two concurrent
`ns-train` processes contend for it, and the symptom is a CUDA OOM in whichever
process next crosses splatfacto's `resolution_schedule` boundary and jumps to
full rasterisation resolution. That failure names memory, not contention, so it
reads exactly like a dataset or configuration problem. Someone will spend an
evening on the wrong hypothesis.

The loop above is safe on its own. What is unsafe is a **second** invocation
from another terminal while the loop is running, and nothing currently refuses
it. The runner already fails loudly about every other prerequisite (`colmap`,
`ffmpeg`, `ns-train`, GPU architecture); the one resource that physically cannot
be shared is the one it does not guard.

Fix: a lockfile acquired in `check_prerequisites` for any stage that touches the
GPU (`train`, `export`), holding the pid, and refused with a message naming the
run that holds it. Stale locks from a killed process must be detectable, so
store the pid and check whether it is alive rather than trusting the file's
existence.

## 2. There is no resume, and re-running silently duplicates work

Run ids auto-increment when `[run] date` and `run_count` are blank. So if a
batch fails on its third dataset and the loop is simply re-run, the first two
execute **again** under new ids. Nothing is overwritten and nothing warns: you
get a second full copy of the COLMAP workspace, the training output and the
export, several GB apiece, and two sets of results that differ only by id.

The existing escape is manual: pass `--run-id <existing>` with `--from-stage` to
resume one dataset. That does not compose with a loop, because the id is derived
per dataset and the operator would have to know each one.

Fix, in increasing order of effort: have the loop record each dataset's resolved
run id as it goes; or teach the runner a `--skip-if-complete` that treats a
dataset with a valid export as done.

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
current columns).

## 4. The training resolution is not recorded anywhere

Separate from the CSV problem above, and worth its own line because it bites
even for a single run. With `[train] downscale_factor = 0`, nerfstudio picks the
factor itself by probing the image pyramid on disk, and the value it picked is
written to no artefact: `config.yml` records the *request* (`downscale_factor:
null`), not the decision. Recovering it after the fact means re-deriving the
probe by hand.

So the resolution a reconstruction was trained at, which changes the result more
than most settings, is currently a property of what happened to be on disk that
day and is recorded nowhere. See "What to change next".

## 5. The `PATH` export is operator memory

`colmap` and `ffmpeg` live in a prefix that is not on the default `PATH`, by
design (`QUICKSTART.md`). Omitting the export is what caused the 2026-08-17
failure: `ns-process-data` ran with no `ffmpeg`, built no image pyramid, and
training silently fell back to native resolution and died mid-run. The
prerequisite check added the same day now catches it before COLMAP starts, but
the batch driver still has to remember the line.

Fix: none needed in the runner. A driver script should export it itself and log
which binaries resolved, so the log answers the question later.

## What to change next

In priority order:

1. Single-instance GPU lock (gap 1). It prevents a misdiagnosis, not just a
   failure.
2. Record the **effective** downscale factor: resolve it the way nerfstudio
   does, log it at the start of the train stage, put it in the run log, and add
   it to the export filename, which already carries platform, environment, step
   count and data type. That makes it a property of the artefact rather than of
   the operator's notes (gaps 3 and 4).
3. `--skip-if-complete`, so re-running a batch is cheap and idempotent (gap 2).

None of these should be implemented while a batch is running: every dataset
launches `run_pipeline.py` fresh, so editing it mid-sequence changes the code
under the remaining datasets.
