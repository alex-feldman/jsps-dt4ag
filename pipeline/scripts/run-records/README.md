# Run records

Tools for the per-run record under `<data_root>/configs/runs/`.

## The two kinds of file, and why they must not be confused

| File | Written by | Trust |
|---|---|---|
| `<run-id>.ini` | the pipeline, on every run since 2026-08-21 | the config, verbatim. Runnable |
| `<run-id>.reconstructed.ini` | `backfill-run-records.py`, after the fact | evidence about a run. **Not** runnable, and not the config |

A genuine record is the config file byte for byte, plus a `[run-record]`
section carrying what resolved at run time. A reconstructed record is an
inference from what a historical run left on disk, because the config it used
was gitignored, never versioned, and has since been edited or deleted.

Reconstructed records carry no `[paths]` or `[dataset]` section, so
`load_config` refuses them. That is deliberate and it is the safety property
that matters here: the pipeline-level settings of an old run (COLMAP flags,
masking route, file extensions, export labels) are often unknown, so a file
that *looked* runnable would invite someone to re-run it believing they had
reproduced the original.

## backfill-run-records.py

```bash
python pipeline/scripts/run-records/backfill-run-records.py --data-root <path> [--dry-run]
```

Walks `outputs/**/<run-id>/splatfacto/<timestamp>/config.yml`, joins each run
against `run-log.csv` where a row exists, and writes one reconstructed record
per run. Every field carries its source in a `[sources]` section, and each
record ends with an `[unrecoverable]` section naming what no surviving evidence
can establish.

Evidence, in descending order of authority:

1. **`config.yml`** — written BY the run, so it cannot drift or overstate.
   Authoritative for iteration count, downscale factor, seed, method and the
   data path. A run with several timestamped attempts has all of them recorded
   and the newest read.
2. **`run-log.csv`** — written at run START, so it states intent. The only
   source for the masking route, the COLMAP version and the config file's path.
   Not every run appears: the log postdates the oldest runs.
3. **The filesystem** — exports and COLMAP workspaces, which corroborate that a
   run produced something.

Idempotent, and read-only against everything except `configs/runs/`. **It never
overwrites a genuine `<run-id>.ini`**, it skips and says so, so re-running it
after new reconstructions is safe.

Zero dependencies, stdlib only, matching the rest of the config tooling. It
reads nerfstudio's `config.yml` line-wise rather than with a YAML loader,
because that file is a python-tagged dump and parsing it properly would mean
importing nerfstudio.

### What it will never tell you

The contents of the config a historical run used. That is gone. What the record
gives you instead is every fact that survived, and an explicit account of what
did not, which is the honest maximum. The 2026-08-21 backfill covered 62 runs,
of which only 32 appeared in `run-log.csv` at all.
