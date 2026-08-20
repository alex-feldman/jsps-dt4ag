# Run records

Tools for the per-run config archive under `<data_root>/configs/runs/`.

Every run since 2026-08-21 freezes its own config there, written by the pipeline
(`archive_run_config` in `dt4ag_config.py`). Runs before that froze nothing,
because nothing did. These tools put the real config of an already-executed run
into the same place, where it can be proved to be the real one.

## The rule that governs all of this

**Archive a config only when it can be shown to be the config that run used.**
The archive sits under real run ids and its files are runnable, so a plausible
but wrong entry is worse than an absent one: it would silently reproduce
something other than the run it names. Nothing here infers a config, and a run
whose config cannot be established is reported, not filled in.

| `record_type` | Written by | Claim |
|---|---|---|
| (absent) | the pipeline, at run time | the config, frozen as the run started |
| `recovered-verified` | `recover-run-configs.py` | the file on disk, proved unedited against the run log |
| `recovered-exact` | a person, by hand | the original, reconstructed from a known edit, with the derivation recorded |

## recover-run-configs.py

```bash
python pipeline/scripts/run-records/recover-run-configs.py --data-root <path> [--dry-run]
```

Archives the REAL config of every already-executed run whose file still matches.

A config is only lost if it was EDITED since the run, and many were not: the six
arabidopsis configs have not been touched since the runs that used them, so the
real file is still on disk and is simply copied, verbatim.

**It only claims a config it can prove.** For each run it compares the file
against what `run-log.csv` recorded at run time: images subpath, iteration
count, downscale factor, masking route and training method. Every checkable
field must agree; one mismatch means the file has been edited and the run is
reported rather than archived. The record names the fields it verified, so the
strength of the claim is visible rather than implied.

The check is necessarily partial, since it can only compare what the run log
carries: a change confined to a COLMAP flag would pass unnoticed. It is worth
having anyway, because every config edit seen on this project has moved one of
those five fields, which are the ones anybody has reason to change.

Runs whose config HAS changed are listed with the differing field, never
guessed at. On the 2026-08-21 run these were 19 runs, and all 19 differ for one
reason: the canonical-layout migration on 2026-08-18 rewrote `images_subpath`
in every config. Their originals also predate the masking rework, which removed
a config key outright, so reverting the path alone would not reproduce them.

Idempotent, never overwrites an existing `<run-id>.ini`, and read-only against
everything except `configs/runs/`.

## Recovering a config by hand

When the tool refuses, and you know exactly what changed, write the file
yourself and say how you derived it. The five 2026-08-18 tomato runs were
recovered this way: the only edit since was `max_num_iterations` 10000 to
30000, applied by a targeted `sed`, so reverting that one line reproduces the
original exactly. Each of those records carries a `derivation` field stating
that, and the claim is corroborated by `run-log.csv` and by the run's own
`config.yml`.

Record `record_type = recovered-exact` for a hand recovery,
`recovered-verified` for one the tool made.
