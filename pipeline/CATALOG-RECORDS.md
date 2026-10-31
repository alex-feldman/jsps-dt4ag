# Reconstruction invocation records

Enable catalog emission with all three arguments (or their matching
`PLANT_CATALOG_ROOT`, `PLANT_CATALOG_ROOT_ID`,
`PLANT_CATALOG_CAPTURE_ID` environment variables):

`--catalog-root /absolute/data/root --catalog-root-id data-root --catalog-capture-id capture-a`

The root must contain the capture and linked outputs. An explicit root is now
required instead of defaulting to the loaded configuration's data root: the
record must be created before config loading so invalid configs can be recorded.
The capture ID is supplied by the caller, never invented from a path. Without
these prerequisites, catalog setup refuses and cannot promise a durable record.
Dry runs write no catalog. Ordinary runs without catalog enablement are unchanged.

## Additive progress contract

`plant_catalog.CatalogJob` owns writes. Existing schema-1 job fields and IDs
are unchanged. New jobs add `progress`, with its own `schema_version`
(`plant_catalog.PROGRESS_SCHEMA_VERSION`, now 2; fields in
`plant_catalog.PROGRESS_FIELDS`):

- `milestones`: ordered verified observations, each with `name` from
  `plant_catalog.MILESTONES` and UTC `verified_at`; `furthest_milestone` names
  the highest-ranked observation on that ladder.
- `active_stage`: stage currently running, or null.
- `stages`: selected stage names mapped to `started_at`, `ended_at`,
  monotonic `elapsed_seconds`, and `status` (running, completed, failed,
  cancelled). A failed stage includes time spent before failure.
- `resolved`: the loaded dataclass configuration including defaults, then
  runtime run ID, paths, stage selection, evaluation mode and effective downscale
  as those become known. Values are not guessed before resolution.
- `dependencies`: Python and installed distribution versions plus bounded
  `plant_catalog.colmap_version` and `ffmpeg_version` probes; unavailable
  versions remain null.
- `failure`, `gpu` and `artifacts`: schema 2 additions, defined in the
  [design note](#progress-schema-2-design-note-o13t5-o13t6) below.

`run_pipeline.run_pipeline` owns milestone placement at existing verification
boundaries. Skipped stages do not acquire synthetic milestones. Exported means
the export command returned; verified is recorded only after all requested
exports pass existing verification and the invocation reaches completion.
Repeated checkpoints do not produce duplicate milestone entries.

Invalid-config jobs retain the caller's capture reference but do not create a
capture record or pretend selected images are known. Handled errors, keyboard
interruptions and SIGTERM (`run_pipeline.Terminated`) finalize jobs. Hard
termination cannot execute finalization: the
last atomic running record, active stage and prior milestones remain evidence of
an incomplete attempt, never proof it is still live or completed. Automated stale
job reconciliation is future work.

Legacy jobs without progress remain readable; `plant_catalog.read_progress`
reads any job's progress in the current shape. Mission Control's existing
schema-1 validator accepts additive fields and retains its existing job summary;
display of detailed progress is separate UI work. No new capture kind or nullable
capture identity is introduced. Indexes, batches and automatic catalog enablement
remain later release-plan work.

`run_collection.py` turns emission on for each capture it runs and is a reader
of these records: it skips a capture only when a job passes
`run_collection.why_incomplete`, which reads `execution_status`, `exit_code`,
the `verified` milestone, export artifact records with their
`progress.artifacts` sizes, `config.ini`, `selected_images` and `started_at`
([SEQUENTIAL-RUNS.md](SEQUENTIAL-RUNS.md#when-a-capture-counts-as-completed)).
Changing those fields changes what a collection rerun skips.

## Verification

Run `python3 -m unittest discover -s pipeline/tests -p test_plant_catalog.py`.
Fixtures use temporary storage and mocked subprocesses. They cover atomic-write
failure, a truncated write, unfinalized progress, failed-stage timings, invalid
configuration, prerequisite failure, stage failure, interruption, a real
SIGTERM delivered to the test process, dry-run suppression, a verified mocked
COLMAP completion with artifact sizes, opt-in hashing, tool-version and GPU
probes that fail or return unexpected output, and old-record reading. They do
not establish a real GPU, COLMAP or nvidia-smi run.

### Later-stage and external consumer checks

The catalog suite also exercises a real partial-pyramid verification failure,
successful processing followed by failed training, zero-vertex export refusal,
and valid export acceptance. External commands remain mocked; filesystem and
verification functions are real.

For optional integration with a Mission Control checkout, generate fixtures into
a new temporary directory:

    python3 pipeline/tests/catalog_consumer_fixtures.py /tmp/new-catalog-fixtures

Then, on a host that can read that directory, run:

    node pipeline/tests/catalog-consumer-compat.cjs /path/to/mission-control /path/to/new-catalog-fixtures

The first path names the app directory containing `lib/plant-catalog.ts`
and its existing TypeScript installation. The check transpiles the actual
consumer module, supplies isolated workspace/config roots, and calls its
validator and `loadCatalog`. It neither edits nor starts the app. No package
installation is part of this check. Fixtures include a producer-generated
completed job, the legacy shape without progress, and a real early config failure.
Expected outcome: all records load; the early job retains its explicit unresolved
capture diagnostic. Malformed required fields still fail validation. Detailed
progress UI rendering is outside this check.

## Progress schema 2 design note (O13.T5, O13.T6)

Scope: release-plan step 2 only. The board tasks name a `runs/` store; ROADMAP's
record architecture decision (2026-10-20) and the release plan's step 1 replace
it with the existing `catalog/jobs/` records, so schema 2 extends those and adds
no second store. Top-level job `schema_version` stays 1, so the catalog v1
validator and existing job IDs, evidence and log pointers are untouched; only
`progress.schema_version` moves from 1 to 2.

**Already implemented (progress schema 1):** incremental atomic writes
(`CatalogJob.write`), the job created before config loading, `milestones`,
`furthest_milestone`, `active_stage`, per-stage `stages` timings, `resolved`
(loaded config plus runtime values) and `dependencies`; config, prerequisite and
stage failures and `KeyboardInterrupt` finalize the job.

**New in schema 2, all additive inside `progress`:**

| Field | Meaning |
|---|---|
| ladder | `plant_catalog.MILESTONES` is the explicit ladder; `milestone` refuses names outside it. |
| `furthest_milestone` | Highest-ranked observed milestone on that ladder (schema 1 said latest; both agree because the runner places milestones in ladder order). |
| `failure` | Null while running and on success. On a nonzero finish: `cause` (`config`, `error`, `interrupted`, `terminated`), `stage` (the stage active when it failed, or null) and `after_milestone` (furthest milestone at that moment). The reason stays in the existing `error_summary`, the code in `exit_code`. |
| `gpu` | Null when no probe could run. Otherwise `devices` (each `name`, `compute_capability`, `memory_total_mib`, `driver_version`, null when unreadable) and `cuda_visible_devices`. Read from `nvidia-smi`; torch is never imported for it. |
| `artifacts` | One entry per linked artifact: `artifact_id`, `role`, `relative_path`, `size_bytes`, and `file_count` for directories. Sizes are null when unreadable. |
| `dependencies` | Same keys. The COLMAP probe also reads the bare-banner version, which COLMAP 3.12 prints instead of honoring `--version`. Unknown stays null; no probe raises. |

Artifact hashing stays off. `--catalog-hash-artifacts` fills the existing
artifact-record `sha256` for regular files only; directories are never hashed.

**Early failure.** Unchanged and already implemented: with catalog emission
enabled, the job exists before config loading, so a config failure leaves a
failed job with the caller's capture ID, null `selected_images`, no milestones
and `failure.cause = config`. No output directory or capture record is needed.
Failures before the job exists (argument parsing, an invalid catalog ID, a
missing catalog root) have no destination and produce no record.

**Interruption.** SIGINT finalizes as before: `execution_status = cancelled`,
exit 130, `failure.cause = interrupted`. New: while a job is active the runner
turns SIGTERM (for example a `wsl-guard` stop) into the same finalization with
exit 143 and `failure.cause = terminated`, then restores the previous handler
before finalizing, so a second SIGTERM still kills. SIGKILL and machine loss
cannot run code: the last atomic record stays `running` with its
`active_stage` and milestones, and is evidence of an incomplete attempt.

**Reading old records.** `plant_catalog.read_progress` returns every schema 2
progress field for any job record; a field the record's schema did not have is
null, and a job without `progress` reads as `schema_version` 0. Callers use the
version to tell "not recorded" from "probed and unknown".

**DECISION NEEDED (Alex).**

- Artifact sizes live in the job's `progress.artifacts`, not as a new
  artifact-record field, because the Mission Control artifact validator could
  not be run here. Promoting `size_bytes` into artifact records is a later,
  consumer-checked change.
- SIGTERM maps to the existing `cancelled` status rather than a new
  `execution_status` value, so no consumer enum changes.
- Emission is still opt-in. "Failed runs get records too" holds only for
  catalog-enabled invocations until automatic enablement (with a capture-ID
  rule) is decided.
