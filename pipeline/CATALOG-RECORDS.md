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
are unchanged. New jobs add `progress`, with its own `schema_version: 1`:

- `milestones`: ordered verified observations, each with `name` and
  UTC `verified_at`; `furthest_milestone` names the latest observation.
- `active_stage`: stage currently running, or null.
- `stages`: selected stage names mapped to `started_at`, `ended_at`,
  monotonic `elapsed_seconds`, and `status` (running, completed, failed,
  cancelled). A failed stage includes time spent before failure.
- `resolved`: the loaded dataclass configuration including defaults, then
  runtime run ID, paths, stage selection, evaluation mode and effective downscale
  as those become known. Values are not guessed before resolution.
- `dependencies`: Python and installed distribution versions plus bounded
  COLMAP/ffmpeg version probes; unavailable versions remain null.

`run_pipeline.run_pipeline` owns milestone placement at existing verification
boundaries. Skipped stages do not acquire synthetic milestones. Exported means
the export command returned; verified is recorded only after all requested
exports pass existing verification and the invocation reaches completion.
Repeated checkpoints do not produce duplicate milestone entries.

Invalid-config jobs retain the caller's capture reference but do not create a
capture record or pretend selected images are known. Handled errors and keyboard
interruptions finalize jobs. Hard termination cannot execute finalization: the
last atomic running record, active stage and prior milestones remain evidence of
an incomplete attempt, never proof it is still live or completed. Automated stale
job reconciliation is future work.

Legacy jobs without progress remain readable. Mission Control's existing
schema-1 validator accepts additive fields and retains its existing job summary;
display of detailed progress is separate UI work. No new capture kind or nullable
capture identity is introduced. Indexes, batches and automatic catalog enablement
remain later release-plan work.

## Verification

Run `python3 -m unittest discover -s pipeline/tests -p test_plant_catalog.py`.
Fixtures use temporary storage and mocked subprocesses. They cover atomic-write
failure, unfinalized progress, failed-stage timings, invalid configuration,
prerequisite failure, stage failure, interruption, dry-run suppression and a
verified mocked COLMAP completion. They do not establish a real GPU run.
