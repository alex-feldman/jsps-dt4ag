# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/): one section per
release, newest first, with changes grouped under **Added**, **Changed**,
**Deprecated**, **Removed**, **Fixed** and **Security** (only the groups that
apply appear). What each version *promises* is in `ROADMAP.md`; this file records
what actually changed.

Versioning is [SemVer](https://semver.org/) while pre-1.0: MINOR for new
capability, PATCH for fixes. Milestones are marked by annotated git tags, not by
long-lived branches, because this repository has one committer working across two
machines and a tag records a milestone permanently at zero merge cost.

Dates are the tag date, not the commit date, where they differ.

## [0.3.0-alpha.14] - 2026-10-20

### Added

- `pipeline/scripts/video-frames/video-frames.py`, a stdlib-only preparation tool
  for a movie that will become a capture. `review` hashes the movie, records
  `ffprobe` output and makes contact and candidate-interval sheets, writing only
  to a new review folder. `extract` stages a hash-checked copy into a new capture
  and extracts PNG frames at a chosen interval and range, with `extraction.json`,
  `ffprobe.json`, `ffmpeg.log` and `frames.csv`. It refuses an existing capture or
  a wrong hash, and stops with a "do not train" status on no frames or mixed image
  shapes. It does not choose the interval or verify overlap, and never starts a
  pipeline stage.
- `review` decodes the clip once and writes whole-clip thumbnails plus a preview
  of the first 30 seconds (`--preview-seconds`, for example 60) at the finest
  interval, and an offline `index.html` that prints the time on every tile, lets
  you choose every 0.5, 1 or 2 s from that one preview, and shows the matching
  `extract` command (quoted in Python, ending in `--dry-run`).
- `subset`: makes a sparser capture (every Nth frame) from an extracted one as
  relative symlinks, with provenance and no copied photograph. It refuses, rather
  than copies, where links are impossible. Relative links survive moving or
  restoring the whole `datasets/` tree.
- Sampling uses `fps=...:round=up`. ffmpeg's default rounding takes each frame from
  the end of its time slot, so frames were late by up to half an interval and the
  whole-clip overview labeled "0 s" and "5 s" showed 2.4 s and 7.4 s. Verified on a
  clip whose brightness encodes its time: overview, preview and extracted frames now
  sit within 0.1 s of their labels. Captures extracted before this have the old
  lateness (visible in the command in their `extraction.json`). The frame-count
  estimate is now `floor((duration - one frame) / interval) + 1`, exact for 23 of 24
  whole-clip test cases and one or two high for a seeked range.
- `extract` refuses a range that does not fit the clip (`--start` at or past the
  end, `--end` past it) before creating or copying anything. Frame-count
  estimates use `round(duration / interval)`, which matched ffmpeg's `fps` filter
  in every measured case. `review` validates its inputs before creating its
  folder, and the page escapes every character the HTML parser treats specially.

- `VIDEO-WORKFLOW.md` troubleshooting: a run refused or stopped for low Windows RAM while little is in use
  inside WSL is usually WSL's file cache (measured: 9.6 GiB cached, 841 MiB used), and how to free it.
- `pipeline/VIDEO-WORKFLOW.md`: the whole path from a video file to a splat as plain,
  copy-paste commands (review page, choose the interval, extract, config, solve
  cameras, camera review, train and export), usable by a person or an agent alike.
- `pipeline/scripts/camera-review/camera-review.py`: the camera path over the sparse
  points plus registration counts, reprojection error and missing frames, written after
  the `colmap` and `process` stages and before training. It reproduces the recorded
  Welzo bed-video solve exactly (180 of 181 registered, 152,783 points, 1.151 px). It
  needs the pipeline environment (nerfstudio, matplotlib), so it has no unit test here
  and was checked by that run.

### Changed

- The video-derived frame workflow in `pipeline/QUICKSTART.md` now uses this tool
  for steps 1 to 3 instead of inline `ffprobe` and `ffmpeg` commands.

## [0.3.0-alpha.13] - 2026-10-19

### Added

- Draw each leaf-cluster ID beside its 3D point cloud in all organ-analysis
  viewers, using a leader from a rendered sample point and collision-aware
  placement so the ID remains readable as the view rotates.

## [0.3.0-alpha.12] - 2026-10-18

### Fixed

- Composite transparent PNG pixels over white in the offline click viewer so photographs remain legible without changing source pixels or click coordinates.

## [0.3.0-alpha.11] - 2026-10-18

### Added

- Opt-in Plant Reconstruction Catalog v1 emission for user-run reconstruction: atomic running/final job records, stable capture references, exact INI settings, selected inputs, frozen configs, command logs and verified export artifacts. Portable pointers use an explicit logical root ID. Execution completion does not imply human acceptance. Dry runs do not emit records.

## [0.3.0-alpha.10] - 2026-10-18

### Added

- Experimental `frame-calibration` record (schema version 1) and
  `calibrate_frame.py` with `orient`, `scale`, `validate` and `apply`. It maps
  dataparser-frame points into a plant frame by `p = s * R @ (p_source - o)`.
  `orient` takes up from the orientation stage's soil plane (camera mean
  image-up as a recorded fallback) and the origin from the soil base projected
  onto the fitted plane. `scale` ingests a `scale-estimate` file. The record is
  hash-bound to the run's `dataparser_transforms.json`, validated on load and
  save, and written as deterministic bytes.
- Target up axis setting `target_frame.up` (`+y` or `+z`; files without it read
  as `+z`, schema version unchanged) and `orient --up y|z`, default `y`. Y-up is
  the z-up rotation followed by `[[1,0,0],[0,0,1],[0,-1,0]]`; the validator
  checks that R maps the recorded up vector to the target axis.
- `transform_splat.py`: transform a 3DGS export PLY by a frame calibration into
  a new PLY plus a hash sidecar, without retraining. Centers, normals, wxyz
  quaternions, log scales (`--no-scale` keeps scene units) and band-wise rotated
  SH `f_rest` coefficients (fitted on the exact gsplat basis); `--inverse` maps
  back.
- **`pipeline/scripts/organ-analysis/click_points.py` and `ruler_scale.py`**: human-clicked image
  points (an `image-clicks` v1 file) to 3D points in the export's dataparser frame. A label clicked
  in two or more images is triangulated (DLT, then reprojection-error refinement with the camera's
  distortion; per-view residuals, triangulation angle, covariance). A label clicked once is placed
  where its ray meets a robust local plane through the COLMAP points observed in that image near the
  click (radius ladder 40/80/160 px, median depth for small support, refusal below a minimum), with
  an optional surface offset for an object lying above that surface. `ruler_scale.py` turns
  `tape_{N}cm` marks into a `scale-estimate` v1 (baseline-weighted median over all pairs, spread,
  propagated uncertainty, collinearity and leave-one-out checks). `click_points.py triangulate`
  writes all points, `{leaf}_base`/`{leaf}_tip` chords and a report-only nearest plant-asset leaf.
- `transform_splat.py --viewer supersplat` (opt-in): also writes a display-only
  `{out stem}-supersplat.ply`, the plain output rotated 180 degrees about z by
  the same Gaussian transform (centers, normals, quaternions, SH), because
  SuperSplat shows y-up files upside down, plus a `splat-viewer-copy` sidecar
  marked `display_only`. The plain output is byte-identical with or without the
  option and stays the analysis file. Refused with `--inverse` or a calibration
  whose target up is not `+y`. On the October 2 export the display copies match
  the ad hoc copies Alex checked vertex for vertex (all differences 0.0).
- `calibrate_frame.py orient --origin-from-asset PLANT_JSON` or
  `--origin-point X,Y,Z`, with `--origin-decided-by Alex --origin-decision TEXT`:
  an explicit, human-decided origin (for example the main-axis root, "bottom of
  the base stem") projected along the soil normal onto the soil plane, with the
  decision, unprojected point, projection distance and shift from the default
  recorded in `origin.diagnostics`. The asset route checks that the asset's
  hash-listed evidence names the same dataparser transforms. Without the option
  the origin is unchanged.
- `click_points.html`: single offline HTML tool for labelled pixel clicks on
  photographs (zoom, pan, quick `tape_{N}cm` and `{leaf}_base`/`{leaf}_tip`
  labels, COLMAP pixel convention, no EXIF rotation), exporting the
  `image-clicks` v1 JSON that `click_points.py` and `ruler_scale.py` read.

### Changed

- Scope: the plant frame's up and origin are geometric estimates from the
  orientation stage, and its scale rests on Alex's ruler clicks and his stated
  tape offset. This release validates neither against gravity or manual
  measurements, and adds no leaf length, area or curvature code.

## [0.3.0-alpha.9] - 2026-10-18

### Added

- Experimental additive partial semantic package and build/validate/query CLI.
  The complete schema-1 source is copied unchanged; a saved opaque identity
  registry binds every source file and preserves IDs through identical-source
  rebuilds and relocation. Changed sources require explicit correspondence.
- Candidate-to-leaf queries and geometric candidate groups preserve unresolved
  insertions, inferred geometry, pair annotations and combined apex regions.
  Source-built packages contain no accepted botanical nodes or internodes.
- Separate topology consistency helpers and synthetic fixtures exercise
  explicit accepted endpoints, axis-local ranks, zero-to-many leaves and
  below-node grouping without promoting source observations to anatomy.

### Changed

- Clarify MTG/Helios as the first candidate to evaluate with other architectures
  still open. This preview selects no backend, establishes no cross-date
  correspondence, and does not complete the 0.3 collection/run-record milestone.

## [0.3.0-alpha.8] - 2026-10-17

### Added

- Opt-in orientation-aware base and apex candidate diagnostics for organ
  analysis. A completed, unchanged orientation artifact binds the capture and
  camera frame; soil-primary up supplies the network root direction. Camera
  fallback requires explicit permission in the config.
- A conservative morphology candidate mode reports supported-leaf height,
  associated endpoint geodesic extent and terminal height. It proceeds only
  when their unique extrema agree; otherwise it keeps a candidate graph/report
  and records a failed run. An explicit apex override retains disagreement.
- Seven CLI integration tests cover tilted orientation, agreement, refusal,
  override, ties and orientation/source-integrity rejection.

### Changed

- Document the anatomy-informed, node-first semantic direction and consideration
  of MTG/OpenAlea and Helios as a preference, not a current feature gate.
- The legacy explicit-apex path reproduces the October 2 skeleton exactly.
  Both Welzo real-data automatic trials remain ambiguous: this release does
  not establish autonomous biological main-axis or apex recovery. The October
  15 retained network starts well above the soil plane, so its base is also
  unvalidated. No thresholds were tuned to recover a known answer.

## [0.3.0-alpha.7] - 2026-10-17

### Added

- `orient_plants.py` and `plant_orientation.py`: the scene's up direction, each plant's vertical axis and the stacking test for detached clusters, all recorded and algorithmic. The stage edits no derivative, and no visual judgment selects anything.
  - **Soil plane, the primary up.** The ground plane is orthogonal to up by definition. The method is ported from the October 2 soil pilot: soil votes at each COLMAP point's own feature-track observations, plant masks subtracted in memory, then deterministic RANSAC and SVD.
    - The camera mean image-up only bootstraps the neighborhood. The fit then re-takes the base along its own normal until the inlier set stops changing (at most 5 fits). Every iteration and the convergence are recorded.
    - Given the pilot's base and sign reference, the port reproduces the pilot's point IDs, votes, inlier set and normal exactly.
  - **Camera mean image-up.** The fallback, with its angular spread as the quality figure, and a cross-check. A disagreement above 10 degrees is reported, never switched.
  - **Plant axes.** Slice medians along up, every 0.02 scene units, with a 95th-percentile radius profile, height range and straightness.
  - **Detached-cluster assignment.** A cluster goes to plant P only when r/R is at most 1.0 for P, it lies within P's height range, and r/R is at least 2.0 for every other plant. Otherwise it is unassigned, with every number written. Axes and assignments are also written under the camera up, labeled as a cross-check.
  - **Welzo October 15.** With newly generated soil masks (prompt `soil`, threshold 0.50, 60 of 61 views), the soil normal lies 5.2 degrees from dataparser z. Cluster c6 is assigned to M1 under the soil up (r/R 0.544 against 2.606) and is unassigned under the camera up (0.887 against 1.408).
  - **Experimental defaults.** All numeric settings are experimental defaults (`plant_orientation.DEFAULTS`).

### Changed

- `geometry.py`: the pose mapping of `Geometry.cameras` is factored into `dataparser_frame` and `camera_to_world`, unchanged.

## [0.3.0-alpha.6] - 2026-10-16

### Added

- `plant_asset.py`: a versioned, capture-scoped plant-asset contract
  (`schema: plant-asset`, `schema_version: 1`) with a validator, also callable
  as `validate_plant_asset.py`. The whole `plant-asset/` folder is the object:
  relative paths, scoped organ IDs, extracted versus inferred status, original
  filtered-array row IDs, source hashes, scale-free units and nulls for unknown
  scale, radii and surfaces. Inferred geometry cannot be relabeled extracted or
  measured.
- `package_plant_asset.py`: emits that package from a checked branch-review
  folder, the endpoint review CSV and an explicit `plant-asset-decisions` file,
  reusing the reviewed pilot's cleanup, petiole and inference logic. Output is
  byte-reproducible and checked after relocation. Regenerating the Welzo M1
  pilot reproduced every leaf array, row ID, graph edge and petiole/attachment
  coordinate exactly; the intentional differences are listed in `DESIGN.md`.
- Single-leaf export: `export_leaf.py` writes one leaf's points, petiole and
  attachment to a self-contained folder, and `plant_asset.load_leaf_export`
  reloads it. The offline viewer selects a leaf by click or list and downloads
  the same record as JSON. Checked in headless Chrome on Welzo leaf 2 (B2 route)
  and inferred cotyledons 5 and 7.

## [0.3.0-alpha.5] - 2026-10-16

### Added

- `cluster_plants.py` and `plant_clusters.py`: memory-lean separation of plants in a multi-plant export.
  - **Clustering.** Single-linkage clustering of opacity-kept Gaussian centers at a radius given in multiples of the median nearest-neighbor spacing. A radius sweep, exact pairwise join radii (opacity-kept and over all rows) and direct gaps all come from one Delaunay-based minimum spanning tree. No step builds a list of all pairs within a large radius.
  - **Guard.** A sampled pair-count guard refuses over-budget radius queries.
  - **Derivatives.** Given an explicit selection with a recorded basis (cluster IDs, opacity-kept rows, optional nearby unclustered centers), the command writes per-plant derivative PLYs. They are byte-for-byte row copies with every property, and faint Gaussians are attached by nearest kept center.
  - **No automatic choice.** The command never picks a target cluster.
  - **Provenance.** It records hashes, settings and per-step peak memory, and it refuses existing outputs.
  - **Verification.** On the October 15 Welzo M1 export it reproduced the benchmarked clustering and gaps, and wrote M1 and neighbor derivatives byte-identical to the separations made by hand, at a 260 MiB peak. Synthetic tests compare it against brute-force radius graphs and minimax distances.

## [0.3.0-alpha.4] - 2026-10-16

### Added

- Per-endpoint review dropdowns and notes, prefilled only from existing human
  dispositions. Browser-local saving is scoped to the source graph and checked
  source node IDs, with a visible unavailable-state fallback. A persistent CSV
  download control exports all endpoints and raw IDs/coordinates, including
  partial reviews. Browser checks exercised reload persistence, CSV quoting,
  storage failure and unchanged geometry.

## [0.3.0-alpha.3] - 2026-10-16

### Added

- Unique IDs for every source-graph endpoint, including branch suffixes such as
  `b5-a`, and additive human review with retained leaf-connected branches kept
  out of the candidate set. IDs remain stable when a group is retained.
- `review_branches.py`: records explicit dispositions, can flag an incoming
  path for review even when it overlaps a saved leaf route, and optionally
  exports a separate graph pruned only at explicitly confirmed-false groups.
  The original graph and raw labels remain unchanged. The viewer distinguishes
  confirmed-cleanup preview, candidate preview and endpoint labels.

## [0.3.0-alpha.2] - 2026-10-16

### Added

- `identify_branches.py`: additive leaf-free terminal-subtree review from an
  existing organ asset, with source IDs, main/leaf-path protection, leaf-nearby
  and uncertain-base guards, JSON/CSV evidence, amber candidate overlays and a
  reversible display-only preview. No candidate is deleted by detection.
- A separately callable pure pruning function for explicit candidate selections,
  refusing protected paths and preserving its source graph. Fixtures cover
  shared leaf-carrying axes, terminal guards, multi-tip subtrees and source
  immutability. Real Welzo candidate detection was inspected in a browser.

## [0.3.0-alpha.1] - 2026-10-16

Experimental organ-analysis preview. This prerelease adds a separate CPU
post-processing command; it does not complete the v0.3.0 batching milestone.
The earlier unreleased reconstruction features listed below are also present
in this source snapshot.

### Added

- `pipeline/scripts/organ-analysis/analyze_organs.py`: explicit JSON-configured
  leaf-instance lifting and confirmed-apex geodesic skeletonization from an
  existing Gaussian PLY, cameras, plant masks and per-photo leaf instances.
  Supports leaf-only inspection followed by checked skeleton-only reuse, or
  one full analysis. No SAM model or training is launched.
- Source/code hashes, effective parameters, failure records, a losslessly
  checked semantic evidence JSON, and a self-contained Canvas inspection viewer.
  Unknown petiole paths and small residual clusters remain explicit.
- Focused CPU fixtures and a real-data Welzo regression against the preserved
  label arrays, source-instance associations and full skeleton JSON. This is
  exploratory geometry, not biological accuracy validation.

## [Unreleased]

### Added

- Video-derived image-sequence guidance in `README.md`, `pipeline/QUICKSTART.md`,
  and `pipeline/LAYOUT.md`. It covers frame sampling and provenance, the
  explicit COLMAP image-sequence mode, camera review before training, and the
  unverified direct-video route.

- Add the MIT license, copyright 2026 Alexander Feldman.

- **`pipeline/scripts/reconstruction-compare/`**: `compare_masked_reconstructions.py` renders two masked
  reconstructions of one capture from the same cameras and scores each against the undistorted ORIGINAL
  photograph inside the region where both masks are foreground (PSNR, SSIM, LPIPS, per photograph and
  per eval token), plus a render-versus-render score. `ns-eval` scores each run against its own mask, so
  its numbers are not comparable across mask variants. `train_mask_variant_arm.sh` trains one arm from a
  copy of a donor run's COLMAP model so both arms share camera poses. Tested by one real comparison
  (Welzo M1, plant against leaf masks); no unit tests, the script needs a CUDA GPU and nerfstudio.

- **`[train] vis`** (`viewer` default, `viewer+tensorboard`, `tensorboard`):
  passes nerfstudio `--vis` so a run can write training-time TensorBoard event
  files (gaussian count, train PSNR, mean eval PSNR/SSIM/LPIPS every 1000 steps)
  into its run directory. `viewer` adds no flag, so existing configs produce the
  same command. Recorded in `run-log.csv` (new `vis` column, widened in place)
  and the archived config's `[run-record]`. The total `Train Loss` tag read NaN
  in testing, and the logged eval has no standard deviation; `ns-eval` remains
  the per-checkpoint tool.

- **`[train] eval_mode`, `train_split_fraction`, `eval_interval`.** `fraction`
  (default, 0.9) holds a share out for `ns-eval`, `interval` holds out every
  n-th, `all` trains on every photograph (and `ns-eval` then scores fit on
  photographs it trained on, not generalization). Refused at load: `fraction` of
  1.0 or outside (0, 1), which would hold nothing out and make `ns-eval` crash
  (use `all`), and `interval` below 2. The train stage also counts the
  photographs COLMAP registered, logs how many train and how many are held out,
  and refuses to start training that would hold out none under a held-out mode.
  `run-log.csv` gained an `eval_mode` column (an existing log is widened in place
  with a `.bak`).

- **`[export] checkpoint_interval`: export every checkpoint that is a multiple of
  it, plus the final one.** `0` (default) is the final checkpoint only. Must be a
  multiple of `steps_per_save` and cannot be combined with
  `save_only_latest_checkpoint = true`. For each earlier checkpoint the export
  stage writes a sibling `config-step-{N}.yml` with `load_step` set (the original
  `config.yml` is never changed), runs `ns-export` and verifies the file as before.
  Exercised end to end with a 300-step run saving and exporting every 100 steps:
  three distinct PLYs. The point-cloud conversion still runs for the final
  checkpoint only.

- **`[train] steps_per_save` and `save_only_latest_checkpoint`: a checkpoint
  every 2500 steps, all kept, by default.** nerfstudio saves every 2000 steps and
  deletes all but the newest, which leaves nothing to export an earlier step
  from, so comparing step counts meant one full training run per count.
  Training once to the largest count and exporting the checkpoints on the way is
  equivalent, because splatfacto's schedules do not depend on
  `max_num_iterations` (learning-rate decay is fixed at 30,000 steps,
  densification stops at 15,000). Both keys are passed to `ns-train` as
  `--steps-per-save` and `--save-only-latest-checkpoint`, and both are
  TrainerConfig fields, so every method accepts them.

  Measured 2026-10-05 on a 24-photo capture: a checkpoint is about 740 bytes per
  gaussian (300,000 gaussians: ~220 MB) and writes in 0.2 to 0.3 seconds, so
  the cost is disk and not time. `save_only_latest_checkpoint = true` restores
  the old behavior. QUICKSTART "Several step counts from one run" has the export
  recipe (`ns-export` has no step option; a copy of `config.yml` with
  `load_step` set does it).

  **Documented alongside it, because it is the obvious next thing to try:
  resuming a finished run (`ns-train splatfacto --load-dir`) does not work on the
  pinned nerfstudio 1.1.5.** The trainer builds its optimizers before the
  checkpoint loads and splatfacto's loader then replaces every gaussian
  parameter, so the optimizers hold stale tensors and the first densification
  step crashes in gsplat's `duplicate()` (index out of bounds). Reproduced twice,
  with `CUDA_LAUNCH_BLOCKING=1` to place the fault.
- **`[dataset] mask_variant`, so one capture can carry several mask sets.**
  `masks_<variant>/` siblings beside `masks/`, each the same parallel tree,
  selected by name: empty reads `masks/` exactly as before, `X` reads
  `masks_X/`. A single directory name and never a path (`/` and `..` are
  refused), so unlike the retired `mask_subpath` it cannot contradict the
  layout or leave the capture. Motivated by mask-set comparison on captures
  that already hold two sets for the same photographs, and by prompt families
  (whole plant, leaf, fruit) wanted simultaneously on one capture.

  Composites are keyed by the variant at the `masked/` level,
  `derived/masked/<variant>/<capture_rel>/`, and deliberately NOT under the
  capture: `composite_masked_images` reuses an existing set only when the
  capture's composite directory, globbed recursively, holds exactly this run's
  files, and a variant nested beneath would make every default run refuse with
  an instruction to delete the whole directory, every variant included. As a
  sibling tree, no variant can see or reuse another's composites, which is the
  failure that mattered: two variants silently sharing composites reconstruct
  identically and "the prompt made no difference" is a believable wrong answer.

  Three refusals and one note come with it. `use_masks = true` with no variant
  named and more than one `masks*` directory present is refused with the sets
  listed, so which masks a run used is never a guess. A variant on a capture
  outside the canonical layout is refused, since it would be read, validated
  and never consulted. A variant that is not one directory name is refused. A
  variant set while `use_masks = false` is accepted, with the fact written to
  the console and to the run log's `note` column, since nothing else in the
  pipeline is a warning surface.

  Recorded in a new `mask_variant` column of `run-log.csv` (an existing log is
  widened in place with a `.bak`, as the header migration already did) and in
  the archived per-run config's `[run-record]`. The `masks` column keeps its
  `used`/`none` vocabulary, which `recover-run-configs.py` compares against;
  that tool now also verifies `mask_variant` for rows that carry it.

  samask has no variant concept and defaults to writing `masks/`, so generating
  a variant means passing its `--masks` explicitly. Stated in every place the
  key is introduced, because it is cheap to say and expensive to discover.

- `Dt4agConfig.mask_variant` and `Dt4agConfig.notes`, the latter the loader's
  accepted-with-reservations lines, logged by the runner and written to the run
  log's `note` column.

- **`[dataset] masked_images_parent_subpath`**, replacing
  `masked_images_subpath` with changed semantics: it names a PARENT that the
  pipeline appends `<mask_variant>/<capture_rel>` to, rather than the composite
  directory itself. One resolution rule now serves the default and the override
  alike, which is what lets a variant apply to both; under the old key the
  override replaced the default path outright and a variant added to the
  default branch would have been silently bypassed.

- **Every run freezes its config** at `<data_root>/configs/runs/<run-id>.ini`:
  the config file verbatim, plus a `[run-record]` section holding what resolves
  only at run time (run id, resolved input/workspace/output paths, and the
  pipeline git commit with a `+dirty` marker). `[run-record]` is an unknown
  section to the loader, so an archived file is still a runnable config and an
  archived run can be reproduced from it.

  Run configs are gitignored on purpose, since this repository is public and a
  real config carries absolute paths and dataset names, so nothing versioned
  them and editing one silently rewrote the record of every earlier run using
  it. `run-log.csv` names the config file per run but never its contents. Found
  by doing it on 2026-08-21: the five tomato configs went from 10,000 to 30,000
  iterations and the previous settings survived only in the per-run `config.yml`
  nerfstudio writes.

  The `pipeline_commit` field closes a second gap. The five 2026-08-18 tomato
  reconstructions were produced by a working tree committed 83 minutes later, so
  the code behind them was identifiable only by timestamp and behaviour.

  Like the run log's row, this states INTENT and not outcome: it is written
  before the stages run, so a file here means a run started, never that it
  finished. The incremental, milestone-bearing per-run record remains v0.3.0's
  job. Archiving never fails a reconstruction; an `OSError` is logged and the
  run continues.

- `Dt4agConfig.config_archive_dir` and `Dt4agConfig.archive_run_config()`, plus
  `_pipeline_commit()`, which reports `unknown` rather than raising when git is
  unavailable.

### Changed

- **Export filenames now carry `{x}steps_from{y}run` and an eval-mode token in
  every case**, replacing the single `{iterations}steps` component: `x` is the
  training steps the exported checkpoint holds, `y` the run's
  `max_num_iterations`, and the token (`evalall`, `evalfrac90`, `evalint8`) says
  which photographs the run trained on, read from the exported run's own
  `config.yml` so it records what training did and not what the INI says now. A
  finished run's final checkpoint reads `30000steps_from30000run_evalall`.
  Anything that matches the old `_{N}steps_` token in a filename must be updated;
  nothing in this repository did. The reason is the per-checkpoint export: a file
  from step 10,000 of a 30,000-step run (`10000steps_from30000run`) must not look
  like a run configured to stop at 10,000, and two runs of one COLMAP workspace
  that differ only in which photographs were held out must not share a name.

- **The per-run config archive is now `{run-id}_{yymmdd-HHMMSS}.ini`, one file
  per invocation, never overwritten.** It was the bare `{run-id}.ini`, so training
  a second time on one run id (an `eval_mode = all` run and a held-out run share a
  COLMAP workspace, hence a run id) overwrote the first training's record. A
  timestamp cannot clash across machines the way a counter would. Files written
  before keep their bare names and remain valid; `recover-run-configs.py` treats
  either form as already archived. `[run-record]` now also carries `stages`,
  `eval_mode`, `steps_per_save` and `checkpoint_interval`.

- **`ns-train` now always receives `nerfstudio-data --eval-mode ...`** (and the
  matching fraction or interval), so which photographs a run trained on is
  stated, not inherited from nerfstudio's default.

### Fixed

- `pipeline/configs/README.md` told you to add your own config to `.gitignore`.
  That had been unnecessary since the blanket `pipeline/configs/*.ini` rule with
  its `!example.ini` exception was written, so the file documented a step the
  repository already took.

### Removed

- **`[dataset] masked_images_subpath`.** Superseded by
  `masked_images_parent_subpath` above. A config carrying the old key with a
  VALUE is refused with a message that says the meaning changed, not just the
  name: renamed mechanically, the same value would put composites one or two
  directories below where it said. An empty leftover `masked_images_subpath =`
  is accepted, because every working config and every archived per-run config
  written before the rename carries it empty, and archived configs are promised
  to stay runnable.

## [0.2.0] — 2026-08-18

**Supports separate raw images and mask images.** The pipeline accepts raw
photographs plus separate mask files and does the masking itself, so the
manual pre-processing step v0.1.0 required is gone.

Verified on eleven captures across two collections, all reconstructing from
the migrated canonical layout, with the five carrying recorded reference
gaussian counts landing within 6% of them.

### Added

- `[dataset] image_extensions` / `mask_extensions`, so a dataset that stores masks
  beside its photographs no longer feeds them to SfM as if they were photographs.
- `[dataset] use_masks`, turning raw photographs plus separate masks into the
  masked-image dataset the pipeline already consumed, by compositing each mask
  into its photograph's alpha channel as a pre-step. Delegates to
  `scripts/rgb-mask/rgb-mask-batch.py` rather than reimplementing it.
- `[dataset] masked_images_subpath`, and optional per-capture provenance in
  `capture.ini`, read and recorded but never affecting behaviour.
- `[paths] derived_dirname`, naming the tree that holds the pipeline's
  rebuildable intermediates. Defaults to `derived`.
- `Dt4agConfig.capture_rel`, the capture's path relative to the datasets
  directory. It is `images_subpath` with the canonical trailing `images`
  component dropped, and it is what the `colmap/`, `outputs/` and `derived/`
  trees are keyed by, since all three describe the capture rather than the
  directory of photographs inside it.
- `pipeline/LAYOUT.md` and `ROADMAP.md`.
- `--compress-level` on `rgb-mask-batch.py`, defaulting to 1. PNG is lossless at
  every level (verified: bit-identical pixels and alpha at levels 0 through 9);
  only speed and size change. Encoding is 96% of that script's runtime, so this
  is roughly 4x faster for 18% more bytes on rebuildable output.
- `[train] downscale_factor`, pinning the training resolution instead of leaving it
  to nerfstudio's on-disk probe.
- The effective downscale factor is now resolved before training, logged to the
  console, and carried in the export filename as `dsN`, so two resolutions of
  one dataset can no longer overwrite each other. The run log's
  `downscale_factor` column holds the CONFIGURED value, not the effective one:
  its row is appended before the stages run, so an unpinned run records the
  literal `auto`. Capturing the resolved value belongs to v0.3.0's per-run
  records.
- `pipeline/MASKING.md` and `pipeline/SEQUENTIAL-RUNS.md`.

### Changed

- Masking is now alpha compositing only. An earlier implementation in this same
  unreleased cycle wired masks in as nerfstudio `mask_path` entries, which
  suppresses background *supervision* but not background *geometry*: measured
  2,385 gaussians via alpha against 96,938 via mask file on one capture. Evidence
  and reasoning in `pipeline/MASKING.md`, which also records that keeping a loss
  masking mode was considered and deliberately rejected.
- **Composited masked images are written under `derived/`, not into
  `datasets/`.** The default is now
  `<data_root>/derived/masked/<collection>/<capture>/`. They were briefly written
  into the input tree earlier in this same unreleased cycle, which put
  rebuildable multi-gigabyte data (roughly 2.3 GB per capture measured) inside
  the one tree that most needs backing up. A `masked_images_subpath` override is
  now resolved against `data_root` rather than the datasets directory, and one
  that resolves inside `datasets/` is refused.
- **Masks are located by layout instead of by configuration.** `<capture>/masks/`
  when `images_subpath` ends in `images`, mirroring it; beside each photograph
  otherwise. Both real arrangements follow from the one rule, so the key that
  used to state it could only ever agree with the filesystem or be wrong.
- The `colmap/`, `outputs/` and `derived/` trees are keyed by `capture_rel`
  rather than `images_subpath`, so a canonical capture's workspace is
  `colmap/<collection>/<capture>/<run-id>/` and not
  `colmap/<collection>/<capture>/images/<run-id>/`.
- The export filename now leads with the capture, via `capture_rel.name`. It
  used to use `images_path.parent.name`, which names the capture only under the
  canonical layout; under a legacy one it named the collection, so every capture
  in a collection shared a filename prefix.

### Removed

- **`[dataset] mask_subpath`.** Superseded by the layout rule above. A config
  still carrying the key is REFUSED with a message naming the replacement,
  rather than ignored: a mask directory silently not read is a run that trains
  against the wrong supervision and still exits 0.
- The `mask_path` route and everything that served it: the COLMAP `images.bin`
  parser, the `colmap_im_id` frame-to-source mapping, and the per-file ffmpeg
  mask downscaling. Compositing happens before ns-process-data renames anything,
  so nothing needs re-pairing afterwards and the mask pyramid is just the image
  pyramid.

### Fixed

- The process stage now verifies that ns-process-data actually built the downscale
  pyramid. Its single ffmpeg image2 sequence stops at the first extension change, so
  a dataset mixing `.jpg` and `.png` silently produced a one-file pyramid, which
  nerfstudio read as no pyramid at all, which made training fall back to native
  resolution and die mid-run with a CUDA OOM on a 6 GB card.
- `ffmpeg` is now a checked prerequisite of the process stage. nerfstudio shells out
  to it and does not check, so without it the stage reported success and produced no
  pyramid.
- The train stage passes `nerfstudio-data --downscale-factor N`, the tyro subcommand
  form. The nested `--pipeline.datamanager.dataparser.downscale-factor` path matches
  how `config.yml` stores the value and is rejected as an unrecognised option.
- The run log upgrades a narrower existing header in place (keeping a `.csv.bak`)
  rather than dropping newer columns forever.
- The staged SfM tree is removed once the process stage no longer needs it. On a
  filesystem without symlinks or hardlinks it is a full copy of the photographs.

## [0.1.0] — 2026-08-13

**Alpha: supports pre-made masked images.**

The Linux alpha, tagged retroactively on 2026-08-18 at the `pipeline-alpha` merge,
which is the real boundary; `pyproject.toml` already declared `0.1.0` there.

- An external tester can install and run the pipeline on a clean Linux machine from
  the repository alone (`pipeline/QUICKSTART.md`).
- Four stages driven from one INI config: COLMAP, ns-process-data, ns-train,
  ns-export, with every subprocess return code checked and every claimed artefact
  verified on disk.
- Masking is supported by *starting from* pre-made masked images (RGBA, alpha
  channel), produced out of band by `pipeline/scripts/rgb-mask/`. The pipeline does
  not composite masks itself.
- GPU architecture is verified against the installed gsplat binary before a run
  starts, rather than failing cryptically during training.
