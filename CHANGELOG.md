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

## [Unreleased]

### Added

- Add the MIT license, copyright 2026 Alexander Feldman.

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
