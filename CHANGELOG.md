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

- Camera review adds per-image retained sparse 3D observation counts and descriptive 4-by-4 image-cell occupancy, with camera-specific dimensions and explicit boundary semantics. Per-image grids stay in the JSON file while the terminal prints a compact summary.

- Camera review surfaces folder-specific warnings and marks flagged peaks in the plot. It reports per-folder step summaries and retains valid expected pairs when unexpected names interleave. It separates folder paths and omits steps across missing expected frames; reports its ordering and step policy.

- Experimental per-config COLMAP executable selection, actual executable provenance, refusal of auto threads for unmeasured versions, and refusal of nerfstudio reconstruction on unverified COLMAP versions.

- Add a camera-motion comparison companion for two saved COLMAP solves, with folder-local relative rotations, normalized baselines and explicit missing-frame refusals.

### Changed

- Rescoped the roadmap (2026-11-12). v0.3.0 is now "runs you can trust and account for": it adds a
  required camera review before training (with a recorded override), an operator verdict on each run,
  default catalog records for collection runs and a documented capture-identity limit, and moves batch
  provenance to v0.5.0. Criteria implemented in alpha.15 are marked partly done pending the real
  acceptance run. New milestones: v0.4.0 Blackwell GPUs and higher resolution, v0.5.0 sources, captures
  and selections, v0.6.0 an outside install (Linux or WSL2). Native Windows moves to future work, and
  direct video-file ingestion becomes a non-goal. `docs/v0.3.0-release-plan.md` follows.

### Fixed

- Fix camera review for nested camera folders and JPEG originals composited to PNG, refusing ambiguous frame stems.

## [0.3.0-alpha.16] - 2026-11-12

### Added

- `pipeline/scripts/frame-selection/frame-selection.py`, standard library only. `preview` inventories an
  existing capture's `images/` (any capture, before or after a COLMAP solve), assigns frames to clips
  (`--clips` CSV, `provenance/frames.csv`, or one clip), makes FFmpeg thumbnails and writes an offline page
  in a new folder: every frame starts kept, each clip can have several excluded ranges (also in the
  middle) and an every-Nth sampling with an offset that is fixed on the original sequence before
  exclusions, with a sticky `Download selection JSON` button, an inventory CSV download with a `kept`
  column, and browser autosave. An optional `camera-checks.json` from `camera-review.py` adds tags labeled
  as coming from that prior solve; they never decide the selection. `build` validates the
  `dt4ag-frame-selection/1` JSON (unchanged inventory hash, real filenames, sound ranges and rules, a kept
  list that follows from them, at least 3 frames) and writes a new capture of relative symlinks, or copies
  with `--materialize`, with `provenance/selection.json`, `frame-selection.json` and a `frames.csv` that
  `camera-review.py` reads. `--dry-run` writes nothing; an existing destination is refused; masks are not
  copied. Documented in its README, `VIDEO-WORKFLOW.md` (before the solve, and as a retry after a bad
  one) and `LAYOUT.md`; tested in `pipeline/tests/test_frame_selection.py`.

### Changed

- Clarified WSL memory and COLMAP thread evidence, including the limits of
  the capped-Windows runs and earlier emulation.
- `[train] steps_per_save` now defaults to `5000` (it was `2500`), named `DEFAULT_STEPS_PER_SAVE` in
  `pipeline/dt4ag_config.py`. With `save_only_latest_checkpoint = false`, a 30,000-step run keeps six checkpoints instead of
  twelve, about half the disk (by default it now writes six and keeps one, see the next entry),
  and an `[export] checkpoint_interval` of `15000` (what `auto` resolves to with every checkpoint kept, see
  the next entry) still lands on a saved checkpoint. A config that
  does not set `steps_per_save` but sets `checkpoint_interval` to a multiple of 2500 that is not a multiple of
  5000 (2500, 7500, 12500, ...) now fails validation with the existing "not a multiple of steps_per_save"
  message; set `steps_per_save = 2500` to keep the old cadence. Configs that set `steps_per_save` explicitly
  are unaffected. `example.ini`, `configs/README.md`, QUICKSTART and the run builder's embedded schema follow.
- `[train] save_only_latest_checkpoint` now defaults to `true` (it was `false`), named
  `DEFAULT_SAVE_ONLY_LATEST_CHECKPOINT` in `pipeline/dt4ag_config.py`: a normal run keeps only its newest
  checkpoint instead of six (each about 740 bytes per gaussian: about 220 MB at 300,000 gaussians,
  740 MB at a million). `[export] checkpoint_interval` accepts `auto`, which is its new
  default (it was `15000`): `auto` resolves to `0`, the final export only, when `save_only_latest_checkpoint`
  is `true`, and to `DEFAULT_CHECKPOINT_INTERVAL` (`15000`) when it is `false`. `steps_per_save` stays `5000`.
  A step study is now one key, `save_only_latest_checkpoint = false`, which keeps a checkpoint every 5000
  steps and exports 15,000 and 30,000; adding `checkpoint_interval = 10000` exports 10,000, 20,000 and
  30,000. An explicit number still wins with the same validation (a multiple of `steps_per_save`, and a
  nonzero one is refused with `save_only_latest_checkpoint = true`); `auto` with `false` refuses a
  `steps_per_save` that does not divide 15,000.
  **Behavior change:** a config that sets neither key now keeps one checkpoint and exports the final
  checkpoint only. This includes `--from-stage export` (or `--stage export`) on a run trained under an older
  config that sets neither key: it now exports only the final checkpoint, where it exported 15,000 plus the
  final one before, and the new `checkpoints :` line at the start of the run says so. A config that sets a
  nonzero `checkpoint_interval` but not `save_only_latest_checkpoint` now fails validation; the message
  says that the default changed and how to fix it. To get the old behavior, set
  `save_only_latest_checkpoint = false`, which with `auto` restores the 15,000-step interval. A config that
  sets `save_only_latest_checkpoint = false` explicitly behaves as before. One that sets it to `true`
  without a `checkpoint_interval` was refused before (the old default interval of 15000 needed every
  checkpoint) and now loads with a final-only export.
  **Collections:** `run_collection.py` decides whether a capture is already done by comparing the raw INI
  sections of the base config with those recorded for the earlier run (`run_pipeline.ini_sections`,
  `run_collection.why_incomplete`), not the loaded values. Editing a collection's base config, for example
  adding `save_only_latest_checkpoint = false` or `checkpoint_interval = auto`, therefore makes every
  capture report "settings changed since that run" and run again, even when the effective values are
  unchanged. The `[run-record]` of an archived config now records `save_only_latest_checkpoint`, the
  `checkpoint_interval` the export stage used (the resolved number) and `checkpoint_interval_setting`
  (`auto` or `explicit`); `dt4ag_config.py` and the run's opening lines (dry run included) print the
  resolved value. The run builder lists `save_only_latest_checkpoint` among its common settings and warns
  when a nonzero interval meets `save_only_latest_checkpoint = true`. `example.ini`, `configs/README.md`,
  QUICKSTART ("Several step counts from one run") and the run builder's embedded schema follow.

## [0.3.0-alpha.15] - 2026-11-02

### Added

- `[colmap] num_threads`, default `auto`: the runner chooses COLMAP's
  `--num_threads` just before the colmap stage from the memory available then,
  the smallest of Linux's `MemAvailable`, the room the stage's cgroup leaves
  (`[host] memory_high`, or a limit on the runner's own cgroup such as
  `wsl-guard.py --emulate-vm`) and, on WSL2, Windows' available RAM and commit
  headroom less 2 GiB each. 256 MiB is kept back, each thread is allowed
  1100 MiB per 3.84 MP extracted (the largest photograph after `--quality
  high` shrinks its long side to 2400 px, read from the JPEG or PNG header),
  and the count never exceeds the usable CPUs. Below one thread the run refuses
  to start, with the figures and the override. A number is used as given;
  `all` is COLMAP's own every-core default. An explicit `--num_threads` in
  `extra_args` always wins and the log says so. The log line, `--dry-run`, and
  the archived run record (`colmap_threads`, `colmap_threads_source`) show the
  choice. Only the pipeline's own colmap stage is affected. The constants,
  measured on COLMAP 3.12.0, are kept together in `pipeline/colmap_threads.py`
  and are to be measured again after the COLMAP 4.2.1 upgrade. The run builder
  warns when `num_threads = all` and `extra_args` sets no count. Covered by
  `tests/test_colmap_threads.py`.
- `pipeline/windows_host.py`: `wsl-guard.py`'s Windows memory probe and its
  default thresholds, moved into one module that the guard and the runner both
  use. The guard's behavior is unchanged.
- `pipeline/copy_config.py`, a helper that copies `configs/example.ini` (or
  `--source`) to a new run config. `--destination-dir` picks the folder,
  `--name` or `--name-prefix` picks the file name, and repeated
  `--set SECTION.KEY=VALUE` changes keys that already exist while keeping
  comments. It never overwrites a file or edits the source, and rejects a
  malformed or unknown `--set`. Covered by `tests/test_copy_config.py`.
- `pipeline/scripts/run-builder/`, a run builder: a local page that asks for
  the data root, one capture or a collection (or a video, through the existing
  `video-frames` commands), masks (none, an existing set, or new sets from samask
  prompts, one `masks_{variant}/` per prompt, never merged) and the run settings,
  then writes the config (or one per mask variant, to compare them) and the
  commands in order: samask, dry run, real run. It works opened as a file; an
  optional standard-library server bound to 127.0.0.1 browses folders, counts
  photographs, discovers captures with `run_collection.discover_captures`,
  validates with `load_config` and the real dry run, and saves configs without
  overwriting, refusing any path outside the data root. It runs no stage and no
  samask; samask's location and command template come from a gitignored settings
  file or environment variables. `dt4ag_config.CONFIG_SCHEMA` and
  `config_schema()` (`python pipeline/dt4ag_config.py --schema`) export every
  key the loader reads with its default, allowed values and help from
  `configs/README.md`; tests fail when the schema, the loader and the page's
  embedded copy disagree. `configs/README.md` gains the `downscale_factor` row it
  was missing.

- A test fails when any tracked `.ipynb` has stored cell outputs or a
  non-null execution count, naming the file and cell and pointing to
  `nbstripout`. The notebook was already committed clean; this keeps it so.

- `pipeline/CAMERA-PROFILES.md` now opens with guidance for any camera: the
  pipeline does not correct lens distortion itself, Nerfstudio 1.1.5 undistorts
  and crops images in memory during training, and four practical rules for
  capture settings and for checking a camera solve before training. QUICKSTART
  links to it from the capture-quality section.

- `pipeline/run_collection.py` runs every capture beneath a collection, one at
  a time, from one base config. Captures are discovered by the `images/` rule;
  each capture's config is the base with `images_subpath` replaced, validated
  before anything runs; one failing capture does not stop the rest, and the
  summary names each failure from its catalog job record. A rerun skips a
  capture only when a catalog job shows it verified with the same settings and
  photographs and its export still intact; anything else runs again under a
  new run ID, and nothing is deleted or overwritten. `--dry-run` lists the
  plan. Its decided behaviors are listed in `pipeline/SEQUENTIAL-RUNS.md`
  under "Decisions".

- Add a programmatic camera-path generator that sorts solved video-frame poses
  chronologically, smooths and resamples the route, adds a small end reveal, and
  records source-hash provenance; document terminal rendering from the generated
  path or a custom viewer path.

- Added a z-up to y-up Gaussian-splat conversion command with provenance sidecars
  and a SuperSplat display copy. Alex confirmed the plain y-up file appears upside-down
  in SuperSplat, while the display copy appears upright (2026-10-26).

- Camera review now reports the largest adjacent-camera step relative to the
  median, names the frame transition, and highlights extreme jumps as review
  alerts instead of leaving the spike only in the plot.

- `pipeline/CAMERA-PROFILES.md` records the DJI Osmo Action 6's COLMAP
  `SIMPLE_RADIAL` precedent separately from Nerfstudio's `OPENCV` transform
  representation, with measured reference and failed-solve estimates.

- The video workflow now includes an interactive COLMAP sparse-cloud review
  before training, with a clear stop condition when the camera path looks
  plausible but the reconstructed points do not.

- Every invocation that uses Nerfstudio records the installed distribution
  version in `run-log.csv`, its archived `[run-record]`, and the invocation log.
  The version comes from the runner's Python package metadata; older CSV rows
  remain blank after header migration.

- Repo-local agent instructions provide a task-routing entry point and human-run default; the README and video workflow point to them.
- `[host]` config section, optional, for how stages may use the machine. Empty or
  absent changes no command. `memory_high` (for example `8G`) runs each external
  stage command (`colmap`, `ns-process-data`, `ns-train`, `ns-export`) inside
  `systemd-run --user --scope --quiet -p MemoryHigh=<value>`, so the file cache a
  stage creates is charged to it and evicted at the cap; on WSL2 that keeps the VM
  from growing into, and emptying, Windows' RAM. `process_cpus` (for example
  `0,1`) pins `ns-process-data` alone with `taskset -c`. Both are validated at
  load (malformed values, a cap below 512M and unknown `[host]` keys are refused),
  shown by `--dry-run` in the printed commands, and recorded in the archived
  config's `[run-record]` when set. Before the first stage the runner checks that
  a capped scope can be created and that the memory controller is delegated, and
  refuses to run rather than drop a cap it was asked for.
- `pipeline/scripts/wsl-guard/wsl-guard.py`, a stdlib-only wrapper that runs
  `run_pipeline.py` on WSL2 under a Windows-RAM (and optional drive free-space)
  guard read through `powershell.exe`: it refuses to start without headroom
  (exit 76) after trying a cache release, releases WSL's file cache when RAM runs
  low during the run, and stops only its own process group before Windows runs
  out (exit 75). Writes the run's log, a host-sample JSONL and a status JSON.
  Refuses to run off WSL2.
- `wsl-guard.py` reads the memory of the run's own cgroup every sample
  (`memory.current`, `memory.peak`, anon and file from `memory.stat`, swap, and
  the `oom_kill` count), tolerating any missing file, and records maxima and the
  OOM count in the status JSON (`scope_peak_mib`, `scope_anon_max_mib`,
  `scope_swap_max_mib`, `scope_oom_kills`). Kept only when the run has a cgroup of
  its own; also flags a SIGKILL the guard did not send.
- `wsl-guard.py --emulate-vm 8G:4G` runs the whole invocation in a hard
  `MemoryMax` plus `MemorySwapMax` scope, which is what WSL gives a 16 GB Windows
  machine by default, to check that a run fits. Validated, probed before the run,
  refused together with `[host] memory_high` (a nested scope escapes the outer
  one), and `--print-command` shows the wrapped command. QUICKSTART's 16 GB
  paragraph has a "check your machine" recipe and says what pass means.
- `pipeline/scripts/ply-check/ply-check.py`, a read-only integrity check for
  exported splat PLY files (binary little-endian). Per file it reports vertex
  count, size, sha256, non-finite values per property, opacity and largest-axis
  scale statistics and the 1-99 percentile extent of the centers, as text or
  `--json`. Exits 1 on non-finite values, zero vertices, a truncated body or an
  unreadable file. `--preview {png}` writes a three-view picture of the Gaussian
  centers (not a render). Tests in `pipeline/tests/test_ply_check.py` use
  synthetic PLYs and are skipped where numpy is missing.
- Catalog integration tests for incomplete image pyramids, failed training,
  invalid and valid exports; an optional compatibility check exercises the actual
  Mission Control validator and reader against generated legacy/current/failure fixtures.

- Additive catalog job progress: verified milestones, active and completed stage
  timings, loaded and runtime-resolved configuration, and dependency versions.
  Create enabled catalog jobs before config loading so early configuration
  failures survive; preserve interrupted progress and existing job identities.
- Dedicated hermetic catalog failure-path tests and `pipeline/CATALOG-RECORDS.md`.
- Catalog job progress schema 2 (additive; job records stay catalog v1): an
  explicit milestone ladder (`plant_catalog.MILESTONES`) ranks
  `furthest_milestone`; a `failure` entry names the cause, the active stage and
  the furthest milestone at failure; `gpu` records the driver's devices from
  `nvidia-smi` or null; `artifacts` lists each linked output with its size.
  SIGTERM, including a `wsl-guard` stop, now finalizes an active job as
  cancelled with exit 143. `--catalog-hash-artifacts` opts in to sha256 for file
  artifacts. `plant_catalog.read_progress` reads older jobs with absent fields
  as null. See `pipeline/CATALOG-RECORDS.md`.

- User instructions that existed only outside the repository are now in it, each
  checked against the code. New `pipeline/AGENT-OPERATION.md`: driving the
  pipeline with OpenCode on a local Ollama model, with no hosted-model
  subscription (model context, data-folder permissions, a first dry-run
  request); linked from `README.md` and `AGENTS.md`. New
  `pipeline/COORDINATE-FRAMES.md`: which frame the COLMAP model,
  `transforms.json` and the export are in, and how to project an export back
  into the photographs. `QUICKSTART.md` adds: watching free space on the Windows
  drive that holds WSL; one image shape per camera (`CAMERA_SINGLE_DIM_ERROR`)
  and how to normalize rotated photographs; which COLMAP 3.12.0 options
  `extra_args` accepts; that a dry run leaves no record and how to save its
  output; what a finished run leaves on disk; training all-images and held-out
  passes on one camera solve; using the main checkout's venv from a git
  worktree; the GPU lost inside WSL (Troubleshooting); and timings for a
  24-photograph capture on an RTX 4050 Laptop. `configs/README.md` says
  `mask_extensions` is required with `use_masks = true` and lists the
  `run-log.csv` columns and how an older log is widened.

- `pipeline/scripts/organ-analysis/color_splat.py`, a colored display splat
  export. It applies a frame calibration to a raw 3DGS export, then recolors the
  selected leaf regions from `leaf_ids2.npz` or a validated plant-asset directory,
  in variant A (selected regions colored, other appearance kept), variant B
  (other rows also grayed) or both, and writes a new output directory with a
  `.color.json` sidecar, `legend.md` and `legend.json`. `--viewer supersplat`
  adds a rotated, `display_only` copy for SuperSplat. The analysis copy changes
  only `f_dc_*` and `f_rest_*`. It refuses a missing or mismatched source PLY
  hash and a label-count mismatch. `leaf_lift.opacity_keep` and
  `viewer.PALETTE` are now shared by the export and the existing code. See
  `pipeline/scripts/organ-analysis/README.md`.

- Reviewed leaf-group display layer for the organ-analysis viewers. A new
  `organ-leaf-groups` record (schema version 1) holds a human reviewer's decision
  that several capture-local clusters are one leaf (`same_leaf`) or that two
  leaves sit on one side branch (`side_branch_pair`). `render_viewer.py` renders
  a fresh viewer and `viewer-record.json` from an existing output folder without
  rerunning anything; `viewer.check_leaf_groups` refuses a record whose capture
  or labels hash differs from the displayed data, or that names an unknown or
  doubly grouped cluster. Same-leaf groups get colors outside the cluster
  palette. The groups are display only: labels, cluster IDs and measurements are
  unchanged, and without `--leaf-groups` the page is byte-identical to the
  ungrouped viewer. See `pipeline/scripts/organ-analysis/README.md`.

### Fixed

- QUICKSTART, "COLMAP threads and memory", said 12 MP photographs need
  "roughly two thirds" of the memory per COLMAP thread that 18 MP photographs
  need. `--quality high` shrinks a copy of each photograph to 2400 px on the
  long side before feature extraction, so a 4000x3000 photograph is extracted
  at 2400x1800, slightly more pixels than an 18 MP 3:2 photograph's 2400x1600.
  The section is rewritten around `num_threads = auto`, and keeps
  `--num_threads 4` as the manual setting for a 16 GB machine.
- A run log whose tail a crash left as NUL bytes no longer stops every later
  run at `append_run_log`: the NUL bytes are removed, the damaged file is kept
  beside it as `run-log.csv.nul-{stamp}`, and the new row's note says so.

- `wsl-guard.py` refuses to start a run that trains or exports when the
  interpreter it would use cannot find torch, and names that interpreter. The
  COLMAP prefix ships its own `python`, so after `export
  PATH=$HOME/opt/colmap-prefix/bin:$PATH` a bare `python` or `$(which python)`
  was COLMAP's and the run failed its prerequisite check with a hint that did
  not say which Python ran. `run_pipeline.py`'s torch check now names the
  interpreter and its environment, its hint warns about the prefix's `python`,
  and QUICKSTART (section 3, the wsl-guard examples, Troubleshooting) and the
  wsl-guard README start the guard with `uv run python`.

- `VIDEO-WORKFLOW.md` told readers to drop WSL's file cache before a run to avoid
  running Windows out of RAM. That does not work: the run refills the cache. The
  troubleshooting entry, its config table and a new QUICKSTART subsection ("Memory:
  WSL can empty Windows' RAM") now describe the mechanism and the fixes that do
  work (`[host] memory_high`, the guard, and releasing leftover cache between
  runs, which Windows sees about 40 seconds later).
- Restrict the object-agnostic source check exception to the existing catalog API
  namespace; the previously added integration otherwise made the baseline fail.
- Catalog job dependency probes now read COLMAP's version from its banner when
  `colmap --version` is rejected, as COLMAP 3.12 does; the previous probe read
  only `--version` output. A broken package's metadata now records an unknown
  version instead of failing the run.

### Changed

- QUICKSTART: on WSL2, limit COLMAP's threads (`--num_threads 4` in
  `[colmap] extra_args`) instead of raising the WSL memory cap. COLMAP's
  feature extraction needs about 1 GiB per thread on 18 MP photographs; at its
  default of one thread per core, the worst capture overflowed an 8 GiB VM in
  16 seconds, and with 4 threads it peaked at 3.7 GiB (new section "COLMAP
  threads and memory"). The old `memory=22GB` recipe is withdrawn. Also new:
  "Windows' commit limit: the other way WSL dies". VIDEO-WORKFLOW's thread
  advice now matches.
- The run builder's example mask prompts are `plant` and `leaf`, the variants
  used for stem extraction (plant minus leaf).
- The run builder marks each copied command with a checkbox beside its Copy
  button, and says on each real run that it is one of several separate
  reconstructions when there is more than one config.

- Default exports now include the 15,000-step checkpoint and the final checkpoint
  in a standard 30,000-step run. DEFAULT_CHECKPOINT_INTERVAL in
  pipeline/dt4ag_config.py owns the default; an explicit zero still means
  final-only export. Configs using save_only_latest_checkpoint = true must set
  checkpoint_interval = 0 explicitly.

- `camera-review.py` now has a README and tests. The pure helpers (frame times
  from `frames.csv` or image order, missing and unexpected frames, model folders)
  load without numpy or nerfstudio, which are imported only when the script runs.
  New `--model` option to review a `sparse/` folder other than `0`; a model with
  no images or no points now stops with a message instead of a traceback, and
  frames with no known time no longer break the plot's colors. Checked on a real
  109-frame solve: 109 registered, 73,930 points, 1.30 px mean reprojection.
  The report's keys are unchanged, with `reviewed_model` added.
- Remove the mandatory global GPU-lock release requirement. Keep batches
  sequential internally, allow independent concurrent invocations, and document
  shared-GPU contention risks; optional per-GPU coordination remains future work.

- Catalog emission now requires an explicit storage root as well as root and
  capture identities, so early failures have a known durable destination.


- Adopt catalog job records as the authoritative v0.3.0 invocation record model;
  replace the parallel run-store proposal and document remaining implementation gaps.

- Reconcile the alpha release history and adopt release-time versioning. Define
  the v0.3.0 alpha, beta and release gates, with an ordered release plan.

- `wsl-guard` also checks Windows commit headroom (`FreeVirtualMemory`): it
  refuses to start below `--start-commit-mib` (8192) and stops the run below
  `--stop-commit-mib` (2048). Running out of commit makes Windows stop the whole
  WSL VM even with RAM available, which `--start-mib` could not see.
- The run builder can print a single capture's real run through `wsl-guard`
  (a checkbox, defaulted by the `[run-builder] wsl_guard` setting) and warns
  when `[host] memory_high` is empty.

### Removed

- The subject-neutral wording rule (Alex, 2026-11-01). Tests no longer fail
  when a committed file names a subject such as plants; the rule dated from
  an AI model that balked at biological terms. The pipeline still contains no
  subject-specific logic.

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

## [Pre-alpha, included in 0.3.0-alpha.1]

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
