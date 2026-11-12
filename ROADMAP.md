# Roadmap

What each version **means**, and what has to be true before it ships.

This file holds acceptance criteria, not tasks. Day-to-day execution state (what
is in progress, blocked, or due) lives in the maintainer's project tracker and
deliberately does not appear here: two lists of the same work drift, and the one
in the repository is the one that would go stale unnoticed. If you want to know
what a version promises, read this. If you want to know what is being worked on
today, read the commit log.

Versioning is [SemVer](https://semver.org/) while pre-1.0: MINOR for new
capability, PATCH for fixes. Milestones are annotated git tags.

**Versioning at release, not per feature (adopted 2026-10-20).** A feature commit adds its changelog
entry under Unreleased and does not change the version or create a tag. A release is one commit on
`main`, made when a coherent set of features is finished and tested: it renames Unreleased to the new
version, bumps `pyproject.toml` and `uv.lock`, and is tagged. Published tags are never renumbered. Before
this rule each feature branch bumped the version itself, so parallel branches took out-of-order numbers:
on local `main`, `alpha.10` arrived after `alpha.12`, and `alpha.14` was numbered before `alpha.13` had
been merged.

---

## v0.1.0 — alpha: supports pre-made masked images

**Released 2026-08-13** (tagged retroactively 2026-08-18).

- [x] Four stages driven from one INI config: COLMAP, ns-process-data, ns-train,
      ns-export
- [x] Every subprocess return code checked and every claimed artefact verified on
      disk, including a PLY that exists but declares zero vertices
- [x] Nothing hardcoded; the environment is reproducible from `pyproject.toml`
      and `uv.lock`
- [x] Installs and runs on a clean Linux machine from the repository alone
- [x] GPU architecture verified against the installed gsplat before a run starts
- [x] Masking supported by *starting from* pre-made masked images; the pipeline
      does not create them

**Known limitation, by design:** the pipeline began at masked images, so
producing them was a manual step outside it.

---

## v0.2.0 — supports separate raw images and mask images

**Released 2026-08-18.** The pipeline accepts raw photographs plus separate mask files
and does the masking itself.

- [x] Masks are composited into the photograph's alpha channel as a pre-step,
      after which every stage is unchanged and unaware
- [x] Compositing reuses the existing batch script rather than reimplementing it
- [x] Alpha compositing verified to remove background geometry: 2,385 gaussians
      against 96,938 for the same capture via the rejected mask-file route
- [x] Masks may sit beside the photographs or in a parallel tree
- [x] The training resolution is carried in the export filename, so two
      resolutions of one capture cannot collide. The run log records the
      *configured* value, which is the resolved one whenever it is pinned and
      the literal `auto` when it is not: the row is written before the stages
      run, so it states intent, and a per-run outcome record is v0.3.0's job
- [x] The process stage verifies the downscale pyramid it depends on
- [x] Optional per-capture provenance (`capture.ini`), read and recorded, never
      load-bearing
- [x] Composited output written to `derived/`, not into the input tree
- [x] Canonical layout implemented: a capture is a directory containing
      `images/` (see `pipeline/LAYOUT.md`)
- [x] Verified end to end on a multi-capture collection: eleven captures across
      two collections reconstructed from the migrated layout on 2026-08-18, all
      succeeding, and the five with recorded reference counts landed within 6%
      of them. No composite was written under `datasets/` by any of the eleven

**Acceptance:** someone with photographs and masks, and no knowledge of this
pipeline's history, can reconstruct a masked subject without pre-processing
anything by hand.

---

## The v0.3.0-alpha line: plant-analysis tools (experimental preview)

Since 2026-10-16 the `v0.3.0-alpha.N` prereleases have shipped plant-analysis tools, viewer fixes,
capture preparation and one reconstruction-side addition (the catalog, alpha.11). These releases do not yet fulfill the collection-batching and run-record acceptance criteria
for v0.3.0 below. The preview capabilities are grouped here:

| Area | What it does | Alpha |
|---|---|---|
| Organ analysis | leaf-label lifting and a provisional skeleton from an export, cameras and masks; leaf-free branch review; per-endpoint review viewer | 1 to 4 |
| Plant separation and orientation | memory-lean clustering of a multi-plant export; the soil plane as the scene's up, plant axes, detached-cluster stacking | 5, 7 |
| Plant asset | a versioned, capture-scoped plant-asset contract with a validator; base and apex candidates; a partial semantic package | 6, 8, 9 |
| Frame calibration and measurement | a frame-calibration record, clicked image points to 3D, ruler scale, splat transform with a SuperSplat display copy | 10 |
| Reconstruction catalog | opt-in atomic job records for a run (see "overlap" under v0.3.0) | 11 |
| Viewers | click-viewer legibility, leaf-cluster IDs drawn beside the point cloud | 12, 13 |
| Video to capture | `video-frames` (review page, extract, subset), `camera-review`, `VIDEO-WORKFLOW.md` (serves the v1.0.0 video criteria) | 14 |
| Collection runs and run records (v0.3.0 work, not yet accepted) | `run_collection.py`; catalog job progress schema 2: incremental records, milestone ladder, failure records, per-stage timings | 15 |
| Running on a workstation | `[host]` limits and `wsl-guard`, COLMAP `num_threads = auto`, `copy_config.py`, the run builder | 15 |
| Frame selection and storage defaults | `frame-selection` (offline exclusion preview, selection file, linked capture build); latest-only checkpoints with `checkpoint_interval = auto` | 16 |

**Status:** experimental. Schemas carry their own version (`plant-asset`, `frame-calibration`,
`image-clicks`, `scale-estimate`, `Plant Reconstruction Catalog v1`, `plant-asset-decisions`,
`splat-viewer-copy`); there is no stability promise between alphas.

**What the number means:** an alpha counts previews, nothing more. It does not measure distance to
v0.3.0, whose criteria are below.

**Alpha exit (decided 2026-10-20, scope revised 2026-11-12).** Stay on `0.3.0-alpha.N` until the
v0.3.0 criteria below work end to end. Then publish `0.3.0-beta.1` for independent testing. Publish
`0.3.0` after the documented acceptance test passes and release validation is complete. A
representation choice or a second species asset alone does not satisfy this gate. Plant-analysis
schemas remain experimental unless their own documentation explicitly promises stability.

See [the ordered path to v0.3.0](docs/v0.3.0-release-plan.md).

---

## Milestone plan (revised 2026-11-12)

The earlier plan tied v0.3.0 to unattended collection runs and v0.4.0 to native Windows. Use showed
two different needs. The expensive failures were camera solves that looked healthy by their counts and
were wrong, and a run whose acceptance nobody could reconstruct afterward. The hardware limit was the
GPU, not the operating system: WSL2 works, while the RTX 50 series cannot run the pipeline at all.
So the milestones now follow what makes a reconstruction trustworthy, then what makes it better,
then what lets someone else use it.

| Version | Theme | Gate in one line |
|---|---|---|
| v0.3.0 | Runs you can trust and account for | Every run says how far it got, whether its camera solve was reviewed, and what the operator decided |
| v0.4.0 | Bigger GPUs, higher resolution | Runs on RTX 50-series (Blackwell) GPUs, verified at a higher training resolution |
| v0.5.0 | Sources, captures and selections | Captures are selections of preserved source datasets, with stable identity across roots |
| v0.6.0 | Someone else can use it | A person other than the author installs it (WSL2 or Linux) and runs a capture from the docs alone |
| v1.0.0 | The public research release | A cited dataset and its reconstruction are published together |

Each is a MINOR release under the rule above, with its own alpha and beta line when the work warrants it.
The order of v0.4.0 and v0.5.0 may swap: v0.4.0 depends on whether gsplat can be built for Blackwell,
which is not yet known.

---

## v0.3.0: runs you can trust and account for

Point the pipeline at a capture or a collection, and afterwards know for every run how far it got,
whether its camera solve was reviewed before training, and whether the operator accepted it.
`[~]` means implemented and tested in fixtures, awaiting the real acceptance run.

**Collections**

- [~] Point the pipeline at a collection and every capture beneath it reconstructs, one at a time,
      discovered by the `images/` rule (`pipeline/LAYOUT.md`). `run_collection.py`, alpha.15; a
      partial real GPU run covered three captures
- [~] One base config for a collection, instead of one near-identical config per capture
- [~] Re-running a partly failed batch skips what already completed instead of silently duplicating
      it under new run ids
- [~] Collection execution is sequential within each batch, and the documentation explains
      shared-GPU contention risks for independent concurrent invocations. No mandatory global GPU
      lock (decided 2026-10-20)

**Camera review before training**

- [ ] Every run that trains has a written camera review of its solve (`camera-review`). Training
      refuses without one unless explicitly overridden, and the override is recorded
- [ ] A collection can stop after every capture's camera solve, with a review written for each, then
      train and export only the captures the operator approves; without the option a collection runs
      straight through

**Run records**

Record architecture (decided 2026-10-20): the Plant Reconstruction Catalog's `catalog/jobs/` records
are authoritative for invocation outcomes; frozen configs, command logs and linked artifacts are
supporting evidence.

- [~] A per-invocation JSON job record, written incrementally so a crashed or killed run still shows
      how far it got (progress schema 2, alpha.15)
- [~] An explicit milestone ladder with the furthest milestone reached surfaced as one field
- [~] Failed runs get records too, with the failing stage and the last milestone
- [~] The record carries the resolved settings, tool versions, per-stage timings and the artifacts
      produced; each run also freezes its config verbatim at `{data_root}/configs/runs/{run-id}.ini`
- [ ] Catalog records are written by default for collection runs, not only when opted in
- [ ] **Operator verdict.** A run can carry the operator's decision (`accepted`, `rejected` or
      `failed`, with who, when, why and an evidence pointer), set only by a person, never inferred.
      A camera-path approval and a final verdict are different fields
- [ ] `catalog/index.csv`, a queryable summary REBUILDABLE from the job records, including the verdict
- [ ] Capture identity is documented: catalog capture IDs derive from the capture's path, and that may
      change in v0.5.0

**Acceptance:** run a real collection with one deliberate failure and a rerun. Afterwards, for every
capture, say whether it succeeded, how long it took, what it produced, at which milestone it failed if
it did, whether its solve was reviewed, and what the operator decided.

**Moved out of v0.3.0:** batch provenance (one record linking a collection's jobs) goes to v0.5.0,
where capture identity is settled first.

**Superseded:** `run-log.csv` becomes a frozen beta log. It records intent only and is not
back-filled.

---

## v0.4.0: bigger GPUs, higher resolution

The prebuilt gsplat wheel has no code for RTX 50-series (Blackwell) GPUs, so a 16 GB RTX 5060 Ti
cannot run training or export, and work stays on smaller cards at quarter resolution.

- [ ] Find out whether gsplat can be built for Blackwell (sm_120) with a supported CUDA and torch, and
      record the result either way. If it cannot, this milestone waits and v0.5.0 goes first
- [ ] The install supports Blackwell alongside the existing Volta-to-Hopper range, and the startup GPU
      check reflects what was actually built
- [ ] A reference capture trains at downscale 2 on a 16 GB card, with measured time, memory and disk
- [ ] QUICKSTART states GPU memory needs by downscale and capture size from measurements

**Acceptance:** the same capture reconstructs on a Blackwell card at a higher resolution than before,
and the result is reviewed against the quarter-resolution run.

---

## v0.5.0: sources, captures and selections

Direction approved 2026-11-12: original videos in `videos/`, source photographs or extracted frames
in `datasets/`, and reconstruction inputs in `captures/` as selections of preserved sources. The
`frame-selection` tool (alpha.16) builds such a selection today inside the existing layout.

- [ ] Opt-in `videos/`, `datasets/` and `captures/` roots, with existing layouts still working
- [ ] Capture identity is stable across roots, so two captures with the same relative name cannot
      share output keys or catalog IDs
- [ ] A selection is reproducible from its recorded source, rules and resolved frame list, and can be
      materialized for standalone sharing
- [ ] Masks follow a selection with provenance
- [ ] Batch provenance (moved from v0.3.0): one record linking a collection's jobs, built on the
      settled capture identity

**Acceptance:** two sampling choices of one video become two captures without copying frames, each
traceable to the same source, and both run and record correctly.

---

## v0.6.0: someone else can use it

- [ ] Installs and runs on Linux and on Windows through WSL2 from the repository alone
- [ ] The install path and one capture run are verified by someone other than the author, following
      the documentation without help
- [x] LICENSE present
- [ ] CITATION.cff present

**Acceptance:** a collaborator can install it, run it on their own capture, and cite it.

---

## v1.0.0: the public research release

- [ ] Everything in v0.6.0, stable across at least two independent installs
- [ ] The published documentation describes a pipeline someone has actually run from it, start to
      finish, without the author present
- [ ] A dataset and its reconstruction are published together, each citable
- [x] **Video-derived frame sequences run through the image pipeline.** Frames are sampled and
      reviewed outside the pipeline, stored in the canonical `images/` layout, and run through the
      ordinary image stages. Exercised on real captures in October and November 2026
- [ ] The video workflow (review, extract, select, solve, review the solve) has been followed end to
      end by someone other than the author

**Acceptance:** the repository stands on its own as a research artefact.

---

## Deferred decisions

Considered, deliberately not done, recorded so they are not re-litigated from scratch.

- **Architecture Decision Records** (numbered, immutable `docs/adr/NNNN-*.md`, Nygard format).
  Considered 2026-08-18 and **held off**: the reasoning they would capture is already in the commit
  history and in the topic documents, and ADRs carry a standing discipline (never edit a decision,
  write a superseding record instead) whose cost looked higher than the convenience bought. Worth
  revisiting if either becomes true: a second person starts making architectural decisions here, or a
  decision gets reversed twice and nobody can reconstruct why.
- **Separate versioning for the plant-analysis tools.** They ship inside this repository as
  experimental modules and do not move the pipeline's version gates. If they need their own stability
  promise, give them their own package and version rather than holding the pipeline's number.

## Future, not scheduled

- [ ] Native Windows, without WSL2 (was v0.4.0 until 2026-11-12). WSL2 is the supported route on
      Windows hosts; native support means maintaining COLMAP and gsplat builds for Windows
- [ ] Launch and monitor runs from the run builder
- [ ] Submit jobs to a pipeline host over the LAN. Input data placement is the open problem: the
      photographs must already be on the host, or travel there with the job

## Explicit non-goals

- **macOS support.** Blocked upstream, not by this code. Revisit only if gsplat publishes a non-CUDA
  backend.
- **Mask generation.** Producing masks is a separate concern with its own repository. This pipeline
  applies masks; it does not create them.
- **Direct video-file ingestion** (removed from v1.0.0 on 2026-11-12). Passing a movie straight to the
  pipeline skips the human checks the video workflow is built on: choosing the range and interval,
  selecting frames, and reviewing the solve. `[nerfstudio] scene_type = video` was never verified with
  this pipeline's COLMAP reuse; remove it or document it as unsupported.

**No longer a non-goal: a GUI** (Alex, 2026-11-01). A local run builder writes configs and commands
(`pipeline/scripts/run-builder/`); the CLI stays the interface that runs things.
