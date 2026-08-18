# Changelog

Versioning is [SemVer](https://semver.org/) while pre-1.0: MINOR for new
capability, PATCH for fixes. Milestones are marked by annotated git tags, not by
long-lived branches, because this repository has one committer working across two
machines and a tag records a milestone permanently at zero merge cost.

Dates are the tag date, not the commit date, where they differ.

## Unreleased

Work toward **0.2.0 — supports separate raw images + mask images**.

### Added

- `[dataset] image_extensions` / `mask_extensions`, so a dataset that stores masks
  beside its photographs no longer feeds them to SfM as if they were photographs.
- `[dataset] mask_subpath`, for masks kept in a parallel tree of identical shape
  rather than next to each image.
- `[dataset] use_masks`, wiring masks into `transforms.json` as nerfstudio
  `mask_path` entries. **Known defect:** this suppresses background *supervision*,
  not background *geometry*, so masked runs still carry large amounts of background.
  Not equivalent to the pre-made alpha route. See `pipeline/MASKING.md`; the fix is
  to composite masks as an alpha channel instead.
- `[train] downscale_factor`, pinning the training resolution instead of leaving it
  to nerfstudio's on-disk probe.
- The effective downscale factor is now resolved before training, logged, written
  to the run log, and carried in the export filename as `dsN`, so two resolutions of
  one dataset can no longer overwrite each other.
- `pipeline/MASKING.md` and `pipeline/SEQUENTIAL-RUNS.md`.

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
