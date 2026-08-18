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
