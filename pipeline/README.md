# jsps-dt4ag pipeline (alpha, unreviewed)

This directory is the first commit of the actual reconstruction pipeline. Until
now this repository held only the demo page in `docs/`, while the working code
sat unversioned on a removable drive.

**Treat this as an alpha snapshot, not a release.** It is committed so the code
is under version control and reviewable, not because it is ready to be run by
someone else. It has known defects, hardcoded paths, and no environment
lockfile. A public release needs the work listed under "Known gaps" below.

## What is here

| Path | What it is |
|---|---|
| `notebooks/nerfstudio-pipeline-06.ipynb` | The pipeline: COLMAP reconstruction, `ns-process-data`, `ns-train splatfacto`, `ns-export gaussian-splat`. Kernel `ns-l-oci`. |
| `scripts/rgb-mask/` | Applies pre-made masks to RGB images as an alpha channel, producing the masked images the notebook consumes. **Has a known defect, see that directory's README.** |
| `scripts/sam/` | SAM helper scripts used while producing masks. |
| `scripts/mask-analysis/` | The UT-vs-KU mask agreement study: per-image IoU comparison and CSV analysis. |

## What is deliberately not here

- **Notebooks `01` to `05`.** The notebook file is one pipeline in six
  copy-and-modify generations, not six stages of a run. `06` is the only one
  that reflects current practice. The earlier five still exist on the source
  drive and are not included, both to keep this commit readable and because two
  of them exceed 5 MB.
- **The data.** Roughly 99 GB of datasets, COLMAP workspaces and nerfstudio
  outputs, plus archived runs. It is excluded by `.gitignore` and needs a
  separate home, most likely a dataset DOI.
- **`samask`.** The segmentation code that produces the masks lives in a
  separate repository of its own. Note that `scripts/mask-analysis/compare_masks.py`
  is a duplicate of the copy in that repository; which one is canonical is
  not yet decided.

## Known gaps

- **Paths are hardcoded** to an absolute mount point on a removable drive
  (`/media/alex/T5Red/DT-data/`). Dataset selection walks a four-level hierarchy
  by substring matching, sets a loop variable without breaking, and so silently
  takes the last match when two siblings match. One cell requires the reader to
  uncomment a line for a three-level hierarchy. Nothing here runs unmodified on
  another machine.
- **The notebook carries scratch cells** (repeated `!ls` variations) and an
  empty `TODO` cell in the middle of the pipeline.
- **Run identity is manual.** The run date, run count and COLMAP version are
  typed by hand each run and concatenated into a project id. There is no run log
  and no auto-incrementing run id.
- **The masking script fails silently.** See `scripts/rgb-mask/README.md`.
- **No environment lockfile, LICENSE, or citation file yet**, and no evaluation
  step: the pipeline ends at export.
