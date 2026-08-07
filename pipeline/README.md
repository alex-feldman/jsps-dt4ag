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

## Running it

Everything the notebook used to have typed into cells now comes from an INI
file; see `configs/README.md`. Point `DT4AG_CONFIG` at your own copy of
`configs/example.ini` and run the notebook on the `ns-l-oci` kernel. Real run
configs are gitignored because they carry real dataset names and paths.

Two environment traps, both of which produce confusing failures:

- **COLMAP was built into the conda environment prefix** and links its CUDA
  libraries from there, so activating the environment on `PATH` alone is not
  enough. Without `LD_LIBRARY_PATH` it dies with
  `libcudart.so.12: cannot open shared object file`:

      export PATH=<conda-prefix>/bin:$PATH
      export LD_LIBRARY_PATH=<conda-prefix>/lib:$LD_LIBRARY_PATH

- **gsplat JIT-compiles its CUDA extension on first import.** CUDA 12.1's
  `nvcc` refuses any host compiler newer than GCC 12, so on a distribution
  shipping GCC 13 the first `ns-train splatfacto` fails with
  `unsupported GNU version! gcc versions later than 12 are not supported!`.
  Point the build at an older compiler that is already installed; nothing in
  the conda environment needs to change:

      export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10

  A full rebuild is 26 objects. The result is cached under
  `~/.cache/torch_extensions/`, so this is a one-off per environment, but the
  variables must be present on any run that might trigger a rebuild.

**Commit the notebook with its cell outputs cleared.** Executed cells store
real dataset paths inside the `.ipynb`; `tests/test_dt4ag_config.py` fails the
build if subject-specific terms reach the committed copy.

## Known gaps

- **The notebook carries scratch cells** (repeated `!ls` variations) and an
  empty `TODO` cell in the middle of the pipeline.
- **The masking script fails silently.** See `scripts/rgb-mask/README.md`.
- **No environment lockfile, LICENSE, or citation file yet**, and no evaluation
  step: the pipeline ends at export.
