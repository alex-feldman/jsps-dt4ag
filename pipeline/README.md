# jsps-dt4ag pipeline (alpha, unreviewed)

This directory is the first commit of the actual reconstruction pipeline. Until
now this repository held only the demo page in `docs/`, while the working code
sat unversioned on a removable drive.

**Treat this as an alpha snapshot, not a release.** It is committed so the code
is under version control and reviewable, not because it is ready to be run by
someone else. It has known defects and hardcoded paths. A public release needs the work
listed under "Known gaps" below.

The Python environment **is** now reproducible: `pyproject.toml` and `uv.lock`
at the repository root, installed with `uv sync --frozen`. See
`QUICKSTART.md` section 0a.

## What is here

| Path | What it is |
|---|---|
| `notebooks/nerfstudio-pipeline-06.ipynb` | The same four stages, interactively. Kernel `dt4ag-uv`, the SAME uv environment the CLI uses. Kept for exploration; `run_pipeline.py` is the supported path. |
| `run_pipeline.py` | The same pipeline as a command-line runner, no Jupyter. See "Running it from the command line" below. |
| `dt4ag_config.py` | The INI config loader both of the above read, so they cannot drift. |
| `scripts/rgb-mask/` | Applies pre-made masks to RGB images as an alpha channel. **Run by hand: the pipeline does NOT call it.** The pipeline starts at masked images; see QUICKSTART section 4. |
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

- **The development machine's COLMAP was built into the conda environment
  prefix** and links its CUDA libraries from there, so activating the
  environment on `PATH` alone is not enough. Without `LD_LIBRARY_PATH` it dies
  with `libcudart.so.12: cannot open shared object file`:

      export PATH=<conda-prefix>/bin:$PATH
      export LD_LIBRARY_PATH=<conda-prefix>/lib:$LD_LIBRARY_PATH

  This applies to the conda fallback only. The supported route installs the
  conda-forge COLMAP build, which resolves its own libraries via an
  `$ORIGIN`-relative RPATH and needs nothing but `PATH`. See
  `pipeline/QUICKSTART.md`.

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

## Running it from the command line

`run_pipeline.py` runs the same five stages as the notebook, reading the same
INI file through the same loader, with no Jupyter and no kernel involved:

```bash
conda activate ns-l-oci
python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

| Option | What it does |
|---|---|
| `--config PATH` | The config to use. Without it: `$DT4AG_CONFIG`, then `configs/example.ini` found by walking up from the working directory. |
| `--stage NAME` | Run only these stages, from `colmap`, `process`, `train`, `export`. Repeatable and comma-separated (`--stage train,export`). They always execute in pipeline order whatever order you name them in. |
| `--from-stage NAME` | Resume: this stage and everything after it. Mutually exclusive with `--stage`. |
| `--run-id ID` | Address an existing run instead of deriving a new id. |
| `--dry-run` | Print the exact commands and paths, execute nothing. |
| `--allow-viewer-hang` | Permit `[train] quit_on_train_completion = false`. |

Start with `--dry-run`. It resolves the config, derives the run id, counts the
input images and prints every command verbatim, in about a second.

The runner exits non-zero on the first failure and never continues past one:
`2` for a configuration error, `1` for a stage failure, `130` on Ctrl-C. It
checks every subprocess return code, and then checks the artefact each stage
claims to have produced: a sparse model after COLMAP, `transforms.json` after
`ns-process-data`, and after export a `.ply` that exists, is bigger than a bare
header, and declares a non-zero vertex count in that header. A success message
from a tool is not accepted as evidence.

Three things worth knowing before a long run:

- **Prerequisites are checked first**, and only the ones the selected stages
  need. A missing `colmap`, a COLMAP that cannot load `libcudart.so.12`, a
  missing `ns-*` command or a torch that reports no CUDA device all fail
  immediately with a message naming the fix, rather than forty minutes in.
- **Set `[train] quit_on_train_completion = true` for command-line runs.**
  Otherwise `ns-train` keeps its viewer alive after training finishes and never
  exits, which deadlocks a non-interactive run. The runner refuses to start the
  train stage otherwise; `--allow-viewer-hang` overrides that.
- **Resuming needs the run id.** With `[run] date` and `run_count` left blank,
  the id auto-increments, so a second invocation derives a *new* id rather than
  the one you meant. Either resume within one invocation (`--from-stage`), or
  pass `--run-id`, or pin the values in the config. A stage that needs a COLMAP
  workspace which does not exist says exactly this instead of failing obscurely.

Checkpoint discovery before export requires **both** a `config.yml` and a real
checkpoint file: a crashed `ns-train` leaves a run directory holding `config.yml`
and no weights, and `ns-export` against it dies deep inside the checkpoint
loader. Run directories predating the training this invocation performed are
also refused, so a failed training run cannot be papered over by exporting an
earlier one.

**Commit the notebook with its cell outputs cleared.** Executed cells store
real dataset paths inside the `.ipynb`; `tests/test_dt4ag_config.py` fails the
build if subject-specific terms reach the committed copy.

## Known gaps

- **The notebook carries scratch cells** (repeated `!ls` variations) and an
  empty `TODO` cell in the middle of the pipeline.
- **The masking script fails silently.** See `scripts/rgb-mask/README.md`.
- **No LICENSE or citation file yet**, and no evaluation step: the pipeline ends
  at export. (`ns-eval` has to be run by hand.)
- **COLMAP and ffmpeg are still outside the lockfile.** They are binaries and
  cannot come from uv. See the COLMAP recipe in `QUICKSTART.md`.
