# jsps-dt4ag pipeline (alpha)

This directory holds the reconstruction pipeline. Until it was committed this
repository held only the demo page in `docs/`, while the working code sat
unversioned on a removable drive.

**Treat this as an alpha, not a release.** What that means concretely, as of
2026-08-11:

- **It has been run end to end and had its output verified**, including twice on
  a clean `ubuntu:24.04` machine carrying no conda, no CUDA toolkit and no
  compiler.
- **Nothing is hardcoded any more.** Every path and parameter comes from one INI
  file, and the Python environment is reproducible from `pyproject.toml` and
  `uv.lock` at the repository root via `uv sync --frozen`.
- **Nobody but its author has installed it from scratch**, which is the open
  question the alpha exists to answer.
- **It accepts raw photographs plus separate mask files** and composites them
  itself (`[dataset] use_masks`, since v0.2.0). It does not *create* masks;
  producing them is a separate concern with its own repository.
- **There is no LICENSE or citation file yet.** See "Known gaps" below.

## What is here

| Path | What it is |
|---|---|
| `notebooks/nerfstudio-pipeline-06.ipynb` | The same four stages, interactively. Kernel `dt4ag-uv`, the SAME uv environment the CLI uses. Kept for exploration; `run_pipeline.py` is the supported path. |
| `run_pipeline.py` | The same pipeline as a command-line runner, no Jupyter. See "The runner's options" below. |
| `dt4ag_config.py` | The INI config loader both of the above read, so they cannot drift. |
| `scripts/rgb-mask/` | Applies masks to RGB images as an alpha channel. **The pipeline calls this itself** when `[dataset] use_masks = true`, as a pre-step before any stage runs (`composite_masked_images`). Still runnable by hand for masking a set outside a pipeline run. This is the route that actually removes background geometry: see [`MASKING.md`](MASKING.md). |
| `RUN-FLOW.md` | The runner's control flow and every point at which it refuses to continue. For debugging a failed run or extending the runner. |
| `LAYOUT.md` | The directory structure the pipeline expects: what a capture is, the `images/` discovery rule, and the input-versus-derived split. |
| `MASKING.md` | Why a separate mask file and a premade alpha image produce very different reconstructions (76x difference in gaussian count), which mode to use when, and the fix. |
| `SEQUENTIAL-RUNS.md` | What running several reconstructions in sequence does and does not do for you. No batch mode exists; the gaps are listed. |
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

## Installing and running it

**Not here. See [`QUICKSTART.md`](QUICKSTART.md).**

That file is the single place the install and run procedure is written down, so
that the two cannot drift. It covers the whole path from a bare machine: the
prerequisites, cloning, `uv sync --frozen`, the COLMAP recipe, the config keys,
and what to check afterwards.

This file describes what is *in* this directory and how the runner behaves. It
deliberately carries no install steps.

Everything the notebook used to have typed into cells now comes from an INI
file; see `configs/README.md`. Real run configs are gitignored because they
carry real dataset names and paths.

## The runner's options

`run_pipeline.py` runs the same four stages as the notebook, reading the same
INI file through the same loader, with no Jupyter and no kernel involved:

```bash
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
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

## Settings for a test run

**When you are testing the pipeline rather than producing a result, use
`downscale_factor = 4` and `max_num_iterations = 10000`** (Alex, 2026-08-18).

```ini
[train]
max_num_iterations = 10000
downscale_factor = 4
```

Measured on a 120-image session at 5184x3456 on a 6 GB card: downscale 4 with
30000 iterations takes ~22 minutes end to end, downscale 2 with 30000 takes ~45.
Dropping to 10000 iterations cuts that again. A test is asking "did the pipeline
do the right thing", and that question is answered at 10000 iterations and
quarter resolution just as well as at 30000 and half, for a third of the wall
clock.

Two caveats worth knowing rather than rediscovering:

- **Pin `downscale_factor` explicitly for tests.** Left at `0`, nerfstudio picks
  by probing the pyramid on disk, so a test's resolution depends on what happens
  to be there and is not reproducible between datasets.
- **Do not compare a test run against a production run.** Resolution and step
  count both change gaussian counts, so a comparison is only meaningful between
  runs that share them. This is what the `dsN` segment in the export filename is
  for.

## Known gaps

- **The notebook carries scratch cells** (repeated `!ls` variations) and an
  empty `TODO` cell in the middle of the pipeline.
- **The masking script fails silently.** See `scripts/rgb-mask/README.md`.
- **No LICENSE or citation file yet**, and no evaluation step: the pipeline ends
  at export. (`ns-eval` has to be run by hand.)
- **COLMAP and ffmpeg are still outside the lockfile.** They are binaries and
  cannot come from uv. See the COLMAP recipe in `QUICKSTART.md`.
