# Quickstart

Run the reconstruction pipeline from a set of photos to an exported Gaussian
splat. No Jupyter required.

This pipeline is **object-agnostic**: it reconstructs arbitrary scenes and
objects and contains no subject-specific logic.

## 0. What you need

| Requirement | Notes |
|---|---|
| Linux x86-64 with an NVIDIA GPU | Developed on Ubuntu 24.04, RTX 2060 (6 GB), driver 580.173.02 |
| **COLMAP**, CUDA-enabled | **Not installable by pip or uv.** See "COLMAP" below |
| **ffmpeg** | Not installable by uv. `sudo apt install ffmpeg` |
| Python environment | `uv sync --frozen` from the committed lockfile. See section 0a |
| Your images | Any number, in one directory. Sub-directories are supported and are treated as one camera each |

No system CUDA toolkit is needed, and **no C or C++ compiler is needed**: torch
ships its own CUDA 12.1 runtime and gsplat comes as a prebuilt wheel.

## 0a. The environment

There are two, and they are independent. Neither touches the other.

### The uv environment (default, reproducible)

`pyproject.toml` and `uv.lock` at the repository root pin the whole Python side:
Python 3.10, torch 2.4.1+cu121, gsplat 1.4.0+pt24cu121 (**prebuilt**, not
compiled), nerfstudio 1.1.5, numpy 1.26.4.

```bash
cd <repo root>
uv sync --frozen              # ~250 packages, 7.4 GB venv, a few minutes

export DT4AG_COLMAP_PREFIX=/path/to/prefix/containing/bin/colmap
export PATH="$PWD/pipeline/scripts/uv-env:$PATH"

uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

`pipeline/scripts/uv-env/` holds a `colmap` wrapper that scopes
`LD_LIBRARY_PATH` to the COLMAP call. Read that directory's README before
setting `LD_LIBRARY_PATH` yourself: exporting it globally under uv makes torch
load the wrong `libcudart.so.12`.

Why torch 2.4 and not 2.5: 2.4 is the newest torch that gsplat publishes a
prebuilt sm_75 wheel for. On torch 2.5 gsplat compiles itself from source on
first use, which needs nvcc plus a gcc no newer than 12, and cost roughly ten
minutes at the start of every training run. That is all gone. Full rationale,
verification numbers and revert instructions:
`knowledge-base/pm/kibanc-dt/artifacts/2026-08-08_torch24-migration-record.md`
in the notes repository.

### The conda environment `ns-l-oci` (fallback)

The original environment: Python 3.10, torch 2.5.0+cu121, nerfstudio 1.1.5,
gsplat 1.4.0 compiled by JIT. It is deliberately left intact and unmodified, and
it still works:

```bash
conda activate ns-l-oci
python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

Use it if the uv environment misbehaves. It is not reproducible from this repo
(no lockfile, and it depends on `gcc-10` being installed), which is why it is the
fallback rather than the default.

### COLMAP

COLMAP is a C++ binary. It has to be installed separately, whatever you use for
Python packages. Two important notes:

- `colmap --version` **does not work** in 3.12. Run bare `colmap` and read line 2,
  which also tells you whether it was built `with CUDA`. You need the CUDA build.
- If COLMAP was installed *into* a conda environment prefix (as it was on the
  development machine), it links its CUDA libraries from there and needs
  `LD_LIBRARY_PATH` set to that environment's `lib`. Activating the environment
  normally does this for you. Under uv, use the wrapper in
  `pipeline/scripts/uv-env/` instead, which scopes the variable to the COLMAP
  call.

Exactly seven of COLMAP 3.12.0's 76 shared libraries do not resolve against the
system loader path and must come from its prefix: `libcudart.so.12`,
`libGLEW.so.2.3`, `libboost_program_options.so.1.84.0`, `libceres.so.4`,
`libglog.so.2`, `libmetis.so`, `libfreeimage.so.3`.

## 1. Make a config

Everything the pipeline needs lives in one INI file. Nothing is hardcoded.

```bash
cp pipeline/configs/example.ini pipeline/configs/my-run.ini
```

Edit two keys and you are done:

```ini
[paths]
data_root = /path/to/your/data       # everything hangs off this

[dataset]
images_subpath = scene-01/images      # relative to <data_root>/datasets
```

`configs/README.md` documents every other key. The ones you are most likely to
touch:

```ini
[train]
max_num_iterations = 30000            # see "How long" below
[paths]
exports_dirname = auto-exports        # where finished .ply files collect
```

Real config files are gitignored, because they carry real paths and dataset
names. Only `example.ini` is committed, and it stays neutral.

Validate a config without running anything:

```bash
python pipeline/dt4ag_config.py pipeline/configs/my-run.ini
```

## 2. Run it

```bash
# uv (default)
export DT4AG_COLMAP_PREFIX=/path/to/colmap/prefix
export PATH="$PWD/pipeline/scripts/uv-env:$PATH"
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini

# conda (fallback)
conda activate ns-l-oci
python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

That runs four stages in order: COLMAP structure-from-motion, `ns-process-data`,
`ns-train splatfacto`, and `ns-export gaussian-splat`. It exits non-zero if any
stage fails, and it verifies the output file rather than trusting a success
message.

Useful flags:

```bash
--dry-run                 # print the exact commands, run nothing
--stage train             # run one stage only
--from-stage train        # resume partway (needs --run-id)
--run-id val_260808-01-3120
```

### Where the output goes

Each run gets an auto-generated id like `val_260808-01-3120`
(prefix, date, run number that day, COLMAP version). Every path derives from it,
so runs never overwrite each other:

```
<data_root>/colmap/<dataset>/<run-id>/          COLMAP workspace + transforms.json
<data_root>/outputs/<dataset>/<run-id>/         training run, checkpoints
<data_root>/<exports_dirname>/                  the finished .ply files
<data_root>/run-log.csv                         one row per run, for provenance
```

Change `[run] id_prefix` to keep a set of runs separate from another set.

## 3. Check it actually worked

**Do not judge quality by gaussian count.** On data where the subject fills a
small part of the frame, more gaussians means *worse*. COLMAP seeds tens of
thousands of points across the background, and training correctly deletes them.
A good reconstruction of a small subject may hold only ~1,400 gaussians while a
useless one holds 69,000.

Judge with held-out views instead:

```bash
ns-eval --load-config <output-dir>/<run-id>/splatfacto/<timestamp>/config.yml
```

For reference, on the development dataset: a 30,000-step run scored **PSNR
46.5**, and a 500-step run of the same data scored **10.4**.

### How long, and why 500 is not a shortcut

`splatfacto` does not start densifying or culling until step 500. **A run of 500
iterations or fewer produces the raw COLMAP seed cloud, not a reconstruction.**
It is useful for proving the pipeline executes, and useless as a result.

Rough timings on an RTX 2060 with 120 images at 5184x3456:

| Stage | Time |
|---|---|
| COLMAP | ~6 min |
| `ns-process-data` | ~3 min |
| `ns-train` @ 500 | ~25 s (plumbing test only) |
| `ns-train` @ 30,000 | tens of minutes |

## 4. Masks (optional)

If you want to reconstruct a subject without its background, mask the images
first and point `images_subpath` at the masked set.

- `pipeline/scripts/rgb-mask/` applies existing masks to RGB images as an alpha
  channel. **Read its README before using it: it has a known defect** where the
  documented directory layout silently produces nothing.
- Mask generation itself is a separate repository.
- COLMAP ignores the alpha channel, so it still seeds points across the
  background. Training removes them, so this costs efficiency rather than
  quality.

## Troubleshooting

**`libcudart.so.12: cannot open shared object file`**
COLMAP cannot find its CUDA libraries. Activate the conda environment, or set
`LD_LIBRARY_PATH` to its `lib` directory.

**`unsupported GNU version! gcc versions later than 12 are not supported!`**
This means gsplat is trying to compile itself, which should never happen in the
uv environment. Either you are in the conda fallback, or the wrong gsplat wheel
got installed. Check which:

```bash
python -c "import gsplat; from gsplat.cuda._backend import _C; print(gsplat.__version__, _C.__file__)"
# uv env, correct:  1.4.0+pt24cu121  .../site-packages/gsplat/csrc.so
# wrong wheel:      1.4.0            .../torch_extensions/.../gsplat_cuda.so
```

A bare `1.4.0` in the uv environment means the pure-Python PyPI wheel shadowed
the prebuilt one. Do not work around it with `--index-strategy
unsafe-best-match`; that reopens dependency confusion for every package. Re-sync
from the lockfile: the `explicit = true` indexes in `pyproject.toml` are what
prevent it.

In the conda fallback the compile is expected, and CUDA 12.1 refuses a host
compiler newer than GCC 12:

```bash
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
```

**`ModuleNotFoundError: No module named 'setuptools'` on `import gsplat`**
`gsplat/cuda/_backend.py` imports `torch.utils.cpp_extension` at module top
level even when nothing is compiled, and that needs `setuptools`. It is an
explicit dependency in `pyproject.toml` for this reason. If you see this, you
built a venv by hand rather than with `uv sync --frozen`.

**Training starts and never finishes, with no output**
`[train] quit_on_train_completion = false` leaves the viewer running forever. The
runner refuses to start in that state; set it to `true` for unattended runs.

**Running the notebook and COLMAP is not found, but it works in your shell**
The Jupyter kernel does not activate the conda environment, so it inherits
neither `PATH` nor `LD_LIBRARY_PATH` from it. Launch Jupyter from an activated
shell, or use `run_pipeline.py` instead.

**No images found, but the directory clearly has images**
Images may be nested one directory per camera. The check is recursive; if this
still happens, verify `images_subpath` resolves by running the config validator
in step 1.

## The notebook

`notebooks/nerfstudio-pipeline-06.ipynb` runs the same pipeline interactively and
reads the same config file. It is kept for exploration. **For anything
repeatable, use `run_pipeline.py`**: it is diffable, testable, runnable headless,
and does not depend on how Jupyter was launched.

## Tests

```bash
python -m unittest discover -s pipeline/tests -v
```

They cover config loading and the runner's own logic. They deliberately do not
need a GPU, COLMAP, or any real data, so they prove nothing about a real
reconstruction.
