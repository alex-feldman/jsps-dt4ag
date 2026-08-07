# Quickstart

Run the reconstruction pipeline from a set of photos to an exported Gaussian
splat. No Jupyter required.

This pipeline is **object-agnostic**: it reconstructs arbitrary scenes and
objects and contains no subject-specific logic.

## 0. What you need

| Requirement | Notes |
|---|---|
| Linux with an NVIDIA GPU | Developed on Ubuntu 24.04, RTX 2060 (6 GB) |
| **COLMAP**, CUDA-enabled | **Not installable by pip or uv.** See "COLMAP" below |
| Conda environment | Python 3.10, torch 2.5.0+cu121, nerfstudio 1.1.5, gsplat 1.4.0 |
| Your images | Any number, in one directory. Sub-directories are supported and are treated as one camera each |

There is no lockfile yet, so the environment is not reproducible from this repo
alone. That is known and tracked.

### COLMAP

COLMAP is a C++ binary. It has to be installed separately, whatever you use for
Python packages. Two important notes:

- `colmap --version` **does not work** in 3.12. Run bare `colmap` and read line 2,
  which also tells you whether it was built `with CUDA`. You need the CUDA build.
- If COLMAP was installed *into* a conda environment prefix (as it was on the
  development machine), it links its CUDA libraries from there and needs
  `LD_LIBRARY_PATH` set to that environment's `lib`. Activating the environment
  normally does this for you.

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
`gsplat` compiles a CUDA extension on first use and CUDA 12.1 refuses a host
compiler newer than GCC 12. Point it at an older one that is already installed:

```bash
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
```

The result is cached, so this is a one-off per environment.

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
