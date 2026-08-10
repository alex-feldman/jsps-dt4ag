# uv-env: the two things uv cannot install

`uv sync --frozen` reproduces the entire Python side of the pipeline. Two
non-Python binaries remain, and this directory is the glue for them.

| Binary | Why not uv | What to do |
|---|---|---|
| **COLMAP 3.12.0, CUDA build** | C++ binary, no usable CUDA build on PyPI | Install system-wide, or point `DT4AG_COLMAP_PREFIX` at an existing prefix and put this directory on `PATH`. See `colmap` in this directory. |
| **ffmpeg** | Binary. `ns-process-data` shells out to it for image downscaling | `sudo apt install ffmpeg`. On the development machine it exists **only** inside the conda prefix and is not on the system `PATH`, which is easy to miss because activating that environment hides the problem. |

## Using it

```bash
cd <repo root>
uv sync --frozen

export DT4AG_COLMAP_PREFIX=/home/alex/miniconda3/envs/ns-l-oci
export PATH="$PWD/pipeline/scripts/uv-env:$PWD/.venv/bin:$PATH"

uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

## Do NOT export LD_LIBRARY_PATH globally

The old conda `activate.d` did `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`. Under
uv that is actively harmful: the uv environment's torch bundles its own
`nvidia-cuda-runtime-cu12` (`libcudart.so.12`) and the conda prefix carries a
different `libcudart.so.12`. A global setting lets torch's loader pick up the
conda copy. The `colmap` wrapper here scopes the variable to the COLMAP exec
alone, which is the only place it is needed.

## What the wrapper replaced

The conda environment's `etc/conda/activate.d/envvars.sh` exported five things.
Four of them are gone entirely under torch 2.4 + prebuilt gsplat, because
nothing compiles any more:

| Old export | Still needed? |
|---|---|
| `CC=gcc-10` | No. Verified: the environment imports and rasterizes with no `gcc` on `PATH` at all. |
| `CXX=g++-10` | No. |
| `CUDAHOSTCXX=g++-10` | No. |
| `TCNN_CUDA_ARCHITECTURES="75;86"` | No. tinycudann is not a nerfstudio dependency and splatfacto does not use it. |
| `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` | Only for COLMAP, and only scoped, which is what the `colmap` wrapper does. |
