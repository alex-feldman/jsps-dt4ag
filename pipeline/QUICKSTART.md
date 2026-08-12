# Quickstart

Run the reconstruction pipeline from a set of photos to an exported Gaussian
splat. No Jupyter required.

This pipeline is **object-agnostic**: it reconstructs arbitrary scenes and
objects and contains no subject-specific logic.

## Install it with the script

> **VALIDATED END TO END on 2026-08-13.** `scripts/install.sh` was run on a bare
> `ubuntu:24.04` container holding no compiler, no `curl`, no `git` and no
> Python. It completed unaided (exit 0), and the four-stage pipeline then
> reproduced the reference reconstruction: **1,324 gaussians at PSNR 46.530**,
> against references of 1,301 at 46.537 and 1,343 at 46.570 on the same GPU.
> Re-running the script took 3.4 seconds and correctly skipped every completed
> step.

**`scripts/install.sh` is the supported install route.** Sections 0 and 3 below
describe the same procedure by hand. Read them when you need to adapt a step,
when the script fails and you want to know what it was doing, or when you are
installing somewhere it does not fit.

```bash
git clone https://github.com/alex-feldman/jsps-dt4ag.git
cd jsps-dt4ag
scripts/install.sh
```

It does the apt prerequisites, `uv`, the micromamba COLMAP 3.12.0 CUDA prefix
with ffmpeg beside it, and `uv sync --frozen`, then **verifies the result**:
that the GPU is one the prebuilt gsplat wheel can actually run, that COLMAP
reports `with CUDA`, that ffmpeg resolves, and that the installed gsplat is the
prebuilt wheel rather than the pure-Python PyPI one. It skips whatever is
already installed, so it is safe to re-run after a partial failure, and it stops
at the first thing that does not check out rather than continuing.

Only the apt step needs root, and it is skipped entirely when those packages are
already present, so a normal desktop Ubuntu can run the whole thing with no
sudo.

**Two things must exist before the script can run, and a minimal container has
neither:** `git`, because that is how you obtain this repository, and `sudo` (or
a root shell) for the apt step. A desktop or server Ubuntu has both. If `sudo`
is genuinely absent the script says so and prints the exact `apt-get` line to
hand an administrator, after which it needs no root at all.

To see what it would do without doing it:

```bash
scripts/install.sh --dry-run
```

That prints every command, and nothing else, as a runnable shell script. The
intention is that the manual steps in this document get generated from that
output, so the two routes cannot drift apart; today they are still maintained by
hand and kept in step deliberately. One deliberate difference: the script fetches
micromamba with `curl -fLs` rather than `-Ls`, so an HTTP error cannot be saved
as if it were the binary.

Note that the verification commands in the `--dry-run` output only print their
results for you to read. The script itself checks them and stops. So running the
generated output as a script installs the same thing but does not verify it.

Everything below still applies: read section 1 before you buy or borrow a GPU,
and section 4 before your first run.

## Pre-steps: getting the machine ready

Two things must be true before anything in this document works, and neither the
script nor `uv` can supply them. Both need a reboot, which is why they are
separate steps rather than part of the installer.

**Check first. If both of these already work, skip this whole section.**

```bash
nvidia-smi        # must print a table naming your GPU
git --version     # you need git to clone the repository
```

### The NVIDIA driver, on a normal Ubuntu machine

A fresh Ubuntu install usually has no NVIDIA driver. Ubuntu's own tool picks a
suitable one for your card:

```bash
sudo ubuntu-drivers install
sudo reboot
```

Then check `nvidia-smi` again. Two things to look for in its output:

- It names your GPU. If instead you get
  `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA
  driver`, the driver modules were built for a different kernel than the one
  you booted. This happened on the development machine: the driver metapackage
  was pinned to an older kernel while a newer one was running. The fix is to
  install the modules package matching your running kernel and reboot again,
  for example `sudo apt install
  linux-modules-nvidia-580-open-generic-hwe-24.04`.
- The **`CUDA Version`** field in the top-right reads **12.1 or higher**. torch
  ships its own CUDA 12.1 runtime, but the driver still has to be new enough to
  run it. An older driver fails later, during training, rather than here.

### WSL2, on Windows

**Untested as of 2026-08-12.** WSL2 is Linux x86-64, so everything in this
document should apply unchanged once the GPU is visible, but nobody has run this
pipeline there yet. Please report the result either way.

The driver rule is the **opposite** of the one above, and getting it wrong is
the classic WSL mistake:

1. **On Windows**, install the normal NVIDIA Windows driver. That is what gives
   WSL the GPU.
2. In PowerShell: `wsl --install -d Ubuntu-24.04`, then reboot.
3. **Inside WSL, do NOT install any Linux NVIDIA driver.** No
   `apt install nvidia-driver-*`, no `ubuntu-drivers install`. The GPU arrives
   through `/usr/lib/wsl/lib` from the Windows side, and installing a Linux
   driver over it is how people break a working setup.
4. Inside WSL, check:

```bash
nvidia-smi
```

If that prints your GPU, continue with this document from section 0 exactly as
written. A WSL Ubuntu is minimal in the same way a container is, so it will need
the `apt` packages in section 0 rather than already having them.

Budget disk space on the **Windows** drive: the environment alone is about 7.4 GB
and the COLMAP prefix another 4 GB, before any data.

## 0. What you need

| Requirement | Notes |
|---|---|
| Linux x86-64 | Developed on Ubuntu 24.04, driver 580.173.02. **Linux only**, see "Which platforms work" below |
| An NVIDIA GPU, Volta through Hopper | **RTX 50 series does NOT work.** See "Which GPUs work" below before anything else |
| **A host C/C++ compiler** | `sudo apt install build-essential`. Needed at install time AND at training time. See "You do need a host compiler" below |
| **X11 and OpenGL runtime libraries** | `sudo apt install libx11-6 libgl1 libgomp1`. `open3d` will not import without them. See "System libraries" below |
| **`git`, `curl`, `ca-certificates`** | `sudo apt install git curl ca-certificates`. A bare `ubuntu:24.04` has none of them, and you need `git` before you can even clone. See "Get the code" |
| **An NVIDIA driver** | `nvidia-smi` must work, reporting CUDA 12.1 or higher. Nothing here installs it. See "Pre-steps" above |
| **`uv`** | `curl -LsSf https://astral.sh/uv/install.sh \| sh`. See "Installing uv" below |
| **COLMAP 3.12.0**, CUDA-enabled | **Not installable by pip or uv.** Copy-paste recipe under "COLMAP" below |
| **ffmpeg** | Not installable by uv. Comes with the COLMAP recipe below, or `sudo apt install ffmpeg` |
| Python | **Nothing to install.** uv downloads its own CPython 3.10; the system Python, if any, is not used |
| Your images, or the sample dataset | A published 2.2 GB sample exists and is the best way to verify an install. See "Where to get images" in section 4. Sub-directories are supported and are treated as one camera each |

**No system CUDA toolkit is needed, and no CUDA compiler (`nvcc`) is ever
needed**, at install time or at run time: torch ships its own CUDA 12.1 runtime
and gsplat comes as a prebuilt wheel. Verified on a bare `ubuntu:24.04`
container: the full 30,000-iteration run completes with no `nvcc` anywhere on
`PATH` and with `TORCH_EXTENSIONS_DIR` still empty afterwards, which is the
proof that gsplat did not build itself.

**A host C/C++ compiler is a different question, and the answer is yes.** See
the next section. Getting rid of `nvcc` and the ten-minute gsplat JIT is what
the torch 2.4 migration bought; it did not remove `cc`.

### Which platforms work

**Linux x86-64 only, and that is a deliberate scope decision for this release,
not an oversight.** Two things pin it, and neither is a matter of effort:

- The prebuilt `gsplat 1.4.0+pt24cu121` wheel this environment depends on is
  published for `linux_x86_64` and nothing else.
- COLMAP publishes no macOS binary in any release, and the conda-forge CUDA
  build used here is `linux-64`.

macOS and Windows are tracked as beta work rather than abandoned. Expect macOS
to be the hard one: no CUDA at all, so it may end up supporting viewing and
analysis rather than training.

**WSL2 has not been tested, and is expected to work.** It is stated as a
hypothesis so nobody mistakes it for a supported route. WSL2 *is* Linux x86-64,
so the wheel platform tag, the conda-forge COLMAP and the whole uv path should
apply unchanged. The part that is genuinely unproven is the GPU, which WSL2
reaches through a passthrough driver stack rather than a native one. Setup steps
and the do-not-install-a-Linux-driver rule are under "Pre-steps" above. If
`nvidia-smi` works inside WSL and
`uv run python -c "import torch; print(torch.cuda.is_available())"` prints
`True`, the rest of this document should apply verbatim. Report the result
either way.

### You do need a host compiler

```bash
sudo apt install build-essential
```

**Install this before `uv sync --frozen`, not after.** It is needed at two
separate moments, and the second one is easy to miss because it only bites on a
machine that has never run the pipeline before.

**At install time**, two of nerfstudio's transitive dependencies ship no Linux
wheel and are compiled from their sdist. Without a toolchain the sync dies
partway through, after the several-minute download, and you have to run the
whole thing again:

| Package | Pulled in by | Needs | Failure without it |
|---|---|---|---|
| `pyliblzfse 0.4.1` | `nerfstudio` -> `viser` | a C compiler | `error: [Errno 2] No such file or directory: 'cc'` |
| `fpsample 1.0.2` | `nerfstudio` | a C++ compiler | `CMake Error: ... Could not find the compiler specified in the environment variable CXX: c++.` |

`pyliblzfse 0.4.1` publishes wheels for macOS and Windows only; `fpsample 1.0.2`
publishes no wheel for any platform, only an sdist. Neither is going to change,
so this is a permanent property of the lockfile, not a transient. CMake arrives
as a build-time wheel and does not need installing.

**At training time**, `splatfacto` compiles a small C file too.
`nerfstudio/models/splatfacto.py` decorates `get_viewmat` with
`@torch_compile()`, so the first training step runs `torch.compile`, which goes
through inductor to triton, and triton builds its `cuda_utils` driver module
with the host compiler. Thirty-one seconds into stage 3, on a machine with no
`cc`:

```text
File ".../triton/runtime/build.py", line 32, in _build
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: ...
```

The result is cached in `~/.triton/cache/`, so it happens once per machine and
never again, which is exactly why it was invisible on the development machine.
This is **not** gsplat compiling: gsplat stays prebuilt and
`TORCH_EXTENSIONS_DIR` stays empty. Do not go looking for a gsplat problem when
you see this.

### System libraries

```bash
sudo apt install libx11-6 libgl1 libgomp1
```

`open3d` is a nerfstudio dependency and
`nerfstudio/process_data/metashape_utils.py` imports it unconditionally, so
**stage 2 cannot start** without these. A desktop Ubuntu already has them; a
minimal server image or a container does not, and the failure is a bare
`OSError` several minutes into the run:

```text
File ".../open3d/__init__.py", line 38, in <module>
OSError: libX11.so.6: cannot open shared object file: No such file or directory
```

The three above are the complete set for `open3d 0.19.0`, confirmed with `ldd`.
Separately, `pymeshlab` prints `Cannot load library ... libio_x3d.so:
(libOpenGL.so.0: ...)` during export. That one is **harmless**: the export
completes and the `.ply` is correct. Install `libopengl0` if you want it quiet.

### Installing uv

`uv` is a single static binary and needs no Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"      # or: source $HOME/.local/bin/env
uv --version
```

Verified with uv 0.12.3. uv also supplies the interpreter: `uv sync --frozen`
downloads CPython 3.10 itself, so a base system with **no** `python3` at all is
fine.

**`uv sync --frozen` is not the whole install, and cannot be.** Two native
binaries must be on `PATH` before stage 2 will start: `colmap` and `ffmpeg`.
`ns-process-data` checks for **both** unconditionally, in
`ColmapConverterToNerfstudioDataset.__post_init__`, at argument-parse time. That
check runs **even when the pipeline passes `--skip-colmap`**, which it always
does, and it calls `sys.exit(1)` if `colmap -h` returns nonzero. So a COLMAP
binary is required even for a run that never asks COLMAP to do anything.

## 1. Which GPUs work

The pipeline compiles no **CUDA** code, which is what makes it install in
minutes rather than needing a CUDA toolkit. The cost of that is a **closed**
list of supported GPUs: the device code is whatever the prebuilt gsplat wheel
already contains, and it cannot be extended at install time. (It does compile a
little host C, see "You do need a host compiler"; that is unrelated to which
GPUs work.)

The installed wheel, `gsplat 1.4.0+pt24cu121`, embeds compiled code for
`sm_70`, `sm_75`, `sm_80`, `sm_86` and `sm_90`, **and no PTX**. PTX is the
intermediate form a driver can just-in-time compile for a GPU it was not built
for; without it there is no forward compatibility whatsoever.

Applying CUDA's binary-compatibility rule (code built for `sm_X.y` runs on any
device `sm_X.z` where `z >= y`, and never across a major version):

| GPU family | Compute | Works |
|---|---|---|
| Maxwell (GTX 900 series) | 5.x | **No** |
| Pascal (GTX 10 series, P100) | 6.x | **No** |
| Volta (V100, Titan V) | 7.0 | Yes |
| Turing (RTX 20 series, GTX 16 series, T4) | 7.5 | Yes, this is the development GPU |
| Ampere (A100) | 8.0 | Yes |
| Ampere (RTX 30 series, A10, A40) | 8.6 | Yes |
| Ada Lovelace (RTX 40 series, L4, L40) | 8.9 | Yes, via the `sm_86` code |
| Hopper (H100, H200) | 9.0 | Yes |
| Blackwell datacenter (B200) | 10.0 | **No** |
| Blackwell (RTX 50 series) | 12.0 | **No** |

`run_pipeline.py` checks this at startup and refuses to begin on an unsupported
GPU, naming the reason. It reads the architecture list out of the installed
gsplat binary rather than from a hardcoded table, so the check stays honest if
the wheel is ever changed. Without that check the failure surfaces deep inside
training, after COLMAP and `ns-process-data` have already spent about ten
minutes.

Note this is a limitation of the *prebuilt wheel*, not of the pipeline or of
gsplat itself, and no configuration setting can work around it. Supporting
newer hardware means finding or building a gsplat wheel that includes those
architectures, which reintroduces a compiler.

**This is a known gap with a committed fix, not an accepted permanent limit.**
Blackwell support is targeted for the beta and is a hard gate on the 1.0
release: 1.0 will not be cut while current-generation consumer GPUs cannot run
the pipeline. It is out of scope only for this alpha, whose purpose is to prove
the install works at all on one platform.

## 2. Get the code

```bash
git clone https://github.com/alex-feldman/jsps-dt4ag.git
cd jsps-dt4ag
```

**A plain clone is now enough.** The pipeline was merged into the default
branch on 2026-08-13, so `main` carries both the code and the project's demo web
page. Instructions written before that date tell you to clone
`-b pipeline-alpha`; that branch still exists and still works, but the flag is
no longer needed.

**Where you clone to does not matter.** Any directory you can write to is fine.
But every command from here on assumes you are **at the repository root**, the
directory containing `pyproject.toml` and `uv.lock`. Check with:

```bash
ls pyproject.toml uv.lock pipeline/run_pipeline.py
```

If that errors, you are in the wrong directory. Nothing below will work until
it succeeds.

## 3. Set up the environment

**`uv` is the environment. There is one supported path and this is it.** A conda
environment (`ns-l-oci`) also exists and still works, but it is a **fallback**
for the case where the uv environment misbehaves, it is not reproducible from
this repository, and a first-time install should ignore it entirely. It is
documented at the end of this section so the older route is not lost, not
because it is an equal choice.

### The uv environment (this is the one)

`pyproject.toml` and `uv.lock` at the repository root pin the whole Python side:
Python 3.10, torch 2.4.1+cu121, gsplat 1.4.0+pt24cu121 (**prebuilt**, not
compiled), nerfstudio 1.1.5, numpy 1.26.4.

```bash
# from the repository root
uv sync --frozen              # ~250 packages, 7.4 GB venv, a few minutes

export PATH="$HOME/opt/colmap-prefix/bin:$PATH"

uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

That one `PATH` line is the whole COLMAP and ffmpeg setup; see "COLMAP" below
for how the prefix is built. Do **not** also put the prefix's `lib/` on
`LD_LIBRARY_PATH`: nothing needs it, and "Why the prefix must not go on
LD_LIBRARY_PATH" below explains what it breaks.

Why torch 2.4 and not 2.5: 2.4 is the newest torch that gsplat publishes a
prebuilt sm_75 wheel for. On torch 2.5 gsplat compiles itself from source on
first use, which needs nvcc plus a gcc no newer than 12, and cost roughly ten
minutes at the start of every training run. That is all gone. Full rationale,
verification numbers and revert instructions:
`knowledge-base/pm/kibanc-dt/artifacts/2026-08-08_torch24-migration-record.md`
in the notes repository.

### COLMAP

COLMAP is a C++ binary. It has to be installed separately, whatever you use for
Python packages.

**Use the conda-forge CUDA build, fetched with `micromamba` into a standalone
prefix.** Do not use `apt install colmap` (Ubuntu 24.04 ships 3.9.1, too old)
and do not build from source (hours, and it needs a full CUDA toolkit). This is
the supported route for the alpha:

```bash
# 1. micromamba: one static binary. Not a conda distribution: no base
#    environment, no activate.d, nothing on your PATH afterwards.
mkdir -p ~/opt/bin
curl -fLs https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
     -o ~/opt/bin/micromamba
chmod +x ~/opt/bin/micromamba

# 2. COLMAP 3.12.0 CUDA build, plus ffmpeg, into one standalone prefix.
#    ~2 GB to download, a few minutes. Budget ~6 GB on disk, not 4: the
#    prefix itself is 4.1 GB and micromamba leaves another 1.9 GB of
#    package cache in MAMBA_ROOT_PREFIX that nothing cleans up.
export MAMBA_ROOT_PREFIX=~/opt/mamba-root
~/opt/bin/micromamba create -y -p ~/opt/colmap-prefix -c conda-forge \
    colmap=3.12.0=cuda_126h825ca31_0 ffmpeg

# 3. Put that prefix's bin/ on PATH. This is the only configuration step,
#    and it covers ffmpeg as well as colmap.
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"
```

Do **not** fetch micromamba from `micro.mamba.pm`: that endpoint serves a
`.tar.bz2`, and a minimal Ubuntu image has no `bzip2`, so `tar` fails. The
GitHub URL above is the bare binary.

That `curl` is silent while it pulls about 18 MB, so on a slow link it looks
like a hung terminal for a few minutes. It is not.

Confirm it is the CUDA build. Run bare `colmap` and read line 2:

```bash
colmap 2>&1 | head -2
# COLMAP 3.12.0 -- Structure-from-Motion and Multi-View Stereo
# (Commit Unknown on Unknown with CUDA)
```

`with CUDA` is the part that matters. A `without CUDA` build will run and will
be uselessly slow.

The bare invocation is not an oversight: `colmap --version` **does not work** in
3.12.

**`PATH` is all this build needs.** Its binary carries an `$ORIGIN`-relative
RPATH (`readelf -d` shows `RPATH [$ORIGIN/../lib]`), so it resolves every one of
its libraries from its own `lib/`. Verified 2026-08-11 on a clean container with
`LD_LIBRARY_PATH` explicitly unset: zero unresolved libraries, and a bare
`colmap` prints the `with CUDA` banner. A COLMAP built by hand into a conda
environment, as the development machine's was, has no RPATH and does need
`LD_LIBRARY_PATH`; that is the only reason the older instructions asked for it.

### Why the prefix must not go on LD_LIBRARY_PATH

Nothing in this quickstart tells you to, and this is why. The conda-forge
`cuda_126` build carries **CUDA 12.9.79**'s `libcudart.so.12` in its `lib/`
(`cuda_126` is the build target, not the shipped runtime version), while torch
bundles its own CUDA 12.1 runtime. Export the prefix's `lib/` globally and
torch's loader picks up the COLMAP copy. `PATH` alone cannot do that, which is
what makes the single `PATH` export safe. Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.4.1+cu121 12.1 True
```

### ffmpeg

`ns-process-data` shells out to ffmpeg even for an image dataset, and it also
refuses to start without it (see the check described in section 0). Either
route works:

- `sudo apt install ffmpeg`, if you have root and want it system-wide, or
- let the COLMAP recipe above install it into the same prefix, which it does by
  default.

For the second route the `PATH` export from the recipe already resolves it: no
extra configuration. Verified on the clean machine with conda-forge ffmpeg 9.0
and no system ffmpeg at all.

### The conda environment `ns-l-oci` (fallback only, skip on a first install)

**Do not use this to install the pipeline.** It exists on the development
machine only, it cannot be recreated from this repository, and following it on a
new machine is the single most common way to end up somewhere this document
cannot help you.

It is the environment that predates the uv migration: Python 3.10, torch
2.5.0+cu121, nerfstudio 1.1.5, and gsplat 1.4.0 **compiled by JIT** rather than
prebuilt. It is deliberately left intact and unmodified, and it still works:

```bash
conda activate ns-l-oci
python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

Reach for it in exactly one situation: the uv environment misbehaves and you
need to know whether the pipeline itself or the environment is at fault. Three
things it costs you, all of which the uv path removed:

- **It is not reproducible.** No lockfile, and it depends on a `gcc-10` that
  happens to be installed on that machine.
- **gsplat compiles itself on first use**, roughly ten minutes at the start of a
  training run, and CUDA 12.1 refuses any host compiler newer than GCC 12.
- **Its COLMAP was built by hand into the conda prefix** and has no RPATH, so it
  needs `LD_LIBRARY_PATH` where the supported route needs only `PATH`.

That is why it is the fallback and `uv` is the path.

## 4. Make a config

Everything the pipeline needs lives in one INI file. Nothing is hardcoded.

```bash
cp pipeline/configs/example.ini pipeline/configs/my-run.ini
```

### The keys a first run must set

These five are the complete set for a fresh machine. Every other key in
`example.ini` has a working default, and `configs/README.md` documents them all.

```ini
[paths]
data_root = /path/to/your/data        # 1. everything else hangs off this

[dataset]
images_subpath = scene-01/images      # 2. see "where this path is rooted" below

[train]
max_num_iterations = 30000            # 3. NOT the default. See "How long" below
quit_on_train_completion = true       # 4. NOT the default. See below, this one bites

[paths]
exports_dirname = exports             # 5. where finished .ply files collect
```

**Two of those five ship with the wrong value for a command-line run**, so
copying `example.ini` and editing only the paths is not enough:

- **`max_num_iterations`** is 30000 in `example.ini`, which is what you want,
  but confirm it: a run of 500 or fewer produces the raw COLMAP seed cloud
  rather than a reconstruction. See "How long" in section 6.
- **`quit_on_train_completion` defaults to `false`**, which leaves `ns-train`'s
  viewer running forever after training finishes and deadlocks any unattended
  run. **Set it to `true`.** The runner refuses to start the train stage
  otherwise rather than letting you discover this an hour in.

**Where `images_subpath` is rooted, which matters if your data is laid out
differently from the development machine.** It is relative to
`<data_root>/<datasets_dirname>`, **not** to `data_root` itself, and
`datasets_dirname` is a key too:

```ini
[paths]
data_root        = /media/you/drive/DT-data
datasets_dirname = datasets           # default; change it if yours differs

[dataset]
images_subpath   = scene-01/images    # resolves to
                                      # /media/you/drive/DT-data/datasets/scene-01/images
```

Any depth works, including a single level. If your images do not sit under a
`datasets/` directory at all, set `datasets_dirname` to the directory they *do*
sit under, or set it to `.` and put the whole relative path in `images_subpath`.
Three sibling keys work the same way and are worth knowing about before the
first run writes anything: `colmap_dirname`, `outputs_dirname` and
`exports_dirname`, which name the working directories the pipeline creates under
`data_root`.

### Where to get images

**A sample dataset is published, and it is the right thing to test an install
against**, because it is the exact data that produced this project's reference
reconstructions:

[jsps-dt4ag alpha sample dataset](https://drive.google.com/file/d/1Co9RLorlKGWBHSfN6WihAmzLryW_W2Ji/view?usp=sharing)
(2.2 GB, 120 masked images, five cameras)

Unzip it, then set **two** keys in your config:

```ini
[paths]
data_root      = /full/path/to/jsps-dt4ag-alpha-sample   # the dir CONTAINING datasets
[dataset]
images_subpath = masked-images/test_251128
```

`data_root` is the directory that **contains** `datasets`, not `datasets`
itself: the image path is built as `data_root / datasets_dirname /
images_subpath`, so those levels come from the other two keys. And
`example.ini` ships a neutral placeholder for `images_subpath`, so that one
must change too. If you get either wrong, the config validator in step 3 names
the exact key and the path it resolved to, in about a second.

On a supported GPU expect roughly 1,300 to 1,360 gaussians at **PSNR ~46.5**,
in about 20 minutes. The zip's `README.txt` repeats all of this.

**You need `unzip`.** A minimal container has neither it nor a system Python.
`scripts/install.sh` installs it; if you installed by hand,
`sudo apt install unzip`, or use the environment you just built:
`uv run --frozen python -m zipfile -e <zip> <destination>`.

**Your own images work too**, with two caveats. There is no reference value to
compare against, so a different PSNR means different data rather than a broken
install. And read section 7 first: the pipeline starts at **masked** images, so
raw photos reconstruct the background along with the subject, which is a
supported thing to do but gives you far more gaussians and something to crop
afterwards.

Two more keys are worth setting even though nothing fails without them, because
they are baked into the export filename and are how a `.ply` records what
produced it:

```ini
[export]
env_label      = uv                   # e.g. uv, conda
platform_label = ubuntu2404           # e.g. ubuntu2404, ubuntu2404-container
```

Real config files are gitignored, because they carry real paths and dataset
names. Only `example.ini` is committed, and it stays neutral.

Validate a config without running anything. It resolves the whole path tree and
fails if the data root, the datasets directory or the image directory does not
exist, which is much cheaper than finding out during stage 1:

```bash
uv run python pipeline/dt4ag_config.py pipeline/configs/my-run.ini
```

Use `uv run`, not a bare `python`: on a clean machine there is no system Python
at all, because uv downloaded its own.

## 5. Run it

From the repository root:

```bash
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini
```

The `PATH` export is needed in **every new shell**, because it is what puts
`colmap` and `ffmpeg` where `ns-process-data` looks for them. Nothing else has
to be activated: `uv run` supplies the whole Python environment.

That runs four stages in order: COLMAP structure-from-motion, `ns-process-data`,
`ns-train splatfacto`, and `ns-export gaussian-splat`. It exits non-zero if any
stage fails, and it verifies the output file rather than trusting a success
message.

**Do a dry run first.** It resolves the config, derives the run id, counts your
input images and prints every command it would execute, in about a second:

```bash
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini --dry-run
```

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

## 6. Check it actually worked

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
| `ns-train` @ 30,000 | ~10 min |
| `ns-export` | ~10 s |

Measured stage by stage on a clean `ubuntu:24.04` container, 2026-08-11, same
120-image dataset: COLMAP 5m49s, `ns-process-data` 2m55s, `ns-train` @ 30,000
10m21s, `ns-export` 10s. A second clean run on 2026-08-13 through
`scripts/install.sh` agreed: 6m00s, 3m00s, 10m42s, 10s.

**Budget separately, and generously, for `uv sync --frozen` the first time.**
249 packages and 7.4 GB. It has taken **4 hours 43 minutes** on a link where
`files.pythonhosted.org` throttled to 150-425 KB/s while
`download.pytorch.org` served 9-11 MB/s to the same machine. On a healthy link
it is minutes. Both extremes are normal; the variable is the PyPI CDN, not this
project.

**It gives no progress output for the whole duration**, so a slow link is
indistinguishable from a hang by watching the log. To tell them apart, check
that the download cache is still growing:

```bash
du -sb "${UV_CACHE_DIR:-$HOME/.cache/uv}"     # run twice, a minute apart
```

Growing means alive.

## 7. Where this pipeline starts, and what it does NOT do

**The pipeline begins at MASKED images. It does not create masks.**

Its four stages are `colmap`, `process`, `train`, `export`. There is no masking
stage, and neither the runner nor the notebook invokes any masking script. If you
point it at raw photos it will happily reconstruct them, background and all.

The full capture-to-splat workflow is six steps. This repository automates the
last four:

```
  1. capture photos                         you
  2. generate masks (SAM3)                  samask, a SEPARATE repository
  3. apply masks as an alpha channel        pipeline/scripts/rgb-mask/   MANUAL
  ----------------------------------------- the pipeline starts here ---
  4. COLMAP structure-from-motion           run_pipeline.py
  5. ns-process-data                        run_pipeline.py
  6. ns-train splatfacto + ns-export        run_pipeline.py
```

Steps 2 and 3 are run by hand. `pipeline/scripts/rgb-mask/rgb-mask-batch.py`
applies existing masks; read its README first. Mask *generation* needs the
`samask` repository, which at the time of writing is **not self-contained**: a
fresh clone cannot run it.

If you already have masked images, point `[dataset] images_subpath` at them and
ignore all of the above.

Closing this gap so the pipeline covers photos to splat is tracked, and it is
blocked on samask becoming installable.

### A note on background removal

You do not need to remove the background by hand after export. When the input is
masked, training drives the background gaussians to zero opacity and the export
drops them. Expect a small gaussian count on a small subject: about 1,300 on the
development dataset. **More gaussians means worse, not better**, see section 6.

## Troubleshooting

**`this GPU is sm_120 (Blackwell (RTX 50 series)), which the installed gsplat cannot run`**
Not a configuration problem and not fixable by one. The prebuilt gsplat wheel
contains no code for your GPU and no PTX to JIT from. See "Which GPUs work"
above. The pipeline stops before doing any work, which is deliberate.

**`colmap: error while loading shared libraries: ...`**
This COLMAP does not resolve its own libraries, so it was not installed by the
recipe in section 3. The conda-forge build there carries an `$ORIGIN`-relative
RPATH and needs nothing but `PATH`. Install it that way rather than reaching for
`LD_LIBRARY_PATH`, which breaks torch (see "Why the prefix must not go on
LD_LIBRARY_PATH").

**`error: [Errno 2] No such file or directory: 'cc'` during `uv sync`**
**or `CMake Error: ... Could not find the compiler ... CXX: c++`**
No compiler. `sudo apt install build-essential` and run `uv sync --frozen`
again. `pyliblzfse` and `fpsample` have no Linux wheels and are always built
from source. See "You do need a host compiler" in section 0.

**`RuntimeError: Failed to find C compiler` about half a minute into `ns-train`**
Same cause, different moment. `splatfacto` runs `torch.compile` on
`get_viewmat`, and triton needs the host compiler to build its driver module.
`sudo apt install build-essential`. It is built once and cached in
`~/.triton/cache/`. This is not gsplat: check `TORCH_EXTENSIONS_DIR`, it will be
empty.

**`OSError: libX11.so.6: cannot open shared object file` at the start of stage 2**
`open3d` needs X11 and OpenGL runtime libraries that a minimal Linux image does
not ship. `sudo apt install libx11-6 libgl1 libgomp1`. See "System libraries".

**`Cannot load library .../libio_x3d.so: (libOpenGL.so.0: ...)` during export**
Harmless. It is a `pymeshlab` plugin the export does not use. The `.ply` is
still written and still correct. `sudo apt install libopengl0` silences it.

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
A Jupyter kernelspec activates nothing and inherits only the environment it is
given, so the COLMAP prefix has to be written into the kernelspec's own `PATH`.
See `pipeline/kernels/README.md`. Or use `run_pipeline.py` instead, which is the
supported path.

**No images found, but the directory clearly has images**
Images may be nested one directory per camera. The check is recursive; if this
still happens, verify `images_subpath` resolves by running the config validator
in step 1.

## The notebook

`notebooks/nerfstudio-pipeline-06.ipynb` runs the same stages interactively and
reads the same config file. **It now runs on the SAME uv environment as the CLI**,
via the `dt4ag-uv` kernel, so there is one dependency set and no torch-version
split between the two paths.

Install the kernel from `pipeline/kernels/kernel-uv.json.example`; see that
directory's README. The kernel carries the COLMAP prefix's `bin/` on its own
`PATH`, because a Jupyter kernelspec does not activate anything and inherits
only what it is given. **The notebook path has not been run on a clean machine**,
unlike the command line, so treat the kernelspec as unverified.

**For anything repeatable, prefer `run_pipeline.py`.** It is diffable, testable,
runs headless, exits non-zero on failure, and verifies its own output. The
notebook is kept for exploration, and retiring it is the intended direction.

## Tests

```bash
python -m unittest discover -s pipeline/tests -v
```

They cover config loading and the runner's own logic. They deliberately do not
need a GPU, COLMAP, or any real data, so they prove nothing about a real
reconstruction.
