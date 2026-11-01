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

**Tested 2026-08-28 and it works**, on Ubuntu 24.04 under WSL2 with an RTX 4050
Laptop. The pipeline itself needed no change. What follows is everything the run
turned out to need that a plain Linux install does not.

The driver rule is the **opposite** of the one above, and getting it wrong is
the classic WSL mistake:

1. **On Windows**, install the normal NVIDIA Windows driver. That is what gives
   WSL the GPU.
2. In PowerShell, install the WSL feature itself if you never have:
   `wsl --install --no-distribution`, **then reboot**. The reboot belongs to this
   step, which enables the optional Windows components. Installing a distro onto
   a WSL2 that already works needs no reboot.
3. **Choose the drive now, because you cannot choose it later.**
   `wsl --install -d Ubuntu-24.04` puts the distro on **C:**, and the install
   plus the COLMAP prefix is about 14 GB before any data. If C: is tight, place
   it elsewhere (WSL 2.4.4 and later):

   ```powershell
   wsl --install Ubuntu-24.04 --location D:\wsl\ubuntu-24.04 --name Ubuntu-24.04 --no-launch
   ```

   **Then watch free space on that Windows drive, not inside WSL.** The distro
   lives in one virtual disk file, `ext4.vhdx`, which grows as you write and by
   default does not shrink when you delete files inside WSL. `df` inside WSL
   reports the virtual disk's ceiling, not what the Windows drive has left. If
   the drive fills during a run, the stage dies with no message, and WSL may
   then refuse to start (`Wsl/Service/CreateInstance/E_FAIL`) or `bash` fails
   with an I/O error. Before a run, check the drive in PowerShell against the
   run's expected outputs (a finished install plus one 120-image capture measured
   23 GB of `ext4.vhdx`, see "COLMAP's GPU matcher on WSL2" below) plus any swap
   growth:

   ```powershell
   "{0:N1} GB free" -f ((Get-PSDrive D).Free/1GB)
   ```

   During a run, `wsl-guard.py --drive D` watches it and stops the run before
   the drive is full (see "The backstop: the guard" below). If WSL will not
   start, check that drive first.

4. **Limit COLMAP's threads before the first run; you do not need to raise
   WSL's memory cap.** **WSL2 gives the guest 50 percent of host RAM by
   default**: 8 GB on a 16 GB machine. The whole pipeline fits in that, but only
   if COLMAP runs on a few threads. By default COLMAP extracts features on one
   thread per CPU core, and every thread holds a full-size photograph. On 120
   images at 18 MP, 24 threads filled 8 GB within 16 seconds, and 4 threads
   peaked at 3.8 GB (see "COLMAP threads and memory" below). Running out is nasty
   to diagnose: the Linux OOM killer takes COLMAP mid `automatic_reconstructor`,
   the pipeline prints nothing, the process simply disappears, and the only
   evidence is in `dmesg`. It is invisible from inside Linux, where `free -h`
   just reports a small machine. Put this in your run config:

   ```ini
   [colmap]
   extra_args = --gpu_index 0 --num_threads 4
   ```

   (`--gpu_index 0` is the WSL2 matcher fix described below.)

   **Leave the guest memory cap at its default, or cap it at 8 GB on a larger
   machine.** A higher cap gains this pipeline nothing, and lets WSL's file
   cache take RAM that Windows needs (see "Memory: WSL can empty Windows' RAM"
   below). To set it, create `%USERPROFILE%\.wslconfig`:

   ```ini
   [wsl2]
   memory=8GB
   ```

   Then `wsl --shutdown` and start the distro again. If C: is short of space,
   `swapfile=D:\\wsl\\swap.vhdx` in the same section moves WSL's swap file to
   another drive.

   An earlier version of this guide recommended `memory=22GB`. That was sized
   for COLMAP at its default thread count, which peaked near 13 GB on the
   capture above. With the thread limit it is no longer needed.

5. **Inside WSL, do NOT install any Linux NVIDIA driver.** No
   `apt install nvidia-driver-*`, no `ubuntu-drivers install`. The GPU arrives
   through `/usr/lib/wsl/lib` from the Windows side, and installing a Linux
   driver over it is how people break a working setup.
6. Inside WSL, check:

```bash
nvidia-smi
```

If that prints your GPU, continue with this document from section 0 exactly as
written. A WSL Ubuntu is minimal in the same way a container is, so it will need
the `apt` packages in section 0 rather than already having them, and the `apt`
step **will** prompt for a sudo password. That matters only if you are scripting
an unattended install.

**Keep your images on the WSL filesystem, not on `/mnt/c` or `/mnt/d`.** Those
cross a 9p bridge slow enough to distort every timing in this document.

### COLMAP's GPU matcher on WSL2

**Set `[colmap] extra_args = --gpu_index 0` in your run config.** Without it,
COLMAP's feature *extraction* succeeds and its *matcher* then refuses:

```
E sift.cc:1217] Not enough GPU memory to match 14327 features. Reduce the maximum number of matches.
E feature_matching_utils.cc:82] Failed to create feature matcher.
W incremental_pipeline.cc:276] No images with matches found in the database
```

**And `automatic_reconstructor` still exits 0.** It writes an empty database and
reports success, so anything that trusts the exit code proceeds into nothing.
`run_pipeline.py` catches it, because it verifies the sparse model exists rather
than trusting the return code, and stops with "colmap reported success but
produced no sparse model". Driving `colmap` by hand gets you silence.

The message is misleading: this is not a machine that is short of VRAM. It was
refused on an RTX 4050 Laptop with **5,920 MiB free and 0 MiB in use**, and then
succeeded on the same images at the same free VRAM with nothing else changed but
the flag. COLMAP's default is `--gpu_index -1`, meaning "use every GPU", and
under WSL2's paravirtualised GPU that enumeration makes its available-memory
check misread the device. Naming the index explicitly makes it query the real
one.

Measured 2026-09-05 on Ubuntu 24.04 under WSL2: the failure reproduced twice on
fresh databases, and the fix reproduced. It does not depend on login versus
non-login shells, which was an earlier and wrong diagnosis of the same symptom.
If you have more than one CUDA GPU, name the one you want rather than assuming 0.

**One cosmetic wart.** `ns-export` prints roughly twenty multi-line Qt plugin
errors, because `pymeshlab` cannot load `libOpenGL.so.0`. The export succeeds
anyway. `sudo apt install libopengl0` silences it.

Budget disk space on whichever drive holds the distro: the environment alone is
about 7.4 GB and the COLMAP prefix another 4 GB, before any data. A finished
install plus one 120-image capture measured 23 GB of `ext4.vhdx`.

### COLMAP threads and memory

**COLMAP's thread count, not the number of photographs, decides how much
memory the camera solve needs.** `automatic_reconstructor` runs at
`--quality high` by default. That setting turns on affine-shape estimation,
which runs feature extraction on the CPU ("Covariant SIFT") even when a GPU is
present. It runs one extractor per thread, and each extractor holds a decoded
full-size photograph plus its feature structures. The default thread count is
every CPU core. So memory grows with cores times image size, which is why a
many-core machine with modest RAM is the worst case.

`--num_threads N` in `[colmap] extra_args` sets the count. Measured 2026-11-01
on Ubuntu 24.04 under WSL2 (24 hardware threads, RTX 4050 Laptop), COLMAP stage
only, 120 unmasked photographs at 5184x3456, inside a hard 8 GiB memory plus
4 GiB swap limit (`wsl-guard.py --emulate-vm 8G:4G`, the default VM on a 16 GB
Windows machine):

| `--num_threads` | Program memory, peak | Swap | Result | COLMAP time |
|---|---|---|---|---|
| default (24) | over 8 GiB within 16 s | 1.5 GiB and climbing | did not fit (stopped) | n/a |
| 8 | 6.0 GiB after 22 s and still rising | none yet | not finished, see below | n/a |
| 4 | 3.7 GiB | none | fits | 9m24s |
| 2 | 1.7 GiB | none | fits | 14m02s |

"Program memory" is the scope's anonymous memory (`scope_anon_max_mib` in the
guard's status file). The file cache fills the rest of the limit and is
evicted, so it is not counted. The 8-thread run was stopped by `wsl-guard.py`
because Windows itself ran out of RAM, not because it hit the 8 GiB limit, so
its row is a lower bound.

**The cost is close to linear: about 1 GiB per thread for 18 MP photographs,
with almost nothing fixed.** It scales with the photograph's pixel count, so
12 MP photographs need roughly two thirds of that. By that rate, 8 threads
would need about 8 GiB, which leaves no room in an 8 GiB VM.

**Use `--num_threads 4` on a machine with 16 GB of RAM.** It fits the worst
capture measured with about 4 GiB to spare. Halving it to 2 saves another
2 GiB but made the camera solve about half again as slow. On a machine with
more memory available to Linux, allow about 1 GiB per thread for 18 MP
photographs plus a few GiB for everything else, and stay below the core count.

Changing `--quality` would also move extraction to the GPU, but it changes the
features themselves. Treat that as a separate experiment, not a memory setting.

### Which COLMAP options `extra_args` accepts

`[colmap] extra_args` is appended verbatim to one `colmap
automatic_reconstructor` command (`colmap_command` in `run_pipeline.py`), and
the installed COLMAP is 3.12.0. Two consequences:

- **Use 3.12.0's option names, not the website's.** COLMAP's online
  documentation follows the newest release, whose names (such as
  `--FeatureExtraction.use_gpu`) do not exist in 3.12.0. Check what your binary
  accepts with `colmap automatic_reconstructor -h`.
- **`automatic_reconstructor` takes only its own coarse options**, among them
  `--quality`, `--camera_model`, `--num_threads`, `--use_gpu` and `--gpu_index`.
  It rejects the per-step options that `colmap feature_extractor -h` lists, for
  example `--SiftExtraction.use_gpu` ("unrecognised option"). Per-step control
  needs separate `feature_extractor`, matcher and `mapper` commands, which the
  pipeline does not run.

`--use_gpu 0` is the CPU fallback for COLMAP's extraction and matching.
`--gpu_index` only chooses which CUDA device COLMAP uses. Neither changes
training or export, which always need a supported CUDA GPU (section 1).
Checked 2026-11-01 against COLMAP 3.12.0's `-h` output.

### Memory: WSL can empty Windows' RAM

**The mechanism.** WSL2's Linux VM takes Windows RAM as Linux asks for it, up to
the `.wslconfig` `memory=` limit, and Linux uses every free page as file cache:
every image a stage reads or writes stays cached. Windows counts that cache as
used by the VM (`vmmemWSL` in Task Manager), and WSL does not give it back while
work keeps running. Its `autoMemoryReclaim` setting only acts after about five
idle minutes. Inside WSL nothing looks wrong: `free -m` shows little `used` and a
large `buff/cache`.

Measured 2026-10-20 on a 32 GB Windows 11 laptop with `memory=22GB`:
`ns-process-data` copying and downscaling a 55-frame 3840x2880 capture took
Windows' available RAM from 6.5 GiB to 59 MiB in about 25 seconds. **Dropping the
cache before the run does not help**: the run refills it. A run also leaves its
cache behind (9.6 GiB after one job, with 841 MiB actually in use inside WSL),
which then starves the next stage.

**The fix: cap each stage in a cgroup.** Set this in the run config:

```ini
[host]
memory_high = 8G
```

Each stage command then runs inside
`systemd-run --user --scope --quiet -p MemoryHigh=8G ...`. The cache the stage
creates is charged to that scope, and at the cap the kernel evicts the scope's
own cache instead of growing the VM. On the capture above, a full camera solve
under an 8G cap kept Windows above 2.7 GiB available. MemoryHigh throttles and
never kills. It needs cgroup v2 and systemd as PID 1 (if `ps -p 1 -o comm=` does
not print `systemd`, set `systemd=true` under `[boot]` in `/etc/wsl.conf` and
restart the distro); the runner
checks the scope works before the first stage and refuses to run uncapped if it
does not. `--dry-run` shows the wrapped commands. To check by hand:

```bash
systemd-run --user --scope --quiet -p MemoryHigh=64M true && echo scope works
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.subtree_control   # must list memory
```

`ns-process-data` also runs ffmpeg's downscale on every core, which fills the
cache fastest. `[host] process_cpus = 0,1` pins that stage alone to two CPUs
(`taskset -c 0,1`); that was enough on its own for a 28-frame capture, not for 55.
`configs/README.md` documents both keys.

**Choosing a value.** Start from Windows' available RAM before the run (Task
Manager, or `wsl-guard.py`'s first sample) and leave about 3 GiB of it to
Windows. Too high and the VM can still empty Windows; too low and the stage is
throttled, spending its time evicting and re-reading cache. A camera solve that
stalls for minutes with no log output is the sign; raise the cap. Measured on
the laptop above: `8G` for a 55-frame camera solve, and a 55-frame training run
under `10G` kept Windows above 4.0 GiB available; of that training run's 9 GiB in
the cap, only about 2 GiB was the program's own memory, the rest file cache. One value applies to every
stage, so pick the larger of what the stages need. If Windows' available RAM is
already low before the run, release leftover cache first (below); the cap limits
what the run adds, not what an earlier run left.

**Checking it works.** The log shows each stage command starting with
`systemd-run --user --scope --quiet -p MemoryHigh=`. During a stage, `free -m` in
another WSL terminal shows `buff/cache` levelling off near the cap instead of
climbing, and Windows' available RAM (Task Manager, or the guard's host-sample
JSONL) stays roughly flat through `ns-process-data`, where it used to collapse.

**Windows machines with 16 GB of RAM.** WSL's default VM limit is half of
Windows' RAM, so on a 16 GB machine the VM gets 8 GB and Windows keeps the
rest. The whole pipeline fits in that. Measured 2026-10-20 by emulation on the
laptop above: colmap, process, train and export on the 55-frame 3840x2880
capture (`downscale_factor` 4, `[host] process_cpus = 0,1`), run inside a hard
`MemoryMax=8G` with `MemorySwapMax=4G`, finished in 27 minutes with no
out-of-memory kills and no swap used. The programs' own memory peaked at
2.4 GiB; the rest of the 8 GiB was file cache, which the kernel evicted at the
limit. The heaviest camera solve measured so far also fits: 120 photographs at
5184x3456 with no masks peaked at 3.7 GiB of program memory with
`--num_threads 4`, but overflowed 8 GiB within 16 seconds at COLMAP's default
thread count (2026-11-01, same emulation; see "COLMAP threads and memory"). On
such a machine keep the `.wslconfig` default (or `memory=8GB`) and set
`--num_threads` as in step 4 of "WSL2, on Windows". A `[host] memory_high` below
the VM limit, such as `6G`, is optional and only throttles earlier. Not yet
tested on real 16 GB hardware, and the emulation used a 6 GB GPU: on a smaller
GPU, VRAM rather than RAM is the likelier limit.

*Check your machine.* On any machine with systemd in WSL you can test the 8 GB
case before you rely on it, whatever your real RAM. Run one full invocation
under `--emulate-vm`, with a config that has no `[host] memory_high` (the
runner's own scope would start inside the emulated one and escape it, so the
guard refuses that combination):

```bash
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"
uv run python pipeline/scripts/wsl-guard/wsl-guard.py --emulate-vm 8G:4G -- \
    --config my-run.ini --stage colmap,process,train,export
```

Start it with `uv run python`, as in section 5, and the guard runs the pipeline
with the venv's Python. Do not use a bare `python` or `--python "$(which
python)"`: the COLMAP prefix has its own `python`, so after the `export` above
those can be COLMAP's, which has no torch (see "torch could not be imported"
under Troubleshooting). The guard refuses that case before starting.

`--print-command` shows the wrapped command without running anything. **Pass
means exit 0 and `"scope_oom_kills": 0`** in the run's `-status.json` (next to
the log). The same file shows `scope_anon_max_mib`, the program's own peak, and
`scope_swap_max_mib`; a peak well under 8 GiB, with little or no swap, means
there is headroom. The scope's `scope_peak_mib` will read close to the limit
even on a pass, because file cache fills what the program leaves. Exit 137, a
nonzero `scope_oom_kills`, or `"killed_by_sigkill_unprompted": true` means the
run did not fit.

**The backstop: the guard.** `scripts/wsl-guard/wsl-guard.py` runs the pipeline
while watching Windows' available RAM from the Windows side, refuses to start
without headroom, releases the cache when RAM runs low and stops its own run
before Windows runs out:

```bash
uv run python pipeline/scripts/wsl-guard/wsl-guard.py --drive D -- --config my-run.ini --stage colmap,process
```

**Releasing the cache by hand,** after a run or between stages:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

It drops only clean cache, so it is safe while other work runs. **Windows sees
the memory about 40 seconds later** (measured: 4.5 GiB available, then 13.1 GiB),
so wait before judging whether it worked.

**Whole-VM fallback.** A lower `memory=` in `%USERPROFILE%\.wslconfig` caps the
entire VM, cache included, and needs `wsl --shutdown` to take effect. It also caps
COLMAP and training: `8GB` is measured to be enough with COLMAP's threads limited
(step 4 of "WSL2, on Windows"), and lower values are untested. The per-stage cap
is the finer tool.

### Windows' commit limit: the other way WSL dies

**WSL can also be stopped by Windows itself, and this happens with plenty of
"free" RAM showing.** Windows promises memory to programs up to its *commit
limit*, which is RAM plus the page file. Everything that runs counts against
it: browsers and every app built on a browser engine (Chrome and Edge, and
Electron or WebView2 apps such as Slack, Discord, WhatsApp and Zotero) commit
far more than they are using at the moment. When a growing WSL VM needs memory
and the commit limit is reached, Windows stops the whole VM: every WSL process
dies at once, and the event log shows Resource-Exhaustion-Detector event 2004
with `vmmemWSL` at the top. Measured 2026-11-01 on a 32 GB laptop with a 31 GB
page file: about 41 GB was committed by Windows apps (Chrome alone 16.7 GB,
across about 99 processes), and WSL was stopped twice in one afternoon.

Check the headroom before a big run, in PowerShell:

```powershell
"{0:N1} GB free" -f ((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)
```

That number has to exceed what the WSL VM may still grow by, which is at most
its `memory=` limit. Closing a browser or chat apps raises it quickly. A smaller
WSL `memory=` lowers what WSL can claim. A larger Windows page file raises the
limit itself. `wsl-guard.py` checks this for you. It refuses to start with less
than 8 GiB of commit headroom (`--start-commit-mib`), and stops its own run if
headroom falls below 2 GiB (`--stop-commit-mib`), so the run ends with a message
instead of the VM dying.

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

**WSL2 works, tested 2026-08-28.** Ubuntu 24.04 under WSL2 on Windows 11, with an
RTX 4050 Laptop: `scripts/install.sh` completed unaided and the pipeline
reproduced the reference reconstruction at 1,360 gaussians and PSNR 46.532, in
22m37s. WSL2 *is* Linux x86-64, so the wheel platform tag, the conda-forge COLMAP
and the whole uv path applied unchanged. The GPU arrives through the Windows
driver and needed no configuration at all.

**Two things are host-side, and neither is visible from inside Linux.** First,
**WSL2 caps guest RAM at 50 percent of host RAM**, and COLMAP's default thread
count (one per core) can overflow that cap on large photographs, with no message;
limiting the threads fixes it. Second, `wsl --install -d <distro>` puts the
distro on C:. Both are covered in "WSL2, on Windows" under "Pre-steps" above,
and you want to read that section **before** the first run rather than after
diagnosing an OOM.

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

**The prefix also contains a `python`** (`python`, `python3` and `python3.14`,
pulled in as conda-forge dependencies). After the `export`, a bare `python` or
`python3` is COLMAP's, not the pipeline's, and it has no torch or nerfstudio.
That is harmless as long as the pipeline is started with `uv run python ...`,
which puts the venv first, or with the venv's full path
(`.venv/bin/python`, which also needs the venv's `bin/` on `PATH`, see section 5).
It bites when a command uses whatever `python` is first on `PATH`, for example
`$(which python)`. If you activate the venv instead (`source .venv/bin/activate`),
do it after the `export`, so the venv comes first.

**A git worktree has no `.venv` of its own.** `.venv` is untracked, so a second
checkout made with `git worktree add` starts without one, and `uv run` there
builds a new environment of about 7.4 GB. To run the pipeline or the tests from
a worktree, use the main checkout's interpreter with that venv's `bin/` first on
`PATH`:

```bash
export PATH="{main checkout}/.venv/bin:$HOME/opt/colmap-prefix/bin:$HOME/.local/bin:$PATH"
{main checkout}/.venv/bin/python pipeline/run_pipeline.py --config my-run.ini --dry-run
```

This is only safe while the worktree's `pyproject.toml` and `uv.lock` match the
main checkout's. A branch that changes dependencies needs its own `uv sync`.

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

Or let a page write it: `pipeline/scripts/run-builder/` asks for the capture,
the masks and the settings, and prints the config with the dry-run and run
commands (see its README). The keys below are what it fills in.

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

**Your own still photos or video-derived frames work too.** There is no
reference score for your capture, and the pipeline reads image files rather
than a movie. Masks are optional: without them, the reconstruction includes
background from the photos. See section 7 for the video-frame workflow and the
optional mask pre-step.

### What makes a capture reconstructable, and what no setting can fix

**A bad capture cannot be rescued by tuning, and it is worth knowing that before
you spend an afternoon trying.** Measured 2026-09-05 on a deliberately poor
capture: 19 photographs of a car, walked along one side.

| symptom | value | what it means |
|---|---|---|
| Angular coverage | **44.8 deg of 360** | 315 degrees of the subject was never photographed |
| Mean track length | 2.74 | the average 3D point survives fewer than three frames |
| Features becoming points | ~12% | correspondences are failing, not features |

The result was 630,000 gaussians of mostly background and floaters, against
1,360 for the reference capture. Masking removed the background and left a
recognisable subject only on the side that was photographed.

**`--quality extreme` does not help, and was tested rather than assumed.** On the
same images, SfM only: 8,938 points, track length 2.749, reprojection error
1.123 px, against `high`'s 9,120 / 2.738 / 1.079 px. Marginally worse, for 4.5
extra minutes. Feature counts were unchanged at roughly 10,000 per image, which
is the point: **the extractor was never the constraint.** What fails is matching
features across views, and that is a property of the surface and the camera path,
not of any COLMAP setting.

So, for a capture that will actually reconstruct:

1. **Go all the way around the subject.** This matters more than everything else
   combined. Roughly 10 degrees per shot, about 36 frames per ring. Wide steps
   over a full circle beat fine steps over a narrow arc.
2. **Two or three heights**, so the top and the underside edges are seen. 70 to
   110 photographs total for an object the size of a car.
3. **Fill the frame with the subject**, and keep the distance roughly constant.
   Standing far back spends your pixels and your parallax on the background.
4. **Walk, do not pivot.** Rotating on the spot gives no parallax and therefore
   no depth.
5. **Avoid glossy and specular surfaces in direct sun.** Highlights move with the
   viewpoint, so features on a shiny panel do not match between frames. Shade or
   overcast light is worth more than any parameter in this document. Matte,
   textured subjects such as plants are the easy case, which is what this
   pipeline was built for.

**One image shape per camera.** The default `[colmap] single_camera = 1` and
`single_camera_per_folder = 1` give every image directory one shared camera.
If the images in one directory do not all have the same pixel dimensions,
COLMAP rejects them with `CAMERA_SINGLE_DIM_ERROR: Single camera specified, but
images have different dimensions`. Mixed shapes usually mean a phone stored
some photographs rotated: a few landscape files among portrait ones, or the
other way round. Check before masking, because masks inherit the shape. List
each file's stored size and EXIF orientation tag (1 means upright as stored):

```bash
uv run python - /path/to/capture/images <<'EOF'
import sys
from pathlib import Path
from PIL import Image
for p in sorted(Path(sys.argv[1]).iterdir()):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        with Image.open(p) as im:
            print(p.name, im.size, im.getexif().get(274, 1))
EOF
```

(`file *.jpg` also prints each JPEG's size and orientation.) To fix it, keep the
originals in a sibling directory outside `images/`, then make the working copies
upright in pixels:

- **When the EXIF tag describes the intended rotation**, apply Pillow's
  `ImageOps.exif_transpose` to each working copy and save it with orientation 1.
- **When there is no reliable tag**, rotate the pixels of the odd images, not
  the tag, and reset the tag to 1. Choose the direction by looking at one rotated
  image beside one reference image.

Then list the sizes again and look at a contact sheet of the set to confirm
every image is upright. Normalize to the orientation the capture actually has;
do not rotate a whole set only to match an older capture. Splitting the shapes
into separate subdirectories also works, but gives each shape its own camera,
which is worse for a handheld set from one phone.

Lens distortion, camera models and how to spot a bad camera solve:
[`CAMERA-PROFILES.md`](CAMERA-PROFILES.md).

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

**Running this unattended, from a script or an agent, needs two more things.**
Both were found on WSL2 and both apply to any non-interactive shell.

First, `uv` itself is not on `PATH`. The installer puts it in `~/.local/bin` and
relies on your shell rc, which a non-interactive shell does not read, so the
command dies with `uv: command not found`. Export both directories:

```bash
export PATH="$HOME/opt/colmap-prefix/bin:$HOME/.local/bin:$PATH"
```

Second, on WSL2 set `[colmap] extra_args = --gpu_index 0 --num_threads 4` in
your config. `--gpu_index 0` is explained in "COLMAP's GPU matcher on WSL2", and
`--num_threads` in "COLMAP threads and memory" (both in "Pre-steps" above). Neither is
specific to unattended runs, but this is where scripted runs hit them. The
thread limit is worth setting on any machine with many cores and little RAM,
not only under WSL2.

That runs four stages in order: COLMAP structure-from-motion, `ns-process-data`,
`ns-train splatfacto`, and `ns-export gaussian-splat`. It exits non-zero if any
stage fails, and it verifies the output file rather than trusting a success
message.

**Do a dry run first.** It resolves the config, derives the run id, counts your
input images and prints every command it would execute, in about a second:

```bash
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini --dry-run
```

**A dry run leaves no record.** It writes no `run-log.csv` row and no archived
config (`run_pipeline()` skips both while `dry_run` is set), and no catalog
record. It also skips the prerequisite and host-limit checks, does not composite
masks, and does not count training views, so it cannot show that a real run
will pass those gates. Its output exists only on the terminal. To keep it, for
example to review it later or attach it to a question, capture both streams:

```bash
set -o pipefail
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini --dry-run 2>&1 | tee dry-run.log
```

`pipefail` keeps the dry run's own exit status, which `tee` would otherwise
replace. A real run also prints only to the terminal unless catalog emission is
on ([CATALOG-RECORDS.md](CATALOG-RECORDS.md)); capture it the same way, or run
it under `wsl-guard.py`, which writes a log.

Useful flags:

```bash
--dry-run                 # print the exact commands, run nothing
--stage train             # run one stage only
--from-stage train        # resume partway (needs --run-id)
--run-id val_260808-01-3120
```

To run every capture beneath a collection from one config, use
`pipeline/run_collection.py` ([SEQUENTIAL-RUNS.md](SEQUENTIAL-RUNS.md#collection-runs)).

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

**What a finished run leaves on disk**, checked 2026-11-01 against
`run_pipeline.py` and real run directories:

- **The COLMAP workspace**, `colmap/.../{run-id}/`: COLMAP's `database.db` and
  sparse model (`sparse/0/`), then what `ns-process-data` adds: `images/` holding
  copies of the photographs renamed `frame_00001` onward in sorted path order,
  the downscaled pyramid (`images_2/`, `images_4/`, `images_8/`),
  `transforms.json` and `sparse_pc.ply`.
- **The training run**, `outputs/.../{run-id}/splatfacto/{timestamp}/`:
  nerfstudio's `config.yml`, `dataparser_transforms.json`, the checkpoints in
  `nerfstudio_models/`, and one `config-step-{step}.yml` per earlier checkpoint
  the export stage exported (see "Several step counts from one run").
- **The exports** in `{exports_dirname}/`, one `run-log.csv` row, and the
  frozen config in `configs/runs/` (the config plus resolved paths and the
  pipeline's git commit; `configs/README.md`, "Archive your configs").
- **For a masked run**, the composites in `derived/masked/...`, reused by later
  runs of the same capture and mask variant, and safe to delete (`LAYOUT.md`).

One temporary tree is removed. When `image_extensions` or `mask_extensions` is
set and `use_masks` is false, the runner stages the photographs into a sibling
`{run-id}_sfm-input/` beside the workspace (`sfm_input_path`,
`stage_sfm_inputs`) and deletes it after a successful process stage. A run that
stops before then, or an invocation that runs only `--stage colmap`, leaves it
behind; it is safe to delete, and `--from-stage process` rebuilds it.

**No quality metric is logged** with the default `[train] vis = viewer`:
training computes none. Use `ns-eval` (section 6).

### Several step counts from one run

**To compare how many training steps a reconstruction needs, train once to the
largest count and export the checkpoints on the way. Do not train once per
count, and do not try to resume.**

The pipeline keeps a checkpoint every `[train] steps_per_save` steps (default
2500) and keeps all of them (`save_only_latest_checkpoint = false`), plus one at
the final step. nerfstudio's own defaults are every 2000 and newest only, which
deletes the earlier ones and leaves nothing to export an earlier step from.

A checkpoint at step N of a longer run is the same training as a run configured
for N steps, because splatfacto's schedules do not depend on
`max_num_iterations`: the learning-rate decay is fixed at 30,000 steps and
densification stops at step 15,000 either way. (Equivalent, not bit-identical:
nerfstudio pins its seed, `machine.seed = 42`, but two runs of one configuration still
differ slightly. The cause was not isolated; non-deterministic GPU kernels are the
likely one. To size that noise, repeat one configuration and compare, rather than
assuming zero.)

**Have the pipeline export them: set `[export] checkpoint_interval = 2500`.** The
export stage then writes one PLY for every checkpoint whose step is a multiple of
that number, plus the final one, each named `{x}steps_from{y}run` and carrying the
eval-mode token. The default is `15000` (`DEFAULT_CHECKPOINT_INTERVAL` in
`pipeline/dt4ag_config.py`), which exports step 15,000 and the final step of a
30,000-step run. `0` exports the final checkpoint only. It must be a
multiple of `steps_per_save`. Exporting takes about 20 seconds per checkpoint.

To export a single earlier step by hand instead, point a copy of the run's
`config.yml` at it. `ns-export` has no step option of its own, but it honours
`load_step` in the config (this is what the export stage does for each checkpoint):

```bash
RUN=<data_root>/outputs/<dataset>/<run-id>/splatfacto/<timestamp>
ls $RUN/nerfstudio_models                      # step-000002500.ckpt, ...
sed 's/^load_step: .*/load_step: 12500/' $RUN/config.yml > $RUN/config-step-12500.yml
ns-export gaussian-splat --load-config $RUN/config-step-12500.yml \
    --output-dir <data_root>/exports --output-filename <name>
```

When exporting by hand, name the file with the pipeline's own pattern
(`configs/README.md`, "Export filenames"):
`..._<x>steps_from<y>run_<eval token>_...`, where `x` is the checkpoint's step and
`y` the run's `max_num_iterations`, for example
`..._12500steps_from30000run_evalall_ds4_individual.ply`. Checkpoint files are
named by the zero-based step they were saved at, so a run's final checkpoint is one
below its maximum (`step-000029999` for a 30,000-step run); count it as `x = y`.

**Score each checkpoint with `ns-eval` to choose a step count.** Same trick, one
config copy per step:

```bash
sed 's/^load_step: .*/load_step: 10000/' $RUN/config.yml > $RUN/config-step-10000.yml
ns-eval --load-config $RUN/config-step-10000.yml --output-path eval-step10000.json
```

`ns-eval` loads that checkpoint, renders every HELD-OUT image from its own pose
and compares it with the photograph: PSNR, SSIM and LPIPS, mean and standard
deviation over those images (about 11 seconds per checkpoint here). Which images
are held out is `[train] eval_mode` (default `fraction`, `train_split_fraction =
0.9`): the first and last photographs and evenly spaced ones train, **the remainder
are never seen in training**. The split is made from the photographs sorted by
filename and involves no randomness, so it is the same for every run and every
`ns-eval` on one capture. With `eval_mode = all` every photograph trains and
`ns-eval` evaluates on those same photographs, so its score measures fit and not
generalization; never compare it with a held-out score. With 24 photographs the
default is 22 train and 2 evaluate, so the standard
deviation is large and a 0.5 dB difference between two checkpoints is noise.
Read the LPIPS and SSIM trend and the plateau, not the third digit of PSNR.

**Comparing all-images and held-out training on one capture.** Train both on
the same COLMAP workspace, so they share camera poses: COLMAP is not
bit-reproducible, so a second solve would make the two runs incomparable. Run
the first pass with `[train] eval_mode = all`, then set the held-out value
(for example `eval_mode = fraction`) and train again on the first run's
workspace:

```bash
uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini \
    --from-stage train --run-id {the first run's id}
```

COLMAP and `ns-process-data` are not repeated. Each pass gets its own
timestamped nerfstudio run directory, its own `run-log.csv` row and its own
archived config, and its exports carry its own eval token. Confirm the split from
the train stage's log line, not from the config:

```
training set      : evalall: 24 train, 0 held out of 24 (every photograph trains; ns-eval then scores fit, not generalization)
training set      : evalfrac90: 22 train, 2 held out of 24
```

A run that trains with the viewer only (`quit_on_train_completion`, as the
pipeline does by default) computes NO metrics during training; `ns-eval` is the
only source of them. Setting `[train] vis = viewer+tensorboard` changes that:
nerfstudio then writes TensorBoard event files (gaussian count, train PSNR, a
mean eval over the held-out images every 1000 steps) into the run directory,
read with `tensorboard --logdir {run directory}` or `EventAccumulator`. The
total `Train Loss` tag read NaN in testing, and the logged eval is a mean with
no standard deviation, so `ns-eval` is still the per-checkpoint tool. See
`configs/README.md`, "Training-time logging".

Measured 2026-10-05, 24 photographs of one cucumber plant (22 train, 2 held
out), 4x downscale, one 20,000-step run with a checkpoint every 2500 steps:

| step | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 2500 | 17.7 | 0.579 | 0.418 |
| 5000 | 19.1 | 0.659 | 0.337 |
| 7500 | 19.6 | 0.689 | 0.310 |
| 10000 | 19.5 | 0.685 | 0.298 |
| 15000 | 19.6 | 0.680 | 0.292 |
| 19999 | 19.5 | 0.673 | 0.293 |

Everything flattens by step 7500 to 10000, and SSIM drifts down after 10000,
which is consistent with overfitting the 22 training views (two held-out
images cannot confirm it). For this capture 10,000 steps was
enough and 20,000 bought nothing measurable. A separate 10,000-step run scored
20.1 against 19.5 for the 10,000-step checkpoint of the longer run: that 0.6 dB
is how far two identical-setting runs differ.

**Resuming a finished run to train further does not work** with the pinned
nerfstudio 1.1.5 (`ns-train splatfacto ... --load-dir`). The trainer builds its
optimizers before the checkpoint is loaded, and splatfacto's loader then
replaces every gaussian parameter with a new tensor sized to the checkpoint, so
the optimizers still hold the old ones. Resuming after any densification
therefore crashes in gsplat's `duplicate()` at the first densification step
(`device-side assert`, an index out of bounds). Seen 2026-10-05 resuming a
10,000-step run. If you need more steps, retrain to the larger count.

**What a checkpoint costs.** About 740 bytes per gaussian: 300,000 gaussians is
about 220 MB (80 MB of model, 140 MB of optimizer state). Writing one takes
about 0.2 to 0.3 seconds on ext4, so 12 checkpoints in a 30,000-step run add a
few seconds to roughly 10 minutes of training. The cost that matters is disk:
12 checkpoints of a 300,000-gaussian scene is about 2.6 GB per run, and it
scales with the gaussian count, so a scene of a million gaussians is about
740 MB per checkpoint. Delete the checkpoints you no longer need after
exporting, or set `save_only_latest_checkpoint = true` when you will not
compare step counts. Lower `steps_per_save` only for short exploratory runs
where the early steps are the question.

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

**A different capture on a laptop GPU.** Measured 2026-10-05 on an RTX 4050
Laptop (6 GB) under WSL2, with 24 unmasked photographs at 4096x3072 trained at
4x downscale (nerfstudio's automatic choice). These numbers are not comparable
row by row with the table above: fewer, smaller images, a lower training
resolution and 10,000 steps rather than 30,000.

| Stage | Time |
|---|---|
| COLMAP | 1m57s |
| `ns-process-data` | 15 s |
| `ns-train` @ 10,000 | 5m15s |
| `ns-export` | 18 s |

The whole run took 7m46s and exported about 294,000 gaussians. A 20,000-step
run of the same capture took about 13 minutes while other GPU work overlapped
it.

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

**The pipeline starts from an image sequence. Masks are optional. It does not
decode a movie or generate masks.**

Its four stages are `colmap`, `process`, `train`, and `export`. When masks are
enabled, the runner first composites each mask into the corresponding image's
alpha channel. The later stages do not know whether those images were
photographed individually or extracted from a video.

For a video-derived capture, the workflow before the pipeline is:

```
source movie
    -> choose and extract reviewed frames             you, using FFmpeg or another decoder
    -> optionally make and review masks                samask, a SEPARATE repository
    ----------------------------------------------- the image pipeline starts here
    -> optionally composite masks as alpha             run_pipeline.py, pre-step
    -> COLMAP -> ns-process-data -> ns-train -> ns-export
```

Mask compositing was manual until v0.2.0 and is now automatic. Set
`[dataset] use_masks = true` and `mask_extensions` to your masks' extension
(the config is refused if it is empty), put the masks in `{capture}/masks/` mirroring
`{capture}/images/`, and the runner composites them, reusing an existing
complete set rather than rebuilding it. See `LAYOUT.md` for the directory rule
and `MASKING.md` for why alpha compositing rather than a mask file.

To keep more than one mask set for a capture (another prompt, another tool,
leaf against whole plant), put each further set in `<capture>/masks_<variant>/`
with the same shape as `masks/` and select it with `[dataset] mask_variant =
<variant>`. Empty keeps reading `masks/`. Each variant composites into its own
`derived/masked/<variant>/...` tree, so switching variants never reuses the
other one's composites. samask has no variant concept and writes `masks/` by
default, so generate a variant by passing its `--masks <capture>/masks_<variant>`
explicitly, or the pipeline will refuse with "no masks_<variant> directory".

Mask generation remains external. It needs the `samask` repository, which at
the time of writing is **not self-contained**: a fresh clone cannot run it.
Video decoding and frame sampling are also preparation steps outside this
pipeline; the preceding section gives the verified image-sequence route.

If you already have premade masked images, point `[dataset] images_subpath` at
them, leave `use_masks = false`, and ignore all of the above. That is what
v0.1.0 required of everyone and it still works.

### Video-derived image sequences

**The whole video path as one set of copy-paste commands is [`VIDEO-WORKFLOW.md`](VIDEO-WORKFLOW.md).**
This section is the reference for the extraction step it uses.

The supported video workflow is to make a stable image capture first, then run
the normal image pipeline. Store still frames under a new capture's `images/`
directory. Keep the source movie outside `images/`; the pipeline will not decode
a movie or extract frames for this workflow.

Steps 1 to 3 are done by [`pipeline/scripts/video-frames/`](scripts/video-frames/README.md),
which owns the commands, the files it writes and what it refuses. It uses the
FFmpeg and `ffprobe` from section 3, decodes on the CPU, and never launches a
pipeline stage.

1. Review the footage:

   ```bash
   python3 pipeline/scripts/video-frames/video-frames.py review \
       --source /path/to/source.mp4 --review-root /path/to/review
   ```

   This hashes the movie, records its stream metadata, and in one decode writes
   whole-clip thumbnails and a preview of the first 30 seconds
   (`--preview-seconds 60` for a minute), plus an `index.html` that labels every
   tile with its time, lets you pick the interval and prints the `extract`
   command, all outside any capture. The
   overview helps choose a segment and sampling plan; it is not the final
   reconstruction frame set. A long, high-resolution clip takes time to decode.
2. Choose sampling from camera movement, overlap, blur and scene motion.
   Do not extract every decoded frame by default. There is no universal
   frame-rate or overlap threshold that guarantees a successful solve. The
   choice is yours; the tool records it and does not verify it.
3. Extract into a new capture, using the hash from the review. Run it once
   with `--dry-run` first:

   ```bash
   python3 pipeline/scripts/video-frames/video-frames.py extract \
       --source /path/to/source.mp4 --sha256 {hash from the review} \
       --capture /path/to/data/datasets/{new-capture-id} --interval 0.5
   ```

   The capture directory must not exist. Source movies, previews and manifests
   stay out of `images/`. The tool saves the exact command, source hash, stream,
   range, interval, frame count and a frame manifest under `provenance/`. FFmpeg's
   default pixel autorotation is enabled and recorded. The `fps` filter's
   output-grid timestamps are not an exact mapping to original source-frame
   PTS; keep an explicit source-frame/PTS mapping if that precision is required.
   To compare sampling densities of one video, extract at the finest interval
   once and make sparser captures from it with `video-frames.py subset`, which
   links frames instead of copying them (see its README).
4. Review the extracted set for consistent orientation and dimensions, blur,
   duplicate views, abrupt zoom/exposure changes, moving subjects and adequate
   camera motion. Freeze the reviewed set; a changed sampling plan is a new
   versioned capture or derivative.
5. Set `[dataset] images_subpath` and `image_extensions` for the extracted
   files. For ordered video frames, set `[colmap] data_type = video` explicitly
   if using COLMAP's video matching mode. The `auto` heuristic does not infer
   it from a canonical `images/` directory. This setting is passed to COLMAP
   for the image sequence; it does not decode the movie.
6. Keep `[nerfstudio] scene_type = images` and `video_path` empty. The separate
   `scene_type = video` branch is a different route and has not been validated
   end to end with this pipeline's existing COLMAP workflow. Do a pipeline
   `--dry-run` and confirm it points at the extracted images.
7. Before training, inspect the sparse camera solution and registered-frame
   coverage. Review the camera path for continuity, unexpected jumps or
   disconnected views, and confirm that views cover the intended scene.
   Registration count and reprojection error alone do not establish correct
   alignment. Use this camera review as a gate before the more expensive
   training stage.

If masks are used, they must correspond one-to-one with the extracted images.
Use distinct configured image and mask suffixes; the config loader refuses an
extension configured for both. The mask filename and directory layout rules
are the same as for still-photo captures. With `use_masks = false`, the pipeline
trains on the unmasked frames and retains their visible background.

A run with `[train] eval_mode = all` trains on every frame that COLMAP
registered. It has no held-out views, so it cannot make a held-out
generalization claim. The default split and its evaluation meaning are
documented in section 6. A valid PLY export confirms that the file was written
and passed structural checks; it does not establish scene coverage or visual
usefulness.

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

**On WSL2: `Failed to initialize NVML`, `torch is installed but reports no CUDA device`, or COLMAP's `CUDA driver version is insufficient for CUDA runtime version`, while Windows still lists the GPU**
WSL has lost the GPU. COLMAP's message is the misleading one: it can appear
after minutes of CPU feature extraction, and the driver is not too old. Check
inside WSL, not on Windows, before any GPU job:

```bash
nvidia-smi -L        # or /usr/lib/wsl/lib/nvidia-smi -L if it is not on PATH
```

A laptop power mode that switches the discrete GPU off (an "eco" or hybrid
graphics mode in the vendor's control utility) causes it; switch to a mode that
keeps the GPU on and check again, which needed no WSL restart when this was
seen (2026-10-20). After a Windows driver update or sleep, stop other WSL work
and run `wsl --shutdown` in PowerShell, then start the distro again.

**Training starts and never finishes, with no output**
`[train] quit_on_train_completion = false` leaves the viewer running forever. The
runner refuses to start in that state; set it to `true` for unattended runs.

**`torch could not be imported (No module named 'torch')` although `uv sync` succeeded**

The pipeline is running under a different Python from the venv's. The message
names the interpreter and its environment; if that is
`$HOME/opt/colmap-prefix/bin/python`, the COLMAP prefix's own `python` came
first on `PATH` (see section 3, "The prefix also contains a `python`"). Start
the command with `uv run python ...`, or give the venv's full path. Found
2026-10-21 with `wsl-guard.py --python "$(which python)"`, which now refuses
that interpreter before starting. On Ubuntu a bare `python` may not exist at
all (`Command 'python' not found`); the fix is the same, not
`apt install python-is-python3`.

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
