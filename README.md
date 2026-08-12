# jsps-dt4ag

Turn a set of photographs into a 3D Gaussian splat. Built for time-series
digital-twin work on plants, but **object-agnostic**: it reconstructs arbitrary
scenes and contains no subject-specific logic.

### [See a reconstruction in your browser](https://alex-feldman.github.io/jsps-dt4ag/)

No install needed to look. The demo page renders finished splats interactively.

## What it does

Four stages, run from one command and one INI config file:

```
COLMAP structure-from-motion  ->  ns-process-data  ->  ns-train splatfacto  ->  ns-export
```

It exits non-zero on the first failure, and verifies the artefact each stage
claims to have produced rather than trusting a success message.

## Run it yourself

```bash
git clone https://github.com/alex-feldman/jsps-dt4ag.git
cd jsps-dt4ag
scripts/install.sh
```

Then follow **[`pipeline/QUICKSTART.md`](pipeline/QUICKSTART.md)**, which is the
only place the procedure is written down, so it cannot drift from the version
you cloned.

`scripts/install.sh` handles the system packages, `uv`, the COLMAP 3.12.0 CUDA
prefix and the locked Python environment, then verifies the result. It was run
on a bare `ubuntu:24.04` holding no compiler, `curl`, `git` or Python and
completed unaided; the pipeline then reproduced this project's reference
reconstruction to within 0.008 dB. Re-running it takes about three seconds and
skips whatever is already installed.

A **[sample dataset](https://drive.google.com/file/d/1Co9RLorlKGWBHSfN6WihAmzLryW_W2Ji/view?usp=sharing)**
(2.2 GB, 120 masked images) is published. It is the exact data behind the
reference numbers, so it is what a fresh install should be checked against:
expect roughly 1,300 to 1,360 gaussians at **PSNR ~46.5**, in about 20 minutes.

## Check these before investing time

Each one stops the pipeline rather than slowing it, and none is fixable by
configuration.

| | |
|---|---|
| **Linux x86-64 only** | Deliberate scope for this release. Windows and macOS are later work; WSL2 is untested and expected to work. |
| **NVIDIA GPU, Volta through Hopper** | **RTX 50 series (Blackwell) and Pascal do not work.** The prebuilt gsplat wheel contains no code for them and no PTX to JIT from. The runner checks at startup and refuses to begin. |
| **A GPU is required, not preferred** | Training and export have no CPU path: the rasteriser is CUDA-only. The first two stages do run on CPU. |
| **Disk and time** | About 7.4 GB for the environment and 6 GB for the COLMAP prefix. The first install is dominated by downloads and has taken anywhere from minutes to several hours depending on the PyPI CDN. |

Blackwell support is a hard gate on the 1.0 release, not an accepted permanent
limit.

## What this repository does not do

**The pipeline begins at MASKED images and does not create masks.** Mask
generation lives in a separate repository (`samask`) and mask application is a
manual pre-step. Point it at raw photographs and it will happily reconstruct
the background along with the subject, which is a supported thing to do; you
then crop or delete those splats yourself.

The datasets are not here either: roughly 99 GB of images, COLMAP workspaces
and training outputs live outside version control.

## Status: alpha

The pipeline has been run end to end and had its output verified, including on
clean machines that started with nothing. It is installable and reproducible.

What that does **not** yet mean: nobody outside this project has installed it,
there is no LICENSE or citation file yet, and the evaluation methodology
described on parts of the wiki has not been run. Treat published numbers as
coming from this project's own runs.

## Where things are

| | |
|---|---|
| [`pipeline/QUICKSTART.md`](pipeline/QUICKSTART.md) | **Start here to run it.** Install and use, from nothing. |
| [`pipeline/README.md`](pipeline/README.md) | What is in the pipeline directory, and the runner's options. |
| [`pipeline/configs/README.md`](pipeline/configs/README.md) | Every configuration key. |
| [`scripts/install.sh`](scripts/install.sh) | The automated install. `--dry-run` prints every command it would run. |
| [Project wiki](https://github.com/alex-feldman/jsps-dt4ag/wiki) | Background and context. It carries no install steps by design. |
| [Demo page](https://alex-feldman.github.io/jsps-dt4ag/) | Published from `docs/` on this branch. |
