# jsps-dt4ag

A reconstruction pipeline that turns a set of photographs into a 3D Gaussian
splat, built for time-series digital-twin work on plants. It is
**object-agnostic**: it reconstructs arbitrary scenes and contains no
subject-specific logic.

Four stages, run from one command and one INI config file: COLMAP
structure-from-motion, `ns-process-data`, `ns-train splatfacto`, `ns-export`.

## Install and run it

**Everything you need is in
[`pipeline/QUICKSTART.md`](pipeline/QUICKSTART.md).** It starts from a bare
machine with nothing installed and no clone, and it is the only place the
install procedure is written down, so it cannot drift from the version you
cloned.

```bash
git clone -b pipeline-alpha https://github.com/alex-feldman/jsps-dt4ag.git
cd jsps-dt4ag
# then follow pipeline/QUICKSTART.md
```

**The `-b pipeline-alpha` matters.** The default branch, `main`, holds only the
project's demo web page and contains no pipeline code at all.

## Before you invest any time, three hard limits

Check these first. Each one stops the pipeline rather than slowing it down, and
none is fixable by configuration.

| | |
|---|---|
| **Linux x86-64 only** | This is the alpha's deliberate scope. macOS and Windows are beta work. WSL2 is untested and expected to work. |
| **NVIDIA GPU, Volta through Hopper** | **RTX 50 series (Blackwell) and Pascal do not work.** The prebuilt gsplat wheel contains no code for them and no PTX to JIT from. The runner checks this at startup and refuses to begin. |
| **A GPU is required, not merely preferred** | Stages 3 and 4 have no CPU path: the rasterizer is CUDA-only. Stages 1 and 2 run on CPU. |

Blackwell support is a hard gate on the 1.0 release, not an accepted permanent
limit.

## What this repository does not do

**The pipeline begins at MASKED images and does not create masks.** Mask
generation lives in a separate repository (`samask`) and mask application is a
manual pre-step. Point it at raw photographs and it will happily reconstruct the
background too. See `pipeline/QUICKSTART.md` section 7.

The datasets are not here either: roughly 99 GB of images, COLMAP workspaces and
training outputs live outside version control.

## Status

**Alpha.** The pipeline has been run end to end and had its output verified,
including once on a clean `ubuntu:24.04` machine with no conda, no CUDA toolkit
and no compiler. It has not yet been installed from scratch by anyone other than
its author. There is no LICENSE or citation file yet.

The pipeline code lives on the **`pipeline-alpha`** branch. `main` is the demo
site and the two have diverged.

## Where things are

| Path | What it is |
|---|---|
| `pipeline/QUICKSTART.md` | **Start here.** Install and run, from nothing. |
| `pipeline/README.md` | What is in the pipeline directory, and the runner's options. |
| `pipeline/configs/README.md` | Every configuration key. |
| `pipeline/run_pipeline.py` | The command-line runner. The supported entry point. |
| `docs/` | The demo web page, published from `main` by GitHub Pages. |
| [Project wiki](https://github.com/alex-feldman/jsps-dt4ag/wiki) | Background and context. It carries no install steps by design; those live in QUICKSTART. |
