# Jupyter kernelspec

`kernel.json.example` is a kernelspec that carries the environment the pipeline
needs, so the notebook works regardless of how Jupyter was started.

## The problem it solves

A conda kernelspec normally invokes the environment's Python **directly**. It does
not activate the environment, so `etc/conda/activate.d/` hooks never run and the
kernel inherits whatever environment launched Jupyter.

That silently breaks two things:

- **COLMAP** cannot load `libcudart.so.12`, because it was built into the conda
  prefix and links its CUDA libraries from there. Needs `LD_LIBRARY_PATH`.
- **gsplat** fails to JIT-compile, because CUDA 12.1's `nvcc` refuses a host
  compiler newer than GCC 12. Needs `CC`/`CXX`/`CUDAHOSTCXX`.

Both work when Jupyter is launched from an already-activated shell, because the
shell ran the activation hooks. Any other launch path fails, with errors that
point at COLMAP or gsplat rather than at the real cause. This cost a full
debugging cycle on 2026-08-07.

## Installing it

Replace `/CONDA/ENV/PREFIX` with your environment prefix (`conda info --base` plus
`/envs/<name>`, or `echo $CONDA_PREFIX` inside the activated environment), then:

```bash
mkdir -p ~/.local/share/jupyter/kernels/<kernel-name>
cp kernel.json.example ~/.local/share/jupyter/kernels/<kernel-name>/kernel.json
# then edit the prefix in that copy
```

Verify it, rather than assuming:

```bash
python - <<'EOF'
import json, subprocess, os
k = json.load(open(os.path.expanduser("~/.local/share/jupyter/kernels/<kernel-name>/kernel.json")))
env = dict(os.environ); env.update(k["env"])
r = subprocess.run(["colmap"], env=env, capture_output=True, text=True)
print((r.stdout + r.stderr).splitlines()[0])
EOF
```

It should print the COLMAP banner. If it prints a `libcudart` error, the
`LD_LIBRARY_PATH` entry is wrong.

## Note on the compiler variables

`CC`, `CXX` and `CUDAHOSTCXX` matter only when gsplat rebuilds its CUDA
extension, which is normally once per environment. They are harmless otherwise,
and cheaper to carry than to debug when a rebuild is triggered unexpectedly.

If the environment ever moves to a prebuilt gsplat wheel, JIT compilation goes
away and these three become unnecessary. `LD_LIBRARY_PATH` is still required for
COLMAP either way.

## Two kernels, and which one to use

| Template | Kernel name | Stack | Use |
|---|---|---|---|
| `kernel-uv.json.example` | `dt4ag-uv` | uv, torch 2.4.1, prebuilt gsplat | **Default.** Same environment as the CLI |
| `kernel.json.example` | `ns-l-oci` | conda, torch 2.5.0, JIT gsplat | Fallback only |

The notebook is committed pointing at `dt4ag-uv`, so there is ONE dependency set
across the notebook and the command line. Before 2026-08-11 the notebook was
pinned to the conda kernel while the CLI defaulted to uv, which meant the two
paths silently ran different torch versions.

### Installing the uv kernel

Replace `/REPO` with the repository root and `/PREFIX/CONTAINING/bin/colmap`
with the prefix holding COLMAP and ffmpeg, then:

```bash
mkdir -p ~/.local/share/jupyter/kernels/dt4ag-uv
cp kernel-uv.json.example ~/.local/share/jupyter/kernels/dt4ag-uv/kernel.json
# edit the two placeholders in that copy
```

Verify it resolves everything, rather than assuming:

```bash
python - <<'EOF'
import json, os, subprocess
k = json.load(open(os.path.expanduser("~/.local/share/jupyter/kernels/dt4ag-uv/kernel.json")))
env = dict(os.environ); env.update(k["env"])
for v in ("CONDA_PREFIX", "LD_LIBRARY_PATH"): env.pop(v, None)
code = ("import shutil, torch, gsplat\n"
        "print(torch.__version__, gsplat.__version__, torch.cuda.is_available())\n"
        "from gsplat.cuda._backend import _C\n"
        "print({t: shutil.which(t) for t in ('colmap','ffmpeg','ns-train')})\n")
print(subprocess.run([k["argv"][0], "-c", code], env=env, capture_output=True, text=True).stdout)
EOF
```

Expect torch 2.4.1+cu121, gsplat 1.4.0+pt24cu121, `True`, and all three commands
resolving. Dropping `CONDA_PREFIX` and `LD_LIBRARY_PATH` is deliberate: it proves
the kernel does not depend on a conda environment being active.
