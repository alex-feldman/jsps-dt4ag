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
