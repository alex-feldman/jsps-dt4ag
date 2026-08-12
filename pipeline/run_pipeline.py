#!/usr/bin/env python3
"""Command-line runner for the jsps-dt4ag reconstruction pipeline.

Same five stages as ``notebooks/nerfstudio-pipeline-06.ipynb``, no Jupyter:

    1. colmap  : ``colmap automatic_reconstructor``
    2. process : ``ns-process-data images --skip-colmap``
    3. train   : ``ns-train <method>``
    4.           latest-checkpoint discovery
    5. export  : ``ns-export <format>``

Usage, with the reconstruction environment already activated::

    conda activate ns-l-oci
    python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini

Configuration comes from the same INI loader the notebook uses
(``dt4ag_config.py``), so the two cannot drift. Nothing is configured on the
command line except which stages to run and whether to execute anything at all.

Design rule, inherited from the config loader: **fail loudly**. Every
subprocess return code is checked, every artefact a stage claims to have
produced is looked for on disk, and the run stops at the first failure with a
non-zero exit status. This codebase has a documented history of succeeding
loudly while producing nothing; nothing here is allowed to repeat it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from dt4ag_config import (  # noqa: E402
    ConfigError,
    Dt4agConfig,
    find_config,
    load_config,
)

__all__ = [
    "STAGES",
    "StageError",
    "build_parser",
    "parse_args",
    "resolve_stages",
    "discover_checkpoint",
    "colmap_command",
    "process_command",
    "train_command",
    "export_command",
    "export_filename",
    "count_images",
    "read_ply_vertex_count",
    "main",
]

#: The pipeline's stages, in the only order they can legally run.
STAGES: Tuple[str, ...] = ("colmap", "process", "train", "export")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

#: A gaussian-splat PLY smaller than this is a header with no geometry behind
#: it. The real thing runs to tens of megabytes.
MIN_EXPORT_BYTES = 1024

#: Timestamped nerfstudio run directories are named like ``2026-08-07_170701``.
RUN_DIR_TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"


class StageError(Exception):
    """A stage failed, or produced nothing that can be handed to the next one."""


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

def _now() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def log_banner(message: str) -> None:
    print(f"\n[{_now()}] {'=' * 8} {message}", flush=True)


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


# --------------------------------------------------------------------------
# command construction, mirroring the notebook cell for cell
# --------------------------------------------------------------------------

def colmap_command(cfg: Dt4agConfig, workspace: Path) -> List[str]:
    command = [
        "colmap",
        "automatic_reconstructor",
        "--workspace_path", str(workspace),
        "--image_path", str(cfg.images_path),
        "--data_type", cfg.resolve_colmap_data_type(),
        "--single_camera", str(cfg.colmap_single_camera),
        "--single_camera_per_folder", str(cfg.colmap_single_camera_per_folder),
        "--dense", str(cfg.colmap_dense),
    ]
    command.extend(shlex.split(cfg.colmap_extra_args))
    return command


def process_command(cfg: Dt4agConfig, workspace: Path) -> List[str]:
    if cfg.scene_type == "video":
        # Not exercised by this pipeline; kept so the config value is honoured
        # rather than silently ignored.
        return [
            "ns-process-data", "video",
            "--data", str(cfg.video_path),
            "--output-dir", str(workspace),
        ]
    command = [
        "ns-process-data", "images",
        "--data", str(cfg.images_path),
        "--output-dir", str(workspace),
    ]
    if cfg.skip_colmap:
        command.append("--skip-colmap")
    command += ["--colmap-model-path", cfg.colmap_model_path]
    return command


def train_command(cfg: Dt4agConfig, workspace: Path) -> List[str]:
    return [
        "ns-train", cfg.train_method,
        "--data", str(workspace),
        "--pipeline.model.use_scale_regularization", str(cfg.use_scale_regularization),
        "--pipeline.model.background_color", cfg.background_color,
        "--output-dir", str(cfg.output_parent),
        "--viewer.quit-on-train-completion", str(cfg.quit_on_train_completion),
        "--max-num-iterations", str(cfg.max_num_iterations),
        "--logging.local-writer.max-log-size", str(cfg.max_log_size),
    ]


def export_filename(cfg: Dt4agConfig, run_id: str) -> str:
    """The notebook's export filename: the run's whole provenance in one name."""
    return "_".join([
        cfg.images_path.parent.name,
        run_id,
        "splat",
        cfg.platform_label,
        cfg.env_label,
        f"{cfg.max_num_iterations}steps",
        cfg.resolve_colmap_data_type(),
    ]) + ".ply"


def export_command(
    cfg: Dt4agConfig, config_yml: Path, export_dir: Path, filename: str
) -> List[str]:
    return [
        "ns-export", cfg.export_format,
        "--load-config", str(config_yml),
        "--output-dir", str(export_dir),
        "--output-filename", filename,
    ]


def gauss_to_pc_command(cfg: Dt4agConfig, ply: Path, workspace: Path) -> List[str]:
    output = str(ply.parent / ply.stem) + "_3dgs-to-pc.ply"
    return [
        sys.executable, os.path.expanduser(cfg.gauss_to_pc_script),
        "--input_path", str(ply),
        "--transform_path", str(workspace),
        "--output_path", output,
    ]


# --------------------------------------------------------------------------
# artefact inspection
# --------------------------------------------------------------------------

def count_images(images_path: Path) -> List[Path]:
    """Every image under ``images_path``, searched RECURSIVELY.

    Recursion is not incidental. With ``single_camera_per_folder`` the images
    sit in one subdirectory per camera rather than flat in this directory, and
    a non-recursive check reports zero images on a perfectly good dataset.
    """
    return sorted(
        p for p in images_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _run_dir_time(path: Path) -> float:
    """Best available wall-clock time for a nerfstudio run directory."""
    try:
        return _dt.datetime.strptime(
            path.name, RUN_DIR_TIMESTAMP_FORMAT
        ).timestamp()
    except ValueError:
        return path.stat().st_mtime


def discover_checkpoint(
    method_dir: Path, not_before: Optional[float] = None
) -> Tuple[Path, Path, Path]:
    """Pick the training run to export from: ``(run_dir, config_yml, checkpoint)``.

    Timestamped run directories sort chronologically, so the newest is last.
    "Newest" is not the same as "usable": a crashed ``ns-train`` leaves behind a
    directory holding ``config.yml`` and ``dataparser_transforms.json`` and no
    weights at all, and ``ns-export`` against that dies deep inside the
    checkpoint loader with an unhelpful message. So BOTH a ``config.yml`` and at
    least one checkpoint are required.

    ``not_before`` is a POSIX timestamp. When given, run directories older than
    it are rejected outright. Pass the moment training started, so that a
    training run which crashed without saving weights cannot be papered over by
    exporting some earlier run's checkpoint and calling it a success.
    """
    if not method_dir.is_dir():
        raise StageError(
            f"no training output directory at {method_dir}. ns-train either did "
            f"not run or wrote somewhere else; run the train stage first."
        )
    run_dirs = sorted(p for p in method_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise StageError(f"no training run directories under {method_dir}")

    usable: List[Tuple[Path, Path]] = []
    stale: List[str] = []
    for path in run_dirs:
        config_yml = path / "config.yml"
        checkpoints = sorted((path / "nerfstudio_models").glob("*.ckpt"))
        detail = (
            f"  {path.name}  config.yml={config_yml.is_file()}  "
            f"checkpoints={len(checkpoints)}"
            + (f"  latest={checkpoints[-1].name}" if checkpoints else "")
        )
        if not (config_yml.is_file() and checkpoints):
            log(detail + "  -> unusable")
            continue
        if not_before is not None and _run_dir_time(path) < not_before - 1:
            log(detail + "  -> predates this run, ignored")
            stale.append(path.name)
            continue
        log(detail + "  -> usable")
        usable.append((path, checkpoints[-1]))

    if not usable:
        extra = ""
        if stale:
            extra = (
                f" Usable but older run directories exist ({', '.join(stale)}); "
                f"they are NOT being exported, because they are not what this "
                f"invocation trained."
            )
        raise StageError(
            f"no run directory under {method_dir} has both a config.yml and a "
            f"checkpoint in nerfstudio_models/. Training did not get far enough "
            f"to save weights; re-run the train stage before exporting.{extra}"
        )

    run_dir, checkpoint = usable[-1]
    return run_dir, run_dir / "config.yml", checkpoint


def read_ply_vertex_count(path: Path) -> Optional[int]:
    """Vertex count from a PLY header, or None if the header does not say."""
    with path.open("rb") as handle:
        header = handle.read(4096)
    text = header.split(b"end_header", 1)[0].decode("ascii", "replace")
    match = re.search(r"element\s+vertex\s+(\d+)", text)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# prerequisites
# --------------------------------------------------------------------------

# Note the uv case comes FIRST and is spelled out. Running the venv's
# interpreter directly (.venv/bin/python run_pipeline.py) puts the right Python
# on the path but NOT the venv's bin/, so none of the ns-* commands resolve and
# this hint fires. That happened on 2026-08-11 and the old text, which only
# mentioned conda, sent the reader to the fallback environment instead of to
# `uv run`. Keep both paths, uv first, since uv is the default.
_HINT = (
    "Activate an environment that has the pipeline installed.\n"
    "  With uv (the default), run through uv so the venv's bin/ is on PATH:\n"
    "    uv run python pipeline/run_pipeline.py --config <your.ini>\n"
    "  Running .venv/bin/python directly is NOT enough: it gives you the right\n"
    "  interpreter but leaves ns-train, ns-process-data and ns-export off PATH.\n"
    "  COLMAP and ffmpeg are separate binaries that uv does not install. Put\n"
    "  the prefix holding them on PATH:\n"
    "    export PATH=$HOME/opt/colmap-prefix/bin:$PATH\n"
    "  With conda (the fallback):\n"
    "    conda activate ns-l-oci\n"
    "  See pipeline/QUICKSTART.md section 3."
)


def _check_colmap(problems: List[str]) -> None:
    if shutil.which("colmap") is None:
        problems.append(f"'colmap' is not on PATH.\n  {_HINT}")
        return
    # COLMAP 3.12 rejects --version and prints its banner on a bare invocation,
    # so ask for the banner and read the version out of it.
    try:
        proc = subprocess.run(
            ["colmap"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        problems.append(f"'colmap' could not be executed: {exc}\n  {_HINT}")
        return
    output = (proc.stdout or "") + (proc.stderr or "")
    # Kept generic on purpose. The conda-forge build the QUICKSTART recipe
    # installs carries an $ORIGIN-relative RPATH and resolves every library from
    # its own prefix, so this cannot fire for the supported route. A COLMAP the
    # user built or installed some other way still can, and the loader error is
    # otherwise easy to mistake for a missing binary.
    if "error while loading shared libraries" in output:
        problems.append(
            "'colmap' is on PATH but cannot load its shared libraries:\n"
            f"    {output.strip().splitlines()[0] if output.strip() else '(no output)'}\n"
            "  This COLMAP was not installed by the recipe in "
            "pipeline/QUICKSTART.md, which resolves its own libraries.\n"
            f"  {_HINT}"
        )
        return
    match = re.search(r"COLMAP\s+\S+", output)
    log(f"colmap            : {match.group(0) if match else 'present (version not parsed)'}")


def _check_nerfstudio(problems: List[str], commands: Iterable[str]) -> None:
    for name in commands:
        found = shutil.which(name)
        if found is None:
            problems.append(
                f"'{name}' is not on PATH, so nerfstudio is not importable from "
                f"this shell.\n  {_HINT}"
            )
        else:
            log(f"{name:<18}: {found}")


#: Compute capability -> the GPU family a user would recognise. Only used to
#: make the unsupported-GPU message readable; the check itself is numeric.
_CUDA_SERIES = {
    50: "Maxwell", 52: "Maxwell", 53: "Maxwell",
    60: "Pascal (GTX 10 series)", 61: "Pascal (GTX 10 series)", 62: "Pascal",
    70: "Volta (V100)", 72: "Xavier", 75: "Turing (RTX 20 / GTX 16 series, T4)",
    80: "Ampere (A100)", 86: "Ampere (RTX 30 series, A10/A40)", 87: "Orin",
    89: "Ada Lovelace (RTX 40 series, L4/L40)",
    90: "Hopper (H100/H200)",
    100: "Blackwell datacenter (B200)", 120: "Blackwell (RTX 50 series)",
}


def _describe_capability(cap: int) -> str:
    series = _CUDA_SERIES.get(cap)
    return f"sm_{cap} ({series})" if series else f"sm_{cap}"


def _gsplat_binary_archs() -> Tuple[Optional[set], bool]:
    """Read the CUDA architectures actually baked into the installed gsplat.

    Deliberately MEASURED, not hardcoded. The set of GPUs this pipeline can run
    on is a property of whichever prebuilt gsplat wheel happens to be installed,
    not of this repository, and it changes the moment that wheel changes. A
    hardcoded list would silently rot into a lie.

    Reads the CUDA cubins embedded in gsplat's compiled extension. Each one is
    an ELF with EI_OSABI 0x33 (ELFOSABI_CUDA) whose low e_flags byte is the
    compute capability it was compiled for.

    Returns (architectures, has_ptx). `architectures` is None when the binary
    could not be located or parsed, in which case the caller MUST NOT block the
    run: an unreadable binary is not evidence of an unsupported GPU. `has_ptx`
    reports whether the fatbin also carries PTX, which the driver can JIT for
    newer architectures; without it the supported set is closed.
    """
    try:
        import gsplat  # noqa: PLC0415
    except Exception:
        return None, False
    root = Path(gsplat.__file__).resolve().parent
    candidates = sorted(root.glob("csrc*.so")) + sorted(root.glob("*.so"))
    if not candidates:
        return None, False
    try:
        blob = candidates[0].read_bytes()
    except OSError:
        return None, False

    import struct  # noqa: PLC0415

    archs = set()
    for match in re.finditer(b"\x7fELF", blob):
        off = match.start()
        if off + 0x34 > len(blob):
            continue
        # EI_CLASS == 2 (64-bit) and EI_OSABI == 0x33 (CUDA)
        if blob[off + 4] != 2 or blob[off + 7] != 0x33:
            continue
        archs.add(struct.unpack_from("<I", blob, off + 0x30)[0] & 0xFF)
    # Fatbin entry kinds: 1 == PTX, 2 == cubin. PTX means forward compatibility.
    has_ptx = False
    for match in re.finditer(b"\x50\xed\x55\xba", blob):
        off = match.start()
        hdr = struct.unpack_from("<H", blob, off + 6)[0] if off + 8 <= len(blob) else 0
        if hdr not in (0x08, 0x10) or off + hdr + 2 > len(blob):
            continue
        if struct.unpack_from("<H", blob, off + hdr)[0] == 1:
            has_ptx = True
            break
    return (archs or None), has_ptx


def _check_gpu_arch(problems: List[str], capability: Tuple[int, int]) -> None:
    """Refuse an unsupported GPU now, rather than cryptically during training.

    Without this the failure lands inside gsplat after COLMAP and
    ns-process-data have already run, which on the development dataset is about
    ten minutes of wasted work and an error that does not name the cause.
    """
    archs, has_ptx = _gsplat_binary_archs()
    if archs is None:
        log("gpu support       : not determined (gsplat binary unreadable), continuing")
        return
    major, minor = capability
    device_cap = major * 10 + minor
    # CUDA binary compatibility: a cubin built for sm_X.y runs on any device
    # sm_X.z where z >= y. It never crosses a major version.
    supported = any(a // 10 == major and a % 10 <= minor for a in archs)
    listed = ", ".join(f"sm_{a}" for a in sorted(archs))
    if supported or has_ptx:
        note = " (+PTX, so newer GPUs can JIT)" if has_ptx else ""
        log(f"gpu support       : {_describe_capability(device_cap)} supported by gsplat [{listed}]{note}")
        return
    # Every capability the embedded cubins actually cover, in hardware order,
    # so the message names the GPUs a reader recognises rather than sm numbers.
    covered = sorted(
        cap
        for cap in _CUDA_SERIES
        if any(a // 10 == cap // 10 and a % 10 <= cap % 10 for a in archs)
    )
    families: List[str] = []
    for cap in covered:
        name = _CUDA_SERIES[cap]
        if name not in families:
            families.append(name)
    problems.append(
        f"this GPU is {_describe_capability(device_cap)}, which the installed "
        f"gsplat cannot run.\n"
        f"  The gsplat wheel embeds code for [{listed}] and carries no PTX, so "
        f"there is no just-in-time fallback\n"
        f"  and the supported set is closed. Supported GPU families:\n"
        f"    {'; '.join(families)}\n"
        f"  This is a property of the prebuilt wheel, not of this repository, "
        f"so it cannot be fixed by configuration.\n"
        f"  See pipeline/QUICKSTART.md, 'Which GPUs work'."
    )


def _check_cuda(problems: List[str]) -> None:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        problems.append(
            f"torch could not be imported ({exc}).\n  {_HINT}"
        )
        return
    if not torch.cuda.is_available():
        problems.append(
            "torch is installed but reports no CUDA device. Training and export "
            "both need a GPU.\n"
            "  Check 'nvidia-smi' reports a driver and a device. torch ships "
            "its own CUDA runtime, so no\n"
            "  system CUDA toolkit is involved."
        )
        return
    capability = torch.cuda.get_device_capability(0)
    log(
        f"cuda              : {torch.cuda.get_device_name(0)}, "
        f"compute {capability[0]}.{capability[1]}, torch {torch.__version__}"
    )
    _check_gpu_arch(problems, capability)


def check_prerequisites(stages: Sequence[str]) -> None:
    """Verify the tools each selected stage needs, before anything long starts.

    Alex's stated assumption is that the conda environment is active. This
    verifies that rather than trusting it, because every failure mode here is
    cheap to detect now and expensive to discover forty minutes into a run.
    """
    log_banner("prerequisites")
    problems: List[str] = []
    if "colmap" in stages:
        _check_colmap(problems)
    needed = []
    if "process" in stages:
        needed.append("ns-process-data")
    if "train" in stages:
        needed.append("ns-train")
    if "export" in stages:
        needed.append("ns-export")
    if needed:
        _check_nerfstudio(problems, needed)
    if {"train", "export"} & set(stages):
        _check_cuda(problems)
    if problems:
        raise StageError(
            "prerequisite check failed:\n\n"
            + "\n\n".join(f"- {p}" for p in problems)
        )
    log("all prerequisites satisfied")


# --------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the reconstruction pipeline from the command line: COLMAP, "
            "ns-process-data, ns-train, ns-export. No Jupyter involved."
        ),
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --config configs/my-run.ini\n"
            "  python run_pipeline.py --config configs/my-run.ini --dry-run\n"
            "  python run_pipeline.py --config configs/my-run.ini "
            "--from-stage train\n"
            "  python run_pipeline.py --config configs/my-run.ini "
            "--stage train,export --run-id run_260807-01-3120\n"
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "path to the INI config. Defaults to $DT4AG_CONFIG, then to "
            "configs/example.ini found by walking up from the working directory."
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        metavar="NAME",
        help=(
            "run only these stages: " + "|".join(STAGES) + ". Repeatable, and "
            "comma-separated lists are accepted. Stages always execute in "
            "pipeline order regardless of the order given."
        ),
    )
    parser.add_argument(
        "--from-stage",
        metavar="NAME",
        choices=STAGES,
        help="resume: run this stage and every stage after it.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "address an existing run instead of deriving a new id from [run]. "
            "Needed when resuming a run whose config leaves date/run_count blank."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact commands and the paths they use, execute nothing.",
    )
    parser.add_argument(
        "--allow-viewer-hang",
        action="store_true",
        help=(
            "permit [train] quit_on_train_completion = false. Off by default "
            "because ns-train then keeps its viewer alive after training ends, "
            "which never returns in a non-interactive run."
        ),
    )
    return parser


def resolve_stages(
    stage_args: Optional[Sequence[str]], from_stage: Optional[str]
) -> List[str]:
    """Turn ``--stage`` / ``--from-stage`` into the stage list, in run order."""
    if stage_args and from_stage:
        raise SystemExit(
            "error: --stage and --from-stage are mutually exclusive; use one or "
            "the other."
        )
    if from_stage:
        return list(STAGES[STAGES.index(from_stage):])
    if not stage_args:
        return list(STAGES)
    requested = []
    for chunk in stage_args:
        for name in chunk.split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name not in STAGES:
                raise SystemExit(
                    f"error: unknown stage {name!r}; choose from "
                    + ", ".join(STAGES)
                )
            if name not in requested:
                requested.append(name)
    if not requested:
        raise SystemExit("error: --stage was given but named no stage")
    return [name for name in STAGES if name in requested]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.stages = resolve_stages(args.stage, args.from_stage)
    return args


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def run_command(command: Sequence[str], label: str, dry_run: bool) -> None:
    """Run a subprocess, streaming its output, and raise on any failure."""
    print(f"    $ {render(command)}", flush=True)
    if dry_run:
        return
    started = time.time()
    try:
        completed = subprocess.run(list(command), check=False)
    except OSError as exc:
        raise StageError(f"{label}: could not execute {command[0]!r}: {exc}") from exc
    elapsed = format_elapsed(time.time() - started)
    if completed.returncode != 0:
        raise StageError(
            f"{label}: exited {completed.returncode} after {elapsed}\n"
            f"  command: {render(command)}"
        )
    log(f"{label}: exit 0 after {elapsed}")


def run_pipeline(cfg: Dt4agConfig, args: argparse.Namespace) -> int:
    stages = args.stages
    dry_run = args.dry_run

    if "train" in stages and not cfg.quit_on_train_completion and not args.allow_viewer_hang:
        raise StageError(
            f"[train] quit_on_train_completion is false in {cfg.source}.\n"
            "  ns-train then keeps its viewer running after training finishes and "
            "never exits, which deadlocks a non-interactive run.\n"
            "  Set it to true for command-line runs, or pass --allow-viewer-hang "
            "if you really intend to stop the run by hand."
        )

    if not dry_run:
        check_prerequisites(stages)

    run_id = args.run_id or cfg.make_run_id()
    workspace = cfg.colmap_workspace(run_id)
    method_dir = cfg.output_dir(run_id) / cfg.train_method

    log_banner(f"run {run_id}")
    log(f"config            : {cfg.source}")
    log(f"stages            : {', '.join(stages)}"
        + ("  (dry run, nothing will execute)" if dry_run else ""))
    log(f"images            : {cfg.images_path}")
    log(f"colmap workspace  : {workspace}")
    log(f"training output   : {cfg.output_parent}")

    images = count_images(cfg.images_path)
    if not images:
        raise StageError(
            f"no image files under {cfg.images_path} (searched recursively). "
            f"Check [dataset] images_subpath in {cfg.source}."
        )
    camera_dirs = sorted({
        p.parent.relative_to(cfg.images_path).as_posix() for p in images
    })
    log(f"images found      : {len(images)} in "
        + (f"{len(camera_dirs)} subdirectories" if camera_dirs != ["."] else "one flat directory"))

    if stages[0] != "colmap" and not workspace.is_dir() and not dry_run:
        raise StageError(
            f"stage '{stages[0]}' needs the COLMAP workspace {workspace}, which "
            f"does not exist.\n"
            "  A run id with a blank [run] date/run_count auto-increments, so "
            "resuming in a second invocation derives a NEW id rather than the "
            "one you meant.\n"
            "  Pass --run-id <existing id>, or pin [run] date and run_count."
        )

    if not dry_run:
        log(f"run log           : {cfg.append_run_log(run_id)}")

    overall_start = time.time()
    train_started: Optional[float] = None

    for stage in stages:
        stage_start = time.time()
        log_banner(f"stage {STAGES.index(stage) + 1}/{len(STAGES)}: {stage}")

        if stage == "colmap":
            if not dry_run:
                workspace.mkdir(parents=True, exist_ok=True)
            run_command(colmap_command(cfg, workspace), "colmap", dry_run)
            sparse = workspace / "sparse" / "0"
            if not dry_run and not (sparse / "cameras.bin").is_file():
                raise StageError(
                    f"colmap reported success but produced no sparse model at "
                    f"{sparse}. Nothing downstream can use this workspace."
                )

        elif stage == "process":
            run_command(process_command(cfg, workspace), "ns-process-data", dry_run)
            transforms = workspace / "transforms.json"
            if not dry_run and not transforms.is_file():
                raise StageError(
                    f"ns-process-data reported success but {transforms} does not "
                    f"exist. ns-train has nothing to read."
                )
            if not dry_run:
                log(f"transforms.json   : {transforms.stat().st_size} bytes")

        elif stage == "train":
            train_started = time.time()
            run_command(train_command(cfg, workspace), "ns-train", dry_run)

        elif stage == "export":
            if dry_run:
                run_dir = method_dir / "<timestamp>"
                config_yml = run_dir / "config.yml"
            else:
                run_dir, config_yml, checkpoint = discover_checkpoint(
                    method_dir, not_before=train_started
                )
                log(f"selected run dir  : {run_dir.name}")
                log(f"config.yml        : {config_yml}")
                log(f"checkpoint        : {checkpoint.name} "
                    f"({checkpoint.stat().st_size} bytes)")
            # Exports go to the configured [paths] exports_dirname, NOT into the
            # training run directory. Until 2026-08-07 this was hardcoded to
            # run_dir / "exports", which made the config key decorative: it was
            # documented, settable, and did nothing. The export filename already
            # carries the run id, so a single flat directory cannot collide and
            # gives one place to find every export.
            export_dir = cfg.exports_dir
            filename = export_filename(cfg, run_id)
            ply = export_dir / filename
            run_command(
                export_command(cfg, config_yml, export_dir, filename),
                "ns-export",
                dry_run,
            )
            if not dry_run:
                verify_export(ply, export_dir)
                if cfg.export_3dgs:
                    run_command(
                        gauss_to_pc_command(cfg, ply, workspace),
                        "gauss_to_pc",
                        dry_run,
                    )

        log(f"stage {stage}: done in {format_elapsed(time.time() - stage_start)}")

    log_banner(
        f"pipeline finished in {format_elapsed(time.time() - overall_start)}"
        + (" (dry run)" if dry_run else "")
    )
    return 0


def verify_export(ply: Path, export_dir: Path) -> None:
    """A zero exit status from ns-export is not proof a splat landed on disk."""
    if not ply.is_file():
        listing = (
            sorted(p.name for p in export_dir.iterdir())
            if export_dir.is_dir() else "(no such directory)"
        )
        raise StageError(
            f"ns-export reported success but {ply} does not exist.\n"
            f"  {export_dir} contains: {listing}"
        )
    size = ply.stat().st_size
    if size < MIN_EXPORT_BYTES:
        raise StageError(
            f"{ply} is only {size} bytes, which is a header and no geometry. "
            f"Training almost certainly produced nothing usable."
        )
    vertices = read_ply_vertex_count(ply)
    log(f"exported file     : {ply}")
    log(f"bytes             : {size} ({size / 1024 / 1024:.2f} MiB)")
    log(f"vertices          : {vertices if vertices is not None else 'not stated in header'}")
    if vertices == 0:
        raise StageError(
            f"{ply} declares 0 vertices in its PLY header: the export is empty."
        )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config_path = find_config(args.config)
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        return run_pipeline(cfg, args)
    except StageError as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"\nCONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
