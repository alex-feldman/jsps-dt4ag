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
import json
import os
import re
import shlex
import shutil
import struct
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
    "resolve_downscale_factor",
    "count_images",
    "read_ply_vertex_count",
    "verify_downscale_pyramid",
    "sfm_input_path",
    "stage_sfm_inputs",
    "attach_masks",
    "read_colmap_image_names",
    "main",
]

#: The pipeline's stages, in the only order they can legally run.
STAGES: Tuple[str, ...] = ("colmap", "process", "train", "export")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

#: nerfstudio's ``MAX_AUTO_RESOLUTION``, the longest edge its dataparser will
#: train at when it picks the downscale factor itself. Mirrored rather than
#: imported because this runner must be able to report the problem in an
#: environment where nerfstudio is not importable, and because a value that
#: drifts is caught by ``test_run_pipeline.py``, which asserts the two agree
#: whenever nerfstudio IS importable.
MAX_AUTO_RESOLUTION = 1600

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

def sfm_input_path(cfg: Dt4agConfig, workspace: Path) -> Path:
    """The directory COLMAP and ns-process-data are pointed at.

    The dataset directory itself unless the config filters files out of it, in
    which case a staged tree holding only the photographs, built beside the
    workspace so it is obvious which run it belongs to and is thrown away with
    it.
    """
    if not cfg.filters_dataset_files:
        return cfg.images_path
    return workspace.parent / f"{workspace.name}_sfm-input"


def colmap_command(cfg: Dt4agConfig, workspace: Path) -> List[str]:
    command = [
        "colmap",
        "automatic_reconstructor",
        "--workspace_path", str(workspace),
        "--image_path", str(sfm_input_path(cfg, workspace)),
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
        "--data", str(sfm_input_path(cfg, workspace)),
        "--output-dir", str(workspace),
    ]
    if cfg.skip_colmap:
        command.append("--skip-colmap")
    command += ["--colmap-model-path", cfg.colmap_model_path]
    return command


def train_command(cfg: Dt4agConfig, workspace: Path) -> List[str]:
    command = [
        "ns-train", cfg.train_method,
        "--data", str(workspace),
        "--pipeline.model.use_scale_regularization", str(cfg.use_scale_regularization),
        "--pipeline.model.background_color", cfg.background_color,
        "--output-dir", str(cfg.output_parent),
        "--viewer.quit-on-train-completion", str(cfg.quit_on_train_completion),
        "--max-num-iterations", str(cfg.max_num_iterations),
        "--logging.local-writer.max-log-size", str(cfg.max_log_size),
    ]
    # Left off entirely when 0, so nerfstudio keeps choosing for itself and
    # configs written before this key behave exactly as they did.
    #
    # The dataparser is a tyro SUBCOMMAND, not a nested config path: the flag
    # only exists as `... nerfstudio-data --downscale-factor N`, appended after
    # every option of the parent command. `--pipeline.datamanager.dataparser.
    # downscale-factor` looks right, matches how config.yml nests it, and is
    # rejected as an unrecognized option.
    #
    # `nerfstudio-data` is also the default subcommand, and it is what a
    # workspace with a transforms.json resolves to anyway, so naming it changes
    # nothing else about the run.
    if cfg.downscale_factor:
        command += [
            "nerfstudio-data",
            "--downscale-factor", str(cfg.downscale_factor),
        ]
    return command


def resolve_downscale_factor(cfg: Dt4agConfig, workspace: Path) -> int:
    """The factor training will ACTUALLY use, not the one the config requested.

    ``[train] downscale_factor = 0`` delegates the choice to nerfstudio, which
    picks it by probing the pyramid on disk (``nerfstudio_dataparser._get_fname``):
    step down while the long edge is over ``MAX_AUTO_RESOLUTION`` AND the next
    level actually holds the first frame. The chosen value is then written to no
    artefact, because ``config.yml`` records the request (``null``), not the
    decision.

    That matters beyond tidiness: resolution changes a reconstruction more than
    most settings, and without this the resolution a run trained at is a
    property of what happened to be on disk that day. Mirrored here, rather
    than read back from nerfstudio, so the value is known BEFORE training and
    can be stamped into the artefacts training produces.

    Returns 0 only when it genuinely cannot be determined (no transforms.json),
    which callers should report rather than treat as a factor.
    """
    if cfg.downscale_factor:
        return cfg.downscale_factor
    transforms = workspace / "transforms.json"
    if not transforms.is_file():
        return 0
    try:
        meta = json.loads(transforms.read_text())
    except (OSError, ValueError):
        return 0
    frames = meta.get("frames") or []
    if not frames:
        return 0
    max_edge = 0
    for frame in frames:
        for key in ("w", "h"):
            value = frame.get(key, meta.get(key))
            if value:
                max_edge = max(max_edge, int(value))
    if not max_edge:
        return 0
    # The probe tests ONE filename, the first frame as listed (not sorted),
    # exactly as the dataparser does.
    first = Path(frames[0]["file_path"]).name
    exponent = 0
    while True:
        if max_edge / (2 ** exponent) <= MAX_AUTO_RESOLUTION:
            break
        if not (workspace / f"images_{2 ** (exponent + 1)}" / first).is_file():
            break
        exponent += 1
    return 2 ** exponent


def export_filename(cfg: Dt4agConfig, run_id: str, downscale_factor: int = 0) -> str:
    """The notebook's export filename: the run's whole provenance in one name.

    ``downscale_factor`` joined the name on 2026-08-17. Without it, the same
    dataset reconstructed at two resolutions produces the same filename twice
    and the second export silently overwrites the first, which is exactly the
    comparison anyone changing the resolution is trying to make.
    """
    parts = [
        cfg.images_path.parent.name,
        run_id,
        "splat",
        cfg.platform_label,
        cfg.env_label,
        f"{cfg.max_num_iterations}steps",
    ]
    if downscale_factor:
        parts.append(f"ds{downscale_factor}")
    parts.append(cfg.resolve_colmap_data_type())
    return "_".join(parts) + ".ply"


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


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialise ``source`` at ``destination`` as cheaply as this OS allows.

    Symlink, then hardlink, then copy. Windows only permits symlinks with
    Developer Mode or elevation, and this pipeline is meant to run there too,
    so a failure to link is expected rather than exceptional.
    """
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(source, destination)
        return
    except OSError:
        shutil.copy2(source, destination)


def stage_sfm_inputs(cfg: Dt4agConfig, staged_root: Path) -> List[Path]:
    """Mirror only the photographs into a clean tree for COLMAP and nerfstudio.

    Neither ``colmap automatic_reconstructor`` nor ``ns-process-data images``
    can be told to ignore some of the files under the directory it is given:
    both take everything. A dataset that keeps per-image masks beside its
    photographs, which is what every masking tool writes by default, therefore
    feeds the masks to SfM as if they were photographs.

    On 2026-08-17 that cost a run twice over. COLMAP wasted its time trying to
    register 92 binary masks, and ns-process-data copied them into ``images/``
    interleaved by extension, which broke its ffmpeg downscale sequence and
    ultimately put 15 MP frames through a 6 GB card.

    The relative directory layout is preserved because
    ``single_camera_per_folder`` reads camera grouping from it.
    """
    photographs = [p for p in count_images(cfg.images_path) if cfg.is_photograph(p)]
    if not photographs:
        raise StageError(
            f"no photographs under {cfg.images_path} after applying "
            f"[dataset] image_extensions "
            f"({', '.join(sorted(cfg.image_extensions)) or 'any'}) and "
            f"mask_extensions "
            f"({', '.join(sorted(cfg.mask_extensions)) or 'none'})."
        )
    if staged_root.exists():
        shutil.rmtree(staged_root)
    for photograph in photographs:
        destination = staged_root / photograph.relative_to(cfg.images_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(photograph, destination)
    # exFAT, which is what an external data drive usually is, supports neither
    # symlinks nor hardlinks, so this is a full copy of the photographs. Say so
    # rather than letting a silent multi-hundred-megabyte write look like a
    # pause. The process stage removes the tree when it is done with it.
    sample = staged_root / photographs[0].relative_to(cfg.images_path)
    if not sample.is_symlink() and sample.stat().st_nlink == 1:
        megabytes = sum(p.stat().st_size for p in photographs) / (1 << 20)
        log(f"sfm input tree    : copied ({megabytes:.0f} MiB); "
            f"{staged_root.parent} supports no links")
    return photographs


def read_colmap_image_names(model_dir: Path) -> dict:
    """Map COLMAP image id -> the image's original name, from ``images.bin``.

    ``images.bin`` is a length-prefixed binary record per registered image:
    uint32 image_id, 4+3 doubles of pose, uint32 camera_id, a NUL-terminated
    name, then uint64 num_points2D followed by that many (double, double,
    uint64) observations.

    Parsed here with ``struct`` rather than imported from nerfstudio, because
    this runner deliberately depends on nerfstudio being on PATH, not on being
    importable from the same interpreter. The parse is self-checking: it must
    consume the file exactly, which is what catches a format change instead of
    returning plausible garbage.
    """
    path = model_dir / "images.bin"
    if not path.is_file():
        raise StageError(
            f"{path} does not exist, so processed frames cannot be mapped back "
            f"to their source photographs.\n"
            f"  A text-format COLMAP model is not supported here; re-run the "
            f"colmap stage, or set [dataset] use_masks = false."
        )
    blob = path.read_bytes()
    offset = 0
    (count,) = struct.unpack_from("<Q", blob, offset)
    offset += 8
    names = {}
    try:
        for _ in range(count):
            (image_id,) = struct.unpack_from("<I", blob, offset)
            offset += 4 + 8 * 7 + 4          # pose (7 doubles) and camera_id
            end = blob.index(b"\x00", offset)
            names[image_id] = blob[offset:end].decode("utf-8")
            offset = end + 1
            (num_points,) = struct.unpack_from("<Q", blob, offset)
            offset += 8 + num_points * 24
    except (struct.error, ValueError, UnicodeDecodeError) as exc:
        raise StageError(f"{path} could not be parsed: {exc}") from exc
    if offset != len(blob) or len(names) != count:
        raise StageError(
            f"{path} did not parse cleanly: read {len(names)} of {count} "
            f"images and consumed {offset} of {len(blob)} bytes.\n"
            f"  This COLMAP writes a format this runner does not understand, "
            f"so frames cannot be mapped back to source photographs safely."
        )
    return names


def attach_masks(cfg: Dt4agConfig, workspace: Path) -> List[str]:
    """Wire the dataset's per-image masks into the processed workspace.

    ns-process-data has no per-image mask support: its only ``--mask`` style
    option is a synthetic crop mask. What nerfstudio's dataparser reads is a
    ``mask_path`` on each frame pointing into ``masks/``, with downscaled
    copies in ``masks_N/`` matching the image pyramid.

    ns-process-data renames every photograph to ``frame_%05d`` and does not
    persist the rename map, so the pairing has to be recovered. It is recovered
    from ``colmap_im_id``, which nerfstudio stamps on every frame it writes
    (``colmap_utils.colmap_to_json``), resolved against the names in the COLMAP
    model. That is nerfstudio's own identifier for the image, so it is exact.

    Comparing file contents does NOT work here, which is worth recording
    because it looks like it should. ``copy_images_list`` copies each file into
    place and then re-encodes it: its ffmpeg downscale chain writes level 0
    back over ``images/`` at ``-q:v 2``, so every processed frame is a lossy
    re-encode of its source and never byte-identical to it.

    Pairing by sort position would work today and fail silently the day
    nerfstudio changes its traversal, attaching the wrong mask to every frame
    with nothing visibly wrong in the output.

    Masks are downscaled one ffmpeg call per file with nearest-neighbour
    sampling: one call per file because the image2 sequence is what broke the
    image pyramid, and nearest-neighbour because interpolating a binary mask
    invents partial coverage along every edge.
    """
    transforms_path = workspace / "transforms.json"
    meta = json.loads(transforms_path.read_text())
    frames = meta.get("frames") or []
    if not frames:
        raise StageError(f"{transforms_path} lists no frames; cannot attach masks.")

    model_dir = workspace / cfg.colmap_model_path
    if not model_dir.is_dir():
        # skip_colmap = false puts nerfstudio's own reconstruction one level
        # deeper, under colmap/.
        model_dir = workspace / "colmap" / cfg.colmap_model_path
    colmap_names = read_colmap_image_names(model_dir)

    masks_dir = workspace / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted(
        int(p.name.split("_")[-1])
        for p in workspace.glob("images_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    )
    for level in levels:
        (workspace / f"masks_{level}").mkdir(parents=True, exist_ok=True)

    unmatched: List[str] = []
    unmasked: List[str] = []
    attached = 0
    for frame in frames:
        name = Path(frame["file_path"]).name
        image_id = frame.get("colmap_im_id")
        if image_id is None:
            unmatched.append(f"{name} (no colmap_im_id)")
            continue
        original = colmap_names.get(image_id)
        if original is None:
            unmatched.append(f"{name} (colmap_im_id {image_id} not in the model)")
            continue
        source = cfg.images_path / original
        if not source.is_file():
            unmatched.append(f"{name} -> {original} (not under the dataset)")
            continue
        mask = cfg.mask_for(source)
        if mask is None:
            unmasked.append(f"{name} <- {source.name}")
            continue
        target = masks_dir / f"{Path(name).stem}{mask.suffix}"
        shutil.copy2(mask, target)
        for level in levels:
            run_command(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-noautorotate",
                    "-i", str(target),
                    "-vf", f"scale=iw/{level}:ih/{level}:flags=neighbor",
                    str(workspace / f"masks_{level}" / target.name),
                ],
                "ffmpeg (mask downscale)",
                dry_run=False,
            )
        frame["mask_path"] = f"masks/{target.name}"
        attached += 1

    if unmatched:
        shown = ", ".join(unmatched[:3])
        raise StageError(
            f"{len(unmatched)} of {len(frames)} processed frames could not be "
            f"mapped back to a source photograph: {shown}.\n"
            f"  The mapping comes from each frame's colmap_im_id resolved "
            f"against {model_dir}/images.bin.\n"
            f"  Masks cannot be attached safely: pairing them by position "
            f"instead would silently mask the wrong frames."
        )
    if unmasked:
        shown = ", ".join(unmasked[:3])
        raise StageError(
            f"{len(unmasked)} of {len(frames)} photographs have no paired mask "
            f"({', '.join(sorted(cfg.mask_extensions))} with the same "
            f"filename stem): {shown}.\n"
            f"  A partially masked dataset trains against inconsistent "
            f"supervision. Supply the missing masks, or set "
            f"[dataset] use_masks = false to reconstruct the full scene."
        )

    transforms_path.write_text(json.dumps(meta, indent=2))
    return [
        f"masks             : {attached} attached, "
        f"downscaled to {levels or ['none']}"
    ]


def verify_downscale_pyramid(
    workspace: Path, allow_full_resolution: bool = False
) -> List[str]:
    """Prove ns-process-data actually built the downscaled image pyramid.

    ns-process-data downscales with a SINGLE ffmpeg ``image2`` sequence per
    level (``frame_%05d`` plus the suffix of the first copied file, see
    nerfstudio ``process_data/process_data_utils.py``, ``copy_images_list``).
    An image2 sequence stops at the first index it cannot open, and ffmpeg
    still exits 0. So any input set whose file extensions are not uniform, for
    example RGB ``.jpg`` frames interleaved with ``.png`` masks, yields an
    ``images_N/`` directory holding exactly the frames before the first
    extension change, with no error anywhere.

    Nothing downstream notices. nerfstudio's dataparser probes the pyramid by
    testing ONE filename (``nerfstudio_dataparser.py``, ``_get_fname``), so a
    pyramid that is one file deep reads as absent, the auto downscale factor
    collapses to 1, and training runs at native resolution. On 2026-08-17 that
    put 15 MP images through a 6 GB card: it survived to exactly step 3000,
    where splatfacto's ``resolution_schedule`` ends and rasterisation jumps to
    full resolution, and died there with a CUDA OOM that named none of this.

    The check is against ``transforms.json`` rather than a file count, because
    a count cannot tell a mask set apart from a missing frame. Only the frames
    the reconstruction actually references have to resolve.

    Returns the log lines to emit. Raises StageError if any level is partial,
    or if there is no pyramid at all and the images are large enough that
    nerfstudio would silently train at native resolution.
    """
    transforms = workspace / "transforms.json"
    if not transforms.is_file():
        raise StageError(f"{transforms} does not exist; nothing to verify.")
    try:
        meta = json.loads(transforms.read_text())
    except (OSError, ValueError) as exc:
        raise StageError(f"{transforms} could not be read as JSON: {exc}") from exc

    frames = meta.get("frames") or []
    if not frames:
        raise StageError(f"{transforms} lists no frames.")
    names = [Path(frame["file_path"]).name for frame in frames]

    # Longest edge of the registered frames. Per-frame intrinsics win over the
    # top-level ones, matching how the dataparser reads them.
    max_edge = 0
    for frame in frames:
        for key in ("w", "h"):
            value = frame.get(key, meta.get(key))
            if value:
                max_edge = max(max_edge, int(value))

    levels = sorted(
        int(p.name.split("_")[-1])
        for p in workspace.glob("images_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    )

    if not levels:
        if max_edge > MAX_AUTO_RESOLUTION and not allow_full_resolution:
            raise StageError(
                f"no downscaled image directories in {workspace}, and the "
                f"registered frames are {max_edge}px on the longest edge.\n"
                f"  nerfstudio only downscales to <= {MAX_AUTO_RESOLUTION}px "
                f"when a pyramid already exists on disk; with none it trains "
                f"at native resolution,\n"
                f"  which caches every training image on the GPU at full size "
                f"and typically dies mid-run with a CUDA OOM.\n"
                f"  Re-run the process stage with ffmpeg available, or pass "
                f"--allow-full-resolution if that is genuinely what you want."
            )
        return [f"downscale pyramid : none (frames are {max_edge}px, allowed)"]

    lines: List[str] = []
    for level in levels:
        directory = workspace / f"images_{level}"
        present = {p.name for p in directory.iterdir() if p.is_file()}
        missing = [name for name in names if name not in present]
        if missing:
            shown = ", ".join(missing[:3])
            more = f", and {len(missing) - 3} more" if len(missing) > 3 else ""
            suffixes = sorted({Path(name).suffix.lower() for name in names})
            hint = (
                "  The registered frames use a single extension, so the usual "
                "cause is an ffmpeg failure mid-sequence.\n"
                if len(suffixes) == 1
                else f"  The input set mixes extensions ({', '.join(suffixes)}), "
                "which is exactly what breaks ns-process-data's image2 "
                "sequence.\n  Separate them (masks belong in masks/, not "
                "images/) and re-run the process stage.\n"
            )
            raise StageError(
                f"{directory} is missing {len(missing)} of {len(names)} "
                f"registered frames: {shown}{more}.\n"
                f"{hint}"
                f"  A partial pyramid is worse than none: nerfstudio probes it "
                f"with one filename, so it reads as absent and training falls "
                f"back to native resolution."
            )
        lines.append(f"images_{level:<11}: {len(names)}/{len(names)} registered frames present")
    return lines


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


def _check_ffmpeg(problems: List[str]) -> None:
    """ns-process-data shells out to ffmpeg; nerfstudio itself does not check.

    Without ffmpeg the copy still runs and ``transforms.json`` still appears,
    so every existing artefact check passes. What silently does not happen is
    the downscale pyramid, which is what keeps training inside a small card's
    memory. NUC1 shipped without ffmpeg and this is how it was found.
    """
    found = shutil.which("ffmpeg")
    if found is None:
        problems.append(
            "'ffmpeg' is not on PATH. ns-process-data shells out to it to "
            "build the downscaled image\n"
            "  pyramid, and does not check for it: without ffmpeg the stage "
            "still reports success and still\n"
            "  writes transforms.json, but no images_2/4/8 are produced and "
            "training silently runs at native\n"
            f"  resolution.\n  {_HINT}"
        )
    else:
        log(f"{'ffmpeg':<18}: {found}")


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
        _check_ffmpeg(problems)
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
    parser.add_argument(
        "--allow-full-resolution",
        action="store_true",
        help=(
            "permit a [process] result with no downscaled image pyramid. Off "
            "by default because nerfstudio then trains at native resolution, "
            "which on a small card dies with a CUDA OOM part-way through the "
            "run. A PARTIAL pyramid is always an error and this does not "
            "override it."
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
    photographs = [p for p in images if cfg.is_photograph(p)]
    camera_dirs = sorted({
        p.parent.relative_to(cfg.images_path).as_posix() for p in photographs
    })
    log(f"images found      : {len(photographs)} in "
        + (f"{len(camera_dirs)} subdirectories" if camera_dirs != ["."] else "one flat directory"))
    if len(photographs) != len(images):
        excluded = len(images) - len(photographs)
        log(f"excluded from sfm : {excluded} file(s) "
            + ("used as masks" if cfg.use_masks else "ignored"))
    if not photographs:
        raise StageError(
            f"every image under {cfg.images_path} was excluded by [dataset] "
            f"image_extensions / mask_extensions in {cfg.source}."
        )

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
    effective_downscale: int = cfg.downscale_factor

    for stage in stages:
        stage_start = time.time()
        log_banner(f"stage {STAGES.index(stage) + 1}/{len(STAGES)}: {stage}")

        if stage == "colmap":
            if not dry_run:
                workspace.mkdir(parents=True, exist_ok=True)
                if cfg.filters_dataset_files:
                    staged = sfm_input_path(cfg, workspace)
                    staged_files = stage_sfm_inputs(cfg, staged)
                    log(f"sfm input tree    : {staged} ({len(staged_files)} photographs)")
            run_command(colmap_command(cfg, workspace), "colmap", dry_run)
            sparse = workspace / "sparse" / "0"
            if not dry_run and not (sparse / "cameras.bin").is_file():
                raise StageError(
                    f"colmap reported success but produced no sparse model at "
                    f"{sparse}. Nothing downstream can use this workspace."
                )

        elif stage == "process":
            # Rebuilt here too, so --from-stage process works on a workspace
            # whose staged tree was cleaned up or never built.
            if not dry_run and cfg.filters_dataset_files:
                staged = sfm_input_path(cfg, workspace)
                staged_files = stage_sfm_inputs(cfg, staged)
                log(f"sfm input tree    : {staged} ({len(staged_files)} photographs)")
            run_command(process_command(cfg, workspace), "ns-process-data", dry_run)
            transforms = workspace / "transforms.json"
            if not dry_run and not transforms.is_file():
                raise StageError(
                    f"ns-process-data reported success but {transforms} does not "
                    f"exist. ns-train has nothing to read."
                )
            if not dry_run:
                log(f"transforms.json   : {transforms.stat().st_size} bytes")
                for line in verify_downscale_pyramid(
                    workspace, allow_full_resolution=args.allow_full_resolution
                ):
                    log(line)
                if cfg.use_masks:
                    for line in attach_masks(cfg, workspace):
                        log(line)
                # ns-process-data has copied the photographs into images/ and
                # nothing downstream reads the staged tree, so drop it. On a
                # filesystem that supports neither symlinks nor hardlinks it is
                # a full copy of the dataset (exFAT, which is what an external
                # data drive usually is), and leaving one behind per run adds up
                # fast. --from-stage process rebuilds it, so this is not a
                # one-way door.
                staged = sfm_input_path(cfg, workspace)
                if cfg.filters_dataset_files and staged.is_dir():
                    shutil.rmtree(staged)
                    log(f"sfm input tree    : removed {staged.name} (no longer needed)")

        elif stage == "train":
            if not dry_run and cfg.downscale_factor > 1:
                # nerfstudio does not generate a level that is missing: it just
                # fails to open the images, several minutes in.
                level = workspace / f"images_{cfg.downscale_factor}"
                if not level.is_dir():
                    raise StageError(
                        f"[train] downscale_factor is "
                        f"{cfg.downscale_factor} but {level} does not exist.\n"
                        f"  nerfstudio reads a pinned level straight off disk "
                        f"and does not build a missing one.\n"
                        f"  Re-run the process stage, or set downscale_factor "
                        f"to 0 to let nerfstudio pick from what is there."
                    )
            train_started = time.time()
            if not dry_run:
                effective_downscale = resolve_downscale_factor(cfg, workspace)
            log(f"train resolution  : "
                + (f"downscale {cfg.downscale_factor} (pinned)"
                   if cfg.downscale_factor
                   else f"downscale {effective_downscale or '?'} "
                        f"(auto, resolved from the pyramid on disk)"))
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
            # Export can run without train in the same invocation
            # (--stage export), so resolve it here too rather than relying on
            # the train stage having set it.
            if not dry_run and not effective_downscale:
                effective_downscale = resolve_downscale_factor(cfg, workspace)
            filename = export_filename(cfg, run_id, effective_downscale)
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
