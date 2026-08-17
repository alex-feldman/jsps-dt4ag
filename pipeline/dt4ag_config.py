"""Configuration loader for the jsps-dt4ag reconstruction pipeline.

Everything the pipeline used to have typed into notebook cells lives in an INI
file instead. This module reads that file, validates it, and exposes the values
as typed attributes.

Why INI and not TOML or YAML: the pipeline environment runs Python 3.10, where
``tomllib`` is not in the standard library, and this repository deliberately
carries no third-party configuration dependency. ``configparser`` is stdlib on
every supported Python.

Design rule: fail loudly. A missing key, an unparseable value or a path that
does not exist raises ``ConfigError`` naming both the key and the config file.
The pipeline has a history of silent no-ops (see the rgb-mask defect in the
2026-08-06 inventory); nothing here is allowed to repeat that pattern.

Typical use::

    from dt4ag_config import load_config
    cfg = load_config("../configs/my-run.ini")
    project_id = cfg.make_run_id()

Run ``python dt4ag_config.py <path-to.ini>`` to validate a config from the
shell and print the resolved values.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import configparser

__all__ = [
    "ConfigError",
    "Dt4agConfig",
    "load_config",
    "detect_colmap_version",
]


class ConfigError(Exception):
    """Raised for any missing, malformed or unsatisfiable configuration value."""


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


# --------------------------------------------------------------------------
# low-level typed accessors
# --------------------------------------------------------------------------

def _require_section(parser: configparser.ConfigParser, section: str, source: Path) -> None:
    if not parser.has_section(section):
        raise ConfigError(
            f"missing required section [{section}] in config file {source}"
        )


def _get_str(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    source: Path,
    default: Optional[str] = None,
    allow_empty: bool = False,
) -> str:
    _require_section(parser, section, source)
    if not parser.has_option(section, key):
        if default is None:
            raise ConfigError(
                f"missing required key '{key}' in section [{section}] "
                f"of config file {source}"
            )
        return default
    value = parser.get(section, key).strip()
    if not value and not allow_empty:
        if default is None:
            raise ConfigError(
                f"key '{key}' in section [{section}] of config file {source} "
                f"is empty, but a value is required"
            )
        return default
    return value


def _parse_extensions(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    source: Path,
) -> frozenset:
    """Read a comma-separated extension list, normalised to lowercase '.ext'.

    Empty means "no opinion", which for image_extensions is every supported
    image type and for mask_extensions is no masks at all. Accepts 'jpg' and
    '.JPG' alike, because requiring the leading dot is a rule nobody would
    remember and the error would arrive four stages later.
    """
    raw = _get_str(parser, section, key, source, "", allow_empty=True)
    if not raw:
        return frozenset()
    parsed = set()
    for part in raw.replace(",", " ").split():
        ext = part.strip().lower()
        if not ext:
            continue
        parsed.add(ext if ext.startswith(".") else f".{ext}")
    return frozenset(parsed)


def _get_int(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    source: Path,
    default: Optional[int] = None,
) -> int:
    raw = _get_str(
        parser,
        section,
        key,
        source,
        default=None if default is None else str(default),
    )
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"key '{key}' in section [{section}] of config file {source} "
            f"must be an integer, got {raw!r}"
        ) from exc


def _get_bool(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    source: Path,
    default: Optional[bool] = None,
) -> bool:
    raw = _get_str(
        parser,
        section,
        key,
        source,
        default=None if default is None else ("true" if default else "false"),
    )
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(
        f"key '{key}' in section [{section}] of config file {source} "
        f"must be a boolean (true/false), got {raw!r}"
    )


# --------------------------------------------------------------------------
# COLMAP version detection
# --------------------------------------------------------------------------

_VERSION_RE = re.compile(r"COLMAP\s+(\d+)\.(\d+)(?:\.(\d+))?", re.IGNORECASE)


def detect_colmap_version() -> Optional[str]:
    """Return the installed COLMAP version as a compact string, e.g. ``312``.

    Tries ``colmap --version`` first, then the bare ``colmap`` banner, which is
    where COLMAP 3.x prints its version when given no subcommand. Returns
    ``None`` if COLMAP is not on PATH or nothing parseable comes back; callers
    are expected to fall back to the configured value rather than guess.
    """
    if shutil.which("colmap") is None:
        return None
    for argv in (["colmap", "--version"], ["colmap"]):
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = _VERSION_RE.search((proc.stdout or "") + "\n" + (proc.stderr or ""))
        if match:
            major, minor, patch = match.group(1), match.group(2), match.group(3)
            return f"{major}{minor}{patch or ''}"
    return None


# --------------------------------------------------------------------------
# the config object
# --------------------------------------------------------------------------

@dataclass
class Dt4agConfig:
    """Resolved, validated pipeline configuration."""

    source: Path

    # [paths]
    data_root: Path
    datasets_dir: Path
    colmap_dir: Path
    outputs_dir: Path
    exports_dir: Path

    # [dataset]
    images_path: Path
    images_rel: Path
    image_extensions: frozenset
    mask_extensions: frozenset
    masks_path: Path
    use_masks: bool

    # [run]
    run_id_prefix: str
    run_date: str
    run_count: str
    colmap_version: str
    run_log: Path

    # [colmap]
    colmap_data_type: str
    colmap_single_camera: int
    colmap_single_camera_per_folder: int
    colmap_dense: int
    colmap_extra_args: str

    # [nerfstudio]
    scene_type: str
    video_path: str
    skip_colmap: bool
    colmap_model_path: str

    # [train]
    train_method: str
    max_num_iterations: int
    downscale_factor: int
    use_scale_regularization: bool
    background_color: str
    quit_on_train_completion: bool
    max_log_size: int

    # [export]
    export_format: str
    env_label: str
    platform_label: str
    export_3dgs: bool
    gauss_to_pc_script: str

    _resolved_run_id: Optional[str] = field(default=None, repr=False)

    # -- derived paths -----------------------------------------------------

    @property
    def colmap_workspace_parent(self) -> Path:
        """``<colmap_dir>/<images_rel>``, the per-dataset COLMAP workspace root."""
        return self.colmap_dir / self.images_rel

    @property
    def output_parent(self) -> Path:
        """``<outputs_dir>/<images_rel>``, the per-dataset nerfstudio output root."""
        return self.outputs_dir / self.images_rel

    def colmap_workspace(self, run_id: str) -> Path:
        return self.colmap_workspace_parent / run_id

    def output_dir(self, run_id: str) -> Path:
        return self.output_parent / run_id

    # -- run identity ------------------------------------------------------

    @property
    def filters_dataset_files(self) -> bool:
        """True when the dataset directory holds files SfM must not see.

        When false the pipeline points COLMAP and ns-process-data straight at
        the dataset directory, exactly as it always has. Existing configs set
        neither key, so they take that path and nothing about them changes.
        """
        return bool(self.image_extensions or self.mask_extensions)

    def is_photograph(self, path: Path) -> bool:
        """Is this file one of the photographs the reconstruction is built from?"""
        suffix = path.suffix.lower()
        if self.image_extensions:
            return suffix in self.image_extensions
        return suffix not in self.mask_extensions

    def is_mask(self, path: Path) -> bool:
        return path.suffix.lower() in self.mask_extensions

    def mask_for(self, photograph: Path) -> Optional[Path]:
        """The mask paired with a photograph, or None.

        Pairing is by filename stem at the same position under ``masks_path``.
        Both real layouts fall out of that one rule: with ``mask_subpath``
        unset, ``masks_path`` IS the image directory and the mask is the file
        beside the photograph, which is what a tool writing next to its input
        produces. With it set, the masks live in a parallel tree of identical
        shape, which is what a batch segmentation run produces.

        Anything looser would need a mapping file kept by hand.
        """
        try:
            relative = photograph.relative_to(self.images_path)
        except ValueError:
            relative = Path(photograph.name)
        for extension in sorted(self.mask_extensions):
            candidate = self.masks_path / relative.with_suffix(extension)
            if candidate.is_file():
                return candidate
        return None

    def resolve_colmap_data_type(self) -> str:
        """Return the ``--data_type`` value for ``colmap automatic_reconstructor``.

        ``auto`` reproduces the notebook's original heuristic: a directory whose
        name mentions "frames" came from a video, anything else is treated as
        individual images.
        """
        configured = self.colmap_data_type.lower()
        if configured != "auto":
            return configured
        name = self.images_path.name.lower()
        return "video" if "frames" in name else "individual"

    def resolve_colmap_version(self) -> str:
        """``[run] colmap_version`` when set, else the installed COLMAP version.

        Config wins on purpose. This value goes into the run id, and the whole
        point of pinning a run id is to address a workspace produced by an
        earlier run, possibly by an earlier COLMAP. Detecting first would
        silently rewrite the pinned id on a machine that has since upgraded
        COLMAP, and the pipeline would build a new workspace next to the one
        the operator meant to reuse. Leave the key blank for the normal case of
        recording whatever COLMAP is actually installed.
        """
        if self.colmap_version:
            return self.colmap_version
        detected = detect_colmap_version()
        if detected:
            return detected
        if not self.colmap_version:
            raise ConfigError(
                "could not determine the COLMAP version by running 'colmap', and "
                f"key 'colmap_version' in section [run] of config file "
                f"{self.source} is empty. Install COLMAP or set colmap_version "
                "explicitly so the run id records real provenance."
            )
        return self.colmap_version

    def _existing_run_ids(self, date: str) -> List[str]:
        prefix = f"{self.run_id_prefix}_{date}-"
        found: List[str] = []
        for parent in (self.colmap_workspace_parent, self.output_parent):
            if not parent.is_dir():
                continue
            for entry in parent.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix):
                    found.append(entry.name)
        return found

    def next_run_count(self, date: str) -> str:
        """Smallest unused two-digit run counter for ``date`` on this dataset.

        Looks at the COLMAP workspace and nerfstudio output directories for this
        dataset, both of which are named after the run id, and returns one more
        than the highest counter already present.
        """
        prefix = f"{self.run_id_prefix}_{date}-"
        highest = 0
        for name in self._existing_run_ids(date):
            tail = name[len(prefix):]
            counter = tail.split("-", 1)[0]
            if counter.isdigit():
                highest = max(highest, int(counter))
        return f"{highest + 1:02d}"

    def make_run_id(self) -> str:
        """Build (and memoise) the run id: ``<prefix>_<date>-<count>-<colmapver>``.

        Each component comes from config when set and is derived automatically
        when left blank: the date from today's clock, the count by inspecting
        existing output directories, the COLMAP version by invoking ``colmap``.
        """
        if self._resolved_run_id is not None:
            return self._resolved_run_id
        date = self.run_date or _dt.date.today().strftime("%y%m%d")
        count = self.run_count or self.next_run_count(date)
        version = self.resolve_colmap_version()
        self._resolved_run_id = f"{self.run_id_prefix}_{date}-{count}-{version}"
        return self._resolved_run_id

    # -- run log -----------------------------------------------------------

    def append_run_log(self, run_id: str, **extra: object) -> Path:
        """Append one row describing this run to the CSV run log.

        Dependency-free by design: stdlib ``csv``, header written on creation.
        """
        self.run_log.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "timestamp",
            "run_id",
            "dataset",
            "images_path",
            "colmap_version",
            "train_method",
            "max_num_iterations",
            "downscale_factor",
            "masks",
            "config_file",
            "note",
        ]
        row = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "dataset": str(self.images_rel),
            "images_path": str(self.images_path),
            "colmap_version": run_id.rsplit("-", 1)[-1],
            "train_method": self.train_method,
            "max_num_iterations": self.max_num_iterations,
            "downscale_factor": self.downscale_factor or "auto",
            "masks": "used" if self.use_masks else "none",
            "config_file": str(self.source),
            "note": "",
        }
        row.update({k: v for k, v in extra.items() if k in columns})
        is_new = not self.run_log.exists()
        if not is_new:
            # An existing log was written under whatever columns were current
            # then. Keep writing under ITS header rather than this one: a row
            # with more fields than the header is a corrupt CSV, and silently
            # corrupting the provenance log is worse than omitting a column
            # from it. Delete the log to start one with the current columns.
            with self.run_log.open("r", newline="", encoding="utf-8") as handle:
                existing = next(csv.reader(handle), None)
            if existing:
                columns = existing
        with self.run_log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        return self.run_log

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"config file        : {self.source}",
            f"data_root          : {self.data_root}",
            f"datasets_dir       : {self.datasets_dir}",
            f"colmap_dir         : {self.colmap_dir}",
            f"outputs_dir        : {self.outputs_dir}",
            f"exports_dir        : {self.exports_dir}",
            f"images_path        : {self.images_path}",
            f"images_rel         : {self.images_rel}",
            f"image_extensions   : "
            f"{', '.join(sorted(self.image_extensions)) or '(any)'}",
            f"mask_extensions    : "
            f"{', '.join(sorted(self.mask_extensions)) or '(none)'}",
            f"masks_path         : "
            f"{self.masks_path if self.masks_path != self.images_path else '(beside the images)'}",
            f"use_masks          : {self.use_masks}",
            f"run_id_prefix      : {self.run_id_prefix}",
            f"run_date           : {self.run_date or '(auto: today)'}",
            f"run_count          : {self.run_count or '(auto: increment)'}",
            f"colmap_version     : {self.colmap_version or '(auto: detect)'}",
            f"run_log            : {self.run_log}",
            f"colmap data_type   : {self.colmap_data_type} "
            f"-> {self.resolve_colmap_data_type()}",
            f"scene_type         : {self.scene_type}",
            f"train_method       : {self.train_method}",
            f"max_num_iterations : {self.max_num_iterations}",
            f"export_format      : {self.export_format}",
            f"export_3dgs        : {self.export_3dgs}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _expand(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def load_config(path, validate_paths: bool = True) -> Dt4agConfig:
    """Read, validate and return the pipeline configuration.

    Args:
        path: path to the INI file.
        validate_paths: when True (the default) the data root, the datasets
            directory and the dataset image directory must all exist, and a
            missing one raises ``ConfigError``. Set False only for tests or for
            inspecting a config on a machine without the data drive attached.

    Raises:
        ConfigError: for a missing file, a missing section or key, a value of
            the wrong type, or a required path that does not exist.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise ConfigError(f"config file not found: {source}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    try:
        with source.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except configparser.Error as exc:
        raise ConfigError(f"could not parse config file {source}: {exc}") from exc

    for section in ("paths", "dataset", "run", "colmap", "nerfstudio", "train", "export"):
        _require_section(parser, section, source)

    # [paths]
    data_root = _expand(_get_str(parser, "paths", "data_root", source))
    if not data_root.is_absolute():
        data_root = (source.parent / data_root).resolve()
    datasets_dir = data_root / _get_str(parser, "paths", "datasets_dirname", source, "datasets")
    colmap_dir = data_root / _get_str(parser, "paths", "colmap_dirname", source, "colmap")
    outputs_dir = data_root / _get_str(parser, "paths", "outputs_dirname", source, "outputs")
    exports_dir = data_root / _get_str(parser, "paths", "exports_dirname", source, "exports")

    # [dataset]
    images_rel_raw = _get_str(parser, "dataset", "images_subpath", source)
    images_rel = Path(images_rel_raw)
    if images_rel.is_absolute():
        raise ConfigError(
            f"key 'images_subpath' in section [dataset] of config file {source} "
            f"must be relative to the datasets directory, got the absolute path "
            f"{images_rel}"
        )
    images_path = datasets_dir / images_rel

    image_extensions = _parse_extensions(
        parser, "dataset", "image_extensions", source
    )
    mask_extensions = _parse_extensions(
        parser, "dataset", "mask_extensions", source
    )
    overlap = image_extensions & mask_extensions
    if overlap:
        raise ConfigError(
            f"keys 'image_extensions' and 'mask_extensions' in section "
            f"[dataset] of config file {source} both list "
            f"{', '.join(sorted(overlap))}. A file cannot be both a "
            f"photograph and a mask."
        )
    mask_subpath_raw = _get_str(
        parser, "dataset", "mask_subpath", source, "", allow_empty=True
    )
    if mask_subpath_raw:
        mask_subpath = Path(mask_subpath_raw)
        masks_path = (
            mask_subpath if mask_subpath.is_absolute() else datasets_dir / mask_subpath
        )
    else:
        masks_path = images_path

    use_masks = _get_bool(parser, "dataset", "use_masks", source, False)
    if use_masks and not mask_extensions:
        raise ConfigError(
            f"key 'use_masks' in section [dataset] of config file {source} is "
            f"true but 'mask_extensions' is empty, so no file would ever be "
            f"treated as a mask."
        )

    if validate_paths:
        for label, candidate, key in (
            ("data root", data_root, "[paths] data_root"),
            ("datasets directory", datasets_dir, "[paths] datasets_dirname"),
            ("dataset image directory", images_path, "[dataset] images_subpath"),
            *(
                [("mask directory", masks_path, "[dataset] mask_subpath")]
                if mask_subpath_raw
                else []
            ),
        ):
            if not candidate.is_dir():
                raise ConfigError(
                    f"{label} does not exist: {candidate}\n"
                    f"  set by {key} in config file {source}\n"
                    f"  (if the data drive is not mounted, mount it; this "
                    f"pipeline will not silently continue with a missing path)"
                )

    # [run]
    run_id_prefix = _get_str(parser, "run", "id_prefix", source, "run")
    run_date = _get_str(parser, "run", "date", source, "", allow_empty=True)
    if run_date and not re.fullmatch(r"\d{6}", run_date):
        raise ConfigError(
            f"key 'date' in section [run] of config file {source} must be six "
            f"digits in yymmdd format or empty for today, got {run_date!r}"
        )
    run_count = _get_str(parser, "run", "run_count", source, "", allow_empty=True)
    if run_count and not run_count.isdigit():
        raise ConfigError(
            f"key 'run_count' in section [run] of config file {source} must be "
            f"digits or empty for auto-increment, got {run_count!r}"
        )
    colmap_version = _get_str(parser, "run", "colmap_version", source, "", allow_empty=True)
    run_log_raw = _get_str(parser, "run", "log_file", source, "run-log.csv")
    run_log = _expand(run_log_raw)
    if not run_log.is_absolute():
        run_log = data_root / run_log

    # [colmap]
    colmap_data_type = _get_str(parser, "colmap", "data_type", source, "auto").lower()
    if colmap_data_type not in ("auto", "individual", "video"):
        raise ConfigError(
            f"key 'data_type' in section [colmap] of config file {source} must be "
            f"one of auto, individual, video; got {colmap_data_type!r}"
        )
    colmap_single_camera = _get_int(parser, "colmap", "single_camera", source, 1)
    colmap_single_camera_per_folder = _get_int(
        parser, "colmap", "single_camera_per_folder", source, 1
    )
    colmap_dense = _get_int(parser, "colmap", "dense", source, 0)
    colmap_extra_args = _get_str(parser, "colmap", "extra_args", source, "", allow_empty=True)

    # [nerfstudio]
    scene_type = _get_str(parser, "nerfstudio", "scene_type", source, "images").lower()
    if scene_type not in ("images", "video"):
        raise ConfigError(
            f"key 'scene_type' in section [nerfstudio] of config file {source} "
            f"must be images or video, got {scene_type!r}"
        )
    video_path = _get_str(parser, "nerfstudio", "video_path", source, "", allow_empty=True)
    if scene_type == "video" and not video_path:
        raise ConfigError(
            f"key 'video_path' in section [nerfstudio] of config file {source} is "
            f"required when scene_type = video"
        )
    skip_colmap = _get_bool(parser, "nerfstudio", "skip_colmap", source, True)
    colmap_model_path = _get_str(
        parser, "nerfstudio", "colmap_model_path", source, "sparse/0"
    )

    # [train]
    train_method = _get_str(parser, "train", "method", source, "splatfacto")
    max_num_iterations = _get_int(parser, "train", "max_num_iterations", source, 30000)
    if max_num_iterations <= 0:
        raise ConfigError(
            f"key 'max_num_iterations' in section [train] of config file {source} "
            f"must be positive, got {max_num_iterations}"
        )
    # 0 means "let nerfstudio choose", which it does by probing the downscale
    # pyramid on disk. Pinning it makes the training resolution a recorded
    # input of the run rather than a property of which files happen to be
    # present, which is what a reconstruction anyone has to reproduce needs.
    downscale_factor = _get_int(parser, "train", "downscale_factor", source, 0)
    if downscale_factor < 0 or (downscale_factor > 1 and downscale_factor & (downscale_factor - 1)):
        raise ConfigError(
            f"key 'downscale_factor' in section [train] of config file {source} "
            f"must be 0 (auto), 1 (native), or a power of two, got "
            f"{downscale_factor}"
        )
    use_scale_regularization = _get_bool(
        parser, "train", "use_scale_regularization", source, True
    )
    background_color = _get_str(parser, "train", "background_color", source, "random")
    quit_on_train_completion = _get_bool(
        parser, "train", "quit_on_train_completion", source, False
    )
    max_log_size = _get_int(parser, "train", "max_log_size", source, 0)

    # [export]
    export_format = _get_str(parser, "export", "format", source, "gaussian-splat")
    env_label = _get_str(parser, "export", "env_label", source, "env")
    platform_label = _get_str(parser, "export", "platform_label", source, sys.platform)
    export_3dgs = _get_bool(parser, "export", "export_3dgs", source, False)
    gauss_to_pc_script = _get_str(
        parser, "export", "gauss_to_pc_script", source, "", allow_empty=True
    )
    if export_3dgs and not gauss_to_pc_script:
        raise ConfigError(
            f"key 'gauss_to_pc_script' in section [export] of config file {source} "
            f"is required when export_3dgs = true"
        )

    return Dt4agConfig(
        source=source,
        data_root=data_root,
        datasets_dir=datasets_dir,
        colmap_dir=colmap_dir,
        outputs_dir=outputs_dir,
        exports_dir=exports_dir,
        images_path=images_path,
        images_rel=images_rel,
        image_extensions=image_extensions,
        mask_extensions=mask_extensions,
        masks_path=masks_path,
        use_masks=use_masks,
        run_id_prefix=run_id_prefix,
        run_date=run_date,
        run_count=run_count,
        colmap_version=colmap_version,
        run_log=run_log,
        colmap_data_type=colmap_data_type,
        colmap_single_camera=colmap_single_camera,
        colmap_single_camera_per_folder=colmap_single_camera_per_folder,
        colmap_dense=colmap_dense,
        colmap_extra_args=colmap_extra_args,
        scene_type=scene_type,
        video_path=video_path,
        skip_colmap=skip_colmap,
        colmap_model_path=colmap_model_path,
        train_method=train_method,
        max_num_iterations=max_num_iterations,
        downscale_factor=downscale_factor,
        use_scale_regularization=use_scale_regularization,
        background_color=background_color,
        quit_on_train_completion=quit_on_train_completion,
        max_log_size=max_log_size,
        export_format=export_format,
        env_label=env_label,
        platform_label=platform_label,
        export_3dgs=export_3dgs,
        gauss_to_pc_script=gauss_to_pc_script,
    )


def find_config(explicit=None, start=None) -> Path:
    """Locate a config file.

    Order of precedence: the explicit argument, then the ``DT4AG_CONFIG``
    environment variable, then ``configs/example.ini`` found by walking up from
    ``start`` (default: the current working directory). Raises ``ConfigError``
    if nothing is found, rather than returning None for a caller to trip over.
    """
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get("DT4AG_CONFIG")
    if from_env:
        return Path(from_env).expanduser()
    here = Path(start or Path.cwd()).resolve()
    for candidate_dir in [here, *here.parents]:
        candidate = candidate_dir / "configs" / "example.ini"
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "no config file given, DT4AG_CONFIG is unset, and no configs/example.ini "
        f"was found by walking up from {here}. Pass a path explicitly."
    )


def _main(argv: List[str]) -> int:
    try:
        path = find_config(argv[1] if len(argv) > 1 else None)
        cfg = load_config(path)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1
    print(cfg.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
