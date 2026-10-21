#!/usr/bin/env python3
"""Read-only integrity check of exported Gaussian-splat PLY files.

For each file it reports the vertex count, size and sha256, counts non-finite
values per property, and prints a few plain statistics: opacity (after the
sigmoid), the largest-axis scale (after exp) and the 1-99 percentile extent of
the Gaussian centers. With ``--preview {png}`` it also writes a crude picture of
the centers. This is an integrity check, not a quality test, and the preview is
not a render.

    python pipeline/scripts/ply-check/ply-check.py {ply} [{ply} ...] [--json] [--preview {png}]

Exit codes: 0 when every file is sound, 1 when a file has non-finite values, no
vertices, a truncated body or cannot be read, 2 for a usage error.

Needs numpy; ``--preview`` also needs matplotlib. Both are in the pipeline
environment. See README.md in this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EXIT_PROBLEM = 1
EXIT_USAGE = 2

#: PLY scalar type names (both spellings) to numpy type codes.
PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

#: Spherical-harmonics DC term to RGB: the constant C0 = 1 / (2 * sqrt(pi)).
SH_C0 = 0.28209479177387814

#: Most centers drawn in a preview, so the picture stays quick to make.
PREVIEW_MAX_POINTS = 250_000

#: A header longer than this is not a PLY header.
MAX_HEADER_BYTES = 1 << 20


class PlyError(Exception):
    """The file is not a PLY this tool can read."""


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def parse_header(handle) -> Tuple[int, List[Tuple[str, str]], int]:
    """Read a binary little-endian PLY header from an open binary file.

    Returns the vertex count, the vertex properties as ``(name, numpy type)``
    pairs, and the byte offset where the vertex data starts. The vertex element
    must come first, which is what every splat exporter writes.
    """
    first = handle.readline().decode("latin-1").strip()
    if first != "ply":
        raise PlyError("not a PLY file (the first line is not 'ply')")
    fmt = None
    count: Optional[int] = None
    props: List[Tuple[str, str]] = []
    in_vertex = False
    seen_elements = 0
    while True:
        raw = handle.readline()
        if not raw or handle.tell() > MAX_HEADER_BYTES:
            raise PlyError("no 'end_header' line found")
        line = raw.decode("latin-1").strip()
        words = line.split()
        if not words:
            continue
        if words[0] == "format":
            fmt = words[1] if len(words) > 1 else None
        elif words[0] == "element":
            seen_elements += 1
            in_vertex = len(words) >= 3 and words[1] == "vertex"
            if in_vertex:
                if seen_elements != 1:
                    raise PlyError("the vertex element is not the first element")
                try:
                    count = int(words[2])
                except ValueError:
                    raise PlyError(f"bad vertex count in {line!r}")
        elif words[0] == "property" and in_vertex:
            if len(words) != 3:
                raise PlyError(f"unsupported property line {line!r} (list properties "
                               "are not allowed on splat vertices)")
            if words[1] not in PLY_TYPES:
                raise PlyError(f"unknown property type {words[1]!r}")
            props.append((words[2], PLY_TYPES[words[1]]))
        elif words[0] == "end_header":
            break
    if fmt != "binary_little_endian":
        raise PlyError(f"unsupported format {fmt!r}; only binary_little_endian is read")
    if count is None:
        raise PlyError("no 'element vertex' line")
    if not props:
        raise PlyError("the vertex element has no properties")
    return count, props, handle.tell()


def read_ply(path: Path) -> Tuple[np.ndarray, int]:
    """The vertex records and the count the header declares.

    The array can be shorter than the count when the file is truncated; the
    caller compares the two.
    """
    with open(path, "rb") as handle:
        count, props, _ = parse_header(handle)
        dtype = np.dtype([(name, "<" + code) for name, code in props])
        return np.fromfile(handle, dtype=dtype, count=count), count


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# checking
# --------------------------------------------------------------------------

def _number(value) -> Optional[float]:
    """A plain float, or None for NaN and infinity (which JSON cannot carry)."""
    value = float(value)
    return value if math.isfinite(value) else None


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values.astype("f8"), -500, 500)))


def non_finite_counts(vertices: np.ndarray) -> Dict[str, int]:
    """Per float property, how many values are NaN or infinite. Zero counts are left out."""
    counts = {}
    for name in vertices.dtype.names:
        if vertices.dtype[name].kind == "f":
            bad = int((~np.isfinite(vertices[name])).sum())
            if bad:
                counts[name] = bad
    return counts


def _finite_rows(columns: np.ndarray) -> np.ndarray:
    return columns[np.isfinite(columns).all(axis=1)]


def center_extent(vertices: np.ndarray) -> Optional[List[float]]:
    """Size of the 1-99 percentile box around the centers, per axis (scene units)."""
    names = vertices.dtype.names
    if not all(axis in names for axis in "xyz"):
        return None
    xyz = _finite_rows(np.stack([vertices[a] for a in "xyz"], axis=1).astype("f8"))
    if len(xyz) == 0:
        return None
    low, high = np.percentile(xyz, [1, 99], axis=0)
    return [round(float(v), 2) for v in (high - low)]


def check_ply(path: Path, with_hash: bool = True) -> dict:
    """Everything this tool says about one file, as a plain dict.

    ``ok`` is False, with the reasons in ``problems``, for a file that cannot
    be read, has no vertices, has fewer vertex records than it declares, or
    holds a NaN or infinite value.
    """
    path = Path(path)
    result: dict = {"path": str(path), "name": path.name, "ok": False, "problems": []}
    try:
        result["bytes"] = path.stat().st_size
        vertices, declared = read_ply(path)
    except (OSError, PlyError, ValueError) as exc:
        result["problems"].append(f"cannot read: {exc}")
        return result
    result["vertices"] = declared
    result["properties"] = len(vertices.dtype.names)
    if with_hash:
        result["sha256"] = sha256_of(path)
    if declared == 0:
        result["problems"].append("zero vertices")
    if len(vertices) < declared:
        result["problems"].append(
            f"truncated: the header declares {declared:,} vertices but the file holds "
            f"{len(vertices):,}")
    bad = non_finite_counts(vertices)
    result["non_finite"] = bad
    if bad:
        total = sum(bad.values())
        result["problems"].append(f"{total:,} non-finite values in {', '.join(sorted(bad))}")
    names = vertices.dtype.names
    with np.errstate(all="ignore"):
        if "opacity" in names and len(vertices):
            opacity = sigmoid(vertices["opacity"])
            result["opacity_median"] = _number(np.nanmedian(opacity))
            result["opacity_share_above_half"] = _number(np.mean(opacity > 0.5))
        if all(f"scale_{i}" in names for i in range(3)) and len(vertices):
            scales = np.stack([vertices[f"scale_{i}"] for i in range(3)], axis=1).astype("f8")
            largest = _finite_rows(np.exp(np.clip(scales, -500, 500))).max(axis=1)
            if len(largest):
                result["scale_median"] = _number(np.median(largest))
                result["scale_p99"] = _number(np.percentile(largest, 99))
        if len(vertices):
            result["xyz_extent_1_99"] = center_extent(vertices)
    result["ok"] = not result["problems"]
    return result


def format_report(result: dict) -> str:
    """The human-readable block for one file."""
    lines = [result["name"]]
    if "vertices" in result:
        line = f"  vertices {result['vertices']:,}  bytes {result['bytes']:,}"
        if "sha256" in result:
            line += f"  sha256 {result['sha256']}"
        lines.append(line)
        bad = result.get("non_finite") or {}
        lines.append("  non-finite: " + (", ".join(f"{k} {v:,}" for k, v in sorted(bad.items()))
                                         if bad else "none"))
    if result.get("opacity_median") is not None:
        lines.append(f"  opacity median {result['opacity_median']:.3f}, share above 0.5: "
                     f"{result['opacity_share_above_half']:.2%}")
    if result.get("scale_median") is not None:
        lines.append(f"  largest-axis scale median {result['scale_median']:.4f}, "
                     f"p99 {result['scale_p99']:.4f}")
    if result.get("xyz_extent_1_99") is not None:
        lines.append(f"  xyz 1-99 pct extent {result['xyz_extent_1_99']} (scene units)")
    for problem in result["problems"]:
        lines.append(f"  PROBLEM: {problem}")
    lines.append("  result: " + ("ok" if result["ok"] else "FAILED"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------

def preview_points(vertices: np.ndarray, max_points: int = PREVIEW_MAX_POINTS,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    """Centers and colors for the preview, and how many Gaussians passed the filter.

    Keeps the Gaussians with opacity above 0.5 (all of them when the file has no
    opacity), drops non-finite centers, keeps the region inside the 5-95
    percentile box widened by 10 percent so far-out strays do not shrink the
    picture, and takes a seeded random sample of at most ``max_points``. Colors
    come from the spherical-harmonics DC term when it exists, else gray.
    """
    names = vertices.dtype.names
    xyz = np.stack([vertices[a] for a in "xyz"], axis=1).astype("f8")
    if all(f"f_dc_{i}" in names for i in range(3)):
        dc = np.stack([vertices[f"f_dc_{i}"] for i in range(3)], axis=1).astype("f8")
        rgb = np.clip(0.5 + SH_C0 * dc, 0, 1)
    else:
        rgb = np.full((len(xyz), 3), 0.7)
    keep = np.isfinite(xyz).all(axis=1)
    if "opacity" in names:
        with np.errstate(all="ignore"):
            keep &= sigmoid(vertices["opacity"]) > 0.5
    xyz, rgb = xyz[keep], rgb[keep]
    kept = int(len(xyz))
    if kept == 0:
        return xyz, rgb, 0
    low, high = np.percentile(xyz, [5, 95], axis=0)
    pad = (high - low) * 0.1
    inside = np.all((xyz > low - pad) & (xyz < high + pad), axis=1)
    xyz, rgb = xyz[inside], rgb[inside]
    pick = np.random.default_rng(seed).permutation(len(xyz))[:max_points]
    return xyz[pick], rgb[pick], kept


def write_preview(path: Path, png: Path, title: Optional[str] = None) -> int:
    """Write a three-view picture of the Gaussian centers; returns how many were drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices, declared = read_ply(path)
    xyz, rgb, kept = preview_points(vertices)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    for ax, (a, b, name) in zip(axes, [(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]):
        if len(xyz):
            order = np.argsort(xyz[:, 3 - a - b])     # far points first
            ax.scatter(xyz[order, a], xyz[order, b], c=rgb[order], s=0.4, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_facecolor("#808080")
    fig.suptitle(
        f"{title or Path(path).name}\n{kept:,} of {declared:,} Gaussians with opacity > 0.5, "
        f"central region, up to {PREVIEW_MAX_POINTS:,} shown. Centers only, NOT a render; "
        "view the PLY in a splat viewer for the real look.")
    fig.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return int(len(xyz))


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ply-check.py",
        description="Read-only integrity check of exported Gaussian-splat PLY files: "
                    "vertex count, non-finite values, opacity and scale statistics, "
                    "sha256, and an optional centers preview.",
        epilog="Exit codes: 0 all files sound, 1 a file has a problem, 2 usage error.")
    parser.add_argument("ply", nargs="+", type=Path, help="One or more .ply files.")
    parser.add_argument("--json", action="store_true",
                        help="Print one JSON document (a list, one object per file) "
                             "instead of text.")
    parser.add_argument("--preview", type=Path, default=None, metavar="PNG",
                        help="Also write a preview of the Gaussian centers to this PNG. "
                             "Only with a single PLY. Not a render.")
    parser.add_argument("--no-hash", action="store_true",
                        help="Skip the sha256 (faster on very large files).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.preview is not None and len(args.ply) != 1:
        parser.error("--preview works with exactly one PLY")
    results = [check_ply(path, with_hash=not args.no_hash) for path in args.ply]
    if args.preview is not None and results[0].get("vertices"):
        try:
            drawn = write_preview(args.ply[0], args.preview)
            results[0]["preview"] = {"path": str(args.preview), "centers_drawn": drawn}
        except ImportError as exc:
            results[0]["preview_error"] = f"matplotlib is needed for --preview ({exc})"
        except (OSError, ValueError, PlyError) as exc:
            results[0]["preview_error"] = str(exc)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(format_report(result))
            if "preview" in result:
                print(f"  preview: {result['preview']['path']} "
                      f"({result['preview']['centers_drawn']:,} centers, not a render)")
            if "preview_error" in result:
                print(f"  preview not written: {result['preview_error']}", file=sys.stderr)
    return 0 if all(r["ok"] for r in results) else EXIT_PROBLEM


if __name__ == "__main__":
    raise SystemExit(main())
