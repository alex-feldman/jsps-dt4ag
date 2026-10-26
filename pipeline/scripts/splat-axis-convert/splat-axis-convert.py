#!/usr/bin/env python3
"""Convert a Nerfstudio z-up Gaussian-splat PLY to y-up without retraining.

The source PLY must declare comment Vertical Axis: z. The conversion changes
coordinate convention only: it adds no scale or translation. --viewer
supersplat also writes a separate display-only copy for SuperSplat.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ORGAN_ANALYSIS = SCRIPT_DIR.parent / "organ-analysis"
if str(ORGAN_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ORGAN_ANALYSIS))

import frame_calibration as frame_calibration  # noqa: E402
import transform_splat as splat  # noqa: E402

SCHEMA = "splat-axis-conversion"
SCHEMA_VERSION = 1
Z_UP_TO_Y_UP = np.asarray(frame_calibration.Z_UP_TO_Y_UP, dtype=np.float64)
ZERO_ORIGIN = np.zeros(3, dtype=np.float64)


def replace_axis_header(lines: list[str], expected: str, target: str, provenance: str) -> list[str]:
    """Replace exactly one axis declaration and add a pointer to the sidecar."""
    axis_lines = [line for line in lines if line.startswith("comment Vertical Axis:")]
    if len(axis_lines) != 1:
        raise ValueError("Expected exactly one PLY comment Vertical Axis declaration")
    actual = axis_lines[0].split(":", 1)[1].strip().lower()
    if actual != expected:
        raise ValueError(
            f"Expected source PLY Vertical Axis: {expected}, found {actual!r}; refusing to guess the source frame"
        )

    result = []
    for line in lines:
        if line.startswith("comment Vertical Axis:"):
            line = f"comment Vertical Axis: {target}"
        result.append(line)
        if line.startswith("format "):
            result.append(provenance)
    return result


def write_json_new(path: Path, data: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def axis_sidecar_name(out: Path) -> Path:
    return out.with_name(out.stem + ".axis-conversion.json")


def viewer_paths(out: Path) -> tuple[Path, Path]:
    display = out.with_name(out.stem + "-supersplat.ply")
    sidecar = out.with_name(out.stem + "-supersplat.viewer.json")
    return display, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=Path, help="Nerfstudio Gaussian-splat PLY, read only")
    parser.add_argument("--out", required=True, type=Path, help="new y-up PLY; output files must not exist")
    parser.add_argument(
        "--viewer",
        choices=["supersplat"],
        help="also write a display-only SuperSplat copy and its sidecar",
    )
    return parser


def convert(source: Path, out: Path, viewer: str | None = None) -> tuple[Path, Path | None]:
    source = source.resolve()
    out = out.resolve()
    if source == out:
        raise ValueError("The conversion never writes in place")
    if source.suffix.lower() != ".ply" or out.suffix.lower() != ".ply":
        raise ValueError("Input and output must end in .ply")
    if not source.is_file():
        raise ValueError(f"Input PLY does not exist: {source}")
    if not out.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {out.parent}")

    sidecar = axis_sidecar_name(out)
    display, display_sidecar = viewer_paths(out) if viewer else (None, None)
    targets = [out, sidecar]
    if display is not None and display_sidecar is not None:
        targets.extend([display, display_sidecar])
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise ValueError("Output already exists; choose new output names: " + ", ".join(existing))

    source_sha = splat.sha(source)
    lines, data = splat.read_ply(source)
    converted_header = replace_axis_header(
        lines,
        expected="z",
        target="y",
        provenance=f"comment jsps-dt4ag axis conversion v{SCHEMA_VERSION}: +z-up to +y-up, no scale or translation; sidecar {sidecar.name}",
    )
    converted, transform_record = splat.transform(data, Z_UP_TO_Y_UP, ZERO_ORIGIN, 1.0)
    splat.write_ply(out, converted_header, converted)
    if splat.sha(source) != source_sha:
        raise ValueError("Input changed during conversion")

    source_record = {
        "path": str(source),
        "sha256": source_sha,
        "vertical_axis": "+z",
        "vertices": int(len(data)),
    }
    output_record = {
        "path": str(out),
        "sha256": splat.sha(out),
        "vertical_axis": "+y",
        "vertices": int(len(converted)),
    }
    sidecar_record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "name": "jsps-dt4ag",
            "script": "pipeline/scripts/splat-axis-convert/splat-axis-convert.py",
        },
        "source": source_record,
        "output": output_record,
        "conversion": {
            "method": "fixed right-handed z-up to y-up coordinate convention",
            "matrix": Z_UP_TO_Y_UP.tolist(),
            "origin": [0.0, 0.0, 0.0],
            "scale": 1.0,
            "equation": "p_y_up = R @ p_z_up",
            "transform": transform_record,
            "unchanged_properties": sorted(set(data.dtype.names or ()) - splat.transformed_properties(data)),
        },
    }
    write_json_new(sidecar, sidecar_record)
    print(f"Wrote y-up PLY {out} ({len(converted)} Gaussians) and {sidecar}")

    if viewer and display is not None and display_sidecar is not None:
        viewer_rotation = np.asarray(splat.VIEWERS[viewer]["rotation"], dtype=np.float64)
        display_data, display_transform = splat.transform(converted, viewer_rotation, ZERO_ORIGIN, 1.0)
        display_header = replace_axis_header(
            converted_header,
            expected="y",
            target="y",
            provenance=(
                f"comment DISPLAY ONLY for SuperSplat: pre-rotated 180 degrees about z from {out.name}; "
                f"sidecar {display_sidecar.name}"
            ),
        )
        splat.write_ply(display, display_header, display_data)
        if splat.sha(source) != source_sha:
            raise ValueError("Input changed while writing the SuperSplat display copy")
        viewer_record = {
            "schema": "splat-viewer-copy",
            "schema_version": 1,
            "display_only": True,
            "purpose": "SuperSplat display; use the y-up PLY for analysis and measurement",
            "viewer": "supersplat",
            "source_input": source_record,
            "analysis_file": output_record,
            "output": {
                "path": str(display),
                "sha256": splat.sha(display),
                "vertices": int(len(display_data)),
            },
            "display_rotation": {
                "matrix": viewer_rotation.tolist(),
                "description": "180 degrees about z",
                "equation": "p_display = R @ p_y_up; no origin or scale",
                "transform": display_transform,
            },
        }
        write_json_new(display_sidecar, viewer_record)
        print(f"Wrote display-only SuperSplat PLY {display} and {display_sidecar}")

    return out, display


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        convert(args.input, args.out, args.viewer)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"splat-axis-convert: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())