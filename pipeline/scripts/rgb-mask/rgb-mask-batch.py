#!/usr/bin/env python3
"""Apply pre-made masks to RGB images as an alpha channel, in batch.

Each image under ``--images`` is paired with a mask of the same relative path
under the mask root, the mask is used as the image's alpha channel, and the
result is written as PNG under ``--output`` with the same relative structure.

Two directory layouts are supported, and the script detects which one is
present instead of assuming:

* **sibling** -- ``<root>/images/`` and ``<root>/masks/``
* **nested**  -- ``<images>/masks/``

Both occur in real data. If neither resolves, or both resolve at once, the
script exits non-zero and names the directories it looked in rather than
quietly finding no masks.

Failures are counted, reported, and fatal by default. A run that processes
zero images is a failure too. Use ``--allow-partial`` when continuing past
failures is genuinely what you want.

Exit codes:

* ``0`` -- every image processed, at least one image processed
* ``1`` -- at least one failure, or zero images processed (unless
  ``--allow-partial`` and at least one image was processed)
* ``2`` -- usage or layout error; nothing was processed

History: an earlier version of this script hardcoded the nested layout while
its header comment described the sibling one, and reported the resulting
total no-op with exit code 0. See ``README.md`` in this directory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# Image extensions this script will attempt to mask. Anything else found under
# --images is counted as an unsupported extension, which is a failure.
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

MASKS_DIRNAME = "masks"

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2


class LayoutError(Exception):
    """Raised when the mask directory cannot be resolved unambiguously."""


@dataclass
class Counters:
    """Tally of what happened to every candidate file under --images."""

    processed: int = 0
    missing_mask: int = 0
    size_mismatch: int = 0
    unsupported_ext: int = 0
    problems: list = field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.missing_mask + self.size_mismatch + self.unsupported_ext

    @property
    def total(self) -> int:
        return self.processed + self.failed


# --------------------------------------------------------------------------
# layout detection
# --------------------------------------------------------------------------

def candidate_mask_roots(images_dir: Path) -> dict:
    """Return the candidate mask roots for each supported layout."""
    return {
        "nested": images_dir / MASKS_DIRNAME,
        "sibling": images_dir.parent / MASKS_DIRNAME,
    }


def resolve_mask_root(images_dir: Path, explicit: Path = None):
    """Resolve the mask root for ``images_dir``.

    Returns ``(mask_root, layout_name)``. ``layout_name`` is ``"explicit"``
    when ``explicit`` is given. Raises :class:`LayoutError` when detection
    finds no candidate, or more than one, naming every directory it looked in.
    """
    if explicit is not None:
        if not explicit.is_dir():
            raise LayoutError(f"Mask directory does not exist: {explicit}")
        return explicit, "explicit"

    candidates = candidate_mask_roots(images_dir)
    found = {name: path for name, path in candidates.items() if path.is_dir()}

    # The sibling candidate of a directory that is itself named "masks" would
    # be that same directory; treat identical candidates as one.
    if len(found) == 2 and candidates["nested"].resolve() == candidates["sibling"].resolve():
        found = {"nested": candidates["nested"]}

    looked_in = ", ".join(f"{name} -> {path}" for name, path in candidates.items())

    if not found:
        raise LayoutError(
            "No mask directory found. Looked in: "
            f"{looked_in}. Pass --masks to name it explicitly."
        )
    if len(found) > 1:
        raise LayoutError(
            "Ambiguous layout: both a nested and a sibling mask directory "
            f"exist. Looked in: {looked_in}. Pass --masks to choose one."
        )

    name, path = next(iter(found.items()))
    return path, name


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------

def normalize_ext(ext: str) -> str:
    """Return ``ext`` with a leading dot, lowercased."""
    ext = ext.strip().lower()
    if not ext:
        return ext
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def binarize_mask(mask: "Image.Image", threshold: int) -> "Image.Image":
    """Return a hard-edged copy of ``mask``: 0 at or below threshold, else 255."""
    return mask.point(lambda value: 255 if value > threshold else 0)


def is_within(path: Path, parent: Path) -> bool:
    """True if ``path`` is ``parent`` or lives underneath it."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def iter_candidates(images_dir: Path, excluded: list):
    """Yield every non-hidden file under ``images_dir``, outside ``excluded``."""
    for path in sorted(images_dir.rglob("*")):
        if path.is_dir():
            continue
        if any(part.startswith(".") for part in path.relative_to(images_dir).parts):
            continue
        if excluded and any(is_within(path.resolve(), ex) for ex in excluded):
            continue
        yield path


def apply_masks(
    images_dir: Path,
    mask_root: Path,
    output_dir: Path,
    mask_ext: str,
    binarize: int = None,
    verbose: bool = True,
) -> Counters:
    """Mask every image under ``images_dir``; return the tally."""
    counters = Counters()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Never walk into the mask root or the output directory, either of which
    # may sit inside the image root.
    excluded = [mask_root.resolve(), output_dir.resolve()]
    resolved_images = images_dir.resolve()
    excluded = [ex for ex in excluded if ex != resolved_images and is_within(ex, resolved_images)]

    for img_path in iter_candidates(images_dir, excluded):
        relative_path = img_path.relative_to(images_dir)

        if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            counters.unsupported_ext += 1
            counters.problems.append(f"unsupported extension: {relative_path}")
            if verbose:
                print(f"UNSUPPORTED EXTENSION: {relative_path}")
            continue

        mask_path = (mask_root / relative_path).with_suffix(mask_ext)

        if not mask_path.exists():
            counters.missing_mask += 1
            counters.problems.append(f"missing mask: {relative_path} (expected {mask_path})")
            if verbose:
                print(f"MISSING MASK: {relative_path} (expected {mask_path})")
            continue

        img = Image.open(img_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")

        if img.size != mask.size:
            counters.size_mismatch += 1
            counters.problems.append(
                f"size mismatch: {relative_path} image {img.size} vs mask {mask.size}"
            )
            if verbose:
                print(
                    f"SIZE MISMATCH: {relative_path} image {img.size} vs mask {mask.size}"
                )
            continue

        if binarize is not None:
            mask = binarize_mask(mask, binarize)

        img.putalpha(mask)

        save_path = (output_dir / relative_path).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

        counters.processed += 1
        if verbose:
            print(f"processed: {relative_path}")

    return counters


def print_summary(counters: Counters, images_dir: Path, mask_root: Path,
                  output_dir: Path, layout: str, stream=None) -> None:
    out = stream if stream is not None else sys.stdout
    print("", file=out)
    print("summary", file=out)
    print(f"  images dir        : {images_dir}", file=out)
    print(f"  mask dir          : {mask_root} (layout: {layout})", file=out)
    print(f"  output dir        : {output_dir}", file=out)
    print(f"  processed         : {counters.processed}", file=out)
    print(f"  missing mask      : {counters.missing_mask}", file=out)
    print(f"  size mismatch     : {counters.size_mismatch}", file=out)
    print(f"  unsupported ext   : {counters.unsupported_ext}", file=out)
    print(f"  failed (total)    : {counters.failed}", file=out)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def threshold_type(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"threshold must be an integer 0-255, got {value!r}")
    if not 0 <= number <= 255:
        raise argparse.ArgumentTypeError(f"threshold must be in 0-255, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgb-mask-batch.py",
        description=(
            "Apply pre-made masks to RGB images as an alpha channel, writing PNGs "
            "that mirror the input directory structure. Supports both the sibling "
            "layout (<root>/images and <root>/masks) and the nested layout "
            "(<images>/masks), and detects which one is present."
        ),
        epilog=(
            "Failures are fatal by default: the script exits non-zero if any image "
            "failed, and also if zero images were processed, so a silent no-op "
            "cannot look like success. Soft mask edges are preserved by default; "
            "grey mask pixels pass straight through into the alpha channel unless "
            "--binarize is given.\n"
            "\n"
            "Examples:\n"
            "  rgb-mask-batch.py --images data/images\n"
            "  rgb-mask-batch.py --images data/scene --masks data/scene/masks \\\n"
            "      --output data/masked-images --binarize 127\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Root directory of source images, walked recursively.",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=None,
        help=(
            "Mask root directory. Omit to auto-detect: <images>/masks (nested) "
            "or <images>/../masks (sibling). Detection fails loudly if neither "
            "or both exist."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("masked-images"),
        help="Directory for the masked PNGs (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-ext",
        default=".png",
        help=(
            "Extension of the mask files, substituted for the image's own "
            "extension when looking a mask up (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--binarize",
        type=threshold_type,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Binarize the mask at THRESHOLD (0-255) before using it as alpha: "
            "values above THRESHOLD become fully opaque, the rest fully "
            "transparent. OFF by default, which preserves the existing "
            "behaviour of passing soft mask edges through unchanged. Note that "
            "the threshold choice moves boundary-level metrics, so make it "
            "deliberately."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Exit 0 even if some images failed, as long as at least one image "
            "was processed. Counts are still reported. Zero processed images "
            "remains a failure."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-file lines; print only the summary.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    images_dir = args.images
    if not images_dir.is_dir():
        print(f"ERROR: images directory does not exist: {images_dir}", file=sys.stderr)
        return EXIT_USAGE

    try:
        mask_root, layout = resolve_mask_root(images_dir, args.masks)
    except LayoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    mask_ext = normalize_ext(args.mask_ext)
    if not mask_ext:
        print("ERROR: --mask-ext must not be empty", file=sys.stderr)
        return EXIT_USAGE

    counters = apply_masks(
        images_dir=images_dir,
        mask_root=mask_root,
        output_dir=args.output,
        mask_ext=mask_ext,
        binarize=args.binarize,
        verbose=not args.quiet,
    )

    print_summary(counters, images_dir, mask_root, args.output, layout)

    if counters.processed == 0:
        print(
            "ERROR: zero images processed. Nothing downstream should consume "
            f"{args.output}.",
            file=sys.stderr,
        )
        return EXIT_FAILURES

    if counters.failed:
        if args.allow_partial:
            print(
                f"WARNING: {counters.failed} of {counters.total} files failed; "
                "continuing because --allow-partial was given.",
                file=sys.stderr,
            )
            return EXIT_OK
        print(
            f"ERROR: {counters.failed} of {counters.total} files failed. "
            "Re-run with --allow-partial only if a partial dataset is genuinely "
            "acceptable downstream.",
            file=sys.stderr,
        )
        return EXIT_FAILURES

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
