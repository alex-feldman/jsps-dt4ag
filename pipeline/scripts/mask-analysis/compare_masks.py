from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict

import numpy as np
from PIL import Image


@dataclass
class MaskCompareResult:
    identical: bool
    tp: int
    fp: int
    fn: int
    tn: int
    iou: float
    dice: float
    diff_path: Optional[Path] = None


def _load_binary_mask(path: Path, threshold: int = 0) -> np.ndarray:
    """
    Loads an image and returns a boolean array mask (H, W).
    Any pixel value > threshold is considered foreground (True).
    """
    img = Image.open(path).convert("L")  # grayscale
    arr = np.array(img)
    return arr > threshold


def compare_binary_masks(
    pred_path: Path,
    ref_path: Path,
    *,
    threshold: int = 0,
    save_diff_to: Optional[Path] = None,
    require_same_shape: bool = True,
) -> MaskCompareResult:
    pred = _load_binary_mask(pred_path, threshold=threshold)
    ref = _load_binary_mask(ref_path, threshold=threshold)

    if pred.shape != ref.shape:
        if require_same_shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, ref={ref.shape} "
                             f"({pred_path} vs {ref_path})")
        else:
            ref_img = Image.open(ref_path).convert("L")
            ref_img = ref_img.resize((pred.shape[1], pred.shape[0]), resample=Image.NEAREST)
            ref = (np.array(ref_img) > threshold)

    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())

    identical = (fp == 0 and fn == 0)

    union = tp + fp + fn
    iou = (tp / union) if union > 0 else 1.0

    denom = (2 * tp + fp + fn)
    dice = (2 * tp / denom) if denom > 0 else 1.0

    diff_path = None
    if save_diff_to is not None and not identical:
        save_diff_to.parent.mkdir(parents=True, exist_ok=True)

        # TP white, FP red, FN green, TN black
        diff = np.zeros((*pred.shape, 3), dtype=np.uint8)
        diff[np.logical_and(pred, ref)] = (255, 255, 255)  # TP
        diff[np.logical_and(pred, ~ref)] = (255, 0, 0)      # FP
        diff[np.logical_and(~pred, ref)] = (0, 255, 0)      # FN

        Image.fromarray(diff, mode="RGB").save(save_diff_to)
        diff_path = save_diff_to

    return MaskCompareResult(
        identical=identical,
        tp=tp, fp=fp, fn=fn, tn=tn,
        iou=iou, dice=dice,
        diff_path=diff_path,
    )


def iter_mask_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def compare_mask_trees(
    pred_root: Path,
    ref_root: Path,
    *,
    diff_root: Optional[Path],
    threshold: int,
    exts: Tuple[str, ...],
    csv_path: Optional[Path],
    fail_on_missing: bool,
) -> Dict[str, int]:
    pred_root = pred_root.resolve()
    ref_root = ref_root.resolve()

    pred_files = list(iter_mask_files(pred_root, exts=exts))
    pred_rel = {p.relative_to(pred_root): p for p in pred_files}

    ref_files = list(iter_mask_files(ref_root, exts=exts))
    ref_rel = {p.relative_to(ref_root): p for p in ref_files}

    all_keys = sorted(set(pred_rel.keys()) | set(ref_rel.keys()))

    rows = []
    counts = {
        "total_compared": 0,
        "identical": 0,
        "different": 0,
        "missing_in_pred": 0,
        "missing_in_ref": 0,
        "shape_mismatch": 0,
    }

    for rel in all_keys:
        pred_path = pred_rel.get(rel)
        ref_path = ref_rel.get(rel)

        if pred_path is None:
            counts["missing_in_pred"] += 1
            if fail_on_missing:
                raise FileNotFoundError(f"Missing in pred_root: {rel}")
            continue
        if ref_path is None:
            counts["missing_in_ref"] += 1
            if fail_on_missing:
                raise FileNotFoundError(f"Missing in ref_root: {rel}")
            continue

        counts["total_compared"] += 1

        diff_path = None
        if diff_root is not None:
            diff_path = (diff_root / rel).with_suffix(".png")

        try:
            res = compare_binary_masks(
                pred_path,
                ref_path,
                threshold=threshold,
                save_diff_to=diff_path,
                require_same_shape=True,
            )
            note = ""
        except ValueError as e:
            counts["shape_mismatch"] += 1
            res = None
            note = f"shape_mismatch: {e}"

        if res is None:
            rows.append({
                "relpath": str(rel),
                "identical": False,
                "tp": "",
                "fp": "",
                "fn": "",
                "tn": "",
                "iou": "",
                "dice": "",
                "note": note,
            })
            continue

        if res.identical:
            counts["identical"] += 1
        else:
            counts["different"] += 1

        rows.append({
            "relpath": str(rel),
            "identical": res.identical,
            "tp": res.tp,
            "fp": res.fp,
            "fn": res.fn,
            "tn": res.tn,
            "iou": f"{res.iou:.6f}",
            "dice": f"{res.dice:.6f}",
            "note": "",
        })

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["relpath", "identical", "tp", "fp", "fn", "tn", "iou", "dice", "note"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Compare two directory trees of binary masks.")
    parser.add_argument("--pred-root", type=Path, required=True, help="Root directory of your masks.")
    parser.add_argument("--ref-root", type=Path, required=True, help="Root directory of collaborator masks.")
    parser.add_argument("--diff-root", type=Path, default=None, help="Where to write diff PNGs (only for mismatches).")
    parser.add_argument("--csv", type=Path, default=Path("mask_compare_summary.csv"), help="CSV output path.")
    parser.add_argument("--threshold", type=int, default=0, help="Pixels > threshold are treated as foreground.")
    parser.add_argument("--ext", action="append", default=[".png"], help="Mask file extension(s). Repeatable.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Stop if any file is missing in either tree.")
    args = parser.parse_args()

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext)

    counts = compare_mask_trees(
        pred_root=args.pred_root,
        ref_root=args.ref_root,
        diff_root=args.diff_root,
        threshold=args.threshold,
        exts=exts,
        csv_path=args.csv,
        fail_on_missing=args.fail_on_missing,
    )

    print("Done.")
    for k, v in counts.items():
        print(f"{k}: {v}")
    if args.csv is not None:
        print(f"CSV: {args.csv.resolve()}")
    if args.diff_root is not None:
        print(f"Diffs: {args.diff_root.resolve()}")


if __name__ == "__main__":
    main()

