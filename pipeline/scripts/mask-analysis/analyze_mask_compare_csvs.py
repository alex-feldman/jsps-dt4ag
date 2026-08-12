# analyze_mask_compare_csvs.py
#
# Reads one or more compare_masks.py CSV outputs and generates:
# 1) per-dataset summary table
# 2) overall summary row
# 3) error-character table (FP/FN behavior)
# 4) quick plots (IoU histogram, IoU by dataset, FP vs FN by dataset)
#
# Usage examples:
#   python analyze_mask_compare_csvs.py \
#       --inputs results/test_251128/mask_compare_summary_ku-ut.csv \
#       --dataset-ids test_251128 \
#       --outdir analysis_out
#
#   python analyze_mask_compare_csvs.py \
#       --inputs results/test_251128.csv results/test_251129.csv results/test_251130.csv \
#       --dataset-ids test_251128 test_251129 test_251130 \
#       --outdir analysis_out
#
# Notes:
# - Assumes compare direction is fixed across all CSVs (e.g., pred=KU, ref=UT).
# - Expected columns:
#   relpath,identical,tp,fp,fn,tn,iou,dice,note

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {"relpath", "identical", "tp", "fp", "fn", "tn", "iou", "dice", "note"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="CSV files from compare_masks.py")
    p.add_argument(
        "--dataset-ids",
        nargs="+",
        required=False,
        help="Dataset IDs matching --inputs order. If omitted, stem of each CSV is used.",
    )
    p.add_argument("--outdir", required=True, help="Output directory for summary CSVs and plots")
    p.add_argument("--pred-label", default="KU", help="Pred label (for metadata)")
    p.add_argument("--ref-label", default="UT", help="Ref label (for metadata)")
    p.add_argument(
        "--exclude-notes",
        nargs="*",
        default=["shape_mismatch"],
        help="Rows with note in this list will be excluded from metric summaries",
    )
    return p.parse_args()


def load_one_csv(csv_path: Path, dataset_id: str, pred_label: str, ref_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset_id"] = dataset_id
    df["pred_label"] = pred_label
    df["ref_label"] = ref_label

    # Normalize dtypes
    df["identical"] = df["identical"].astype(bool)
    for c in ["tp", "fp", "fn", "tn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["iou", "dice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize note column
    df["note"] = df["note"].fillna("").astype(str).str.strip()

    # Derived helpers
    df["foreground_ref_pixels"] = df["tp"] + df["fn"]   # UT foreground size (since ref=UT)
    df["foreground_pred_pixels"] = df["tp"] + df["fp"]  # KU foreground size (since pred=KU)

    # Safe normalized rates (relative to ref foreground)
    denom = df["foreground_ref_pixels"].replace(0, np.nan)
    df["fp_rate_vs_ref_fg"] = df["fp"] / denom
    df["fn_rate_vs_ref_fg"] = df["fn"] / denom

    return df


def iqr(series: pd.Series) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return float(q1), float(q3)


def pct(series: pd.Series, p: float) -> float:
    return float(series.quantile(p))


def summarize_dataset(df_all: pd.DataFrame, exclude_notes: List[str]) -> pd.DataFrame:
    rows = []

    for dataset_id, dfg in df_all.groupby("dataset_id", sort=True):
        # QC counts from all rows in that dataset
        total_rows = len(dfg)
        note_counts = dfg["note"].value_counts(dropna=False).to_dict()

        valid_mask = ~dfg["note"].isin(exclude_notes)
        d = dfg.loc[valid_mask].copy()

        # For MVP, also exclude rows with any non-empty note from metric summaries
        # (but still report them in QC columns)
        d = d[d["note"] == ""]

        n_compared = len(d)
        if n_compared == 0:
            rows.append({
                "dataset_id": dataset_id,
                "n_rows_total": total_rows,
                "n_compared": 0,
                "n_excluded": total_rows,
                "pct_identical": np.nan,
                "mean_iou": np.nan,
                "sd_iou": np.nan,
                "median_iou": np.nan,
                "iqr_iou_low": np.nan,
                "iqr_iou_high": np.nan,
                "p05_iou": np.nan,
                "p95_iou": np.nan,
                "mean_dice": np.nan,
                "sd_dice": np.nan,
                "median_dice": np.nan,
                "pct_iou_ge_0_99": np.nan,
                "pct_iou_ge_0_95": np.nan,
                "mean_fp": np.nan,
                "mean_fn": np.nan,
                "median_fp": np.nan,
                "median_fn": np.nan,
                "mean_fp_rate_vs_ref_fg": np.nan,
                "mean_fn_rate_vs_ref_fg": np.nan,
                "count_note_nonempty": int((dfg["note"] != "").sum()),
                "count_shape_mismatch": int((dfg["note"] == "shape_mismatch").sum()),
            })
            continue

        iqr_low, iqr_high = iqr(d["iou"])

        row = {
            "dataset_id": dataset_id,
            "n_rows_total": total_rows,
            "n_compared": n_compared,
            "n_excluded": total_rows - n_compared,
            "pct_identical": 100.0 * d["identical"].mean(),
            "mean_iou": d["iou"].mean(),
            "sd_iou": d["iou"].std(ddof=1),
            "median_iou": d["iou"].median(),
            "iqr_iou_low": iqr_low,
            "iqr_iou_high": iqr_high,
            "p05_iou": pct(d["iou"], 0.05),
            "p95_iou": pct(d["iou"], 0.95),
            "mean_dice": d["dice"].mean(),
            "sd_dice": d["dice"].std(ddof=1),
            "median_dice": d["dice"].median(),
            "pct_iou_ge_0_99": 100.0 * (d["iou"] >= 0.99).mean(),
            "pct_iou_ge_0_95": 100.0 * (d["iou"] >= 0.95).mean(),
            "mean_fp": d["fp"].mean(),
            "mean_fn": d["fn"].mean(),
            "median_fp": d["fp"].median(),
            "median_fn": d["fn"].median(),
            "mean_fp_rate_vs_ref_fg": d["fp_rate_vs_ref_fg"].mean(),
            "mean_fn_rate_vs_ref_fg": d["fn_rate_vs_ref_fg"].mean(),
            "count_note_nonempty": int((dfg["note"] != "").sum()),
            "count_shape_mismatch": int((dfg["note"] == "shape_mismatch").sum()),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)
    return out


def summarize_overall(df_all: pd.DataFrame, exclude_notes: List[str]) -> pd.DataFrame:
    d = df_all.copy()
    valid_mask = ~d["note"].isin(exclude_notes)
    d = d.loc[valid_mask]
    d = d[d["note"] == ""]

    if len(d) == 0:
        return pd.DataFrame([{"dataset_id": "OVERALL", "n_compared": 0}])

    iqr_low, iqr_high = iqr(d["iou"])
    row = {
        "dataset_id": "OVERALL",
        "n_rows_total": len(df_all),
        "n_compared": len(d),
        "n_excluded": len(df_all) - len(d),
        "pct_identical": 100.0 * d["identical"].mean(),
        "mean_iou": d["iou"].mean(),
        "sd_iou": d["iou"].std(ddof=1),
        "median_iou": d["iou"].median(),
        "iqr_iou_low": iqr_low,
        "iqr_iou_high": iqr_high,
        "p05_iou": pct(d["iou"], 0.05),
        "p95_iou": pct(d["iou"], 0.95),
        "mean_dice": d["dice"].mean(),
        "sd_dice": d["dice"].std(ddof=1),
        "median_dice": d["dice"].median(),
        "pct_iou_ge_0_99": 100.0 * (d["iou"] >= 0.99).mean(),
        "pct_iou_ge_0_95": 100.0 * (d["iou"] >= 0.95).mean(),
        "mean_fp": d["fp"].mean(),
        "mean_fn": d["fn"].mean(),
        "median_fp": d["fp"].median(),
        "median_fn": d["fn"].median(),
        "mean_fp_rate_vs_ref_fg": d["fp_rate_vs_ref_fg"].mean(),
        "mean_fn_rate_vs_ref_fg": d["fn_rate_vs_ref_fg"].mean(),
        "count_note_nonempty": int((df_all["note"] != "").sum()),
        "count_shape_mismatch": int((df_all["note"] == "shape_mismatch").sum()),
    }
    return pd.DataFrame([row])


def save_table_with_rounding(df: pd.DataFrame, out_csv: Path, float_digits: int = 6) -> None:
    df_to_save = df.copy()
    float_cols = df_to_save.select_dtypes(include=[np.number]).columns
    for col in float_cols:
        df_to_save[col] = df_to_save[col].round(float_digits)
    df_to_save.to_csv(out_csv, index=False)


def plot_iou_histogram(df_all: pd.DataFrame, outpath: Path) -> None:
    d = df_all[df_all["note"] == ""].copy()
    if d.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(d["iou"], bins=30)
    plt.axvline(0.99, linestyle="--")
    plt.axvline(0.95, linestyle="--")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("IoU distribution (all datasets)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_iou_by_dataset(df_all: pd.DataFrame, outpath: Path) -> None:
    d = df_all[df_all["note"] == ""].copy()
    if d.empty:
        return

    datasets = sorted(d["dataset_id"].unique())
    data = [d.loc[d["dataset_id"] == ds, "iou"].dropna().values for ds in datasets]

    plt.figure(figsize=(max(8, 1.6 * len(datasets)), 5))
    plt.boxplot(data, tick_labels=datasets, showfliers=True)
    plt.ylabel("IoU")
    plt.xlabel("Dataset")
    plt.title("IoU by dataset")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_fp_fn_by_dataset(df_summary: pd.DataFrame, outpath: Path) -> None:
    d = df_summary[df_summary["dataset_id"] != "OVERALL"].copy()
    if d.empty:
        return

    x = np.arange(len(d))
    width = 0.38

    plt.figure(figsize=(max(8, 1.6 * len(d)), 5))
    plt.bar(x - width/2, d["mean_fp"], width=width, label="Mean FP")
    plt.bar(x + width/2, d["mean_fn"], width=width, label="Mean FN")
    plt.xticks(x, d["dataset_id"], rotation=30, ha="right")
    plt.ylabel("Pixels per image (mean)")
    plt.xlabel("Dataset")
    plt.title("Mean FP vs FN by dataset (pred=KU, ref=UT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in args.inputs]
    if args.dataset_ids is not None and len(args.dataset_ids) != len(input_paths):
        raise ValueError("--dataset-ids must match number of --inputs")

    dataset_ids = (
        args.dataset_ids
        if args.dataset_ids is not None
        else [p.stem for p in input_paths]
    )

    dfs = []
    for p, ds in zip(input_paths, dataset_ids):
        df = load_one_csv(p, ds, args.pred_label, args.ref_label)
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)
    master = master.sort_values(["dataset_id", "relpath"]).reset_index(drop=True)

    # Save master table
    master_out = outdir / "master_mask_comparisons.csv"
    master.to_csv(master_out, index=False)

    # Summaries
    per_dataset = summarize_dataset(master, args.exclude_notes)
    overall = summarize_overall(master, args.exclude_notes)

    per_dataset_with_overall = pd.concat([per_dataset, overall], ignore_index=True)

    # Error-character table (subset columns)
    error_character = per_dataset_with_overall[
        [
            "dataset_id",
            "n_compared",
            "mean_fp",
            "mean_fn",
            "median_fp",
            "median_fn",
            "mean_fp_rate_vs_ref_fg",
            "mean_fn_rate_vs_ref_fg",
        ]
    ].copy()

    # Save CSVs
    save_table_with_rounding(per_dataset_with_overall, outdir / "summary_per_dataset_with_overall.csv")
    save_table_with_rounding(error_character, outdir / "summary_error_character_fp_fn.csv")

    # Plots
    plot_iou_histogram(master, outdir / "plot_iou_hist_all.png")
    plot_iou_by_dataset(master, outdir / "plot_iou_by_dataset.png")
    plot_fp_fn_by_dataset(per_dataset_with_overall, outdir / "plot_mean_fp_vs_fn_by_dataset.png")

    # Console summary
    print(f"Saved master CSV: {master_out}")
    print(f"Saved summary table: {outdir / 'summary_per_dataset_with_overall.csv'}")
    print(f"Saved error-character table: {outdir / 'summary_error_character_fp_fn.csv'}")
    print("Saved plots:")
    print(f"  - {outdir / 'plot_iou_hist_all.png'}")
    print(f"  - {outdir / 'plot_iou_by_dataset.png'}")
    print(f"  - {outdir / 'plot_mean_fp_vs_fn_by_dataset.png'}")

    # Quick preview
    print("\nPer-dataset summary (preview):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(per_dataset_with_overall.to_string(index=False))


if __name__ == "__main__":
    main()
