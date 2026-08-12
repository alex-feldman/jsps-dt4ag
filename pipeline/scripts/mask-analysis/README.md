# mask-analysis

Compares two sets of **binary mask images** (for example your masks against a
collaborator's) and reports whether each pair is identical. Where they differ, it
writes a colour-coded diff PNG showing the disagreement pixel by pixel.

This is how the segmentation agreement floor is established: how closely two
independently produced mask sets agree on the same images. Any downstream
accuracy claim has to sit above that floor to mean anything.

- `compare_masks.py` computes per-image agreement and writes a CSV.
- `analyze_mask_compare_csvs.py` pools those per-image CSVs into per-dataset and
  overall summaries, and produces the plots.

The scripts are object-agnostic. They compare mask images and know nothing about
what was photographed.

## compare_masks.py

For each corresponding mask pair it outputs `identical: True/False`, the pixel
counts TP / FP / FN / TN, and the overlap metrics IoU and Dice. When a pair is
not identical it writes a diff image:

| Colour | Meaning |
|---|---|
| White | True positive: both masks foreground |
| Red | False positive: pred is foreground, ref is background |
| Green | False negative: pred is background, ref is foreground |
| Black | True negative: both masks background |

It walks entire directory trees, including nested subfolders, assuming both trees
share the same relative paths.

### Requirements

Python 3.9+, `numpy`, `Pillow`. Setup with `uv` is recommended.

### Usage

```bash
python compare_masks.py \
  --pred-root /path/to/pred_masks \
  --ref-root  /path/to/ref_masks \
  --diff-root ./mask_diffs \
  --csv       ./mask_compare_summary.csv \
  --ext .png
```

**Thresholding.** By default any pixel value greater than 0 counts as
foreground. Raise it if the background carries noise:

```bash
  --threshold 10
```

**Multiple extensions.** Repeat the flag:

```bash
  --ext .png --ext .tif --ext .tiff
```

**Fail fast.** Missing files are counted and skipped by default. To stop at the
first one:

```bash
  --fail-on-missing
```

Note that `compare_masks.py` has this flag and honours it. The masking script in
`../rgb-mask/` does not, and that asymmetry is the subject of its README.

### Troubleshooting

Masks must share dimensions for a pixel-wise comparison to be valid. A
`shape_mismatch` in the CSV means something upstream resized or cropped
differently; investigate there rather than here.

If masks are stored in alpha channels (RGBA), the loader may need adjusting: it
currently loads greyscale via `convert("L")`.

## Provenance

`compare_masks.py` and this documentation previously lived in the `samask`
repository under `mask_compare/`. As of 2026-08-07 **this is the canonical
home**, and the samask copy was removed. The split of responsibility is now:
samask generates masks, this repository runs the pipeline and measures it. That
also keeps the measurement code reachable, since samask is private.

Usage examples above were generalized from the original project-specific paths at
the same time.
