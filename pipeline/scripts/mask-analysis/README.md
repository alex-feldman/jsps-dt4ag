# mask-analysis

The UT-vs-KU mask agreement study: how closely two independently produced sets
of segmentation masks agree on the same images. This establishes the
segmentation noise floor that any downstream accuracy claim has to sit above.

- `compare_masks.py` computes per-image agreement (IoU, false positive and
  false negative character) between two mask sets and writes a CSV.
- `analyze_mask_compare_csvs.py` pools those per-image CSVs into per-dataset and
  overall summaries and produces the plots.

Only the scripts are committed. The inputs (mask sets) and the outputs
(per-image CSVs, summary CSVs, plots) live with the data, outside this
repository.

**`compare_masks.py` is a duplicate.** A byte-identical copy lives in the
separate `samask` repository as `mask_compare/compare_masks.py`. Which of the
two is canonical, and whether the two repositories should merge, is not yet
decided. Do not edit one copy without checking the other.
