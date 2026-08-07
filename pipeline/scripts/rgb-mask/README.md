# rgb-mask

Applies a pre-made mask to an RGB image as an alpha channel, producing the
masked images the reconstruction notebook consumes. Masks themselves come from
SAM, via the separate `samask` repository, not from here.

- `rgb-mask.py` operates on a single image.
- `rgb-mask-batch.py` walks a directory.

## KNOWN DEFECT: `rgb-mask-batch.py` silently does nothing

**The script's header comment and its code disagree about the directory layout,
and the mismatch fails silently.**

- Line 2 says to run the script "in the root folder containing `images/` and
  `masks/` dirs", which is a **sibling** layout. That is also the layout the
  real project data uses.
- Line 10 computes `mask_root = base_dir / "masks"`, where `base_dir` is
  `images`. So the code actually requires `images/masks/`, a **nested** layout.

Verified by running the script unmodified against a fixture of each shape:

| Layout | Result |
|---|---|
| Sibling (`images/`, `masks/`) | Prints `Mask missing for: <file>` for every image, creates an empty `masked-images/`, and **exits 0** |
| Nested (`images/masks/`) | Prints `Processed: <file>` and writes the masked output |

So following the header comment produces a successful-looking no-op, and the
reconstruction then runs on nothing.

This generalizes. All three failure modes (missing mask, size mismatch,
unsupported extension) print a line and continue. There are no counts and no
non-zero exit, so a partially masked dataset flows into reconstruction unnoticed
and silently changes what any downstream evaluation measures.

Two smaller issues in the same file:

- The mask extension is hardcoded to `.png` (there is a "for now, hardcode"
  comment saying so).
- The mask is converted to `L` and used directly as alpha with no
  binarization, so soft edges pass through. Binarization threshold is a known
  source of boundary-level IoU differences, so it should be an explicit choice
  rather than an implicit one.

**The script is committed here as-is, unfixed, on purpose.** This commit is
about getting the code under version control, not about correcting it. The fix
is tracked separately as `kibanc-dt/O2.T13`. Until then, use the nested layout
that the code requires, and check that `masked-images/` is actually non-empty
before running anything downstream.
