# rgb-mask

Applies a pre-made mask to an RGB image as an alpha channel, producing the
masked images the reconstruction notebook consumes. Masks themselves come from
SAM, via the separate `samask` repository, not from here.

- `rgb-mask.py` operates on a single image.
- `rgb-mask-batch.py` walks a directory.

## `rgb-mask-batch.py` usage

    rgb-mask-batch.py --images <dir> [--masks <dir>] [--output <dir>]
                      [--mask-ext .png] [--binarize THRESHOLD]
                      [--allow-partial] [--quiet]

`--images` is required and is walked recursively. Every image is paired with a
mask of the same relative path under the mask root, with the mask extension
substituted, and the masked result is written as PNG under `--output`
(default `masked-images` in the current directory) with the same relative
structure.

### Layouts

Two directory layouts are supported, and the script detects which one is
present rather than assuming:

| Layout | Shape |
|---|---|
| sibling | `<root>/images/` and `<root>/masks/` |
| nested | `<images>/masks/` |

Both occur in the real data, which is why both are supported. If neither
candidate directory exists, or if both exist at once, the script exits `2` and
names both directories it looked in. Pass `--masks` to skip detection and
name the mask root yourself.

### Failure handling

Failures are counted, listed, summarised, and fatal by default:

| Situation | Counted as | Default exit |
|---|---|---|
| Mask file not found | `missing mask` | `1` |
| Image and mask differ in size | `size mismatch` | `1` |
| File is not `.jpg`/`.jpeg`/`.png` | `unsupported ext` | `1` |
| Zero images processed | (nothing to count) | `1` |
| Mask root undetectable or ambiguous | n/a | `2` |

Hidden files and directories (names starting with `.`) are skipped entirely
and are not counted. The mask root and the output directory are never walked as
input, so both layouts and a re-run over an output directory nested inside the
image root behave the same way.

`--allow-partial` downgrades a partial failure to exit `0`, for the case where
a partial dataset is genuinely acceptable downstream. It still reports the full
counts, and zero images processed remains a failure even with it.

### Options

- `--mask-ext` (default `.png`) is the extension substituted when looking up a
  mask. A leading dot is optional.
- `--binarize THRESHOLD` (0-255) makes the alpha channel hard-edged: mask
  values above `THRESHOLD` become fully opaque, the rest fully transparent. It
  is **off by default**, which preserves the historical behaviour of passing
  soft mask edges straight through into the alpha channel. The threshold choice
  moves boundary-level metrics, so it is an explicit opt-in rather than a
  silent default.
- `--quiet` prints only the summary, not the per-file lines.

### Examples

    # sibling layout, detected
    rgb-mask-batch.py --images data/images --output data/masked-images

    # nested layout, detected
    rgb-mask-batch.py --images data/scene --output data/scene/masked-images

    # explicit mask root and a hard-edged alpha
    rgb-mask-batch.py --images data/scene --masks data/scene/masks \
        --output data/masked-images --binarize 127

## FIXED DEFECT (history): `rgb-mask-batch.py` used to silently do nothing

Kept here on purpose. The trap existed, it cost real time, and a future reader
should be able to recognise the shape of it.

**The script's header comment and its code disagreed about the directory
layout, and the mismatch failed silently.**

- Line 2 said to run the script "in the root folder containing `images/` and
  `masks/` dirs", which is a **sibling** layout.
- Line 10 computed `mask_root = base_dir / "masks"`, where `base_dir` was the
  hardcoded `Path("images")`. So the code actually required `images/masks/`, a
  **nested** layout.

Verified by running the old script unmodified against a fixture of each shape:

| Layout | Old result |
|---|---|
| Sibling (`images/`, `masks/`) | Printed `Mask missing for: <file>` for every image, created an empty `masked-images/`, and **exited 0** |
| Nested (`images/masks/`) | Printed `Processed: <file>` and wrote the masked output |

So following the header comment produced a successful-looking no-op, and the
reconstruction then ran on nothing.

It generalised. All three failure modes (missing mask, size mismatch,
unsupported extension) printed a line and continued. There were no counts and
no non-zero exit, so a partially masked dataset could flow into reconstruction
unnoticed and silently change what any downstream evaluation measured.

Two smaller issues in the same file:

- The mask extension was hardcoded to `.png` (there was a "for now, hardcode"
  comment saying so).
- The mask was converted to `L` and used directly as alpha with no
  binarization, so soft edges passed through. Binarization threshold is a known
  source of boundary-level IoU differences, so it should be an explicit choice
  rather than an implicit one.

### What the fix changed

- Both layouts are supported and **detected**; an undetectable or ambiguous
  mask root is a hard error naming the directories searched, not a silent
  zero-mask run.
- The hardcoded `Path("images")` is gone, replaced by an argparse interface.
- Every outcome is counted and summarised.
- Any failure, and any run that processed zero images, exits non-zero.
  `--allow-partial` is the explicit opt-out, so the strict behaviour is the
  default.
- `--mask-ext` replaces the hardcoded `.png`.
- `--binarize` makes the soft-edge decision explicit, defaulting to the old
  pass-through behaviour so results do not change unless asked for.

The masking itself is unchanged: open the image as RGBA, open the mask as `L`,
use it as the alpha channel, write PNG.

One correction to the earlier account of the defect: it stated that the sibling
layout "is also the layout the real project data uses". Both layouts are in use
in the real data. Some datasets keep `images/` and `masks/` as siblings; others
keep `masks/` inside the image root alongside the per-session capture
directories, which is the nested shape the old code required. That is exactly
why the fix detects the layout instead of picking one.

Tracked as `kibanc-dt/O2.T13`. Covered by `pipeline/tests/test_rgb_mask_batch.py`.
