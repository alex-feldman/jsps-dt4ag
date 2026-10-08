#!/usr/bin/env python
"""Compare two masked 3DGS reconstructions of one capture on a COMMON region.

Why this exists: ``ns-eval`` composites each ground-truth photograph onto the model's background colour
using that run's OWN mask, so the scores of two runs made with different masks are measured on different
pixels and are not like for like (see knowledge-base procedure comparing-reconstruction-runs.md, section 2).
This script renders both models from the same cameras and scores each against the undistorted ORIGINAL
photograph (the RGB under the alpha channel) only where BOTH runs' masks are foreground. It also scores the
two renders against each other in that region.

Inputs: two runs of one capture, each trained with ``eval_mode`` ``all`` and/or ``fraction`` (the "token").
Tokens are never mixed: token T of run A is compared only with token T of run B, because the two trained on
different photographs otherwise.

Run it with the pipeline's own interpreter (it needs nerfstudio, torch, torchmetrics), for example:

    uv run python pipeline/scripts/reconstruction-compare/compare_masked_reconstructions.py \\
        --outputs-root ~/data/outputs/welzo-M1-masktest \\
        --a-run masktest-plant_261008-01-3120 --a-label plant \\
        --b-run masktest-leaf_261008-01-3120  --b-label leaf \\
        --out-dir ~/data/evals/welzo-M1-masktest

Both runs' images must carry an alpha channel (they do whenever ``use_masks`` was true). Needs a CUDA GPU.
Object-agnostic: it knows nothing about what was photographed.
"""
import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

TOKEN_OF_EVAL_MODE = {"all": "evalall", "fraction": "evalfrac", "interval": "evalint"}


def configs_by_token(outputs_root, run, method):
    """{token: newest config.yml}, where the token is read from the run's own config, not guessed from the order."""
    found = {}
    for cfg in sorted(glob.glob(f"{outputs_root}/{run}/{method}/*/config.yml")):
        text = Path(cfg).read_text()
        mode = None
        for line in text.splitlines():
            if line.strip().startswith("eval_mode:"):
                mode = line.split(":", 1)[1].strip()
                break
        if mode in TOKEN_OF_EVAL_MODE:
            tok = TOKEN_OF_EVAL_MODE[mode]
            if mode == "fraction":
                frac = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                             if l.strip().startswith("train_split_fraction:")), "?")
                tok += str(round((1 - float(frac)) * 100)) if frac != "?" else ""
                tok = tok.replace("evalfrac", "evalfrac")  # e.g. train fraction 0.9 -> evalfrac10 held out
            found[tok] = cfg  # sorted by timestamp, so a later config of the same token wins
    return found


def load_model_and_images(cfg, device):
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(Path(cfg), test_mode="test")
    dm = pipeline.datamanager
    items = {}
    train_names = set()
    for split, ds, cache in (("train", dm.train_dataset, dm.cached_train), ("heldout", dm.eval_dataset, dm.cached_eval)):
        names = [Path(p).name for p in ds._dataparser_outputs.image_filenames]
        for i, nm in enumerate(names):
            img = cache[i]["image"]
            if img.shape[-1] != 4:
                sys.exit(f"{nm}: the cached image has no alpha channel, so there is no mask to intersect "
                         f"(was use_masks false for this run?)")
            # nerfstudio caches undistorted images as uint8; scale to [0, 1] the way its own metrics do.
            scale = 255.0 if img.dtype == torch.uint8 else 1.0
            img = img.float() / scale
            # eval_mode = all makes the eval set the whole training set, so those photographs are not held out.
            if split == "heldout" and nm in train_names:
                items[nm]["split"] = "all"
                continue
            if split == "train":
                train_names.add(nm)
            items[nm] = dict(split=split, camera=ds.cameras[i:i + 1].to(device),
                             rgb=img[..., :3].clamp(0, 1).to(device), alpha=img[..., 3:4].to(device))
    return pipeline, items


@torch.no_grad()
def render(pipeline, camera):
    return pipeline.model.get_outputs_for_camera(camera)["rgb"].clamp(0, 1)


def psnr_region(a, b, region):
    mse = ((a - b) ** 2)[region.expand_as(a)].mean().item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def ssim_region(a, b, region):
    _, full = ssim_fn(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None], data_range=1.0, return_full_image=True)
    full = full[0].permute(1, 2, 0)
    return full[region.expand_as(full)].mean().item()


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    s = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    return m, s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outputs-root", required=True, help="the capture's training output directory, e.g. ~/data/outputs/{capture}")
    ap.add_argument("--a-run", required=True)
    ap.add_argument("--b-run", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--method", default="splatfacto", help="the sub-directory of a run that holds the timestamped training directories")
    ap.add_argument("--out-dir", required=True, help="where the per-photograph CSV is written")
    ap.add_argument("--min-region", type=float, default=0.005,
                    help="exclude a photograph whose common region is below this share of the frame (default 0.005)")
    ap.add_argument("--tokens", default="", help="comma list of tokens to compare; default every token both runs have")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required to render")
    dev = torch.device("cuda")
    root = os.path.expanduser(args.outputs_root)
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    ca = configs_by_token(root, args.a_run, args.method)
    cb = configs_by_token(root, args.b_run, args.method)
    tokens = [t for t in (args.tokens.split(",") if args.tokens else sorted(set(ca) & set(cb))) if t]
    if not tokens:
        sys.exit(f"no eval token in common: {args.a_run} has {sorted(ca)}, {args.b_run} has {sorted(cb)}")
    A, B = args.a_label, args.b_label
    lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)

    def lpips_region(a, b, region):
        # outside the region both images become the same pixels, so only the region can differ
        a2 = torch.where(region, a, b)
        return lp(a2.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]).item()

    rows = []
    for tok in tokens:
        pa, ia = load_model_and_images(ca[tok], dev)
        pb, ib = load_model_and_images(cb[tok], dev)
        if set(ia) != set(ib):
            sys.exit(f"{tok}: the two runs hold different photographs")
        for nm in sorted(ia):
            X, Y = ia[nm], ib[nm]
            if X["split"] != Y["split"]:
                sys.exit(f"{tok} {nm}: the runs split this photograph differently (held out in one, trained in the other)")
            gt = X["rgb"]
            region = (X["alpha"] > 0.5) & (Y["alpha"] > 0.5)
            frac = region.float().mean().item()
            row = dict(token=tok, photo=nm, split=X["split"], region_frac=round(frac, 4),
                       gt_rgb_diff=round((gt - Y["rgb"]).abs().max().item(), 4))
            if frac < args.min_region:
                row["excluded"] = "common region too small"
                rows.append(row)
                continue
            ra, rb = render(pa, X["camera"]), render(pb, Y["camera"])
            for lab, r in ((A, ra), (B, rb)):
                row[f"{lab}_psnr"] = psnr_region(r, gt, region)
                row[f"{lab}_ssim"] = ssim_region(r, gt, region)
                row[f"{lab}_lpips"] = lpips_region(r, gt, region)
            row["m2m_psnr"] = psnr_region(ra, rb, region)
            row["m2m_ssim"] = ssim_region(ra, rb, region)
            rows.append(row)
        del pa, pb
        torch.cuda.empty_cache()

    keys = ["token", "photo", "split", "region_frac", "gt_rgb_diff", "excluded",
            f"{A}_psnr", f"{B}_psnr", f"{A}_ssim", f"{B}_ssim", f"{A}_lpips", f"{B}_lpips", "m2m_psnr", "m2m_ssim"]
    out_csv = out_dir / f"{args.a_run}__vs__{args.b_run}_common-region_per-photo.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k in keys})
    print("wrote", out_csv)

    for tok in tokens:
        present = [sp for sp in ("heldout", "train", "all") if any(r["token"] == tok and r["split"] == sp for r in rows)]
        for split in present + ([None] if len(present) > 1 else []):
            sel = [r for r in rows if r["token"] == tok and "excluded" not in r and (split is None or r["split"] == split)]
            if not sel:
                continue
            print(f"\n{tok}, {'all valid photographs' if split is None else split + ' photographs'}: n={len(sel)}")
            for met, better in (("psnr", "higher"), ("ssim", "higher"), ("lpips", "LOWER")):
                am, asd = mean_std([r[f"{A}_{met}"] for r in sel])
                bm, bsd = mean_std([r[f"{B}_{met}"] for r in sel])
                dm, dsd = mean_std([r[f"{B}_{met}"] - r[f"{A}_{met}"] for r in sel])
                print(f"  {met:5} ({better} is better)  {A} {am:.4f} (sd {asd:.3f})  {B} {bm:.4f} (sd {bsd:.3f})  {B}-{A} {dm:+.4f} (sd {dsd:.3f})")
            mp, _ = mean_std([r["m2m_psnr"] for r in sel])
            ms, _ = mean_std([r["m2m_ssim"] for r in sel])
            print(f"  {A} render vs {B} render, in the region: PSNR {mp:.2f}, SSIM {ms:.4f}")
    excluded = sorted({(r["photo"], r["region_frac"]) for r in rows if "excluded" in r})
    print(f"\nexcluded (common region below {args.min_region * 100:.1f}% of the frame): {excluded}")
    for tok in tokens:
        held = sorted(r["photo"] for r in rows if r["token"] == tok and r["split"] == "heldout")
        print(f"held out in {tok}: {held}")
    print("largest difference between the two runs' ground-truth RGB (0 means the same photographs):",
          max(r["gt_rgb_diff"] for r in rows))


if __name__ == "__main__":
    main()
