# reconstruction-compare

Tools for comparing two masked 3DGS reconstructions of one capture, for example the
same photographs masked with two different prompts or tools.

- `compare_masked_reconstructions.py` renders both models from the same cameras and
  scores each against the undistorted ORIGINAL photograph, inside the region where
  both masks are foreground. It also scores the two renders against each other.
- `train_mask_variant_arm.sh` trains one arm of such a comparison: it copies a donor
  run's COLMAP model so that both arms share camera poses, re-runs the pipeline from
  the `process` stage on a different mask variant, and trains the held-out and
  `eval_mode = all` passes with `ns-eval` after each.

Both are object-agnostic. They know nothing about what was photographed.

## Why a separate tool: `ns-eval` is not like for like across masks

`ns-eval` composites each ground-truth photograph onto the model's background color
using that run's OWN mask, so two runs made with different masks are scored on
different pixels. The higher number can simply mean "a smaller or easier region".
This tool scores both models on ONE region, the intersection of the two masks, so
the same pixels are judged for both.

## Preconditions

- Both runs trained on the same capture with `use_masks = true`, so the cached images
  carry an alpha channel.
- Both runs share one COLMAP solve (camera poses). COLMAP is not bit-reproducible, so
  two separate solves make the models differ for reasons that have nothing to do with
  the masks. `train_mask_variant_arm.sh` arranges this.
- A token (`evalall`, `evalfrac10`, ...) is only ever compared with the same token of
  the other run. The two trained on different photographs otherwise. The token is read
  from each run's own `config.yml`, not from file order.
- A CUDA GPU, and the pipeline's own environment (nerfstudio, torch, torchmetrics).

## compare_masked_reconstructions.py

```bash
uv run python pipeline/scripts/reconstruction-compare/compare_masked_reconstructions.py \
    --outputs-root ~/data/outputs/{capture} \
    --a-run {run id of arm A} --a-label plant \
    --b-run {run id of arm B} --b-label leaf \
    --out-dir ~/data/evals/{capture}
```

Writes `{a-run}__vs__{b-run}_common-region_per-photo.csv` and prints, per token and
split (held-out, train, all), the mean and standard deviation over photographs of
PSNR, SSIM and LPIPS (computed inside the region) for each arm, and the paired
difference B minus A. LPIPS is lower-is-better. It also prints the photographs it
excluded (common region below `--min-region`, default 0.5% of the frame) and which
photographs each token held out.

What the numbers are, so they are not over-read:

- Scores are over the common region only, against the original photograph, so they
  are comparable between the two arms. They are NOT comparable with `ns-eval` numbers.
- LPIPS inside a region is approximate: pixels outside the region are replaced by the
  ground truth in both images, so only the region can differ.
- Photographs held out are few (2 of 24 at a 0.9 train fraction). The standard
  deviation over them is large, and one run per arm gives no noise floor. Run
  replicates of each arm before calling a difference real
  (`knowledge-base/sci/procedures/comparing-reconstruction-runs.md`, Procedure C).
- Photograph names in the output are the renamed frames (`frame_00004.png`), which are the sorted
  original file names, so frame N is the Nth photograph in sorted order.
- The script was first run on Welzo M1 (24 photographs, plant against leaf masks); that result and its
  caveats are in the workspace report `sam-welzo-leaf-mask-test-report.md`, Parts 4 and 4b.
- The render-versus-render score has no ground-truth noise. High agreement means the
  two models are practically the same inside the region.

## train_mask_variant_arm.sh

```bash
pipeline/scripts/reconstruction-compare/train_mask_variant_arm.sh \
    --template ~/data/configs/{base}.ini \
    --donor-run {run id that already holds the COLMAP model} \
    --arm-run {new run id} --variant {mask variant name} \
    --iterations 10000
```

Run `--help` for every option. It refuses to overwrite an existing arm workspace.
