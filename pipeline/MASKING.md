# Masking: why a mask file and a premade alpha image are not the same thing

Status: **known defect in the `use_masks` implementation**, recorded 2026-08-18.
The current mask support suppresses background *supervision*; it does not suppress
background *geometry*. The earlier premade-alpha route did. This file records the
evidence and the fix, because the two look interchangeable and are not.

## Symptom

Reconstructions produced from a separate mask file contain a large amount of
background geometry. Reconstructions produced from premade RGBA images of the same
scene contain the subject alone. In the worst observed case the subject was hard to
find at all inside the surrounding junk.

Measured on one session, same subject, same camera rig:

| route | gaussians | within 1.0 of median | 99th pct radius |
|---|---|---|---|
| premade RGBA images | 1,276 | 100% | 0.05 |
| separate mask file | 96,688 | 47.5% | 9.14 |

A 76x difference in count. This is not a tuning difference.

## Cause

Two different mechanisms, only one of which removes geometry.

**Premade RGBA image → the background is SUPERVISED to be empty.**
`nerfstudio/models/splatfacto.py`, `composite_with_background`:

```python
if image.shape[2] == 4:
    alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
    return alpha * image[..., :3] + (1 - alpha) * background
```

The ground truth's background is replaced by the same `background` value the
renderer composites behind the gaussians (`rgb = render[..., :3] + (1 - alpha) *
background`). A background region containing zero gaussians therefore matches the
ground truth exactly. With `background_color = random` that colour is redrawn every
iteration, so any opaque gaussian in the background is penalised on every iteration
and is driven to zero opacity and culled.

**Separate mask file → the background is IGNORED.**
The mask removes those pixels from the loss. Background gaussians receive no
gradient: not rewarded, not punished. Whatever the COLMAP seeding placed there
survives training untouched.

Ruled out, so nobody re-tests them:

- **COLMAP seeding is not the difference.** Both routes seeded comparably (69,631
  points premade vs 71,730 separate-mask), because the RGB underneath a premade
  image's transparent region is the ORIGINAL background, not black, so COLMAP sees
  the full scene either way.
- **Mask file location is not a factor.** Masks beside the images and masks in a
  parallel tree behave identically. Both are `mask_path` entries in the end.
- **Resolution is not a factor.** The effect is the same at downscale 2 and 4.

## Consequence for the two modes

They are not two ways of doing one thing. They answer different questions:

- **Loss masking** (`mask_path`): "do not let these pixels influence the model."
  Correct when the masked region is *unreliable* (motion, glare, a moving operator)
  but the scene is still the subject.
- **Alpha compositing** (RGBA): "there is nothing here." Correct when the masked
  region is genuinely not part of the reconstruction target.

For subject-only phenotyping the second is what is wanted, and it is what the
pipeline currently cannot do from a separate mask file.

## Fix

Composite each mask into its photograph as an alpha channel while staging the SfM
inputs, writing RGBA PNGs, so nerfstudio receives premade alpha images and takes the
supervised path. The mask files stay the input format; the compositing becomes the
pipeline's job rather than a manual pre-step.

Notes for whoever implements it:

- The staged tree already exists (`stage_sfm_inputs`) and is already the only thing
  COLMAP and ns-process-data see, so this is the natural place.
- RGBA requires PNG output, so `transforms.json` will carry `.png` paths. Confirm
  ffmpeg's downscale chain preserves the alpha channel, since the image pyramid is
  built by ffmpeg and a silently flattened alpha reproduces the current defect while
  looking correct.
- COLMAP can keep running on plain RGB. It ignores alpha anyway, and the premade
  route proved a full-background feature set still yields usable poses.
- Keep loss masking available; do not replace one mode with the other. Suggested
  config shape: `use_masks = false | loss | alpha`, with `alpha` recommended for
  subject-only work and `loss` for unreliable-region work.
