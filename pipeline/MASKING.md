# Masking: why a mask file and a premade alpha image are not the same thing

Status: **resolved in v0.2.0.** `use_masks` is alpha compositing, applied as a
pre-step. What this file documents is why that is the only mode, since the
alternative it replaced looks interchangeable with it and is not.

The defect this file was opened for, recorded earlier on 2026-08-18: mask support
was first built on nerfstudio `mask_path` entries, which suppress background
*supervision* but not background *geometry*. The evidence below is why that route
was removed the same day rather than kept as an option. Read it before proposing
any masking change; the two mechanisms are easy to conflate and the difference is
two orders of magnitude in the result.

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

**Confirmed a second time against the pipeline's own compositing**, once it
existed, which is the comparison that actually justifies the design rather than
comparing against a hand-made artefact. Same capture, same settings, downscale 2,
30,000 iterations:

| route | gaussians |
|---|---|
| pipeline alpha compositing | 2,385 |
| separate mask file (`mask_path`) | 96,938 |

A 41x reduction, into the range the manual route produced. Cite this pair, not
the first, when the claim being supported is about what the pipeline does today;
the first pair compares the *manual* route against `mask_path` and predates the
implementation.

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
pipeline does.

## What was built

Each mask is composited into its photograph's alpha channel as a **pre-step**,
before any stage runs, turning raw photographs plus separate masks into the
masked-image dataset the pipeline has always consumed. Every stage afterwards is
unchanged and unaware. It delegates to `scripts/rgb-mask/rgb-mask-batch.py`,
which had already done exactly this by hand for months.

Two things about the shape are worth recording, because an earlier draft of this
section proposed otherwise and was wrong:

- **A pre-step, not part of SfM staging.** Compositing before `ns-process-data`
  renames anything means nothing has to be re-paired afterwards, and the mask
  pyramid is simply the image pyramid. Building it into `stage_sfm_inputs`
  instead would have kept the frame-to-source mapping problem: a `struct` parser
  for COLMAP's `images.bin`, a `colmap_im_id` lookup, and per-file ffmpeg mask
  downscaling. Choosing the pre-step deleted about 165 lines net.
- **COLMAP still sees plain RGB.** It ignores alpha, and the premade route proved
  a full-background feature set still yields usable poses.

## Loss masking was deliberately NOT kept

An earlier draft of this file recommended keeping both modes, with a config shape
of `use_masks = false | loss | alpha`. **That recommendation was considered and
overridden on 2026-08-18 (Alex).** The `mask_path` route was removed outright and
`use_masks` stayed a boolean.

The reasoning: the route does not do what a reader would assume it does, the
assumption is expensive to discover (a 41x wrong answer that looks like a
successful run), and no current work needs the unreliable-region case that would
justify carrying it. A mode nobody uses, which silently produces a plausible
wrong result when misunderstood, is worse than an absent one.

**Do not reintroduce a loss-masking mode without a concrete use case**, and if
one arrives, name it `loss` explicitly rather than folding it back under
`use_masks`, so no one can select it by accident.

## Where the masks come from, and why that stays a subprocess

This pipeline **consumes** masks. It does not make them. Generation lives in a
separate repository, `samask`, which drives SAM3 with a text prompt for the
subject and refines the box with SAM2-HQ. Today that is a manual step you run
before the pipeline; making it a pipeline stage is open work.

**When it does become a stage, it must invoke samask as a subprocess, never as an
import.** This is a hard constraint, not a style preference, and it is worth
writing down because "just import it" is the obvious first instinct:

- samask is Python 3.12 on torch 2.7.0.
- This pipeline is Python 3.10 on torch 2.4.1+cu121, and **the torch version is
  forced**: 2.4 is the newest torch with a prebuilt `sm_75` gsplat wheel, which is
  what removes the `nvcc` requirement and the ten-minute gsplat JIT.

Two torch versions cannot share one interpreter, so the two tools cannot share an
environment. A subprocess in its own `uv` environment is the only shape that
works, and it is the shape this pipeline already uses for `colmap`, `ffmpeg` and
every `ns-*` command.

That boundary has a second benefit worth preserving deliberately: **no samask
code enters this repository**, so this repository's licence and authorship story
stay independent of samask's, which is unsettled.
