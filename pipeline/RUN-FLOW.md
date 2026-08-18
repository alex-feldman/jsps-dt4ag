# What a run actually does

The order of operations in `run_pipeline.py`, and every point at which it refuses
to continue.

**Scope, deliberately narrow.** The four stages are described for a reader in the
wiki page *Running the 3D Reconstruction Pipeline*; installing and running is
`QUICKSTART.md`; the flags are `README.md`. This file covers the thing none of
those do: the runner's own control flow and its verification gates, which is what
you need when a run fails somewhere unexpected or when you are extending it.

**This file names functions rather than restating their logic**, because the code
owns the mechanism and a prose copy of it would drift. When the two disagree, the
code is right and this file is a bug.

## The shape

Three kinds of work happen, and they are not the same kind:

1. **Resolution and refusal** — everything before any tool runs. Config, run id,
   prerequisites, the image census. Cheap, and it is where a run should die.
2. **A pre-step** — masking. Not a stage. It turns raw photographs plus masks
   into the masked-image dataset the stages already consume, so no stage knows it
   happened. Runs before the stage loop, and its time is not counted in the
   reported pipeline elapsed.
3. **Four stages** — colmap, process, train, export. Selectable and resumable.

## Order of operations

| # | What | Refuses when |
|---|---|---|
| 1 | Load and validate config (`load_config`) | any path missing, any key mistyped, masked output would overwrite the source |
| 2 | Derive the run id, or take `--run-id` | a non-`colmap` first stage names a workspace that does not exist |
| 3 | Prerequisites for the SELECTED stages only (`check_prerequisites`) | `colmap`, `ffmpeg` or an `ns-*` command missing; no CUDA device; GPU architecture absent from the installed gsplat binary |
| 4 | Image census (`count_images`, filtered by `is_photograph`) | no photographs survive the extension filter |
| 5 | Capture provenance (`read_capture_metadata`) | file present but unparseable. Absent is normal and silent |
| 6 | **Pre-step:** composite masks (`composite_masked_images`) | any photograph has no paired mask; the output count does not match afterwards. Skipped entirely when a complete set already exists |
| 7 | Append the run-log row | — |
| 8 | **Stage colmap** | no `sparse/cameras.bin` afterwards |
| 9 | **Stage process** | no `transforms.json`; the downscale pyramid is incomplete (`verify_downscale_pyramid`) |
| 10 | **Stage train** | a pinned `downscale_factor` names a level that is not on disk |
| 11 | Checkpoint discovery (`discover_checkpoint`) | only a `config.yml` and no weights; the newest run directory predates this invocation |
| 12 | **Stage export** | the PLY is missing, is no bigger than a bare header, or declares zero vertices (`verify_export`) |

## The two rules behind all of it

**A tool reporting success is not evidence.** Every subprocess return code is
checked, and then the artefact that stage claims to have produced is looked for on
disk and inspected. This codebase has a documented history of succeeding loudly
while producing nothing.

**Fail early and name the fix.** Prerequisites are checked before anything long
starts, and only for the stages actually selected, so a missing binary costs
seconds rather than being discovered forty minutes into a reconstruction.

## Where masking sits, and why it is not a stage

Masking is applied to the *inputs*, before COLMAP. Both COLMAP and
`ns-process-data` are then pointed at the composited directory
(`sfm_input_path`), which they must share: with `--skip-colmap`, nerfstudio
converts the COLMAP model by looking each of its image names up in the rename map
built from the files it copied, so a name COLMAP saw and `ns-process-data` did not
is a `KeyError`.

Making masking a pre-step rather than a stage is what removes the hardest part of
the problem. Compositing happens before anything is renamed to `frame_NNNNN`, so
nothing has to be re-paired afterwards, and the mask pyramid is simply the image
pyramid. See `MASKING.md`.

## Known reporting gap

The pipeline elapsed time printed at the end excludes the masking pre-step,
because the timer starts after it. On a capture that composites from scratch that
is a material understatement: roughly 10 minutes at the time of writing against a
12 minute reported total.
