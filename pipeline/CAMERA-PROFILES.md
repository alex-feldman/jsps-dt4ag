# Camera profiles

This is an empirical record of COLMAP models and Nerfstudio camera data seen
in this project. A fitted focal length or distortion coefficient is an output
from one reconstruction, not a factory calibration to copy into another run.
Record the capture mode as well as the camera body: field of view, electronic
stabilization and in-camera dewarping can change the image geometry.

## Lens distortion and camera models: what the pipeline does

**The pipeline does not correct lens distortion itself, and you do not need to
correct it first.** Stage 1 runs `colmap automatic_reconstructor`, which fits a
camera model, including distortion, to your images. Stage 2 runs
`ns-process-data images --skip-colmap`, which copies that fit into
`transforms.json` (see the header of `pipeline/run_pipeline.py`). Your image
files are never rewritten.

During training, Nerfstudio 1.1.5, the version this repository pins, undistorts
each image in memory with the fitted parameters. It then crops the result to the
region that has no empty border and adjusts the intrinsics to match
(`_undistort_image` in `nerfstudio/data/datamanagers/full_images_datamanager.py`,
which the `splatfacto` method uses). So the training images can show a slightly
smaller field of view than your originals.

What to do with that:

1. **Use one camera at one setting for a capture.** By default the pipeline asks
   COLMAP to share one set of intrinsics (`single_camera` and
   `single_camera_per_folder`, both `1`; see `pipeline/configs/README.md`). Do not
   mix cameras, lenses, zoom levels or field-of-view modes in one capture. Lock
   zoom and focus if the camera lets you.
2. **Prefer the least wide mode that still frames the subject.** Strong
   wide-angle distortion is harder for the fit to get right; the second capture
   in the Osmo precedent below is the example of it going wrong. Record the
   field-of-view and stabilization mode with the capture, because they change the
   image geometry.
3. **Do not crop, rescale, dewarp or stabilize in post unless you must.** The fit
   assumes every frame of a camera shares one geometry. If you do change geometry
   after capture, apply exactly the same change to every frame of that camera and
   note it with the capture. Keep one camera's frames at one size and
   orientation (step 4 of the video-derived image sequences section in
   `QUICKSTART.md` asks you to check this).
4. **Check the solve before you train.** Run
   `pipeline/scripts/camera-review/camera-review.py` after stages 1 and 2 and read
   `cameras` in `camera-checks.json`. A focal length or distortion coefficient far
   from what the camera should give, or a flagged camera step, means the solve is
   suspect. Compare the two Osmo captures below: for that camera the usable fit had
   `f = 1510.7 px` and `k1 = 0.0034`, the failed one `f = 9316.4 px` and
   `k1 = 13.3`. Fix the capture or the sampling and solve again; do not train on
   it hoping the training will repair it.

## DJI Osmo Action 6

The two captures below came from the same camera and have the same
autorotated image size and sampling interval. The saved capture records do not
identify the selected FOV or stabilization mode, so that part of the match is
unverified.

Both configs used `[colmap] data_type = video`, `single_camera = 1`, and
`single_camera_per_folder = 1`. Neither config overrides
`ImageReader.camera_model`; COLMAP 3.12's feature-extractor default is
`SIMPLE_RADIAL`. The raw sparse models confirm that model. The measurements
below came from each model's `cameras.bin`, converted to `cameras.txt` with
COLMAP's `model_converter`.

| Capture | Raw COLMAP camera estimate | Sparse-model evidence | Nerfstudio `transforms.json` |
|---|---|---|---|
| Reference video A, 500 ms sampling | `SIMPLE_RADIAL`, 2880 x 3840; f=1510.707 px, cx=1440, cy=1920, k1=0.003421 | 180/181 registered; 152,783 points; mean track length 4.858; mean reprojection error 1.151 px; largest/median camera step ratio 2.80. | Says `camera_model = OPENCV`; the recorded focal length and k1 match the fitted values at left. |
| Video B, 500 ms sampling; points judged incoherent on visual review | `SIMPLE_RADIAL`, 2880 x 3840; f=9316.425 px, cx=1440, cy=1920, k1=13.305047 | 160/161 registered; 40,873 points; mean track length 2.539; mean reprojection error 1.270 px; step from `frame_000056.png` to `frame_000057.png` is 5.060 scene units, 18.96x the median. COLMAP also logged 71 Ceres linear-solver failures. | Also says `camera_model = OPENCV`; the focal length and k1 match the fitted values at left. |

**Starting choice for this camera:** use COLMAP `SIMPLE_RADIAL` with one
shared camera for a video whose FOV and lens processing match these captures.
Reference video A is the useful precedent. Video B used the same raw COLMAP
model, so switching to COLMAP `OPENCV` is not supported by this comparison;
its fitted parameters went badly wrong despite the matching model. Its camera
path also had an 18.96x-median step at 27.5 to 28.0 seconds, and the COLMAP log
recorded repeated Ceres solver failures. If a new solve produces similarly
extreme intrinsics or a large step warning, inspect the named frames and
matching results before training. Do not copy video A's intrinsics as fixed
calibration values.

### Reading the camera model

The model name in Nerfstudio's `transforms.json` is not necessarily the raw
COLMAP model name. In these two runs, Nerfstudio's file says `OPENCV`, while
the actual COLMAP model in `sparse/0/cameras.bin` is `SIMPLE_RADIAL`.
Inspect the COLMAP model directly when choosing a model for a manual solve:

```bash
MODEL_TXT=$(mktemp -d)
colmap model_converter \
    --input_path "$COLMAP_WS/sparse/0" \
    --output_path "$MODEL_TXT" \
    --output_type TXT
cat "$MODEL_TXT/cameras.txt"
```

COLMAP's [camera-model documentation](https://colmap.github.io/legacy/3.12/cameras.html)
explains the projection families and warns that grossly wrong fitted focal
lengths or distortion coefficients can indicate a model or solve problem.
DJI lists Standard (Dewarp), Wide and Natural Wide FOV options for the
[Osmo Action 6](https://www.dji.com/support/product/osmo-action-6); save that
setting and the stabilization mode in future capture notes.
