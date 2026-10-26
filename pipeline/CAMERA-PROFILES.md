# Camera profiles

This is an empirical record of COLMAP models and Nerfstudio camera data seen
in this project. A fitted focal length or distortion coefficient is an output
from one reconstruction, not a factory calibration to copy into another run.
Record the capture mode as well as the camera body: field of view, electronic
stabilization and in-camera dewarping can change the image geometry.

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
