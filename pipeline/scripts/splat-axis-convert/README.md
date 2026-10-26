# Splat axis conversion

Convert a Nerfstudio Gaussian-splat PLY whose header declares Vertical Axis: z
into the equivalent y-up coordinate convention, without retraining. The conversion
uses a fixed right-handed rotation, preserves scene units and origin, and rotates
Gaussian positions, normals, orientations and view-dependent spherical harmonics.

The script refuses an input with a missing or non-z Vertical Axis header. It writes a new PLY and an axis-conversion JSON provenance sidecar. It never changes the source or overwrites outputs.

## Command

From the repository root, set the source export and a new output path:

    SOURCE=/home/alex/data/exports/{source-export}.ply
    Y_UP="$(dirname "$SOURCE")/$(basename "$SOURCE" .ply)-yup.ply"

    uv run python pipeline/scripts/splat-axis-convert/splat-axis-convert.py \
      --input "$SOURCE" \
      --out "$Y_UP" \
      --viewer supersplat

The viewer option also writes a separate SuperSplat display-only PLY and a viewer
JSON sidecar. SuperSplat applies its own rotation on load. Use the plain y-up PLY
for other viewers, analysis and measurement.

This converts coordinate convention. It does not measure gravity or calibrate scale.
It uses the source PLY's declared vertical axis and does not translate the scene.
For a measured plant frame with an explicit origin or scale, use the organ-analysis
README section Frame calibration.