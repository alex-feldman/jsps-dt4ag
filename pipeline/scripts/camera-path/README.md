# Programmatic Nerfstudio camera paths

Build a camera-path JSON from a solved capture's Nerfstudio transforms.json. The generator sorts frames by numeric filename, smooths camera positions and gaze, resamples the path to a chosen duration and frame rate, and writes a provenance sidecar. It does not edit the capture, COLMAP workspace or trained model.

The camera follows the southwest-to-southeast capture corridor. A shallow bow along the smoothed camera gaze adds modest parallax. The last part eases into a small rise, pan and downward tilt. Each position offset is capped at one median source-camera step so the route stays near observed camera positions.

The script uses only Python's standard library. Nerfstudio's ns-render camera-path reads the JSON and renders with the trained run config, not the exported PLY.

## Build the teaching-greenhouse preview path

From the repository root in WSL:

    DATA=/home/alex/data
    CAPTURE=dji-20261022051311-0107-d-teaching-gh-0500ms-v1
    RUN_ID=ua-ceac-teaching-gh-manual-colmap_261025-01-3120
    PATH_DIR="$DATA/evals/$CAPTURE/flythrough"
    mkdir -p "$PATH_DIR"

    uv run python pipeline/scripts/camera-path/camera-path.py \
      --transforms "$DATA/colmap/$CAPTURE/$RUN_ID/transforms.json" \
      --out "$PATH_DIR/sw-se-flythrough-camera-path.json"

Defaults make a 24-second, 24 fps path at 720x960. The provenance sidecar records the source hash, frame order, smoothing, offsets, field of view and output settings. Use --dry-run to inspect the planned path without writing files. The script refuses to overwrite an existing path or sidecar.

## Render the low-resolution preview

Use the completed 30k ds4 training config for this capture:

    RUN_CONFIG="$DATA/outputs/$CAPTURE/$RUN_ID/splatfacto/2026-10-26_000741/config.yml"

    uv run ns-render camera-path \
      --load-config "$RUN_CONFIG" \
      --camera-path-filename "$PATH_DIR/sw-se-flythrough-camera-path.json" \
      --output-path "$DATA/exports/$CAPTURE-$RUN_ID-sw-se-flythrough-preview.mp4" \
      --downscale-factor 2 \
      --output-format video

The JSON path is 720x960; --downscale-factor 2 makes the preview 360x480. Review framing and motion before making a larger render or increasing path offsets. Raise output resolution by regenerating the path with --width and --height at the same 3:4 aspect ratio, then use --downscale-factor 1.

wsl-guard.py currently wraps run_pipeline.py, not ns-render. Before rendering on WSL, check Windows available memory and the free space on the drive holding the WSL disk; use the guard's documented start thresholds as the minimum. Rendering uses the RTX 4050 GPU when the installed Nerfstudio config resolves to CUDA.

## Other captures

Point --transforms at a different Nerfstudio transforms.json and use a new --out. Keep the transforms from the same COLMAP/Nerfstudio run as the model config that will render the path. The script expects numeric image frame ids and Nerfstudio's z-up camera coordinates. Begin with these conservative defaults; review the output before increasing the bow or end-reveal angles.
