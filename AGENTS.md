# Agent instructions

This file is the repo-local entry point for AI agents launched from this checkout.

## Read for the task

- `README.md` for the project boundary and document map.
- `pipeline/QUICKSTART.md` for setup, WSL2, GPUs, memory limits, and terminal commands.
- `pipeline/VIDEO-WORKFLOW.md` for video review, frame extraction, camera review, and reconstruction.
- `pipeline/configs/README.md` for config keys and run archives.
- `pipeline/LAYOUT.md` for capture, input, and derived-data placement.
- `pipeline/RUN-FLOW.md` for stage order and refusal gates.
- `pipeline/COORDINATE-FRAMES.md` before joining an exported splat to per-photograph masks or points.
- `pipeline/AGENT-OPERATION.md` for how this agent session is set up on a local model, the data-folder permissions it runs under, and the dry-run request it should handle.

The code owns mechanisms. These files route the agent to the current implementation and its operating procedure; do not recreate their commands or config defaults from memory.

## Operator and decisions

- Alex runs long pipeline stages in his terminal by default. Prepare the config, preflight, and exact commands. Run a stage as an agent only when Alex explicitly asks for that stage.
- For video work, follow `pipeline/VIDEO-WORKFLOW.md`. Wait for Alex to select the video range and sampling interval, and to review the extracted frames. Stop for camera-path review before training. Ask for the training budget, evaluation split, and usefulness criterion before the first training run.
- Treat task-specific source identity, mask choice, sharing rights, and output purpose as inputs from Alex or the named study record. Do not infer them from sample data or filenames.
- A request for an unmasked full scene means preserve the full image content by setting `use_masks = false`. Training split choices are separate and remain governed by the training procedure.

## Data and runtime

- Keep capture data, runtime configs, logs, models, and exports outside the Git checkout, following `pipeline/LAYOUT.md` and `pipeline/configs/README.md`.
- Do not install software, edit pipeline code, delete data, or push changes during a capture task unless Alex explicitly asks.
- On WSL2, follow the setup, GPU, memory, and guarded-run instructions in `pipeline/QUICKSTART.md`. Use the pipeline dry run before a real stage and preserve its run ID and logs.
