# Operator instructions

Use this file for plain-language requests to run a photoset. The INI and existing scripts own pipeline behavior. Do not write new pipeline code for a run.

## Config

- Use the directory containing photographs. If the user gives a capture directory with `images/`, use that child. For `{data_root}/datasets/{collection}/{capture}/images`, set `data_root` to the parent of `datasets` and `images_subpath` to the path after it. Example: `/home/alex/data/datasets/welzo-M1/images` maps to `/home/alex/data` and `welzo-M1/images`. `{capture_dir}` is the parent of `images/`.
- If no existing config or `datasets/` ancestor identifies where outputs belong, ask for the data root.
- Copy the matching config, or `pipeline/configs/example.ini` if none matches. Save a new config under `{data_root}/configs/`, outside Git. In a fresh copy, clear `[run] date`, `run_count` and `colmap_version`; after dry-run, verify the new run id has no existing `colmap/` or `outputs/` directory.
- Map iterations to `[train] max_num_iterations`, downscale to `[train] downscale_factor`, split to `[train] eval_mode` and its split value. For test runs, explicitly set downscale to 4 unless the user specifies otherwise. Keep `quit_on_train_completion = true`. Map other settings only to keys in `pipeline/configs/README.md`.
- Before noninteractive WSL commands, export `PATH="$HOME/opt/colmap-prefix/bin:$HOME/.local/bin:$PATH"` so `uv`, COLMAP and ffmpeg resolve. Preserve `[colmap] extra_args = --gpu_index 0` on WSL2.

## Optional mask prompt

SAMask is a separate pre-step. This machine's checkout is `/home/alex/Downloads/samask`; run from its root. Do not install code or weights.

- `{capture_dir}` is the parent of the photoset's `images/`. For a mask prompt, require canonical `.../capture/images` layout; `mask_variant` is refused otherwise.
- `{mask_dir}` is `{capture_dir}/masks` or `{capture_dir}/masks_{slug}`. Reuse only if its `samask-runs.jsonl` has a completed `end` record whose `content_settings.prompt` and `score_thresh` match, `written` equals the photo count, and `failed = 0`. Leave `mask_variant` blank for `masks/`; name the slug for `masks_{slug}/`. If several sets exist and the default set is ambiguous, stop and ask.
- For a new set, choose an unused slug with lowercase letters, digits and hyphens. Run `uv run python 01_segmask_sam3_hq.py --images {photoset_dir} --masks {capture_dir}/masks_{slug} --prompt "{mask_prompt}" --preview-ratio 0 --dry-run`; review, then rerun without `--dry-run`. Add `--score-thresh {score_thresh}` only if requested. Require exit 0; keep CPU default and never overwrite an existing set.
- For masked runs, set `use_masks = true`, `mask_extensions = .png`, and `image_extensions` to the photo extension, such as `.jpg`; the two lists cannot overlap. If photos are PNG, stop and report this config limitation. Without a prompt, do not run Samask.
## Run

- Run inside the Linux distro holding the checkout. Check no other reconstruction uses the single GPU. If the local model is also on that GPU, unload it in the same Bash call as the pipeline command, using the endpoint and model supplied by the session. If those are unavailable, stop after dry-run and report the command.
- First run `uv run python pipeline/run_pipeline.py --config {config} --dry-run`. Check inputs, masks, run id and output paths. Stop on refusals or unexpected paths; never bypass a gate.
- If the user asked to run, execute `uv run python pipeline/run_pipeline.py --config {config}`. Project comparisons use `eval_mode = all`, then `fraction` with `train_split_fraction = 0.9` on the same run id and COLMAP workspace. Use `--from-stage train --run-id {run_id}` for pass two. Only the fraction pass measures held-out views.
- Report config, run id, masks, export and held-out metrics. The full `QUICKSTART.md` is for human installation reference, not routine agent context.
