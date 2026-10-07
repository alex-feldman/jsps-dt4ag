# Operator instructions

Use this for plain-language requests to run a photoset. Existing INI files and scripts own pipeline behavior. Do not write pipeline code for a run.

## Config

- Use photos only under {data_root}/datasets/. If the photoset is elsewhere, ask for a safe data root. Never use an absolute path or .. in images_subpath.
- For {data_root}/datasets/{collection}/{capture}/images, set data_root to the parent of datasets and images_subpath to {collection}/{capture}/images. {capture_dir} is the parent of images.
- Copy the matching config, or pipeline/configs/example.ini. Never edit or overwrite the source. Save the copy under {data_root}/configs/. Clear [run] date, run_count and colmap_version so the runner derives a new run id.
- Map iterations to max_num_iterations, downscale to downscale_factor, and split to eval_mode and its split value. For tests, use downscale_factor=4 unless asked otherwise. Keep quit_on_train_completion=true. Map other options only to pipeline/configs/README.md.
- Without a mask request, set use_masks=false and clear mask_variant unless the user names an existing mask set.
- In WSL2, ensure [colmap] extra_args includes --gpu_index 0. Before noninteractive commands, export PATH="$HOME/opt/colmap-prefix/bin:$HOME/.local/bin:$PATH".

## Optional SAMask prompt

SAMask is separate. Use /home/alex/Downloads/samask. Do not install code or weights.

- A mask prompt requires {capture_dir}/images. Stop and ask if that layout is absent.
- Use {capture_dir}/masks for the default set or {capture_dir}/masks_{slug}. If several sets exist and the default is unclear, ask.
- Inspect existing provenance with uv run python 01_segmask_sam3_hq.py --images {capture_dir}/images --masks {mask_dir} --show-records. Reuse only a completed end record whose prompt, score_thresh (requested value or default 0.50), and input extension match; require written=photo_count and failed=0. A start with no matching end is incomplete.
- For a new set, choose an unused slug of lowercase letters, digits and hyphens. Run uv run python 01_segmask_sam3_hq.py --images {capture_dir}/images --masks {mask_dir} --prompt "{mask_prompt}" and add --score-thresh only if requested. Keep CPU default and preview default. On nonzero exit, stop, report failures, and do not use the set.
- Set use_masks=true, mask_variant to the slug (blank for masks/), mask_extensions=.png, and image_extensions to the photo extension. The lists cannot overlap; if photos are PNG, stop and report the limitation.
- Report the preview directory and say whether a human visually reviewed it. Do not claim visual review from the provenance record.

## Run

- Check that no other reconstruction uses the single GPU. If Ollama is using it, unload the session's model before the pipeline command in the same blocking shell call. If the endpoint/model or a sufficiently long blocking command is unavailable, stop after dry-run and give the exact run command. Do not background a GPU run and then continue local-model inference.
- Run uv run python pipeline/run_pipeline.py --config {config} --dry-run first. Check config, input count, generated run id, and output paths. A dry-run does not run prerequisites, composite masks, verify mask coverage or downscale files, or check registered-view holdout counts.
- If asked to run, execute the same command without --dry-run. For fraction mode, require photos - ceil(photos * train_split_fraction) >= 1 before training.
- Project comparison: run eval_mode=all, then fraction with train_split_fraction=0.9 on the same run id and COLMAP workspace. Pass two uses --from-stage train --run-id {run_id}. Only fraction holds out views; all-mode evaluation measures fit, not generalization.
- Held-out metrics are produced only with vis=viewer+tensorboard. Set it when requested; otherwise report "none produced".
- Report config, run id, masks and review status, exports, and metrics or why none were produced.
