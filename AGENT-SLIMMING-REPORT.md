# Agent Slimming Branch Report

Date: 2026-10-07
Branch: agent-slimming
Base: main at d0ecbfc

## Goal

Make the repository's only agent-facing instructions an operator guide for translating plain requests into existing dataset configs and pipeline scripts. The operator should not need developer setup or pipeline internals in context.

## Repository changes

- Added root AGENTS.md as the single agent-facing operator guide. It maps a photoset path to dataset configuration, copies a matching INI into the external data-root config directory, clears run metadata, maps iteration/downscale/split requests to existing keys, and requires a pipeline dry-run before execution.
- Added optional SAMask handling: canonical capture/images layout, variant naming, provenance/completeness checks in samask-runs.jsonl, preview suppression, no overwrite, mask extension settings, and explicit stop conditions.
- Added WSL execution requirements: PATH setup, COLMAP GPU index, check for competing GPU use, and unload the local Ollama model in the same shell call before a GPU run when the endpoint and model are available.
- Added the same-run all-then-fraction comparison procedure and the held-out metric reporting requirement.
- Updated README.md and pipeline/README.md to point operators to AGENTS.md. Updated CHANGELOG.md to record the operator guide and this trial report.
- Added this report to record the implementation, trial evidence, and remaining limitations.
- Kept pipeline implementation and existing scripts unchanged. Removed the temporary project opencode.json used for model testing. The earlier draft pipeline/AGENT-RUN.md was removed so AGENTS.md remains the sole agent-facing document.

## Review and safety corrections

An independent high-effort review caught several places where the guide needed to match existing code behavior. The guide now requires mask provenance to include a completed end record, matching prompt and score threshold, a written count equal to the photo count, and zero failures. It uses .png for masks and the actual photo extension for images, and stops if the extensions overlap. It also restricts masks to the canonical images layout, prevents overwriting existing mask sets, requires a dry-run, and does not bypass pipeline refusals. The comparison instructions keep the same run id and COLMAP workspace between all and fraction passes.

## Local model and harness trial

The test used native WSL OpenCode 1.18.34 with local Ollama model alias qwen35-agent-48k at a 49,152-token context. The Windows OpenCode shim resolved to a different version, so the Linux binary was invoked directly. A temporary project OpenCode config granted access to external data directories and routed to the WSL-only Ollama proxy; that config has been removed.

A 48K-context request successfully had the model read only AGENTS.md, copy the matching Welzo M1 config to /home/alex/data/configs/welzo-m1_20261007.ini, set downscale_factor=4, and run the pipeline dry-run. The dry-run accepted run id welzo-m1_261007-01-3120, 24 images, evalfrac90, downscale 4, and GPU 0.

The next request exposed an instruction-following weakness: the model ignored the requested eval_mode=all, tried importing a nonexistent PipelineConfig symbol, and ultimately invoked the existing config as evalfrac90. It also failed to unload the Ollama model before the first attempt; the manager unloaded it before allowing the GPU run. This is evidence that deterministic gates help, but the model still needs close supervision around multi-step comparisons and GPU cleanup.

The model then started a 10,000-iteration evalfrac90 run for welzo-m1_261007-01-3120. Windows restarted before it completed. The output contains a COLMAP database and sparse/project.ini only; there is no completed sparse model, training output, export, or metric result. The run did not reproduce the previous reconstruction. The dump analysis found a 0x50 PAGE_FAULT_IN_NONPAGED_AREA at nvlddmkm.sys, preceded by GPU watchdog dumps 0x141 and 0x117 in the same driver. Current nvidia-smi reports driver 617.14. The crash interrupted the trial, but the dumps do not prove the pipeline itself caused the driver failure.

## What this establishes

- A small local model can follow the single-file operator guide for config creation and a valid dry-run at 48K context.
- A real completed reconstruction has not yet been demonstrated with this model.
- The two-pass comparison and SAMask-plus-pipeline prompt remain unvalidated end to end.
- The model's mistake on eval_mode=all shows that the guide cannot replace checking the generated config and command before a GPU run.
- GPU training should be retried only after the NVIDIA driver instability is addressed. Preserve the incomplete run as evidence; do not treat it as a successful result.

## Test cleanup and next step

The temporary opencode.json is removed from the worktree. The WSL proxy stopped when Windows rebooted. The temporary Windows firewall rule named Temporary WSL Ollama test may still exist; remove it from an elevated PowerShell session when WSL Ollama testing is finished:

    Remove-NetFirewallRule -DisplayName 'Temporary WSL Ollama test'

After addressing the NVIDIA driver issue, rerun the two requested operator prompts against a known photoset: a plain 10,000-iteration run, then a SAMask prompt such as plant followed by the same pipeline. Compare outputs with the known prior 10,000-iteration reconstruction and record config, run id, context size, VRAM placement, performance, and final artifacts.
