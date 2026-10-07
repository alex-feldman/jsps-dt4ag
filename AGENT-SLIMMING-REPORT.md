# Agent Slimming Branch Report

Date: 2026-10-07
Branch: agent-slimming
Base: main at d0ecbfc

## Goal

Keep one concise agent-facing document for operators. It should map plain requests to existing configuration keys and pipeline scripts without loading developer setup or pipeline internals.

## Repository changes

- Added root AGENTS.md as the only agent-facing guide. It covers safe photoset-to-config mapping, copying rather than editing source configs, mask defaults, SAMask provenance, WSL GPU setup, dry-run limits, long-running GPU calls, two-pass comparisons, and result reporting.
- Updated README.md and pipeline/README.md to point to AGENTS.md and this report. Updated CHANGELOG.md.
- Added this report with the implementation and trial evidence.
- Pipeline code and scripts were not changed. The temporary project opencode.json used for testing was removed. An earlier pipeline/AGENT-RUN.md draft was removed so AGENTS.md remains the only agent-facing guide.

## Code reviews and corrections

Two independent high-effort reviews were checked against repository and SAMask source. The guide now handles copied configs that already enable masks, requires the WSL GPU index flag even when the template leaves extra_args empty, rejects photosets outside the data-root datasets tree and unsafe relative paths, and keeps the source config intact. It also uses the actual SAMask record schema and default 0.50 threshold, keeps the default preview output, marks visual review as a human check, and refuses partial mask results.

The guide also says what the pipeline dry-run does not validate: runtime prerequisites, mask compositing and coverage, downscale files, and held-out counts from registered images. Fraction runs must have at least one held-out view. Metrics are reported only when viewer+tensorboard produces them. Run IDs are generated from existing output directories, so the separate manual collision check was removed.

## Local model and harness trial

The test used native WSL OpenCode 1.18.34 with Ollama alias qwen35-agent-48k at 49,152-token context. The Windows OpenCode shim resolved to a different version, so the Linux binary was invoked directly. A temporary project config granted access to external data directories and routed through the WSL-only Ollama proxy; it has been removed.

Before the final review corrections, the model read only AGENTS.md, copied the Welzo M1 config to /home/alex/data/configs/welzo-m1_20261007.ini, set downscale_factor=4, and ran a valid dry-run. It accepted run id welzo-m1_261007-01-3120, 24 images, evalfrac90, downscale 4, and GPU 0. The final guide changes were not retested through OpenCode.

A later request exposed an instruction-following failure: the model ignored eval_mode=all, attempted to import a nonexistent PipelineConfig symbol, and ultimately invoked the config as evalfrac90. It also did not unload Ollama before its first GPU attempt; the manager unloaded the model before proceeding.

The model then started a 10,000-iteration evalfrac90 run at about 03:34. Windows restarted before completion. Only a COLMAP database and sparse/project.ini remain; there is no completed sparse model, training output, export, or metric. The run did not reproduce the prior reconstruction.

## Restart evidence and test limits

The 0x50 crash dump faults inside nvlddmkm.sys. Two earlier live dumps, 0x141 and 0x117, also identify the NVIDIA driver and GPU timeout recovery. The current post-restart driver is 617.14 (Windows driver version 32.0.16.1714). This is strong evidence of an NVIDIA graphics-driver failure during GPU work, but does not prove the pipeline caused the failure or distinguish driver, hardware, and memory corruption.

A completed reconstruction, the all/fraction comparison, and the SAMask-plus-pipeline prompt remain unvalidated. Do not count the interrupted run as a result.

## Next step

After addressing NVIDIA driver stability, rerun the plain 10,000-iteration prompt and the SAMask plant prompt against the known photoset. Compare with the previous 10,000-iteration reconstruction. Record config, run id, context, VRAM placement, speed, and final artifacts.

The WSL proxy stopped at reboot. The temporary Windows firewall rule may still exist; remove it when WSL Ollama testing is finished with elevated PowerShell:

    Remove-NetFirewallRule -DisplayName 'Temporary WSL Ollama test'
