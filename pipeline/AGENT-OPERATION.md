# Operating the pipeline with an agent on a local model

The pipeline is built to be driven by an AI coding agent as well as by hand:
one command, every decision in the config file, no interactive prompts, and
exit codes a caller can branch on (`RUN-FLOW.md`, "Driving it: by hand or by
agent"). **You do not need a subscription to a hosted model for that.** This
page sets up [OpenCode](https://opencode.ai/), an open-source terminal agent,
against a model served on your own machine by [Ollama](https://ollama.com/).

Everything here is optional. Every step an agent performs can also be done by
hand with `QUICKSTART.md`.

## What the agent reads

**`AGENTS.md` at the repository root is the agent's entry point.** OpenCode
looks for `AGENTS.md` in the directory it starts in and the directories above
it, and loads it as instructions, so launch it from the repository root. That file routes the agent to
the right document for the task and sets the operator rules: a person runs long
stages unless they explicitly ask the agent to, the agent prepares configs and
commands and runs the dry run, and it does not install software, edit pipeline
code, delete data or push changes during a capture task unless asked.

## 1. Install the two programs

Install both inside the same Linux or WSL2 environment as the pipeline:

- **Ollama**, following its Linux instructions at <https://ollama.com/download>.
  It serves models on `http://127.0.0.1:11434` and exposes an
  OpenAI-compatible API under `/v1`.
- **OpenCode**, following <https://opencode.ai/docs/>. The commands below were
  checked against OpenCode 1.18.35 (`opencode run --help`).

**The model and the pipeline share one GPU.** On a small GPU, a model loaded in
VRAM can leave too little for COLMAP's matcher or for training. Use the agent to
prepare and dry-run, and have nothing else on the GPU while a stage that needs
it runs. `ollama ps` shows what is loaded, and `ollama stop {model}` unloads it.

## 2. Choose and prepare a model

Pick a model from Ollama's library that supports **tool calling**: an agent
works by calling tools (read a file, run a command), and a model without that
cannot act. This repository does not certify any particular model; small models
make more mistakes, so review what the agent changed (section 5).

```bash
ollama pull {model}
```

**Give it a context window large enough for an agent.** The agent's
instructions, `AGENTS.md`, the documents it opens and every command's output all
share the model's context. Ollama's default window is set for chat, not for an
agent session, and it differs between Ollama versions, so set it explicitly.
Make a variant with a larger window:

```bash
printf 'FROM {model}\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create {model}-32k -f Modelfile
```

A larger window needs more memory. Choose the largest the machine holds
alongside the pipeline's own needs.

## 3. Point OpenCode at the local model

OpenCode reads a user config at `~/.config/opencode/opencode.json`. Merge a
provider entry for Ollama into it (keep anything already there):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://127.0.0.1:11434/v1"
      },
      "models": {
        "{model}-32k": {
          "name": "{model} with a 32k context",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

The key under `models` must match the Ollama model name exactly, and
`limit.context` should match the `num_ctx` you set. The model is then addressed
as `ollama/{model}-32k`. `opencode models ollama` lists what OpenCode sees.

## 4. Let the agent reach the data folder, and protect the photographs

Your captures, configs and outputs live under the config's `[paths] data_root`,
outside the repository (`LAYOUT.md`). OpenCode asks for approval before it
touches a directory outside the one it was started in (its `external_directory`
permission), and an "allow always" answer lasts only for the current session.
To avoid answering on every run, add persistent rules to the same
`~/.config/opencode/opencode.json`:

```json
{
  "permission": {
    "external_directory": {
      "~/data/**": "allow"
    },
    "edit": {
      "~/data/datasets/**": "deny"
    }
  }
}
```

Replace `~/data` with your own `data_root` and `datasets` with your
`datasets_dirname`; `~` expands to each user's home directory. The first rule
lets the agent read and write under the data root. The second stops OpenCode's
file-editing tools from changing your input photographs and masks, which the
pipeline itself never writes (`LAYOUT.md`, "Input versus derived").

Two limits to know:

- **The `edit` rule does not cover shell commands.** A command the agent runs
  can still write into `datasets/`. Keep instructions that protect the inputs in
  your request, and review commands before you approve them.
- **A config file outside both the repository and the data root** triggers its
  own approval. Approve that one path, or add a separate, narrow rule for it.

Keep these rules limited to the directories you actually use for pipeline data.
OpenCode's permission syntax is documented at
<https://opencode.ai/docs/permissions/>. Do not use `opencode run --auto`, which
approves every request that is not explicitly denied.

## 5. A first request: prepare a dry run

Start OpenCode from the repository root with the same `PATH` a scripted run
needs (`QUICKSTART.md` section 5): the agent's shell is non-interactive, so `uv`
and the COLMAP prefix are not on `PATH` otherwise, and even a dry run runs
`colmap` to put its version in the run id (`Dt4agConfig.resolve_colmap_version`
in `dt4ag_config.py`, unless `[run] colmap_version` is set).

```bash
cd {repository root}
export PATH="$HOME/opt/colmap-prefix/bin:$HOME/.local/bin:$PATH"
opencode run --interactive --model ollama/{model}-32k "{request}"
```

`--interactive` runs OpenCode's interactive terminal mode, so you see and answer
permission requests as they come. The request is the positional argument;
`-p` is a server password in `opencode run`, not a prompt. An example request, with the placeholders filled in for
your capture:

```text
Prepare a dry run for the capture at {data_root}/datasets/{capture}/images with
10,000 iterations and no masks. Follow AGENTS.md. Copy
pipeline/configs/example.ini to {data_root}/configs/{capture}.ini and set only
data_root, images_subpath, max_num_iterations = 10000,
quit_on_train_completion = true and use_masks = false. Validate it with
uv run python pipeline/dt4ag_config.py {data_root}/configs/{capture}.ini, then run
uv run python pipeline/run_pipeline.py --config {data_root}/configs/{capture}.ini --dry-run.
Do not search for the photographs elsewhere, install packages, train, or start
any stage. Report the config path, the resolved image directory, the image
count, the run id, the iterations and the mask setting, then stop.
```

Then check the result yourself: read the config the agent wrote against
`configs/README.md`, and read the dry run's output. A dry run validates the
config and prints every command, but it leaves no record and skips several
checks a real run makes (`QUICKSTART.md` section 5, "Do a dry run first", which
also shows how to save its output). The real run is yours to start, or the
agent's only if you ask for it explicitly.

## What this page does not cover

Running Ollama on a different machine or on the Windows side of WSL2, choosing
between specific models, and judging a model's reliability as an operator.
Those depend on your hardware and are not tested here.
