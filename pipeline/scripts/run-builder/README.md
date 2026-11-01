# run-builder

A local web page that asks for a run's inputs (where the photographs are, which
masks, the run settings) and writes a validated pipeline config plus the exact
terminal commands to run, in order, each with one line saying what it does. It
never runs a pipeline stage or samask. The command line stays the interface
that runs things.

| File | What it is |
|---|---|
| `run-builder.html` | The page. One file, inline CSS and JavaScript, no network. Builds the configs and commands in the browser, so it also works opened directly as a file. |
| `serve.py` | Optional local server (standard library only). Serves the page and adds what a file cannot: folder browsing, photograph counts, capture discovery, validation with the pipeline, and saving. |
| `build.py` | Embeds the pipeline's config schema in the page. Run it after the schema or `pipeline/configs/README.md` changes. |
| `settings.example.ini` | The settings the server reads. Copy to `settings.ini` (gitignored). |

## Start it

From the repository root, in the pipeline environment, in a terminal where
`colmap` is on PATH (the validation dry run reads its version):

```bash
cp pipeline/scripts/run-builder/settings.example.ini pipeline/scripts/run-builder/settings.ini
```

Edit `settings.ini` (at least `data_root`), then:

```bash
uv run python pipeline/scripts/run-builder/serve.py
```

Open the address it prints, `http://127.0.0.1:8765/`. `--port` picks another
port; `--settings FILE` reads another settings file. Ctrl-C stops it.

Without the server, open `run-builder.html` in a browser and type the paths.
Copy or download the config, and use the printed dry run as the check.

On WSL2, open the address in the Windows browser; WSL forwards `127.0.0.1`.

## Use it: a first run

1. **Photographs.** Choose one capture and type its path under `datasets/`
   (or browse to it), then press **Check**. The page shows the photograph
   count, camera folders and any mask sets already in the capture.
2. **Masks.** Choose **Generate** and enter one prompt per line, for example
   `plant` and `leaf`. Choose **One config per mask set** to compare them.
   To reuse masks that exist, choose **Existing** and click a mask set.
3. **Run settings.** For a test run, set `[train] downscale_factor = 4` and
   `max_num_iterations = 10000` (`pipeline/README.md`, "Settings for a test
   run"). The other keys start at the pipeline's defaults.
4. **Validate.** Each config is loaded by the pipeline and dry-run. Fix
   anything it reports; the messages are the pipeline's own.
5. **Save.** Writes the configs into the configs directory. An existing file
   is never overwritten.
6. **Run the commands** in the order shown: the samask commands (when masks
   are generated), the dry runs, then the real runs.

## What it does

1. **Photographs.** One capture (a directory containing `images/`,
   `pipeline/LAYOUT.md`), a collection of captures run by
   `pipeline/run_collection.py`, or a video made into a capture first.
2. **Masks.** None, existing mask sets (`masks/` or `masks_{variant}/`), or new
   ones generated with samask from one or more text prompts. **Each prompt is its
   own mask variant:** prompt `plant` writes `masks_plant/`, prompt `leaf`
   writes `masks_leaf/`; they are never merged. The variant name is the
   prompt in lowercase with every run of other characters replaced by `-`.
   Choose the set a run uses (`[dataset] mask_variant`), or one config per set
   so the sets can be compared; each such config then gets the set's name added
   to `[run] id_prefix` (a checkbox turns that off).
3. **Run settings.** The common keys first, every other key folded by section.
4. **Output.** The config (one per mask set when comparing), then the commands:
   terminal setup, the samask commands (one per capture and prompt), the dry
   runs, then the real runs. A collection uses `run_collection.py` with the
   catalog flags; one capture uses `run_pipeline.py`.

A video adds two steps in front, from `pipeline/VIDEO-WORKFLOW.md`: the
`video-frames.py review` command, then the extract command that the review page
prints for the interval you choose. Then it continues as a capture. A video
config can be validated and saved only after the extraction exists.

## Where its knowledge comes from

- **Keys, sections, defaults, allowed values and help** come from
  `dt4ag_config.config_schema()` (`python pipeline/dt4ag_config.py --schema`):
  `CONFIG_SCHEMA` in `dt4ag_config.py` declares the keys and the help text is
  each key's row in `pipeline/configs/README.md`. `build.py` embeds that JSON in
  the page. `tests/test_config_schema.py` fails when the schema and
  `load_config` disagree; `tests/test_run_builder.py` fails when the page's copy
  is stale.
- **Validation is the pipeline's own.** The page checks nothing itself. The
  server loads each config with `load_config` and runs the real dry run; the
  page shows their messages as they are. With the page opened as a file, the
  printed dry run is the check.
- **Capture discovery** is `run_collection.discover_captures`, and the capture
  ID shown for a catalog job is `run_collection.capture_id_for`.
- **samask** is a separate repository on its own Python and torch
  (`pipeline/MASKING.md`); nothing of it is imported here. Its location and
  command template come from the settings below, never from the page.

The one value the page sets that differs from the loader's default is `[train]
quit_on_train_completion = true`, because `run_pipeline.py` refuses the train
stage otherwise and a collection run treats false as invalid. It can be changed
like any other setting.

## Settings

`settings.ini` beside `serve.py` (or `--settings FILE`), then environment
variables, which win. `settings.example.ini` documents each key.

| Key | Environment variable | Meaning |
|---|---|---|
| `[run-builder] data_root` | `RUN_BUILDER_DATA_ROOT` | Required. Every path the server touches must be inside it. |
| `[run-builder] configs_dir` | `RUN_BUILDER_CONFIGS_DIR` | Where Save writes; default `<data_root>/configs`. |
| `[run-builder] catalog_root` | `RUN_BUILDER_CATALOG_ROOT` | Catalog root for collection runs; default the data root. |
| `[run-builder] catalog_root_id` | `RUN_BUILDER_CATALOG_ROOT_ID` | Catalog root ID, kebab-case. |
| `[run-builder] shell_setup` | `RUN_BUILDER_SHELL_SETUP` | One shell line printed before the commands, for example the PATH export from QUICKSTART section 5. |
| `[samask] directory` | `RUN_BUILDER_SAMASK_DIR` | The samask checkout; its commands run inside it. |
| `[samask] command` | `RUN_BUILDER_SAMASK_COMMAND` | Command template with `{images}`, `{masks}`, `{prompt}`, `{capture}`, `{variant}`. |
| `[samask] mask_extension` | `RUN_BUILDER_SAMASK_MASK_EXTENSION` | The extension samask writes; becomes `[dataset] mask_extensions`. |

The page's own form is remembered in the browser (local storage); the server's
settings fill only empty fields.

## What the server does and refuses

`POST /api/<action>` with a JSON body, one handler per action on
`serve.Builder`: `info`, `list`, `capture`, `discover`, `validate`, `save`.

- Binds `127.0.0.1` only. Refuses a request whose `Host` or `Origin` is not
  this server, and anything that is not JSON.
- Refuses every path outside the data root, after resolving symbolic links.
- `validate` writes the configs to a temporary directory, loads each with
  `load_config`, and for each that loads runs `run_pipeline.py --dry-run` (or
  `run_collection.py --dry-run` for a collection) with the Python running the
  server. A dry run executes no stage.
- `save` refuses a name that exists (nothing is written if any does), a
  directory outside the data root, and a config that does not load. When masks
  are still to be generated, the config is checked twice instead: as written,
  without the path checks, and with `use_masks` off, with them; the dry run
  after samask checks the masks.
- It never deletes a file and never starts a pipeline stage or samask.

The API is shaped so job submission could be added later as another action.
None exists: no submission, queue, process control, authentication or network
binding.

## Tests

`pipeline/tests/test_run_builder.py` and `pipeline/tests/test_config_schema.py`,
in the stdlib suite (`python -m unittest discover -s pipeline/tests`). The tests
of what the page generates run its `builder-core` script under Node and are
skipped when `node` is not on PATH; the schema and server tests need only
Python.
