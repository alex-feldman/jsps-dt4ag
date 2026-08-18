# Pipeline configuration

Everything the pipeline used to have typed into notebook cells now lives in an
INI file. `example.ini` is the committed template; it carries neutral
placeholder paths and is not meant to run as-is.

## Making your own config

```bash
cd pipeline
cp configs/example.ini configs/my-run.ini
$EDITOR configs/my-run.ini            # at minimum set data_root and images_subpath
export DT4AG_CONFIG=$PWD/configs/my-run.ini
python dt4ag_config.py                # validate and print the resolved values
```

The last command prints every resolved path and setting, or exits non-zero with
a `CONFIG ERROR:` line naming the offending key and file. Run it before you
start a reconstruction; it is much cheaper than finding out at cell 39.

Your own configs are yours. Only `example.ini` is committed; if you want to keep
a personal one out of git, name it and add it to `.gitignore`, because real
configs contain absolute paths and dataset names.

## Why INI

The pipeline environment is Python 3.10, where `tomllib` is not in the standard
library. INI via stdlib `configparser` needs no third-party dependency at all.
That is a deliberate choice, not an accident: do not swap it for YAML or TOML
without also accepting a new dependency in an environment that is already hard
to reproduce.

## How the pipeline finds a config

In order of precedence:

1. An explicit path passed to `load_config()`.
2. The `DT4AG_CONFIG` environment variable.
3. `configs/example.ini`, found by walking up from the working directory.

If none of those resolve, loading raises `ConfigError` rather than falling back
to defaults. Silent fallback is exactly the failure mode this refactor exists to
remove.

## Directory layout the config assumes

```
<data_root>/
  datasets/   INPUT photographs and masks     <- [dataset] images_subpath lives here
  derived/    composited masked images        <- written per capture, disposable
  colmap/     COLMAP workspaces               <- written per run
  outputs/    nerfstudio training outputs     <- written per run
  exports/    exported splats
  run-log.csv appended to on every run
```

`datasets/` is input and the pipeline never writes to it. The other four trees
are the pipeline's own and can all be rebuilt from `datasets/` plus a config
file. `pipeline/LAYOUT.md` is the specification, including the rule that makes a
directory a capture; this file only covers the keys.

The COLMAP workspace and the nerfstudio output directory for a run are both

```
<colmap_dir | outputs_dir>/<capture_rel>/<run_id>
```

so the capture's path is mirrored across all three trees and each run gets its
own leaf. Nothing is overwritten between runs. `capture_rel` is `images_subpath`
with a trailing `images` component dropped, since that component names a
directory inside the capture rather than the capture itself
(`Dt4agConfig.capture_rel`).

## Keys

### `[paths]`

| Key | Required | Default | Controls |
|---|---|---|---|
| `data_root` | yes | none | Root of the working data tree. Absolute, or relative to the config file. Must exist. Replaces the hardcoded `/media/alex/T5Red/DT-data/` and the implicit `Path.cwd().parent` the notebook actually relied on. |
| `datasets_dirname` | no | `datasets` | Name of the input-images directory under `data_root`. Must exist. |
| `colmap_dirname` | no | `colmap` | Name of the COLMAP workspace directory. Created on demand. |
| `outputs_dirname` | no | `outputs` | Name of the nerfstudio output directory. Created on demand. |
| `exports_dirname` | no | `exports` | Name of the top-level exports directory. |
| `derived_dirname` | no | `derived` | Name of the directory holding the pipeline's rebuildable intermediates, currently the composited masked images. Created on demand, and safe to delete at any time. |

### `[dataset]`

| Key | Required | Default | Controls |
|---|---|---|---|
| `images_subpath` | yes | none | Path of the image directory, relative to `<data_root>/<datasets_dirname>`. Must exist. Must be relative; an absolute path is an error. Ends in `images` under the canonical layout. |
| `image_extensions` | no | empty | Which files under `images_subpath` are photographs, comma separated. Empty means every supported image type. |
| `mask_extensions` | no | empty | Which files are masks. Always kept out of SfM. |
| `use_masks` | no | `false` | Composite each mask into its photograph's alpha channel as a pre-step. Every photograph must have a mask; a partial set is an error. |
| `masked_images_subpath` | no | `<derived_dirname>/masked/<capture_rel>` | Where composites are written. Relative values resolve against `data_root`, not the datasets directory. A path inside `datasets/` is refused, because that tree is input. |

**There is no key for where the masks are.** It is derived from the layout:
`<capture>/masks/` when `images_subpath` ends in `images`, mirroring it, and
beside each photograph otherwise. Both real arrangements follow from that one
rule, so a key stating it could only ever agree with the filesystem or be wrong.
The `mask_subpath` key that used to set it was retired in v0.2.0, and a config
still carrying it is refused rather than ignored.

This one key replaces the notebook's `subdir_a_id` through `subdir_d_id` and the
four substring-matching loops that resolved them, along with both escape
hatches: the commented-out `subdir_d = subdir_c` line for three-deep
hierarchies, and the `use_specified_folder_names` block. Any depth of nesting
now works, because the path is stated rather than searched for.

### `[run]`

| Key | Required | Default | Controls |
|---|---|---|---|
| `id_prefix` | no | `run` | First component of the run id. |
| `date` | no | empty | `yymmdd`. Empty means today's date. Set it to reproduce a past run id. |
| `run_count` | no | empty | Empty means auto-increment: one more than the highest counter already present for that date under this dataset's `colmap/` and `outputs/` directories. |
| `colmap_version` | no | empty | Empty means detect by running `colmap --version` (then the bare `colmap` banner). If detection fails and this is empty, loading raises rather than recording a fake version. |
| `log_file` | no | `run-log.csv` | CSV run log, appended to once per run. Relative paths resolve under `data_root`. |

Run ids are `<id_prefix>_<date>-<run_count>-<colmap_version>`, for example
`run_260807-01-312`.

### `[colmap]`

Feeds `colmap automatic_reconstructor`.

| Key | Required | Default | Controls |
|---|---|---|---|
| `data_type` | no | `auto` | `auto`, `individual` or `video`. `auto` reproduces the old heuristic: a directory name containing `frames` means video, anything else means individual images. |
| `single_camera` | no | `1` | `--single_camera`. |
| `single_camera_per_folder` | no | `1` | `--single_camera_per_folder`. |
| `dense` | no | `0` | `--dense`. Sparse only by default. |
| `extra_args` | no | empty | Passed verbatim after the generated flags. |

### `[nerfstudio]`

Feeds `ns-process-data`.

| Key | Required | Default | Controls |
|---|---|---|---|
| `scene_type` | no | `images` | `images` or `video`. |
| `video_path` | conditional | empty | Required when `scene_type = video`. |
| `skip_colmap` | no | `true` | Adds `--skip-colmap`, reusing the COLMAP reconstruction instead of re-running SfM. |
| `colmap_model_path` | no | `sparse/0` | `--colmap-model-path`. |

### `[train]`

Feeds `ns-train`.

| Key | Required | Default | Controls |
|---|---|---|---|
| `method` | no | `splatfacto` | The nerfstudio method name. |
| `max_num_iterations` | no | `30000` | `--max-num-iterations`. Must be positive. |
| `use_scale_regularization` | no | `true` | `--pipeline.model.use-scale-regularization`. |
| `background_color` | no | `random` | `--pipeline.model.background-color`. |
| `quit_on_train_completion` | no | `false` | `--viewer.quit-on-train-completion`. |
| `max_log_size` | no | `0` | `--logging.local-writer.max-log-size`. |

### `[export]`

Feeds `ns-export` and the optional point-cloud conversion.

| Key | Required | Default | Controls |
|---|---|---|---|
| `format` | no | `gaussian-splat` | The `ns-export` subcommand. |
| `env_label` | no | `env` | Free-text label baked into the export filename to record which environment produced it. Was the hardcoded `conda_env_name`. |
| `platform_label` | no | the running platform | Second label in the export filename. |
| `export_3dgs` | no | `false` | Whether to run the 3DGS-to-point-cloud conversion. Inflates file size by roughly 1000x. |
| `gauss_to_pc_script` | conditional | empty | Path to `gauss_to_pc.py`. Required when `export_3dgs = true`. |

Export filenames are built as (`export_filename` in `run_pipeline.py`)

```
<capture>_<run_id>_splat_<platform_label>_<env_label>_<iterations>steps[_dsN]_<colmap data_type>.ply
```

`<capture>` is `capture_rel`'s last component, so it names the capture under
both layouts. `dsN` appears whenever the effective downscale factor is known,
which is what stops two resolutions of one capture overwriting each other.

## Validation behaviour

`load_config()` raises `ConfigError` for a missing file, a missing section, a
missing or empty required key, a value of the wrong type, an out-of-range
enumerated value, or a required path that does not exist. Every message names
both the key and the config file. Nothing falls back silently.

The three paths checked for existence are `data_root`, the datasets directory
and the dataset image directory. Pass `validate_paths=False` to inspect a config
on a machine where the data is not mounted.
