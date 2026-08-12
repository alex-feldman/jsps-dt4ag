# pipeline tests

Tests for the pipeline's configuration layer (`pipeline/dt4ag_config.py`) and
for the command-line runner (`pipeline/run_pipeline.py`).

## Running them

From the repository root:

    python -m unittest discover -s pipeline/tests -v

Stdlib `unittest` only, no pytest and no other third-party package. That is
deliberate: the pipeline's config handling is zero-dependency by design and runs
on Python 3.10 in the reconstruction environment, where a test runner cannot be
assumed to be installed. If you add tests here, keep them importable with
nothing but the standard library.

To run one class or one test:

    python -m unittest pipeline.tests.test_dt4ag_config.TestFindConfig -v

## What they cover

`test_dt4ag_config.py` covers the loader end to end: the happy path, every
default applied when a key is omitted, the derived COLMAP workspace and output
paths, all the documented failure modes (each asserting both that `ConfigError`
is raised and that the message names the offending key and the config file),
run-id derivation and auto-increment, COLMAP version detection, `data_type =
auto` inference, `find_config` precedence, the CSV run log, and the committed
`configs/example.ini`.

`test_run_pipeline.py` covers the runner's own logic: argument parsing, stage
selection (`--stage`, `--from-stage`, ordering, duplicates, bad names, the
mutually exclusive pair), the checkpoint-discovery rule, the exact command line
built for each of the four stages, recursive image counting, and export
verification including a PLY that exists at a plausible size but declares zero
vertices. It runs no subprocess at all; the stages themselves can only be proven
by a real run.

Every test is hermetic. Each one builds its own data tree in a
`tempfile.TemporaryDirectory` and throws it away afterwards.

## What they deliberately do not cover

- **No GPU.** Nothing here trains, exports, or touches CUDA.
- **No COLMAP binary.** `detect_colmap_version` is exercised by monkeypatching
  `shutil.which` and `subprocess.run` inside the module under test, including
  the real-world COLMAP 3.12 case where `--version` is rejected and the version
  only appears in the banner printed by a bare `colmap`. The real binary is
  never invoked, so the suite passes identically on a machine without COLMAP.
- **No real data and no data drive.** No test reads or writes the working data
  drive, or any path outside its own temporary directory. `configs/example.ini`
  is loaded with `validate_paths=False` because its `data_root` is a deliberate
  placeholder; one test asserts that placeholder does **not** exist, so the
  committed example can never be a runnable config by accident.
- **No network.**
- **No notebook execution.** `notebooks/nerfstudio-pipeline-06.ipynb` is not
  imported, executed, or linted here. Note in particular that the image
  discovery and empty-directory guard live in the notebook, not in
  `dt4ag_config.py`; the loader does no image counting at all.
- **No nerfstudio.** `ns-process-data`, `ns-train` and `ns-export` are outside
  the scope of this suite. `test_run_pipeline.py` asserts on the argument
  vectors the runner would execute, and executes none of them.
