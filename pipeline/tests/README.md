# pipeline tests

Tests for the pipeline's configuration layer (`pipeline/dt4ag_config.py`), for
the command-line runner (`pipeline/run_pipeline.py`) and for the stdlib-only
helper scripts under `pipeline/scripts/`.

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

`test_video_frames.py` covers `scripts/video-frames/video-frames.py`: the exact
`fps` expression and the frame-count estimate, the PNG header reader, the
refusals (existing capture, wrong or malformed hash, impossible range), the
records left by a successful, a failed and a mixed-shape extraction, the review
folder's contents, the single decode and its preview window, the command the page shows, and `subset` (linked, never copied, with
its refusals and its cleanup when links are impossible). ffmpeg and ffprobe are replaced by fakes, so the tests
need neither; the real tools were checked only by a real run on a synthetic clip.

`test_wsl_guard.py` covers the WSL2 guard's decisions, probe parsing and
process-group handling with a fake Windows and a stand-in child. Its cgroup
tests read a fake `/proc` and `/sys/fs/cgroup` tree built in a temporary
directory (missing files, cgroup v1, shared versus dedicated cgroup, maxima),
and its `--emulate-vm` tests cover value validation, the exact wrapped command,
the refusal with `[host] memory_high` and `--print-command`. No scope is created.

`test_camera_review.py` covers the helpers of `scripts/camera-review/camera-review.py`
that need no COLMAP data: frame times from `frames.csv` or image order, missing
and unexpected frames, model folders and the arguments. Stdlib only.

`test_ply_check.py` covers `scripts/ply-check/ply-check.py` with small synthetic
PLYs written by the test (3DGS property names, binary little-endian): the header
reader and its refusals, non-finite and zero-vertex and truncated files, the
statistics, the preview point selection, exit codes and `--json`. It needs
numpy and is skipped without it, so run the suite with the pipeline
environment's Python to include it.

The optional `[host]` section is covered in both files. `test_dt4ag_config.py`
(`TestHostSection`): an absent or empty section means no limits, valid values
are normalized, malformed sizes and CPU lists, a cap below 512M and an unknown
key are refused naming the key and file, and the archived `[run-record]`
carries `memory_high` and `process_cpus` only when set.
`test_run_pipeline.py` (`TestHostLimits`): empty keys leave all four stage
commands unchanged, `memory_high` prefixes each with the `systemd-run` scope,
`process_cpus` pins `ns-process-data` only (inside the scope), `--dry-run`
prints the prefixed commands and runs no probe, and the run-start check refuses
a failing scope, a missing `systemd-run`, an undelegated memory controller and
a CPU the process cannot use. `subprocess.run` is mocked; no scope is created.

`test_wsl_guard.py` covers `scripts/wsl-guard/wsl-guard.py`: argument
splitting at `--` and threshold validation, the PowerShell probe script and the
parsing of its output (BOM, stray lines, missing values), WSL and
`powershell.exe` detection, the start, release and stop decisions including the
once-per-period release limit, the pre-start release-and-wait loop on a fake
clock, where the log, status and host-sample files go, and four end-to-end runs
against a few-line stand-in for `run_pipeline.py`: a refusal that starts
nothing, a clean run passing its exit code through, a low-RAM stop that signals
only the guard's own process group (exit 75), and an in-run cache release. The
Windows side is a fake and sudo is never called, so it needs neither WSL nor
Windows.

`test_notebook_outputs.py` fails if any notebook listed by `git ls-files` has
stored cell outputs or a non-null execution count, and names the file and cell
and the `nbstripout` command that fixes it. The check is a pure function over
a parsed notebook, tested on in-memory dirty and clean notebooks; the
repository test is skipped outside a git checkout.

Every test is hermetic. Each one builds its own data tree in a
`tempfile.TemporaryDirectory` and throws it away afterwards.

The separate experimental organ-analysis command has its own dependency-using
CPU fixtures under `pipeline/scripts/organ-analysis/tests/`. Run those with the
pipeline environment using the command in that directory's README; they are
not part of this stdlib-only suite.

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

Catalog job progress, failure-path, SIGTERM, probe and old-record fixtures are
in `test_plant_catalog.py`; see [CATALOG-RECORDS.md](../CATALOG-RECORDS.md) for
the producer contract and what the fixtures cover.

The optional external-reader check uses `catalog_consumer_fixtures.py` and
`catalog-consumer-compat.cjs`; invocation and prerequisites are in
[CATALOG-RECORDS.md](../CATALOG-RECORDS.md#later-stage-and-external-consumer-checks).
