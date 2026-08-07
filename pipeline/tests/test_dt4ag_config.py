"""Tests for ``pipeline/dt4ag_config.py``.

Stdlib ``unittest`` only, on purpose: the pipeline's config handling is
deliberately zero-dependency and pytest is not guaranteed to be installed in
the reconstruction environment. Run with::

    python -m unittest discover -s pipeline/tests -v

Every test is hermetic. Nothing here reads or writes the real data drive,
invokes COLMAP, touches a GPU, or uses the network. Anything that would need a
real COLMAP binary is simulated by monkeypatching ``shutil.which`` and
``subprocess.run`` inside the module under test.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# The module under test lives one directory above this package.
PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import dt4ag_config  # noqa: E402
from dt4ag_config import (  # noqa: E402
    ConfigError,
    Dt4agConfig,
    detect_colmap_version,
    find_config,
    load_config,
)

EXAMPLE_INI = PIPELINE_DIR / "configs" / "example.ini"

# Terms that must never appear in the committed example config. The pipeline is
# object-agnostic; the example must not imply a subject.
FORBIDDEN_TERMS = [
    "plant",
    "tomato",
    "tanashi",
    "soybean",
    "houseplant",
    "arabidopsis",
]

SECTIONS = ("paths", "dataset", "run", "colmap", "nerfstudio", "train", "export")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def render_ini(sections) -> str:
    """Render an ordered mapping of section -> {key: value} as INI text."""
    chunks = []
    for name in SECTIONS:
        chunks.append(f"[{name}]")
        for key, value in (sections.get(name) or {}).items():
            chunks.append(f"{key} = {value}")
        chunks.append("")
    return "\n".join(chunks)


class ConfigFixture:
    """A temporary data tree plus a config file pointing at it."""

    def __init__(self, tmp: Path, images_subpath: str = "scene-01/images",
                 make_images: bool = True, extra=None, omit=None):
        self.tmp = tmp
        self.data_root = tmp / "data"
        self.datasets = self.data_root / "datasets"
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.images_subpath = images_subpath
        self.images_path = self.datasets / images_subpath
        if make_images:
            self.images_path.mkdir(parents=True, exist_ok=True)
            (self.images_path / "0001.png").write_bytes(b"not-a-real-png")
        sections = {
            "paths": {"data_root": str(self.data_root)},
            "dataset": {"images_subpath": images_subpath},
            "run": {},
            "colmap": {},
            "nerfstudio": {},
            "train": {},
            "export": {},
        }
        for section, values in (extra or {}).items():
            sections.setdefault(section, {}).update(values)
        for section, keys in (omit or {}).items():
            for key in keys:
                sections.get(section, {}).pop(key, None)
        self.path = tmp / "test.ini"
        self.path.write_text(render_ini(sections), encoding="utf-8")

    def load(self, **kwargs) -> Dt4agConfig:
        return load_config(self.path, **kwargs)


class FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TempDirTestCase(unittest.TestCase):
    """Base class giving each test its own throwaway directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def patch_module(self, name, value):
        """Temporarily replace an attribute on the module under test."""
        original = getattr(dt4ag_config, name)
        setattr(dt4ag_config, name, value)
        self.addCleanup(setattr, dt4ag_config, name, original)

    def patch_which(self, result):
        class _Shutil:
            @staticmethod
            def which(_name):
                return result

        self.patch_module("shutil", _Shutil)

    def assertConfigError(self, ini_path, *needles):
        """Assert ConfigError is raised and its message names each needle."""
        with self.assertRaises(ConfigError) as caught:
            load_config(ini_path)
        message = str(caught.exception)
        for needle in needles:
            self.assertIn(str(needle), message)
        return message


# --------------------------------------------------------------------------
# happy path and defaults
# --------------------------------------------------------------------------

class TestHappyPath(TempDirTestCase):

    def test_loads_a_minimal_valid_config(self):
        fixture = ConfigFixture(self.tmp)
        cfg = fixture.load()
        self.assertEqual(cfg.source, fixture.path)
        self.assertEqual(cfg.data_root, fixture.data_root)
        self.assertEqual(cfg.images_path, fixture.images_path)
        self.assertEqual(cfg.images_rel, Path("scene-01/images"))

    def test_every_documented_default_is_applied_when_key_omitted(self):
        fixture = ConfigFixture(self.tmp)
        cfg = fixture.load()
        # [paths] directory names
        self.assertEqual(cfg.datasets_dir, fixture.data_root / "datasets")
        self.assertEqual(cfg.colmap_dir, fixture.data_root / "colmap")
        self.assertEqual(cfg.outputs_dir, fixture.data_root / "outputs")
        self.assertEqual(cfg.exports_dir, fixture.data_root / "exports")
        # [run]
        self.assertEqual(cfg.run_id_prefix, "run")
        self.assertEqual(cfg.run_date, "")
        self.assertEqual(cfg.run_count, "")
        self.assertEqual(cfg.colmap_version, "")
        self.assertEqual(cfg.run_log, fixture.data_root / "run-log.csv")
        # [colmap]
        self.assertEqual(cfg.colmap_data_type, "auto")
        self.assertEqual(cfg.colmap_single_camera, 1)
        self.assertEqual(cfg.colmap_single_camera_per_folder, 1)
        self.assertEqual(cfg.colmap_dense, 0)
        self.assertEqual(cfg.colmap_extra_args, "")
        # [nerfstudio]
        self.assertEqual(cfg.scene_type, "images")
        self.assertEqual(cfg.video_path, "")
        self.assertTrue(cfg.skip_colmap)
        self.assertEqual(cfg.colmap_model_path, "sparse/0")
        # [train]
        self.assertEqual(cfg.train_method, "splatfacto")
        self.assertEqual(cfg.max_num_iterations, 30000)
        self.assertTrue(cfg.use_scale_regularization)
        self.assertEqual(cfg.background_color, "random")
        self.assertFalse(cfg.quit_on_train_completion)
        self.assertEqual(cfg.max_log_size, 0)
        # [export]
        self.assertEqual(cfg.export_format, "gaussian-splat")
        self.assertEqual(cfg.env_label, "env")
        self.assertEqual(cfg.platform_label, sys.platform)
        self.assertFalse(cfg.export_3dgs)
        self.assertEqual(cfg.gauss_to_pc_script, "")

    def test_explicit_values_override_defaults(self):
        fixture = ConfigFixture(self.tmp, extra={
            "paths": {"colmap_dirname": "sfm", "outputs_dirname": "runs"},
            "train": {"method": "nerfacto", "max_num_iterations": "1234",
                      "quit_on_train_completion": "yes"},
            "colmap": {"dense": "1", "extra_args": "--quality high"},
        })
        cfg = fixture.load()
        self.assertEqual(cfg.colmap_dir, fixture.data_root / "sfm")
        self.assertEqual(cfg.outputs_dir, fixture.data_root / "runs")
        self.assertEqual(cfg.train_method, "nerfacto")
        self.assertEqual(cfg.max_num_iterations, 1234)
        self.assertTrue(cfg.quit_on_train_completion)
        self.assertEqual(cfg.colmap_dense, 1)
        self.assertEqual(cfg.colmap_extra_args, "--quality high")

    def test_relative_data_root_resolves_against_the_config_file(self):
        data_root = self.tmp / "data"
        (data_root / "datasets" / "scene-01" / "images").mkdir(parents=True)
        ini = self.tmp / "nested" / "test.ini"
        ini.parent.mkdir()
        ini.write_text(render_ini({
            "paths": {"data_root": "../data"},
            "dataset": {"images_subpath": "scene-01/images"},
        }), encoding="utf-8")
        cfg = load_config(ini)
        self.assertEqual(cfg.data_root, data_root.resolve())

    def test_describe_mentions_the_source_and_the_dataset(self):
        cfg = ConfigFixture(self.tmp).load()
        text = cfg.describe()
        self.assertIn(str(cfg.source), text)
        self.assertIn("scene-01/images", text)
        self.assertIn("(auto: today)", text)


# --------------------------------------------------------------------------
# derived paths
# --------------------------------------------------------------------------

class TestDerivedPaths(TempDirTestCase):

    def test_colmap_workspace_and_output_dir_land_where_the_notebook_expects(self):
        fixture = ConfigFixture(self.tmp, images_subpath="scene-01/capture-a/images")
        cfg = fixture.load()
        run_id = "run_260807-01-312"
        self.assertEqual(
            cfg.colmap_workspace(run_id),
            fixture.data_root / "colmap" / "scene-01/capture-a/images" / run_id,
        )
        self.assertEqual(
            cfg.output_dir(run_id),
            fixture.data_root / "outputs" / "scene-01/capture-a/images" / run_id,
        )
        self.assertEqual(cfg.colmap_workspace_parent, cfg.colmap_dir / cfg.images_rel)
        self.assertEqual(cfg.output_parent, cfg.outputs_dir / cfg.images_rel)

    def test_derived_paths_follow_renamed_directories(self):
        fixture = ConfigFixture(self.tmp, extra={
            "paths": {"colmap_dirname": "sfm", "outputs_dirname": "runs"}})
        cfg = fixture.load()
        self.assertEqual(
            cfg.colmap_workspace("r"), fixture.data_root / "sfm" / "scene-01/images" / "r")
        self.assertEqual(
            cfg.output_dir("r"), fixture.data_root / "runs" / "scene-01/images" / "r")


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------

class TestFailureModes(TempDirTestCase):

    def test_missing_config_file(self):
        missing = self.tmp / "nope.ini"
        with self.assertRaises(ConfigError) as caught:
            load_config(missing)
        self.assertIn(str(missing), str(caught.exception))

    def test_missing_section(self):
        ini = self.tmp / "test.ini"
        ini.write_text("[paths]\ndata_root = /tmp\n", encoding="utf-8")
        self.assertConfigError(ini, "dataset", ini)

    def test_missing_required_key(self):
        fixture = ConfigFixture(self.tmp, omit={"dataset": ["images_subpath"]})
        self.assertConfigError(fixture.path, "images_subpath", "dataset", fixture.path)

    def test_empty_required_value(self):
        fixture = ConfigFixture(self.tmp, extra={"dataset": {"images_subpath": ""}})
        message = self.assertConfigError(
            fixture.path, "images_subpath", "dataset", fixture.path)
        self.assertIn("empty", message)

    def test_nonexistent_data_root(self):
        fixture = ConfigFixture(self.tmp, extra={
            "paths": {"data_root": str(self.tmp / "not-mounted")}})
        self.assertConfigError(fixture.path, "data_root", "not-mounted", fixture.path)

    def test_nonexistent_image_directory(self):
        fixture = ConfigFixture(self.tmp, images_subpath="scene-99/images",
                                make_images=False)
        self.assertConfigError(fixture.path, "images_subpath", "scene-99", fixture.path)

    def test_absolute_images_subpath_is_rejected(self):
        fixture = ConfigFixture(self.tmp, extra={
            "dataset": {"images_subpath": str(self.tmp / "elsewhere")}})
        message = self.assertConfigError(
            fixture.path, "images_subpath", "dataset", fixture.path)
        self.assertIn("relative", message)

    def test_malformed_int(self):
        fixture = ConfigFixture(self.tmp, extra={
            "train": {"max_num_iterations": "thirty thousand"}})
        message = self.assertConfigError(
            fixture.path, "max_num_iterations", "train", fixture.path)
        self.assertIn("integer", message)

    def test_malformed_bool(self):
        fixture = ConfigFixture(self.tmp, extra={"train": {"use_scale_regularization": "maybe"}})
        message = self.assertConfigError(
            fixture.path, "use_scale_regularization", "train", fixture.path)
        self.assertIn("boolean", message)

    def test_malformed_enum(self):
        fixture = ConfigFixture(self.tmp, extra={"colmap": {"data_type": "timelapse"}})
        message = self.assertConfigError(
            fixture.path, "data_type", "colmap", fixture.path)
        self.assertIn("timelapse", message)

    def test_malformed_enum_scene_type(self):
        fixture = ConfigFixture(self.tmp, extra={"nerfstudio": {"scene_type": "pointcloud"}})
        self.assertConfigError(fixture.path, "scene_type", "nerfstudio", fixture.path)

    def test_malformed_date(self):
        fixture = ConfigFixture(self.tmp, extra={"run": {"date": "2026-08-07"}})
        message = self.assertConfigError(fixture.path, "date", "run", fixture.path)
        self.assertIn("yymmdd", message)

    def test_malformed_run_count(self):
        fixture = ConfigFixture(self.tmp, extra={"run": {"run_count": "first"}})
        self.assertConfigError(fixture.path, "run_count", "run", fixture.path)

    def test_non_positive_max_num_iterations(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                fixture = ConfigFixture(
                    self.tmp, extra={"train": {"max_num_iterations": value}})
                message = self.assertConfigError(
                    fixture.path, "max_num_iterations", "train", fixture.path)
                self.assertIn("positive", message)

    def test_video_scene_type_requires_a_video_path(self):
        fixture = ConfigFixture(self.tmp, extra={"nerfstudio": {"scene_type": "video"}})
        self.assertConfigError(fixture.path, "video_path", "nerfstudio", fixture.path)

    def test_export_3dgs_requires_the_conversion_script(self):
        fixture = ConfigFixture(self.tmp, extra={"export": {"export_3dgs": "true"}})
        self.assertConfigError(
            fixture.path, "gauss_to_pc_script", "export", fixture.path)

    def test_validate_paths_false_tolerates_a_missing_data_drive(self):
        fixture = ConfigFixture(self.tmp, extra={
            "paths": {"data_root": "/nonexistent-drive/data"}})
        cfg = fixture.load(validate_paths=False)
        self.assertEqual(cfg.data_root, Path("/nonexistent-drive/data"))


# --------------------------------------------------------------------------
# run identity
# --------------------------------------------------------------------------

class TestRunIdentity(TempDirTestCase):

    def _cfg(self, **run_keys):
        fixture = ConfigFixture(self.tmp, extra={"run": run_keys})
        self.fixture = fixture
        return fixture.load()

    def test_run_id_uses_pinned_date_count_and_version(self):
        cfg = self._cfg(date="260807", run_count="07", colmap_version="312")
        self.assertEqual(cfg.make_run_id(), "run_260807-07-312")

    def test_run_id_is_memoised(self):
        cfg = self._cfg(date="260807", run_count="07", colmap_version="312")
        first = cfg.make_run_id()
        self.assertIs(first, cfg.make_run_id())

    def test_run_id_prefix_is_configurable(self):
        cfg = self._cfg(id_prefix="valid", date="260807", run_count="01",
                        colmap_version="312")
        self.assertEqual(cfg.make_run_id(), "valid_260807-01-312")

    def test_date_is_derived_from_today_when_blank(self):
        import datetime as dt

        cfg = self._cfg(run_count="01", colmap_version="312")
        expected = dt.date.today().strftime("%y%m%d")
        self.assertEqual(cfg.make_run_id(), f"run_{expected}-01-312")

    def test_run_count_starts_at_01_with_no_existing_runs(self):
        cfg = self._cfg(date="260807", colmap_version="312")
        self.assertEqual(cfg.next_run_count("260807"), "01")
        self.assertEqual(cfg.make_run_id(), "run_260807-01-312")

    def test_run_count_increments_past_runs_in_the_colmap_tree(self):
        cfg = self._cfg(date="260807", colmap_version="312")
        (cfg.colmap_workspace_parent / "run_260807-01-312").mkdir(parents=True)
        (cfg.colmap_workspace_parent / "run_260807-02-312").mkdir(parents=True)
        self.assertEqual(cfg.next_run_count("260807"), "03")

    def test_run_count_increments_past_runs_in_the_outputs_tree(self):
        cfg = self._cfg(date="260807", colmap_version="312")
        (cfg.output_parent / "run_260807-04-312").mkdir(parents=True)
        self.assertEqual(cfg.next_run_count("260807"), "05")

    def test_run_count_considers_both_trees_together(self):
        cfg = self._cfg(date="260807", colmap_version="312")
        (cfg.colmap_workspace_parent / "run_260807-02-312").mkdir(parents=True)
        (cfg.output_parent / "run_260807-05-312").mkdir(parents=True)
        self.assertEqual(cfg.next_run_count("260807"), "06")
        self.assertEqual(cfg.make_run_id(), "run_260807-06-312")

    def test_run_count_ignores_other_dates_files_and_prefixes(self):
        cfg = self._cfg(date="260807", colmap_version="312")
        parent = cfg.colmap_workspace_parent
        parent.mkdir(parents=True)
        (parent / "run_260806-09-312").mkdir()
        (parent / "other_260807-09-312").mkdir()
        (parent / "run_260807-09-312.txt").write_text("a file, not a run", encoding="utf-8")
        self.assertEqual(cfg.next_run_count("260807"), "01")

    def test_resolve_colmap_version_prefers_the_detected_version(self):
        cfg = self._cfg(date="260807", run_count="01", colmap_version="999")
        self.patch_module("detect_colmap_version", lambda: "312")
        self.assertEqual(cfg.resolve_colmap_version(), "312")

    def test_resolve_colmap_version_falls_back_to_config(self):
        cfg = self._cfg(date="260807", run_count="01", colmap_version="311")
        self.patch_module("detect_colmap_version", lambda: None)
        self.assertEqual(cfg.resolve_colmap_version(), "311")

    def test_resolve_colmap_version_raises_rather_than_inventing_a_version(self):
        cfg = self._cfg(date="260807", run_count="01")
        self.patch_module("detect_colmap_version", lambda: None)
        with self.assertRaises(ConfigError) as caught:
            cfg.resolve_colmap_version()
        message = str(caught.exception)
        self.assertIn("colmap_version", message)
        self.assertIn(str(cfg.source), message)

    def test_make_run_id_propagates_the_undetectable_version_error(self):
        cfg = self._cfg(date="260807", run_count="01")
        self.patch_module("detect_colmap_version", lambda: None)
        with self.assertRaises(ConfigError):
            cfg.make_run_id()


# --------------------------------------------------------------------------
# COLMAP version detection (simulated, never the real binary)
# --------------------------------------------------------------------------

class TestDetectColmapVersion(TempDirTestCase):

    def test_returns_none_when_colmap_is_not_on_path(self):
        self.patch_which(None)

        def explode(*_args, **_kwargs):  # pragma: no cover - must not be called
            raise AssertionError("subprocess.run must not run when colmap is absent")

        self.patch_module("subprocess", type("S", (), {
            "run": staticmethod(explode),
            "SubprocessError": subprocess.SubprocessError,
        }))
        self.assertIsNone(detect_colmap_version())

    def _install_fake_run(self, responses):
        """Map argv tuple -> FakeCompleted, and record what was tried."""
        self.calls = []

        def fake_run(argv, **_kwargs):
            self.calls.append(tuple(argv))
            return responses[tuple(argv)]

        self.patch_module("subprocess", type("S", (), {
            "run": staticmethod(fake_run),
            "SubprocessError": subprocess.SubprocessError,
        }))

    def test_version_flag_supported(self):
        """The easy shape: `colmap --version` prints the banner."""
        self.patch_which("/usr/local/bin/colmap")
        self._install_fake_run({
            ("colmap", "--version"): FakeCompleted(stdout="COLMAP 3.8 -- SfM and MVS\n"),
        })
        self.assertEqual(detect_colmap_version(), "38")
        self.assertEqual(self.calls, [("colmap", "--version")])

    def test_colmap_312_rejects_version_flag_and_prints_banner_on_bare_invocation(self):
        """The real-world 3.12 shape.

        COLMAP 3.12 does not accept ``--version``: it exits with a usage error
        and no version string. The version only appears in the banner printed by
        a bare ``colmap``. Detection must try both, in that order.
        """
        self.patch_which("/usr/local/bin/colmap")
        self._install_fake_run({
            ("colmap", "--version"): FakeCompleted(
                stderr="ERROR: unrecognised option '--version'\n", returncode=1),
            ("colmap",): FakeCompleted(
                stdout="COLMAP 3.12 -- Structure-from-Motion and Multi-View Stereo\n"
                       "\nUsage:\n  colmap [command]\n"),
        })
        self.assertEqual(detect_colmap_version(), "312")
        self.assertEqual(self.calls, [("colmap", "--version"), ("colmap",)])

    def test_version_is_read_from_stderr_as_well_as_stdout(self):
        self.patch_which("/usr/local/bin/colmap")
        self._install_fake_run({
            ("colmap", "--version"): FakeCompleted(stderr="COLMAP 3.11.1\n"),
        })
        self.assertEqual(detect_colmap_version(), "3111")

    def test_returns_none_when_nothing_parseable_comes_back(self):
        self.patch_which("/usr/local/bin/colmap")
        self._install_fake_run({
            ("colmap", "--version"): FakeCompleted(stderr="bad option\n", returncode=1),
            ("colmap",): FakeCompleted(stdout="usage: colmap [command]\n"),
        })
        self.assertIsNone(detect_colmap_version())

    def test_a_failing_invocation_does_not_abort_detection(self):
        self.patch_which("/usr/local/bin/colmap")
        calls = []

        def fake_run(argv, **_kwargs):
            calls.append(tuple(argv))
            if argv == ["colmap", "--version"]:
                raise OSError("exec format error")
            return FakeCompleted(stdout="COLMAP 3.12\n")

        self.patch_module("subprocess", type("S", (), {
            "run": staticmethod(fake_run),
            "SubprocessError": subprocess.SubprocessError,
        }))
        self.assertEqual(detect_colmap_version(), "312")
        self.assertEqual(calls, [("colmap", "--version"), ("colmap",)])


# --------------------------------------------------------------------------
# data_type inference
# --------------------------------------------------------------------------

class TestDataTypeInference(TempDirTestCase):

    def _resolved(self, images_subpath, data_type="auto"):
        tmp = Path(tempfile.mkdtemp(dir=self.tmp))
        fixture = ConfigFixture(tmp, images_subpath=images_subpath,
                                extra={"colmap": {"data_type": data_type}})
        return fixture.load().resolve_colmap_data_type()

    def test_auto_infers_video_from_a_frames_directory(self):
        self.assertEqual(self._resolved("scene-01/frames"), "video")

    def test_auto_infers_video_from_a_frames_suffixed_directory(self):
        self.assertEqual(self._resolved("scene-01/capture-a-frames"), "video")

    def test_auto_infers_individual_from_anything_else(self):
        for name in ("images", "masked-images", "session-001", "capture-a"):
            with self.subTest(name=name):
                self.assertEqual(self._resolved(f"scene-01/{name}"), "individual")

    def test_explicit_data_type_overrides_the_inference(self):
        self.assertEqual(self._resolved("scene-01/frames", "individual"), "individual")
        self.assertEqual(self._resolved("scene-01/images", "video"), "video")

    def test_only_the_last_path_component_is_inspected(self):
        self.assertEqual(self._resolved("frames-project/images"), "individual")


# --------------------------------------------------------------------------
# find_config precedence
# --------------------------------------------------------------------------

class TestFindConfig(TempDirTestCase):

    def setUp(self):
        super().setUp()
        # Preserve and restore whatever the caller's environment had.
        original = os.environ.get("DT4AG_CONFIG")
        self.addCleanup(self._restore_env, original)
        os.environ.pop("DT4AG_CONFIG", None)

    @staticmethod
    def _restore_env(original):
        if original is None:
            os.environ.pop("DT4AG_CONFIG", None)
        else:
            os.environ["DT4AG_CONFIG"] = original

    def test_explicit_argument_wins_over_everything(self):
        explicit = self.tmp / "explicit.ini"
        os.environ["DT4AG_CONFIG"] = str(self.tmp / "from-env.ini")
        walkup = self.tmp / "configs" / "example.ini"
        walkup.parent.mkdir()
        walkup.write_text("", encoding="utf-8")
        self.assertEqual(find_config(explicit, start=self.tmp), explicit)

    def test_environment_variable_wins_over_the_walk_up(self):
        from_env = self.tmp / "from-env.ini"
        os.environ["DT4AG_CONFIG"] = str(from_env)
        walkup = self.tmp / "configs" / "example.ini"
        walkup.parent.mkdir()
        walkup.write_text("", encoding="utf-8")
        self.assertEqual(find_config(start=self.tmp), from_env)

    def test_walk_up_finds_configs_example_ini_in_a_parent(self):
        root = self.tmp / "repo"
        (root / "configs").mkdir(parents=True)
        example = root / "configs" / "example.ini"
        example.write_text("", encoding="utf-8")
        deep = root / "notebooks" / "scratch"
        deep.mkdir(parents=True)
        self.assertEqual(find_config(start=deep), example)

    def test_raises_when_nothing_can_be_found(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(ConfigError) as caught:
            find_config(start=empty)
        self.assertIn("DT4AG_CONFIG", str(caught.exception))

    def test_user_home_shorthand_is_expanded(self):
        result = find_config("~/some-config.ini")
        self.assertTrue(result.is_absolute())
        self.assertNotIn("~", str(result))


# --------------------------------------------------------------------------
# run log
# --------------------------------------------------------------------------

class TestRunLog(TempDirTestCase):

    def _cfg(self):
        fixture = ConfigFixture(self.tmp, extra={
            "run": {"log_file": "logs/run-log.csv", "date": "260807",
                    "run_count": "01", "colmap_version": "312"}})
        return fixture.load()

    @staticmethod
    def _rows(cfg):
        with cfg.run_log.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_creates_the_file_with_a_header_on_first_use(self):
        cfg = self._cfg()
        self.assertFalse(cfg.run_log.exists())
        written = cfg.append_run_log("run_260807-01-312")
        self.assertEqual(written, cfg.run_log)
        self.assertTrue(cfg.run_log.is_file())
        rows = self._rows(cfg)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run_260807-01-312")
        self.assertEqual(rows[0]["colmap_version"], "312")
        self.assertEqual(rows[0]["dataset"], "scene-01/images")
        self.assertEqual(rows[0]["config_file"], str(cfg.source))

    def test_appends_rather_than_truncating(self):
        cfg = self._cfg()
        cfg.append_run_log("run_260807-01-312")
        cfg.append_run_log("run_260807-02-312")
        cfg.append_run_log("run_260807-03-312")
        rows = self._rows(cfg)
        self.assertEqual([r["run_id"] for r in rows],
                         ["run_260807-01-312", "run_260807-02-312", "run_260807-03-312"])
        # The header must appear exactly once, not once per append.
        text = cfg.run_log.read_text(encoding="utf-8")
        self.assertEqual(text.count("timestamp,run_id"), 1)

    def test_extra_fields_are_recorded_and_unknown_ones_ignored(self):
        cfg = self._cfg()
        cfg.append_run_log("run_260807-01-312", note="smoke test", bogus="dropped")
        text = cfg.run_log.read_text(encoding="utf-8")
        self.assertIn("smoke test", text)
        self.assertNotIn("dropped", text)

    def test_relative_log_file_resolves_under_data_root(self):
        cfg = self._cfg()
        self.assertEqual(cfg.run_log, cfg.data_root / "logs" / "run-log.csv")


# --------------------------------------------------------------------------
# nested image layouts (the single_camera_per_folder case)
# --------------------------------------------------------------------------

class TestNestedImageLayout(TempDirTestCase):
    """The 2026-08-07 defect: a non-recursive image count sees zero images.

    With ``single_camera_per_folder``, the captured images sit in one
    subdirectory per camera under the dataset directory rather than flat in it.
    A guard written with a non-recursive ``Path.iterdir()`` file scan therefore
    reported "0 images" for a valid 120-image dataset.

    IMPORTANT for future readers: the buggy guard is NOT in ``dt4ag_config``.
    That module does no image discovery and no image counting of any kind; the
    guard lives in the pipeline notebook. The loader is not responsible for the
    defect and there is no loader code here to fix. What these tests pin is that
    the loader accepts the nested layout, plus a flat-versus-recursive
    demonstration that keeps the defect documented and executable next to the
    config it applies to.
    """

    def _make_nested_dataset(self, cameras=4, per_camera=30):
        fixture = ConfigFixture(self.tmp, images_subpath="scene-01/masked-images",
                                make_images=False)
        images = fixture.images_path
        images.mkdir(parents=True)
        for cam in range(cameras):
            cam_dir = images / f"cam{cam:02d}"
            cam_dir.mkdir()
            for index in range(per_camera):
                (cam_dir / f"{index:04d}.png").write_bytes(b"x")
        return fixture

    def test_loader_accepts_a_dataset_whose_images_live_one_level_down(self):
        fixture = self._make_nested_dataset()
        cfg = fixture.load()
        self.assertEqual(cfg.images_path, fixture.images_path)
        self.assertEqual(cfg.colmap_single_camera_per_folder, 1)

    def test_non_recursive_scan_undercounts_a_per_camera_layout(self):
        fixture = self._make_nested_dataset(cameras=4, per_camera=30)
        images = fixture.load().images_path
        flat = [p for p in images.iterdir() if p.is_file()]
        recursive = [p for p in images.rglob("*") if p.is_file()]
        self.assertEqual(len(flat), 0, "the non-recursive scan is what saw zero images")
        self.assertEqual(len(recursive), 120)


# --------------------------------------------------------------------------
# the committed example config
# --------------------------------------------------------------------------

class TestCommittedExampleConfig(unittest.TestCase):

    def test_example_ini_exists(self):
        self.assertTrue(EXAMPLE_INI.is_file(), f"missing {EXAMPLE_INI}")

    def test_example_ini_parses(self):
        # validate_paths=False: the example points at a placeholder data root
        # that deliberately does not exist on any machine.
        cfg = load_config(EXAMPLE_INI, validate_paths=False)
        self.assertEqual(cfg.run_id_prefix, "run")
        self.assertEqual(cfg.train_method, "splatfacto")
        self.assertEqual(cfg.max_num_iterations, 30000)
        self.assertEqual(cfg.colmap_data_type, "auto")
        self.assertFalse(cfg.images_rel.is_absolute())

    def test_example_ini_leaves_run_identity_on_auto(self):
        cfg = load_config(EXAMPLE_INI, validate_paths=False)
        self.assertEqual(cfg.run_date, "")
        self.assertEqual(cfg.run_count, "")
        self.assertEqual(cfg.colmap_version, "")

    def test_example_ini_is_a_template_not_a_runnable_config(self):
        """The example must never be the config a real run picks up by accident.

        Its data_root is the placeholder ``/path/to/data``, which must not exist
        on this or any machine. If this test ever fails, someone has pointed the
        committed example at real data.
        """
        cfg = load_config(EXAMPLE_INI, validate_paths=False)
        self.assertFalse(
            cfg.data_root.exists(),
            f"the committed example config points at an existing data root "
            f"({cfg.data_root}); it is a template and must not be runnable")
        with self.assertRaises(ConfigError):
            load_config(EXAMPLE_INI)

    def test_example_ini_contains_no_subject_specific_term(self):
        text = EXAMPLE_INI.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(
                    term, text,
                    f"the committed example config must stay object-agnostic, "
                    f"but it mentions {term!r}")


# --------------------------------------------------------------------------
# public surface
# --------------------------------------------------------------------------

class TestPublicSurface(unittest.TestCase):
    """Pin the module's advertised public surface.

    Note a known inconsistency, asserted here as it currently stands rather than
    as it arguably should be: ``find_config`` is documented in the module
    docstring, used by ``_main``, and imported by the notebook, yet it is absent
    from ``__all__``. This test records the present behaviour so the gap cannot
    drift silently. Fixing ``__all__`` is a change to the module, not to this
    test; whoever makes it should update the second assertion below.
    """

    def test_all_lists_the_documented_names(self):
        self.assertEqual(
            sorted(dt4ag_config.__all__),
            ["ConfigError", "Dt4agConfig", "detect_colmap_version", "load_config"])

    def test_find_config_is_public_but_currently_missing_from_all(self):
        self.assertTrue(callable(dt4ag_config.find_config))
        self.assertNotIn(
            "find_config", dt4ag_config.__all__,
            "find_config has been added to __all__; that is an improvement, so "
            "update this test to assert its presence instead")

    def test_every_name_in_all_actually_exists(self):
        for name in dt4ag_config.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(dt4ag_config, name))


if __name__ == "__main__":
    unittest.main()
