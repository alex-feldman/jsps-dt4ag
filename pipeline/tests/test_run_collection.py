"""Tests for ``pipeline/run_collection.py``.

Stdlib ``unittest`` only and hermetic: every tree is built in a temporary
directory, and no reconstruction tool runs. The per-capture child is either
``FakeChild``, which writes the catalog job record a real run would, or
``run_pipeline.main`` itself with every external command mocked
(``real_child``), so completion is checked against records the actual
producer wrote.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_collection  # noqa: E402
import run_pipeline as runner  # noqa: E402
from dt4ag_config import load_config  # noqa: E402
from plant_catalog import CatalogJob, MILESTONES  # noqa: E402
from run_collection import CollectionError, discover_captures  # noqa: E402
from test_run_pipeline import CONFIG_TEMPLATE  # noqa: E402
import colmap_threads  # noqa: E402


def setUpModule():
    # These tests run the colmap stage, whose thread choice reads the machine's
    # memory (and Windows', on WSL). Fixed figures keep them hermetic.
    figures = colmap_threads.HostMemory(linux_available_mib=65536, usable_cpus=8)
    patcher = patch.object(colmap_threads, "read_host_memory", return_value=figures)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    version = patch.object(runner, "detect_colmap_version", return_value="3120")
    version.start()
    unittest.addModuleCleanup(version.stop)

ROOT_ID = "test-root"
#: Photographs are dated well before any job starts; see why_incomplete's mtime check.
OLD = time.time() - 3600
PLY = "ply\nformat binary_little_endian 1.0\nelement vertex {n}\nend_header\n"


def write_ply(path: Path, vertices: int = 1, padding: int = 2048) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PLY.format(n=vertices) + " " * padding)


class FakeChild:
    """Stands in for one ``run_pipeline.py`` invocation; writes its catalog job, runs no tool.

    ``outcomes`` maps a capture id to ``ok`` (default), ``fail``, ``cancelled``,
    ``interrupt`` (raises KeyboardInterrupt) or ``silent`` (exit 0, no record).
    """

    def __init__(self, repo: Path, outcomes=None):
        self.repo = repo
        self.outcomes = outcomes or {}
        self.calls = []
        self.argvs = []
        self.active = False

    def __call__(self, argv):
        assert not self.active, "captures must run one at a time"
        self.active = True
        try:
            return self.run(argv)
        finally:
            self.active = False

    def run(self, argv):
        args = runner.build_parser().parse_args(argv[2:])
        capture_id = args.catalog_capture_id
        self.calls.append(capture_id)
        self.argvs.append(list(argv))
        outcome = self.outcomes.get(capture_id, "ok")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "silent":
            return 0
        cfg = load_config(args.config)
        with patch("plant_catalog.git_value", return_value=None):
            job = CatalogJob(args.catalog_root, args.catalog_root_id, capture_id, self.repo,
                             title="test", job_type="reconstruction", command=" ".join(argv),
                             config={"stages": list(runner.STAGES), "run_id_override": None})
        job.record["config"]["ini"] = runner.ini_sections(cfg.source)
        job.record["selected_images"] = runner.selected_images(cfg)
        job.capture(source=cfg.images_path, image_count=len(job.record["selected_images"]))
        for name in ("config_ok", "prereqs_ok", "colmap_done", "transforms_ok", "pyramid_ok"):
            job.milestone(name)
        if outcome in ("fail", "cancelled"):
            job.start_stage("train")
            code = 1 if outcome == "fail" else 130
            job.finish(code, "training failed\nsecond line")
            return code
        for name in ("trained", "checkpoint_found", "exported"):
            job.milestone(name)
        ply = cfg.exports_dir / f"{capture_id}_splat.ply"
        write_ply(ply)
        job.artifact(ply, role="point-cloud", media_type="application/octet-stream")
        job.milestone("verified")
        job.finish(0)
        return 0


def real_child(argv):
    """``run_pipeline.main`` for one capture, with every external command mocked."""
    args = runner.build_parser().parse_args(argv[2:])
    cfg = load_config(args.config)
    run_id = cfg.make_run_id()
    workspace = cfg.colmap_workspace(run_id)

    def command(argv_, label, dry_run):
        if label == "colmap":
            (workspace / "sparse/0").mkdir(parents=True, exist_ok=True)
            (workspace / "sparse/0/cameras.bin").write_bytes(b"camera")
        elif label == "ns-process-data":
            (workspace / "images").mkdir(exist_ok=True)
            (workspace / "images/frame_00001.jpg").write_bytes(b"image")
            (workspace / "transforms.json").write_text(
                json.dumps({"frames": [{"file_path": "images/frame_00001.jpg"}]}))
        elif label == "ns-train":
            models = cfg.output_dir(run_id) / cfg.train_method / "trained" / "nerfstudio_models"
            models.mkdir(parents=True, exist_ok=True)
            (models / "step-000000499.ckpt").write_bytes(b"checkpoint")
            (models.parent / "config.yml").write_text("load_step: null\n")
        elif label == "ns-export":
            write_ply(cfg.exports_dir / runner.export_filename(cfg, run_id, 1, cfg.max_num_iterations, None))

    with patch.object(runner, "check_prerequisites"), \
            patch.object(runner, "run_command", side_effect=command), \
            patch.object(runner, "read_run_eval_token", return_value=None), \
            patch("plant_catalog.git_value", return_value=None), \
            patch("plant_catalog.subprocess.run", side_effect=FileNotFoundError):
        return runner.main(argv[2:])


def tree_digest(folder: Path) -> dict:
    """Content hash and mtime of every file under ``folder``."""
    return {p.relative_to(folder).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


class CollectionFixture(unittest.TestCase):
    """A data root that is also the catalog root, a collection and a base config."""

    def setUp(self):
        self.fresh_root()

    def fresh_root(self):
        """A new, empty data and catalog root; subtests call it to start clean."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.datasets = self.root / "datasets"
        self.collection = self.datasets / "coll"
        self.collection.mkdir(parents=True)
        self.base_path = self.write_base()
        self.child = FakeChild(self.root)

    def write_base(self, train_extra="eval_mode = all\ndownscale_factor = 1", data_root=None,
                   path=None, **overrides):
        values = dict(data_root=data_root or self.root, extra_args="", skip_colmap="true",
                      dataset_extra="", train_extra=train_extra, images_subpath="ignored/images",
                      host_section="")
        values.update(overrides)
        path = path or self.root / "base.ini"
        path.parent.mkdir(parents=True, exist_ok=True)
        # A collection needs auto-incremented run ids; the shared template pins one.
        text = CONFIG_TEMPLATE.format(**values).replace("run_count = 03", "run_count =")
        path.write_text(text, encoding="utf-8")
        return path

    def edit_base(self, old, new):
        text = self.base_path.read_text()
        self.assertIn(old, text)
        self.base_path.write_text(text.replace(old, new))

    def capture(self, rel, photos=("cam1/a.jpg", "cam1/b.jpg")):
        images = self.collection / rel / "images"
        images.mkdir(parents=True, exist_ok=True)
        for name in photos:
            self.photo(images / name)
        return self.collection / rel

    @staticmethod
    def photo(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"photo " + path.name.encode())
        os.utime(path, (OLD, OLD))

    def args(self, *extra, collection="coll", catalog_root=None):
        return ["--config", str(self.base_path), "--collection", collection,
                "--catalog-root", str(catalog_root or self.root), "--catalog-root-id", ROOT_ID, *extra]

    def main(self, *extra, child=None, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_collection.main(self.args(*extra, **kwargs), runner=child or self.child)
        return code, out.getvalue() + err.getvalue()

    def plan(self, **kwargs):
        args = run_collection.build_parser().parse_args(self.args(**kwargs))
        return {str(c.rel): c for c in run_collection.plan_collection(args).captures}

    def jobs(self, capture_id=None):
        return [job for job in run_collection.read_jobs(self.root)
                if capture_id is None or job["capture_id"] == capture_id]

    def summary_line(self, output, rel):
        lines = output.split("collection summary")[-1].splitlines()
        return next(line for line in lines if f" {rel} " in line)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

class TestDiscovery(CollectionFixture):
    def found(self, collection=None):
        captures, ignored = discover_captures(collection or self.collection)
        base = collection or self.collection
        return ([c.relative_to(base).as_posix() for c in captures],
                [(p.relative_to(base).as_posix(), why) for p, why in ignored])

    def test_only_a_directory_holding_images_is_a_capture(self):
        self.capture("site/a")
        self.capture("site/b")
        (self.collection / "notes").mkdir()
        self.photo(self.collection / "site" / "stray.jpg")  # image files alone do not make a capture
        self.assertEqual(self.found()[0], ["site/a", "site/b"])

    def test_a_capture_can_itself_be_the_collection(self):
        capture = self.capture("site/a")
        self.assertEqual(self.found(capture)[0], ["."])

    def test_camera_subdirectories_inside_images_are_not_captures(self):
        self.capture("a", photos=("cam1/x.jpg", "cam2/deep/y.jpg"))
        self.assertEqual(self.found()[0], ["a"])

    def test_capture_inside_a_capture_is_refused(self):
        for nested in ("a/images/cam1/images/x.jpg", "a/extra/images/x.jpg", "images/images/x.jpg"):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as folder:
                collection = Path(folder)
                self.photo(collection / "a/images/cam1/y.jpg")
                self.photo(collection / nested)
                with self.assertRaisesRegex(CollectionError, "inside another capture"):
                    discover_captures(collection)

    def test_hidden_directories_are_not_searched(self):
        self.capture("a")
        self.capture(".trash/old")
        self.photo(self.collection / "a/.cache/images/x.jpg")
        self.assertEqual(self.found(), (["a"], []))

    def test_symlinks_are_not_followed_and_are_reported(self):
        self.capture("a")
        (self.collection / "b").mkdir()
        try:
            (self.collection / "link-to-a").symlink_to(self.collection / "a", target_is_directory=True)
            (self.collection / "b/images").symlink_to(self.collection / "a/images", target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"no symlinks here: {exc}")
        captures, ignored = self.found()
        self.assertEqual(captures, ["a"])
        self.assertEqual([path for path, _ in ignored], ["b/images", "link-to-a"])
        self.assertIn("its parent is not a capture", ignored[0][1])

    def test_order_is_by_path_whatever_the_creation_order(self):
        for rel in ("z", "m/b", "a", "m/a", "B"):
            self.capture(rel)
        self.assertEqual(self.found()[0], ["B", "a", "m/a", "m/b", "z"])
        code, output = self.main("--dry-run")
        listed = [line.split()[1] for line in output.splitlines() if line.startswith("  run ")]
        self.assertEqual(listed, ["coll/B", "coll/a", "coll/m/a", "coll/m/b", "coll/z"])

    def test_empty_images_is_found_then_refused_without_stopping_the_rest(self):
        self.capture("a")
        (self.collection / "empty/images").mkdir(parents=True)
        self.capture("z")
        code, output = self.main()
        self.assertEqual(code, 1)
        self.assertEqual(self.child.calls, ["coll-a", "coll-z"])
        self.assertIn("no photographs", self.summary_line(output, "coll/empty"))
        self.assertIn("invalid", self.summary_line(output, "coll/empty"))

    def test_no_capture_is_a_collection_error(self):
        (self.collection / "a/raw-images").mkdir(parents=True)
        code, output = self.main()
        self.assertEqual(code, 2)
        self.assertIn("no capture", output)
        self.assertEqual(self.child.calls, [])


# --------------------------------------------------------------------------
# one base config, derived per capture
# --------------------------------------------------------------------------

class TestDerivedConfigs(CollectionFixture):
    def test_each_capture_overrides_images_subpath_from_one_base(self):
        for rel in ("a", "site/b"):
            self.capture(rel)
        code, _ = self.main()
        self.assertEqual(code, 0)
        base = runner.ini_sections(self.base_path)
        for argv, rel in zip(self.child.argvs, ("coll/a", "coll/site/b")):
            config = Path(argv[argv.index("--config") + 1])
            self.assertEqual(config.parent, self.root / "derived/collection-configs" / rel)
            cfg = load_config(config)
            self.assertEqual(cfg.images_path, self.datasets / rel / "images")
            self.assertEqual(cfg.capture_rel, Path(rel))
            derived = runner.ini_sections(config)
            self.assertEqual(derived["dataset"].pop("images_subpath"), f"{rel}/images")
            self.assertEqual(derived["paths"].pop("data_root"), str(self.root))
            base["dataset"].pop("images_subpath", None)
            base["paths"].pop("data_root", None)
            self.assertEqual(derived, base)

    def test_relative_data_root_resolves_against_the_base_config(self):
        self.capture("a")
        self.base_path = self.write_base(data_root="..", path=self.root / "configs/base.ini")
        code, _ = self.main()
        self.assertEqual(code, 0)
        argv = self.child.argvs[0]
        self.assertEqual(load_config(argv[argv.index("--config") + 1]).data_root, self.root)

    def test_child_receives_catalog_identity_and_passthrough_flags(self):
        self.capture("site/Cap_01")
        self.main("--allow-full-resolution")
        argv = self.child.argvs[0]
        self.assertEqual(argv[:2], [sys.executable, str(PIPELINE_DIR / "run_pipeline.py")])
        self.assertEqual(argv[argv.index("--catalog-capture-id") + 1], "coll-site-cap-01")
        self.assertEqual(argv[argv.index("--catalog-root-id") + 1], ROOT_ID)
        self.assertIn("--allow-full-resolution", argv)
        self.assertNotIn("--allow-viewer-hang", argv)

    def test_derived_config_is_never_rewritten(self):
        self.capture("a")
        self.main()
        [config] = (self.root / "derived/collection-configs/coll/a").glob("*.ini")
        before = config.stat().st_mtime_ns
        self.edit_base("max_num_iterations = 500", "max_num_iterations = 600")
        self.main()
        self.assertEqual(config.stat().st_mtime_ns, before)
        self.assertEqual(len(list(config.parent.glob("*.ini"))), 2)

    def test_invalid_base_settings_mark_every_capture_and_run_nothing(self):
        for edit, reason in ((("eval_mode = all", "eval_mode = sometimes"), "eval_mode"),
                             (("quit_on_train_completion = true", "quit_on_train_completion = false"), "never return"),
                             (("run_count =", "run_count = 03"), "run_count is pinned")):
            with self.subTest(reason=reason):
                self.fresh_root()
                self.capture("a")
                self.capture("b")
                self.edit_base(*edit)
                code, output = self.main()
                self.assertEqual(code, 1)
                self.assertEqual(self.child.calls, [])
                self.assertIn(reason, self.summary_line(output, "coll/a"))

    def test_dry_run_writes_and_runs_nothing(self):
        self.capture("a")
        (self.collection / "empty/images").mkdir(parents=True)
        code, output = self.main("--dry-run")
        self.assertEqual(code, 1)  # an invalid capture is reported in the exit code
        self.assertEqual(self.child.calls, [])
        self.assertFalse((self.root / "derived").exists())
        self.assertFalse((self.root / "catalog").exists())
        self.assertIn("  run      coll/a  [coll-a]  no catalog job", output)

    def test_capture_id_collision_refuses_the_collection(self):
        self.capture("Cap_1")
        self.capture("cap-1")
        code, output = self.main()
        self.assertEqual(code, 2)
        self.assertIn("same catalog capture id", output)
        self.assertEqual(self.child.calls, [])

    def test_capture_record_naming_another_source_is_refused(self):
        self.capture("a")
        record = self.root / "catalog/captures/coll-a.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"kind": "capture", "id": "coll-a",
                                      "source": {"root_id": ROOT_ID, "relative_path": "datasets/elsewhere/images"}}))
        code, output = self.main()
        self.assertEqual((code, self.child.calls), (1, []))
        self.assertIn("already names", self.summary_line(output, "coll/a"))

    def test_data_outside_the_catalog_root_is_refused(self):
        self.capture("a")
        with tempfile.TemporaryDirectory() as other:
            code, output = self.main(catalog_root=other)
        self.assertEqual((code, self.child.calls), (1, []))
        self.assertIn("not beneath the catalog root", output)

    def test_collection_errors(self):
        self.capture("a")
        for kwargs, extra, message in (
                (dict(collection="missing"), (), "does not exist"),
                (dict(collection="../outside"), (), "must not climb"),
                (dict(collection="/"), (), "not inside the datasets directory")):
            with self.subTest(message=message):
                code, output = self.main(*extra, **kwargs)
                self.assertEqual(code, 2)
                self.assertIn(message, output)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PLANT_CATALOG_ROOT_ID", None)
            code = run_collection.main(["--config", str(self.base_path), "--collection", "coll",
                                        "--catalog-root", str(self.root)], runner=self.child)
        self.assertEqual(code, 2)
        self.assertIn("--catalog-root-id", err.getvalue())


# --------------------------------------------------------------------------
# sequential execution and failure handling
# --------------------------------------------------------------------------

class TestExecution(CollectionFixture):
    def test_one_failure_does_not_stop_the_rest_and_the_summary_names_it(self):
        for rel in ("a", "b", "c"):
            self.capture(rel)
        self.child.outcomes = {"coll-b": "fail"}
        code, output = self.main()
        self.assertEqual(code, 1)
        self.assertEqual(self.child.calls, ["coll-a", "coll-b", "coll-c"])
        line = self.summary_line(output, "coll/b")
        self.assertTrue(line.strip().startswith("failed"))
        self.assertIn("exit 1, error in stage train after pyramid_ok: training failed", line)
        self.assertIn(self.jobs("coll-b")[0]["id"], line)
        for rel in ("coll/a", "coll/c"):
            self.assertTrue(self.summary_line(output, rel).strip().startswith("completed"))

    def test_interrupt_stops_the_batch_and_reports_what_did_not_start(self):
        for outcome in ("interrupt", "cancelled"):
            with self.subTest(outcome=outcome):
                self.fresh_root()
                for rel in ("a", "b", "c"):
                    self.capture(rel)
                self.child.outcomes = {"coll-b": outcome}
                code, output = self.main()
                self.assertEqual(code, 130)
                self.assertEqual(self.child.calls, ["coll-a", "coll-b"])
                self.assertIn("not started", self.summary_line(output, "coll/c"))
                self.assertIn("interrupted", self.summary_line(output, "coll/b"))

    def test_exit_zero_without_a_verifying_record_is_a_failure(self):
        self.capture("a")
        self.child.outcomes = {"coll-a": "silent"}
        code, output = self.main()
        self.assertEqual(code, 1)
        self.assertIn("no catalog job was written", self.summary_line(output, "coll/a"))

    def test_input_photographs_are_unchanged(self):
        for rel in ("a", "b"):
            self.capture(rel)
        before = tree_digest(self.datasets)
        code, _ = self.main(child=real_child)
        self.assertEqual(code, 0)
        self.assertEqual(tree_digest(self.datasets), before)

    def test_run_child_returns_the_exit_code(self):
        self.assertEqual(run_collection.run_child([sys.executable, "-c", "raise SystemExit(3)"]), 3)


# --------------------------------------------------------------------------
# resume: completion from catalog records
# --------------------------------------------------------------------------

class TestResume(CollectionFixture):
    def complete_one(self):
        self.capture("a")
        code, _ = self.main()
        self.assertEqual(code, 0)
        [job] = self.jobs("coll-a")
        return job

    def job_path(self, job):
        return self.root / "catalog/jobs" / f"{job['id']}.json"

    def export_path(self, job):
        return self.root / job["progress"]["artifacts"][0]["relative_path"]

    def assert_runs_again(self, reason):
        capture = self.plan()["coll/a"]
        self.assertIsNone(capture.complete_job)
        self.assertIn(reason, capture.reason)

    def test_rerun_skips_completed_work_without_a_duplicate(self):
        for child in (FakeChild, real_child):
            with self.subTest(child=getattr(child, "__name__", "fake")):
                self.fresh_root()
                if child is FakeChild:
                    child = self.child
                for rel in ("a", "b"):
                    self.capture(rel)
                self.assertEqual(self.main(child=child)[0], 0)
                outputs = tree_digest(self.root / "exports")
                calls = []
                code, output = self.main(child=lambda argv: calls.append(argv) or 0)
                self.assertEqual((code, calls), (0, []))
                self.assertTrue(self.summary_line(output, "coll/a").strip().startswith("skipped"))
                completed = [job for job in self.jobs() if job["execution_status"] == "completed"]
                self.assertEqual(sorted(job["capture_id"] for job in completed), ["coll-a", "coll-b"])
                self.assertEqual(tree_digest(self.root / "exports"), outputs)

    def test_rerun_of_a_partly_failed_batch_runs_only_the_failure(self):
        for rel in ("a", "b", "c"):
            self.capture(rel)
        self.child.outcomes = {"coll-b": "fail"}
        self.assertEqual(self.main()[0], 1)
        outputs = tree_digest(self.root / "exports")
        retry = FakeChild(self.root)
        code, _ = self.main(child=retry)
        self.assertEqual((code, retry.calls), (0, ["coll-b"]))
        after = tree_digest(self.root / "exports")
        self.assertEqual({k: v for k, v in after.items() if k in outputs}, outputs)
        self.assertEqual(len(self.jobs("coll-b")), 2)  # the failed attempt is kept, not replaced

    def test_incomplete_or_invalid_records_are_not_completion(self):
        def edit(change):
            def apply(job):
                change(job)
                self.job_path(job).write_text(json.dumps(job))
            return apply
        cases = {
            "failed": (edit(lambda j: j.update(execution_status="failed", exit_code=1)), "status failed"),
            "running": (edit(lambda j: j.update(execution_status="running", exit_code=None)), "status running"),
            "unverified": (edit(lambda j: j["progress"].update(
                milestones=[m for m in j["progress"]["milestones"] if m["name"] != "verified"],
                furthest_milestone="exported")), "never verified"),
            "missing export": (lambda j: self.export_path(j).unlink(), "no longer exists"),
            "empty export": (lambda j: self.export_path(j).write_bytes(b""), "below MIN_EXPORT_BYTES"),
            "zero vertices": (lambda j: write_ply(self.export_path(j), vertices=0, padding=2048), "0 vertices"),
            "replaced export": (lambda j: write_ply(self.export_path(j), padding=4096), "its record says"),
            "missing artifact": (lambda j: (self.root / "catalog/artifacts" / f"{j['output_artifact_ids'][0]}.json").unlink(),
                                 "missing or unreadable"),
            "no export": (edit(lambda j: j.update(output_artifact_ids=[])), "no export artifact"),
        }
        for name, (damage, reason) in cases.items():
            with self.subTest(name):
                self.fresh_root()
                damage(self.complete_one())
                self.assert_runs_again(reason)
                retry = FakeChild(self.root)
                self.assertEqual(self.main(child=retry)[0], 0)
                self.assertEqual(retry.calls, ["coll-a"])

    def test_changed_base_config_is_not_skipped(self):
        self.complete_one()
        self.edit_base("max_num_iterations = 500", "max_num_iterations = 600")
        self.assert_runs_again("[train] max_num_iterations")

    def test_rerun_after_a_change_writes_beside_earlier_outputs(self):
        self.capture("a")
        self.assertEqual(self.main(child=real_child)[0], 0)
        before = tree_digest(self.root / "exports")
        self.edit_base("env_label = my-env", "env_label = other-env")
        self.assertEqual(self.main(child=real_child)[0], 0)
        after = tree_digest(self.root / "exports")
        self.assertEqual({k: v for k, v in after.items() if k in before}, before)
        self.assertEqual(len(after), 2)
        self.assertEqual(len(list((self.root / "colmap/coll/a").iterdir())), 2)  # two run ids
        self.assertEqual(len(self.jobs("coll-a")), 2)

    def test_unrelated_comment_or_layout_change_still_skips(self):
        self.complete_one()
        self.base_path.write_text("# a comment\n" + self.base_path.read_text().replace(
            "env_label = my-env", "env_label    =   my-env   ; inline note"))
        self.assertIsNotNone(self.plan()["coll/a"].complete_job)

    def test_changed_image_set_is_not_skipped(self):
        self.complete_one()
        self.photo(self.collection / "a/images/cam1/c.jpg")
        self.assert_runs_again("1 added, 0 removed")
        (self.collection / "a/images/cam1/c.jpg").unlink()
        (self.collection / "a/images/cam1/a.jpg").unlink()
        self.assert_runs_again("0 added, 1 removed")

    def test_photograph_modified_after_the_run_is_not_skipped(self):
        self.complete_one()
        photo = self.collection / "a/images/cam1/a.jpg"
        later = time.time() + 60
        os.utime(photo, (later, later))
        self.assert_runs_again("cam1/a.jpg was modified after that run started")

    def test_newest_completing_job_wins_over_a_later_failure(self):
        job = self.complete_one()
        failed_record = dict(job, id="job-later", execution_status="failed", exit_code=1,
                             started_at="2999-01-01T00:00:00+00:00")
        self.job_path(failed_record).write_text(json.dumps(failed_record))
        self.assertEqual(self.plan()["coll/a"].complete_job, job["id"])

    def test_old_record_shapes(self):
        job = self.complete_one()
        legacy = {k: v for k, v in job.items() if k != "progress"}
        self.job_path(job).write_text(json.dumps(legacy))
        self.assert_runs_again("predates progress milestones")
        schema_one = dict(job, progress={k: v for k, v in job["progress"].items()
                                         if k not in ("gpu", "artifacts", "failure")})
        schema_one["progress"]["schema_version"] = 1
        self.job_path(job).write_text(json.dumps(schema_one))
        self.assertEqual(self.plan()["coll/a"].complete_job, job["id"])
        without_config = dict(job, config={"stages": list(runner.STAGES)})
        self.job_path(job).write_text(json.dumps(without_config))
        self.assert_runs_again("no configuration")

    def test_unreadable_job_records_prove_nothing(self):
        self.capture("a")
        jobs = self.root / "catalog/jobs"
        jobs.mkdir(parents=True)
        (jobs / "job-broken.json").write_text("{not json")
        (jobs / "job-list.json").write_text("[]")
        self.assert_runs_again("no catalog job")

    def test_milestone_ladder_still_names_verified(self):
        # The completion rule depends on these names; a rename must fail here first.
        self.assertIn("verified", MILESTONES)
        self.assertEqual(set(run_collection.EXPORT_ROLES), {"gaussian-splat", "point-cloud"})


if __name__ == "__main__":
    unittest.main()
