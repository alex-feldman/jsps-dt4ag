"""Tests for ``pipeline/scripts/run-records/recover-run-configs.py``.

Stdlib ``unittest`` only, matching the rest of the suite, and hermetic.

This tool archives a config file and asserts it is the one a given run used.
The failure that matters is therefore not a crash: it is claiming a config that
has been edited since the run, which would put a plausible, runnable, WRONG
recipe in the archive under a real run id. Most of what follows tests the
refusal path.
"""

from __future__ import annotations

import configparser
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "recover_run_configs",
    PIPELINE_DIR / "scripts" / "run-records" / "recover-run-configs.py",
)
recover = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recover)

LOG_HEADER = ("timestamp,run_id,dataset,images_path,colmap_version,train_method,"
              "max_num_iterations,downscale_factor,masks,object_id,imaging_date,"
              "config_file,note\n")

CONFIG = """\
[paths]
data_root = {data_root}

[dataset]
images_subpath = coll/cap/images
image_extensions = .jpg
mask_extensions = .png
use_masks = true

[run]
id_prefix = t

[colmap]

[nerfstudio]

[train]
method = splatfacto
max_num_iterations = {iters}
downscale_factor = {ds}

[export]
"""


class RecoverTestCase(unittest.TestCase):

    RUN = "t_260818-01-3120"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.data_root = self.root / "data"
        (self.data_root / "datasets").mkdir(parents=True)
        self.repo = self.root / "repo"
        (self.repo / "configs").mkdir(parents=True)

    def write_config(self, iters="10000", ds="4", subpath="coll/cap/images"):
        text = CONFIG.format(data_root=self.data_root, iters=iters, ds=ds)
        text = text.replace("images_subpath = coll/cap/images",
                            f"images_subpath = {subpath}")
        path = self.repo / "configs" / "run.ini"
        path.write_text(text, encoding="utf-8")
        return path

    def write_log(self, iters="10000", ds="4", masks="used",
                  images="coll/cap/images", method="splatfacto",
                  config="configs/run.ini"):
        (self.data_root / "run-log.csv").write_text(
            LOG_HEADER
            + f"2026-08-18T18:49:03,{self.RUN},{images},"
              f"{self.data_root / 'datasets' / images},3120,{method},"
              f"{iters},{ds},{masks},,,{config},\n",
            encoding="utf-8")

    def run_tool(self, *extra):
        return recover.main(["--data-root", str(self.data_root),
                             "--repo-root", str(self.repo),
                             "--stamp", "2026-08-21", *extra])

    @property
    def archived(self):
        return self.data_root / "configs" / "runs" / f"{self.RUN}.ini"


class TestRecoversAMatchingConfig(RecoverTestCase):

    def test_copies_the_file_verbatim_and_appends_a_record(self):
        source = self.write_config()
        self.write_log()
        self.run_tool()
        text = self.archived.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(source.read_text(encoding="utf-8")))
        parser = configparser.ConfigParser()
        parser.read_string(text)
        self.assertEqual(parser["run-record"]["record_type"], "recovered-verified")
        self.assertEqual(parser["run-record"]["run_id"], self.RUN)

    def test_names_the_fields_it_verified(self):
        """The record must show the strength of its own claim."""
        self.write_config()
        self.write_log()
        self.run_tool()
        parser = configparser.ConfigParser()
        parser.read_string(self.archived.read_text(encoding="utf-8"))
        verified = parser["run-record"]["verified_fields"]
        for field in ("images_subpath", "max_num_iterations", "downscale_factor",
                      "use_masks", "train_method"):
            self.assertIn(field, verified)

    def test_the_archived_file_is_still_loadable_as_a_config(self):
        from dt4ag_config import load_config
        capture = self.data_root / "datasets" / "coll" / "cap"
        (capture / "images").mkdir(parents=True)
        (capture / "images" / "a.jpg").write_bytes(b"x")
        # use_masks = true, so the loader requires the sibling masks directory.
        (capture / "masks").mkdir()
        (capture / "masks" / "a.png").write_bytes(b"x")
        self.write_config()
        self.write_log()
        self.run_tool()
        self.assertEqual(load_config(self.archived).max_num_iterations, 10000)


class TestRefusesAnEditedConfig(RecoverTestCase):
    """One disagreeing field is enough. A near-miss is still the wrong config."""

    def assert_refused(self):
        self.run_tool()
        self.assertFalse(self.archived.exists())

    def test_refuses_when_the_iteration_count_changed(self):
        self.write_config(iters="30000")
        self.write_log(iters="10000")
        self.assert_refused()

    def test_refuses_when_the_images_subpath_changed(self):
        """The real case: the 2026-08-18 canonical-layout migration."""
        self.write_config(subpath="coll/cap/images")
        self.write_log(images="coll/cap")
        self.assert_refused()

    def test_refuses_when_the_downscale_factor_changed(self):
        self.write_config(ds="4")
        self.write_log(ds="2")
        self.assert_refused()

    def test_refuses_when_the_masking_route_changed(self):
        self.write_config()
        self.write_log(masks="none")
        self.assert_refused()

    def test_refuses_when_the_config_file_is_gone(self):
        self.write_log(config="configs/vanished.ini")
        self.assert_refused()

    def test_refuses_when_the_log_row_carries_nothing_checkable(self):
        self.write_config()
        self.write_log(iters="", ds="", masks="", images="", method="")
        self.assert_refused()

    def test_an_unpinned_downscale_matches_the_literal_auto(self):
        """The log writes `auto` when the key is 0 or absent, not `0`."""
        self.write_config(ds="0")
        self.write_log(ds="auto")
        self.run_tool()
        self.assertTrue(self.archived.exists())


class TestSafety(RecoverTestCase):

    def test_never_overwrites_an_existing_record(self):
        """A genuine frozen config outranks any after-the-fact recovery."""
        self.write_config()
        self.write_log()
        archive = self.data_root / "configs" / "runs"
        archive.mkdir(parents=True)
        self.archived.write_text("GENUINE", encoding="utf-8")
        self.run_tool()
        self.assertEqual(self.archived.read_text(encoding="utf-8"), "GENUINE")

    def test_dry_run_writes_nothing(self):
        self.write_config()
        self.write_log()
        self.run_tool("--dry-run")
        self.assertFalse(self.archived.exists())

    def test_is_idempotent(self):
        self.write_config()
        self.write_log()
        self.run_tool()
        before = self.archived.read_text(encoding="utf-8")
        self.run_tool()
        self.assertEqual(self.archived.read_text(encoding="utf-8"), before)

    def test_exits_non_zero_without_a_run_log(self):
        self.assertEqual(self.run_tool(), 2)


if __name__ == "__main__":
    unittest.main()
