"""Tests for ``pipeline/scripts/run-records/backfill-run-records.py``.

Stdlib ``unittest`` only, matching the rest of the suite, and hermetic: every
test builds its own throwaway data root and nothing reads the real drive.

The properties worth guarding are about HONESTY rather than mechanics. A
reconstructed record asserts provenance for a run whose config is gone, so the
ways it can be wrong are: claiming an artefact that belongs to another run,
silently overwriting a genuine record, or looking runnable when it is not.
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

# The script has a hyphenated filename, so it cannot be imported by name.
_SPEC = importlib.util.spec_from_file_location(
    "backfill_run_records",
    PIPELINE_DIR / "scripts" / "run-records" / "backfill-run-records.py",
)
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)


CONFIG_YML = """\
experiment_name: {run_id}
method_name: splatfacto
max_num_iterations: 30000
data: &id003 !!python/object/apply:pathlib.PosixPath
- /
- data
- colmap
- {quoted}
machine:
  seed: 42
  num_devices: 1
pipeline:
  datamanager:
    dataparser:
      downscale_factor: {downscale}
  model:
    background_color: random
timestamp: 2026-02-24_231152
"""


class BackfillTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "exports").mkdir(parents=True)
        (self.root / "colmap").mkdir(parents=True)

    def add_run(self, run_id, capture="cap", downscale="4", quoted=None):
        out = (self.root / "outputs" / "coll" / capture / run_id
               / "splatfacto" / "2026-02-24_231152")
        out.mkdir(parents=True)
        (out / "config.yml").write_text(
            CONFIG_YML.format(run_id=run_id, downscale=downscale,
                              quoted=quoted or run_id),
            encoding="utf-8")
        return out / "config.yml"

    def add_export(self, name):
        (self.root / "exports" / name).write_bytes(b"ply")

    def record(self, run_id):
        path = self.root / "configs" / "runs" / f"{run_id}.reconstructed.ini"
        parser = configparser.ConfigParser()
        parser.read_string(path.read_text(encoding="utf-8"))
        return parser

    def run_backfill(self, *extra):
        return backfill.main(["--data-root", str(self.root), *extra])


class TestExportOwnership(unittest.TestCase):
    """The substring bug: run id ``02`` claimed eight unrelated exports.

    Legacy run ids are short, and every export name contains digits, so a
    substring match attributes other captures' outputs to a run. That is a
    provenance record asserting something false, which is worse than a gap.
    """

    def test_matches_the_run_id_at_the_underscore_splat_boundary(self):
        self.assertTrue(backfill.export_belongs_to(
            "cap_t251128_260818-02-3120_splat_ubuntu_uv_10000steps.ply",
            "t251128_260818-02-3120"))

    def test_rejects_a_bare_substring_match(self):
        self.assertFalse(backfill.export_belongs_to(
            "test_251208_t251208_260818-02-3120_splat_ubuntu_uv.ply", "02"))

    def test_rejects_a_run_id_appearing_after_splat(self):
        self.assertFalse(backfill.export_belongs_to(
            "cap_other_splat_ubuntu_02_individual.ply", "02"))

    def test_accepts_an_export_whose_whole_prefix_is_the_run_id(self):
        self.assertTrue(backfill.export_belongs_to("02_splat_ubuntu.ply", "02"))

    def test_rejects_a_file_that_is_not_an_export(self):
        self.assertFalse(backfill.export_belongs_to("02.ply", "02"))


class TestTrainingConfigParsing(BackfillTestCase):

    def test_lifts_the_scalars_and_rebuilds_the_data_path(self):
        parsed = backfill.parse_training_config(self.add_run("run_a"))
        self.assertEqual(parsed["method_name"], "splatfacto")
        self.assertEqual(parsed["max_num_iterations"], "30000")
        self.assertEqual(parsed["downscale_factor"], "4")
        self.assertEqual(parsed["seed"], "42")
        self.assertEqual(parsed["data"], "/data/colmap/run_a")

    def test_strips_quotes_the_yaml_dumper_added_to_a_component(self):
        """`- '02'` is the component ``02``, not ``'02'``."""
        parsed = backfill.parse_training_config(
            self.add_run("02", quoted="'02'"))
        self.assertEqual(parsed["data"], "/data/colmap/02")

    def test_a_null_downscale_is_reported_as_unrecoverable(self):
        self.add_run("run_b", downscale="null")
        self.run_backfill()
        text = " ".join(self.record("run_b")["unrecoverable"].values())
        self.assertIn("downscale_factor", text)


class TestRecordContent(BackfillTestCase):

    def test_every_recorded_field_names_its_source(self):
        self.add_run("run_c")
        self.run_backfill()
        parser = self.record("run_c")
        recorded = set(parser["run-record"]) - {"run_id", "record_type"}
        self.assertTrue(recorded)
        self.assertTrue(recorded <= set(parser["sources"]))

    def test_marks_itself_reconstructed(self):
        self.add_run("run_d")
        self.run_backfill()
        self.assertEqual(self.record("run_d")["run-record"]["record_type"],
                         "reconstructed")

    def test_is_not_loadable_as_a_config(self):
        """The safety property: it must not look like a recipe for a re-run."""
        from dt4ag_config import ConfigError, load_config
        self.add_run("run_e")
        self.run_backfill()
        path = self.root / "configs" / "runs" / "run_e.reconstructed.ini"
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_says_so_when_a_run_is_absent_from_the_run_log(self):
        self.add_run("run_f")
        self.run_backfill()
        parser = self.record("run_f")
        self.assertNotIn("masks", parser["run-record"])
        self.assertIn("run log", " ".join(parser["unrecoverable"].values()))

    def test_reads_the_run_log_when_the_run_is_in_it(self):
        self.add_run("run_g")
        (self.root / "run-log.csv").write_text(
            "timestamp,run_id,dataset,images_path,colmap_version,train_method,"
            "max_num_iterations,downscale_factor,masks,object_id,imaging_date,"
            "config_file,note\n"
            "2026-08-18T18:49:03,run_g,coll/cap/images,/data/datasets/cap/images,"
            "3120,splatfacto,10000,4,used,,,configs/x.ini,\n",
            encoding="utf-8")
        self.run_backfill()
        record = self.record("run_g")["run-record"]
        self.assertEqual(record["masks"], "used")
        self.assertEqual(record["colmap_version"], "3120")
        self.assertEqual(record["config_source"], "configs/x.ini")
        # config.yml outranks the log for what actually trained.
        self.assertEqual(record["max_num_iterations"], "30000")
        self.assertEqual(record["configured_max_num_iterations"], "10000")

    def test_records_every_training_attempt_under_one_run_id(self):
        self.add_run("run_h")
        second = (self.root / "outputs" / "coll" / "cap" / "run_h"
                  / "splatfacto" / "2026-02-24_235959")
        second.mkdir(parents=True)
        (second / "config.yml").write_text(
            CONFIG_YML.format(run_id="run_h", downscale="4", quoted="run_h"),
            encoding="utf-8")
        self.run_backfill()
        record = self.record("run_h")["run-record"]
        self.assertEqual(record["training_attempts"], "2")
        self.assertIn("2026-02-24_235959", record["training_configs"])


class TestGenuineRecordsAreNeverOverwritten(BackfillTestCase):
    """A real frozen config outranks any reconstruction of it, always."""

    def test_skips_a_run_that_already_has_a_frozen_config(self):
        self.add_run("run_i")
        archive = self.root / "configs" / "runs"
        archive.mkdir(parents=True)
        genuine = archive / "run_i.ini"
        genuine.write_text("[paths]\ndata_root = /data\n", encoding="utf-8")
        self.run_backfill()
        self.assertEqual(genuine.read_text(encoding="utf-8"),
                         "[paths]\ndata_root = /data\n")
        self.assertFalse((archive / "run_i.reconstructed.ini").exists())


class TestDryRun(BackfillTestCase):

    def test_writes_nothing(self):
        self.add_run("run_j")
        self.assertEqual(self.run_backfill("--dry-run"), 0)
        self.assertFalse((self.root / "configs").exists())

    def test_is_idempotent(self):
        self.add_run("run_k")
        self.run_backfill()
        first = (self.root / "configs" / "runs" / "run_k.reconstructed.ini")
        before = first.read_text(encoding="utf-8")
        self.run_backfill()
        self.assertEqual(first.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
