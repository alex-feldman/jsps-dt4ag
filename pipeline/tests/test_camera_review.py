"""Tests for the pure helpers in ``pipeline/scripts/camera-review/camera-review.py``.

Stdlib ``unittest`` only. The script's heavy imports (numpy, matplotlib, nerfstudio) happen
inside ``main``, so the module loads here without them. Drawing the plot and reading COLMAP
binaries need the pipeline environment and a real solve, and are covered by a real run only.
"""

from __future__ import annotations

import importlib.util
import io
import math
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "camera_review", PIPELINE_DIR / "scripts" / "camera-review" / "camera-review.py")
cr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cr)


class CaptureCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.capture = Path(self._tmp.name) / "capture"
        (self.capture / "images").mkdir(parents=True)

    def images(self, *names):
        for name in names:
            (self.capture / "images" / name).write_bytes(b"x")

    def frames_csv(self, text):
        folder = self.capture / "provenance"
        folder.mkdir()
        (folder / "frames.csv").write_text(text)


class TestFrameTimes(CaptureCase):

    def test_uses_the_clip_seconds_column(self):
        self.frames_csv("filename,source_seconds_approx\na.png,1.5\nb.png,3.0\n")
        times, label = cr.frame_times(self.capture)
        self.assertEqual(times, {"a.png": 1.5, "b.png": 3.0})
        self.assertEqual(label, "clip seconds (approx)")

    def test_falls_back_to_the_grid_seconds_column(self):
        self.frames_csv("filename,output_grid_seconds\na.png,0.5\n")
        self.assertEqual(cr.frame_times(self.capture)[0], {"a.png": 0.5})

    def test_without_a_csv_the_label_is_frame_order_over_the_images(self):
        self.images("b.png", "a.PNG", "notes.txt")
        times, label = cr.frame_times(self.capture)
        self.assertEqual(times, {"a.PNG": 0.0, "b.png": 1.0})
        self.assertEqual(label, "frame order")

    def test_a_csv_without_a_known_time_column_or_with_no_rows_falls_back(self):
        self.images("a.png", "b.png")
        self.frames_csv("filename,something\na.png,9\n")
        self.assertEqual(cr.frame_times(self.capture)[1], "frame order")
        (self.capture / "provenance" / "frames.csv").write_text("filename,source_seconds_approx\n")
        self.assertEqual(cr.frame_times(self.capture)[1], "frame order")

    def test_images_in_subfolders_are_found(self):
        (self.capture / "images" / "sub").mkdir()
        (self.capture / "images" / "sub" / "c.jpg").write_bytes(b"x")
        self.assertEqual(list(cr.frame_times(self.capture)[0]), ["c.jpg"])


class TestComparison(unittest.TestCase):

    def test_missing_and_unexpected_are_sorted_lists(self):
        result = cr.compare_frames(["c", "a", "b"], ["b", "z", "a"])
        self.assertEqual(result, {"missing": ["c"], "unexpected": ["z"]})

    def test_a_full_registration_has_nothing_to_report(self):
        self.assertEqual(cr.compare_frames(["a", "b"], ["b", "a"]),
                         {"missing": [], "unexpected": []})

    def test_finite_span_ignores_nan_and_infinity(self):
        nan = float("nan")
        self.assertEqual(cr.finite_span([nan, 2.0, 5.0, float("inf"), 1.0]), [1.0, 5.0])
        self.assertEqual(cr.finite_span([nan, nan]), [])
        self.assertEqual(cr.finite_span([]), [])


class TestCameraStepReview(unittest.TestCase):

    def test_flags_a_disproportionate_jump_and_names_its_transition(self):
        result = cr.summarize_camera_steps(
            ["frame_000055.png", "frame_000056.png", "frame_000057.png", "frame_000058.png"],
            [27.0, 27.5, 28.0, 28.5],
            [0.25, 5.06, 0.27],
            "clip seconds (approx)",
        )
        self.assertAlmostEqual(
            result["median_step_between_adjacent_registered_cameras"], 0.27)
        self.assertAlmostEqual(
            result["largest_step_ratio_to_median"], 5.06 / 0.27)
        self.assertEqual(result["largest_step_from"], "frame_000056.png")
        self.assertEqual(result["largest_step_to"], "frame_000057.png")
        self.assertEqual(result["largest_step_time_interval"], [27.5, 28.0])
        self.assertEqual(len(result["camera_path_warnings"]), 1)
        self.assertIn("before training", result["camera_path_warnings"][0])

    def test_does_not_warn_for_ordinary_variation(self):
        result = cr.summarize_camera_steps(
            ["a", "b", "c", "d"], [0.0, 0.5, 1.0, 1.5],
            [0.2, 0.25, 0.7], "clip seconds (approx)")
        self.assertAlmostEqual(result["largest_step_ratio_to_median"], 2.8)
        self.assertEqual(result["camera_path_warnings"], [])

    def test_nonzero_step_with_zero_median_warns_without_infinity_in_json(self):
        result = cr.summarize_camera_steps(
            ["a", "b", "c", "d"], [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.0, 1.0], "clip seconds (approx)")
        self.assertIsNone(result["largest_step_ratio_to_median"])
        self.assertIn("unbounded relative to the median", result["camera_path_warnings"][0])

    def test_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            cr.summarize_camera_steps(["a", "b"], [0.0], [1.0], "frame order")


class TestRotationSteps(unittest.TestCase):

    def test_identical_and_sign_flipped_quaternions_are_zero_degrees(self):
        angles = cr.rotation_steps([(1, 0, 0, 0), (-1, 0, 0, 0)])
        self.assertAlmostEqual(angles[0], 0.0, places=6)

    def test_a_half_turn_is_180_degrees(self):
        angles = cr.rotation_steps([(1, 0, 0, 0), (0, 0, 1, 0)])
        self.assertAlmostEqual(angles[0], 180.0, places=6)

    def test_a_ten_degree_turn(self):
        half = math.radians(10) / 2
        angles = cr.rotation_steps([(1, 0, 0, 0), (math.cos(half), 0, math.sin(half), 0)])
        self.assertAlmostEqual(angles[0], 10.0, places=6)

    def test_one_camera_has_no_steps(self):
        self.assertEqual(cr.rotation_steps([(1, 0, 0, 0)]), [])


class TestPointsPerImage(unittest.TestCase):

    def test_counts_only_features_with_a_3d_point(self):
        self.assertEqual(cr.points_per_image([[-1, 4, 7], [-1, -1], []]), [2, 0, 0])

    def test_low_point_frames_are_under_a_tenth_of_the_median(self):
        result = cr.low_point_frames(["a", "b", "c", "d"], [1000, 1100, 90, 1200])
        self.assertEqual(result["median_points_per_image"], 1050.0)
        self.assertEqual(result["low_point_frames"], [{"frame": "c", "points": 90}])

    def test_no_frames_has_no_median(self):
        self.assertEqual(cr.low_point_frames([], []),
                         {"median_points_per_image": None, "low_point_frames": []})

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            cr.low_point_frames(["a"], [1, 2])


class TestFlagCameraSteps(unittest.TestCase):

    def names(self, n):
        return [f"f{i}" for i in range(n)]

    def test_lists_every_flagged_step_with_its_turn_not_only_the_largest(self):
        names = self.names(8)
        steps = [0.3, 0.3, 9.0, 0.3, 0.3, 12.0, 0.3]
        turns = [5, 5, 170, 5, 5, 20, 5]
        result = cr.flag_camera_steps(names, [float(i) for i in range(8)], steps, turns, [1.0] * 6)
        self.assertEqual(result["flagged_step_count"], 2)
        self.assertEqual([s["to"] for s in result["flagged_steps"]], ["f3", "f6"])
        self.assertTrue(result["flagged_steps"][0]["also_large_turn"])
        self.assertFalse(result["flagged_steps"][1]["also_large_turn"])
        self.assertEqual(result["large_turn_count"], 1)
        self.assertEqual(len(result["warnings"]), 1)

    def test_a_gap_in_registration_is_not_mistaken_for_a_jump(self):
        names = ["a", "b", "c", "d", "e"]
        times = [0.0, 1.0, 2.0, 13.0, 14.0]
        steps = [0.3, 0.3, 3.3, 0.3]
        result = cr.flag_camera_steps(names, times, steps, [5] * 4, [0.6] * 3)
        self.assertEqual(result["flagged_step_count"], 0)
        self.assertEqual(result["warnings"], [])

    def test_a_jump_away_and_straight_back_names_the_misplaced_camera(self):
        names = self.names(6)
        steps = [0.3, 8.0, 8.0, 0.3, 0.3]
        spans = [8.2, 0.3, 8.2, 0.6]
        result = cr.flag_camera_steps(names, [float(i) for i in range(6)], steps,
                                      [5, 170, 170, 5, 5], spans)
        self.assertEqual(result["out_and_back"], ["f2"])
        self.assertIn("f2", result["warnings"][0])

    def test_a_clean_walk_has_nothing_to_report(self):
        names = self.names(5)
        result = cr.flag_camera_steps(names, [0.0, 1.0, 2.0, 3.0, 4.0],
                                      [0.3, 0.35, 0.3, 0.4], [4, 6, 7, 5], [0.6, 0.6, 0.7])
        self.assertEqual(result["flagged_steps"], [])
        self.assertEqual(result["large_turns"], [])
        self.assertEqual(result["out_and_back"], [])
        self.assertEqual(result["warnings"], [])

    def test_a_large_turn_alone_is_reported(self):
        names = self.names(4)
        result = cr.flag_camera_steps(names, [0.0, 1.0, 2.0, 3.0], [0.3, 0.3, 0.3],
                                      [4, 150, 4], [0.6, 0.6])
        self.assertEqual(result["flagged_steps"], [])
        self.assertEqual(result["large_turn_count"], 1)
        self.assertEqual(len(result["warnings"]), 1)

    def test_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            cr.flag_camera_steps(["a", "b", "c"], [0.0, 1.0, 2.0], [1.0, 1.0], [1.0], [1.0])
        with self.assertRaises(ValueError):
            cr.flag_camera_steps(["a", "b", "c"], [0.0, 1.0, 2.0], [1.0, 1.0], [1.0, 1.0], [])


class TestDistantPairs(unittest.TestCase):

    def make_database(self, directory, names, pairs):
        path = Path(directory) / "database.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        for image_id, name in enumerate(names, start=1):
            connection.execute("INSERT INTO images VALUES (?, ?)", (image_id, name))
        for first, second, rows in pairs:
            connection.execute("INSERT INTO two_view_geometries VALUES (?, ?)",
                               (first * 2147483647 + second, rows))
        connection.commit()
        connection.close()
        return path

    def test_counts_verified_pairs_that_join_distant_frames(self):
        names = [f"f{i}.png" for i in range(10)]
        order = {n: float(i) for i, n in enumerate(names)}
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(tmp, names, [(1, 2, 50), (2, 3, 40), (1, 9, 30), (3, 10, 20), (4, 5, 0)])
            result = cr.distant_pair_summary(database, order)
        self.assertEqual(result["verified_pairs"], 4)
        self.assertEqual(result["distant_pairs"], 2)
        self.assertAlmostEqual(result["distant_pair_fraction"], 0.5)

    def test_a_database_with_no_verified_pairs_has_no_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(tmp, ["a.png", "b.png"], [(1, 2, 0)])
            result = cr.distant_pair_summary(database, {"a.png": 0.0, "b.png": 1.0})
        self.assertEqual(result["verified_pairs"], 0)
        self.assertIsNone(result["distant_pair_fraction"])


class TestSparseModels(unittest.TestCase):

    def test_lists_only_folders_that_hold_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self.assertEqual(cr.sparse_models(workspace), [])
            for name, has_model in (("0", True), ("1", True), ("2", False)):
                folder = workspace / "sparse" / name
                folder.mkdir(parents=True)
                if has_model:
                    (folder / "images.bin").write_bytes(b"")
            (workspace / "sparse" / "stray.txt").write_text("x")
            self.assertEqual(cr.sparse_models(workspace), ["0", "1"])


class TestArguments(unittest.TestCase):

    def test_required_and_default_options(self):
        args = cr.build_parser().parse_args(
            ["--capture", "/c", "--workspace", "/w", "--out", "/o"])
        self.assertEqual((args.model, args.label), ("0", ""))
        self.assertEqual(args.capture, Path("/c"))

    def test_missing_required_option_is_a_usage_error(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cr.build_parser().parse_args(["--capture", "/c"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
