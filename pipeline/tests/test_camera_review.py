"""Tests for the pure helpers in ``pipeline/scripts/camera-review/camera-review.py``.

Stdlib ``unittest`` only. The script's heavy imports (numpy, matplotlib, nerfstudio) happen
inside ``main``, so the module loads here without them. Drawing the plot and reading COLMAP
binaries need the pipeline environment and a real solve, and are covered by a real run only.
"""

from __future__ import annotations

import importlib.util
import io
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
        self.assertEqual(list(cr.frame_times(self.capture)[0]), ["sub/c.jpg"])


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

    def test_masked_pngs_match_nested_original_jpegs(self):
        times, result = cr.reconcile_frame_names({"a/frame.jpg": 0.0, "b/frame.jpg": 1.0}, ["a/frame.png", "b/frame.png"])
        self.assertEqual(times, {"a/frame.png": 0.0, "b/frame.png": 1.0})
        self.assertEqual(result, {"missing": [], "unexpected": []})

    def test_ambiguous_stems_are_refused(self):
        with self.assertRaisesRegex(ValueError, "ambiguous frame stem"):
            cr.reconcile_frame_names({"a.jpg": 0.0, "a.png": 1.0}, ["a.png"])


class TestSegments(unittest.TestCase):

    def test_folder_boundaries_and_missing_frames(self):
        expected = ["a/01.jpg", "a/02.jpg", "a/03.jpg", "b/01.jpg", "b/02.jpg"]
        registered = ["a/01.png", "a/03.png", "b/01.png", "b/02.png"]
        self.assertEqual(cr.adjacent_expected_pairs(expected, registered),
                         [("b/01.png", "b/02.png")])

    def test_single_video_sequence_and_filtered_statistics(self):
        names = ["01.png", "02.png", "03.png"]
        self.assertEqual(cr.adjacent_expected_pairs(names, names), list(zip(names, names[1:])))
        summary = cr.summarize_camera_steps(names, [0, 1, 2], [1, 100], "frame order", [0])
        self.assertEqual(summary["largest_step_between_adjacent_registered_cameras"], 1)
        self.assertEqual(summary["largest_step_to"], "02.png")


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
