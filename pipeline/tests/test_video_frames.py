"""Tests for ``pipeline/scripts/video-frames/video-frames.py``.

Stdlib ``unittest`` only, matching the rest of the suite, and hermetic: ffmpeg
and ffprobe are replaced by fakes, so these pass on a machine that has neither.

What matters most here is the refusal path. The tool must not overwrite a
capture, must not extract from a source whose hash is not the one named, and
must leave a record (never a half-trusted frame set) when extraction fails or
yields frames of more than one shape. What the real ffmpeg does with a real
movie can only be proved by a real run.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "video_frames",
    PIPELINE_DIR / "scripts" / "video-frames" / "video-frames.py",
)
vf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vf)

MOVIE_BYTES = b"not really a movie, only bytes to hash" * 50
MOVIE_SHA = hashlib.sha256(MOVIE_BYTES).hexdigest()

PROBE = {
    "format": {"format_name": "mov,mp4", "duration": "54.4", "size": "123456"},
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "hevc", "width": 3840,
         "height": 2880, "r_frame_rate": "24000/1001", "nb_frames": "1305",
         "pix_fmt": "yuv420p10le", "color_transfer": "bt709", "duration": "54.429375"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        {"index": 5, "codec_type": "video", "codec_name": "mjpeg", "width": 960,
         "height": 720, "disposition": {"attached_pic": 1}},
    ],
}


def png_header(width: int, height: int) -> bytes:
    """The 24 bytes the tool reads: signature, IHDR length and tag, then size."""
    return vf.PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


class FakeRun:
    """Stands in for ``subprocess.run``; records commands and writes outputs."""

    def __init__(self, shapes=((8, 6),) * 5, returncode=0):
        self.shapes = list(shapes)
        self.returncode = returncode
        self.commands = []

    def __call__(self, cmd, stdout=None, stderr=None, **_kw):
        self.commands.append(list(cmd))
        out = Path(cmd[-1])
        if cmd[-1].endswith("frame_%06d.png"):
            for index, (w, h) in enumerate(self.shapes, start=1):
                (out.parent / f"frame_{index:06d}.png").write_bytes(png_header(w, h))
        else:
            name = out.name.replace("%02d", "01")
            (out.parent / name).write_bytes(b"jpeg")

        class Result:
            pass
        result = Result()
        result.returncode = self.returncode
        return result


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "clip.mp4"
        self.source.write_bytes(MOVIE_BYTES)
        self.capture = self.tmp / "datasets" / "new-capture"
        self.fake = FakeRun()
        self._saved = (vf.find_tool, vf.probe, vf.ffmpeg_version, vf.subprocess.run)
        vf.find_tool = lambda name, prefix: f"/fake/{name}"
        vf.probe = lambda ffprobe, path: PROBE
        vf.ffmpeg_version = lambda ffmpeg: "ffmpeg version test"
        vf.subprocess.run = self.fake
        self.addCleanup(self._restore)

    def _restore(self):
        vf.find_tool, vf.probe, vf.ffmpeg_version, vf.subprocess.run = self._saved

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = vf.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def extract(self, *extra, sha=MOVIE_SHA, interval="0.5"):
        return self.run_main("extract", "--source", str(self.source), "--sha256", sha,
                             "--capture", str(self.capture), "--interval", interval, *extra)


class TestHelpers(unittest.TestCase):
    def test_fps_expression_is_exact(self):
        self.assertEqual(vf.fps_expression(Fraction("0.5")), "2")
        self.assertEqual(vf.fps_expression(Fraction(5)), "1/5")
        self.assertEqual(vf.fps_expression(Fraction(2, 3)), "3/2")
        self.assertEqual(vf.fps_expression(Fraction(1, 4)), "4")

    def test_interval_label(self):
        self.assertEqual(vf.interval_label(Fraction("0.5")), "0500ms")
        self.assertEqual(vf.interval_label(Fraction(2)), "2000ms")

    def test_estimate_frames_matches_what_ffmpeg_produced(self):
        # Measured against ffmpeg 9.0.1's fps filter on synthetic clips (the filter's
        # output is round(duration / interval), halves up). A ceil() estimate was wrong
        # for most of these, and agrees with round() only on the easy cases.
        measured = [
            (10.0, "0.3", 33), (10.0, "0.7", 14), (10.0, "3", 3), (10.0, "1", 10),
            (10.0, "0.5", 20), (10.2917, "1", 10), (10.2917, "2", 5),
            # recorded runs: a 3 s range at 0.5 s, the Welzo bed video, the cactus clip
            (3.0, "0.5", 6), (90.4, "0.5", 181), (54.43, "0.5", 109),
        ]
        for duration, interval, frames in measured:
            self.assertEqual(vf.estimate_frames(duration, Fraction(interval)), frames,
                             f"{duration} s at {interval} s")
        self.assertEqual(vf.estimate_frames(0, Fraction(1)), 0)

    def test_parse_interval_refuses_nonsense(self):
        for bad in ("0", "-1", "abc", ""):
            with self.assertRaises(vf.ToolError, msg=bad):
                vf.parse_interval(bad)

    def test_plan_candidates_covers_the_window_in_sheets_of_twenty(self):
        plan = vf.plan_candidates(90.0, Fraction(0), Fraction(30), [Fraction(2), Fraction(1), Fraction("0.5")])
        self.assertEqual([(g["frames"], g["sheet_count"]) for g in plan], [(15, 1), (30, 2), (60, 3)])
        self.assertEqual(plan[1]["label"], "1000ms")
        self.assertEqual(plan[1]["pattern"], "candidate-1000ms-%02d.jpg")
        sixty = vf.plan_candidates(90.0, Fraction(0), Fraction(60), [Fraction("0.5")])
        self.assertEqual((sixty[0]["frames"], sixty[0]["sheet_count"]), (120, 6))

    def test_plan_candidates_stops_at_the_end_of_the_clip(self):
        plan = vf.plan_candidates(20.0, Fraction(10), Fraction(30), [Fraction(1)])
        self.assertEqual((plan[0]["window_seconds"], plan[0]["frames"]), (10.0, 10))
        with self.assertRaises(vf.ToolError):
            vf.plan_candidates(20.0, Fraction(25), Fraction(30), [Fraction(1)])

    def test_capture_root_is_guessed_only_from_an_evals_layout(self):
        self.assertEqual(vf.guess_capture_root(Path("/d/evals/subject")), str(Path("/d/datasets")))
        self.assertEqual(vf.guess_capture_root(Path("/somewhere/else")), "")

    def test_png_size_read_and_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "a.png", Path(tmp) / "b.png"
            good.write_bytes(png_header(2880, 3840))
            bad.write_bytes(b"GIF89a" + b"\x00" * 30)
            self.assertEqual(vf.read_png_size(good), (2880, 3840))
            with self.assertRaises(vf.ToolError):
                vf.read_png_size(bad)

    def test_sha256_is_streamed_and_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            path.write_bytes(MOVIE_BYTES * 30000)
            self.assertEqual(vf.sha256_of(path), hashlib.sha256(MOVIE_BYTES * 30000).hexdigest())

    def test_summary_skips_attached_picture_and_reads_rotation(self):
        facts = vf.summarize_probe(PROBE)
        self.assertEqual((facts["width"], facts["height"], facts["codec"]), (3840, 2880, "hevc"))
        self.assertAlmostEqual(facts["duration_seconds"], 54.429375)
        rotated = json.loads(json.dumps(PROBE))
        rotated["streams"][0]["side_data_list"] = [{"rotation": -90}]
        self.assertEqual(vf.summarize_probe(rotated)["rotation"], -90)

    def test_summary_refuses_a_source_with_no_video(self):
        with self.assertRaises(vf.ToolError):
            vf.summarize_probe({"format": {}, "streams": [{"codec_type": "audio"}]})


class TestExtract(Base):
    def test_dry_run_writes_nothing(self):
        code, out, _ = self.extract("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("dry run: nothing written", out)
        self.assertFalse(self.capture.exists())
        self.assertEqual(self.fake.commands, [])

    def test_refuses_an_existing_capture_and_leaves_it_alone(self):
        self.capture.mkdir(parents=True)
        (self.capture / "keep.txt").write_text("mine")
        code, _, err = self.extract()
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertEqual([p.name for p in self.capture.iterdir()], ["keep.txt"])

    def test_refuses_a_wrong_hash_before_creating_anything(self):
        code, _, err = self.extract(sha="0" * 64)
        self.assertEqual(code, 2)
        self.assertIn("hash", err)
        self.assertFalse(self.capture.exists())

    def test_refuses_a_malformed_hash(self):
        code, _, err = self.extract(sha="abc")
        self.assertEqual(code, 2)
        self.assertIn("64-character", err)

    def test_refuses_a_range_outside_the_clip_before_creating_anything(self):
        # The probed clip is 54.43 s long.
        for argv, fragment in (
            (["--start", "100", "--end", "110"], "past the end"),
            (["--start", "54.5"], "past the end"),
            (["--end", "90"], "past the end"),
        ):
            code, _, err = self.extract(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn(fragment, err)
            self.assertFalse(self.capture.exists(), argv)
            self.assertEqual(self.fake.commands, [])

    def test_an_end_within_rounding_of_the_clip_end_is_accepted(self):
        code, out, _ = self.extract("--end", "54.45", "--dry-run")
        self.assertEqual(code, 0, out)

    def test_refuses_a_range_that_ends_before_it_starts(self):
        code, _, err = self.extract("--start", "10", "--end", "5")
        self.assertEqual(code, 2)
        self.assertFalse(self.capture.exists())

    def test_happy_path_leaves_the_full_record(self):
        code, out, _ = self.extract("--identity", "test subject")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(list((self.capture / "images").glob("frame_*.png"))), 5)
        staged = self.capture / "source" / "clip.mp4"
        self.assertEqual(hashlib.sha256(staged.read_bytes()).hexdigest(), MOVIE_SHA)
        for name in ("extraction.json", "ffprobe.json", "ffmpeg.log", "frames.csv"):
            self.assertTrue((self.capture / "provenance" / name).is_file(), name)
        record = json.loads((self.capture / "provenance" / "extraction.json").read_text())
        self.assertEqual(record["status"], vf.STATUS_PENDING_REVIEW)
        self.assertEqual(record["actual_frame_count"], 5)
        self.assertEqual(record["source_identity"], "test subject")
        self.assertEqual(record["source_sha256"], MOVIE_SHA)
        self.assertEqual(record["dimensions"], [[8, 6]])
        self.assertIn("timestamp_warning", record)
        self.assertFalse(any(p.suffix == ".mp4" for p in (self.capture / "images").iterdir()))

    def test_command_is_no_overwrite_explicit_stream_and_cpu(self):
        self.extract()
        cmd = self.fake.commands[0]
        self.assertIn("-n", cmd)
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:v:0")
        self.assertEqual(cmd[cmd.index("-hwaccel") + 1], "none")
        self.assertEqual(cmd[cmd.index("-vf") + 1], "fps=2,showinfo")
        self.assertEqual(cmd[cmd.index("-threads") + 1], "2")
        self.assertNotIn("-ss", cmd)
        self.assertNotIn("-t", cmd)

    def test_range_adds_seek_and_duration_and_shifts_recorded_times(self):
        code, _, _ = self.extract("--start", "10", "--end", "12")
        self.assertEqual(code, 0)
        cmd = self.fake.commands[0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "10.0")
        self.assertEqual(cmd[cmd.index("-t") + 1], "2.0")
        with (self.capture / "provenance" / "frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["output_grid_seconds"] for r in rows[:3]], ["0.0", "0.5", "1.0"])
        self.assertEqual([r["source_seconds_approx"] for r in rows[:3]], ["10.0", "10.5", "11.0"])

    def test_mixed_shapes_stop_the_run_and_say_do_not_train(self):
        self.fake.shapes = [(8, 6), (6, 8), (8, 6)]
        code, _, err = self.extract()
        self.assertEqual(code, 2)
        self.assertIn("do not train", err)
        record = json.loads((self.capture / "provenance" / "extraction.json").read_text())
        self.assertEqual(record["status"], vf.STATUS_NEEDS_REVIEW)

    def test_no_frames_stops_the_run(self):
        self.fake.shapes = []
        code, _, _ = self.extract()
        self.assertEqual(code, 2)
        record = json.loads((self.capture / "provenance" / "extraction.json").read_text())
        self.assertEqual(record["status"], vf.STATUS_NEEDS_REVIEW)

    def test_ffmpeg_failure_is_recorded_and_preserved(self):
        self.fake.returncode = 1
        code, _, err = self.extract()
        self.assertEqual(code, 2)
        self.assertIn("Preserve this attempt", err)
        record = json.loads((self.capture / "provenance" / "extraction.json").read_text())
        self.assertEqual(record["status"], vf.STATUS_FAILED)
        self.assertTrue((self.capture / "source" / "clip.mp4").is_file())


class TestSubset(Base):
    """``subset`` thins an extracted capture by symlink; it must never copy."""

    def setUp(self):
        super().setUp()
        self.fake.shapes = [(8, 6)] * 12
        code, _, _ = self.extract()
        self.assertEqual(code, 0)
        self.child = self.tmp / "datasets" / "child"

    def subset(self, *extra, every="2"):
        return self.run_main("subset", "--from", str(self.capture), "--capture", str(self.child),
                             "--every", every, *extra)

    def test_every_second_frame_is_linked_not_copied(self):
        code, out, _ = self.subset()
        self.assertEqual(code, 0, out)
        links = sorted((self.child / "images").iterdir())
        self.assertEqual([p.name for p in links],
                         [f"frame_{i:06d}.png" for i in (1, 3, 5, 7, 9, 11)])
        for link in links:
            self.assertTrue(link.is_symlink(), link.name)
            self.assertEqual(link.resolve(), (self.capture / "images" / link.name).resolve())
        record = json.loads((self.child / "provenance" / "subset.json").read_text())
        self.assertEqual((record["every"], record["offset"], record["frame_count"]), (2, 0, 6))
        self.assertEqual(record["interval_seconds"], 1.0)
        self.assertEqual(record["parent_source_sha256"], MOVIE_SHA)
        self.assertEqual(record["status"], vf.STATUS_PENDING_REVIEW)
        with (self.child / "provenance" / "frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["filename"] for r in rows], record["frames"])
        self.assertEqual(rows[1]["output_grid_seconds"], "1.0")

    def test_links_are_relative_and_survive_moving_the_whole_datasets_tree(self):
        self.subset()
        link = self.child / "images" / "frame_000001.png"
        self.assertFalse(Path(os.readlink(link)).is_absolute())
        moved = self.tmp / "restored" / "datasets"
        moved.parent.mkdir()
        shutil.move(str(self.tmp / "datasets"), str(moved))
        for name in ("frame_000001.png", "frame_000011.png"):
            self.assertTrue((moved / "child" / "images" / name).is_file(), name)
        self.assertEqual((moved / "child" / "images" / "frame_000001.png").resolve(),
                         (moved / "new-capture" / "images" / "frame_000001.png").resolve())

    def test_every_fourth_with_an_offset(self):
        code, _, _ = self.subset("--offset", "1", every="4")
        self.assertEqual(code, 0)
        self.assertEqual([p.name for p in sorted((self.child / "images").iterdir())],
                         [f"frame_{i:06d}.png" for i in (2, 6, 10)])
        record = json.loads((self.child / "provenance" / "subset.json").read_text())
        self.assertEqual(record["interval_seconds"], 2.0)

    def test_dry_run_writes_nothing(self):
        code, out, _ = self.subset("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("dry run", out)
        self.assertFalse(self.child.exists())

    def test_refuses_nonsense_selections(self):
        for argv, fragment in (
            (["--every", "1"], "at least 2"),
            (["--every", "4", "--offset", "4"], "--offset"),
            (["--every", "6"], "need at least"),
        ):
            code, _, err = self.run_main("subset", "--from", str(self.capture),
                                         "--capture", str(self.child), *argv)
            self.assertEqual(code, 2, argv)
            self.assertIn(fragment, err)
            self.assertFalse(self.child.exists())

    def test_refuses_an_existing_capture(self):
        self.child.mkdir(parents=True)
        code, _, err = self.subset()
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)

    def test_refuses_a_parent_that_is_not_an_extracted_capture(self):
        code, _, err = self.run_main("subset", "--from", str(self.tmp), "--capture", str(self.child),
                                     "--every", "2")
        self.assertEqual(code, 2)
        self.assertIn("not a capture made by extract", err)

    def test_refuses_a_parent_that_needs_review(self):
        manifest = self.capture / "provenance" / "extraction.json"
        record = json.loads(manifest.read_text())
        record["status"] = vf.STATUS_NEEDS_REVIEW
        manifest.write_text(json.dumps(record))
        code, _, err = self.subset()
        self.assertEqual(code, 2)
        self.assertIn("not good", err)
        self.assertFalse(self.child.exists())

    def test_refuses_a_parent_whose_frames_changed_since_extraction(self):
        (self.capture / "images" / "frame_000003.png").unlink()
        code, _, err = self.subset()
        self.assertEqual(code, 2)
        self.assertIn("changed since extraction", err)

    def test_never_falls_back_to_copying_when_links_are_impossible(self):
        real = Path.symlink_to

        def refuse(self_path, target, *a, **k):
            raise OSError("symlinks not permitted")
        Path.symlink_to = refuse
        self.addCleanup(setattr, Path, "symlink_to", real)
        code, _, err = self.subset()
        self.assertEqual(code, 2)
        self.assertIn("never copies", err)
        self.assertFalse(self.child.exists())
        self.assertEqual(len(list((self.capture / "images").glob("frame_*.png"))), 12)


class TestReview(Base):
    def run_review(self, *extra):
        return self.run_main("review", "--source", str(self.source),
                             "--review-root", str(self.tmp / "evals"), *extra)

    def review_dir(self):
        found = list((self.tmp / "evals").glob("sampling-review-*"))
        self.assertEqual(len(found), 1)
        return found[0]

    def test_review_writes_provenance_and_sheets_only_in_a_new_directory(self):
        code, out, _ = self.run_review()
        self.assertEqual(code, 0, out)
        review = self.review_dir()
        names = {p.name for p in review.iterdir()}
        for expected in ("source.sha256", "ffprobe.json", "summary.json", "review-notes.json",
                         "index.html", "contact-01.jpg", "candidate-2000ms-01.jpg",
                         "candidate-1000ms-01.jpg", "candidate-0500ms-01.jpg"):
            self.assertIn(expected, names)
        self.assertIn(MOVIE_SHA, (review / "source.sha256").read_text())
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["clip.mp4", "evals"])
        self.assertEqual(self.source.read_bytes(), MOVIE_BYTES)

    def test_review_leaves_every_human_decision_empty(self):
        self.run_review()
        notes = json.loads((self.review_dir() / "review-notes.json").read_text())
        for key in ("human_source_identity", "human_selected_range",
                    "human_selected_interval_seconds", "human_verdict", "agent_proposal"):
            self.assertIsNone(notes[key], key)
        self.assertTrue(notes["selected_interval_does_not_verify_overlap"])

    def test_summary_estimates_frames_per_interval(self):
        self.run_review()
        summary = json.loads((self.review_dir() / "summary.json").read_text())
        self.assertEqual(summary["estimated_frames_by_interval_seconds"]["0.5"], 109)
        self.assertEqual(summary["width"], 3840)

    def test_candidate_sheets_cover_the_preview_window_in_numbered_sheets(self):
        self.run_review("--candidate-intervals", "1", "--candidate-start", "20")
        candidate = [c for c in self.fake.commands if c[-1].endswith("candidate-1000ms-%02d.jpg")][0]
        self.assertEqual(candidate[candidate.index("-ss") + 1], "20.0")
        # 54.43 s clip, start 20: the 30 s default window fits.
        self.assertEqual(candidate[candidate.index("-t") + 1], "30.0")
        self.assertNotIn("-frames:v", candidate)
        coarse = [c for c in self.fake.commands if c[-1].endswith("contact-%02d.jpg")][0]
        self.assertIn("fps=1/5,scale=320:-1,tile=5x4", coarse)
        self.assertNotIn("-ss", coarse)

    def test_preview_seconds_widens_the_window_and_is_cut_at_the_clip_end(self):
        self.run_review("--candidate-intervals", "1", "--preview-seconds", "60")
        candidate = [c for c in self.fake.commands if c[-1].endswith("candidate-1000ms-%02d.jpg")][0]
        self.assertEqual(candidate[candidate.index("-t") + 1], str(float(Fraction(str(54.429375)))))

    def test_index_page_carries_the_hash_groups_and_no_placeholder(self):
        self.run_review()
        page = (self.review_dir() / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("__DATA__", page)
        self.assertIn(MOVIE_SHA, page)
        self.assertIn("candidate-1000ms-%02d.jpg", page)
        self.assertIn("Build the extract command", page)
        self.assertIn(str(self.tmp / "datasets"), page)

    def test_index_page_cannot_be_broken_out_of_by_its_data(self):
        out = self.tmp / "page"
        out.mkdir()
        for hostile in ("x</script><b>y", "a<!--<script>b", "q&amp;<z>"):
            page = vf.write_index(out, {"source": hostile, "facts": {}}).read_text(encoding="utf-8")
            self.assertEqual(page.count("</script>"), 1, hostile)
            self.assertNotIn("<!--<script", page, hostile)
            self.assertNotIn(hostile, page, hostile)
            self.assertIn("\\u003c", page)

    def test_page_command_uses_python3_and_an_absolute_script_path(self):
        self.run_review()
        page = (self.review_dir() / "index.html").read_text(encoding="utf-8")
        self.assertIn("python3 " + str(Path(vf.__file__).resolve()), page.replace("\\/", "/"))

    def test_a_bad_candidate_start_leaves_nothing_behind(self):
        code, _, err = self.run_review("--candidate-start", "100")
        self.assertEqual(code, 2)
        self.assertIn("past the end", err)
        self.assertEqual(list((self.tmp / "evals").glob("*")) if (self.tmp / "evals").exists() else [], [])
        self.assertEqual(self.fake.commands, [])

    def test_sheet_failure_is_reported(self):
        self.fake.returncode = 1
        code, _, err = self.run_review()
        self.assertEqual(code, 2)
        self.assertIn("sheets failed", err)

    def test_missing_source_is_refused(self):
        code, _, err = self.run_main("review", "--source", str(self.tmp / "nope.mp4"),
                                     "--review-root", str(self.tmp / "evals"))
        self.assertEqual(code, 2)
        self.assertIn("source not found", err)


if __name__ == "__main__":
    unittest.main()
