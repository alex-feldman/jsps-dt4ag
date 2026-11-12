"""Tests for ``pipeline/scripts/frame-selection/frame-selection.py``.

Stdlib ``unittest`` only and hermetic: every test builds its own capture in a temporary
directory, and ffmpeg is replaced by a fake that writes placeholder JPEGs, so the suite needs no
ffmpeg. What the real ffmpeg makes of real images, and how the page behaves in a browser, are
checked only by a real run.

What matters most: the source images are never modified, the sampling phase is fixed on the
original sequence before exclusions, and ``build`` refuses a selection that does not follow from
its own rules or was made on a different image set.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fs = _load("frame_selection", "scripts/frame-selection/frame-selection.py")
camera_review = _load("camera_review_for_fs", "scripts/camera-review/camera-review.py")


def png_bytes(width=8, height=6, salt=b""):
    return (fs.PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
            + b"rest-of-file" + salt)


class FakeRun:
    """Stands in for ``subprocess.run``: records commands and writes each ``.jpg`` output."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.commands = []

    def __call__(self, cmd, stdout=None, stderr=None, **_kw):
        self.commands.append(list(cmd))
        if self.returncode == 0:
            for arg in cmd:
                if str(arg).endswith(".jpg"):
                    Path(arg).write_bytes(b"jpeg")

        class Result:
            pass
        result = Result()
        result.returncode = self.returncode
        result.stdout = b""
        return result


def tree_state(root: Path):
    """Every path under ``root`` with its bytes hash, size and mtime, links not followed."""
    state = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else None
        state[str(path.relative_to(root))] = (digest, st.st_size, st.st_mtime_ns, path.is_symlink())
    return state


class Base(unittest.TestCase):
    N = 12

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "datasets" / "src-capture"
        (self.source / "images").mkdir(parents=True)
        for i in range(1, self.N + 1):
            (self.source / "images" / f"frame_{i:06d}.png").write_bytes(png_bytes(salt=bytes([i])))
        self.fake = FakeRun()
        self._saved = (fs.find_tool, fs.ffmpeg_version, fs.subprocess.run, fs.tool_version)
        fs.find_tool = lambda name, prefix: f"/fake/{name}"
        fs.ffmpeg_version = lambda ffmpeg: "ffmpeg version test"
        fs.subprocess.run = self.fake
        fs.tool_version = lambda: "test-version"
        self.addCleanup(self._restore)
        self.out = self.tmp / "evals" / "subject" / "preview-1"
        self.dest = self.tmp / "datasets" / "new-capture"

    def _restore(self):
        fs.find_tool, fs.ffmpeg_version, fs.subprocess.run, fs.tool_version = self._saved

    def names(self, n=None):
        return [f"frame_{i:06d}.png" for i in range(1, (n or self.N) + 1)]

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = fs.main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def write_clips(self, rows, name="clips.csv", header=("filename", "clip", "seconds")):
        path = self.tmp / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def selection(self, clips=None, kept=None, **overrides):
        """A selection as the page would export it: kept computed by the rules unless given."""
        entries, _ = fs.inventory_images(self.source)
        if clips is None:
            clips = [{"name": "src-capture", "frames": self.names(), "every": 1, "offset": 0, "exclude": []}]
        if kept is None:
            kept_set = set()
            for c in clips:
                flags = fs.resolve_clip(c["frames"], c["every"], c["offset"], c["exclude"])
                kept_set |= {n for n, k in zip(c["frames"], flags) if k}
            kept = [e["filename"] for e in entries if e["filename"] in kept_set]
        data = {"format": fs.FORMAT,
                "source": {"capture": str(self.source), "inventory_sha256": fs.inventory_sha256(entries)},
                "clips": clips, "kept": kept}
        data.update(overrides)
        return data

    def write_selection(self, data, name="selection.json"):
        path = self.tmp / name
        path.write_text(json.dumps(data))
        return path

    def build(self, data, *extra):
        return self.run_main("build", "--selection", self.write_selection(data), "--capture", self.dest, *extra)


class TestInventory(Base):
    def test_top_level_images_sorted_and_other_entries_reported(self):
        (self.source / "images" / "cam1").mkdir()
        (self.source / "images" / "notes.txt").write_text("x")
        (self.source / "images" / "a.JPG").write_bytes(b"jpeg")
        entries, notes = fs.inventory_images(self.source)
        self.assertEqual([e["filename"] for e in entries], ["a.JPG"] + self.names())
        self.assertTrue(any("subdirectories" in n for n in notes))
        self.assertTrue(any("non-image" in n for n in notes))

    def test_refuses_no_images_and_a_broken_link(self):
        empty = self.tmp / "empty"
        (empty / "images").mkdir(parents=True)
        with self.assertRaises(fs.ToolError):
            fs.inventory_images(empty)
        with self.assertRaises(fs.ToolError):
            fs.inventory_images(self.tmp / "nowhere")
        (self.source / "images" / "zz.png").symlink_to("missing.png")
        with self.assertRaisesRegex(fs.ToolError, "does not resolve"):
            fs.inventory_images(self.source)

    def test_inventory_hash_tracks_names_sizes_and_mtimes(self):
        entries, _ = fs.inventory_images(self.source)
        before = fs.inventory_sha256(entries)
        self.assertEqual(before, fs.inventory_sha256(fs.inventory_images(self.source)[0]))
        target = self.source / "images" / "frame_000003.png"
        st = target.stat()
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        self.assertNotEqual(before, fs.inventory_sha256(fs.inventory_images(self.source)[0]))

    def test_clip_assignment_from_a_clips_csv(self):
        rows = [(n, "a" if i < 5 else "b", str(i * 0.5)) for i, n in enumerate(self.names())]
        inv = fs.build_inventory(self.source, self.write_clips(rows))
        self.assertEqual([(c["name"], len(c["frames"])) for c in inv["clips"]], [("a", 5), ("b", 7)])
        row = next(r for r in inv["rows"] if r["filename"] == "frame_000007.png")
        self.assertEqual((row["clip"], row["clip_index"], row["seconds"], row["width"]), ("b", 1, 3.0, 8))
        self.assertEqual(inv["clip_assignment"]["source"], "clips_csv")

    def test_clips_csv_refusals(self):
        names = self.names()
        cases = {
            "missing": [(n, "a", "") for n in names[:-1]],
            "duplicate": [(n, "a", "") for n in names] + [(names[0], "b", "")],
            "unknown": [(n, "a", "") for n in names] + [("other.png", "a", "")],
            "empty clip": [(n, "" if n == names[2] else "a", "") for n in names],
            "bad seconds": [(n, "a", "soon" if n == names[1] else "") for n in names],
        }
        for label, rows in cases.items():
            with self.assertRaises(fs.ToolError, msg=label):
                fs.build_inventory(self.source, self.write_clips(rows, name=f"{label}.csv"))
        with self.assertRaisesRegex(fs.ToolError, "columns"):
            fs.build_inventory(self.source, self.write_clips([(n, "a") for n in names], header=("file", "clip")))

    def test_one_clip_from_provenance_frames_csv_with_times(self):
        (self.source / "provenance").mkdir()
        with (self.source / "provenance" / "frames.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "output_grid_seconds", "source_seconds_approx", "width", "height"])
            for i, n in enumerate(self.names()):
                writer.writerow([n, i * 0.5, 10 + i * 0.5, 8, 6])
        inv = fs.build_inventory(self.source, None)
        self.assertEqual([c["name"] for c in inv["clips"]], ["src-capture"])
        self.assertEqual(inv["rows"][2]["seconds"], 11.0)
        self.assertEqual(inv["clip_assignment"]["time_column"], "source_seconds_approx")

    def test_one_clip_without_times_otherwise(self):
        inv = fs.build_inventory(self.source, None)
        self.assertEqual(inv["clip_assignment"]["source"], "single_clip_no_times")
        self.assertIsNone(inv["rows"][0]["seconds"])


class TestRules(unittest.TestCase):
    F = [f"f{i:02d}.png" for i in range(10)]

    def kept(self, every, offset, exclude):
        return [n for n, k in zip(self.F, fs.resolve_clip(self.F, every, offset, exclude)) if k]

    def test_sampling_phase_is_set_before_exclusions(self):
        self.assertEqual(self.kept(2, 0, []), ["f00.png", "f02.png", "f04.png", "f06.png", "f08.png"])
        # Excluding the first three frames must not re-anchor the phase on f03.
        self.assertEqual(self.kept(2, 0, [["f00.png", "f02.png"]]), ["f04.png", "f06.png", "f08.png"])
        self.assertEqual(self.kept(3, 1, [["f00.png", "f02.png"]]), ["f04.png", "f07.png"])

    def test_several_ranges_including_one_in_the_middle(self):
        exclude = [["f00.png", "f01.png"], ["f04.png", "f05.png"], ["f09.png", "f09.png"]]
        self.assertEqual(self.kept(1, 0, exclude), ["f02.png", "f03.png", "f06.png", "f07.png", "f08.png"])
        flags = fs.resolve_clip(self.F, 1, 0, exclude)
        ranges = fs.kept_ranges(self.F, flags, exclude)
        self.assertEqual([(r["start"], r["end"], r["kept"]) for r in ranges],
                         [("f02.png", "f03.png", 2), ("f06.png", "f08.png", 3)])

    def test_overlapping_ranges_union(self):
        self.assertEqual(self.kept(1, 0, [["f01.png", "f05.png"], ["f03.png", "f07.png"]]),
                         ["f00.png", "f08.png", "f09.png"])

    def test_filename_check(self):
        for bad in ("../x.png", "a/b.png", "a\\b.png", "..", "", "x\n.png", 5):
            with self.assertRaises(fs.ToolError, msg=repr(bad)):
                fs.check_filename(bad)
        self.assertEqual(fs.check_filename("frame_000001.png"), "frame_000001.png")


class TestPreview(Base):
    def preview(self, *extra):
        return self.run_main("preview", "--capture", self.source, "--out", self.out, *extra)

    def page_data(self):
        page = (self.out / "index.html").read_text(encoding="utf-8")
        match = re.search(r"const DATA = (\{.*?\});\n", page)
        self.assertIsNotNone(match)
        return page, json.loads(match.group(1))

    def test_writes_inventory_thumbnails_and_page_only_in_the_new_folder(self):
        before = tree_state(self.source)
        code, out, err = self.preview()
        self.assertEqual(code, 0, err)
        for name in ("index.html", "inventory.csv", "inventory.sha256", "preview.json", "ffmpeg-thumbs.log"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertEqual(len(list((self.out / "thumbs").glob("*.jpg"))), self.N)
        self.assertEqual(len(list((self.out / "large").glob("*.jpg"))), self.N)
        self.assertEqual(len(self.fake.commands), self.N)
        self.assertEqual(tree_state(self.source), before)
        with (self.out / "inventory.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), fs.INVENTORY_COLUMNS)
        self.assertEqual([r["filename"] for r in rows], self.names())
        self.assertEqual((rows[3]["index"], rows[3]["clip_index"], rows[3]["width"]), ("3", "3", "8"))
        digest = hashlib.sha256((self.out / "inventory.csv").read_bytes()).hexdigest()
        self.assertEqual((self.out / "inventory.sha256").read_text().split()[0], digest)

    def test_page_embeds_the_data_and_carries_no_placeholder(self):
        self.preview("--title", "My title")
        page, data = self.page_data()
        self.assertNotIn("__DATA__", page)
        self.assertEqual(data["format"], fs.FORMAT)
        self.assertEqual(data["title"], "My title")
        entries, _ = fs.inventory_images(self.source)
        self.assertEqual(data["inventory_sha256"], fs.inventory_sha256(entries))
        self.assertEqual([f["f"] for f in data["clips"][0]["frames"]], self.names())
        self.assertEqual(data["clips"][0]["frames"][0]["t"], "thumbs/000000.jpg")
        self.assertIn("Download selection JSON", page)
        self.assertNotIn("http://", page.replace("http://www.w3.org", ""))
        self.assertNotIn("https://", page)
        self.assertIn("--dry-run", data["build_command"])

    def test_page_data_cannot_break_out_of_the_script(self):
        self.out.mkdir(parents=True)
        for hostile in ("x</script><b>y", "a<!--<script>b"):
            page = fs.write_index(self.out, {"title": hostile}).read_text(encoding="utf-8")
            self.assertEqual(page.count("</script>"), 1, hostile)
            self.assertNotIn(hostile, page)

    def test_refuses_an_existing_out_and_out_inside_the_capture(self):
        self.out.mkdir(parents=True)
        code, _, err = self.preview()
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        code, _, err = self.run_main("preview", "--capture", self.source, "--out", self.source / "review")
        self.assertEqual(code, 2)
        self.assertFalse((self.source / "review").exists())

    def test_a_refused_clips_csv_creates_nothing(self):
        code, _, err = self.preview("--clips", self.write_clips([(self.names()[0], "a", "")]))
        self.assertEqual(code, 2)
        self.assertIn("exactly once", err)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.fake.commands, [])

    def test_ffmpeg_failure_is_reported(self):
        self.fake.returncode = 1
        code, _, err = self.preview()
        self.assertEqual(code, 2)
        self.assertIn("ffmpeg failed", err)

    def test_annotations_are_tolerant_labeled_and_recorded(self):
        checks = {"label": "old run", "registered": 10, "expected": 12,
                  "missing": ["frame_000002.png", "not_here.png"],
                  "largest_step_from": "frame_000005.png", "largest_step_to": "frame_000006.png",
                  "largest_step_ratio_to_median": 20.0, "camera_path_warnings": ["big jump"],
                  "low_point_frames": [{"frame": "frame_000009.png", "points": 3}],
                  "something_new": 1}
        path = self.tmp / "camera-checks.json"
        path.write_text(json.dumps(checks))
        code, out, err = self.preview("--annotations", path)
        self.assertEqual(code, 0, err)
        _, data = self.page_data()
        ann = data["annotations"]
        self.assertEqual(ann["path"], str(path.resolve()))
        self.assertEqual(ann["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(ann["tags"]["frame_000002.png"], ["not registered"])
        self.assertIn("largest camera step arrives here (20.0x median)", ann["tags"]["frame_000006.png"])
        self.assertEqual(ann["tags"]["frame_000009.png"], ["sees only 3 points"])
        self.assertEqual(ann["names_not_in_inventory"], ["not_here.png"])
        self.assertIn("something_new", ann["fields_ignored"])
        self.assertIn("flagged_steps", ann["fields_absent"])
        self.assertIn("names not in this inventory", out)
        record = json.loads((self.out / "preview.json").read_text())
        self.assertEqual(record["annotations"]["sha256"], ann["sha256"])

    def test_annotations_that_are_not_a_json_object_are_refused(self):
        path = self.tmp / "bad.json"
        path.write_text("[1, 2]")
        code, _, err = self.preview("--annotations", path)
        self.assertEqual(code, 2)
        self.assertFalse(self.out.exists())


class TestBuild(Base):
    def test_dry_run_validates_and_writes_nothing(self):
        clips = [{"name": "src-capture", "frames": self.names(), "every": 2, "offset": 1,
                  "exclude": [["frame_000001.png", "frame_000003.png"]]}]
        code, out, err = self.build(self.selection(clips), "--dry-run")
        self.assertEqual(code, 0, err)
        self.assertIn("dry run: nothing written", out)
        self.assertIn("keep range frame_000004.png .. frame_000012.png", out)
        self.assertFalse(self.dest.exists())

    def test_dry_run_estimates_bytes_for_materialize(self):
        code, out, _ = self.build(self.selection(), "--dry-run", "--materialize")
        self.assertEqual(code, 0)
        total = sum(p.stat().st_size for p in (self.source / "images").iterdir())
        self.assertIn(f"about {total} bytes", out)

    def test_link_mode_makes_relative_symlinks_and_leaves_the_source_untouched(self):
        before = tree_state(self.source)
        clips = [{"name": "src-capture", "frames": self.names(), "every": 1, "offset": 0,
                  "exclude": [["frame_000001.png", "frame_000002.png"], ["frame_000006.png", "frame_000007.png"]]}]
        data = self.selection(clips)
        code, out, err = self.build(data)
        self.assertEqual(code, 0, err)
        images = sorted((self.dest / "images").iterdir())
        self.assertEqual([p.name for p in images], data["kept"])
        self.assertEqual(len(images), 8)
        for link in images:
            self.assertTrue(link.is_symlink(), link.name)
            self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(link.resolve(), (self.source / "images" / link.name).resolve())
        self.assertEqual(tree_state(self.source), before)
        moved = self.tmp / "moved" / "datasets"
        moved.parent.mkdir()
        shutil.move(str(self.tmp / "datasets"), str(moved))
        self.assertTrue((moved / "new-capture" / "images" / "frame_000003.png").is_file())

    def test_provenance_records_the_selection_verbatim_and_the_build(self):
        data = self.selection()
        path = self.write_selection(data)
        code, _, err = self.run_main("build", "--selection", path, "--capture", self.dest)
        self.assertEqual(code, 0, err)
        prov = self.dest / "provenance"
        self.assertEqual((prov / "selection.json").read_bytes(), path.read_bytes())
        record = json.loads((prov / "frame-selection.json").read_text())
        self.assertEqual(record["mode"], "link")
        self.assertEqual(record["inventory_sha256"], data["source"]["inventory_sha256"])
        self.assertEqual((record["source_frames"], record["kept_frames"]), (self.N, self.N))
        self.assertEqual(record["version"], "test-version")
        self.assertEqual(record["selection_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(record["masks_copied"])
        self.assertEqual(record["source_capture"], str(self.source))

    def test_materialize_copies_real_files(self):
        before = tree_state(self.source)
        code, _, err = self.build(self.selection(), "--materialize")
        self.assertEqual(code, 0, err)
        for path in (self.dest / "images").iterdir():
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.read_bytes(), (self.source / "images" / path.name).read_bytes())
        self.assertEqual(json.loads((self.dest / "provenance" / "frame-selection.json").read_text())["mode"], "copy")
        self.assertEqual(tree_state(self.source), before)

    def test_frames_csv_has_kept_rows_and_camera_review_reads_it(self):
        names = self.names()
        clips = [{"name": "a", "frames": names[:6], "seconds": [i * 0.5 for i in range(6)], "every": 2, "offset": 0,
                  "exclude": [["frame_000003.png", "frame_000003.png"]]},
                 {"name": "b", "frames": names[6:], "seconds": [100 + i * 0.5 for i in range(6)], "every": 1,
                  "offset": 0, "exclude": []}]
        code, _, err = self.build(self.selection(clips))
        self.assertEqual(code, 0, err)
        with (self.dest / "provenance" / "frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()),
                         ["filename", "clip", "clip_base_index", "source_filename", "source_seconds_approx"])
        self.assertEqual([r["filename"] for r in rows[:2]], ["frame_000001.png", "frame_000005.png"])
        self.assertEqual((rows[1]["clip"], rows[1]["clip_base_index"], rows[1]["source_seconds_approx"]),
                         ("a", "4", "2.0"))
        self.assertEqual((rows[2]["clip"], rows[2]["source_seconds_approx"]), ("b", "100.0"))
        times, axis = camera_review.frame_times(self.dest)
        self.assertEqual(axis, "clip seconds (approx)")
        self.assertEqual(len(times), 8)

    def test_frames_csv_omits_times_when_any_is_unknown(self):
        code, _, _ = self.build(self.selection())
        self.assertEqual(code, 0)
        with (self.dest / "provenance" / "frames.csv").open() as handle:
            self.assertNotIn("source_seconds_approx", next(csv.reader(handle)))
        self.assertEqual(camera_review.frame_times(self.dest)[1], "frame order")

    def assertRefused(self, data, fragment, *extra):
        code, _, err = self.build(data, *extra)
        self.assertEqual(code, 2, err)
        self.assertIn(fragment, err)
        self.assertFalse(self.dest.exists())

    def test_refuses_when_the_source_changed(self):
        data = self.selection()
        (self.source / "images" / "frame_000013.png").write_bytes(png_bytes())
        self.assertRefused(data, "changed since the preview")

    def test_refuses_a_tampered_kept_list(self):
        data = self.selection()
        self.assertRefused(dict(data, kept=data["kept"][:-1]), "does not follow from the rules")
        self.assertRefused(dict(data, kept=data["kept"] + [data["kept"][0]]), "duplicates")
        self.assertRefused(dict(data, kept=list(reversed(data["kept"]))), "order")

    def test_refuses_bad_ranges(self):
        names = self.names()
        for exclude, fragment in (
            ([["frame_000005.png", "frame_000002.png"]], "starts after it ends"),
            ([["frame_000001.png", "frame_000099.png"]], "not a frame of that clip"),
            ([["frame_000001.png"]], "not a [start, end] pair"),
        ):
            clips = [{"name": "c", "frames": names, "every": 1, "offset": 0, "exclude": exclude}]
            self.assertRefused(self.selection(clips, kept=names), fragment)

    def test_refuses_a_range_naming_a_frame_of_another_clip(self):
        names = self.names()
        clips = [{"name": "a", "frames": names[:6], "every": 1, "offset": 0,
                  "exclude": [["frame_000001.png", "frame_000008.png"]]},
                 {"name": "b", "frames": names[6:], "every": 1, "offset": 0, "exclude": []}]
        self.assertRefused(self.selection(clips, kept=names), "not a frame of that clip")

    def test_refuses_traversal_filenames(self):
        names = self.names()
        clips = [{"name": "c", "frames": names, "every": 1, "offset": 0, "exclude": []}]
        self.assertRefused(self.selection(clips, kept=names[:-1] + ["../../etc/passwd"]), "path separator")
        clips[0]["frames"] = names[:-1] + ["../frame_000012.png"]
        self.assertRefused(self.selection(clips, kept=names), "path separator")

    def test_refuses_bad_sampling_rules(self):
        names = self.names()
        for every, offset, fragment in ((0, 0, "1 or more"), (2, 2, "offset must be"), (2, -1, "offset must be"),
                                        (1.5, 0, "whole number"), (True, 0, "whole number")):
            clips = [{"name": "c", "frames": names, "every": every, "offset": offset, "exclude": []}]
            self.assertRefused(self.selection(clips, kept=names), fragment)

    def test_refuses_clips_that_do_not_cover_the_source_exactly(self):
        names = self.names()
        self.assertRefused(self.selection([{"name": "c", "frames": names[:-1], "every": 1, "offset": 0, "exclude": []}],
                                          kept=names[:-1]), "do not cover")
        dup = [{"name": "a", "frames": names, "every": 1, "offset": 0, "exclude": []},
               {"name": "b", "frames": names[:1], "every": 1, "offset": 0, "exclude": []}]
        self.assertRefused(self.selection(dup, kept=names), "listed twice")
        unsorted = [{"name": "c", "frames": list(reversed(names)), "every": 1, "offset": 0, "exclude": []}]
        self.assertRefused(self.selection(unsorted, kept=names), "filename order")

    def test_refuses_too_few_frames_wrong_format_and_existing_destination(self):
        names = self.names()
        clips = [{"name": "c", "frames": names, "every": 1, "offset": 0,
                  "exclude": [["frame_000001.png", "frame_000010.png"]]}]
        self.assertRefused(self.selection(clips), "need at least 3")
        self.assertRefused(self.selection(format="dt4ag-frame-selection/0"), "format")
        self.dest.mkdir(parents=True)
        (self.dest / "keep.txt").write_text("mine")
        code, _, err = self.build(self.selection())
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertEqual([p.name for p in self.dest.iterdir()], ["keep.txt"])

    def test_source_can_be_given_when_the_recorded_path_moved(self):
        data = self.selection()
        moved = self.tmp / "elsewhere"
        shutil.copytree(self.source, moved, copy_function=shutil.copy2)
        shutil.rmtree(self.source)
        self.assertRefused(data, "source capture not found")
        code, _, err = self.build(data, "--source", moved, "--dry-run")
        self.assertEqual(code, 0, err)

    def test_never_falls_back_to_copying_when_links_are_impossible(self):
        real = Path.symlink_to

        def refuse(self_path, target, *a, **k):
            raise OSError("symlinks not permitted")
        Path.symlink_to = refuse
        self.addCleanup(setattr, Path, "symlink_to", real)
        self.assertRefused(self.selection(), "does not silently copy")

    def test_masks_are_reported_not_copied(self):
        (self.source / "masks").mkdir()
        code, out, _ = self.build(self.selection())
        self.assertEqual(code, 0)
        self.assertIn("masks are NOT copied", out)
        self.assertFalse((self.dest / "masks").exists())
        record = json.loads((self.dest / "provenance" / "frame-selection.json").read_text())
        self.assertEqual(record["source_mask_dirs"], ["masks"])

    def test_preview_to_build_round_trip_with_clips(self):
        rows = [(n, "a" if i < 7 else "b", "") for i, n in enumerate(self.names())]
        code, _, err = self.run_main("preview", "--capture", self.source, "--out", self.out,
                                     "--clips", self.write_clips(rows))
        self.assertEqual(code, 0, err)
        page = (self.out / "index.html").read_text(encoding="utf-8")
        data = json.loads(re.search(r"const DATA = (\{.*?\});\n", page).group(1))
        clips = [{"name": c["name"], "frames": [f["f"] for f in c["frames"]], "seconds": [f["s"] for f in c["frames"]],
                  "every": 1, "offset": 0, "exclude": []} for c in data["clips"]]
        clips[0]["exclude"] = [["frame_000003.png", "frame_000004.png"]]
        selection = self.selection(clips)
        self.assertEqual(selection["source"]["inventory_sha256"], data["inventory_sha256"])
        code, _, err = self.build(selection)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(list((self.dest / "images").iterdir())), self.N - 2)


@unittest.skipUnless(shutil.which("node"), "node is not on PATH")
class TestPageRulesMatchPython(unittest.TestCase):
    """The page's resolveClip() must keep exactly what resolve_clip() keeps."""

    def test_page_rules_equal_python_rules(self):
        template = (PIPELINE_DIR / "scripts" / "frame-selection" / "frame-selection.template.html").read_text(encoding="utf-8")
        body = re.search(r"(function resolveClip\(clip\) \{.*?\n\})", template, re.S).group(1)
        frames = [f"f{i:02d}" for i in range(15)]
        cases = [(1, 0, []), (2, 0, [["f00", "f02"]]), (3, 2, [["f04", "f06"], ["f10", "f10"]]),
                 (4, 1, [["f00", "f14"]]), (2, 1, [["f03", "f08"], ["f05", "f11"]])]
        script = ("let state = {};\n" + body + "\nconst out = [];\n"
                  + f"const frames = {json.dumps(frames)}.map((f) => ({{ f }}));\n"
                  + f"for (const [every, offset, exclude] of {json.dumps(cases)}) {{\n"
                  + "  state = { c: { every, offset, exclude } };\n"
                  + "  const r = resolveClip({ name: 'c', frames });\n"
                  + "  out.push({ kept: r.kept, ranges: r.ranges.map((g) => [g.start.f, g.end.f, g.kept]) });\n"
                  + "}\nconsole.log(JSON.stringify(out));\n")
        result = subprocess.run(["node", "-e", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        page = json.loads(result.stdout)
        for (every, offset, exclude), got in zip(cases, page):
            flags = fs.resolve_clip(frames, every, offset, exclude)
            self.assertEqual(got["kept"], flags, (every, offset, exclude))
            self.assertEqual(got["ranges"], [[r["start"], r["end"], r["kept"]]
                                             for r in fs.kept_ranges(frames, flags, exclude)])


if __name__ == "__main__":
    unittest.main()
