"""Tests for ``pipeline/copy_config.py``.

Stdlib ``unittest`` only, matching the rest of the suite, and hermetic: every
copy goes to a temporary directory and nothing touches ``pipeline/configs``.
"""

from __future__ import annotations

import configparser
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import copy_config  # noqa: E402

EXAMPLE = PIPELINE_DIR / "configs" / "example.ini"

SMALL = """\
# keep this comment
[paths]
data_root = /path/to/data   # inline note

[train]
Max_Num_Iterations = 30000
vis = viewer
"""


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = copy_config.main(list(argv))
    return code, out.getvalue().strip(), err.getvalue()


def read(path: str | Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path, encoding="utf-8")
    return parser


class CopyConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.dest = self.tmp / "out"
        self.small = self.tmp / "small.ini"
        self.small.write_text(SMALL, encoding="utf-8")

    def test_default_source_is_example_and_copy_is_identical(self):
        code, printed, err = run(
            "--destination-dir", str(self.dest), "--name", "a.ini"
        )
        self.assertEqual(code, 0, err)
        copy = self.dest / "a.ini"
        self.assertEqual(Path(printed), copy.resolve())
        self.assertEqual(copy.read_bytes(), EXAMPLE.read_bytes())

    def test_example_ini_accepts_overrides_and_still_parses(self):
        code, printed, err = run(
            "--destination-dir", str(self.dest), "--name", "b.ini",
            "--set", "train.max_num_iterations=10000",
            "--set", "dataset.images_subpath=scene/images",
        )
        self.assertEqual(code, 0, err)
        parsed = read(printed)
        self.assertEqual(parsed["train"]["max_num_iterations"], "10000")
        self.assertEqual(parsed["dataset"]["images_subpath"], "scene/images")
        self.assertEqual(parsed["train"]["method"], "splatfacto")

    def test_overrides_keep_comments_and_other_lines(self):
        code, printed, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "c.ini", "--set", "train.max_num_iterations=500",
        )
        self.assertEqual(code, 0, err)
        text = Path(printed).read_text(encoding="utf-8")
        self.assertIn("# keep this comment", text)
        self.assertIn("data_root = /path/to/data   # inline note", text)
        self.assertIn("Max_Num_Iterations = 500", text)
        self.assertIn("vis = viewer", text)

    def test_source_is_not_modified(self):
        before = self.small.read_bytes()
        run("--source", str(self.small), "--destination-dir", str(self.dest),
            "--set", "train.vis=off")
        self.assertEqual(self.small.read_bytes(), before)

    def test_named_copy_refuses_to_overwrite(self):
        args = ("--source", str(self.small), "--destination-dir", str(self.dest),
                "--name", "d.ini")
        self.assertEqual(run(*args)[0], 0)
        target = self.dest / "d.ini"
        target.write_text("precious", encoding="utf-8")
        code, _, err = run(*args)
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertEqual(target.read_text(encoding="utf-8"), "precious")

    def test_prefix_picks_unused_numbered_names(self):
        args = ("--source", str(self.small), "--destination-dir", str(self.dest),
                "--name-prefix", "run")
        names = []
        for _ in range(3):
            code, printed, err = run(*args)
            self.assertEqual(code, 0, err)
            names.append(Path(printed).name)
        self.assertEqual(names, ["run.ini", "run-2.ini", "run-3.ini"])

    def test_default_name_derives_from_source(self):
        code, printed, _ = run("--source", str(self.small),
                               "--destination-dir", str(self.dest))
        self.assertEqual(code, 0)
        self.assertEqual(Path(printed).name, "small-copy.ini")

    def test_bad_set_forms_are_rejected_and_write_nothing(self):
        for bad in ("noequals", "nodot=1", ".key=1", "section.=1",
                    "paths.data_root=a\nb"):
            with self.subTest(bad=bad):
                code, _, err = run(
                    "--source", str(self.small), "--destination-dir",
                    str(self.dest), "--name", "e.ini", "--set", bad)
                self.assertEqual(code, 2)
                self.assertIn("CONFIG COPY ERROR", err)
                self.assertFalse((self.dest / "e.ini").exists())

    def test_unknown_key_is_rejected_and_write_nothing(self):
        code, _, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "f.ini", "--set", "train.no_such_key=1")
        self.assertEqual(code, 2)
        self.assertIn("not a key the loader knows", err)
        self.assertFalse((self.dest / "f.ini").exists())

    def test_known_key_missing_from_source_is_added_under_its_section(self):
        code, printed, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "g.ini", "--set", "paths.datasets_dirname=ds",
            "--set", "train.method=splatfacto")
        self.assertEqual(code, 0, err)
        text = Path(printed).read_text(encoding="utf-8")
        parsed = read(printed)
        self.assertEqual(parsed["paths"]["datasets_dirname"], "ds")
        self.assertEqual(parsed["train"]["method"], "splatfacto")
        lines = text.splitlines()
        self.assertEqual(lines.index("datasets_dirname = ds"),
                         lines.index("data_root = /path/to/data   # inline note") + 1)
        self.assertEqual(parsed["train"]["vis"], "viewer")

    def test_known_key_in_absent_section_creates_the_section(self):
        code, printed, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "h.ini", "--set", "run.id_prefix=tomato")
        self.assertEqual(code, 0, err)
        self.assertEqual(read(printed)["run"]["id_prefix"], "tomato")
        self.assertEqual(read(printed)["paths"]["data_root"], "/path/to/data")

    def test_added_key_works_when_source_lacks_trailing_newline(self):
        bare = self.tmp / "bare.ini"
        bare.write_text("[paths]\ndata_root = x", encoding="utf-8")
        code, printed, err = run(
            "--source", str(bare), "--destination-dir", str(self.dest),
            "--name", "i.ini", "--set", "paths.outputs_dirname=o")
        self.assertEqual(code, 0, err)
        self.assertEqual(read(printed)["paths"]["outputs_dirname"], "o")

    def test_unknown_key_in_unknown_section_is_still_refused(self):
        code, _, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "j.ini", "--set", "bogus.thing=1")
        self.assertEqual(code, 2)
        self.assertFalse((self.dest / "j.ini").exists())

    def test_value_that_would_become_an_inline_comment_is_refused(self):
        for value in ("a #b", "a ;b", "a\t#b", "#b", ";b"):
            with self.subTest(value=value):
                code, _, err = run(
                    "--source", str(self.small), "--destination-dir",
                    str(self.dest), "--name", "k.ini",
                    "--set", f"train.vis={value}")
                self.assertEqual(code, 2)
                self.assertIn("inline comment", err)
                self.assertFalse((self.dest / "k.ini").exists())

    def test_section_names_are_case_sensitive_like_the_loader(self):
        code, _, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "m.ini", "--set", "TRAIN.vis=off")
        self.assertEqual(code, 2)
        self.assertFalse((self.dest / "m.ini").exists())

    def test_hash_inside_a_word_is_allowed(self):
        code, printed, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--name", "l.ini", "--set", "train.vis=a#b")
        self.assertEqual(code, 0, err)
        self.assertEqual(read(printed)["train"]["vis"], "a#b")

    def test_duplicate_set_target_is_rejected(self):
        code, _, err = run(
            "--source", str(self.small), "--destination-dir", str(self.dest),
            "--set", "train.vis=a", "--set", "train.VIS=b")
        self.assertEqual(code, 2)
        self.assertIn("duplicate", err)

    def test_bad_names_are_rejected(self):
        for flag, value in (("--name", "../x.ini"), ("--name", "x.txt"),
                            ("--name-prefix", "a/b"), ("--name-prefix", "x.ini")):
            with self.subTest(flag=flag, value=value):
                code, _, err = run("--source", str(self.small),
                                   "--destination-dir", str(self.dest),
                                   flag, value)
                self.assertEqual(code, 2)
                self.assertIn("CONFIG COPY ERROR", err)

    def test_missing_or_non_ini_source_is_rejected(self):
        code, _, _ = run("--source", str(self.tmp / "missing.ini"),
                         "--destination-dir", str(self.dest))
        self.assertEqual(code, 2)
        other = self.tmp / "notes.txt"
        other.write_text("x", encoding="utf-8")
        code, _, err = run("--source", str(other),
                           "--destination-dir", str(self.dest))
        self.assertEqual(code, 2)
        self.assertIn(".ini", err)

    def test_same_file_as_source_is_refused(self):
        code, _, err = run("--source", str(self.small),
                           "--destination-dir", str(self.tmp),
                           "--name", "small.ini")
        self.assertEqual(code, 2)
        self.assertIn("same file", err)
        self.assertEqual(self.small.read_text(encoding="utf-8"), SMALL)

    def test_script_contains_no_private_paths(self):
        text = Path(copy_config.__file__).read_text(encoding="utf-8")
        for needle in ("/home/alex", "PX13", "welzo"):
            self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
