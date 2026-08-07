"""Tests for ``pipeline/scripts/rgb-mask/rgb-mask-batch.py``.

Stdlib ``unittest``, matching ``test_dt4ag_config.py`` and
``test_run_pipeline.py``. Every test is hermetic: each builds its own image and
mask tree in a ``tempfile.TemporaryDirectory`` and throws it away. No GPU, no
COLMAP, no nerfstudio, no network, no data drive, no subprocess.

The one non-stdlib dependency is Pillow, which the script under test needs in
order to exist at all. If Pillow is not importable, this module skips rather
than failing, so the suite still passes in the zero-dependency environment the
rest of the tests target.

What is covered is the defect this script was rewritten to fix: it used to
accept a sibling ``images/`` + ``masks/`` layout in its header comment, require
a nested ``images/masks/`` layout in its code, and report the resulting total
no-op with exit code 0. So the tests pin down layout detection, the failure
counts, and the exit codes.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PIPELINE_DIR / "scripts" / "rgb-mask" / "rgb-mask-batch.py"

# Kept in step with test_dt4ag_config.FORBIDDEN_TERMS. The pipeline
# reconstructs arbitrary scenes; no committed file may imply a subject.
FORBIDDEN_TERMS = [
    "plant",
    "tomato",
    "tanashi",
    "soybean",
    "houseplant",
    "arabidopsis",
]


def _load_script():
    """Import the hyphenated script file as a module, or return None."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return None
    spec = importlib.util.spec_from_file_location("rgb_mask_batch", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rgb_mask_batch"] = module
    spec.loader.exec_module(module)
    return module


rmb = _load_script()

requires_pillow = unittest.skipIf(rmb is None, "Pillow is not installed")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class TreeFixture:
    """A temporary image/mask tree.

    ``layout`` is ``"sibling"`` (``root/images``, ``root/masks``) or
    ``"nested"`` (``root/images``, ``root/images/masks``).
    """

    def __init__(self, layout: str = "sibling"):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.masks = self.images / "masks" if layout == "nested" else self.root / "masks"
        self.masks.mkdir(parents=True)
        self.output = self.root / "masked-images"

    def cleanup(self):
        self._tmp.cleanup()

    # -- content ---------------------------------------------------------
    def add_image(self, relative: str, size=(8, 6)):
        from PIL import Image

        path = self.images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (10, 120, 200)).save(path)
        return path

    def add_mask(self, relative: str, size=(8, 6), soft=False, root=None):
        from PIL import Image

        path = (root if root is not None else self.masks) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", size, 0)
        for x in range(size[0]):
            for y in range(size[1]):
                if x < size[0] // 2:
                    mask.putpixel((x, y), 255)
                elif soft and x == size[0] // 2:
                    mask.putpixel((x, y), 100)
        mask.save(path)
        return path

    def add_pair(self, stem: str, size=(8, 6), soft=False):
        self.add_image(f"{stem}.jpg", size=size)
        self.add_mask(f"{stem}.png", size=size, soft=soft)

    def add_file(self, relative: str, text: str = "not an image"):
        path = self.images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # -- running ---------------------------------------------------------
    def run(self, *extra):
        """Run ``main`` with this fixture's dirs; return ``(code, stdout)``."""
        argv = [
            "--images", str(self.images),
            "--output", str(self.output),
            *extra,
        ]
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
            code = rmb.main(argv)
        return code, buffer.getvalue()

    def alpha_values(self, relative: str):
        from PIL import Image

        image = Image.open(self.output / relative)
        return sorted(set(image.split()[-1].getdata()))


class FixtureCase(unittest.TestCase):
    """Base case that builds a fixture and tears it down."""

    layout = "sibling"

    def setUp(self):
        if rmb is None:
            self.skipTest("Pillow is not installed")
        self.fixture = TreeFixture(self.layout)
        self.addCleanup(self.fixture.cleanup)


# --------------------------------------------------------------------------
# layout detection
# --------------------------------------------------------------------------

@requires_pillow
class TestLayoutDetection(unittest.TestCase):
    """The heart of the old defect: which mask directory is actually used."""

    def test_sibling_layout_detected(self):
        fixture = TreeFixture("sibling")
        self.addCleanup(fixture.cleanup)
        mask_root, layout = rmb.resolve_mask_root(fixture.images)
        self.assertEqual(layout, "sibling")
        self.assertEqual(mask_root, fixture.masks)

    def test_nested_layout_detected(self):
        fixture = TreeFixture("nested")
        self.addCleanup(fixture.cleanup)
        mask_root, layout = rmb.resolve_mask_root(fixture.images)
        self.assertEqual(layout, "nested")
        self.assertEqual(mask_root, fixture.masks)

    def test_no_mask_directory_raises_and_names_both_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            images.mkdir()
            with self.assertRaises(rmb.LayoutError) as ctx:
                rmb.resolve_mask_root(images)
            message = str(ctx.exception)
            self.assertIn(str(images / "masks"), message)
            self.assertIn(str(Path(tmp) / "masks"), message)

    def test_both_layouts_present_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            (images / "masks").mkdir(parents=True)
            (Path(tmp) / "masks").mkdir()
            with self.assertRaises(rmb.LayoutError) as ctx:
                rmb.resolve_mask_root(images)
            message = str(ctx.exception)
            self.assertIn("mbiguous", message)
            self.assertIn(str(images / "masks"), message)
            self.assertIn(str(Path(tmp) / "masks"), message)

    def test_explicit_masks_wins_over_detection(self):
        fixture = TreeFixture("sibling")
        self.addCleanup(fixture.cleanup)
        elsewhere = fixture.root / "other-masks"
        elsewhere.mkdir()
        mask_root, layout = rmb.resolve_mask_root(fixture.images, elsewhere)
        self.assertEqual(layout, "explicit")
        self.assertEqual(mask_root, elsewhere)

    def test_explicit_masks_directory_must_exist(self):
        fixture = TreeFixture("sibling")
        self.addCleanup(fixture.cleanup)
        with self.assertRaises(rmb.LayoutError):
            rmb.resolve_mask_root(fixture.images, fixture.root / "nope")


# --------------------------------------------------------------------------
# the happy paths, one per layout
# --------------------------------------------------------------------------

@requires_pillow
class TestSiblingLayoutRun(FixtureCase):
    """The layout the old code silently no-opped on."""

    layout = "sibling"

    def test_sibling_layout_writes_output_and_exits_zero(self):
        self.fixture.add_pair("cam-1/a")
        self.fixture.add_pair("cam-1/b")
        code, out = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertTrue((self.fixture.output / "cam-1" / "a.png").is_file())
        self.assertTrue((self.fixture.output / "cam-1" / "b.png").is_file())
        self.assertIn("processed         : 2", out)

    def test_output_mirrors_the_input_structure(self):
        self.fixture.add_pair("cam-1/sub/a")
        code, _ = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertTrue((self.fixture.output / "cam-1" / "sub" / "a.png").is_file())

    def test_masked_output_carries_the_mask_as_alpha(self):
        self.fixture.add_pair("a", size=(8, 6))
        code, _ = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertEqual(self.fixture.alpha_values("a.png"), [0, 255])


@requires_pillow
class TestNestedLayoutRun(FixtureCase):
    """The layout the old code required; it must keep working."""

    layout = "nested"

    def test_nested_layout_still_works(self):
        self.fixture.add_pair("cam-1/a")
        code, out = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertTrue((self.fixture.output / "cam-1" / "a.png").is_file())
        self.assertIn("processed         : 1", out)

    def test_masks_inside_the_image_root_are_not_treated_as_images(self):
        self.fixture.add_pair("a")
        code, out = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertIn("processed         : 1", out)
        self.assertIn("failed (total)    : 0", out)
        self.assertFalse((self.fixture.output / "masks").exists())

    def test_output_inside_the_image_root_is_not_reprocessed(self):
        self.fixture.add_pair("a")
        self.fixture.output = self.fixture.images / "masked-images"
        first_code, _ = self.fixture.run()
        self.assertEqual(first_code, 0)
        second_code, out = self.fixture.run()
        self.assertEqual(second_code, 0)
        self.assertIn("processed         : 1", out)
        self.assertIn("failed (total)    : 0", out)


# --------------------------------------------------------------------------
# failure counting and exit codes
# --------------------------------------------------------------------------

@requires_pillow
class TestFailuresAreLoud(FixtureCase):

    def test_missing_mask_is_counted_and_fatal(self):
        self.fixture.add_pair("a")
        self.fixture.add_image("b.jpg")
        code, out = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertIn("missing mask      : 1", out)
        self.assertIn("processed         : 1", out)
        self.assertIn("failed (total)    : 1", out)

    def test_size_mismatch_is_counted_and_fatal(self):
        self.fixture.add_pair("a")
        self.fixture.add_image("b.jpg", size=(8, 6))
        self.fixture.add_mask("b.png", size=(4, 3))
        code, out = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertIn("size mismatch     : 1", out)
        self.assertIn("failed (total)    : 1", out)
        self.assertFalse((self.fixture.output / "b.png").exists())

    def test_unsupported_extension_is_counted_and_fatal(self):
        self.fixture.add_pair("a")
        self.fixture.add_file("notes.txt")
        code, out = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertIn("unsupported ext   : 1", out)

    def test_hidden_files_are_ignored_entirely(self):
        self.fixture.add_pair("a")
        self.fixture.add_file(".DS_Store")
        code, out = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertIn("unsupported ext   : 0", out)

    def test_zero_images_processed_is_fatal(self):
        # A mask directory exists, but there is nothing to mask.
        code, out = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertIn("processed         : 0", out)

    def test_all_masks_missing_is_fatal_not_a_silent_no_op(self):
        self.fixture.add_image("a.jpg")
        self.fixture.add_image("b.jpg")
        code, out = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertIn("missing mask      : 2", out)
        self.assertIn("processed         : 0", out)

    def test_missing_images_directory_is_a_usage_error(self):
        missing = self.fixture.root / "nope"
        with redirect_stderr(io.StringIO()):
            code = rmb.main(["--images", str(missing), "--output", str(self.fixture.output)])
        self.assertEqual(code, 2)

    def test_undetectable_layout_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            images.mkdir()
            with redirect_stderr(io.StringIO()):
                code = rmb.main(["--images", str(images), "--output", str(Path(tmp) / "out")])
            self.assertEqual(code, 2)


@requires_pillow
class TestAllowPartial(FixtureCase):

    def test_allow_partial_downgrades_a_partial_failure(self):
        self.fixture.add_pair("a")
        self.fixture.add_image("b.jpg")
        code, out = self.fixture.run("--allow-partial")
        self.assertEqual(code, 0)
        self.assertIn("missing mask      : 1", out)
        self.assertIn("processed         : 1", out)

    def test_allow_partial_does_not_excuse_zero_processed(self):
        self.fixture.add_image("a.jpg")
        code, out = self.fixture.run("--allow-partial")
        self.assertEqual(code, 1)
        self.assertIn("processed         : 0", out)


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------

@requires_pillow
class TestMaskExtension(FixtureCase):

    def test_default_mask_extension_is_png(self):
        parser = rmb.build_parser()
        args = parser.parse_args(["--images", "x"])
        self.assertEqual(args.mask_ext, ".png")

    def test_alternative_mask_extension(self):
        self.fixture.add_image("a.jpg")
        self.fixture.add_mask("a.tif")
        code, out = self.fixture.run("--mask-ext", ".tif")
        self.assertEqual(code, 0)
        self.assertIn("processed         : 1", out)

    def test_mask_extension_without_a_leading_dot(self):
        self.fixture.add_image("a.jpg")
        self.fixture.add_mask("a.tif")
        code, _ = self.fixture.run("--mask-ext", "tif")
        self.assertEqual(code, 0)

    def test_normalize_ext(self):
        self.assertEqual(rmb.normalize_ext("PNG"), ".png")
        self.assertEqual(rmb.normalize_ext(".PNG"), ".png")
        self.assertEqual(rmb.normalize_ext(" .png "), ".png")
        self.assertEqual(rmb.normalize_ext(""), "")

    def test_empty_mask_extension_is_a_usage_error(self):
        self.fixture.add_pair("a")
        code, _ = self.fixture.run("--mask-ext", "")
        self.assertEqual(code, 2)


@requires_pillow
class TestBinarize(FixtureCase):

    def test_binarize_is_off_by_default_so_soft_edges_pass_through(self):
        self.fixture.add_pair("a", soft=True)
        code, _ = self.fixture.run()
        self.assertEqual(code, 0)
        self.assertIn(100, self.fixture.alpha_values("a.png"))

    def test_binarize_produces_a_hard_edged_alpha(self):
        self.fixture.add_pair("a", soft=True)
        code, _ = self.fixture.run("--binarize", "127")
        self.assertEqual(code, 0)
        self.assertEqual(self.fixture.alpha_values("a.png"), [0, 255])

    def test_binarize_threshold_is_respected(self):
        # A soft pixel of 100 stays opaque when the threshold is below it.
        self.fixture.add_pair("a", soft=True)
        code, _ = self.fixture.run("--binarize", "50")
        self.assertEqual(code, 0)
        self.assertEqual(self.fixture.alpha_values("a.png"), [0, 255])

    def test_binarize_rejects_out_of_range_thresholds(self):
        parser = rmb.build_parser()
        for bad in ("-1", "256", "half"):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--images", "x", "--binarize", bad])


@requires_pillow
class TestArguments(unittest.TestCase):

    def test_images_is_required(self):
        parser = rmb.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = rmb.build_parser()
        args = parser.parse_args(["--images", "x"])
        self.assertIsNone(args.masks)
        self.assertIsNone(args.binarize)
        self.assertFalse(args.allow_partial)
        self.assertEqual(args.output, Path("masked-images"))

    def test_help_documents_the_soft_edge_behaviour(self):
        help_text = rmb.build_parser().format_help().lower()
        self.assertIn("soft", help_text)
        self.assertIn("binarize", help_text)


# --------------------------------------------------------------------------
# committed-file hygiene
# --------------------------------------------------------------------------

class TestObjectAgnostic(unittest.TestCase):
    """The pipeline reconstructs arbitrary scenes; the script names no subject."""

    def test_script_names_no_subject(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, text)

    def test_readme_names_no_subject(self):
        readme = SCRIPT_PATH.parent / "README.md"
        text = readme.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
