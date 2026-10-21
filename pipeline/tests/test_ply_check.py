"""Tests for ``pipeline/scripts/ply-check/ply-check.py``.

Stdlib ``unittest``. The tool itself needs numpy, so the whole module is skipped
where numpy is missing. Every PLY here is a small synthetic file written by the
test: binary little-endian, with the property names a 3D Gaussian splat export
carries (positions, normals, spherical-harmonics terms, opacity, scales and a
rotation quaternion). The picture writer needs matplotlib and is covered only
when that is installed; the point selection behind it is always covered.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:
    import numpy as np
except ImportError:          # pragma: no cover - depends on the environment
    np = None

PIPELINE_DIR = Path(__file__).resolve().parent.parent

if np is not None:
    _SPEC = importlib.util.spec_from_file_location(
        "ply_check", PIPELINE_DIR / "scripts" / "ply-check" / "ply-check.py")
    pc = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(pc)

SPLAT_PROPERTIES = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


def header(count, names=SPLAT_PROPERTIES, fmt="binary_little_endian 1.0", extra=""):
    lines = ["ply", f"format {fmt}", "comment synthetic test file", extra.rstrip("\n"),
             f"element vertex {count}"]
    lines += [f"property float {name}" for name in names]
    lines.append("end_header")
    return ("\n".join(line for line in lines if line) + "\n").encode("ascii")


def splat_array(count, seed=0):
    """Plausible splat values: centers in a unit-ish cloud, opacity logits, log scales."""
    rng = np.random.default_rng(seed)
    data = np.zeros((count, len(SPLAT_PROPERTIES)), dtype="<f4")
    column = {name: i for i, name in enumerate(SPLAT_PROPERTIES)}
    data[:, column["x"]:column["z"] + 1] = rng.normal(0, 1, (count, 3))
    data[:, column["f_dc_0"]:column["f_dc_2"] + 1] = rng.normal(0, 1, (count, 3))
    data[:, column["opacity"]] = rng.normal(0, 3, count)
    data[:, column["scale_0"]:column["scale_2"] + 1] = rng.normal(-4, 0.5, (count, 3))
    data[:, column["rot_0"]] = 1.0
    return data, column


def write_ply(path, data, declared=None, **header_args):
    count = len(data) if declared is None else declared
    path.write_bytes(header(count, **header_args) + data.tobytes())
    return path


class PlyCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def good_file(self, name="good.ply", count=200):
        data, column = splat_array(count)
        return write_ply(self.tmp / name, data), data, column


@unittest.skipIf(np is None, "numpy is not installed")
class TestCheckPly(PlyCase):

    def test_a_sound_file_reports_its_numbers(self):
        path, data, column = self.good_file()
        result = pc.check_ply(path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["vertices"], 200)
        self.assertEqual(result["properties"], len(SPLAT_PROPERTIES))
        self.assertEqual(result["bytes"], path.stat().st_size)
        self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["non_finite"], {})
        logits = data[:, column["opacity"]].astype("f8")
        opacity = 1 / (1 + np.exp(-logits))
        self.assertAlmostEqual(result["opacity_median"], float(np.median(opacity)), places=9)
        self.assertAlmostEqual(result["opacity_share_above_half"],
                               float((opacity > 0.5).mean()), places=9)
        self.assertEqual(len(result["xyz_extent_1_99"]), 3)

    def test_scale_is_the_exp_of_the_largest_axis(self):
        data, column = splat_array(3)
        data[:, column["scale_0"]] = np.log([0.01, 0.02, 0.03])
        data[:, column["scale_1"]] = np.log([0.001, 0.001, 0.001])
        data[:, column["scale_2"]] = np.log([0.005, 0.005, 0.005])
        result = pc.check_ply(write_ply(self.tmp / "s.ply", data))
        self.assertAlmostEqual(result["scale_median"], 0.02, places=6)

    def test_the_extent_is_the_one_to_ninety_nine_percentile_box(self):
        data, column = splat_array(1000)
        data[:, column["x"]] = np.linspace(0, 100, 1000)
        result = pc.check_ply(write_ply(self.tmp / "e.ply", data))
        self.assertAlmostEqual(result["xyz_extent_1_99"][0], 98.0, delta=0.1)

    def test_nan_and_infinity_fail_the_file_and_name_the_property(self):
        data, column = splat_array(50)
        data[3, column["x"]] = np.nan
        data[4, column["opacity"]] = np.inf
        data[5, column["opacity"]] = -np.inf
        result = pc.check_ply(write_ply(self.tmp / "bad.ply", data))
        self.assertFalse(result["ok"])
        self.assertEqual(result["non_finite"], {"x": 1, "opacity": 2})
        self.assertIn("3 non-finite values in opacity, x", result["problems"][0])

    def test_statistics_survive_non_finite_values(self):
        data, column = splat_array(50)
        data[0, column["opacity"]] = np.nan
        data[1, column["scale_1"]] = np.nan
        result = pc.check_ply(write_ply(self.tmp / "bad.ply", data))
        self.assertIsNotNone(result["opacity_median"])
        self.assertIsNotNone(result["scale_median"])
        json.dumps(result, allow_nan=False)      # no NaN may leak into the JSON

    def test_zero_vertices_fail(self):
        data, _ = splat_array(0)
        result = pc.check_ply(write_ply(self.tmp / "empty.ply", data))
        self.assertFalse(result["ok"])
        self.assertEqual(result["vertices"], 0)
        self.assertIn("zero vertices", result["problems"])

    def test_a_truncated_body_fails(self):
        data, _ = splat_array(10)
        result = pc.check_ply(write_ply(self.tmp / "cut.ply", data, declared=20))
        self.assertFalse(result["ok"])
        self.assertIn("truncated", result["problems"][0])
        self.assertIn("20", result["problems"][0])

    def test_unreadable_inputs_fail_without_raising(self):
        text = self.tmp / "text.ply"
        text.write_text("not a ply\n")
        ascii_ply = self.tmp / "ascii.ply"
        ascii_ply.write_bytes(header(0, fmt="ascii 1.0"))
        for path in (text, ascii_ply, self.tmp / "missing.ply"):
            with self.subTest(path=path.name):
                result = pc.check_ply(path)
                self.assertFalse(result["ok"])
                self.assertTrue(result["problems"][0].startswith("cannot read"))

    def test_a_file_without_splat_columns_still_gets_the_basics(self):
        data = np.zeros((4, 3), dtype="<f4")
        path = self.tmp / "xyz.ply"
        path.write_bytes(header(4, names=["x", "y", "z"]) + data.tobytes())
        result = pc.check_ply(path)
        self.assertTrue(result["ok"])
        self.assertNotIn("opacity_median", result)
        self.assertNotIn("scale_median", result)

    def test_no_hash_skips_the_hash(self):
        path, _, _ = self.good_file()
        self.assertNotIn("sha256", pc.check_ply(path, with_hash=False))


@unittest.skipIf(np is None, "numpy is not installed")
class TestHeader(PlyCase):

    def test_reads_count_properties_and_data_offset(self):
        raw = header(7, names=["x", "y", "z"])
        path = self.tmp / "h.ply"
        path.write_bytes(raw)
        with open(path, "rb") as handle:
            count, props, offset = pc.parse_header(handle)
        self.assertEqual((count, props, offset),
                         (7, [("x", "f4"), ("y", "f4"), ("z", "f4")], len(raw)))

    def test_other_scalar_types_and_trailing_elements(self):
        raw = (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
               b"property double x\nproperty uchar red\nproperty int16 tag\n"
               b"element face 0\nproperty list uchar int vertex_indices\nend_header\n")
        path = self.tmp / "t.ply"
        path.write_bytes(raw + np.zeros(1, dtype=[("x", "<f8"), ("r", "u1"), ("t", "<i2")]).tobytes())
        with open(path, "rb") as handle:
            _, props, _ = pc.parse_header(handle)
        self.assertEqual(props, [("x", "f8"), ("red", "u1"), ("tag", "i2")])

    def test_refusals(self):
        cases = {
            "big endian": b"ply\nformat binary_big_endian 1.0\nelement vertex 1\n"
                          b"property float x\nend_header\n",
            "no vertex element": b"ply\nformat binary_little_endian 1.0\nend_header\n",
            "unknown type": b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                            b"property quad x\nend_header\n",
            "vertex not first": b"ply\nformat binary_little_endian 1.0\nelement face 1\n"
                                b"element vertex 1\nproperty float x\nend_header\n",
            "list on vertex": b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                              b"property list uchar int x\nend_header\n",
            "no end": b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n",
        }
        for label, raw in cases.items():
            with self.subTest(label):
                with self.assertRaises(pc.PlyError):
                    pc.parse_header(io.BytesIO(raw))


@unittest.skipIf(np is None, "numpy is not installed")
class TestPreviewPoints(PlyCase):

    def _vertices(self, data):
        path = write_ply(self.tmp / "p.ply", data)
        return pc.read_ply(path)[0]

    def test_keeps_only_opaque_finite_gaussians(self):
        data, column = splat_array(400)
        data[:, column["opacity"]] = -5.0
        data[:100, column["opacity"]] = 5.0
        data[0, column["x"]] = np.nan
        xyz, rgb, kept = pc.preview_points(self._vertices(data))
        self.assertEqual(kept, 99)
        self.assertLessEqual(len(xyz), 99)
        self.assertTrue(np.isfinite(xyz).all())
        self.assertEqual(rgb.shape[1], 3)
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())

    def test_sampling_is_capped_and_repeatable(self):
        data, _ = splat_array(2000)
        vertices = self._vertices(data)
        first = pc.preview_points(vertices, max_points=50)
        second = pc.preview_points(vertices, max_points=50)
        self.assertEqual(len(first[0]), 50)
        self.assertTrue(np.array_equal(first[0], second[0]))

    def test_color_comes_from_the_dc_term(self):
        data, column = splat_array(5)
        data[:, column["opacity"]] = 5.0
        data[:, column["f_dc_0"]:column["f_dc_2"] + 1] = 0.0
        _, rgb, _ = pc.preview_points(self._vertices(data))
        self.assertTrue(np.allclose(rgb, 0.5))

    def test_nothing_opaque_returns_empty(self):
        data, column = splat_array(10)
        data[:, column["opacity"]] = -9.0
        xyz, _, kept = pc.preview_points(self._vertices(data))
        self.assertEqual((len(xyz), kept), (0, 0))


@unittest.skipIf(np is None, "numpy is not installed")
class TestCommandLine(PlyCase):

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = pc.main(list(argv))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_sound_files_exit_zero_and_print_a_block_each(self):
        a, _, _ = self.good_file("a.ply")
        b, _, _ = self.good_file("b.ply")
        code, out, _ = self._run(str(a), str(b))
        self.assertEqual(code, 0)
        self.assertIn("a.ply", out)
        self.assertIn("b.ply", out)
        self.assertEqual(out.count("result: ok"), 2)
        self.assertIn("non-finite: none", out)

    def test_one_bad_file_makes_the_exit_nonzero(self):
        good, _, _ = self.good_file()
        data, column = splat_array(10)
        data[0, column["y"]] = np.inf
        bad = write_ply(self.tmp / "bad.ply", data)
        code, out, _ = self._run(str(good), str(bad))
        self.assertEqual(code, pc.EXIT_PROBLEM)
        self.assertIn("result: ok", out)
        self.assertIn("result: FAILED", out)
        self.assertIn("PROBLEM", out)

    def test_zero_vertices_exit_nonzero(self):
        data, _ = splat_array(0)
        code, _, _ = self._run(str(write_ply(self.tmp / "e.ply", data)))
        self.assertEqual(code, pc.EXIT_PROBLEM)

    def test_json_is_a_list_with_one_object_per_file(self):
        good, _, _ = self.good_file()
        code, out, _ = self._run("--json", str(good), str(self.tmp / "missing.ply"))
        self.assertEqual(code, pc.EXIT_PROBLEM)
        records = json.loads(out)
        self.assertEqual([r["ok"] for r in records], [True, False])
        self.assertEqual(records[0]["vertices"], 200)

    def test_preview_with_two_files_is_a_usage_error(self):
        a, _, _ = self.good_file("a.ply")
        b, _, _ = self.good_file("b.ply")
        code, _, err = self._run("--preview", str(self.tmp / "p.png"), str(a), str(b))
        self.assertEqual(code, 2)
        self.assertIn("exactly one", err)

    def test_no_arguments_is_a_usage_error(self):
        self.assertEqual(self._run()[0], 2)

    def test_the_command_line_leaves_the_input_untouched(self):
        path, _, _ = self.good_file()
        before = path.read_bytes()
        self._run(str(path))
        self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_preview_writes_a_png(self):
        path, _, _ = self.good_file(count=500)
        png = self.tmp / "out" / "preview.png"
        code, out, _ = self._run("--json", "--preview", str(png), str(path))
        self.assertEqual(code, 0)
        self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
        self.assertGreater(json.loads(out)[0]["preview"]["centers_drawn"], 0)


if __name__ == "__main__":
    unittest.main()
