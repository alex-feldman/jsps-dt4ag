import csv
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/camera-review/compare-camera-motion.py'
spec = importlib.util.spec_from_file_location('camera_motion', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rotation(axis, degrees):
    a = math.radians(degrees) / 2
    return (math.cos(a), *(math.sin(a) * x for x in axis))


def pose(x, q=(1, 0, 0, 0)):
    return (q, (-x, 0, 0))


class CameraMotionTest(unittest.TestCase):
    def test_rotation_convention_and_world_frame(self):
        a, b, world = rotation((1, 0, 0), 20), rotation((0, 0, 1), 30), rotation((0, 1, 0), 63)
        self.assertAlmostEqual(m.rotation_difference(m.motion((1, 0, 0, 0), b), b), 0)
        self.assertAlmostEqual(m.rotation_difference(m.motion(a, b), m.motion(m.multiply(a, world), m.multiply(b, world))), 0)
        self.assertAlmostEqual(m.rotation_difference(m.motion((1, 0, 0, 0), b), m.motion((1, 0, 0, 0), rotation((0, 0, 1), 40))), 10)

    def test_folder_boundaries_and_missing_gap(self):
        expected = ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'b/1.jpg', 'b/2.jpg']
        model = {'a/1.png': pose(0), 'a/3.png': pose(2), 'b/1.png': pose(0), 'b/2.png': pose(1)}
        rows, refusals, info = m.compare(expected, model, model)
        self.assertEqual([(r['image_a'], r['image_b']) for r in rows], [('b/1.jpg', 'b/2.jpg')])
        self.assertEqual(len(refusals), 2)
        self.assertEqual(info['missing_a'], ['a/2'])

    def test_scale_invariance_and_zero_refusal(self):
        expected = ['a/1.jpg', 'a/2.jpg', 'a/3.jpg']
        a = {'a/1.png': pose(0), 'a/2.png': pose(1), 'a/3.png': pose(3)}
        b = {'a/1.png': pose(0), 'a/2.png': pose(10), 'a/3.png': pose(30)}
        rows, refused, _ = m.compare(expected, a, b)
        self.assertFalse(refused)
        for row in rows:
            self.assertAlmostEqual(row['normalized_a'], row['normalized_b'])
        rows, refused, _ = m.compare(expected, a, {k: pose(0) for k in b})
        self.assertEqual(len(refused), 2)
        self.assertTrue(all(r['normalized_b'] is None for r in rows))

    def test_rotated_camera_center(self):
        # A +90-degree world-to-camera Z rotation maps C=(1,2,3) to
        # R*C=(-2,1,3), so t=-R*C=(2,-1,-3).
        q = rotation((0, 0, 1), 90)
        for actual, expected in zip(m.center((q, (2, -1, -3))), (1, 2, 3)):
            self.assertAlmostEqual(actual, expected)

    def test_cli_reads_binary_models_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / 'capture' / 'images' / '006'
            capture.mkdir(parents=True)
            for name in ('a.jpg', 'b.jpg', 'c.jpg'):
                (capture / name).touch()
            workspaces = [root / 'first', root / 'second']
            for workspace, scale in zip(workspaces, (1, 2)):
                model = workspace / 'sparse' / '0'
                model.mkdir(parents=True)
                with (model / 'images.bin').open('wb') as f:
                    f.write(struct.pack('<Q', 2))
                    for image_id, name, x in ((1, '006/a.png', 0), (2, '006/b.png', scale)):
                        f.write(struct.pack('<i7di', image_id, 1, 0, 0, 0, -x, 0, 0, 1))
                        f.write(name.encode() + b'\0')
                        # Include a real observation record to exercise binary skipping.
                        f.write(struct.pack('<Qddq', 1, 2.5, 3.5, -1))
            output = root / 'result'
            result = subprocess.run([sys.executable, str(path), '--capture', str(root / 'capture'),
                                     '--workspace-a', str(workspaces[0]), '--workspace-b', str(workspaces[1]),
                                     '--out', str(output)], capture_output=True, text=True, check=True)
            self.assertIn('1 eligible edges; 1 refusals', result.stdout)
            with (output / 'edges.csv').open(newline='') as f:
                edges = list(csv.DictReader(f))
            self.assertEqual((edges[0]['image_a'], edges[0]['image_b']), ('006/a.jpg', '006/b.jpg'))
            self.assertAlmostEqual(float(edges[0]['normalized_a']), 1)
            self.assertAlmostEqual(float(edges[0]['normalized_b']), 1)
            with (output / 'refusals.csv').open(newline='') as f:
                self.assertEqual(list(csv.DictReader(f))[0]['reason'], 'missing registration')
            metadata = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(metadata['missing_a'], ['006/c'])
            self.assertEqual(metadata['eligible_edges'], 1)

    def test_ambiguity_refuses(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            m.compare(['a/1.jpg', 'a/1.png'], {}, {})
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            m.compare(['a/1.jpg'], {'a/1.jpg': pose(0), 'a/1.png': pose(0)}, {})


if __name__ == '__main__':
    unittest.main()
