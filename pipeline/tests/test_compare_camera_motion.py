import importlib.util
import math
from pathlib import Path
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

    def test_ambiguity_refuses(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            m.compare(['a/1.jpg', 'a/1.png'], {}, {})
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            m.compare(['a/1.jpg'], {'a/1.jpg': pose(0), 'a/1.png': pose(0)}, {})


if __name__ == '__main__':
    unittest.main()
