import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmp import DiscreteDMP2D, HoffmannObstacleCoupling  # noqa: E402


def curved_demo():
    x = np.linspace(0.0, 10.0, 151)
    return np.column_stack((x, 1.5 * np.sin(np.pi * x / 10.0)))


class DMPTests(unittest.TestCase):
    def test_dmp_reproduces_demonstrated_shape_and_converges(self):
        model = DiscreteDMP2D.fit(curved_demo())
        path = model.rollout((0.0, 0.0), (10.0, 0.0))
        self.assertLess(np.linalg.norm(path[-1] - np.array((10.0, 0.0))), 0.02)
        self.assertTrue(1.40 < path[:, 1].max() < 1.60)

    def test_dmp_rotates_translates_and_scales_the_motion(self):
        model = DiscreteDMP2D.fit(curved_demo())
        path = model.rollout((2.0, 3.0), (2.0, 13.0))
        self.assertLess(np.linalg.norm(path[-1] - np.array((2.0, 13.0))), 0.02)
        # A +90 degree rotation turns the positive y bow toward negative x.
        self.assertTrue(0.40 < 2.0 - path[:, 0].min() < 1.60)

    def test_forcing_term_vanishes_near_the_goal(self):
        model = DiscreteDMP2D.fit(curved_demo())
        early = np.linalg.norm(model.forcing(1.0, np.array((0.0, 0.0)),
                                            np.array((10.0, 0.0))))
        late = np.linalg.norm(model.forcing(1e-5, np.array((0.0, 0.0)),
                                           np.array((10.0, 0.0))))
        self.assertLess(late, early * 1e-3)

    def test_hoffmann_coupling_increases_clearance_and_preserves_goal(self):
        demonstration = np.column_stack((np.linspace(0.0, 10.0, 151),
                                         np.zeros(151)))
        model = DiscreteDMP2D.fit(demonstration)
        obstacles = np.array(((5.0, 0.40), (5.0, -0.40), (5.3, 0.0)))
        baseline = model.rollout((0.0, 0.0), (10.0, 0.0))
        coupling = HoffmannObstacleCoupling(
            obstacles, influence_distance=2.0, body_radius=0.20)
        avoided = model.rollout((0.0, 0.0), (10.0, 0.0), coupling=coupling)
        baseline_clearance = np.min(np.linalg.norm(
            baseline[:, None, :] - obstacles[None, :, :], axis=2))
        avoided_clearance = np.min(np.linalg.norm(
            avoided[:, None, :] - obstacles[None, :, :], axis=2))
        self.assertGreater(avoided_clearance, baseline_clearance + 0.15)
        self.assertLess(np.linalg.norm(avoided[-1] - np.array((10.0, 0.0))), 0.05)

    def test_invalid_dmp_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            DiscreteDMP2D.fit(np.zeros((4, 2)))


if __name__ == "__main__":
    unittest.main()

