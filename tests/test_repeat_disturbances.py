import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repeat_disturbance_benchmark import (PoseSensor, Scenario, obstacle_for,
                                          smooth_route)  # noqa: E402


class RepeatDisturbanceTests(unittest.TestCase):
    def test_noise_is_paired_and_reproducible(self):
        scenario = Scenario("test", .1, 2.0, .01, .1, .5)
        a = PoseSensor(scenario, np.random.default_rng(7))
        b = PoseSensor(scenario, np.random.default_rng(7))
        truth = np.array((1.0, 2.0, .3))
        for step in range(20):
            np.testing.assert_allclose(a.observe(truth, step * .05),
                                       b.observe(truth, step * .05))

    def test_smoothing_preserves_route_endpoints(self):
        x = np.linspace(0, 10, 31)
        route = np.column_stack((x, np.sin(x)))
        smoothed = smooth_route(route)
        np.testing.assert_allclose(smoothed[0], route[0], atol=1e-9)
        np.testing.assert_allclose(smoothed[-1], route[-1], atol=1e-9)

    def test_added_obstacle_is_inside_route_not_endpoint(self):
        route = np.column_stack((np.linspace(0, 10, 101), np.zeros(101)))
        center, radius = obstacle_for(route)
        self.assertTrue(5.0 < center[0] < 7.0)
        self.assertGreater(radius, 0.4)


if __name__ == "__main__":
    unittest.main()
