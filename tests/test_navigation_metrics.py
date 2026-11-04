import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navigation_metrics import path_geometry, tracking_quality, wilson_interval  # noqa: E402


class NavigationMetricTests(unittest.TestCase):
    def test_straight_path_has_zero_bending(self):
        metrics = path_geometry(np.column_stack((np.linspace(0, 5, 51),
                                                 np.zeros(51))))
        self.assertAlmostEqual(metrics["planned_length_m"], 5.0, places=3)
        self.assertAlmostEqual(metrics["maximum_curvature_1pm"], 0.0, places=5)
        self.assertAlmostEqual(metrics["bending_energy_1pm"], 0.0, places=5)

    def test_tracking_error_has_physical_units(self):
        plan = np.column_stack((np.linspace(0, 5, 51), np.zeros(51)))
        trace = plan + np.array((0.0, 0.2))
        metrics = tracking_quality(trace, plan, 5.0)
        self.assertAlmostEqual(metrics["cross_track_rmse_m"], 0.2, places=3)
        self.assertAlmostEqual(metrics["mean_speed_mps"], 1.0, places=3)

    def test_wilson_interval_is_bounded(self):
        low, high = wilson_interval(25, 27)
        self.assertTrue(0.0 <= low < 25 / 27 < high <= 1.0)


if __name__ == "__main__":
    unittest.main()

