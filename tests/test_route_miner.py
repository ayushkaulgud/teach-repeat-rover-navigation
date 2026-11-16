import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from route_miner import (Candidate, LocalClearanceMap, describe_candidate,
                         resample, select_nonoverlapping)  # noqa: E402


class RouteMinerTests(unittest.TestCase):
    def test_resample_preserves_endpoints(self):
        path = np.array(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)))
        result = resample(path, .25)
        np.testing.assert_allclose(result[0], path[0])
        np.testing.assert_allclose(result[-1], path[-1])
        self.assertGreater(len(result), len(path))

    def test_local_map_clearance_detects_obstacle(self):
        route = np.column_stack((np.linspace(0, 30, 121), np.zeros(121)))
        hits = np.repeat(np.array(((15.0, 1.0),)), 3, axis=0)
        world = LocalClearanceMap(route, hits)
        self.assertLess(world.clearance(np.array((15.0, 1.0))), 0.0)
        self.assertGreater(world.clearance(np.array((5.0, 0.0))), .15)

    def test_candidate_selection_is_geometric_and_nonoverlapping(self):
        def item(start, stop, score):
            return Candidate(start, stop, 40, 25, 1.6, 180, 4, .4, 50, score)
        chosen = select_nonoverlapping([item(0, 200, 10), item(100, 300, 9),
                                        item(350, 550, 8)], 2)
        self.assertEqual([(x.start_scan, x.stop_scan) for x in chosen],
                         [(0, 200), (350, 550)])

    def test_curved_feasible_route_can_be_ranked(self):
        x = np.linspace(0.0, 35.0, 240)
        y = 2.0 * np.sin(2.0 * np.pi * x / 35.0)
        poses = np.column_stack((x, y, np.zeros_like(x)))
        # Repeated returns 1.5 m from the route create nearby occupancy without collision.
        obstacles = np.repeat(np.column_stack((x[::4], y[::4] + 1.5)), 3, axis=0)
        candidate = describe_candidate(0, len(poses), poses, obstacles)
        self.assertIsNotNone(candidate)
        self.assertGreater(candidate.total_heading_change_deg, 30)
        self.assertGreater(candidate.obstacle_exposure_pct, 0)


if __name__ == "__main__":
    unittest.main()
