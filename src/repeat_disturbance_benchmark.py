"""Matched teach-and-repeat robustness benchmark.

The taught route and static scene come from Freiburg.  Repeat-time pose noise,
dropout, wheel slip, and newly introduced obstacles are controlled simulations.
Every method receives the same scenario seed.  This sensitivity benchmark is
separate from the PyBullet benchmark and must not be described as a field test.

Run after route mining:
    python src/route_miner.py --top 3
    python src/repeat_disturbance_benchmark.py --runs 10
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from dmp import DiscreteDMP2D, HoffmannObstacleCoupling
from navigation_metrics import wilson_interval
from route_miner import (DEFAULT_LOG, OUT as ROUTE_OUT, LocalClearanceMap,
                         read_carmen, resample)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/repeat_disturbances"
METHODS = ("Raw waypoints", "Smoothed waypoints", "Online local A*",
           "Online classical DMP", "Online obstacle-aware DMP")
DT = 0.05
ROBOT_RADIUS_M = 0.42
GOAL_TOLERANCE_M = 0.30


@dataclass(frozen=True)
class Scenario:
    name: str
    position_sigma_m: float = 0.0
    heading_sigma_deg: float = 0.0
    drift_m_per_sqrt_s: float = 0.0
    yaw_drift_deg_per_sqrt_s: float = 0.0
    dropout_s: float = 0.0
    traction: float = 1.0
    added_obstacle: bool = False


SCENARIOS = (
    Scenario("clean"),
    Scenario("localization_noise", .12, 3.0, .012, .20, .50),
    Scenario("repeat_obstacle", added_obstacle=True),
    Scenario("wheel_slip", .04, 1.0, traction=.72),
    Scenario("combined", .12, 3.0, .012, .20, .50, .72, True),
)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def nearest_error(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.min(np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=2), axis=1)


def smooth_route(route: np.ndarray) -> np.ndarray:
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-6]
    route, arc = route[keep], arc[keep]
    query = np.linspace(0.0, arc[-1], max(3, int(arc[-1] / .10) + 1))
    return np.column_stack((CubicSpline(arc, route[:, 0])(query),
                            CubicSpline(arc, route[:, 1])(query)))


class PoseSensor:
    """Seeded noisy pose stream with one repeatable mid-route dropout."""

    def __init__(self, scenario: Scenario, rng: np.random.Generator):
        self.scenario = scenario
        self.rng = rng
        self.bias = np.zeros(3)
        self.last: np.ndarray | None = None

    def observe(self, truth: np.ndarray, time_s: float) -> np.ndarray:
        s = self.scenario
        self.bias[:2] += self.rng.normal(0.0, s.drift_m_per_sqrt_s * math.sqrt(DT), 2)
        self.bias[2] += self.rng.normal(0.0, math.radians(s.yaw_drift_deg_per_sqrt_s)
                                       * math.sqrt(DT))
        dropout = s.dropout_s > 0 and 15.0 <= time_s < 15.0 + s.dropout_s
        if dropout and self.last is not None:
            return self.last.copy()
        measured = truth + self.bias
        measured[:2] += self.rng.normal(0.0, s.position_sigma_m, 2)
        measured[2] = wrap(measured[2] + self.rng.normal(0.0, math.radians(s.heading_sigma_deg)))
        self.last = measured
        return measured.copy()


def obstacle_for(route: np.ndarray) -> tuple[np.ndarray, float]:
    """Place a repeat-time obstacle on the taught route, away from endpoints."""
    center = route[int(.58 * (len(route) - 1))].copy()
    return center, .48


def combined_clearance(world: LocalClearanceMap, xy: np.ndarray,
                       added: tuple[np.ndarray, float] | None) -> float:
    clearance = world.clearance(xy)
    if added is not None:
        center, radius = added
        clearance = min(clearance, float(np.linalg.norm(xy - center) - radius - ROBOT_RADIUS_M))
    return clearance


def grid_astar(world: LocalClearanceMap, start_xy: np.ndarray, goal_xy: np.ndarray,
               added: tuple[np.ndarray, float] | None) -> np.ndarray | None:
    """Eight-connected local replan with the same footprint rule as collision checks."""
    resolution = .10
    source = tuple(np.floor((start_xy - world.origin) / resolution).astype(int))
    target = tuple(np.floor((goal_xy - world.origin) / resolution).astype(int))
    frontier = [(0.0, source)]
    cost = {source: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    reached = False
    while frontier:
        _, cell = heapq.heappop(frontier)
        if cell == target:
            reached = True
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            xy = world.origin + (np.asarray(neighbor) + .5) * resolution
            if combined_clearance(world, xy, added) < .15:
                continue
            new_cost = cost[cell] + math.hypot(dx, dy)
            if new_cost < cost.get(neighbor, math.inf):
                cost[neighbor] = new_cost
                parent[neighbor] = cell
                heapq.heappush(frontier, (new_cost + math.dist(neighbor, target), neighbor))
    if not reached:
        return None
    cells = [target]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    cells.reverse()
    return world.origin + (np.asarray(cells) + .5) * resolution


class OnlineDMP:
    """Incremental DMP reference generator with measured-state feedback."""

    def __init__(self, model: DiscreteDMP2D, start: np.ndarray, goal: np.ndarray,
                 coupling: HoffmannObstacleCoupling | None):
        self.model = model
        self.start = start.copy()
        self.goal = goal.copy()
        self.position = start.copy()
        self.velocity = np.zeros(2)
        self.phase = 1.0
        self.coupling = coupling

    def step(self, measured_position: np.ndarray) -> np.ndarray:
        p = self.model.parameters
        forcing = self.model.forcing(self.phase, self.start, self.goal)
        coupling = (np.zeros(2) if self.coupling is None else
                    self.coupling(self.position, self.velocity, self.phase))
        # Feedback keeps the reference generator aware of repeat-pass tracking
        # error; it is applied identically at every integration step.
        feedback = 18.0 * (measured_position - self.position)
        acceleration = (p.alpha_z * (p.beta_z * (self.goal - self.position)
                                     - self.velocity)
                        + forcing + coupling + feedback)
        # Slow phase progression when the physical rover falls behind.
        phase_scale = 1.0 / (1.0 + 2.0 * np.linalg.norm(measured_position - self.position))
        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.phase *= math.exp(-p.alpha_s * DT * phase_scale)
        return self.position.copy()


def path_target(path: np.ndarray, measured_xy: np.ndarray, lookahead_m: float = .75) -> np.ndarray:
    closest = int(np.argmin(np.sum((path - measured_xy) ** 2, axis=1)))
    target = closest
    distance = 0.0
    while target + 1 < len(path) and distance < lookahead_m:
        distance += float(np.linalg.norm(path[target + 1] - path[target]))
        target += 1
    return path[target]


def control(target: np.ndarray, pose: np.ndarray) -> tuple[float, float]:
    error = wrap(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
    speed = .78 * max(.18, math.cos(error))
    turn = float(np.clip(2.2 * error, -1.25, 1.25))
    return speed, turn


def run_trial(route: np.ndarray, world: LocalClearanceMap, scenario: Scenario,
              method: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    heading = math.atan2(route[1, 1] - route[0, 1], route[1, 0] - route[0, 0])
    truth = np.array((route[0, 0], route[0, 1], heading), float)
    sensor = PoseSensor(scenario, rng)
    added = obstacle_for(route) if scenario.added_obstacle else None
    smoothed = smooth_route(route)
    dmp = None
    active_path = route
    replans = 0
    if method == "Smoothed waypoints":
        active_path = smoothed
    elif method.startswith("Online ") and "DMP" in method:
        model = DiscreteDMP2D.fit(smoothed)
        points = world.obstacle_xy
        if added is not None:
            center, radius = added
            angles = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
            circle = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            points = np.vstack((points, circle))
        coupling = None
        if method == "Online obstacle-aware DMP":
            coupling = HoffmannObstacleCoupling(points, influence_distance=1.5,
                                                body_radius=ROBOT_RADIUS_M)
        dmp = OnlineDMP(model, truth[:2], route[-1], coupling)

    trace, turns, speeds, clearances = [], [], [], []
    status = "timeout"
    replan_clock = -math.inf
    max_time = max(80.0, 2.5 * float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()))
    steps = int(max_time / DT)
    for step in range(steps):
        time_s = step * DT
        measured = sensor.observe(truth.copy(), time_s)
        if method == "Online local A*" and time_s - replan_clock >= .50:
            closest = int(np.argmin(np.sum((route - measured[:2]) ** 2, axis=1)))
            target_index = min(closest + 45, len(route) - 1)
            local = grid_astar(world, measured[:2], route[target_index], added)
            if local is None:
                status = "planning_failure"
                break
            active_path = resample(np.vstack((local, route[target_index:])), .10)
            replan_clock = time_s
            replans += 1
        if dmp is not None:
            target = dmp.step(measured[:2])
        else:
            target = path_target(active_path, measured[:2])
        speed, turn = control(target, measured)
        traction_noise = float(np.clip(rng.normal(scenario.traction, .025), .35, 1.05))
        truth[2] = wrap(truth[2] + turn * traction_noise * DT)
        truth[:2] += speed * traction_noise * DT * np.array((math.cos(truth[2]),
                                                              math.sin(truth[2])))
        clearance = combined_clearance(world, truth[:2], added)
        trace.append(truth[:2].copy())
        turns.append(turn)
        speeds.append(speed)
        clearances.append(clearance)
        if clearance <= 0.0:
            status = "collision"
            break
        if np.linalg.norm(truth[:2] - route[-1]) <= GOAL_TOLERANCE_M:
            status = "completed"
            break
    trace_array = np.asarray(trace) if trace else truth[None, :2]
    errors = nearest_error(trace_array, route)
    duration = len(trace_array) * DT
    return {
        "method": method,
        "scenario": scenario.name,
        "seed": seed,
        "status": status,
        "completed": int(status == "completed"),
        "duration_s": round(duration, 3),
        "minimum_clearance_m": round(float(min(clearances, default=math.inf)), 4),
        "cross_track_rmse_m": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "cross_track_p95_m": round(float(np.percentile(errors, 95)), 4),
        "route_shape_error_m": round(float(np.mean(errors)), 4),
        "steering_effort_rad": round(float(np.sum(np.abs(turns)) * DT), 4),
        "rms_turn_rate_radps": round(float(np.sqrt(np.mean(np.square(turns)))) if turns else 0.0, 4),
        "mean_speed_mps": round(float(np.mean(speeds)) if speeds else 0.0, 4),
        "replans": replans,
    }


def load_routes(log_path: Path, limit: int) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    poses, scans = read_carmen(log_path)
    manifest_path = ROUTE_OUT / "selected_routes.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        routes = []
        for index, item in enumerate(manifest["selected"][:limit], 1):
            routes.append((f"mined_{index}", resample(
                poses[item["start_scan"]:item["stop_scan"], :2], .10)))
        return routes, scans
    # Explicit fallback keeps the script runnable before mining, but results
    # identify the route as the historical easy segment.
    return [("historical_300_549", resample(poses[300:550, :2], .10))], scans


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"scenarios": {}}
    for scenario in SCENARIOS:
        scenario_summary = {}
        for method in METHODS:
            subset = [row for row in rows if row["scenario"] == scenario.name
                      and row["method"] == method]
            completed = [row for row in subset if row["status"] == "completed"]
            interval = wilson_interval(len(completed), len(subset))
            scenario_summary[method] = {
                "attempts": len(subset),
                "completion_pct": round(100 * len(completed) / len(subset), 1),
                "completion_95ci_pct": [round(100 * value, 1) for value in interval],
                "collisions": sum(row["status"] == "collision" for row in subset),
                "planning_failures": sum(row["status"] == "planning_failure" for row in subset),
                "median_cross_track_rmse_m": (round(float(np.median(
                    [row["cross_track_rmse_m"] for row in completed])), 4) if completed else None),
                "median_steering_effort_rad": (round(float(np.median(
                    [row["steering_effort_rad"] for row in completed])), 4) if completed else None),
                "median_minimum_clearance_m": (round(float(np.median(
                    [row["minimum_clearance_m"] for row in completed])), 4) if completed else None),
            }
        result["scenarios"][scenario.name] = scenario_summary
    return result


def run(log_path: Path, runs: int, route_limit: int, seed: int) -> None:
    if runs < 1:
        raise ValueError("runs must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    routes, scan_points = load_routes(log_path, route_limit)
    rows: list[dict[str, object]] = []
    for route_index, (route_name, route) in enumerate(routes):
        world = LocalClearanceMap(route, scan_points)
        for scenario_index, scenario in enumerate(SCENARIOS):
            for run_index in range(runs):
                paired_seed = seed + 10000 * route_index + 100 * scenario_index + run_index
                for method in METHODS:
                    row = run_trial(route, world, scenario, method, paired_seed)
                    row["route"] = route_name
                    rows.append(row)
                    print(route_name, scenario.name, run_index, method, row["status"], flush=True)
    with (OUT / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evidence_scope": {
            "recorded": "Freiburg taught route and static laser scene",
            "simulated": "repeat pose noise, dropout, slip, added obstacle, and rover motion",
            "warning": "This is a controlled sensitivity benchmark, not a field trial.",
        },
        "paired_seed": seed,
        "runs_per_route_scenario_method": runs,
        "routes": [name for name, _ in routes],
        "methods": list(METHODS),
        **summarize(rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--routes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    run(args.log, args.runs, args.routes, args.seed)
