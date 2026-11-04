"""Run configurable simulation scenarios for the recorded Freiburg benchmark.

This script extends the default benchmark by sweeping route offsets and heading
offsets across preset scenario difficulty levels. It is designed to support the
repo goals around testing under harder conditions, varying start offsets, and
running larger simulation batches without duplicating a lot of logic.

Examples:
    python src/scenario_benchmark.py --preset baseline
    python src/scenario_benchmark.py --preset hard
    python src/scenario_benchmark.py --preset extended
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as bullet

from recorded_demo import (
    MARGIN,
    METHODS,
    RecordedMap,
    astar_path,
    direct_path,
    dmp_path,
    learn_dmp,
    read_log,
    simulate,
)
from navigation_metrics import path_geometry, tracking_quality, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS = {
    "baseline": {
        "label": "baseline",
        "offsets": [-1.2, -0.6, 0.0, 0.6, 1.2],
        "heading_offsets_deg": [-30, 0, 30],
    },
    "hard": {
        "label": "hard",
        "offsets": [-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8],
        "heading_offsets_deg": [-45, -30, -15, 0, 15, 30, 45],
    },
    "extended": {
        "label": "extended",
        "offsets": [-2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4],
        "heading_offsets_deg": [-60, -45, -30, -15, 0, 15, 30, 45, 60],
    },
}


def build_starts(world: RecordedMap, route_start: int, offsets, heading_offsets_deg):
    heading = float(world.raw_poses[route_start, 2])
    normal = np.array((-math.sin(heading), math.cos(heading)))
    starts = []
    for offset in offsets:
        for angle in heading_offsets_deg:
            x, y = world.reference[0] + offset * normal
            starts.append((float(x), float(y), heading + math.radians(angle), float(offset), float(angle)))
    return starts


def run_benchmark(
    *,
    map_start: int,
    map_stop: int,
    route_start: int,
    route_stop: int,
    offsets,
    heading_offsets_deg,
    result_dir: str,
):
    out = ROOT / "results" / result_dir
    out.mkdir(parents=True, exist_ok=True)

    poses, points = read_log(map_start, map_stop)
    world = RecordedMap(poses, points, route_start, route_stop)
    route_clearance = min(world.clearance(xy) for xy in world.reference)
    if route_clearance < MARGIN:
        raise ValueError(
            f"The route fails the safety screen: minimum clearance is {route_clearance:.3f} m, "
            f"which is below the {MARGIN:.3f} m margin."
        )

    model = learn_dmp(world.reference)
    starts = build_starts(world, route_start, offsets, heading_offsets_deg)

    rows = []
    bullet.connect(bullet.DIRECT)
    try:
        for start in starts:
            for method in METHODS:
                t0 = time.perf_counter()
                if method == "Direct waypoints":
                    path = direct_path(world, start)
                elif method == "Local A*":
                    path = astar_path(world, start)
                else:
                    path = dmp_path(world, start, model, method == "Obstacle-aware DMP")
                planning_ms = (time.perf_counter() - t0) * 1000

                if not world.clear_path(path):
                    status, trace, minimum, duration = "rejected", np.array([start[:2]]), world.clearance(start[:2]), 0.0
                    diagnostics = {"minimum_bullet_clearance_m": None,
                                   "absolute_turn_integral_rad": None,
                                   "rms_turn_command_radps": None}
                else:
                    status, trace, minimum, duration, diagnostics = simulate(
                        world, path, start, return_diagnostics=True)

                row = {
                        "method": method,
                        "offset_m": start[3],
                        "heading_offset_deg": start[4],
                        "status": status,
                        "safe_success": int(status == "completed"),
                        "minimum_grid_clearance_m": round(float(minimum), 3),
                        "goal_error_m": round(float(np.linalg.norm(trace[-1] - world.reference[-1])), 3),
                        "planning_ms": round(float(planning_ms), 2),
                        "duration_s": round(float(duration), 2),
                    }
                row.update(path_geometry(path))
                row.update(tracking_quality(trace, path, duration))
                row.update(diagnostics)
                rows.append(row)
                print(f"{method:20} offset={start[3]:+4.1f} heading={start[4]:+3} {status}", flush=True)
    finally:
        bullet.disconnect()

    with (out / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "Freiburg Campus 2004-07-14 raw CARMEN log",
        "route_scan_indices": [route_start, route_stop - 1],
        "map_scan_indices": [map_start, map_stop - 1],
        "route_length_m": round(float(np.linalg.norm(np.diff(world.reference, axis=0), axis=1).sum()), 2),
        "recorded_obstacle_cells": int(world.occupied.sum()),
        "route_minimum_grid_clearance_m": round(route_clearance, 3),
        "scenario": result_dir,
        "offsets_m": [float(v) for v in offsets],
        "heading_offsets_deg": [float(v) for v in heading_offsets_deg],
        "total_attempts": len(rows),
        "methods": {},
    }
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        completed = [r for r in subset if r["status"] == "completed"]
        interval = wilson_interval(len(completed), len(subset))
        summary["methods"][method] = {
            "attempts": len(subset),
            "completed": len(completed),
            "rejected": sum(r["status"] == "rejected" for r in subset),
            "collisions": sum(r["status"] == "collision" for r in subset),
            "safety_stops": sum(r["status"] == "safety_stop" for r in subset),
            "timeouts": sum(r["status"] == "timeout" for r in subset),
            "completion_rate_pct": round(100 * len(completed) / len(subset), 1),
            "completion_rate_95ci_pct": [round(100 * value, 1) for value in interval],
            "median_goal_error_m": round(float(np.median([r["goal_error_m"] for r in completed])), 3) if completed else None,
            "median_planning_ms": round(float(np.median([r["planning_ms"] for r in subset])), 2),
            "median_duration_s": round(float(np.median([r["duration_s"] for r in completed])), 2) if completed else None,
            "median_cross_track_rmse_m": round(float(np.median([r["cross_track_rmse_m"] for r in completed])), 3) if completed else None,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run larger recorded-route simulation sweeps.")
    parser.add_argument("--preset", choices=sorted(DEFAULT_PRESETS), default="baseline")
    parser.add_argument("--map-start", type=int, default=250)
    parser.add_argument("--map-stop", type=int, default=570)
    parser.add_argument("--route-start", type=int, default=300)
    parser.add_argument("--route-stop", type=int, default=550)
    args = parser.parse_args()

    cfg = DEFAULT_PRESETS[args.preset]
    run_benchmark(
        map_start=args.map_start,
        map_stop=args.map_stop,
        route_start=args.route_start,
        route_stop=args.route_stop,
        offsets=cfg["offsets"],
        heading_offsets_deg=cfg["heading_offsets_deg"],
        result_dir=f"scenario_{cfg['label']}",
    )

