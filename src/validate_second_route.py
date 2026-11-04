"""Repeat the four-method benchmark on a second recorded Freiburg route.

Run ``python src/validate_second_route.py``. The taught route and obstacle
observations are recorded; offset starts and Husky drives are simulated.
"""

import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as bullet

from recorded_demo import (MARGIN, METHODS, RecordedMap, astar_path, direct_path,
                           dmp_path, learn_dmp, read_log, simulate)


def run(route_start=1050, route_stop=1300, map_start=1000, map_stop=1320, result_dir="second_route"):
    out = Path(__file__).resolve().parents[1] / "results" / result_dir
    out.mkdir(parents=True, exist_ok=True)
    poses, points = read_log(map_start, map_stop)
    world = RecordedMap(poses, points, route_start, route_stop)
    route_clearance = min(world.clearance(xy) for xy in world.reference)
    assert route_clearance >= MARGIN, "The recorded route fails the map safety screen"
    model = learn_dmp(world.reference)
    heading = float(poses[route_start, 2])
    normal = np.array((-math.sin(heading), math.cos(heading)))
    starts = [(float(x), float(y), heading + math.radians(angle), offset, angle)
              for offset in (-1.2, -.6, 0, .6, 1.2)
              for angle in (-30, 0, 30)
              for x, y in [world.reference[0] + offset * normal]]
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
                    status, trace, minimum, duration = "rejected", np.array([start[:2]]), world.clearance(start[:2]), 0
                else:
                    status, trace, minimum, duration = simulate(world, path, start)
                rows.append({"method": method, "offset_m": start[3], "heading_offset_deg": start[4],
                             "status": status, "minimum_clearance_m": round(minimum, 3),
                             "goal_error_m": round(float(np.linalg.norm(trace[-1] - world.reference[-1])), 3),
                             "planning_ms": round(planning_ms, 2), "duration_s": round(duration, 2)})
                print(f"{method:20} offset={start[3]:+4.1f} heading={start[4]:+3} {status}", flush=True)
    finally:
        bullet.disconnect()
    assert len(rows) == 60 and all(sum(r["method"] == m for r in rows) == 15 for m in METHODS)
    with (out / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset": "Freiburg Campus 2004-07-14 raw CARMEN log",
               "route_scan_indices": [route_start, route_stop - 1], "map_scan_indices": [map_start, map_stop - 1],
               "route_length_m": round(float(np.linalg.norm(np.diff(world.reference, axis=0), axis=1).sum()), 2),
               "recorded_obstacle_cells": int(world.occupied.sum()),
               "route_minimum_grid_clearance_m": round(route_clearance, 3),
               "selection_note": "The 550-799 and 800-1049 windows had route clearance below 0.15 m and were not driven",
               "data_note": "Route and obstacle endpoints are recorded; start offsets and repeat drives are simulated.",
               "methods": {}}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        completed = [r for r in subset if r["status"] == "completed"]
        summary["methods"][method] = {
            "attempts": 15,
            "completed": len(completed),
            "rejected": sum(r["status"] == "rejected" for r in subset),
            "collisions": sum(r["status"] == "collision" for r in subset),
            "safety_stops": sum(r["status"] == "safety_stop" for r in subset),
            "timeouts": sum(r["status"] == "timeout" for r in subset),
            "median_goal_error_m": round(float(np.median([r["goal_error_m"] for r in completed])), 3) if completed else None,
            "median_planning_ms": round(float(np.median([r["planning_ms"] for r in subset])), 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    x = np.arange(len(METHODS))
    for shift, key, color in ((-.2, "completed", "#27a69a"), (0, "rejected", "#b7822f"),
                              (.2, "safety_stops", "#c5554c")):
        ax.bar(x + shift, [summary["methods"][m][key] for m in METHODS], .19,
               color=color, label=key.replace("_", " ").title())
    ax.set_xticks(x, ["Waypoints", "A*", "DMP", "Obstacle DMP"])
    ax.set(ylabel="Attempts (of 15)", ylim=(0, 16))
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out / "outcomes.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
