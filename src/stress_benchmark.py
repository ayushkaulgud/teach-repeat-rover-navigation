"""Stress the four planners on recorded Freiburg routes in PyBullet.

All route and obstacle inputs are recorded. Starts, rover dynamics, and repeat
drives are controlled simulations; no physical field trial is implied.
"""

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as bullet

from recorded_demo import (METHODS, RecordedMap, adaptive_tracking_command,
                           astar_path, direct_path, dmp_path, learn_dmp,
                           read_log, simulate, tracking_command)
from navigation_metrics import path_geometry, tracking_quality, wilson_interval


OUT = Path(__file__).resolve().parents[1] / "results/stress"
ROUTES = (("300-549", 300, 550, 250, 570),
          ("1050-1299", 1050, 1300, 1000, 1320),
          ("1800-2049", 1800, 2050, 1750, 2070))
STANDARD_STARTS = ((-.6, 0), (0, 0), (.6, 0))
WIDE_STARTS = tuple((offset, heading) for offset in (-2.4, 0, 2.4)
                    for heading in (-60, 0, 60))
SCENARIOS = (
    ("wide starts", WIDE_STARTS, "husky", None, tracking_command),
    ("wide conservative", WIDE_STARTS, "husky", None, tracking_command),
    ("wide proximity guard", WIDE_STARTS, "husky", None, tracking_command),
    ("adaptive wide starts", WIDE_STARTS, "husky", None, adaptive_tracking_command),
    ("matched baseline", STANDARD_STARTS, "husky", None, tracking_command),
    ("adaptive controller", STANDARD_STARTS, "husky", None, adaptive_tracking_command),
    ("low friction", STANDARD_STARTS, "husky", .2, tracking_command),
    ("class rover", STANDARD_STARTS, "fri_rover", None, tracking_command),
)


def make_start(world, pose, offset, heading_offset):
    normal = np.array((-math.sin(pose[2]), math.cos(pose[2])))
    xy = world.reference[0] + offset * normal
    return float(xy[0]), float(xy[1]), float(pose[2] + math.radians(heading_offset))


def candidate(world, model, start, method):
    if method == "Direct waypoints":
        return direct_path(world, start)
    if method == "Local A*":
        return astar_path(world, start)
    return dmp_path(world, start, model, avoid=method == "Obstacle-aware DMP")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    route_lengths = {}
    bullet.connect(bullet.DIRECT)
    try:
        for label, route_start, route_stop, map_start, map_stop in ROUTES:
            poses, points = read_log(map_start, map_stop)
            world = RecordedMap(poses, points, route_start, route_stop)
            model = learn_dmp(world.reference)
            route_lengths[label] = round(float(np.linalg.norm(
                np.diff(world.reference, axis=0), axis=1).sum()), 2)
            for scenario, starts, rover, friction, controller in SCENARIOS:
                if label != ROUTES[0][0] and scenario not in (
                        "wide starts", "wide conservative", "wide proximity guard"):
                    continue
                for offset, heading in starts:
                    start = make_start(world, poses[route_start], offset, heading)
                    for method in METHODS:
                        t0 = time.perf_counter()
                        path = candidate(world, model, start, method)
                        planning_ms = round((time.perf_counter() - t0) * 1000, 2)
                        screen_margin = .38 if scenario == "wide conservative" else .15
                        if world.clear_path(path, margin=screen_margin):
                            status, trace, minimum, duration, diagnostics = simulate(
                                world, path, start, controller=controller,
                                rover_model=rover, lateral_friction=friction,
                                proximity_stop_m=.15 if scenario == "wide proximity guard" else None,
                                return_diagnostics=True)
                        else:
                            status, trace, minimum, duration = (
                                "rejected", np.array([start[:2]]),
                                world.path_minimum_clearance(path), 0)
                            diagnostics = {"minimum_bullet_clearance_m": None,
                                           "absolute_turn_integral_rad": None,
                                           "rms_turn_command_radps": None}
                        row = {
                            "route_scans": label, "scenario": scenario,
                            "method": method, "offset_m": offset,
                            "heading_offset_deg": heading, "rover_model": rover,
                            "wheel_lateral_friction": friction if friction is not None else "default",
                            "controller": "adaptive" if controller is adaptive_tracking_command else "original",
                            "screen_margin_m": screen_margin,
                            "proximity_stop_m": .15 if scenario == "wide proximity guard" else "none",
                            "status": status, "safe_success": int(status == "completed"),
                            "minimum_grid_clearance_m": round(float(minimum), 3),
                            "goal_error_m": round(float(np.linalg.norm(trace[-1] - world.reference[-1])), 3),
                            "duration_s": round(duration, 2), "planning_ms": planning_ms,
                        }
                        row.update(path_geometry(path))
                        row.update(tracking_quality(trace, path, duration))
                        row.update(diagnostics)
                        rows.append(row)
                    print(label, scenario, offset, heading,
                          [r["status"] for r in rows[-4:]], flush=True)
    finally:
        bullet.disconnect()

    expected = sum(len(starts) * len(METHODS) *
                   (len(ROUTES) if scenario in (
                       "wide starts", "wide conservative", "wide proximity guard") else 1)
                   for scenario, starts, *_ in SCENARIOS)
    assert len(rows) == expected
    assert all(math.isfinite(float(row["goal_error_m"])) for row in rows)
    with (OUT / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "Freiburg Campus 2004-07-14 raw CARMEN log",
        "recorded_route_scan_windows": list(route_lengths),
        "route_lengths_m": route_lengths,
        "simulation": "PyBullet DIRECT at 120 Hz; logged routes and laser endpoints are recorded",
        "wide_offsets_m": [-2.4, 0, 2.4],
        "wide_heading_offsets_deg": [-60, 0, 60],
        "matched_offsets_m": [-.6, 0, .6],
        "matched_heading_offset_deg": 0,
        "low_friction_proxy": "Wheel-link lateralFriction=0.2 in PyBullet; not a measured field slip coefficient",
        "conservative_screen": "0.38 m grid clearance after the 0.42 m body-radius proxy; baseline uses 0.15 m",
        "proximity_guard": "Stops the simulated rover when Bullet's body-to-obstacle distance is below 0.15 m",
        "class_rover": "Uncalibrated four-wheel URDF, 40 kg chassis, 0.165 m wheel radius, 0.27 m half-track",
        "attempts": len(rows),
        "simulated_drives": sum(r["status"] != "rejected" for r in rows),
        "scenarios": {},
    }
    for scenario, _, _, _, _ in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        summary["scenarios"][scenario] = {
            "attempts": len(subset),
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "methods": {},
        }
        for method in METHODS:
            method_rows = [r for r in subset if r["method"] == method]
            completed = [r for r in method_rows if r["status"] == "completed"]
            method_summary = dict(Counter(r["status"] for r in method_rows))
            interval = wilson_interval(len(completed), len(method_rows))
            method_summary.update({
                "completion_rate_pct": round(100 * len(completed) / len(method_rows), 1),
                "completion_rate_95ci_pct": [round(100 * value, 1) for value in interval],
                "median_planning_ms": round(float(np.median(
                    [r["planning_ms"] for r in method_rows])), 2),
                "median_duration_s": round(float(np.median(
                    [r["duration_s"] for r in completed])), 2) if completed else None,
                "median_cross_track_rmse_m": round(float(np.median(
                    [r["cross_track_rmse_m"] for r in completed])), 3) if completed else None,
                "median_bending_energy_1pm": round(float(np.median(
                    [r["bending_energy_1pm"] for r in method_rows
                     if r["bending_energy_1pm"] is not None])), 3),
            })
            summary["scenarios"][scenario]["methods"][method] = method_summary
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(11, 4.9), layout="constrained")
    names = ["Wide starts\n3 routes", "Conservative\nscreen", "Proximity\nguard", "Adaptive wide\nfirst route", "Matched\nbaseline",
             "Adaptive\ncontroller", "Low friction\nproxy", "Class rover\nmodel"]
    values = []
    for scenario, *_ in SCENARIOS:
        values.append([])
        for method in METHODS:
            subset = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            values[-1].append((sum(r["status"] == "completed" for r in subset), len(subset)))
    matrix = np.array([[done / total for done, total in row] for row in values])
    ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["Waypoints", "Local A*", "Classical DMP", "Obstacle DMP"])
    ax.set_yticks(range(len(names)), names)
    for i, row in enumerate(values):
        for j, (done, total) in enumerate(row):
            ax.text(j, i, f"{done}/{total}", ha="center", va="center", fontsize=12,
                    color="white" if matrix[i, j] > .7 else "#222222", weight="bold")
    ax.set_title("Completed drives / planning attempts", loc="left", pad=14)
    ax.tick_params(length=0)
    fig.savefig(OUT / "outcomes.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()

