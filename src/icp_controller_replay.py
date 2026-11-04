"""Feed ICP poses from held-out recorded scans to the repeat controller.

This is an offline replay. The controller commands are measured, not applied
to the recorded robot or to a Bullet repeat drive.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recorded_demo import (RecordedMap, astar_path, direct_path, dmp_path,
                           learn_dmp, read_log, scan_icp, tracking_command)


def run():
    out = Path(__file__).resolve().parents[1] / "results" / "icp_controller_replay"
    out.mkdir(parents=True, exist_ok=True)
    poses, points = read_log()
    world = RecordedMap(poses, points)
    scan_ids = [i for i in range(300, 550) if i % 3 != 0][::5]
    assert len(scan_ids) == 34 and all(i % 3 != 0 for i in scan_ids)
    start = poses[300]
    model = learn_dmp(world.reference)
    candidates = {
        "Direct waypoints": direct_path(world, start),
        "Local A*": astar_path(world, start),
        "Classical DMP": dmp_path(world, start, model),
        "Obstacle-aware DMP": dmp_path(world, start, model, avoid=True),
    }
    paths = {name: path for name, path in candidates.items()
             if path is not None and world.clear_path(path)}
    assert set(paths) == {"Direct waypoints", "Local A*", "Obstacle-aware DMP"}

    offsets = (("positive", np.array((.5, -.4, math.radians(8)))),
               ("negative", np.array((-.5, .4, math.radians(-8)))))
    rows = []
    for direction, offset in offsets:
        estimate = poses[scan_ids[0]].copy() + offset
        previous = scan_ids[0]
        for scan_id in scan_ids:
            if scan_id != previous:
                estimate[:2] += poses[scan_id, :2] - poses[previous, :2]
                heading_delta = poses[scan_id, 2] - poses[previous, 2]
                estimate[2] += (heading_delta + math.pi) % (2 * math.pi) - math.pi
            predicted = estimate.copy()
            aligned = scan_icp(world, scan_id, initial_pose=estimate)
            estimate = aligned["estimate_pose"]
            reference = poses[scan_id]
            for method, path in paths.items():
                _, raw_turn = tracking_command(path, predicted[:2], predicted[2])
                _, icp_turn = tracking_command(path, estimate[:2], estimate[2])
                _, reference_turn = tracking_command(path, reference[:2], reference[2])
                rows.append({"scan_index": scan_id, "offset": direction, "method": method,
                             "prediction_error_m": round(aligned["before_m"], 4),
                             "icp_error_m": round(aligned["after_m"], 4),
                             "prediction_turn_error_rad_s": round(abs(raw_turn - reference_turn), 4),
                             "icp_turn_error_rad_s": round(abs(icp_turn - reference_turn), 4)})
            previous = scan_id
    assert len(rows) == 68 * len(paths)
    assert all(math.isfinite(value) for row in rows for value in
               (row["prediction_error_m"], row["icp_error_m"], row["icp_turn_error_rad_s"]))

    with (out / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    unique = rows[::len(paths)]
    summary = {
        "dataset": "Freiburg Campus 2004-07-14 raw CARMEN log",
        "held_out_recorded_scans": len(scan_ids),
        "icp_pose_updates": len(unique),
        "controller_updates": len(rows),
        "accepted_methods": list(paths),
        "rejected_method": "Classical DMP: path failed the pre-drive clearance screen",
        "median_prediction_error_m": round(float(np.median([r["prediction_error_m"] for r in unique])), 3),
        "median_icp_error_m": round(float(np.median([r["icp_error_m"] for r in unique])), 3),
        "methods": {},
        "limit": "Offline real-scan replay with logged odometry increments. Commands were computed but not applied to a robot or Bullet drive. Reference commands are computed from logged odometry, not measured robot commands."
    }
    for method in paths:
        subset = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            "median_prediction_turn_error_rad_s": round(float(np.median(
                [r["prediction_turn_error_rad_s"] for r in subset])), 3),
            "median_icp_turn_error_rad_s": round(float(np.median(
                [r["icp_turn_error_rad_s"] for r in subset])), 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].scatter([r["scan_index"] for r in unique],
                    [r["prediction_error_m"] for r in unique],
                    s=24, color="#bc8537", alpha=.7, label="Odometry prediction")
    axes[0].scatter([r["scan_index"] for r in unique],
                    [r["icp_error_m"] for r in unique],
                    s=24, color="#168d82", alpha=.7, label="After ICP")
    axes[0].set(xlabel="Held-out scan index", ylabel="Position difference vs logged odometry (m)")
    axes[0].legend(frameon=False, fontsize=8)
    names = list(paths)
    x = np.arange(len(names))
    axes[1].bar(x - .18, [summary["methods"][n]["median_prediction_turn_error_rad_s"] for n in names],
                .35, color="#bc8537", label="Prediction")
    axes[1].bar(x + .18, [summary["methods"][n]["median_icp_turn_error_rad_s"] for n in names],
                .35, color="#168d82", label="After ICP")
    axes[1].set_xticks(x, ("Waypoints", "A*", "Obstacle DMP"))
    axes[1].set(ylabel="Median turn-command difference (rad/s)")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out / "controller_replay.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
