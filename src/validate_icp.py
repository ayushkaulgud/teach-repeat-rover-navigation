"""Evaluate scan matching on held-out Freiburg laser scans.

Run ``python src/validate_icp.py``. The map uses every third recorded scan;
these tests use scans excluded from it. Logged odometry is the comparison pose.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recorded_demo import RecordedMap, read_log, scan_icp


def run():
    out = Path(__file__).resolve().parents[1] / "results" / "icp_validation"
    out.mkdir(parents=True, exist_ok=True)
    poses, points = read_log()
    world = RecordedMap(poses, points)
    scan_ids = [i for i in range(300, 550) if i % 3 != 0][::5]
    assert len(scan_ids) >= 30 and all(i % 3 != 0 for i in scan_ids)
    offsets = (("positive", (.5, -.4, math.radians(8))),
               ("negative", (-.5, .4, math.radians(-8))))
    rows = []
    for scan_id in scan_ids:
        for direction, offset in offsets:
            result = scan_icp(world, scan_id, offset)
            rows.append({"scan_index": scan_id, "offset": direction,
                         "initial_xy_error_m": round(result["before_m"], 4),
                         "final_xy_error_m": round(result["after_m"], 4),
                         "scan_map_rmse_m": round(result["residual_rmse_m"], 4)})
    assert all(math.isfinite(r["final_xy_error_m"]) for r in rows)
    assert abs(scan_icp(world)["after_m"] - .044) < .002
    with (out / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    final = np.array([r["final_xy_error_m"] for r in rows])
    initial = np.array([r["initial_xy_error_m"] for r in rows])
    summary = {"dataset": "Freiburg Campus 2004-07-14 raw CARMEN log",
               "scan_selection": "Every fifth route scan held out from the every-third-scan map, indices 300-549",
               "distinct_recorded_scans": len(scan_ids), "registrations": len(rows),
               "initial_offsets_xy_heading": [[.5, -.4, 8], [-.5, .4, -8]],
               "initial_error_m": round(float(np.median(initial)), 3),
               "median_final_error_m": round(float(np.median(final)), 3),
               "p90_final_error_m": round(float(np.percentile(final, 90)), 3),
               "improved_pct": round(float(np.mean(final < initial) * 100), 1),
               "within_0_2_m_pct": round(float(np.mean(final <= .2) * 100), 1),
               "reference_note": "Errors are relative to logged odometry, not surveyed ground truth; the map window uses the logged pose."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for direction, color in (("positive", "#148a7d"), ("negative", "#b7822f")):
        subset = [r for r in rows if r["offset"] == direction]
        ax.scatter([r["scan_index"] for r in subset], [r["final_xy_error_m"] for r in subset],
                   s=32, color=color, alpha=.8, label=f"{direction} initial offset")
    ax.axhline(summary["initial_error_m"], color="#555555", linestyle="--", linewidth=1.2,
               label="initial position error")
    ax.axhline(.2, color="#c5554c", linestyle=":", linewidth=1.2, label="0.2 m reference")
    ax.set(xlabel="Held-out recorded scan index", ylabel="Position error vs logged odometry (m)")
    ax.grid(alpha=.2)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, ncol=2,
              loc="upper right", fontsize=9)
    fig.savefig(out / "errors.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
