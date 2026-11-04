"""Aggregate the three recorded-route benchmarks for the November update."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ROUTES = (("Scans 300–549", "recorded"),
          ("Scans 1050–1299", "second_route"),
          ("Scans 1800–2049", "third_route"))
METHODS = ("Direct waypoints", "Local A*", "Classical DMP", "Obstacle-aware DMP")


def main():
    rows = []
    labels = []
    completed = []
    rejected = []
    for route_label, directory in ROUTES:
        with (ROOT / "results" / directory / "trials.csv").open(newline="") as source:
            trials = list(csv.DictReader(source))
        assert len(trials) == 60
        rows.extend(trials)
        for method in METHODS:
            counts = Counter(r["status"] for r in trials if r["method"] == method)
            assert sum(counts.values()) == 15
            labels.append(f"{route_label}  |  {method}")
            completed.append(counts["completed"])
            rejected.append(counts["rejected"])

    totals = Counter(r["status"] for r in rows)
    assert len(rows) == 180 and totals == {"completed": 165, "rejected": 15}
    summary = {
        "recorded_route_windows": [label for label, _ in ROUTES],
        "planning_attempts": len(rows),
        "simulated_drives": len(rows) - totals["rejected"],
        "statuses": dict(totals),
        "methods": {method: dict(Counter(r["status"] for r in rows if r["method"] == method))
                    for method in METHODS},
        "note": "Routes and static obstacle observations are recorded Freiburg data. Offset starts and PyBullet drives are simulated. Route windows were screened for clearance."
    }
    out = ROOT / "results" / "nov4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(11, 5.6), layout="constrained")
    positions = list(range(len(labels)))[::-1]
    ax.barh(positions, completed, color="#168d82", label="Completed")
    ax.barh(positions, rejected, left=completed, color="#bc8537", label="Rejected before drive")
    ax.set_yticks(positions, labels, fontsize=9)
    ax.set(xlim=(0, 16), xlabel="Attempts per method and route (15 each)")
    ax.set_xticks((0, 5, 10, 15))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    fig.savefig(out / "outcomes.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
