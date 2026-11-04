"""Presentation figures from the recorded-route stress benchmark output."""

import csv
import math
from collections import Counter
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recorded_demo import RecordedMap, read_log
from stress_benchmark import OUT, make_start


def make_figures():
    rows = list(csv.DictReader((OUT / "trials.csv").open(newline="")))
    assert rows and all(row["status"] in {"completed", "rejected", "collision", "safety_stop", "timeout"}
                        for row in rows)

    poses, points = read_log()
    world = RecordedMap(poses, points)
    origin = world.reference[0]
    heading = poses[300, 2]
    forward = np.array((math.cos(heading), math.sin(heading)))
    normal = np.array((-math.sin(heading), math.cos(heading)))
    def local(xy):
        delta = xy - origin
        return np.column_stack((delta @ forward, delta @ normal))
    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    fig.patch.set_facecolor("#fbf6ea")
    cells = local(world.obstacle_xy)
    route = local(world.reference)
    ax.scatter(cells[:, 0], cells[:, 1], s=7, color="#9a9e9c", alpha=.5,
               label="Recorded laser cells")
    ax.plot(route[:, 0], route[:, 1], color="#168d82", lw=2.4,
            label="Logged route")
    colors = ("#c74c47", "#b7822f", "#686fc5")
    for offset, color in zip((-2.4, 0, 2.4), colors):
        x, y, _ = make_start(world, poses[300], offset, 0)
        start_xy = local(np.array(((x, y),)))[0]
        ax.scatter(*start_xy, s=95, color=color,
                   edgecolor="white", lw=.8, zorder=4, label=f"{offset:+.1f} m start")
    ax.set(xlim=(-2, 33), ylim=(-12, 12), xlabel="Along logged route (m)",
           ylabel="Across logged route (m)",
           title="First recorded route and wider start positions")
    ax.grid(alpha=.18)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.savefig(OUT / "scenario_map.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    names = ("Original path screen", "Larger path margin", "Bullet proximity stop")
    scenarios = ("wide starts", "wide conservative", "wide proximity guard")
    statuses = ("completed", "rejected", "safety_stop", "collision")
    colors = {"completed": "#168d82", "rejected": "#b7822f",
              "safety_stop": "#686fc5", "collision": "#c74c47"}
    fig, ax = plt.subplots(figsize=(10.4, 3.9), layout="constrained")
    fig.patch.set_facecolor("#fbf6ea")
    for i, (scenario, name) in enumerate(zip(scenarios, names)):
        subset = [r for r in rows if r["scenario"] == scenario]
        assert len(subset) == 108
        count = Counter(r["status"] for r in subset)
        left = 0
        for status in statuses:
            n = count[status]
            if n:
                ax.barh(i, n, left=left, color=colors[status], height=.55,
                        label=status.replace("_", " ").title() if i == 0 else None)
                if n >= 6:
                    ax.text(left + n / 2, i, str(n), ha="center", va="center",
                            color="white", weight="bold", fontsize=12)
                left += n
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 108), xlabel="Planning attempts across three recorded routes",
           title="Wider starts expose a safety tradeoff")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in statuses]
    ax.legend(handles, [s.replace("_", " ").title() for s in statuses],
              frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.savefig(OUT / "safety_comparison.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    scenarios = ("matched baseline", "adaptive controller", "low friction", "class rover")
    labels = ("Original Husky", "Adaptive controller", "Low friction proxy", "Class rover model")
    durations = []
    for scenario in scenarios:
        subset = [r for r in rows if r["scenario"] == scenario]
        assert len(subset) == 12
        assert Counter(r["status"] for r in subset) == {"completed": 9, "rejected": 3}
        durations.append(median(float(r["duration_s"]) for r in subset
                                if r["status"] == "completed"))
    fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
    fig.patch.set_facecolor("#fbf6ea")
    bars = ax.barh(labels, durations, color=("#168d82", "#686fc5", "#b7822f", "#c74c47"),
                   height=.58)
    ax.invert_yaxis()
    ax.set(xlim=(0, max(durations) * 1.15), xlabel="Median simulated drive time (s)",
           title="Matched starts: 9 completed drives per setting")
    for bar, value in zip(bars, durations):
        ax.text(value + .4, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} s", va="center", fontsize=11)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "matched_checks.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    make_figures()
