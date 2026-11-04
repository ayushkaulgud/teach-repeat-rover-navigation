"""Build the recorded-route and dated timeline figures for Project Update #2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from recorded_demo import RecordedMap, read_log


OUT = Path(__file__).resolve().parents[1] / "results" / "nov4"
OUT.mkdir(parents=True, exist_ok=True)


def recorded_routes():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    fig.patch.set_facecolor("#fbf6ea")
    cases = ((1050, 1300, 1000, 1320, 30.60, 278),
             (1800, 2050, 1750, 2070, 20.92, 640))
    for ax, (start, stop, map_start, map_stop, length, cells) in zip(axes, cases):
        poses, points = read_log(map_start, map_stop)
        world = RecordedMap(poses, points, start, stop)
        measured_length = np.linalg.norm(np.diff(world.reference, axis=0), axis=1).sum()
        assert abs(measured_length - length) < .01 and int(world.occupied.sum()) == cells
        direction = world.reference[-1] - world.reference[0]
        direction /= np.linalg.norm(direction)
        side = np.array((-direction[1], direction[0]))

        def local(xy):
            shifted = xy - world.reference[0]
            return np.column_stack((shifted @ direction, shifted @ side))

        route = local(world.reference)
        obstacles = local(world.obstacle_xy)
        ax.scatter(obstacles[:, 0], obstacles[:, 1],
                   s=5, color="#747e7c", alpha=.8, label="Recorded laser cells")
        ax.plot(route[:, 0], route[:, 1],
                color="#168d82", lw=2.5, label="Logged route")
        ax.scatter(*route[0], s=65, color="#bc8537", zorder=3, label="Start")
        ax.scatter(*route[-1], s=65, marker="X", color="#292929", zorder=3,
                   label="Goal")
        ax.set_title(f"Scans {start}–{stop - 1}  |  {length:.2f} m", fontsize=15, pad=12)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.15)
        ax.set_xlabel("Along route (m)")
        ax.set_ylabel("Across route (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4,
               bbox_to_anchor=(.5, -.02), fontsize=9)
    fig.savefig(OUT / "recorded_routes.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def timeline():
    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    fig.patch.set_facecolor("#fbf6ea")
    ax.set_facecolor("#fbf6ea")
    # Filled diamonds are dated results; outlined bars are future schedule windows.
    for label, day, y in (("October baseline", 0, 4),
                          ("Two new recorded routes", 28, 3),
                          ("Held-out ICP validation", 28, 2)):
        ax.scatter(day, y, s=145, marker="D", color="#168d82", zorder=5)
        ax.text(day + 1, y, label, va="center", ha="left", fontsize=11)
    for label, left, width, y in (("ICP in repeat loop", 28, 7, 1),
                                  ("Stress tests and report", 35, 11, 0)):
        ax.barh(y, width, left=left, height=.56, color="white",
                edgecolor="#168d82", linewidth=1.8)
        ax.text(left + .6, y, label, va="center", ha="left", fontsize=11)
    ax.axvline(28, color="#bc8537", lw=2.3, ls="--", zorder=4)
    ax.annotate("WE ARE HERE\nNOV 4", xy=(28, 4.3), xytext=(28, 5.1),
                ha="center", va="bottom", color="#9b6320", weight="bold", fontsize=12,
                arrowprops={"arrowstyle": "-|>", "color": "#bc8537", "lw": 1.8})
    ax.set_xlim(-1, 47)
    ax.set_ylim(-.65, 5.5)
    ax.set_yticks([])
    ax.set_xticks((0, 14, 28, 35, 46), ("Oct 7", "Oct 21", "Nov 4", "Nov 11", "Nov 22"))
    ax.tick_params(axis="x", labelsize=11, length=0, pad=8)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=(Line2D([], [], marker="D", linestyle="", markersize=8,
                              markerfacecolor="#168d82", markeredgecolor="#168d82",
                              label="Complete milestone"),
                       Patch(facecolor="white", edgecolor="#168d82", label="Planned")),
              frameon=False, loc="lower left", ncol=2)
    fig.savefig(OUT / "timeline.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    recorded_routes()
    timeline()
