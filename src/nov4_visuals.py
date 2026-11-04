"""Build the recorded-route and dated timeline figures for Project Update #2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(figsize=(11.2, 5.5), layout="constrained")
    fig.patch.set_facecolor("#fbf6ea")
    ax.set_facecolor("white")
    indigo, red, amber = "#686fc5", "#c74c47", "#b7822f"
    bands = (
        (6, "#eef0fb"), (5, "#f7f7fd"), (4, "#fffadf"),
        (3, "#f7f7fd"), (2, "#eef0fb"), (1, "#f7f7fd"), (0, "#fffadf"),
    )
    for y, color in bands:
        ax.axhspan(y - .46, y + .46, color=color, zorder=0)

    # Schedule windows follow the class timeline; filled work is complete at Nov 4.
    tasks = (
        (6, 0, 17, "Simulation + route replay", indigo, True, True),
        (5, 7, 17, "Freiburg map + waypoint / A*", indigo, True, False),
        (4, 14, 45, "Four planners on three recorded routes", red, True, True),
        (3, 31, 45, "ICP control replay", red, True, True),
        (3, 45, 52, "ICP drive", red, False, True),
        (2, 17, 45, "180 attempts / 165 drives", indigo, True, True),
        (2, 52, 63, "Stress tests", indigo, False, True),
        (1, 45, 63, "Report + demo", indigo, False, True),
    )
    for y, start, end, label, color, done, inside in tasks:
        ax.barh(y, end - start, left=start, height=.42,
                color=color if done else "white", edgecolor=color,
                linewidth=1.8, zorder=3)
        ax.text(start + .5 if inside else end + .5, y, label,
                va="center", ha="left", fontsize=12,
                color="white" if done and inside else "#292929", zorder=4)

    for x, label in ((17, "M1"), (28, "M2"), (45, "M3"), (63, "M4")):
        ax.scatter(x, 0, s=130, marker="D",
                   facecolor=indigo if x < 63 else "white",
                   edgecolor=indigo, linewidth=1.7, zorder=5)
        ax.text(x + .7 if x < 63 else x - .7, 0, label,
                ha="left" if x < 63 else "right", va="center",
                fontsize=11, color="#292929", zorder=5)

    ax.axvline(45, color=amber, lw=2.2, ls="--", zorder=6)
    ax.annotate("WE ARE HERE\nNOV 4", xy=(45, 6.47), xytext=(45, 7.12),
                ha="center", va="bottom", fontsize=14, color="#946018",
                weight="bold", arrowprops={"arrowstyle": "-|>", "color": amber, "lw": 1.7})
    ax.text(0, 7.2, "Filled: complete     Outline: planned", fontsize=11,
            color="#555555", va="bottom")
    ax.set_xlim(-1.5, 65)
    ax.set_ylim(-.6, 7.65)
    ax.set_yticks((6, 5, 4, 3, 2, 1, 0),
                  ("Phase 1 Foundation", "Phase 2 Data + baselines",
                   "Phase 3 Method comparison", "Phase 4 Localization",
                   "Phase 5 Experimentation", "Phase 6 Finalization",
                   "Milestones"))
    ax.tick_params(axis="y", labelsize=11, length=0, pad=8)
    ax.set_xticks((0, 7, 14, 21, 28, 35, 45, 49, 56, 63),
                  ("Sep 20", "Sep 27", "Oct 4", "Oct 11", "Oct 18",
                   "Oct 25", "Nov 4", "Nov 8", "Nov 15", "Nov 22"))
    ax.tick_params(axis="x", labelsize=10, length=0, pad=8)
    ax.grid(axis="x", color="#555555", alpha=.23)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(OUT / "timeline.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    recorded_routes()
    timeline()
