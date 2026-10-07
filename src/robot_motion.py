"""Reproducible 2-D teach-and-repeat rover demonstration.

Run: python rover_demo.py
This is an initial simulation, not a physical-rover validation.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent.parent / "results"
OBSTACLES = ((3.55, 3.10, 0.72), (6.35, 3.50, 0.72))
ROVER_RADIUS = 0.16
SAFETY_MARGIN = 0.12
LIMITS = (0.0, 10.0, 0.0, 7.0)
METHODS = ("Waypoints", "Local reconnection", "Classical DMP", "Obstacle-aware DMP")
COLORS = ("#ed6a5a", "#54d2a0", "#65b8ff", "#ffc56e")


def demonstration(n=251):
    t = np.linspace(0, 1, n)
    x = 0.8 + 8.4 * t
    y = 1.05 + 4.75 * t - 1.24 * np.sin(2 * np.pi * t)
    return np.column_stack((x, y))


DEMO = demonstration()


def clearance(points):
    points = np.asarray(points)
    c = np.minimum.reduce([np.hypot(points[..., 0] - x, points[..., 1] - y) - r - ROVER_RADIUS
                           for x, y, r in OBSTACLES])
    return c


def inside(points):
    p = np.asarray(points)
    return bool(np.all((p[:, 0] >= 0.16) & (p[:, 0] <= 9.84)
                       & (p[:, 1] >= 0.16) & (p[:, 1] <= 6.84)))


def densify(points, step=0.035):
    points = np.asarray(points)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = np.arange(0, distance[-1], step)
    s = np.r_[s, distance[-1]]
    return np.column_stack([np.interp(s, distance, points[:, i]) for i in range(2)])


def waypoint_path(start):
    # Closest forward taught waypoint, then resume the stored route.
    valid = np.flatnonzero(DEMO[:, 0] >= start[0] + 0.15)
    j = int(valid[np.argmin(np.linalg.norm(DEMO[valid] - start[:2], axis=1))])
    return densify(np.vstack((start[:2], DEMO[j:])))


def reconnect_path(start, resolution=0.12):
    """8-connected A* to the nearest feasible point on the taught route."""
    nx, ny = int(10 / resolution) + 1, int(7 / resolution) + 1

    def cell(p):
        return (round(float(p[0]) / resolution), round(float(p[1]) / resolution))

    def point(c):
        return np.array(c) * resolution

    def free(c):
        if not (2 <= c[0] < nx - 2 and 2 <= c[1] < ny - 2):
            return False
        return bool(clearance(point(c)) >= SAFETY_MARGIN)

    source = cell(start)
    candidates = [i for i in range(0, 176, 5) if DEMO[i, 0] > start[0] + 0.55
                  and free(cell(DEMO[i]))]
    if not free(source) or not candidates:
        return None
    # Search once, stop at the first reachable merge cell among nearby points.
    candidates.sort(key=lambda i: np.linalg.norm(DEMO[i] - start[:2]))
    targets = {cell(DEMO[i]): i for i in candidates[:12]}
    frontier = [(0.0, source)]
    cost = {source: 0.0}
    parent = {}
    goal = None
    while frontier:
        priority, current = heapq.heappop(frontier)
        if current in targets:
            goal = current
            break
        if priority > cost[current] + min(math.dist(current, t) for t in targets):
            continue
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if not free(nxt) or (dx and dy and (not free((current[0] + dx, current[1]))
                                              or not free((current[0], current[1] + dy)))):
                continue
            new_cost = cost[current] + math.hypot(dx, dy)
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt], parent[nxt] = new_cost, current
                heapq.heappush(frontier, (new_cost + min(math.dist(nxt, t) for t in targets), nxt))
    if goal is None:
        return None
    cells = [goal]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    cells.reverse()
    route = np.vstack((start[:2], [point(c) for c in cells[1:]], DEMO[targets[goal]:]))
    return densify(route)


def learn_dmp(reference=DEMO, bases=50):
    """Fit a discrete DMP forcing term to the demonstrated x/y movement."""
    dt = 1 / (len(reference) - 1)
    velocity = np.gradient(reference, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    phase = np.exp(-4 * np.linspace(0, 1, len(reference)))
    centers = np.exp(-4 * np.linspace(0, 1, bases))
    widths = bases ** 1.5 / centers
    psi = np.exp(-widths[None, :] * (phase[:, None] - centers[None, :]) ** 2)
    features = phase[:, None] * psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-12)
    target = acceleration - 25 * (6.25 * (reference[-1] - reference) - velocity)
    weights = np.linalg.solve(features.T @ features + 0.001 * np.eye(bases), features.T @ target)
    return centers, widths, weights


DMP_MODEL = learn_dmp()


def dmp_path(start, obstacle_aware=False):
    centers, widths, weights = DMP_MODEL
    p = np.array(start[:2], dtype=float)
    v = np.zeros(2)
    path = [p.copy()]
    dt = 0.005
    for k in range(350):
        phase = math.exp(-4 * k * dt)
        psi = np.exp(-widths * (phase - centers) ** 2)
        forcing = phase * (psi @ weights) / max(psi.sum(), 1e-12)
        coupling = np.zeros(2)
        if obstacle_aware:
            for x, y, r in OBSTACLES:
                delta = p - (x, y)
                distance = np.linalg.norm(delta)
                surface = distance - r - ROVER_RADIUS
                if surface < 0.85:
                    coupling += 28.0 * max(0, 0.85 - surface) * delta / max(distance, 1e-9)
        acceleration = 25 * (6.25 * (DEMO[-1] - p) - v) + forcing + coupling
        v += acceleration * dt
        p = p + v * dt
        path.append(p.copy())
    return densify(path)


def execute(path, start):
    """Shared bounded unicycle controller; stop on collision or goal."""
    x, y, heading = map(float, start)
    trace = [[x, y]]
    goal = DEMO[-1]
    dt = 0.04
    for _ in range(750):
        position = np.array((x, y))
        if np.linalg.norm(position - goal) < 0.18:
            return "completed", np.asarray(trace)
        if clearance(position) < 0 or not inside(position[None, :]):
            return "collision", np.asarray(trace)
        if clearance(position) < SAFETY_MARGIN:
            return "safety_stop", np.asarray(trace)
        nearest = int(np.argmin(np.sum((path - position) ** 2, axis=1)))
        look = min(nearest + 13, len(path) - 1)
        target = path[look]
        error = math.atan2(target[1] - y, target[0] - x) - heading
        error = (error + math.pi) % (2 * math.pi) - math.pi
        speed = 0.72 * (0.22 + 0.78 * max(0, math.cos(error)))
        turn = float(np.clip(2 * speed * math.sin(error) / 0.48, -1.5, 1.5))
        heading += turn * dt
        x += speed * math.cos(heading) * dt
        y += speed * math.sin(heading) * dt
        trace.append([x, y])
    return "timeout", np.asarray(trace)


def icp_demo(seed=18):
    """Point-to-point 2-D ICP on synthetic terrain landmarks."""
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform((0.5, 0.5), (9.5, 6.5), size=(55, 2))
    angle, shift = math.radians(9), np.array((0.32, -0.24))
    rot = np.array(((math.cos(angle), -math.sin(angle)),
                    (math.sin(angle), math.cos(angle))))
    scan = (landmarks - shift) @ rot + rng.normal(0, 0.008, landmarks.shape)
    corrected = scan.copy()
    estimated_r = np.eye(2)
    estimated_t = np.zeros(2)
    for _ in range(30):
        nearest = np.argmin(np.sum((corrected[:, None] - landmarks[None, :]) ** 2, axis=2), axis=1)
        target = landmarks[nearest]
        a, b = corrected.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((corrected - a).T @ (target - b))
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = u @ vt
        t = b - a @ r
        corrected = corrected @ r + t
        estimated_r = estimated_r @ r
        estimated_t = estimated_t @ r + t
        if np.linalg.norm(t) < 1e-7:
            break
    return {
        "map": landmarks, "scan": scan, "corrected": corrected,
        "translation_error_m": float(np.linalg.norm(estimated_t - shift)),
        "rotation_error_deg": float(abs(math.degrees(math.atan2(estimated_r[0, 1], estimated_r[0, 0])) - 9)),
        "registration_rmse_m": float(np.sqrt(np.mean(np.sum((corrected - landmarks) ** 2, axis=1)))),
    }


def draw_map(ax):
    ax.set_facecolor("#0d1b2b")
    for x, y, r in OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, facecolor="#a74450", alpha=0.95, zorder=2))
        ax.add_patch(plt.Circle((x, y), r + ROVER_RADIUS + SAFETY_MARGIN,
                                fill=False, color="#ed6a5a", lw=1, ls="--", alpha=0.7))
    ax.plot(DEMO[:, 0], DEMO[:, 1], color="white", lw=2.0, ls="--", alpha=0.75, label="Taught route")
    ax.set(xlim=(0, 10), ylim=(0, 7), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    ax.grid(color="white", alpha=0.08)
    ax.tick_params(colors="#c8d7e7")
    for spine in ax.spines.values():
        spine.set_color("#63788b")
    ax.xaxis.label.set_color("#c8d7e7")
    ax.yaxis.label.set_color("#c8d7e7")


def generate():
    OUT.mkdir(exist_ok=True)
    starts = [(x, y, math.radians(h)) for x in (0.8, 1.5, 2.2, 2.8)
              for y in (0.65, 1.55, 2.45, 3.35, 4.25)
              for h in (-35, 0, 35)]
    rows = []
    example = (2.2, 3.35, math.radians(-35))
    example_paths = {}
    for start in starts:
        for method in METHODS:
            t0 = time.perf_counter()
            if method == "Waypoints":
                path = waypoint_path(start)
            elif method == "Local reconnection":
                path = reconnect_path(start)
            else:
                path = dmp_path(start, method == "Obstacle-aware DMP")
            planning_ms = (time.perf_counter() - t0) * 1000
            if path is None or not inside(path) or float(np.min(clearance(path))) < SAFETY_MARGIN:
                status, trace = "rejected", np.array([start[:2]])
            else:
                status, trace = execute(path, start)
            rows.append({
                "method": method, "start_x_m": start[0], "start_y_m": start[1],
                "heading_deg": round(math.degrees(start[2])), "status": status,
                "minimum_clearance_m": round(float(np.min(clearance(trace))), 4),
                "goal_error_m": round(float(np.linalg.norm(trace[-1] - DEMO[-1])), 4),
                "planning_ms": round(planning_ms, 3),
                "distance_m": round(float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1))), 4),
            })
            if start == example:
                example_paths[method] = (path, trace, status)
    with (OUT / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        executed = [r for r in subset if r["status"] != "rejected"]
        completed = [r for r in subset if r["status"] == "completed"]
        summary[method] = {
            "trials": len(subset),
            "completion_pct": round(100 * len(completed) / len(subset), 1),
            "collision_pct": round(100 * sum(r["status"] == "collision" for r in subset) / len(subset), 1),
            "rejection_pct": round(100 * sum(r["status"] == "rejected" for r in subset) / len(subset), 1),
            "safety_stop_pct": round(100 * sum(r["status"] == "safety_stop" for r in subset) / len(subset), 1),
            "minimum_clearance_m": round(min((r["minimum_clearance_m"] for r in executed), default=float("nan")), 3),
            "median_goal_error_m": round(float(np.median([r["goal_error_m"] for r in completed])), 3) if completed else None,
            "median_planning_ms": round(float(np.median([r["planning_ms"] for r in subset])), 2),
            "median_distance_m": round(float(np.median([r["distance_m"] for r in completed])), 2) if completed else None,
        }
    icp = icp_demo()
    metrics = {"scenario": "2-D static map, two circular obstacles, 60 start poses, deterministic controller",
               "rover_radius_m": ROVER_RADIUS, "path_safety_margin_m": SAFETY_MARGIN,
               "icp": {k: round(v, 4) for k, v in icp.items() if isinstance(v, float)},
               "methods": summary}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.facecolor": "#0b1523", "figure.facecolor": "#0b1523",
                         "text.color": "#e9f0f8", "axes.titlecolor": "#e9f0f8"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, method, color in zip(axes.flat, METHODS, COLORS):
        draw_map(ax)
        path, trace, status = example_paths[method]
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], color=color, lw=2.7, label="Planned")
        ax.plot(trace[:, 0], trace[:, 1], color="#f7e9bd", lw=1.3, alpha=0.85, label="Executed")
        ax.scatter([example[0]], [example[1]], s=90, color=color, marker="*", zorder=5)
        ax.set_title(method + "  |  " + status.upper(), fontsize=12, fontweight="bold")
    fig.savefig(OUT / "four_methods.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.1), layout="constrained")
    ax.set_facecolor("#0d1b2b")
    x = np.arange(4)
    for offset, key, label, color in ((-0.30, "completion_pct", "Completed", "#54d2a0"),
                                      (-0.10, "rejection_pct", "Rejected", "#ffc56e"),
                                      (0.10, "safety_stop_pct", "Safety stopped", "#ad99e7"),
                                      (0.30, "collision_pct", "Collided", "#ed6a5a")):
        values = [summary[m][key] for m in METHODS]
        bars = ax.bar(x + offset, values, 0.19, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.0f}%" if v else "" for v in values], padding=3,
                     color="#e9f0f8", fontsize=9)
    ax.set_xticks(x, ["Waypoints", "Local\nreconnection", "Classical\nDMP", "Obstacle-aware\nDMP"])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Share of 60 starts (%)", color="#c8d7e7")
    ax.tick_params(colors="#c8d7e7")
    ax.grid(axis="y", color="white", alpha=0.12)
    ax.legend(facecolor="#17263a", labelcolor="#e9f0f8", frameon=False, ncol=4)
    fig.savefig(OUT / "outcomes.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.set_facecolor("#0d1b2b")
    ax.scatter(icp["map"][:, 0], icp["map"][:, 1], s=25, color="#65b8ff", label="Recorded map")
    ax.scatter(icp["scan"][:, 0], icp["scan"][:, 1], s=20, color="#ed6a5a", marker="x", label="Current scan")
    ax.scatter(icp["corrected"][:, 0], icp["corrected"][:, 1], s=13, color="#54d2a0", label="After ICP")
    ax.set(xlim=(0, 10), ylim=(0, 7), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    ax.tick_params(colors="#c8d7e7")
    ax.xaxis.label.set_color("#c8d7e7")
    ax.yaxis.label.set_color("#c8d7e7")
    ax.legend(facecolor="#17263a", labelcolor="#e9f0f8", frameon=False)
    fig.savefig(OUT / "icp_registration.png", dpi=170)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    assert np.min(clearance(DEMO)) > SAFETY_MARGIN
    assert icp_demo()["translation_error_m"] < 0.05
    generate()
