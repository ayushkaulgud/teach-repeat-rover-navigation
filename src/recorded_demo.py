"""Teach-and-repeat benchmark on recorded Freiburg Campus laser/odometry data.

Run ``python src/recorded_demo.py``. Recorded sensor data determine the route
and obstacles; counterfactual repeat runs use a physical Husky model in Bullet.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as bullet
import pybullet_data
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/freiburg_campus/raw.log.gz"
OUT = ROOT / "results/recorded"
METHODS = ("Direct waypoints", "Local A*", "Classical DMP", "Obstacle-aware DMP")
COLORS = ("#c5554c", "#148a7d", "#337da5", "#b7822f")
RES = 0.25
BODY_RADIUS = 0.42  # Husky footprint approximation for path screening.
MARGIN = 0.15
DT = 1 / 120


def read_log():
    poses, scans = [], []
    with gzip.open(LOG, "rt", errors="replace") as source:
        for line in source:
            if not line.startswith("FLASER "):
                continue
            fields = line.split()
            count = int(fields[1])
            i = len(poses)
            pose = np.array(fields[count + 5:count + 8], float)
            poses.append(pose)
            if 250 <= i < 570 and i % 3 == 0:
                ranges = np.array(fields[2:count + 2:3], float)
                angles = np.linspace(-np.pi / 2, np.pi / 2, count)[::3] + pose[2]
                valid = (ranges > 0.2) & (ranges < 25)
                scans.append(np.column_stack((pose[0] + ranges[valid] * np.cos(angles[valid]),
                                              pose[1] + ranges[valid] * np.sin(angles[valid]))))
    return np.asarray(poses), np.vstack(scans)


def resample(route, step=0.10):
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    distance, route = distance[keep], route[keep]
    s = np.r_[np.arange(0, distance[-1], step), distance[-1]]
    return np.column_stack((np.interp(s, distance, route[:, 0]),
                            np.interp(s, distance, route[:, 1])))


class RecordedMap:
    def __init__(self, poses, points):
        self.reference = resample(poses[300:550, :2])
        self.raw_poses = poses
        self.scan_points = points
        self.origin = self.reference.min(axis=0) - 8
        self.shape = np.ceil((self.reference.max(axis=0) + 8 - self.origin) / RES).astype(int)
        cells = np.floor((points - self.origin) / RES).astype(int)
        valid = np.all((cells >= 0) & (cells < self.shape), axis=1)
        hits = np.zeros(tuple(self.shape), dtype=np.uint16)
        np.add.at(hits, (cells[valid, 0], cells[valid, 1]), 1)
        self.occupied = hits >= 3
        self.distance = distance_transform_edt(~self.occupied) * RES
        self.gradient = np.gradient(self.distance, RES)
        self.obstacle_xy = np.column_stack(np.where(self.occupied)) * RES + self.origin + RES / 2

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy) - self.origin) / RES).astype(int))

    def clearance(self, xy):
        cell = self.cell(xy)
        if any(c < 0 or c >= n for c, n in zip(cell, self.shape)):
            return -BODY_RADIUS
        return float(self.distance[cell] - BODY_RADIUS)

    def gradient_at(self, xy):
        cell = self.cell(xy)
        if any(c < 0 or c >= n for c, n in zip(cell, self.shape)):
            return np.zeros(2)
        return np.array([g[cell] for g in self.gradient])

    def clear_path(self, path):
        if path is None:
            return False
        return all(self.clearance(xy) >= MARGIN for xy in resample(path, 0.08))


def direct_path(world, start):
    route = world.reference
    merge = min(25, len(route) - 1)
    return resample(np.vstack((start[:2], route[merge:])))


def astar_path(world, start):
    source, target = world.cell(start[:2]), world.cell(world.reference[25])
    if world.clearance(start[:2]) < MARGIN:
        return None
    frontier = [(0, source)]
    cost = {source: 0.0}
    parent = {}
    while frontier:
        _, cell = heapq.heappop(frontier)
        if cell == target:
            break
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            xy = world.origin + (np.array(neighbor) + .5) * RES
            if world.clearance(xy) < MARGIN:
                continue
            if dx and dy:
                if world.clearance(world.origin + (np.array((cell[0] + dx, cell[1])) + .5) * RES) < MARGIN:
                    continue
                if world.clearance(world.origin + (np.array((cell[0], cell[1] + dy)) + .5) * RES) < MARGIN:
                    continue
            new_cost = cost[cell] + math.hypot(dx, dy)
            if new_cost < cost.get(neighbor, math.inf):
                cost[neighbor], parent[neighbor] = new_cost, cell
                heuristic = math.dist(neighbor, target)
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
    else:
        return None
    cells = [target]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    cells.reverse()
    xy = world.origin + (np.asarray(cells) + .5) * RES
    return resample(np.vstack((start[:2], xy, world.reference[25:])))


def learn_dmp(route):
    reference = np.column_stack([np.interp(np.linspace(0, 1, 501),
                                           np.linspace(0, 1, len(route)), route[:, i]) for i in range(2)])
    dt = 1 / (len(reference) - 1)
    velocity = np.gradient(reference, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    phase = np.exp(-4 * np.linspace(0, 1, len(reference)))
    centers = np.exp(-4 * np.linspace(0, 1, 60))
    widths = 60 ** 1.5 / centers
    psi = np.exp(-widths[None, :] * (phase[:, None] - centers[None, :]) ** 2)
    features = phase[:, None] * psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-12)
    target = acceleration - 25 * (6.25 * (reference[-1] - reference) - velocity)
    weights = np.linalg.solve(features.T @ features + .001 * np.eye(60), features.T @ target)
    return centers, widths, weights


def dmp_path(world, start, model, avoid=False):
    centers, widths, weights = model
    point, velocity = np.array(start[:2], float), np.zeros(2)
    path = [point.copy()]
    dt = .002
    for k in range(850):
        phase = math.exp(-4 * k * dt)
        psi = np.exp(-widths * (phase - centers) ** 2)
        forcing = phase * (psi @ weights) / max(psi.sum(), 1e-12)
        coupling = np.zeros(2)
        if avoid:
            clearance = world.clearance(point)
            if clearance < 1.2:
                gradient = world.gradient_at(point)
                norm = np.linalg.norm(gradient)
                if norm > 1e-8:
                    coupling = 200 * max(0, 1.2 - clearance) * gradient / norm
        acceleration = 25 * (6.25 * (world.reference[-1] - point) - velocity) + forcing + coupling
        velocity += acceleration * dt
        point = point + velocity * dt
        path.append(point.copy())
    return resample(np.asarray(path))


def mesh_obstacles(world):
    """One static Bullet triangle mesh, each occupied laser cell a low box."""
    vertices, indices = [], []
    corners = ((-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0),
               (-.5, -.5, 1), (.5, -.5, 1), (.5, .5, 1), (-.5, .5, 1))
    faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    for x, y in world.obstacle_xy:
        offset = len(vertices)
        vertices.extend([(x + a * RES, y + b * RES, z) for a, b, z in corners])
        indices.extend(offset + i for face in faces for i in face)
    shape = bullet.createCollisionShape(bullet.GEOM_MESH, vertices=vertices, indices=indices,
                                        flags=bullet.GEOM_FORCE_CONCAVE_TRIMESH)
    visual = bullet.createVisualShape(bullet.GEOM_MESH, vertices=vertices, indices=indices,
                                      rgbaColor=(.42, .48, .46, 1))
    return bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, baseVisualShapeIndex=visual)


def simulate(world, path, start, capture=False):
    bullet.resetSimulation()
    bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    bullet.setGravity(0, 0, -9.81)
    bullet.setTimeStep(DT)
    bullet.loadURDF("plane.urdf")
    mesh = mesh_obstacles(world)
    rover = bullet.loadURDF("husky/husky.urdf", [*start[:2], .35],
                            bullet.getQuaternionFromEuler((0, 0, float(start[2]))))
    trace = []
    minimum = math.inf
    goal = world.reference[-1]
    status = "timeout"
    for step in range(120 * 100):
        pos, quat = bullet.getBasePositionAndOrientation(rover)
        xy = np.array(pos[:2])
        if step % 6 == 0:
            trace.append(xy.copy())
        minimum = min(minimum, world.clearance(xy))
        if bullet.getContactPoints(rover, mesh):
            status = "collision"
            break
        if np.linalg.norm(xy - goal) < .65:
            status = "completed"
            break
        if minimum < .08:
            status = "safety_stop"
            break
        if capture and step == 3000:
            view = bullet.computeViewMatrix([xy[0] + 2.5, xy[1] - 4, 4.3],
                                            [xy[0], xy[1] + .8, 0], [0, 0, 1])
            projection = bullet.computeProjectionMatrixFOV(55, 16 / 9, .1, 50)
            _, _, pixels, _, _ = bullet.getCameraImage(1280, 720, view, projection,
                                                       renderer=bullet.ER_TINY_RENDERER)
            Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(720, 1280, 4)[:, :, :3]).save(OUT / "bullet_scene.png")
        closest = int(np.argmin(np.sum((path - xy) ** 2, axis=1)))
        look = min(closest + 13, len(path) - 1)
        target = path[look]
        heading = bullet.getEulerFromQuaternion(quat)[2]
        error = (math.atan2(target[1] - xy[1], target[0] - xy[0]) - heading + math.pi) % (2 * math.pi) - math.pi
        speed = .78 * (.18 + .82 * max(0, math.cos(error)))
        turn = np.clip(2 * speed * math.sin(error) / 1.3, -1.2, 1.2)
        left, right = (speed - .27 * turn) / .165, (speed + .27 * turn) / .165
        for joint, velocity in ((2, left), (4, left), (3, right), (5, right)):
            bullet.setJointMotorControl2(rover, joint, bullet.VELOCITY_CONTROL,
                                         targetVelocity=float(velocity), force=150)
        bullet.stepSimulation()
    if np.linalg.norm(trace[-1] - xy) > 1e-8:
        trace.append(xy)
    trace = np.asarray(trace)
    return status, trace, minimum, step * DT


def scan_icp(world):
    """Align a held-out recorded scan to other recorded returns, using odometry as reference."""
    scan_id = 430
    with gzip.open(LOG, "rt", errors="replace") as source:
        scans = (line for line in source if line.startswith("FLASER "))
        fields = next(line.split() for i, line in enumerate(scans) if i == scan_id)
    count = int(fields[1])
    ranges = np.array(fields[2:count + 2:3], float)
    angles = np.linspace(-np.pi / 2, np.pi / 2, count)[::3]
    valid = (ranges > .2) & (ranges < 15)
    local = np.column_stack((ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid])))
    truth = world.raw_poses[scan_id]
    target = world.scan_points
    target = target[np.linalg.norm(target - truth[:2], axis=1) < 15]
    tree = cKDTree(target)
    estimate = truth.copy() + np.array((.5, -.4, math.radians(8)))
    before = np.linalg.norm(estimate[:2] - truth[:2])
    for _ in range(30):
        rot = np.array(((math.cos(estimate[2]), -math.sin(estimate[2])),
                        (math.sin(estimate[2]), math.cos(estimate[2]))))
        transformed = local @ rot.T + estimate[:2]
        distances, match = tree.query(transformed)
        use = distances < 1.0
        a, b = transformed[use], target[match[use]]
        ac, bc = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
        fix = vt.T @ u.T
        if np.linalg.det(fix) < 0:
            vt[-1] *= -1
            fix = vt.T @ u.T
        translation = bc - fix @ ac
        estimate[:2] = fix @ estimate[:2] + translation
        estimate[2] += math.atan2(fix[1, 0], fix[0, 0])
        if np.linalg.norm(translation) < 1e-5:
            break
    rot = np.array(((math.cos(estimate[2]), -math.sin(estimate[2])),
                    (math.sin(estimate[2]), math.cos(estimate[2]))))
    aligned = local @ rot.T + estimate[:2]
    residual = tree.query(aligned)[0]
    return {"reference_xy": truth[:2], "before_m": before,
            "after_m": float(np.linalg.norm(estimate[:2] - truth[:2])),
            "residual_rmse_m": float(np.sqrt(np.mean(residual ** 2))),
            "scan_xy": aligned, "map_xy": target}


def figures(world, examples, summary, icp):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    def scene(ax):
        ax.set_facecolor("#fbf9f0")
        ax.scatter(world.obstacle_xy[:, 0], world.obstacle_xy[:, 1], s=3, c="#475454", alpha=.6,
                   label="Recorded laser obstacles")
        ax.plot(world.reference[:, 0], world.reference[:, 1], "--", c="#274b47", lw=1.5,
                label="Recorded odometry route")
        ax.set_aspect("equal")
        ax.set(xlabel="x (m)", ylabel="y (m)")
        ax.grid(alpha=.17)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained", facecolor="#fbf9f0")
    scene(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "recorded_map.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", facecolor="#fbf9f0")
    for ax, method, color in zip(axes.flat, METHODS, COLORS):
        path, trace, status = examples[method]
        scene(ax)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], color=color, lw=2, label="Planned")
        ax.plot(trace[:, 0], trace[:, 1], color="#111111", lw=1.2, label="Bullet rover")
        ax.set_title(f"{method}  |  {status.replace('_', ' ')}", loc="left", fontsize=12)
    fig.savefig(OUT / "four_methods.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained", facecolor="#fbf9f0")
    x = np.arange(len(METHODS))
    for offset, key, title, color in ((-.24, "completion_pct", "Completed", "#27a69a"),
                                      (0, "rejection_pct", "Rejected", "#b7822f"),
                                      (.24, "safety_stop_pct", "Safety stop", "#c5554c")):
        ax.bar(x + offset, [summary[m][key] for m in METHODS], .23, label=title, color=color)
    ax.set_xticks(x, ["Waypoints", "A*", "DMP", "Obstacle DMP"])
    ax.set(ylabel="Trial share (%)", ylim=(0, 105))
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(OUT / "outcomes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained", facecolor="#fbf9f0")
    ax.scatter(icp["map_xy"][:, 0], icp["map_xy"][:, 1], s=1.5, c="#777777", label="Recorded map")
    ax.scatter(icp["scan_xy"][:, 0], icp["scan_xy"][:, 1], s=7, c="#27a69a", label="Aligned scan")
    ax.scatter(*icp["reference_xy"], s=70, marker="*", c="#c5554c", label="Odometry pose")
    ax.set_aspect("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "recorded_icp.png", dpi=180)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    poses, points = read_log()
    world = RecordedMap(poses, points)
    assert len(poses) == 15683 and len(world.reference) > 100
    assert min(world.clearance(p) for p in world.reference) > MARGIN
    model = learn_dmp(world.reference)
    heading = float(poses[300, 2])
    normal = np.array((-math.sin(heading), math.cos(heading)))
    starts = [(float(x), float(y), heading + math.radians(angle), offset, angle)
              for offset in (-1.2, -.6, 0, .6, 1.2)
              for angle in (-30, 0, 30)
              for x, y in [world.reference[0] + offset * normal]]
    bullet.connect(bullet.DIRECT)
    rows, examples = [], {}
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
                    status, trace, minimum, duration = simulate(world, path, start,
                                                               method == "Local A*" and start[3:] == (.6, 0))
                rows.append({"method": method, "offset_m": start[3], "heading_offset_deg": start[4],
                             "status": status, "minimum_clearance_m": round(minimum, 3),
                             "goal_error_m": round(float(np.linalg.norm(trace[-1] - world.reference[-1])), 3),
                             "planning_ms": round(planning_ms, 2), "duration_s": round(duration, 2),
                             "distance_m": round(float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()), 2)})
                if start[3:] == (.6, 0):
                    examples[method] = (path, trace, status)
                print(f"{method:20} offset={start[3]:+4.1f} heading={start[4]:+3} {status}", flush=True)
    finally:
        bullet.disconnect()
    with (OUT / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        completed = [r for r in subset if r["status"] == "completed"]
        executed = [r for r in subset if r["status"] != "rejected"]
        summary[method] = {"trials": len(subset)}
        for status, key in (("completed", "completion_pct"), ("rejected", "rejection_pct"),
                            ("safety_stop", "safety_stop_pct"), ("collision", "collision_pct")):
            summary[method][key] = round(100 * sum(r["status"] == status for r in subset) / len(subset), 1)
        summary[method].update({"minimum_clearance_m": min((r["minimum_clearance_m"] for r in executed), default=None),
                                "median_goal_error_m": round(float(np.median([r["goal_error_m"] for r in completed])), 3) if completed else None,
                                "median_planning_ms": round(float(np.median([r["planning_ms"] for r in subset])), 2),
                                "median_distance_m": round(float(np.median([r["distance_m"] for r in completed])), 2) if completed else None})
    icp = scan_icp(world)
    metrics = {"dataset": "Freiburg Campus 2004-07-14 raw CARMEN log", "recorded_scans_total": len(poses),
               "route_scan_indices": [300, 549], "route_length_m": round(float(np.linalg.norm(np.diff(world.reference, axis=0), axis=1).sum()), 2),
               "obstacle_cells": int(world.occupied.sum()), "map_resolution_m": RES,
               "simulator": "PyBullet DIRECT; Husky URDF; 120 Hz", "starts_per_method": len(starts),
               "start_offsets_m": [-1.2, -.6, 0, .6, 1.2], "heading_offsets_deg": [-30, 0, 30],
               "body_radius_for_screening_m": BODY_RADIUS, "path_safety_margin_m": MARGIN,
               "pose_for_navigation": "exact simulated Bullet pose", "odometry_note": "Recorded odometry is a route reference, not independent ground truth",
               "icp": {key: round(value, 3) for key, value in icp.items() if isinstance(value, float)},
               "methods": summary}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figures(world, examples, summary, icp)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
