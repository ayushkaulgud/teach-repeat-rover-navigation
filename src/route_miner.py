"""Mine difficult, naturally recorded Freiburg route segments.

This script ranks route windows before any navigation method is evaluated.  It
therefore avoids selecting a route because it happens to favor one planner.
The raw log works out of the box; a corrected CARMEN log can be supplied with
``--log`` and is strongly preferred for long windows.

Example:
    python src/route_miner.py --top 5
    python src/route_miner.py --log data/freiburg_campus/corrected.log.gz --top 5
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data/freiburg_campus/raw.log.gz"
OUT = ROOT / "results/route_mining"
RESOLUTION_M = 0.10
ROVER_RADIUS_M = 0.42
SAFETY_MARGIN_M = 0.15


def read_carmen(path: Path, scan_stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Read FLASER poses and a downsampled global laser point cloud."""
    poses: list[np.ndarray] = []
    scans: list[np.ndarray] = []
    with gzip.open(path, "rt", errors="replace") as source:
        for line in source:
            if not line.startswith("FLASER "):
                continue
            fields = line.split()
            count = int(fields[1])
            pose = np.asarray(fields[count + 5:count + 8], float)
            scan_index = len(poses)
            poses.append(pose)
            if scan_index % scan_stride:
                continue
            ranges = np.asarray(fields[2:count + 2:3], float)
            angles = np.linspace(-np.pi / 2, np.pi / 2, count)[::3] + pose[2]
            valid = (ranges > 0.20) & (ranges < 25.0)
            scans.append(np.column_stack((
                pose[0] + ranges[valid] * np.cos(angles[valid]),
                pose[1] + ranges[valid] * np.sin(angles[valid]),
            )))
    if not poses or not scans:
        raise ValueError(f"No usable FLASER data found in {path}")
    return np.asarray(poses), np.vstack(scans)


def resample(path: np.ndarray, step: float = 0.25) -> np.ndarray:
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    path, distance = path[keep], distance[keep]
    if len(path) < 2 or distance[-1] < step:
        return np.asarray(path)
    query = np.r_[np.arange(0.0, distance[-1], step), distance[-1]]
    return np.column_stack([np.interp(query, distance, path[:, axis]) for axis in range(2)])


def smooth(path: np.ndarray, width: int = 9) -> np.ndarray:
    """Small symmetric moving average used only for complexity scoring."""
    if len(path) < width:
        return path.copy()
    pad = width // 2
    kernel = np.ones(width) / width
    return np.column_stack([
        np.convolve(np.pad(path[:, axis], pad, mode="edge"), kernel, mode="valid")
        for axis in range(2)
    ])


class LocalClearanceMap:
    """Local occupancy map; prevents long-window raw odometry errors spreading globally."""

    def __init__(self, route: np.ndarray, scan_points: np.ndarray):
        lower = route.min(axis=0) - 4.0
        upper = route.max(axis=0) + 4.0
        shape = np.maximum(1, np.ceil((upper - lower) / RESOLUTION_M).astype(int))
        nearby = scan_points[np.all((scan_points >= lower) & (scan_points < upper), axis=1)]
        cells = np.floor((nearby - lower) / RESOLUTION_M).astype(int)
        valid = np.all((cells >= 0) & (cells < shape), axis=1)
        hits = np.zeros(tuple(shape), dtype=np.uint16)
        if np.any(valid):
            np.add.at(hits, (cells[valid, 0], cells[valid, 1]), 1)
        occupied = hits >= 3
        self.origin = lower
        self.shape = shape
        self.occupied = occupied
        self.distance = distance_transform_edt(~occupied) * RESOLUTION_M
        self.obstacle_xy = (np.column_stack(np.where(occupied)) * RESOLUTION_M
                            + self.origin + RESOLUTION_M / 2)

    def clearance(self, xy: np.ndarray) -> float:
        cell = np.floor((xy - self.origin) / RESOLUTION_M).astype(int)
        if np.any(cell < 0) or np.any(cell >= self.shape):
            return -ROVER_RADIUS_M
        return float(self.distance[tuple(cell)] - ROVER_RADIUS_M)


@dataclass(frozen=True)
class Candidate:
    start_scan: int
    stop_scan: int
    length_m: float
    chord_m: float
    tortuosity: float
    total_heading_change_deg: float
    turns_over_20_deg: int
    minimum_clearance_m: float
    obstacle_exposure_pct: float
    score: float


def describe_candidate(start: int, stop: int, poses: np.ndarray,
                       scan_points: np.ndarray) -> Candidate | None:
    route = resample(poses[start:stop, :2])
    if len(route) < 10:
        return None
    length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(route[-1] - route[0]))
    if not 25.0 <= length <= 55.0 or chord < 5.0:
        return None
    filtered = smooth(route)
    headings = np.unwrap(np.arctan2(np.diff(filtered[:, 1]), np.diff(filtered[:, 0])))
    changes = np.diff(headings)
    total_turn = float(np.degrees(np.abs(changes).sum()))
    # Aggregate over approximately one metre so logger jitter is not counted as turns.
    stride = max(1, int(round(1.0 / 0.25)))
    coarse = headings[::stride]
    turn_count = int(np.count_nonzero(np.abs(np.degrees(np.diff(coarse))) >= 20.0))
    local_map = LocalClearanceMap(route, scan_points)
    clearances = np.asarray([local_map.clearance(point) for point in route])
    minimum = float(clearances.min())
    if minimum < SAFETY_MARGIN_M:
        return None
    exposure = float(np.mean(clearances <= 1.5))
    tortuosity = length / chord
    # Predeclared geometric score.  It never uses planner outcomes.
    score = (2.0 * min(tortuosity, 3.0)
             + total_turn / 90.0
             + 0.6 * turn_count
             + 2.0 * exposure
             - 0.35 * abs(minimum - 0.45))
    return Candidate(start, stop, length, chord, tortuosity, total_turn,
                     turn_count, minimum, 100.0 * exposure, score)


def candidate_windows(poses: np.ndarray, target_length_m: float = 40.0,
                      start_stride: int = 100) -> list[tuple[int, int]]:
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
    windows = []
    for start in range(0, len(poses) - 20, start_stride):
        target = cumulative[start] + target_length_m
        stop = int(np.searchsorted(cumulative, target))
        if stop < len(poses) and stop - start >= 20:
            windows.append((start, stop))
    return windows


def select_nonoverlapping(candidates: list[Candidate], count: int) -> list[Candidate]:
    chosen: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        overlap = any(not (candidate.stop_scan <= other.start_scan
                           or candidate.start_scan >= other.stop_scan)
                      for other in chosen)
        if not overlap:
            chosen.append(candidate)
        if len(chosen) == count:
            break
    return chosen


def run(log_path: Path = DEFAULT_LOG, top: int = 5, start_stride: int = 100) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    poses, scan_points = read_carmen(log_path)
    candidates = [candidate for start, stop in candidate_windows(poses, start_stride=start_stride)
                  if (candidate := describe_candidate(start, stop, poses, scan_points)) is not None]
    if not candidates:
        raise RuntimeError("No feasible route windows met the predeclared criteria")
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    selected = select_nonoverlapping(ranked, top)
    fieldnames = list(asdict(ranked[0]))
    with (OUT / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in ranked)
    manifest = {
        "source_log": str(log_path),
        "selection_policy": "top-scoring non-overlapping windows selected before planner evaluation",
        "map_resolution_m": RESOLUTION_M,
        "rover_radius_m": ROVER_RADIUS_M,
        "safety_margin_m": SAFETY_MARGIN_M,
        "selected": [asdict(item) for item in selected],
    }
    (OUT / "selected_routes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for rank, item in enumerate(selected, 1):
        route = resample(poses[item.start_scan:item.stop_scan, :2], 0.10)
        np.savetxt(OUT / f"route_{rank}.csv", route, delimiter=",",
                   header="x_m,y_m", comments="")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--start-stride", type=int, default=100)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    run(args.log, args.top, args.start_stride)
