"""Objective, unit-bearing metrics shared by the navigation benchmarks."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def path_geometry(path):
    """Return geometric path metrics without combining them into a score."""
    if path is None or len(path) < 3:
        return {
            "planned_length_m": None,
            "maximum_curvature_1pm": None,
            "bending_energy_1pm": None,
        }
    points = np.asarray(path, float)
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-8
    if valid.sum() < 2:
        return {
            "planned_length_m": float(lengths.sum()),
            "maximum_curvature_1pm": None,
            "bending_energy_1pm": None,
        }
    segments, lengths = segments[valid], lengths[valid]
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    delta_heading = np.diff(headings)
    local_length = np.maximum((lengths[:-1] + lengths[1:]) / 2, 1e-8)
    curvature = delta_heading / local_length
    return {
        "planned_length_m": round(float(lengths.sum()), 4),
        "maximum_curvature_1pm": round(float(np.max(np.abs(curvature))), 4),
        "bending_energy_1pm": round(float(np.sum(curvature ** 2 * local_length)), 4),
    }


def tracking_quality(trace, planned_path, duration_s):
    """Return execution metrics derived from the driven trace."""
    trace = np.asarray(trace, float)
    if len(trace) < 2:
        return {
            "driven_distance_m": 0.0,
            "mean_speed_mps": 0.0,
            "cross_track_rmse_m": None,
            "cross_track_p95_m": None,
        }
    distance = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum())
    metrics = {
        "driven_distance_m": round(distance, 4),
        "mean_speed_mps": round(distance / duration_s, 4) if duration_s > 0 else 0.0,
        "cross_track_rmse_m": None,
        "cross_track_p95_m": None,
    }
    if planned_path is not None and len(planned_path):
        errors = cKDTree(np.asarray(planned_path, float)).query(trace)[0]
        metrics["cross_track_rmse_m"] = round(float(np.sqrt(np.mean(errors ** 2))), 4)
        metrics["cross_track_p95_m"] = round(float(np.percentile(errors, 95)), 4)
    return metrics


def safe_success(status):
    """A safe success is a completion without contact or a triggered stop."""
    return int(status == "completed")


def wilson_interval(successes, attempts, z=1.96):
    """Wilson 95% interval for a binomial rate, returned as proportions."""
    if attempts <= 0:
        return None, None
    p = successes / attempts
    denominator = 1 + z ** 2 / attempts
    center = (p + z ** 2 / (2 * attempts)) / denominator
    radius = z * math.sqrt(p * (1 - p) / attempts + z ** 2 /
                           (4 * attempts ** 2)) / denominator
    return round(center - radius, 4), round(center + radius, 4)

