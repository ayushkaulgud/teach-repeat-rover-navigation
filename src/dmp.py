"""Discrete Dynamic Movement Primitives for planar rover paths.

The implementation follows the classical discrete DMP transformation system
described by Ijspeert et al. (2013).  The learned forcing term is expressed in
the demonstration's local start-to-goal frame.  At rollout it is rotated and
scaled into the requested start-to-goal frame, avoiding the zero-axis and
global-coordinate scaling problems of independent x/y implementations.

Obstacle avoidance uses the velocity-dependent steering coupling introduced
by Hoffmann et al. (2009).  It changes the generated reference path; collision
screening and the rover controller remain separate safety layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


Array = np.ndarray
CouplingFunction = Callable[[Array, Array, float], Array]


def _frame(start: Array, goal: Array) -> tuple[Array, float]:
    """Return a 2-D start-to-goal rotation matrix and displacement magnitude."""
    displacement = np.asarray(goal, float) - np.asarray(start, float)
    length = float(np.linalg.norm(displacement))
    if length < 1e-8:
        raise ValueError("A discrete DMP requires distinct start and goal positions")
    tangent = displacement / length
    normal = np.array((-tangent[1], tangent[0]))
    return np.column_stack((tangent, normal)), length


def _resample(demonstration: Array, samples: int) -> Array:
    """Resample a path by arc length so fitting is not tied to logger frequency."""
    path = np.asarray(demonstration, float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        raise ValueError("demonstration must have shape (n, 2) with n >= 3")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-9]
    path, arc = path[keep], arc[keep]
    if len(path) < 3 or arc[-1] < 1e-8:
        raise ValueError("demonstration must contain a nonzero movement")
    # A minimum-jerk timing law preserves the geometric path while giving the
    # discrete movement the near-zero endpoint velocity assumed by the stable
    # point-attractor formulation.
    normalized_time = np.linspace(0.0, 1.0, samples)
    progress = (10 * normalized_time ** 3 - 15 * normalized_time ** 4
                + 6 * normalized_time ** 5)
    query = arc[-1] * progress
    return np.column_stack([np.interp(query, arc, path[:, axis]) for axis in range(2)])


@dataclass(frozen=True)
class DMPParameters:
    n_basis: int = 60
    alpha_z: float = 25.0
    beta_z: float = 6.25
    alpha_s: float = 8.0
    ridge: float = 1e-6
    fit_samples: int = 501


class DiscreteDMP2D:
    """A spatially scalable discrete DMP for a two-dimensional path."""

    def __init__(self, weights: Array, centers: Array, widths: Array,
                 demonstration_start: Array, demonstration_goal: Array,
                 parameters: DMPParameters):
        self.weights = np.asarray(weights, float)
        self.centers = np.asarray(centers, float)
        self.widths = np.asarray(widths, float)
        self.demonstration_start = np.asarray(demonstration_start, float)
        self.demonstration_goal = np.asarray(demonstration_goal, float)
        self.parameters = parameters

    @classmethod
    def fit(cls, demonstration: Array,
            parameters: DMPParameters = DMPParameters()) -> "DiscreteDMP2D":
        path = _resample(demonstration, parameters.fit_samples)
        time = np.linspace(0.0, 1.0, len(path))
        dt = float(time[1] - time[0])
        velocity = np.gradient(path, dt, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)

        start, goal = path[0], path[-1]
        demo_frame, demo_length = _frame(start, goal)
        forcing_global = (
            acceleration
            - parameters.alpha_z
            * (parameters.beta_z * (goal - path) - velocity)
        )
        # Row-vector form of R.T @ f, normalized by demonstrated displacement.
        forcing_local = forcing_global @ demo_frame / demo_length

        phase = np.exp(-parameters.alpha_s * time)
        centers = np.exp(-parameters.alpha_s * np.linspace(0.0, 1.0,
                                                            parameters.n_basis))
        # Neighboring Gaussian basis functions overlap throughout phase space.
        spacing = np.maximum(np.abs(np.diff(centers, append=centers[-1] ** 2)), 1e-6)
        widths = 1.0 / spacing ** 2
        psi = np.exp(-widths[None, :] * (phase[:, None] - centers[None, :]) ** 2)
        features = phase[:, None] * psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-12)
        normal = features.T @ features + parameters.ridge * np.eye(parameters.n_basis)
        weights = np.linalg.solve(normal, features.T @ forcing_local)
        return cls(weights, centers, widths, start, goal, parameters)

    def forcing(self, phase: float, start: Array, goal: Array) -> Array:
        """Evaluate the learned forcing term after spatial rotation and scaling."""
        psi = np.exp(-self.widths * (phase - self.centers) ** 2)
        local = phase * (psi @ self.weights) / max(float(psi.sum()), 1e-12)
        rollout_frame, rollout_length = _frame(start, goal)
        return rollout_length * (rollout_frame @ local)

    def rollout(self, start: Array, goal: Array, *, samples: int = 1001,
                coupling: Optional[CouplingFunction] = None) -> Array:
        """Generate a stable path from ``start`` to ``goal``.

        Time is normalized to one movement duration.  The canonical phase and
        forcing term vanish near the end, leaving the stable goal attractor.
        """
        if samples < 3:
            raise ValueError("samples must be at least 3")
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        _frame(start, goal)  # Validate before integration.
        dt = 1.0 / (samples - 1)
        y = start.copy()
        velocity = np.zeros(2)
        path = [y.copy()]
        p = self.parameters
        for step in range(samples - 1):
            phase = float(np.exp(-p.alpha_s * step * dt))
            forcing = self.forcing(phase, start, goal)
            feedback = np.zeros(2) if coupling is None else np.asarray(
                coupling(y.copy(), velocity.copy(), phase), float)
            if feedback.shape != (2,) or not np.all(np.isfinite(feedback)):
                raise ValueError("DMP coupling must return a finite two-vector")
            acceleration = p.alpha_z * (p.beta_z * (goal - y) - velocity) + forcing + feedback
            velocity += acceleration * dt
            y += velocity * dt
            path.append(y.copy())
        return np.asarray(path)


class HoffmannObstacleCoupling:
    """Velocity-dependent planar obstacle steering from Hoffmann et al. (2009).

    Only obstacles in front of and within ``influence_distance`` of the rover
    contribute.  The acceleration is perpendicular to velocity and turns away
    from the obstacle, so it deforms the motion without replacing the DMP goal
    attractor.  ``body_radius`` converts obstacle-center distance to an
    approximate rover-surface clearance.
    """

    def __init__(self, obstacle_points: Array, *, influence_distance: float = 1.5,
                 body_radius: float = 0.42, gamma: float = 1000.0,
                 beta: float = 5.0, nearest: int = 12,
                 maximum_acceleration: float = 2000.0):
        points = np.asarray(obstacle_points, float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("obstacle_points must have shape (n, 2)")
        self.obstacle_points = points
        self.influence_distance = float(influence_distance)
        self.body_radius = float(body_radius)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.nearest = int(nearest)
        self.maximum_acceleration = float(maximum_acceleration)

    def __call__(self, position: Array, velocity: Array, phase: float) -> Array:
        del phase  # The Hoffmann steering law depends on state, not phase.
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-6 or not len(self.obstacle_points):
            return np.zeros(2)
        relative = self.obstacle_points - position
        center_distance = np.linalg.norm(relative, axis=1)
        clearance = center_distance - self.body_radius
        candidates = np.flatnonzero((clearance > 0.0) &
                                    (clearance < self.influence_distance))
        if not len(candidates):
            return np.zeros(2)
        candidates = candidates[np.argsort(clearance[candidates])[:self.nearest]]
        direction_of_motion = velocity / speed
        left = np.array((-direction_of_motion[1], direction_of_motion[0]))
        result = np.zeros(2)
        for index in candidates:
            toward = relative[index] / max(center_distance[index], 1e-12)
            cosine = float(np.clip(np.dot(direction_of_motion, toward), -1.0, 1.0))
            if cosine <= 0.0:  # Obstacle is beside or behind the rover.
                continue
            theta = float(np.arccos(cosine))
            cross = direction_of_motion[0] * toward[1] - direction_of_motion[1] * toward[0]
            # Obstacle on the left produces a clockwise turn, and vice versa.
            turn_sign = -1.0 if cross >= 0.0 else 1.0
            distance_weight = (1.0 - clearance[index] / self.influence_distance) ** 2
            magnitude = self.gamma * speed * max(theta, 1e-3) * np.exp(-self.beta * theta)
            result += turn_sign * magnitude * distance_weight * left
        norm = float(np.linalg.norm(result))
        if norm > self.maximum_acceleration:
            result *= self.maximum_acceleration / norm
        return result

