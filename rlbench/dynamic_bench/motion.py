"""
Motion primitives of the dynamic benchmark: pure functions of time, no simulator calls.

Every primitive maps a time t [s] (measured from when the primitive started) to a position, and reports when it is
finished. Speeds are in m/s of simulated time, so the motion does not depend on how fast a controller runs.
"""

from typing import List, Optional, Sequence

import numpy as np


def cosine(s: float) -> float:
    """Smooth 0 -> 1 profile with zero velocity at both ends (peak speed pi/2 times the mean speed)."""
    s = float(np.clip(s, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * s))


def min_jerk(s: float) -> float:
    """Minimum-jerk 0 -> 1 profile (bell-shaped velocity, peak 1.875 times the mean): human point-to-point reaching."""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def linear(s: float) -> float:
    """Constant velocity 0 -> 1 (abrupt start and stop)."""
    return float(np.clip(s, 0.0, 1.0))


PROFILES = {'cosine': cosine, 'min_jerk': min_jerk, 'linear': linear}
# Peak speed / mean speed of each profile: a segment of mean speed v peaks at PEAK_FACTOR * v
PEAK_FACTOR = {'cosine': np.pi / 2.0, 'min_jerk': 1.875, 'linear': 1.0}


class Segment:
    """Point-to-point move from p0 to p1. The duration comes from the PEAK speed, so 'speed' bounds the motion."""

    def __init__(self, p0: Sequence[float], p1: Sequence[float], speed: float, profile: str = 'cosine',
                 min_duration: float = 0.2):
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.profile = PROFILES[profile]
        dist = float(np.linalg.norm(self.p1 - self.p0))
        self.duration = max(min_duration, PEAK_FACTOR[profile] * dist / max(float(speed), 1e-6))

    def at(self, t: float) -> np.ndarray:
        return self.p0 + (self.p1 - self.p0) * self.profile(t / self.duration)

    def done(self, t: float) -> bool:
        return t >= self.duration


class Hold:
    """Stay at p for a duration (inf: forever)."""

    def __init__(self, p: Sequence[float], duration: float):
        self.p0 = self.p1 = np.asarray(p, dtype=np.float64)
        self.duration = float(duration)

    def at(self, t: float) -> np.ndarray:
        return self.p0.copy()

    def done(self, t: float) -> bool:
        return t >= self.duration


class Chain:
    """Chain of Segment / Hold items played one after the other; holds the last point at the end."""

    def __init__(self, items: List):
        self.items = list(items)
        self.duration = float(sum(i.duration for i in self.items))

    def at(self, t: float) -> np.ndarray:
        for item in self.items:
            if t < item.duration:
                return item.at(t)
            t -= item.duration
        return self.items[-1].at(self.items[-1].duration)

    def done(self, t: float) -> bool:
        return t >= self.duration

    def index(self, t: float) -> int:
        """Index of the item played at time t (len(items) once finished)."""
        for k, item in enumerate(self.items):
            if t < item.duration:
                return k
            t -= item.duration
        return len(self.items)


def path(points: Sequence[Sequence[float]], speed: float, profile: str = 'cosine',
         holds: Optional[Sequence[float]] = None) -> Chain:
    """Sequence through points with an optional hold [s] after each point (holds[i] after reaching points[i+1])."""
    items = []
    for i in range(len(points) - 1):
        items.append(Segment(points[i], points[i + 1], speed, profile))
        if holds is not None and i < len(holds) and holds[i] > 0:
            items.append(Hold(points[i + 1], holds[i]))
    return Chain(items)


def yaw_quat(yaw: float) -> List[float]:
    """Quaternion [x, y, z, w] of a rotation about the world z axis."""
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton product of quaternions [x, y, z, w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])
