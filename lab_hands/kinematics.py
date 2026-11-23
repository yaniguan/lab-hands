"""Planar 2R arm kinematics with boundary clamping, angle wrapping, and joint limits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

DEFAULT_JOINT_LIMITS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (-math.pi, math.pi),
    (-2.6, 2.6),
)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) in radians into [-pi, pi)."""
    arr = np.asarray(angle, dtype=np.float64)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


@dataclass(frozen=True)
class Arm2D:
    """Planar 2-link arm with configurable link lengths, joint limits, and capsule clearance."""

    l1: float = 0.30
    l2: float = 0.25
    joint_limits: Tuple[Tuple[float, float], Tuple[float, float]] = DEFAULT_JOINT_LIMITS
    clearance_m: float = 0.0
    margin_m: float = 0.005

    def __post_init__(self) -> None:
        if self.l1 <= 0.0 or self.l2 <= 0.0:
            raise ValueError(f"Link lengths must be > 0, got ({self.l1}, {self.l2})")
        if self.clearance_m < 0.0 or self.margin_m < 0.0:
            raise ValueError("clearance_m and margin_m must be >= 0")
        for idx, (lo, hi) in enumerate(self.joint_limits):
            if lo >= hi:
                raise ValueError(f"Invalid joint_limits[{idx}]=({lo}, {hi})")

    @property
    def total_padding_m(self) -> float:
        return float(self.clearance_m + self.margin_m)

    def within_joint_limits(self, q: Sequence[float] | np.ndarray, tol: float = 1e-12) -> bool:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (2,) or not np.isfinite(arr).all():
            return False
        (q1_lo, q1_hi), (q2_lo, q2_hi) = self.joint_limits
        return bool(q1_lo - tol <= arr[0] <= q1_hi + tol and q2_lo - tol <= arr[1] <= q2_hi + tol)

    def batch_within_joint_limits(self, qs: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        arr = np.asarray(qs, dtype=np.float64)
        (q1_lo, q1_hi), (q2_lo, q2_hi) = self.joint_limits
        finite = np.all(np.isfinite(arr), axis=-1)
        in_q1 = (arr[..., 0] >= q1_lo - tol) & (arr[..., 0] <= q1_hi + tol)
        in_q2 = (arr[..., 1] >= q2_lo - tol) & (arr[..., 1] <= q2_hi + tol)
        return finite & in_q1 & in_q2

    def sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        (q1_lo, q1_hi), (q2_lo, q2_hi) = self.joint_limits
        return np.array([rng.uniform(q1_lo, q1_hi), rng.uniform(q2_lo, q2_hi)], dtype=np.float64)

    def fk(self, q: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError(f"Expected joint vector of shape (2,), got {arr.shape}")
        c1, s1 = math.cos(float(arr[0])), math.sin(float(arr[0]))
        q12 = float(arr[0] + arr[1])
        c12, s12 = math.cos(q12), math.sin(q12)
        e = np.array([self.l1 * c1, self.l1 * s1], dtype=np.float64)
        t = np.array([e[0] + self.l2 * c12, e[1] + self.l2 * s12], dtype=np.float64)
        return e, t

    def batch_fk(self, qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(qs, dtype=np.float64)
        q1 = arr[..., 0]
        q12 = arr[..., 0] + arr[..., 1]
        e = np.stack([self.l1 * np.cos(q1), self.l1 * np.sin(q1)], axis=-1)
        t = e + np.stack([self.l2 * np.cos(q12), self.l2 * np.sin(q12)], axis=-1)
        return e, t

    def ik(self, p: Sequence[float] | np.ndarray, elbow: int = 1) -> Optional[np.ndarray]:
        """Analytical 2-link inverse kinematics with boundary rounding tolerance and joint-limit checks."""
        if elbow not in (1, -1):
            raise ValueError(f"elbow must be +1 or -1, got {elbow!r}")
        pt = np.asarray(p, dtype=np.float64)
        if pt.shape != (2,) or not np.isfinite(pt).all():
            raise ValueError(f"Target position p must be a finite 2-element coordinate, got {p!r}")

        r2 = float(pt[0] * pt[0] + pt[1] * pt[1])
        d = (r2 - self.l1 * self.l1 - self.l2 * self.l2) / (2.0 * self.l1 * self.l2)
        if abs(d) > 1.0 + 1e-9:
            return None
        d = float(np.clip(d, -1.0, 1.0))
        q2 = float(wrap_angle(elbow * math.acos(d)))
        q1_raw = math.atan2(float(pt[1]), float(pt[0])) - math.atan2(
            self.l2 * math.sin(q2),
            self.l1 + self.l2 * math.cos(q2),
        )
        q1 = float(wrap_angle(q1_raw))
        sol = np.array([q1, q2], dtype=np.float64)
        if not self.within_joint_limits(sol):
            return None
        return sol

    def jacobian(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        s1, c1 = math.sin(float(arr[0])), math.cos(float(arr[0]))
        q12 = float(arr[0] + arr[1])
        s12, c12 = math.sin(q12), math.cos(q12)
        return np.array(
            [
                [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                [self.l1 * c1 + self.l2 * c12, self.l2 * c12],
            ],
            dtype=np.float64,
        )

    def batch_tip_kinematics(
        self,
        qs: np.ndarray,
        qds: np.ndarray,
        qdds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute exact Cartesian tip positions, velocities, and accelerations along a trajectory."""
        q1 = qs[..., 0]
        q12 = qs[..., 0] + qs[..., 1]
        qd1 = qds[..., 0]
        qd12 = qds[..., 0] + qds[..., 1]
        qdd1 = qdds[..., 0]
        qdd12 = qdds[..., 0] + qdds[..., 1]

        c1, s1 = np.cos(q1), np.sin(q1)
        c12, s12 = np.cos(q12), np.sin(q12)

        pos = np.stack(
            [self.l1 * c1 + self.l2 * c12, self.l1 * s1 + self.l2 * s12],
            axis=-1,
        )
        vel = np.stack(
            [-self.l1 * s1 * qd1 - self.l2 * s12 * qd12, self.l1 * c1 * qd1 + self.l2 * c12 * qd12],
            axis=-1,
        )
        acc = np.stack(
            [
                -self.l1 * s1 * qdd1 - self.l2 * s12 * qdd12 - self.l1 * c1 * (qd1**2) - self.l2 * c12 * (qd12**2),
                self.l1 * c1 * qdd1 + self.l2 * c12 * qdd12 - self.l1 * s1 * (qd1**2) - self.l2 * s12 * (qd12**2),
            ],
            axis=-1,
        )
        return pos, vel, acc


DEFAULT_ARM = Arm2D()
