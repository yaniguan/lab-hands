"""Planar 2R and 6-DOF SE(3) arm kinematics with boundary clamping, wrapping, and verticality checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

DEFAULT_JOINT_LIMITS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (-math.pi, math.pi),
    (-2.6, 2.6),
)

DEFAULT_SE3_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-math.pi, math.pi),
    (-0.5 * math.pi, 0.85 * math.pi),
    (-2.8, 2.8),
    (-math.pi, math.pi),
    (-0.75 * math.pi, 0.75 * math.pi),
    (-math.pi, math.pi),
)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) in radians into [-pi, pi)."""
    arr = np.asarray(angle, dtype=np.float64)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Construct 3x3 SO(3) rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _rot_to_rpy(rot: np.ndarray) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) in radians from a 3x3 SO(3) matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    sp = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(sp)
    cp = math.cos(pitch)
    if abs(cp) > 1e-9:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rot[0, 1]), float(rot[1, 1]))
    return (float(wrap_angle(roll)), float(wrap_angle(pitch)), float(wrap_angle(yaw)))


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


@dataclass(frozen=True)
class PoseSE3:
    """6-DOF end-effector flange pose in SE(3) with tool-axis verticality checks for tall vessels."""

    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        p = np.asarray(self.xyz, dtype=np.float64)
        a = np.asarray(self.rpy, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f"PoseSE3.xyz must be 3 finite floats, got {self.xyz!r}")
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError(f"PoseSE3.rpy must be 3 finite floats, got {self.rpy!r}")

    def rotation_matrix(self) -> np.ndarray:
        return _rpy_to_rot(float(self.rpy[0]), float(self.rpy[1]), float(self.rpy[2]))

    def homogeneous_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation_matrix()
        mat[:3, 3] = np.asarray(self.xyz, dtype=np.float64)
        return mat

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "PoseSE3":
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"Expected (4, 4) SE(3) matrix, got {arr.shape}")
        rpy = _rot_to_rpy(arr[:3, :3])
        xyz = (float(arr[0, 3]), float(arr[1, 3]), float(arr[2, 3]))
        return cls(xyz=xyz, rpy=rpy)

    def tool_tilt_rad(self) -> float:
        """Angle acos(R_zz) between pipette tool axis and world vertical axis."""
        r_zz = float(np.clip(self.rotation_matrix()[2, 2], -1.0, 1.0))
        return math.acos(r_zz)

    def tip_position(self, attached_tip_length_m: float = 0.0) -> np.ndarray:
        """3D world position of the pipette tip apex suspended `attached_tip_length_m` along -z_tool."""
        if attached_tip_length_m < 0.0:
            raise ValueError(f"attached_tip_length_m must be >= 0, got {attached_tip_length_m}")
        rot = self.rotation_matrix()
        return np.asarray(self.xyz, dtype=np.float64) - float(attached_tip_length_m) * rot[:, 2]

    def check_plunge_verticality(
        self,
        well_radius_m: float,
        plunge_depth_m: float,
        *,
        shaft_radius_m: float = 0.0030,
        attached_tip_length_m: float = 0.0,
        max_tilt_rad: float = math.radians(1.5),
    ) -> bool:
        """Verify tool-axis tilt acos(R_zz) <= max_tilt_rad and shaft avoids vessel rim wall binding."""
        if max_tilt_rad < 0.0:
            raise ValueError("max_tilt_rad must be >= 0")
        tilt = self.tool_tilt_rad()
        if tilt > max_tilt_rad + 1e-12:
            return False
        eff_depth = max(0.0, float(plunge_depth_m))
        eff_lever = max(eff_depth, min(float(attached_tip_length_m), eff_depth))
        if eff_lever > 1e-9 and well_radius_m > 0.0:
            eff_shaft = min(float(shaft_radius_m), 0.75 * float(well_radius_m))
            radial_deflection = eff_lever * math.tan(tilt) + eff_shaft
            if radial_deflection > float(well_radius_m) + 1e-12:
                return False
        return True


@dataclass(frozen=True)
class ArmSE3:
    """6R spherical-wrist cobot kinematic model (SO-ARM100 / UR5e benchtop class) in SE(3)."""

    z_base: float = 0.10
    l_upper: float = 0.35
    l_forearm: float = 0.32
    joint_limits: Tuple[Tuple[float, float], ...] = DEFAULT_SE3_JOINT_LIMITS
    max_plunge_tilt_rad: float = math.radians(1.5)

    def __post_init__(self) -> None:
        if self.l_upper <= 0.0 or self.l_forearm <= 0.0:
            raise ValueError("ArmSE3 link lengths must be > 0")
        if len(self.joint_limits) != 6:
            raise ValueError(f"Expected 6 joint limits, got {len(self.joint_limits)}")

    def within_joint_limits(self, q: Sequence[float] | np.ndarray, tol: float = 1e-9) -> bool:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (6,) or not np.isfinite(arr).all():
            return False
        for val, (lo, hi) in zip(arr, self.joint_limits):
            if not (lo - tol <= float(val) <= hi + tol):
                return False
        return True

    def fk_se3(self, q: Sequence[float] | np.ndarray) -> PoseSE3:
        """Closed-form 6-DOF forward kinematics mapping 6R joint vector q -> PoseSE3."""
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (6,):
            raise ValueError(f"Expected 6-DOF joint vector of shape (6,), got {arr.shape}")
        q1, q2, q3, q4, q5, q6 = (float(v) for v in arr)
        q23 = q2 + q3
        r_reach = self.l_upper * math.cos(q2) + self.l_forearm * math.cos(q23)
        x = r_reach * math.cos(q1)
        y = r_reach * math.sin(q1)
        z = self.z_base + self.l_upper * math.sin(q2) + self.l_forearm * math.sin(q23)

        r_123 = _rot_z(q1) @ _rot_y(-q23)
        r_wrist = _rpy_to_rot(q4, q5, q6)
        r_total = r_123 @ r_wrist
        return PoseSE3(xyz=(x, y, z), rpy=_rot_to_rpy(r_total))

    def ik_se3(self, target: PoseSE3, elbow: int = 1) -> Optional[np.ndarray]:
        """Analytical 6R spherical-wrist inverse kinematics for `target` flange PoseSE3."""
        if elbow not in (1, -1):
            raise ValueError(f"elbow must be +1 or -1, got {elbow!r}")
        x, y, z = (float(v) for v in target.xyz)
        r_xy = math.hypot(x, y)
        dz = z - self.z_base
        dist2 = r_xy * r_xy + dz * dz
        d = (dist2 - self.l_upper * self.l_upper - self.l_forearm * self.l_forearm) / (
            2.0 * self.l_upper * self.l_forearm
        )
        if abs(d) > 1.0 + 1e-9:
            return None
        d = float(np.clip(d, -1.0, 1.0))

        q1 = float(wrap_angle(math.atan2(y, x))) if r_xy > 1e-12 else 0.0
        # elbow = +1 selects elbow-up configuration (q3 <= 0) keeping elbow high above deck labware
        q3 = float(wrap_angle(-elbow * math.acos(d)))
        q2 = float(
            wrap_angle(
                math.atan2(dz, r_xy)
                - math.atan2(self.l_forearm * math.sin(q3), self.l_upper + self.l_forearm * math.cos(q3))
            )
        )
        r_123 = _rot_z(q1) @ _rot_y(-(q2 + q3))
        r_wrist = r_123.T @ target.rotation_matrix()
        q4, q5, q6 = _rot_to_rpy(r_wrist)
        sol = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)
        if not self.within_joint_limits(sol):
            return None
        return sol


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
DEFAULT_ARM_SE3 = ArmSE3()
