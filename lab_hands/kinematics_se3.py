"""Exact SE(3) Product-of-Exponentials & DH/MDH cobot kinematics (6-DoF UR5e, 7-DoF Panda, Opentrons Flex).

Provides:
1. Exact SO(3) / SE(3) Lie group & Lie algebra operators (`exp_so3`, `log_so3`, `exp_se3`,
   `log_se3`, `adjoint_se3`, `pose_error_world`) using linear-first twist ordering
   `[vx, vy, vz, wx, wy, wz]`.
2. Exact 6-DoF Universal Robots UR5e (Standard DH + PoE), 7-DoF Franka Emika Panda
   (Craig Modified-DH + PoE), and 4-DoF Opentrons Flex Cartesian gantry + rotary gripper
   kinematic chains with 3D multi-link capsule geometries and mounted pipette tip offsets.
3. Analytical Space Jacobian `J_space(q)`, Body Jacobian `J_body(q)`, Geometric World-Aligned
   Jacobian `J_world(q)`, and Yoshikawa manipulability `manipulability(q)`.
4. Damped Least-Squares (`DLS`) inverse kinematics with null-space joint-limit avoidance
   and Yoshikawa manipulability maximization (`dls_ik`).
5. Exact 3D end-effector & multi-well rigid-body linear + Euler + centrifugal accelerations
   in local well/plate coordinates for 96-well and 384-well open-meniscus slosh evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .kinematics import PoseSE3, wrap_angle

GRAVITY_M_S2: float = 9.80665
JointKind = Literal["revolute", "prismatic"]


def skew3(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric cross-product matrix `[v]_x`."""
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def unskew3(mat: np.ndarray) -> np.ndarray:
    """Extract 3-vector `v` from skew-symmetric 3x3 matrix `[v]_x`."""
    return np.array(
        [
            0.5 * float(mat[2, 1] - mat[1, 2]),
            0.5 * float(mat[0, 2] - mat[2, 0]),
            0.5 * float(mat[1, 0] - mat[0, 1]),
        ],
        dtype=np.float64,
    )


def exp_so3(omega_theta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Exact Rodrigues exponential map `exp([omega * theta]_x) -> SO(3)`."""
    w = np.asarray(omega_theta, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        # First + second order Taylor series for numerical stability near zero
        K = skew3(w)
        return np.eye(3, dtype=np.float64) + K + 0.5 * (K @ K)
    axis = w / theta
    K = skew3(axis)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def log_so3(rot: np.ndarray) -> np.ndarray:
    """Exact matrix logarithm `SO(3) -> R^3` rotation vector `omega * theta` (`||omega*theta|| <= pi`)."""
    R = np.asarray(rot, dtype=np.float64)
    cos_theta = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    if cos_theta > 1.0 - 1e-10:
        return 0.5 * unskew3(R - R.T)
    if cos_theta < -1.0 + 1e-7:
        # Near pi rotation: extract unit axis from symmetric part (R + I) / 2 = axis @ axis.T
        diag = np.maximum(0.0, 0.5 * (np.diag(R) + 1.0))
        idx = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        axis[idx] = math.sqrt(float(diag[idx]))
        if axis[idx] > 1e-12:
            for j in range(3):
                if j != idx:
                    axis[j] = float(R[idx, j] + R[j, idx]) / (4.0 * axis[idx])
        else:
            axis[0] = 1.0
        norm_a = float(np.linalg.norm(axis))
        if norm_a > 1e-12:
            axis /= norm_a
        theta = math.acos(cos_theta)
        return axis * theta
    theta = math.acos(cos_theta)
    return (theta / (2.0 * math.sin(theta))) * unskew3(R - R.T)


def exp_se3(screw: Sequence[float] | np.ndarray, theta: float) -> np.ndarray:
    """Exact SE(3) matrix exponential `exp([S] * theta)` for screw `S = [vx, vy, vz, wx, wy, wz]`."""
    s = np.asarray(screw, dtype=np.float64)
    v = s[:3]
    w = s[3:]
    T = np.eye(4, dtype=np.float64)
    w_norm = float(np.linalg.norm(w))
    th = float(theta)
    if w_norm < 1e-12:
        # Pure prismatic translation
        T[:3, 3] = v * th
        return T
    w_unit = w / w_norm
    v_scaled = v / w_norm
    ang = w_norm * th
    K = skew3(w_unit)
    K2 = K @ K
    s_ang = math.sin(ang)
    c_ang = math.cos(ang)
    T[:3, :3] = np.eye(3, dtype=np.float64) + s_ang * K + (1.0 - c_ang) * K2
    G = ang * np.eye(3, dtype=np.float64) + (1.0 - c_ang) * K + (ang - s_ang) * K2
    T[:3, 3] = G @ v_scaled
    return T


def log_se3(mat: np.ndarray) -> np.ndarray:
    """Exact SE(3) matrix logarithm returning 6-vector `[vx, vy, vz, wx, wy, wz] * theta`."""
    T = np.asarray(mat, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    wt = log_so3(R)
    theta = float(np.linalg.norm(wt))
    if theta < 1e-10:
        K = skew3(wt)
        G_inv = np.eye(3, dtype=np.float64) - 0.5 * K + (1.0 / 12.0) * (K @ K)
        return np.concatenate([G_inv @ p, wt])
    w_unit = wt / theta
    K = skew3(w_unit)
    half = 0.5 * theta
    cot_half = half / math.tan(half)
    G_inv = (
        np.eye(3, dtype=np.float64) / theta
        - 0.5 * K
        + (1.0 / theta - 0.5 / math.tan(half)) * (K @ K)
    )
    _ = cot_half
    vt = (G_inv @ p) * theta
    return np.concatenate([vt, wt])


def adjoint_se3(mat: np.ndarray) -> np.ndarray:
    """6x6 Adjoint matrix `Ad_T` for linear-first twists `[vx, vy, vz, wx, wy, wz]`."""
    R = mat[:3, :3]
    p = mat[:3, 3]
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3:] = skew3(p) @ R
    out[3:, 3:] = R
    return out


def inverse_se3(mat: np.ndarray) -> np.ndarray:
    """Exact rigid-body inverse `T^{-1}` of a 4x4 SE(3) matrix."""
    R = mat[:3, :3]
    p = mat[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ p)
    return out


def pose_error_world(t_current: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """World-aligned 6D pose error `[dp_x, dp_y, dp_z, dw_x, dw_y, dw_z]` matching `J_world @ dq`."""
    p_cur = t_current[:3, 3]
    r_cur = t_current[:3, :3]
    p_tgt = t_target[:3, 3]
    r_tgt = t_target[:3, :3]
    dp = p_tgt - p_cur
    dw_body = log_so3(r_cur.T @ r_tgt)
    dw_world = r_cur @ dw_body
    return np.concatenate([dp, dw_world])


def _standard_dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _modified_dh_matrix(a_prev: float, d: float, alpha_prev: float, theta: float) -> np.ndarray:
    """Craig Modified-DH transform `Rx(alpha_{i-1}) @ Tx(a_{i-1}) @ Rz(theta_i) @ Tz(d_i)`."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha_prev), math.sin(alpha_prev)
    return np.array(
        [
            [ct, -st, 0.0, a_prev],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LinkCapsuleSpec:
    """3D swept-sphere capsule attached to `frame_index` (`0` = base, `1..n` = joint frames, `n+1` = flange)."""

    name: str
    frame_index: int
    p0_local: Tuple[float, float, float]
    p1_local: Tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True)
class IKResultSE3:
    """Detailed Damped Least-Squares + null-space inverse kinematics result."""

    success: bool
    q: np.ndarray
    pos_error_m: float
    rot_error_rad: float
    iterations: int
    manipulability: float
    seed_index: int


class SerialChainSE3:
    """Exact PoE multi-DoF robot arm with analytical Jacobians, DLS IK, and 3D link capsules."""

    def __init__(
        self,
        *,
        name: str,
        screw_axes_space: np.ndarray,
        home_matrix: np.ndarray,
        joint_kinds: Sequence[JointKind],
        joint_limits: Sequence[Tuple[float, float]],
        qd_max: Sequence[float],
        qdd_max: Sequence[float],
        qddd_max: Sequence[float],
        capsules: Sequence[LinkCapsuleSpec],
        home_q: Sequence[float],
        ik_seed_bank: Sequence[Sequence[float]] = (),
        base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        margin_m: float = 0.005,
        pipette_holder_radius_m: float = 0.012,
    ) -> None:
        self.name = str(name)
        screws = np.asarray(screw_axes_space, dtype=np.float64)
        home = np.asarray(home_matrix, dtype=np.float64)
        if screws.ndim != 2 or screws.shape[1] != 6:
            raise ValueError(f"screw_axes_space must have shape (n, 6), got {screws.shape}")
        if home.shape != (4, 4):
            raise ValueError(f"home_matrix must have shape (4, 4), got {home.shape}")
        n = int(screws.shape[0])
        if len(joint_kinds) != n or len(joint_limits) != n:
            raise ValueError("joint_kinds and joint_limits must match DOF count")
        if len(qd_max) != n or len(qdd_max) != n or len(qddd_max) != n:
            raise ValueError("velocity, acceleration, and jerk bounds must match DOF count")

        self.dof: int = n
        self.joint_kinds: Tuple[JointKind, ...] = tuple(joint_kinds)
        self.joint_limits: Tuple[Tuple[float, float], ...] = tuple(
            (float(lo), float(hi)) for lo, hi in joint_limits
        )
        self.q_lower = np.array([lo for lo, _ in self.joint_limits], dtype=np.float64)
        self.q_upper = np.array([hi for _, hi in self.joint_limits], dtype=np.float64)
        self.q_mid = 0.5 * (self.q_lower + self.q_upper)
        self.q_span = np.maximum(1e-6, self.q_upper - self.q_lower)

        self.qd_max = np.asarray(qd_max, dtype=np.float64)
        self.qdd_max = np.asarray(qdd_max, dtype=np.float64)
        self.qddd_max = np.asarray(qddd_max, dtype=np.float64)
        self.capsules: Tuple[LinkCapsuleSpec, ...] = tuple(capsules)
        self.home_q = np.asarray(home_q, dtype=np.float64)
        self.ik_seed_bank: Tuple[np.ndarray, ...] = tuple(
            np.asarray(s, dtype=np.float64) for s in ik_seed_bank
        )
        self.base_offset_xyz: Tuple[float, float, float] = (
            float(base_offset_xyz[0]),
            float(base_offset_xyz[1]),
            float(base_offset_xyz[2]),
        )
        self.base_transform = np.eye(4, dtype=np.float64)
        self.base_transform[:3, 3] = np.asarray(self.base_offset_xyz, dtype=np.float64)

        # Transform space screw axes & home matrix into world frame via base_transform
        ad_base = adjoint_se3(self.base_transform)
        self.screw_axes_local = screws.copy()
        self.home_matrix_local = home.copy()
        self.screw_axes_space = (ad_base @ screws.T).T
        self.home_matrix = self.base_transform @ home

        self.margin_m = float(margin_m)
        self.pipette_holder_radius_m = float(pipette_holder_radius_m)

    def within_joint_limits(self, q: Sequence[float] | np.ndarray, tol: float = 1e-9) -> bool:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.dof,) or not np.isfinite(arr).all():
            return False
        return bool(np.all(arr >= self.q_lower - tol) and np.all(arr <= self.q_upper + tol))

    def batch_within_joint_limits(self, qs: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        arr = np.asarray(qs, dtype=np.float64)
        finite = np.all(np.isfinite(arr), axis=-1)
        lo_ok = np.all(arr >= self.q_lower - tol, axis=-1)
        hi_ok = np.all(arr <= self.q_upper + tol, axis=-1)
        return finite & lo_ok & hi_ok

    def clamp_joints(self, q: Sequence[float] | np.ndarray, eps: float = 0.0) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        return np.clip(arr, self.q_lower + eps, self.q_upper - eps)

    def sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.q_lower, self.q_upper).astype(np.float64)

    def fk_poe(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Exact Product-of-Exponentials forward kinematics `T(q) = exp([S_1]q_1)...exp([S_n]q_n) M`."""
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.dof,):
            raise ValueError(f"Expected joint vector of shape ({self.dof},), got {arr.shape}")
        T = np.eye(4, dtype=np.float64)
        for i in range(self.dof):
            T = T @ exp_se3(self.screw_axes_space[i], float(arr[i]))
        return T @ self.home_matrix

    def fk_matrix(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.fk_poe(q)

    def fk(self, q: Sequence[float] | np.ndarray) -> PoseSE3:
        return PoseSE3.from_matrix(self.fk_poe(q))

    def fk_se3(self, q: Sequence[float] | np.ndarray) -> PoseSE3:
        return self.fk(q)

    def joint_frames_world(self, q: Sequence[float] | np.ndarray) -> List[np.ndarray]:
        """Return world SE(3) frames `[T_base, T_1(q), ..., T_n(q), T_flange(q)]`."""
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.dof,):
            raise ValueError(f"Expected joint vector of shape ({self.dof},), got {arr.shape}")
        frames: List[np.ndarray] = [self.base_transform.copy()]
        prefix = np.eye(4, dtype=np.float64)
        for i in range(self.dof):
            prefix = prefix @ exp_se3(self.screw_axes_space[i], float(arr[i]))
            frames.append(prefix @ self._home_joint_frames[i])
        frames.append(prefix @ self.home_matrix)
        return frames

    @property
    def _home_joint_frames(self) -> List[np.ndarray]:
        return [self.base_transform.copy() for _ in range(self.dof)]

    def link_capsules(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        attached_tip_length_m: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(p0s, p1s, radii)` in world meters `(L, 3), (L, 3), (L,)` for all robot links + tip."""
        frames = self.joint_frames_world(q)
        p0_list: List[np.ndarray] = []
        p1_list: List[np.ndarray] = []
        r_list: List[float] = []
        for cap in self.capsules:
            F = frames[cap.frame_index]
            R = F[:3, :3]
            t = F[:3, 3]
            p0_list.append(t + R @ np.asarray(cap.p0_local, dtype=np.float64))
            p1_list.append(t + R @ np.asarray(cap.p1_local, dtype=np.float64))
            r_list.append(cap.radius_m)

        if attached_tip_length_m > 1e-6:
            T_ee = frames[-1]
            p_flange = T_ee[:3, 3]
            z_tool = T_ee[:3, 2]
            p_tip = p_flange - float(attached_tip_length_m) * z_tool
            p0_list.append(p_flange)
            p1_list.append(p_tip)
            r_list.append(0.0035)

        return (
            np.stack(p0_list, axis=0),
            np.stack(p1_list, axis=0),
            np.asarray(r_list, dtype=np.float64),
        )

    def space_jacobian(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Analytical 6xn Space Jacobian `J_s(q)` in linear-first twist ordering `[v_s; w_s]`."""
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.dof,):
            raise ValueError(f"Expected joint vector of shape ({self.dof},), got {arr.shape}")
        J_s = np.zeros((6, self.dof), dtype=np.float64)
        T_prefix = np.eye(4, dtype=np.float64)
        for i in range(self.dof):
            if i == 0:
                J_s[:, 0] = self.screw_axes_space[0]
            else:
                J_s[:, i] = adjoint_se3(T_prefix) @ self.screw_axes_space[i]
            T_prefix = T_prefix @ exp_se3(self.screw_axes_space[i], float(arr[i]))
        return J_s

    def body_jacobian(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Analytical 6xn Body Jacobian `J_b(q) = Ad_{T(q)^{-1}} J_s(q)` (`[v_b; w_b]`)."""
        J_s = self.space_jacobian(q)
        T = self.fk_poe(q)
        return adjoint_se3(inverse_se3(T)) @ J_s

    def world_jacobian(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        attached_tip_length_m: float = 0.0,
    ) -> np.ndarray:
        """Analytical 6xn Geometric World-Aligned Jacobian mapping `qd -> [p_dot_world; omega_world]`."""
        J_s = self.space_jacobian(q)
        T = self.fk_poe(q)
        p_ref = T[:3, 3] - float(attached_tip_length_m) * T[:3, 2]
        out = J_s.copy()
        out[:3, :] = J_s[:3, :] - skew3(p_ref) @ J_s[3:, :]
        return out

    def manipulability(self, q: Sequence[float] | np.ndarray) -> float:
        """Yoshikawa manipulability index `w(q) = sqrt(max(0, det(J_w(q) @ J_w(q).T)))`."""
        J = self.world_jacobian(q)
        jjt = J @ J.T
        det_val = float(np.linalg.det(jjt))
        return math.sqrt(max(0.0, det_val))

    def _nullspace_gradient(
        self,
        q: np.ndarray,
        *,
        joint_limit_gain: float = 0.20,
        manipulability_gain: float = 0.05,
    ) -> np.ndarray:
        """Combined gradient moving joints toward mid-range and increasing Yoshikawa manipulability."""
        # Normalized quadratic joint-limit centering gradient
        grad_lim = -joint_limit_gain * (q - self.q_mid) / (self.q_span**2)
        if manipulability_gain <= 0.0:
            return grad_lim
        w0 = self.manipulability(q)
        eps = 1e-5
        grad_m = np.zeros(self.dof, dtype=np.float64)
        for j in range(self.dof):
            dq = np.zeros(self.dof, dtype=np.float64)
            dq[j] = eps
            grad_m[j] = (self.manipulability(q + dq) - w0) / eps
        norm_gm = float(np.linalg.norm(grad_m))
        if norm_gm > 1e-9:
            grad_m = grad_m / norm_gm
        return grad_lim + manipulability_gain * grad_m

    def dls_ik(
        self,
        target: PoseSE3 | np.ndarray,
        *,
        q_seed: Optional[Sequence[float] | np.ndarray] = None,
        attached_tip_length_m: float = 0.0,
        damping: float = 0.025,
        max_iters: int = 140,
        pos_tol_m: float = 1e-5,
        rot_tol_rad: float = 1e-4,
        nullspace_limit_gain: float = 0.25,
        nullspace_manip_gain: float = 0.04,
        multi_start: bool = True,
        collision_free_fn: Optional[ Any ] = None,
    ) -> IKResultSE3:
        """Damped Least-Squares IK with null-space joint-limit & manipulability optimization.

        When `attached_tip_length_m > 0`, `target.xyz` denotes the desired pipette tip apex
        in world coordinates while `target.rpy` denotes the flange/pipette orientation.
        """
        if isinstance(target, PoseSE3):
            T_goal_tip = target.homogeneous_matrix()
        else:
            T_goal_tip = np.asarray(target, dtype=np.float64)
            if T_goal_tip.shape != (4, 4):
                raise ValueError(f"Expected PoseSE3 or (4,4) matrix, got {T_goal_tip.shape}")

        # Convert desired tip frame to desired flange frame (+attached_tip_length_m along +z_tool)
        T_goal = T_goal_tip.copy()
        if attached_tip_length_m > 0.0:
            T_goal[:3, 3] = T_goal_tip[:3, 3] + float(attached_tip_length_m) * T_goal_tip[:3, 2]

        seeds: List[np.ndarray] = []
        if q_seed is not None:
            seeds.append(self.clamp_joints(q_seed, eps=1e-4))
        if multi_start or not seeds:
            seeds.append(self.clamp_joints(self.home_q, eps=1e-4))
            for sb in self.ik_seed_bank:
                seeds.append(self.clamp_joints(sb, eps=1e-4))

        best_res: Optional[IKResultSE3] = None
        eye6 = np.eye(6, dtype=np.float64)
        eye_n = np.eye(self.dof, dtype=np.float64)

        for s_idx, q0 in enumerate(seeds):
            q = q0.copy()
            iters_used = 0
            for it in range(1, max_iters + 1):
                iters_used = it
                T_cur = self.fk_poe(q)
                err = pose_error_world(T_cur, T_goal)
                p_err = float(np.linalg.norm(err[:3]))
                r_err = float(np.linalg.norm(err[3:]))
                if p_err <= pos_tol_m and r_err <= rot_tol_rad and self.within_joint_limits(q, tol=1e-6):
                    # Optional redundant null-space polish steps for 7-DoF arms
                    if self.dof > 6 and (nullspace_limit_gain > 0.0 or nullspace_manip_gain > 0.0):
                        for _ in range(10):
                            J = self.world_jacobian(q)
                            lam2 = (0.25 * damping) ** 2
                            J_pinv = J.T @ np.linalg.solve(J @ J.T + lam2 * eye6, eye6)
                            N = eye_n - J_pinv @ J
                            dq_null = N @ self._nullspace_gradient(
                                q,
                                joint_limit_gain=nullspace_limit_gain,
                                manipulability_gain=nullspace_manip_gain,
                            )
                            q_try = self.clamp_joints(q + 0.35 * dq_null, eps=1e-5)
                            e_try = pose_error_world(self.fk_poe(q_try), T_goal)
                            q_corr = q_try + J_pinv @ e_try
                            if self.within_joint_limits(q_corr, tol=1e-6):
                                e_corr = pose_error_world(self.fk_poe(q_corr), T_goal)
                                if float(np.linalg.norm(e_corr[:3])) <= pos_tol_m and float(
                                    np.linalg.norm(e_corr[3:])
                                ) <= rot_tol_rad:
                                    q = q_corr
                    if collision_free_fn is not None and not bool(collision_free_fn(q)):
                        break
                    return IKResultSE3(
                        success=True,
                        q=q,
                        pos_error_m=p_err,
                        rot_error_rad=r_err,
                        iterations=it,
                        manipulability=self.manipulability(q),
                        seed_index=s_idx,
                    )

                J = self.world_jacobian(q)
                err_norm2 = float(np.dot(err, err))
                # Adaptive Levenberg-Marquardt damping: strong far from target / near singularities,
                # quadratic Gauss-Newton convergence (1e-10) once close to target!
                jjt_raw = J @ J.T
                min_eig = float(np.min(np.linalg.eigvalsh(jjt_raw)))
                sing_damp = (damping * damping) if min_eig < 1e-3 else 1e-10
                lam2 = max(sing_damp, min(damping * damping, 0.5 * err_norm2))
                J_pinv = J.T @ np.linalg.solve(jjt_raw + lam2 * eye6, eye6)
                dq_task = J_pinv @ err

                # Step clamping for large Cartesian jumps
                max_dq = float(np.max(np.abs(dq_task)))
                if max_dq > 0.40:
                    dq_task = dq_task * (0.40 / max_dq)

                if self.dof > 6 and it <= max_iters - 30 and p_err > 2e-4:
                    # Exact kinematic null-space projector during approach; pure Gauss-Newton polish near convergence!
                    J_pinv_exact = J.T @ np.linalg.solve(jjt_raw + 1e-9 * eye6, eye6)
                    N = eye_n - J_pinv_exact @ J
                    g_ns = self._nullspace_gradient(
                        q,
                        joint_limit_gain=nullspace_limit_gain,
                        manipulability_gain=nullspace_manip_gain,
                    )
                    dq = dq_task + 0.35 * (N @ g_ns)
                elif min_eig < 2e-3 and p_err > 2e-4:
                    # Near singularity on 6-DoF arm: gentle joint-limit/manipulability bias
                    N = eye_n - J_pinv @ J
                    g_ns = self._nullspace_gradient(
                        q,
                        joint_limit_gain=0.10 * nullspace_limit_gain,
                        manipulability_gain=0.0,
                    )
                    dq = dq_task + N @ g_ns
                else:
                    dq = dq_task

                q = self.clamp_joints(q + dq, eps=1e-6)

            T_end = self.fk_poe(q)
            err_end = pose_error_world(T_end, T_goal)
            p_end = float(np.linalg.norm(err_end[:3]))
            r_end = float(np.linalg.norm(err_end[3:]))
            ok = bool(
                p_end <= pos_tol_m
                and r_end <= rot_tol_rad
                and self.within_joint_limits(q, tol=1e-6)
                and (collision_free_fn is None or bool(collision_free_fn(q)))
            )
            cand = IKResultSE3(
                success=ok,
                q=q,
                pos_error_m=p_end,
                rot_error_rad=r_end,
                iterations=iters_used,
                manipulability=self.manipulability(q),
                seed_index=s_idx,
            )
            if ok:
                return cand
            if best_res is None or (p_end + 0.1 * r_end) < (best_res.pos_error_m + 0.1 * best_res.rot_error_rad):
                best_res = cand

        assert best_res is not None
        return best_res

    def ik_se3(
        self,
        target: PoseSE3,
        *,
        q_seed: Optional[Sequence[float] | np.ndarray] = None,
        attached_tip_length_m: float = 0.0,
    ) -> Optional[np.ndarray]:
        res = self.dls_ik(target, q_seed=q_seed, attached_tip_length_m=attached_tip_length_m)
        return res.q.copy() if res.success else None

    def trajectory_ee_kinematics(
        self,
        t: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        *,
        attached_tip_length_m: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate exact 3D `(pos, rot_mats, lin_vel, ang_vel, lin_acc, ang_acc)` along a trajectory.

        Uses exact analytical PoE twist velocity `V = J_world(q) @ qd` and exact analytical
        spatial Lie-bracket acceleration recursion so `lin_acc` and `ang_acc` require zero
        numerical differentiation noise even on non-uniform time grids!
        """
        _ = np.asarray(t, dtype=np.float64)
        qs = np.asarray(q, dtype=np.float64)
        qds = np.asarray(qd, dtype=np.float64)
        qdds = np.asarray(qdd, dtype=np.float64)
        n_steps = int(qs.shape[0])

        pos = np.zeros((n_steps, 3), dtype=np.float64)
        rots = np.zeros((n_steps, 3, 3), dtype=np.float64)
        lin_vel = np.zeros((n_steps, 3), dtype=np.float64)
        ang_vel = np.zeros((n_steps, 3), dtype=np.float64)
        lin_acc = np.zeros((n_steps, 3), dtype=np.float64)
        ang_acc = np.zeros((n_steps, 3), dtype=np.float64)

        for k in range(n_steps):
            qk = qs[k]
            qdk = qds[k]
            qddk = qdds[k]

            # Exact PoE forward velocity & acceleration recursion in space frame:
            # V_s = sum_i J_{s,i} qd_i
            # V_s_dot = sum_i (J_{s,i} qdd_i + [V_{s,prefix_{i-1}}, J_{s,i}] qd_i)
            J_s = np.zeros((6, self.dof), dtype=np.float64)
            T_prefix = np.eye(4, dtype=np.float64)
            v_prefix = np.zeros(6, dtype=np.float64)
            vs_dot = np.zeros(6, dtype=np.float64)

            for i in range(self.dof):
                s_col = self.screw_axes_space[0] if i == 0 else (adjoint_se3(T_prefix) @ self.screw_axes_space[i])
                J_s[:, i] = s_col
                # Spatial Lie bracket [V_1, V_2] for linear-first twists [v; w]:
                # [v1, w1; v2, w2] = [w1 x v2 - w2 x v1; w1 x w2]
                v1, w1 = v_prefix[:3], v_prefix[3:]
                v2, w2 = s_col[:3], s_col[3:]
                ad_bracket = np.concatenate([np.cross(w1, v2) - np.cross(w2, v1), np.cross(w1, w2)])
                vs_dot = vs_dot + s_col * float(qddk[i]) + ad_bracket * float(qdk[i])
                v_prefix = v_prefix + s_col * float(qdk[i])
                T_prefix = T_prefix @ exp_se3(self.screw_axes_space[i], float(qk[i]))

            T_ee = T_prefix @ self.home_matrix
            R_ee = T_ee[:3, :3]
            p_ref = T_ee[:3, 3] - float(attached_tip_length_m) * R_ee[:, 2]

            vs = v_prefix[:3]
            ws = v_prefix[3:]
            vs_d = vs_dot[:3]
            ws_d = vs_dot[3:]

            # Convert spatial twist (vs, ws) & spatial acceleration (vs_d, ws_d) to material point p_ref:
            # p_dot = vs + ws x p_ref
            # p_ddot = vs_d + ws_d x p_ref + ws x p_dot
            p_dot = vs + np.cross(ws, p_ref)
            p_ddot = vs_d + np.cross(ws_d, p_ref) + np.cross(ws, p_dot)

            pos[k] = p_ref
            rots[k] = R_ee
            lin_vel[k] = p_dot
            ang_vel[k] = ws
            lin_acc[k] = p_ddot
            ang_acc[k] = ws_d

        return pos, rots, lin_vel, ang_vel, lin_acc, ang_acc

    def multi_well_accelerations_3d(
        self,
        t: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        *,
        well_offsets_xyz: Sequence[Tuple[float, float, float]],
        attached_tip_length_m: float = 0.0,
    ) -> np.ndarray:
        """Compute exact 3D effective accelerations `(T, K, 3)` in local plate coordinates across `K` wells.

        Includes:
        - 3D translational acceleration `R^T a_ee`,
        - 3D Euler angular acceleration `alpha_body x r_well`,
        - 3D centrifugal acceleration `omega_body x (omega_body x r_well)`,
        - Gravity tilt projection `R^T [0, 0, g] - [0, 0, g]` when the plate tilts off horizontal!
        """
        _, rots, _, ang_vel_w, lin_acc_w, ang_acc_w = self.trajectory_ee_kinematics(
            t,
            q,
            qd,
            qdd,
            attached_tip_length_m=attached_tip_length_m,
        )
        offsets = np.asarray(well_offsets_xyz, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise ValueError(f"well_offsets_xyz must have shape (K, 3), got {offsets.shape}")

        n_steps = rots.shape[0]
        k_wells = offsets.shape[0]
        out = np.zeros((n_steps, k_wells, 3), dtype=np.float64)
        g_world = np.array([0.0, 0.0, GRAVITY_M_S2], dtype=np.float64)

        for k in range(n_steps):
            Rt = rots[k].T
            a_body = Rt @ lin_acc_w[k]
            w_body = Rt @ ang_vel_w[k]
            alpha_body = Rt @ ang_acc_w[k]
            # Quasi-static tilt gravity shift in plate frame: when R = Rz(yaw), g_tilt == [0, 0, 0]
            g_tilt = (Rt @ g_world) - g_world
            for j in range(k_wells):
                rw = offsets[j]
                euler = np.cross(alpha_body, rw)
                centrif = np.cross(w_body, np.cross(w_body, rw))
                out[k, j] = a_body + euler + centrif + g_tilt
        return out


class UR5eArm(SerialChainSE3):
    """6-DoF Universal Robots UR5e cobot with exact manufacturer Standard DH and PoE chains.

    Includes standard `Rx(pi)` micropipette flange adapter (`_tool_mount`) so overhead elbow-up
    cobot postures (`z > 0` above benchtop) map directly to `PoseSE3(rpy=(0, 0, yaw))`
    (`tool_tilt_rad == 0`, pipette tip pointing vertically down `-z_world`).
    `fk_dh_iso(q)` returns the unrotated ISO-9409-1-50-4-M6 flange, while `fk_dh(q) == fk_poe(q)`
    includes the micropipette adapter.
    """

    # Official Universal Robots UR5e Standard DH parameters (meters, radians)
    DH_A: Tuple[float, ...] = (0.0, -0.4250, -0.3922, 0.0, 0.0, 0.0)
    DH_D: Tuple[float, ...] = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
    DH_ALPHA: Tuple[float, ...] = (0.5 * math.pi, 0.0, 0.0, 0.5 * math.pi, -0.5 * math.pi, 0.0)

    def __init__(
        self,
        *,
        base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_limits: Optional[Sequence[Tuple[float, float]]] = None,
        margin_m: float = 0.005,
    ) -> None:
        # Standard Rx(pi) pipette tool adapter aligning downward ISO flange (+z_iso = -z_world)
        # with lab_hands PoseSE3 convention (+z_tool = +z_world, tip along -z_tool = -z_world).
        self._tool_mount = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

        screws: List[np.ndarray] = []
        home_frames_local: List[np.ndarray] = []
        M_prefix = np.eye(4, dtype=np.float64)
        ez_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        for a_i, d_i, al_i in zip(self.DH_A, self.DH_D, self.DH_ALPHA):
            home_frames_local.append(M_prefix.copy())
            screws.append(adjoint_se3(M_prefix) @ ez_twist)
            M_prefix = M_prefix @ _standard_dh_matrix(a_i, d_i, al_i, 0.0)
        M_total = M_prefix @ self._tool_mount
        self._ur5e_home_joint_frames_local = home_frames_local

        limits = (
            tuple((-2.0 * math.pi, 2.0 * math.pi) for _ in range(6))
            if joint_limits is None
            else tuple((float(lo), float(hi)) for lo, hi in joint_limits)
        )
        # Manufacturer UR5e bounds: 180 deg/s = pi rad/s, 8..14 rad/s^2, 45..80 rad/s^3 jerk
        qd_max = (math.pi,) * 6
        qdd_max = (8.0, 8.0, 10.0, 12.0, 12.0, 14.0)
        qddd_max = (45.0, 45.0, 55.0, 70.0, 70.0, 80.0)

        # 3D multi-link capsules in joint frames (all above z = 0 in overhead elbow-up configurations)
        capsules = (
            LinkCapsuleSpec("shoulder_column", 1, (0.0, 0.0, 0.04), (0.0, 0.0, 0.1625), 0.045),
            LinkCapsuleSpec("upper_arm", 2, (0.0, 0.0, 0.0), (-0.4250, 0.0, 0.0), 0.038),
            LinkCapsuleSpec("forearm", 3, (0.0, 0.0, 0.0), (-0.3922, 0.0, 0.0), 0.032),
            LinkCapsuleSpec("wrist_1_2", 4, (0.0, 0.0, 0.0), (0.0, 0.0, 0.10), 0.026),
            LinkCapsuleSpec("wrist_flange", 6, (0.0, 0.0, 0.02), (0.0, 0.0, 0.095), 0.022),
        )

        # Overhead elbow-up downward-pipette UR5e configurations (z_elbow ~ 0.45..0.58 m above deck!)
        home_q = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
        seed_bank = (
            (0.0, -1.40, 1.85, -2.02, -1.5708, 0.0),
            (1.20, -1.40, 1.85, -2.02, -1.5708, 1.20),
            (-0.60, -1.40, 1.85, -2.02, -1.5708, -0.60),
            (1.85, -1.40, 1.85, -2.02, -1.5708, 1.85),
            (-1.40, -1.40, 1.85, -2.02, -1.5708, -1.40),
            (2.50, -1.40, 1.85, -2.02, -1.5708, 2.50),
        )

        super().__init__(
            name="ur5e",
            screw_axes_space=np.stack(screws, axis=0),
            home_matrix=M_total,
            joint_kinds=("revolute",) * 6,
            joint_limits=limits,
            qd_max=qd_max,
            qdd_max=qdd_max,
            qddd_max=qddd_max,
            capsules=capsules,
            home_q=home_q,
            ik_seed_bank=seed_bank,
            base_offset_xyz=base_offset_xyz,
            margin_m=margin_m,
        )

    @property
    def _home_joint_frames(self) -> List[np.ndarray]:
        return [self.base_transform @ f for f in self._ur5e_home_joint_frames_local]

    def fk_dh(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Exact Standard-DH forward kinematics verified bit-for-bit against `fk_poe(q)`."""
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (6,):
            raise ValueError(f"Expected 6-DoF UR5e joint vector, got {arr.shape}")
        T = self.base_transform.copy()
        for a_i, d_i, al_i, q_i in zip(self.DH_A, self.DH_D, self.DH_ALPHA, arr):
            T = T @ _standard_dh_matrix(a_i, d_i, al_i, float(q_i))
        return T @ self._tool_mount


class PandaArm(SerialChainSE3):
    """7-DoF Franka Emika Panda cobot with exact Craig Modified-DH and PoE kinematics."""

    # Official Franka Control Interface (FCI) Craig Modified-DH parameters
    MDH_A_PREV: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.0880)
    MDH_D: Tuple[float, ...] = (0.3330, 0.0, 0.3160, 0.0, 0.3840, 0.0, 0.0)
    MDH_ALPHA_PREV: Tuple[float, ...] = (
        0.0,
        -0.5 * math.pi,
        0.5 * math.pi,
        0.5 * math.pi,
        -0.5 * math.pi,
        0.5 * math.pi,
        0.5 * math.pi,
    )
    FLANGE_D_M: float = 0.1070

    DEFAULT_LIMITS: Tuple[Tuple[float, float], ...] = (
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    )

    def __init__(
        self,
        *,
        base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_limits: Optional[Sequence[Tuple[float, float]]] = None,
        margin_m: float = 0.005,
    ) -> None:
        flange_tf = np.eye(4, dtype=np.float64)
        flange_tf[2, 3] = self.FLANGE_D_M
        # Align downward Panda flange (+z_iso = -z_world) with PoseSE3 (+z_tool = +z_world)
        self._iso_flange_tf = flange_tf.copy()
        self._flange_tf = flange_tf @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

        screws: List[np.ndarray] = []
        home_frames_local: List[np.ndarray] = []
        M_prefix = np.eye(4, dtype=np.float64)
        ez_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        for a_p, d_i, al_p in zip(self.MDH_A_PREV, self.MDH_D, self.MDH_ALPHA_PREV):
            # In MDH: A_i(q_i) = Rx(alpha_{i-1}) Tx(a_{i-1}) * Rz(q_i) * Tz(d_i)
            B_i = _modified_dh_matrix(a_p, 0.0, al_p, 0.0)
            axis_frame = M_prefix @ B_i
            home_frames_local.append(axis_frame.copy())
            screws.append(adjoint_se3(axis_frame) @ ez_twist)
            M_prefix = M_prefix @ _modified_dh_matrix(a_p, d_i, al_p, 0.0)

        M_total = M_prefix @ self._flange_tf
        self._panda_home_joint_frames_local = home_frames_local

        limits = self.DEFAULT_LIMITS if joint_limits is None else tuple((float(lo), float(hi)) for lo, hi in joint_limits)
        # Official Franka Emika Panda FCI velocity, acceleration, and jerk bounds
        qd_max = (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100)
        qdd_max = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)
        qddd_max = (7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0)

        capsules = (
            LinkCapsuleSpec("link1_column", 1, (0.0, 0.0, 0.05), (0.0, 0.0, 0.333), 0.050),
            LinkCapsuleSpec("upper_arm", 3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.316), 0.042),
            LinkCapsuleSpec("forearm", 5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.384), 0.036),
            LinkCapsuleSpec("wrist_flange", 7, (0.0, 0.0, 0.0), (0.0, 0.0, 0.107), 0.028),
        )

        home_q = (0.0, -0.55, 0.0, -2.15, 0.0, 1.60, 0.7854)
        seed_bank = (
            (0.0, -0.55, 0.0, -2.25, 0.0, 1.70, 0.0),
            (-0.10, -1.25, 2.10, -2.40, -1.30, 1.00, -0.65),
            (0.10, -1.25, -2.10, -2.40, 1.30, 1.00, 0.65),
            (0.85, -0.55, 0.35, -2.20, 0.0, 1.65, 0.85),
            (-0.65, -0.55, -0.35, -2.20, 0.0, 1.65, -0.65),
            (1.45, -0.50, 0.80, -2.15, -0.30, 1.65, 1.20),
        )

        super().__init__(
            name="panda",
            screw_axes_space=np.stack(screws, axis=0),
            home_matrix=M_total,
            joint_kinds=("revolute",) * 7,
            joint_limits=limits,
            qd_max=qd_max,
            qdd_max=qdd_max,
            qddd_max=qddd_max,
            capsules=capsules,
            home_q=home_q,
            ik_seed_bank=seed_bank,
            base_offset_xyz=base_offset_xyz,
            margin_m=margin_m,
        )

    @property
    def _home_joint_frames(self) -> List[np.ndarray]:
        return [self.base_transform @ f for f in self._panda_home_joint_frames_local]

    def fk_mdh(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Exact Craig Modified-DH forward kinematics verified bit-for-bit against `fk_poe(q)`."""
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (7,):
            raise ValueError(f"Expected 7-DoF Panda joint vector, got {arr.shape}")
        T = self.base_transform.copy()
        for a_p, d_i, al_p, q_i in zip(self.MDH_A_PREV, self.MDH_D, self.MDH_ALPHA_PREV, arr):
            T = T @ _modified_dh_matrix(a_p, d_i, al_p, float(q_i))
        return T @ self._flange_tf


class OpentronsFlexGantry(SerialChainSE3):
    """4-DoF Opentrons Flex / OT-2 3-axis Cartesian gantry + rotary gripper PoE kinematic chain.

    Joint vector `q = (x_m, y_m, z_m, yaw_rad)` in SI meters & radians over the canonical
    Opentrons deck envelope `(0..0.418, 0..0.353, 0..0.218) m` x `[-pi, pi] rad`.
    """

    DECK_BOUNDS_MM: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.0, 418.0),
        (0.0, 353.0),
        (0.0, 218.0),
    )

    def __init__(
        self,
        *,
        base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        margin_m: float = 0.005,
    ) -> None:
        screws = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Prismatic X
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Prismatic Y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Prismatic Z
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Rotary gripper paddle Z-yaw
            ],
            dtype=np.float64,
        )
        home = np.eye(4, dtype=np.float64)
        limits = (
            (0.0, 0.418),
            (0.0, 0.353),
            (0.0, 0.218),
            (-math.pi, math.pi),
        )
        qd_max = (0.60, 0.50, 0.30, 3.0)
        qdd_max = (2.5, 2.5, 1.5, 10.0)
        qddd_max = (25.0, 25.0, 15.0, 60.0)

        capsules = (
            LinkCapsuleSpec("pipette_mount", 4, (0.0, 0.0, 0.0), (0.0, 0.0, 0.14), 0.018),
        )
        home_q = (0.209, 0.1765, 0.180, 0.0)
        super().__init__(
            name="opentrons_flex",
            screw_axes_space=screws,
            home_matrix=home,
            joint_kinds=("prismatic", "prismatic", "prismatic", "revolute"),
            joint_limits=limits,
            qd_max=qd_max,
            qdd_max=qdd_max,
            qddd_max=qddd_max,
            capsules=capsules,
            home_q=home_q,
            base_offset_xyz=base_offset_xyz,
            margin_m=margin_m,
        )

    def analytical_ik(
        self,
        target: PoseSE3,
        *,
        attached_tip_length_m: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Exact closed-form 4-DoF gantry + yaw IK (returns `None` if tilted or out of deck bounds)."""
        if abs(target.rpy[0]) > 1e-6 or abs(target.rpy[1]) > 1e-6:
            return None
        p_flange = np.asarray(target.xyz, dtype=np.float64) - np.asarray(self.base_offset_xyz, dtype=np.float64)
        p_flange[2] += float(attached_tip_length_m)
        q = np.array(
            [float(p_flange[0]), float(p_flange[1]), float(p_flange[2]), float(wrap_angle(target.rpy[2]))],
            dtype=np.float64,
        )
        return q if self.within_joint_limits(q) else None


def make_ur5e_arm(
    *,
    base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    margin_m: float = 0.005,
) -> UR5eArm:
    return UR5eArm(base_offset_xyz=base_offset_xyz, margin_m=margin_m)


def make_panda_arm(
    *,
    base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    margin_m: float = 0.005,
) -> PandaArm:
    return PandaArm(base_offset_xyz=base_offset_xyz, margin_m=margin_m)


def make_opentrons_flex(
    *,
    base_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    margin_m: float = 0.005,
) -> OpentronsFlexGantry:
    return OpentronsFlexGantry(base_offset_xyz=base_offset_xyz, margin_m=margin_m)


DEFAULT_UR5E = UR5eArm()
DEFAULT_PANDA = PandaArm()
DEFAULT_OPENTRONS_FLEX = OpentronsFlexGantry()

__all__ = [
    "DEFAULT_OPENTRONS_FLEX",
    "DEFAULT_PANDA",
    "DEFAULT_UR5E",
    "IKResultSE3",
    "JointKind",
    "LinkCapsuleSpec",
    "OpentronsFlexGantry",
    "PandaArm",
    "SerialChainSE3",
    "UR5eArm",
    "adjoint_se3",
    "exp_se3",
    "exp_so3",
    "inverse_se3",
    "log_se3",
    "log_so3",
    "make_opentrons_flex",
    "make_panda_arm",
    "make_ur5e_arm",
    "pose_error_world",
    "skew3",
    "unskew3",
]
