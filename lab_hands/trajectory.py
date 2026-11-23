"""Droplet-safe C2 time parameterization, 2.5D Z-clearance transfers, and JSONL trajectory schema."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .collision import (
    DEFAULT_SCENE,
    BenchScene,
    batch_in_collision,
    edge_free,
    edge_free_3d,
    path_min_clearance,
)
from .kinematics import DEFAULT_ARM, Arm2D
from .planner import ik_solutions, path_length, rrt, rrt_connect, shortcut

TRAJECTORY_SCHEMA = "lab-hands/trajectory/v1"


@dataclass(frozen=True)
class TimeParameterizedTrajectory:
    """Smooth C2 rest-to-rest trajectory satisfying joint velocity and Cartesian tip slosh bounds."""

    t: np.ndarray  # (N,)
    q: np.ndarray  # (N, 2)
    qd: np.ndarray  # (N, 2)
    qdd: np.ndarray  # (N, 2)
    tip_xy: np.ndarray  # (N, 2)
    tip_vel: np.ndarray  # (N, 2)
    tip_acc: np.ndarray  # (N, 2)
    duration_s: float
    max_joint_vel_rad_s: float
    max_joint_acc_rad_s2: float
    max_tip_acc_m_s2: float


def _build_blended_geometric_curve(
    waypoints: List[np.ndarray],
    arm: Arm2D,
    scene: BenchScene,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct C2-continuous geometric path q(s), dq/ds, d2q/ds2 for s in [0, S_total]."""
    seg_vecs = [waypoints[k + 1] - waypoints[k] for k in range(len(waypoints) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    tangents = [v / max(L, 1e-12) for v, L in zip(seg_vecs, seg_lens)]
    m = len(seg_lens)

    # Start with up to 25% corner blend radius per segment; halve if any blend collides
    radii = [0.0] * (m + 1)
    for k in range(1, m):
        radii[k] = min(0.25 * seg_lens[k - 1], 0.25 * seg_lens[k], 0.08)

    for _ in range(12):
        pieces = []  # (kind, length_s, data)
        for k in range(m):
            r_in = radii[k]
            r_out = radii[k + 1]
            lin_len = max(0.0, seg_lens[k] - r_in - r_out)
            a_lin = waypoints[k] + r_in * tangents[k]
            if lin_len > 1e-12:
                pieces.append(("line", lin_len, (a_lin, tangents[k])))
            if k < m - 1 and radii[k + 1] > 1e-9:
                rk = radii[k + 1]
                a_b = waypoints[k + 1] - rk * tangents[k]
                delta = rk * (tangents[k + 1] - tangents[k])
                pieces.append(("blend", 2.0 * rk, (a_b, tangents[k], delta, rk)))

        s_total = sum(p[1] for p in pieces)
        s_grid = np.linspace(0.0, s_total, n_samples, dtype=np.float64)
        q_s = np.zeros((n_samples, 2), dtype=np.float64)
        qp_s = np.zeros((n_samples, 2), dtype=np.float64)
        qpp_s = np.zeros((n_samples, 2), dtype=np.float64)

        cursor = 0.0
        for idx_p, (kind, plen, pdata) in enumerate(pieces):
            s_lo = cursor
            s_hi = cursor + plen
            mask = (s_grid >= s_lo) & (s_grid <= s_hi if idx_p == len(pieces) - 1 else s_grid < s_hi)
            loc = s_grid[mask] - s_lo
            if kind == "line":
                a_pt, u_vec = pdata
                q_s[mask] = a_pt[None, :] + loc[:, None] * u_vec[None, :]
                qp_s[mask] = u_vec[None, :]
                qpp_s[mask] = 0.0
            else:
                a_b, u_prev, delta, rk = pdata
                tau = loc / (2.0 * rk)
                v0 = 2.0 * rk * u_prev
                q_s[mask] = (
                    a_b[None, :]
                    + tau[:, None] * v0[None, :]
                    + (2.0 * tau**3 - tau**4)[:, None] * delta[None, :]
                )
                # dq/ds = (dq/dtau) / (2*rk)
                qp_s[mask] = u_prev[None, :] + ((6.0 * tau**2 - 4.0 * tau**3) / (2.0 * rk))[:, None] * delta[None, :]
                # d2q/ds2 = (d2q/dtau2) / (2*rk)^2
                qpp_s[mask] = ((12.0 * tau * (1.0 - tau)) / (4.0 * rk * rk))[:, None] * delta[None, :]
            cursor = s_hi

        if not bool(np.any(batch_in_collision(q_s, arm=arm, scene=scene))):
            return q_s, qp_s, qpp_s, s_total
        radii = [0.5 * r for r in radii]

    # Fallback: zero-radius stop-and-go segments (guaranteed identical to collision-free waypoints)
    radii = [0.0] * (m + 1)
    s_total = sum(seg_lens)
    s_grid = np.linspace(0.0, s_total, n_samples, dtype=np.float64)
    q_s = np.zeros((n_samples, 2), dtype=np.float64)
    qp_s = np.zeros((n_samples, 2), dtype=np.float64)
    qpp_s = np.zeros((n_samples, 2), dtype=np.float64)
    cursor = 0.0
    for k in range(m):
        s_lo, s_hi = cursor, cursor + seg_lens[k]
        mask = (s_grid >= s_lo) & (s_grid <= s_hi if k == m - 1 else s_grid < s_hi)
        loc = s_grid[mask] - s_lo
        q_s[mask] = waypoints[k][None, :] + loc[:, None] * tangents[k][None, :]
        qp_s[mask] = tangents[k][None, :]
        cursor = s_hi
    return q_s, qp_s, qpp_s, s_total


def parameterize_path(
    path: Sequence[np.ndarray],
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    qd_max: float = 1.5,
    qdd_max: float = 4.0,
    a_max_liquid: float = 1.5,
    n_samples: int = 401,
) -> TimeParameterizedTrajectory:
    """Time-parameterize a waypoint path with C2 minimum-jerk profile and slosh-safe tip acceleration."""
    if not path:
        raise ValueError("Cannot parameterize empty path")
    if qd_max <= 0.0 or qdd_max <= 0.0 or a_max_liquid <= 0.0:
        raise ValueError("Velocity and acceleration bounds must be > 0")

    pts: List[np.ndarray] = []
    for p in path:
        arr = np.asarray(p, dtype=np.float64).copy()
        if not pts or float(np.linalg.norm(arr - pts[-1])) > 1e-10:
            pts.append(arr)

    if len(pts) == 1:
        q = pts[0][None, :]
        z = np.zeros((1, 2), dtype=np.float64)
        tip_xy, tip_vel, tip_acc = arm.batch_tip_kinematics(q, z, z)
        return TimeParameterizedTrajectory(
            t=np.zeros(1, dtype=np.float64),
            q=q,
            qd=z,
            qdd=z,
            tip_xy=tip_xy,
            tip_vel=tip_vel,
            tip_acc=tip_acc,
            duration_s=0.0,
            max_joint_vel_rad_s=0.0,
            max_joint_acc_rad_s2=0.0,
            max_tip_acc_m_s2=0.0,
        )

    # Evaluate geometric path on dense arc-parameter grid, then apply minimum-jerk warping
    dense_n = max(1001, n_samples * 3)
    q_geom, qp_geom, qpp_geom, s_total = _build_blended_geometric_curve(pts, arm=arm, scene=scene, n_samples=dense_n)
    s_dense = np.linspace(0.0, s_total, dense_n, dtype=np.float64)

    sigma = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    # Minimum-jerk polynomial u(sigma) in [0, 1]
    u = 10.0 * sigma**3 - 15.0 * sigma**4 + 6.0 * sigma**5
    up = 30.0 * sigma**2 - 60.0 * sigma**3 + 30.0 * sigma**4
    upp = 60.0 * sigma - 180.0 * sigma**2 + 120.0 * sigma**3

    s_eval = s_total * u
    ds_dsigma = s_total * up
    d2s_dsigma2 = s_total * upp

    q = np.column_stack([np.interp(s_eval, s_dense, q_geom[:, j]) for j in range(2)])
    qp = np.column_stack([np.interp(s_eval, s_dense, qp_geom[:, j]) for j in range(2)])
    qpp = np.column_stack([np.interp(s_eval, s_dense, qpp_geom[:, j]) for j in range(2)])

    # Unit-duration (T = 1 s) derivatives: dq/dsigma and d2q/dsigma2
    qd_unit = qp * ds_dsigma[:, None]
    qdd_unit = qpp * (ds_dsigma[:, None] ** 2) + qp * d2s_dsigma2[:, None]
    _, _, tip_acc_unit = arm.batch_tip_kinematics(q, qd_unit, qdd_unit)

    peak_qd = float(np.max(np.abs(qd_unit)))
    peak_qdd = float(np.max(np.abs(qdd_unit)))
    peak_tip_acc = float(np.max(np.linalg.norm(tip_acc_unit, axis=-1)))

    # Exact scaling: qd ~ 1/T, qdd ~ 1/T^2, tip_acc ~ 1/T^2
    duration = max(
        1e-3,
        peak_qd / qd_max,
        math.sqrt(peak_qdd / qdd_max),
        math.sqrt(peak_tip_acc / a_max_liquid),
    ) * (1.0 + 1e-6)

    t = sigma * duration
    qd = qd_unit / duration
    qdd = qdd_unit / (duration * duration)
    tip_xy, tip_vel, tip_acc = arm.batch_tip_kinematics(q, qd, qdd)

    return TimeParameterizedTrajectory(
        t=t,
        q=q,
        qd=qd,
        qdd=qdd,
        tip_xy=tip_xy,
        tip_vel=tip_vel,
        tip_acc=tip_acc,
        duration_s=float(duration),
        max_joint_vel_rad_s=float(np.max(np.abs(qd))),
        max_joint_acc_rad_s2=float(np.max(np.abs(qdd))),
        max_tip_acc_m_s2=float(np.max(np.linalg.norm(tip_acc, axis=-1))),
    )


def plan_transfer_3d(
    well: str,
    rng: np.random.Generator,
    *,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    planner: str = "rrt_connect",
    z_aspirate: Optional[float] = None,
    z_dispense: Optional[float] = None,
) -> Optional[List[np.ndarray]]:
    """Plan a 3-DoF (q1, q2, z) retract -> planar transit -> plunge sequence free of rim collisions."""
    starts = ik_solutions(scene.tube_xy, arm=arm, scene=scene)
    target_xy = scene.plate.well_xy(well)
    goals = ik_solutions(target_xy, arm=arm, scene=scene)
    if not starts or not goals:
        return None

    q_start = starts[0]
    solve_fn = rrt_connect if planner == "rrt_connect" else rrt
    raw_2d = solve_fn(q_start, goals, rng, arm=arm, scene=scene)
    if raw_2d is None:
        return None
    smooth_2d = shortcut(raw_2d, rng, arm=arm, scene=scene)

    z_asp = scene.z_tube_bottom + 0.002 if z_aspirate is None else float(z_aspirate)
    z_disp = scene.plate.z_bottom_m + 0.002 if z_dispense is None else float(z_dispense)
    z_safe = scene.z_safe_travel

    waypoints_3d: List[np.ndarray] = [
        np.array([q_start[0], q_start[1], z_asp], dtype=np.float64),
    ]
    for q in smooth_2d:
        waypoints_3d.append(np.array([q[0], q[1], z_safe], dtype=np.float64))
    q_goal = smooth_2d[-1]
    waypoints_3d.append(np.array([q_goal[0], q_goal[1], z_disp], dtype=np.float64))

    for a3, b3 in zip(waypoints_3d, waypoints_3d[1:]):
        if not edge_free_3d(a3, b3, arm=arm, scene=scene, allowed_well_xy=target_xy):
            return None
    return waypoints_3d


@dataclass
class TrajectoryRecord:
    """Machine-checkable trajectory artifact for lab-rehearsal and lab-verify."""

    trial: int
    seed: int
    planner: str
    start_source: str
    goal_well: str
    elbow_branch: int
    joint_waypoints: List[List[float]]
    cartesian_tip_path: List[List[float]]
    raw_length_rad: float
    smooth_length_rad: float
    min_clearance_m: float
    duration_s: float
    max_tip_acc_m_s2: float
    plan_time_ms: float
    joint_limits_satisfied: bool
    collision_free: bool
    schema: str = field(default=TRAJECTORY_SCHEMA)

    def validate(self) -> None:
        if self.schema != TRAJECTORY_SCHEMA:
            raise ValueError(f"Unsupported schema {self.schema!r}, expected {TRAJECTORY_SCHEMA!r}")
        if self.elbow_branch not in (1, -1):
            raise ValueError(f"elbow_branch must be +1 or -1, got {self.elbow_branch!r}")
        if len(self.joint_waypoints) < 1 or len(self.joint_waypoints) != len(self.cartesian_tip_path):
            raise ValueError("joint_waypoints and cartesian_tip_path must be non-empty and equal length")
        for val_name, val in (
            ("raw_length_rad", self.raw_length_rad),
            ("smooth_length_rad", self.smooth_length_rad),
            ("min_clearance_m", self.min_clearance_m),
            ("duration_s", self.duration_s),
            ("max_tip_acc_m_s2", self.max_tip_acc_m_s2),
            ("plan_time_ms", self.plan_time_ms),
        ):
            if not math.isfinite(float(val)) or float(val) < 0.0:
                raise ValueError(f"{val_name} must be finite and >= 0, got {val!r}")
        if not self.joint_limits_satisfied or not self.collision_free:
            raise ValueError("TrajectoryRecord must satisfy joint limits and collision freedom")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "TrajectoryRecord":
        obj = cls(**json.loads(line))
        obj.validate()
        return obj


def build_trajectory_record(
    *,
    trial: int,
    seed: int,
    planner: str,
    goal_well: str,
    raw_path: Sequence[np.ndarray],
    smooth_path: Sequence[np.ndarray],
    plan_time_s: float,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    start_source: str = "tube",
) -> TrajectoryRecord:
    q_end = np.asarray(smooth_path[-1], dtype=np.float64)
    elbow_branch = 1 if q_end[1] >= 0.0 else -1
    _, tip_pts = arm.batch_fk(np.asarray(smooth_path, dtype=np.float64))
    min_cl = path_min_clearance(smooth_path, arm=arm, scene=scene)
    in_lim = all(arm.within_joint_limits(q) for q in smooth_path)
    is_free = (
        (not in_lim and False)
        or (
            in_lim
            and all(edge_free(a, b, arm=arm, scene=scene) for a, b in zip(smooth_path, smooth_path[1:]))
        )
    )
    tp = parameterize_path(smooth_path, arm=arm, scene=scene)
    rec = TrajectoryRecord(
        trial=trial,
        seed=seed,
        planner=planner,
        start_source=start_source,
        goal_well=goal_well,
        elbow_branch=elbow_branch,
        joint_waypoints=[[round(float(q[0]), 6), round(float(q[1]), 6)] for q in smooth_path],
        cartesian_tip_path=[[round(float(p[0]), 6), round(float(p[1]), 6)] for p in tip_pts],
        raw_length_rad=round(path_length(raw_path), 6),
        smooth_length_rad=round(path_length(smooth_path), 6),
        min_clearance_m=round(min_cl, 6),
        duration_s=round(tp.duration_s, 6),
        max_tip_acc_m_s2=round(tp.max_tip_acc_m_s2, 6),
        plan_time_ms=round(plan_time_s * 1000.0, 3),
        joint_limits_satisfied=bool(in_lim),
        collision_free=bool(is_free),
    )
    rec.validate()
    return rec
