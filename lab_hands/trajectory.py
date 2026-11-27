"""Droplet-safe C2 time parameterization, 3D/SE(3) Z-clearance transfers, and JSONL trajectory schema."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .collision import (
    DEFAULT_SCENE,
    BenchScene,
    edge_free,
    edge_free_3d,
    in_collision,
    path_min_clearance,
)
from .kinematics import DEFAULT_ARM, DEFAULT_ARM_SE3, Arm2D, ArmSE3, PoseSE3
from .labware import DeckLayout, PlateGeometry, parse_well_ref
from .planner import ik_solutions, path_length, rrt, rrt_connect, select_ik_branch, shortcut

TRAJECTORY_SCHEMA = "lab-hands/trajectory/v1"
SLOSH_GRAVITY_M_S2 = 9.80665
SLOSH_BESSEL_XI_11 = 1.8411837813406593
MIN_GRAVITY_SLOSH_BOND_NUMBER = 5.0
_GRAVITY_M_S2 = SLOSH_GRAVITY_M_S2
_BESSEL_XI_11 = SLOSH_BESSEL_XI_11


def slosh_natural_frequency(
    well_radius_m: float,
    fill_height_m: float,
    *,
    gravity_m_s2: float = SLOSH_GRAVITY_M_S2,
    xi_11: float = SLOSH_BESSEL_XI_11,
) -> float:
    """First antisymmetric cylindrical gravity-wave natural frequency (`rad/s`):

        omega_11 = sqrt((xi_11 * g / R) * tanh(xi_11 * h / R))

    Single canonical dispersion implementation shared across `trajectory.py`,
    `datasets.py`, `slosh_dynamics.py`, and `community_extensions/slosh_minimum_jerk.py`.
    """
    r = float(well_radius_m)
    h = float(fill_height_m)
    g = float(gravity_m_s2)
    xi = float(xi_11)
    if r <= 0.0 or h <= 0.0:
        raise ValueError(f"well_radius_m and fill_height_m must be strictly positive, got R={r}, h={h}")
    if g <= 0.0 or xi <= 0.0:
        raise ValueError("gravity_m_s2 and xi_11 must be strictly positive")
    return math.sqrt((xi * g / r) * math.tanh(xi * h / r))


def slosh_bond_number(
    well_radius_m: float,
    surface_tension_n_m: float = 0.072,
    density_kg_m3: float = 1000.0,
    *,
    gravity_m_s2: float = SLOSH_GRAVITY_M_S2,
) -> float:
    """Dimensionless Bond number `Bo = rho * g * R^2 / sigma` comparing gravity to surface tension.

    A flat-surface pure-gravity linear Bessel slosh model requires `Bo >= MIN_GRAVITY_SLOSH_BOND_NUMBER`
    (`5.0`) so gravitational restoring pressure dominates capillary meniscus curvature (`Bo ~ 1.6` for
    water in a 96-well plate and `Bo ~ 0.44` in a 384-well plate -> `SLOSH_MODEL_OUT_OF_RANGE`).
    """
    r = float(well_radius_m)
    sigma = float(surface_tension_n_m)
    rho = float(density_kg_m3)
    g = float(gravity_m_s2)
    if r <= 0.0 or sigma <= 0.0 or rho <= 0.0 or g <= 0.0:
        raise ValueError("well_radius_m, surface_tension_n_m, density_kg_m3, and gravity_m_s2 must be > 0")
    return float(rho * g * (r * r) / sigma)


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
    bond_number: Optional[float] = None
    slosh_model_status: str = "NOT_APPLICABLE"


@dataclass(frozen=True)
class PickPlacePlan3D:
    """3D / 6-DOF SE(3) vertical-horizontal-vertical transfer plan over labware rims and tips."""

    source_ref: str
    target_ref: str
    attached_tip_length_m: float
    z_safe_transit_m: float
    z_pick_m: float
    z_place_m: float
    waypoints_3d: List[np.ndarray]
    poses_se3: List[PoseSE3]
    q6_waypoints: Optional[List[np.ndarray]]
    max_tool_tilt_rad: float
    collision_free: bool


CurvePiece = Tuple[str, float, Tuple[Any, ...]]


def _eval_blended_pieces(
    pieces: Sequence[CurvePiece],
    s_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate C2-blended geometric curve q(s), dq/ds, d2q/ds2 analytically at exact `s_eval`."""
    n = s_eval.size
    q_s = np.zeros((n, 2), dtype=np.float64)
    qp_s = np.zeros((n, 2), dtype=np.float64)
    qpp_s = np.zeros((n, 2), dtype=np.float64)

    cursor = 0.0
    n_pieces = len(pieces)
    for idx_p, (kind, plen, pdata) in enumerate(pieces):
        s_lo = cursor
        s_hi = cursor + plen
        if idx_p == n_pieces - 1:
            mask = (s_eval >= s_lo - 1e-12) & (s_eval <= s_hi + 1e-12)
        else:
            mask = (s_eval >= s_lo - 1e-12) & (s_eval < s_hi)
        if not bool(np.any(mask)):
            cursor = s_hi
            continue
        loc = np.clip(s_eval[mask] - s_lo, 0.0, plen)
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
            qp_s[mask] = u_prev[None, :] + ((6.0 * tau**2 - 4.0 * tau**3) / (2.0 * rk))[:, None] * delta[None, :]
            qpp_s[mask] = ((12.0 * tau * (1.0 - tau)) / (4.0 * rk * rk))[:, None] * delta[None, :]
        cursor = s_hi

    return q_s, qp_s, qpp_s


def _pieces_collision_free(pieces: Sequence[CurvePiece], arm: Arm2D, scene: BenchScene) -> bool:
    for kind, plen, pdata in pieces:
        if kind == "line":
            a_pt, u_vec = pdata
            b_pt = a_pt + plen * u_vec
            if not edge_free(a_pt, b_pt, arm=arm, scene=scene):
                return False
        else:
            s_sub = np.linspace(0.0, plen, 17, dtype=np.float64)
            q_sub, _, _ = _eval_blended_pieces([(kind, plen, pdata)], s_sub)
            for qa, qb in zip(q_sub, q_sub[1:]):
                if not edge_free(qa, qb, arm=arm, scene=scene):
                    return False
    return True


def _build_blended_geometric_pieces(
    waypoints: List[np.ndarray],
    arm: Arm2D,
    scene: BenchScene,
) -> Tuple[List[CurvePiece], float]:
    """Construct collision-free C2-blended piecewise polynomial pieces and total arc parameter."""
    seg_vecs = [waypoints[k + 1] - waypoints[k] for k in range(len(waypoints) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    tangents = [v / max(L, 1e-12) for v, L in zip(seg_vecs, seg_lens)]
    m = len(seg_lens)

    radii = [0.0] * (m + 1)
    for k in range(1, m):
        radii[k] = min(0.25 * seg_lens[k - 1], 0.25 * seg_lens[k], 0.08)

    for _ in range(12):
        pieces: List[CurvePiece] = []
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

        s_total = float(sum(p[1] for p in pieces))
        if _pieces_collision_free(pieces, arm=arm, scene=scene):
            return pieces, s_total
        radii = [0.5 * r for r in radii]

    fallback: List[CurvePiece] = [
        ("line", seg_lens[k], (waypoints[k], tangents[k])) for k in range(m) if seg_lens[k] > 1e-12
    ]
    return fallback, float(sum(p[1] for p in fallback))


def _build_blended_geometric_curve(
    waypoints: List[np.ndarray],
    arm: Arm2D,
    scene: BenchScene,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct C2-continuous geometric path q(s), dq/ds, d2q/ds2 on a uniform arc grid."""
    pieces, s_total = _build_blended_geometric_pieces(waypoints, arm=arm, scene=scene)
    s_grid = np.linspace(0.0, s_total, n_samples, dtype=np.float64)
    q_s, qp_s, qpp_s = _eval_blended_pieces(pieces, s_grid)
    return q_s, qp_s, qpp_s, s_total


def _cylindrical_slosh_peak_rise_m(
    q: np.ndarray,
    qd_unit: np.ndarray,
    qdd_unit: np.ndarray,
    tip_acc_unit: np.ndarray,
    duration_s: float,
    *,
    well_radius_m: float,
    fill_height_m: float,
    damping_ratio: float,
    offsets_xy: Sequence[Tuple[float, float]],
    rigid_forearm_mount: bool,
) -> float:
    """Integrate fundamental cylindrical slosh mode across plate offsets at candidate `duration_s`."""
    omega_n = slosh_natural_frequency(well_radius_m, fill_height_m)
    n_samples = q.shape[0]
    dt_s = duration_s / max(1, n_samples - 1)
    inv_t2 = 1.0 / (duration_s * duration_s)
    acc_center = tip_acc_unit * inv_t2

    pts = list(offsets_xy) if (rigid_forearm_mount and offsets_xy) else [(0.0, 0.0)]
    if (0.0, 0.0) not in pts:
        pts.insert(0, (0.0, 0.0))

    q12 = q[:, 0] + q[:, 1]
    w12 = (qd_unit[:, 0] + qd_unit[:, 1]) / duration_s
    a12 = (qdd_unit[:, 0] + qdd_unit[:, 1]) * inv_t2
    c12 = np.cos(q12)
    s12 = np.sin(q12)

    k_wells = len(pts)
    well_acc = np.zeros((n_samples, k_wells, 2), dtype=np.float64)
    for idx_w, (rxb, ryb) in enumerate(pts):
        if not rigid_forearm_mount or (rxb == 0.0 and ryb == 0.0):
            well_acc[:, idx_w, :] = acc_center
        else:
            rx = c12 * rxb - s12 * ryb
            ry = s12 * rxb + c12 * ryb
            well_acc[:, idx_w, 0] = acc_center[:, 0] - a12 * ry - (w12**2) * rx
            well_acc[:, idx_w, 1] = acc_center[:, 1] + a12 * rx - (w12**2) * ry

    n_ring = max(10, int(math.ceil(0.12 / max(dt_s, 1e-4))))
    a_pad = np.concatenate([well_acc, np.zeros((n_ring, k_wells, 2), dtype=np.float64)], axis=0)
    n_sub = max(1, int(math.ceil(omega_n * dt_s / 0.15)))
    h = dt_s / n_sub
    gain = (well_radius_m / _GRAVITY_M_S2) * (omega_n * omega_n)
    w2 = omega_n * omega_n
    two_zw = 2.0 * damping_ratio * omega_n

    eta = np.zeros((k_wells, 2), dtype=np.float64)
    eta_dot = np.zeros((k_wells, 2), dtype=np.float64)
    peak_r = 0.0
    for i in range(a_pad.shape[0]):
        drive = gain * a_pad[i]
        for _ in range(n_sub):
            k1_x = eta_dot
            k1_v = drive - two_zw * eta_dot - w2 * eta
            x2 = eta + 0.5 * h * k1_x
            v2 = eta_dot + 0.5 * h * k1_v
            k2_x = v2
            k2_v = drive - two_zw * v2 - w2 * x2
            x3 = eta + 0.5 * h * k2_x
            v3 = eta_dot + 0.5 * h * k2_v
            k3_x = v3
            k3_v = drive - two_zw * v3 - w2 * x3
            x4 = eta + h * k3_x
            v4 = eta_dot + h * k3_v
            k4_x = v4
            k4_v = drive - two_zw * v4 - w2 * x4
            eta = eta + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            eta_dot = eta_dot + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        r_cur = float(np.max(np.hypot(eta[:, 0], eta[:, 1])))
        if r_cur > peak_r:
            peak_r = r_cur
    return peak_r


def parameterize_path(
    path: Sequence[np.ndarray],
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    qd_max: float = 1.5,
    qdd_max: float = 4.0,
    a_max_liquid: float = 1.5,
    n_samples: int = 401,
    open_plate: Optional[Any] = None,
    well_radius_m: Optional[float] = None,
    fill_height_m: Optional[float] = None,
    well_depth_m: Optional[float] = None,
    damping_ratio: float = 0.028,
    headspace_safety_fraction: float = 0.70,
    corner_offsets_xy: Optional[Sequence[Tuple[float, float]]] = None,
    rigid_forearm_mount: bool = True,
) -> TimeParameterizedTrajectory:
    """Time-parameterize a waypoint path via analytical C2 curve evaluation and slosh bounds."""
    if not path:
        raise ValueError("Cannot parameterize empty path")
    if qd_max <= 0.0 or qdd_max <= 0.0 or a_max_liquid <= 0.0:
        raise ValueError("Velocity and acceleration bounds must be > 0")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

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

    pieces, s_total = _build_blended_geometric_pieces(pts, arm=arm, scene=scene)

    sigma = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    u = 10.0 * sigma**3 - 15.0 * sigma**4 + 6.0 * sigma**5
    up = 30.0 * sigma**2 - 60.0 * sigma**3 + 30.0 * sigma**4
    upp = 60.0 * sigma - 180.0 * sigma**2 + 120.0 * sigma**3

    s_eval = s_total * u
    ds_dsigma = s_total * up
    d2s_dsigma2 = s_total * upp

    # Exact analytical evaluation at warped arc positions s_eval (chain rule exact: dq/dt == qd)
    q, qp, qpp = _eval_blended_pieces(pieces, s_eval)

    qd_unit = qp * ds_dsigma[:, None]
    qdd_unit = qpp * (ds_dsigma[:, None] ** 2) + qp * d2s_dsigma2[:, None]
    _, _, tip_acc_unit = arm.batch_tip_kinematics(q, qd_unit, qdd_unit)

    peak_qd = float(np.max(np.abs(qd_unit)))
    peak_qdd = float(np.max(np.abs(qdd_unit)))
    peak_tip_acc = float(np.max(np.linalg.norm(tip_acc_unit, axis=-1)))

    duration = max(
        1e-3,
        peak_qd / qd_max,
        math.sqrt(peak_qdd / qdd_max),
        math.sqrt(peak_tip_acc / a_max_liquid),
    ) * (1.0 + 1e-6)

    # Optional dynamic cylindrical open-meniscus slosh scaling (including corner-well rotation)
    if open_plate is not None:
        well_radius_m = float(open_plate.well_radius_m)
        fill_height_m = float(open_plate.fill_height_m)
        well_depth_m = float(open_plate.well_depth_m)
        damping_ratio = float(open_plate.damping_ratio)
        if corner_offsets_xy is None and hasattr(open_plate, "corner_offsets_xy"):
            corner_offsets_xy = tuple(open_plate.corner_offsets_xy)
        if hasattr(open_plate, "rigid_forearm_mount"):
            rigid_forearm_mount = bool(open_plate.rigid_forearm_mount)

    if well_radius_m is not None and fill_height_m is not None and well_depth_m is not None:
        h_free = float(well_depth_m) - float(fill_height_m)
        if h_free <= 0.0:
            raise ValueError("Open-well plate has non-positive free headspace")
        eff_offsets = (
            ((0.0495, 0.0315), (0.0495, -0.0315), (-0.0495, 0.0315), (-0.0495, -0.0315))
            if corner_offsets_xy is None
            else tuple(corner_offsets_xy)
        )
        cap_m = max(0.01, headspace_safety_fraction - 0.02) * h_free
        rise = _cylindrical_slosh_peak_rise_m(
            q,
            qd_unit,
            qdd_unit,
            tip_acc_unit,
            duration,
            well_radius_m=float(well_radius_m),
            fill_height_m=float(fill_height_m),
            damping_ratio=float(damping_ratio),
            offsets_xy=eff_offsets,
            rigid_forearm_mount=rigid_forearm_mount,
        )
        if rise > cap_m:
            t_lo = duration
            t_hi = duration * 1.5
            while (
                _cylindrical_slosh_peak_rise_m(
                    q,
                    qd_unit,
                    qdd_unit,
                    tip_acc_unit,
                    t_hi,
                    well_radius_m=float(well_radius_m),
                    fill_height_m=float(fill_height_m),
                    damping_ratio=float(damping_ratio),
                    offsets_xy=eff_offsets,
                    rigid_forearm_mount=rigid_forearm_mount,
                )
                > cap_m
                and t_hi < 30.0
            ):
                t_hi *= 1.5
            for _ in range(14):
                t_mid = 0.5 * (t_lo + t_hi)
                r_mid = _cylindrical_slosh_peak_rise_m(
                    q,
                    qd_unit,
                    qdd_unit,
                    tip_acc_unit,
                    t_mid,
                    well_radius_m=float(well_radius_m),
                    fill_height_m=float(fill_height_m),
                    damping_ratio=float(damping_ratio),
                    offsets_xy=eff_offsets,
                    rigid_forearm_mount=rigid_forearm_mount,
                )
                if r_mid <= cap_m:
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            duration = t_hi
            while (
                _cylindrical_slosh_peak_rise_m(
                    q,
                    qd_unit,
                    qdd_unit,
                    tip_acc_unit,
                    duration,
                    well_radius_m=float(well_radius_m),
                    fill_height_m=float(fill_height_m),
                    damping_ratio=float(damping_ratio),
                    offsets_xy=eff_offsets,
                    rigid_forearm_mount=rigid_forearm_mount,
                )
                >= headspace_safety_fraction * h_free
            ):
                duration *= 1.05

    t = sigma * duration
    qd = qd_unit / duration
    qdd = qdd_unit / (duration * duration)
    tip_xy, tip_vel, tip_acc = arm.batch_tip_kinematics(q, qd, qdd)
    bo_val: Optional[float] = None
    model_status = "NOT_APPLICABLE"
    if well_radius_m is not None:
        bo_val = round(slosh_bond_number(float(well_radius_m)), 4)
        model_status = (
            "IN_GRAVITY_REGIME" if bo_val >= MIN_GRAVITY_SLOSH_BOND_NUMBER else "SLOSH_MODEL_OUT_OF_RANGE"
        )

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
        bond_number=bo_val,
        slosh_model_status=model_status,
    )


def plan_transfer_3d(
    well: str,
    rng: np.random.Generator,
    *,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    planner: str = "rrt_connect",
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    attached_tip_length_m: Optional[float] = None,
    z_aspirate: Optional[float] = None,
    z_dispense: Optional[float] = None,
) -> Optional[List[np.ndarray]]:
    """Plan a 3-DoF (q1, q2, z_flange) retract -> planar transit -> plunge free of rim collisions."""
    eff_scene = (
        scene
        if attached_tip_length_m is None
        else scene.with_attached_tip(float(attached_tip_length_m))
    )
    if start_q is not None:
        q_start = np.asarray(start_q, dtype=np.float64)
        if q_start.shape != (2,) or in_collision(q_start, arm=arm, scene=eff_scene):
            return None
    else:
        q_start = select_ik_branch(eff_scene.tube_xy, arm=arm, scene=eff_scene)
        if q_start is None:
            return None

    target_xy = eff_scene.plate.well_xy(well)
    goals = ik_solutions(target_xy, arm=arm, scene=eff_scene, prev_q=q_start)
    if not goals:
        return None

    solve_fn = rrt_connect if planner == "rrt_connect" else rrt
    raw_2d = solve_fn(q_start, goals, rng, arm=arm, scene=eff_scene)
    if raw_2d is None:
        return None
    smooth_2d = shortcut(raw_2d, rng, arm=arm, scene=eff_scene)

    tip_len = float(eff_scene.attached_tip_length_m)
    z_asp = (
        eff_scene.z_tube_bottom + 0.002 + tip_len
        if z_aspirate is None
        else float(z_aspirate)
    )
    z_disp = (
        eff_scene.plate.z_bottom_m + 0.002 + tip_len
        if z_dispense is None
        else float(z_dispense)
    )
    z_safe = float(eff_scene.z_safe_travel)  # type: ignore[arg-type]

    _, start_xy = arm.fk(q_start)
    waypoints_3d: List[np.ndarray] = [
        np.array([q_start[0], q_start[1], z_asp], dtype=np.float64),
    ]
    for q in smooth_2d:
        waypoints_3d.append(np.array([q[0], q[1], z_safe], dtype=np.float64))
    q_goal = smooth_2d[-1]
    waypoints_3d.append(np.array([q_goal[0], q_goal[1], z_disp], dtype=np.float64))

    allowed_bores = [start_xy, target_xy]
    for a3, b3 in zip(waypoints_3d, waypoints_3d[1:]):
        if not edge_free_3d(
            a3,
            b3,
            arm=arm,
            scene=eff_scene,
            allowed_wells_xy=allowed_bores,
        ):
            return None
    return waypoints_3d


def _resolve_3d_endpoint(
    ref: str,
    scene: BenchScene,
    deck: Optional[DeckLayout],
) -> Tuple[np.ndarray, float, float, float, Optional[PlateGeometry]]:
    """Return (xy_m, z_bottom_m, z_rim_m, bore_radius_m, plate_or_none)."""
    s = ref.strip()
    if s.upper() == "TUBE":
        return (
            np.asarray(scene.tube_xy, dtype=np.float64),
            float(scene.z_tube_bottom),
            float(scene.z_tube_rim),
            float(scene.tube_radius_m),
            None,
        )
    eff_deck = deck if deck is not None else scene.deck
    alias, well = parse_well_ref(s)
    if alias is not None:
        if eff_deck is None:
            raise ValueError(f"Deck-qualified well reference {ref!r} requires a DeckLayout")
        geom, _, xy = eff_deck.resolve_well(s)
        return xy, float(geom.z_bottom_m), float(geom.z_rim_m), float(geom.well_radius_m), geom
    if eff_deck is not None and len(eff_deck.labware_by_alias) == 1:
        geom, _, xy = eff_deck.resolve_well(s)
        return xy, float(geom.z_bottom_m), float(geom.z_rim_m), float(geom.well_radius_m), geom
    xy = scene.plate.well_xy(well)
    return (
        xy,
        float(scene.plate.z_bottom_m),
        max(float(scene.z_plate_rim), float(scene.plate.z_rim_m)),
        float(scene.plate.well_radius_m),
        scene.plate,
    )


def plan_3d_pick_place(
    source_ref: str,
    target_ref: str,
    rng: Optional[np.random.Generator] = None,
    *,
    arm: Arm2D = DEFAULT_ARM,
    arm_se3: Optional[ArmSE3] = DEFAULT_ARM_SE3,
    scene: BenchScene = DEFAULT_SCENE,
    deck: Optional[DeckLayout] = None,
    attached_tip_length_m: Optional[float] = None,
    pickup_tip_from_source: Optional[bool] = None,
    tip_length_m: Optional[float] = None,
    tool_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_tilt_rad: float = math.radians(1.5),
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    planner: str = "rrt_connect",
) -> Optional[PickPlacePlan3D]:
    """Plan a 3D / 6-DOF SE(3) vertical-horizontal-vertical transfer over tall blocks and tipracks."""
    eff_rng = np.random.default_rng(0) if rng is None else rng
    eff_deck = deck if deck is not None else scene.deck

    src_xy, src_z_bot, src_z_rim, src_rad, src_geom = _resolve_3d_endpoint(source_ref, scene, eff_deck)
    dst_xy, dst_z_bot, dst_z_rim, dst_rad, _ = _resolve_3d_endpoint(target_ref, scene, eff_deck)

    pre_tip_len = float(scene.attached_tip_length_m) if attached_tip_length_m is None else float(attached_tip_length_m)
    is_tip_pickup = (
        bool(src_geom is not None and src_geom.is_tiprack and pre_tip_len == 0.0)
        if pickup_tip_from_source is None
        else bool(pickup_tip_from_source)
    )
    if is_tip_pickup:
        if tip_length_m is not None:
            post_tip_len = float(tip_length_m)
        elif src_geom is not None and src_geom.is_tiprack:
            post_tip_len = src_geom.default_tip_length_m()
        else:
            post_tip_len = 0.058
    else:
        post_tip_len = pre_tip_len

    transit_scene = scene.with_attached_tip(post_tip_len, deck=eff_deck)
    z_safe = max(
        float(transit_scene.z_safe_travel),  # type: ignore[arg-type]
        max(src_z_rim, dst_z_rim, transit_scene.max_vessel_rim_m) + post_tip_len + transit_scene.z_margin_m + 0.005,
    )
    if z_safe > transit_scene.z_max:
        return None

    # Enforce 6-DOF SE(3) tool-axis verticality on both source and destination vessel plunges
    src_plunge_depth = max(0.0, src_z_rim - src_z_bot)
    dst_plunge_depth = max(0.0, dst_z_rim - dst_z_bot)
    z_pick = (src_z_rim if is_tip_pickup else src_z_bot + 0.002) + post_tip_len
    z_place = dst_z_bot + 0.002 + post_tip_len

    pose_pick = PoseSE3(xyz=(float(src_xy[0]), float(src_xy[1]), z_pick), rpy=tool_rpy)
    pose_place = PoseSE3(xyz=(float(dst_xy[0]), float(dst_xy[1]), z_place), rpy=tool_rpy)
    if not pose_pick.check_plunge_verticality(
        src_rad,
        src_plunge_depth,
        attached_tip_length_m=post_tip_len,
        max_tilt_rad=max_tilt_rad,
    ):
        return None
    if not pose_place.check_plunge_verticality(
        dst_rad,
        dst_plunge_depth,
        attached_tip_length_m=post_tip_len,
        max_tilt_rad=max_tilt_rad,
    ):
        return None

    if start_q is not None:
        q_src = np.asarray(start_q, dtype=np.float64)
        if q_src.shape != (2,) or in_collision(q_src, arm=arm, scene=transit_scene):
            return None
        if float(np.linalg.norm(arm.fk(q_src)[1] - src_xy)) > 1e-3:
            q_src = select_ik_branch(src_xy, prev_q=q_src, arm=arm, scene=transit_scene)  # type: ignore[assignment]
            if q_src is None:
                return None
    else:
        q_src = select_ik_branch(src_xy, arm=arm, scene=transit_scene)  # type: ignore[assignment]
        if q_src is None:
            return None

    goals = ik_solutions(dst_xy, arm=arm, scene=transit_scene, prev_q=q_src)
    if not goals:
        return None

    solve_fn = rrt_connect if planner == "rrt_connect" else rrt
    raw_2d = solve_fn(q_src, goals, eff_rng, arm=arm, scene=transit_scene)
    if raw_2d is None:
        return None
    smooth_2d = shortcut(raw_2d, eff_rng, arm=arm, scene=transit_scene)
    q_dst = smooth_2d[-1]

    waypoints_3d: List[np.ndarray] = [np.array([q_src[0], q_src[1], z_pick], dtype=np.float64)]
    for q in smooth_2d:
        waypoints_3d.append(np.array([q[0], q[1], z_safe], dtype=np.float64))
    waypoints_3d.append(np.array([q_dst[0], q_dst[1], z_place], dtype=np.float64))

    allowed_bores = [src_xy, dst_xy]
    for a3, b3 in zip(waypoints_3d, waypoints_3d[1:]):
        if not edge_free_3d(a3, b3, arm=arm, scene=transit_scene, allowed_wells_xy=allowed_bores):
            return None

    poses_se3: List[PoseSE3] = []
    q6_list: Optional[List[np.ndarray]] = [] if arm_se3 is not None else None
    max_tilt = 0.0
    for wp in waypoints_3d:
        _, xy = arm.fk(wp[:2])
        pose = PoseSE3(xyz=(float(xy[0]), float(xy[1]), float(wp[2])), rpy=tool_rpy)
        poses_se3.append(pose)
        max_tilt = max(max_tilt, pose.tool_tilt_rad())
        if arm_se3 is not None and q6_list is not None:
            if hasattr(arm_se3, "dls_ik"):
                prev_seed = q6_list[-1] if q6_list else None
                ik_res = arm_se3.dls_ik(pose, q_seed=prev_seed)  # type: ignore[attr-defined]
                if not ik_res.success:
                    return None
                q6_list.append(ik_res.q.copy())
            else:
                q6 = arm_se3.ik_se3(pose, elbow=1)
                if q6 is None:
                    q6 = arm_se3.ik_se3(pose, elbow=-1)
                if q6 is None:
                    return None
                q6_list.append(q6)

    return PickPlacePlan3D(
        source_ref=source_ref,
        target_ref=target_ref,
        attached_tip_length_m=post_tip_len,
        z_safe_transit_m=z_safe,
        z_pick_m=z_pick,
        z_place_m=z_place,
        waypoints_3d=waypoints_3d,
        poses_se3=poses_se3,
        q6_waypoints=q6_list,
        max_tool_tilt_rad=max_tilt,
        collision_free=True,
    )


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
    is_free = bool(
        in_lim and all(edge_free(a, b, arm=arm, scene=scene) for a, b in zip(smooth_path, smooth_path[1:]))
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


# ---------------------------------------------------------------------------
# Canonical HuggingFace LeRobot v2.1 / v3.0 & Opentrons 3D Gantry Interfaces
# ---------------------------------------------------------------------------

OPENTRONS_DECK_BOUNDS_MM: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (0.0, 418.0),
    (0.0, 353.0),
    (0.0, 218.0),
)

# Exact rigid offset aligning lab_hands DeckLayout slots 1..11 (m) with Opentrons deck coordinates (mm):
# Slot 1 A1: (0.1600, -0.1000) m <-> (14.38, 74.24) mm
# Slot 5 A1: (0.2925, -0.0095) m <-> (146.88, 164.74) mm
DEFAULT_DECK_ORIGIN_IN_BASE_M: Tuple[float, float, float] = (0.14562, -0.17424, 0.0)


@dataclass(frozen=True)
class LeRobotFrameTable:
    """Canonical HuggingFace LeRobot v2.1 / v3.0 frame-tabular episode tensors."""

    observation_state: np.ndarray  # float32 (T, D_state), e.g. (T, 7) = 6 UR5e joints + 1 gripper/plunger
    action: np.ndarray  # float32 (T, D_action), e.g. (T, 7)
    timestamp: np.ndarray  # float32 (T,)
    frame_index: np.ndarray  # int64 (T,)
    episode_index: np.ndarray  # int64 (T,)
    next_done: np.ndarray  # bool (T,)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_frames(self) -> int:
        return int(self.observation_state.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observation_state.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    def validate(self) -> None:
        validate_lerobot_frame_tensors(self.to_dict())

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "observation.state": self.observation_state,
            "action": self.action,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "episode_index": self.episode_index,
            "next.done": self.next_done,
        }

    def to_rows(self, *, task_index: int = 0) -> List[Dict[str, Any]]:
        self.validate()
        rows: List[Dict[str, Any]] = []
        for idx in range(self.n_frames):
            rows.append(
                {
                    "observation.state": [float(v) for v in self.observation_state[idx]],
                    "action": [float(v) for v in self.action[idx]],
                    "timestamp": float(self.timestamp[idx]),
                    "frame_index": int(self.frame_index[idx]),
                    "episode_index": int(self.episode_index[idx]),
                    "next.done": bool(self.next_done[idx]),
                    "task_index": int(task_index),
                }
            )
        return rows


def validate_lerobot_frame_tensors(tensors: Mapping[str, Any]) -> None:
    """Strictly validate canonical HuggingFace LeRobot v2.1 / v3.0 tensor keys, dtypes, and shapes."""
    required = (
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "next.done",
    )
    for k in required:
        if k not in tensors:
            raise ValueError(f"Missing canonical LeRobot tensor key {k!r}")
        if not isinstance(tensors[k], np.ndarray):
            raise TypeError(f"LeRobot tensor {k!r} must be a numpy.ndarray, got {type(tensors[k]).__name__}")

    obs = tensors["observation.state"]
    act = tensors["action"]
    ts = tensors["timestamp"]
    fidx = tensors["frame_index"]
    eidx = tensors["episode_index"]
    done = tensors["next.done"]

    if obs.dtype != np.float32 or obs.ndim != 2 or obs.shape[0] < 1 or obs.shape[1] < 1:
        raise ValueError(
            f"observation.state must be float32 of shape (T, D_state), got dtype={obs.dtype}, shape={obs.shape}"
        )
    t_len = int(obs.shape[0])

    if act.dtype != np.float32 or act.ndim != 2 or act.shape[0] != t_len or act.shape[1] < 1:
        raise ValueError(
            f"action must be float32 of shape ({t_len}, D_action), got dtype={act.dtype}, shape={act.shape}"
        )
    if ts.dtype != np.float32 or ts.shape != (t_len,):
        raise ValueError(f"timestamp must be float32 of shape ({t_len},), got dtype={ts.dtype}, shape={ts.shape}")
    if fidx.dtype != np.int64 or fidx.shape != (t_len,):
        raise ValueError(
            f"frame_index must be int64 of shape ({t_len},), got dtype={fidx.dtype}, shape={fidx.shape}"
        )
    if eidx.dtype != np.int64 or eidx.shape != (t_len,):
        raise ValueError(
            f"episode_index must be int64 of shape ({t_len},), got dtype={eidx.dtype}, shape={eidx.shape}"
        )
    if done.dtype != np.bool_ or done.shape != (t_len,):
        raise ValueError(f"next.done must be bool of shape ({t_len},), got dtype={done.dtype}, shape={done.shape}")
    if not (np.isfinite(obs).all() and np.isfinite(act).all() and np.isfinite(ts).all()):
        raise ValueError("LeRobot float32 tensors must contain finite values")
    if t_len >= 2 and bool(np.any(np.diff(ts) < -1e-6)):
        raise ValueError("LeRobot timestamp tensor must be monotonically non-decreasing")
    if not bool(done[-1]):
        raise ValueError("LeRobot next.done[-1] must be True on terminal episode frame")


def trajectory_to_lerobot_tensors(
    timestamps_s: Sequence[float] | np.ndarray,
    joint_positions_rad: Sequence[Sequence[float]] | np.ndarray,
    *,
    gripper_or_plunger: Optional[Sequence[float] | np.ndarray] = None,
    actions: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    episode_index: int = 0,
    append_gripper_if_missing: bool = True,
) -> LeRobotFrameTable:
    """Convert a multi-DoF trajectory `(T, D)` into a validated `LeRobotFrameTable` (`float32`, `int64`, `bool`).

    For 6-DoF UR5e trajectories `(T, 6)`, appends a 7th gripper/plunger channel by default so
    `observation.state` and `action` have canonical shape `(T, 7)` `float32`.
    """
    t_arr = np.asarray(timestamps_s, dtype=np.float32)
    q_arr = np.asarray(joint_positions_rad, dtype=np.float32)
    if q_arr.ndim == 1:
        q_arr = q_arr[:, None]
    if q_arr.ndim != 2 or q_arr.shape[0] < 1:
        raise ValueError(f"joint_positions_rad must have shape (T, D), got {q_arr.shape}")
    n_frames = int(q_arr.shape[0])
    if t_arr.shape != (n_frames,):
        raise ValueError(f"timestamps_s shape {t_arr.shape} must match T={n_frames}")
    t_arr = t_arr - float(t_arr[0])

    if gripper_or_plunger is not None:
        g_arr = np.asarray(gripper_or_plunger, dtype=np.float32).reshape(n_frames, 1)
        obs_state = np.concatenate([q_arr, g_arr], axis=1).astype(np.float32, copy=False)
    elif append_gripper_if_missing and q_arr.shape[1] == 6:
        g_zero = np.zeros((n_frames, 1), dtype=np.float32)
        obs_state = np.concatenate([q_arr, g_zero], axis=1).astype(np.float32, copy=False)
    else:
        obs_state = q_arr.astype(np.float32, copy=False)

    if actions is not None:
        act_arr = np.asarray(actions, dtype=np.float32)
        if act_arr.ndim == 1:
            act_arr = act_arr[:, None]
        if act_arr.shape[0] != n_frames:
            raise ValueError(f"actions length {act_arr.shape[0]} must match T={n_frames}")
        if act_arr.shape[1] == q_arr.shape[1] and obs_state.shape[1] == q_arr.shape[1] + 1:
            g_act = obs_state[:, -1:] if n_frames == 1 else np.vstack([obs_state[1:, -1:], obs_state[-1:, -1:]])
            act_arr = np.concatenate([act_arr, g_act], axis=1).astype(np.float32, copy=False)
    else:
        # Canonical target-position action: next-step state (with terminal hold)
        act_arr = (
            obs_state.copy()
            if n_frames == 1
            else np.vstack([obs_state[1:], obs_state[-1:]]).astype(np.float32, copy=False)
        )

    frame_idx = np.arange(n_frames, dtype=np.int64)
    ep_idx = np.full(n_frames, int(episode_index), dtype=np.int64)
    done = np.zeros(n_frames, dtype=np.bool_)
    done[-1] = True

    return LeRobotFrameTable(
        observation_state=obs_state,
        action=act_arr,
        timestamp=t_arr,
        frame_index=frame_idx,
        episode_index=ep_idx,
        next_done=done,
    )


@dataclass(frozen=True)
class OpentronsWaypointTrajectory:
    """Canonical Opentrons 3D gantry + robot-base SE(3) + plunger trajectory representation."""

    deck_tip_mm: np.ndarray  # float64 (T, 3) in Opentrons deck millimeters (0..418, 0..353, 0..218)
    base_tip_m: np.ndarray  # float64 (T, 3) in robot-base SI meters
    base_pose_se3: np.ndarray  # float64 (T, 6) [x, y, z, roll, pitch, yaw] in SI meters & radians
    plunger_ul: np.ndarray  # float64 (T,) liquid volume held in pipette (uL >= 0)
    timestamps_s: np.ndarray  # float64 (T,) monotonically non-decreasing seconds
    deck_origin_in_base_m: Tuple[float, float, float] = DEFAULT_DECK_ORIGIN_IN_BASE_M

    def __post_init__(self) -> None:
        self.validate(strict_deck_bounds=True)

    @property
    def n_frames(self) -> int:
        return int(self.deck_tip_mm.shape[0])

    def validate(self, *, strict_deck_bounds: bool = True) -> None:
        validate_opentrons_trajectory(
            {
                "deck_tip_mm": self.deck_tip_mm,
                "base_tip_m": self.base_tip_m,
                "base_pose_se3": self.base_pose_se3,
                "plunger_ul": self.plunger_ul,
                "timestamps_s": self.timestamps_s,
            },
            deck_origin_in_base_m=self.deck_origin_in_base_m,
            strict_deck_bounds=strict_deck_bounds,
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "deck_tip_mm": self.deck_tip_mm,
            "base_tip_m": self.base_tip_m,
            "base_pose_se3": self.base_pose_se3,
            "plunger_ul": self.plunger_ul,
            "timestamps_s": self.timestamps_s,
        }

    def to_lerobot_ur5e(
        self,
        *,
        arm: Optional[Any] = None,
        episode_index: int = 0,
        max_pipette_volume_ul: float = 300.0,
        attached_tip_length_m: float = 0.0,
    ) -> LeRobotFrameTable:
        """Solve 6-DoF `UR5eArm` IK along `base_pose_se3` and pack `(T, 7)` LeRobot tensors."""
        from .kinematics_se3 import DEFAULT_UR5E

        cobot = DEFAULT_UR5E if arm is None else arm
        qs: List[np.ndarray] = []
        prev_q: Optional[np.ndarray] = None
        for idx in range(self.n_frames):
            row = self.base_pose_se3[idx]
            pose = PoseSE3(
                xyz=(float(row[0]), float(row[1]), float(row[2])),
                rpy=(float(row[3]), float(row[4]), float(row[5])),
            )
            ik_res = cobot.dls_ik(
                pose,
                q_seed=prev_q,
                attached_tip_length_m=attached_tip_length_m,
            )
            if not ik_res.success:
                raise ValueError(
                    f"UR5e IK infeasible at frame {idx}: xyz={pose.xyz}, pos_err={ik_res.pos_error_m:.5f} m"
                )
            qs.append(ik_res.q)
            prev_q = ik_res.q

        plunger_norm = np.clip(self.plunger_ul / max(float(max_pipette_volume_ul), 1e-6), 0.0, 1.0)
        return trajectory_to_lerobot_tensors(
            self.timestamps_s,
            np.stack(qs, axis=0),
            gripper_or_plunger=plunger_norm,
            episode_index=episode_index,
        )


def validate_opentrons_trajectory(
    data: Mapping[str, Any],
    *,
    deck_origin_in_base_m: Tuple[float, float, float] = DEFAULT_DECK_ORIGIN_IN_BASE_M,
    strict_deck_bounds: bool = True,
) -> None:
    """Strictly verify Opentrons `(T, 3)` deck mm, `(T, 3)` SI m, `(T, 6)` SE(3), and `(T,)` plunger uL."""
    for key in ("deck_tip_mm", "base_tip_m", "base_pose_se3", "plunger_ul", "timestamps_s"):
        if key not in data:
            raise ValueError(f"Missing Opentrons trajectory field {key!r}")
        if not isinstance(data[key], np.ndarray):
            raise TypeError(f"Opentrons field {key!r} must be numpy.ndarray, got {type(data[key]).__name__}")

    deck_mm = data["deck_tip_mm"]
    base_m = data["base_tip_m"]
    pose6 = data["base_pose_se3"]
    plunger = data["plunger_ul"]
    ts = data["timestamps_s"]

    if deck_mm.dtype != np.float64 or deck_mm.ndim != 2 or deck_mm.shape[1] != 3 or deck_mm.shape[0] < 1:
        raise ValueError(
            f"deck_tip_mm must be float64 of shape (T, 3), got dtype={deck_mm.dtype}, shape={deck_mm.shape}"
        )
    t_len = int(deck_mm.shape[0])

    if base_m.dtype != np.float64 or base_m.shape != (t_len, 3):
        raise ValueError(
            f"base_tip_m must be float64 of shape ({t_len}, 3), got dtype={base_m.dtype}, shape={base_m.shape}"
        )
    if pose6.dtype != np.float64 or pose6.shape != (t_len, 6):
        raise ValueError(
            f"base_pose_se3 must be float64 of shape ({t_len}, 6), got dtype={pose6.dtype}, shape={pose6.shape}"
        )
    if plunger.dtype != np.float64 or plunger.shape != (t_len,):
        raise ValueError(
            f"plunger_ul must be float64 of shape ({t_len},), got dtype={plunger.dtype}, shape={plunger.shape}"
        )
    if ts.dtype != np.float64 or ts.shape != (t_len,):
        raise ValueError(
            f"timestamps_s must be float64 of shape ({t_len},), got dtype={ts.dtype}, shape={ts.shape}"
        )

    for arr_name, arr in (
        ("deck_tip_mm", deck_mm),
        ("base_tip_m", base_m),
        ("base_pose_se3", pose6),
        ("plunger_ul", plunger),
        ("timestamps_s", ts),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"Opentrons field {arr_name!r} contains non-finite values")

    if bool(np.any(plunger < -1e-9)):
        raise ValueError("plunger_ul cannot be negative")

    if strict_deck_bounds:
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = OPENTRONS_DECK_BOUNDS_MM
        tol = 1e-6
        if bool(
            np.any(deck_mm[:, 0] < x_lo - tol)
            or np.any(deck_mm[:, 0] > x_hi + tol)
            or np.any(deck_mm[:, 1] < y_lo - tol)
            or np.any(deck_mm[:, 1] > y_hi + tol)
            or np.any(deck_mm[:, 2] < z_lo - tol)
            or np.any(deck_mm[:, 2] > z_hi + tol)
        ):
            raise ValueError(
                f"deck_tip_mm exceeds Opentrons deck envelope (0..418, 0..353, 0..218) mm: "
                f"min={deck_mm.min(axis=0).round(3)}, max={deck_mm.max(axis=0).round(3)}"
            )

    origin = np.asarray(deck_origin_in_base_m, dtype=np.float64)
    expected_base = origin[None, :] + 1e-3 * deck_mm
    if not np.allclose(base_m, expected_base, atol=1e-6):
        raise ValueError("base_tip_m does not match deck_origin_in_base_m + 1e-3 * deck_tip_mm")
    if not np.allclose(pose6[:, :3], base_m, atol=1e-6):
        raise ValueError("base_pose_se3[:, :3] must match base_tip_m")


def plan_to_opentrons_trajectory(
    plan: PickPlacePlan3D,
    *,
    aspirate_volume_ul: float = 100.0,
    samples_per_segment: int = 12,
    dt_s: float = 0.04,
    deck_origin_in_base_m: Tuple[float, float, float] = DEFAULT_DECK_ORIGIN_IN_BASE_M,
) -> OpentronsWaypointTrajectory:
    """Convert a `PickPlacePlan3D` over deck labware into a validated `OpentronsWaypointTrajectory`."""
    if not plan.poses_se3:
        raise ValueError("plan.poses_se3 must be non-empty")
    origin = np.asarray(deck_origin_in_base_m, dtype=np.float64)

    tip_pts_m: List[np.ndarray] = []
    rpy_pts: List[Tuple[float, float, float]] = []
    vols: List[float] = []

    key_tips = [p.tip_position(attached_tip_length_m=plan.attached_tip_length_m) for p in plan.poses_se3]
    key_rpys = [p.rpy for p in plan.poses_se3]

    # Clamp/project waypoints into Opentrons deck gantry workspace when converting off-deck cobot plans
    # (e.g. legacy off-deck TUBE=(-0.05, 0.40) remaps to Slot 2 tube rack (0.2925, -0.1000) m)
    key_tips_clamped: List[np.ndarray] = []
    for pt in key_tips:
        x_m, y_m, z_m = float(pt[0]), float(pt[1]), float(pt[2])
        dx_mm = (x_m - float(origin[0])) * 1000.0
        dy_mm = (y_m - float(origin[1])) * 1000.0
        if dx_mm < 0.0 or dx_mm > 418.0 or dy_mm < 0.0 or dy_mm > 353.0:
            # Map off-deck cobot pickup stand into Opentrons deck envelope
            dx_mm = float(np.clip(dx_mm if dx_mm >= 0.0 else 146.88, 5.0, 413.0))
            dy_mm = float(np.clip(dy_mm if dy_mm <= 353.0 else 74.24, 5.0, 348.0))
            x_m = float(origin[0]) + 1e-3 * dx_mm
            y_m = float(origin[1]) + 1e-3 * dy_mm
        z_clamped = float(np.clip(z_m, float(origin[2]), float(origin[2]) + 0.218))
        key_tips_clamped.append(np.array([x_m, y_m, z_clamped], dtype=np.float64))

    n_seg = len(key_tips_clamped) - 1
    if n_seg == 0:
        tip_pts_m.append(key_tips_clamped[0])
        rpy_pts.append(key_rpys[0])
        vols.append(0.0)
    else:
        for s_idx in range(n_seg):
            a_pt = key_tips_clamped[s_idx]
            b_pt = key_tips_clamped[s_idx + 1]
            a_rpy = np.asarray(key_rpys[s_idx], dtype=np.float64)
            b_rpy = np.asarray(key_rpys[s_idx + 1], dtype=np.float64)
            m_sub = max(2, int(samples_per_segment))
            alphas = np.linspace(0.0, 1.0, m_sub, endpoint=(s_idx == n_seg - 1))
            for al in alphas:
                u = 3.0 * (al**2) - 2.0 * (al**3)
                p_cur = (1.0 - u) * a_pt + u * b_pt
                r_cur = (1.0 - u) * a_rpy + u * b_rpy
                tip_pts_m.append(p_cur)
                rpy_pts.append((float(r_cur[0]), float(r_cur[1]), float(r_cur[2])))
                # Hold aspirated volume during transit segments (1 .. n_seg-1), dispense on final segment
                if s_idx == 0:
                    vols.append(float(aspirate_volume_ul) * float(al))
                elif s_idx == n_seg - 1:
                    vols.append(float(aspirate_volume_ul) * (1.0 - float(al)))
                else:
                    vols.append(float(aspirate_volume_ul))

    base_tip_m = np.stack(tip_pts_m, axis=0).astype(np.float64)
    deck_tip_mm = ((base_tip_m - origin[None, :]) * 1000.0).astype(np.float64)
    rpy_arr = np.asarray(rpy_pts, dtype=np.float64)
    base_pose_se3 = np.concatenate([base_tip_m, rpy_arr], axis=1).astype(np.float64)
    plunger_ul = np.maximum(0.0, np.asarray(vols, dtype=np.float64))
    timestamps_s = np.arange(base_tip_m.shape[0], dtype=np.float64) * float(dt_s)

    return OpentronsWaypointTrajectory(
        deck_tip_mm=deck_tip_mm,
        base_tip_m=base_tip_m,
        base_pose_se3=base_pose_se3,
        plunger_ul=plunger_ul,
        timestamps_s=timestamps_s,
        deck_origin_in_base_m=deck_origin_in_base_m,
    )


def _extract_point_mm_from_opentrons_location(loc: Any) -> Optional[Tuple[float, float, float]]:
    if loc is None:
        return None
    pt = getattr(loc, "point", None)
    if pt is not None and hasattr(pt, "x"):
        return (float(pt.x), float(pt.y), float(pt.z))
    if hasattr(loc, "top"):
        top_loc = loc.top()
        top_pt = getattr(top_loc, "point", None)
        if top_pt is not None:
            return (float(top_pt.x), float(top_pt.y), float(top_pt.z))
    lw = getattr(loc, "labware", None)
    if lw is not None and hasattr(lw, "is_well") and bool(lw.is_well):
        well = lw.as_well()
        top_pt = well.top().point
        return (float(top_pt.x), float(top_pt.y), float(top_pt.z))
    if isinstance(loc, Mapping) and all(k in loc for k in ("x", "y", "z")):
        return (float(loc["x"]), float(loc["y"]), float(loc["z"]))
    if isinstance(loc, (tuple, list)) and len(loc) == 3:
        return (float(loc[0]), float(loc[1]), float(loc[2]))
    return None


def extract_opentrons_runlog_trajectory(
    runlog: Sequence[Mapping[str, Any]],
    *,
    safe_transit_z_mm: float = 115.0,
    samples_per_move: int = 10,
    dt_s: float = 0.04,
    deck_origin_in_base_m: Tuple[float, float, float] = DEFAULT_DECK_ORIGIN_IN_BASE_M,
) -> OpentronsWaypointTrajectory:
    """Parse a live `opentrons.simulate.simulate()` 8.8.x runlog into `OpentronsWaypointTrajectory`.

    Inserts safe Z-retract gantry arches (`safe_transit_z_mm <= 218 mm`) between distinct XY
    wells and tracks cumulative plunger microliter volume across `aspirate` / `dispense` / `drop_tip`.
    """
    if not (0.0 <= safe_transit_z_mm <= 218.0):
        raise ValueError(f"safe_transit_z_mm must be in [0, 218], got {safe_transit_z_mm}")

    key_deck_mm: List[np.ndarray] = []
    key_vol_ul: List[float] = []
    cur_vol = 0.0

    for entry in runlog:
        payload = entry.get("payload", entry) if isinstance(entry, Mapping) else {}
        if not isinstance(payload, Mapping):
            continue
        text = str(payload.get("text", "")).lower()
        pt_mm = _extract_point_mm_from_opentrons_location(payload.get("location"))
        if pt_mm is None:
            pt_mm = _extract_point_mm_from_opentrons_location(payload.get("end_location"))
        if pt_mm is None:
            continue

        cur_pt = np.array(
            [
                float(np.clip(pt_mm[0], 0.0, 418.0)),
                float(np.clip(pt_mm[1], 0.0, 353.0)),
                float(np.clip(pt_mm[2], 0.0, 218.0)),
            ],
            dtype=np.float64,
        )

        # Insert safe-Z retract -> XY gantry travel -> Z plunge when moving laterally between wells
        if key_deck_mm:
            prev_pt = key_deck_mm[-1]
            if float(np.linalg.norm(cur_pt[:2] - prev_pt[:2])) > 0.5:
                z_arch = min(218.0, max(float(safe_transit_z_mm), float(prev_pt[2]) + 15.0, float(cur_pt[2]) + 15.0))
                key_deck_mm.append(np.array([prev_pt[0], prev_pt[1], z_arch], dtype=np.float64))
                key_vol_ul.append(cur_vol)
                key_deck_mm.append(np.array([cur_pt[0], cur_pt[1], z_arch], dtype=np.float64))
                key_vol_ul.append(cur_vol)

        vol_delta = float(payload.get("volume") or 0.0)
        if text.startswith("aspirat"):
            # Arrive at well at pre-aspirate volume, then draw plunger up
            key_deck_mm.append(cur_pt)
            key_vol_ul.append(cur_vol)
            cur_vol = cur_vol + max(0.0, vol_delta)
            key_deck_mm.append(cur_pt.copy())
            key_vol_ul.append(cur_vol)
        elif text.startswith("dispens"):
            key_deck_mm.append(cur_pt)
            key_vol_ul.append(cur_vol)
            cur_vol = max(0.0, cur_vol - max(0.0, vol_delta))
            key_deck_mm.append(cur_pt.copy())
            key_vol_ul.append(cur_vol)
        elif "drop" in text or "blow" in text:
            cur_vol = 0.0
            key_deck_mm.append(cur_pt)
            key_vol_ul.append(0.0)
        else:
            key_deck_mm.append(cur_pt)
            key_vol_ul.append(cur_vol)

    if not key_deck_mm:
        raise ValueError("Opentrons runlog contained zero spatial well/point commands")

    dense_mm: List[np.ndarray] = []
    dense_vol: List[float] = []
    n_seg = len(key_deck_mm) - 1
    if n_seg == 0:
        dense_mm.append(key_deck_mm[0])
        dense_vol.append(key_vol_ul[0])
    else:
        for idx in range(n_seg):
            pa, pb = key_deck_mm[idx], key_deck_mm[idx + 1]
            va, vb = key_vol_ul[idx], key_vol_ul[idx + 1]
            m_sub = 2 if np.allclose(pa, pb, atol=1e-6) else max(2, int(samples_per_move))
            alphas = np.linspace(0.0, 1.0, m_sub, endpoint=(idx == n_seg - 1))
            for al in alphas:
                u = 3.0 * (al**2) - 2.0 * (al**3)
                dense_mm.append((1.0 - u) * pa + u * pb)
                dense_vol.append((1.0 - float(al)) * va + float(al) * vb)

    deck_arr = np.stack(dense_mm, axis=0).astype(np.float64)
    origin = np.asarray(deck_origin_in_base_m, dtype=np.float64)
    base_arr = (origin[None, :] + 1e-3 * deck_arr).astype(np.float64)
    zeros_rpy = np.zeros((deck_arr.shape[0], 3), dtype=np.float64)
    pose6_arr = np.concatenate([base_arr, zeros_rpy], axis=1).astype(np.float64)
    vol_arr = np.maximum(0.0, np.asarray(dense_vol, dtype=np.float64))
    t_arr = np.arange(deck_arr.shape[0], dtype=np.float64) * float(dt_s)

    return OpentronsWaypointTrajectory(
        deck_tip_mm=deck_arr,
        base_tip_m=base_arr,
        base_pose_se3=pose6_arr,
        plunger_ul=vol_arr,
        timestamps_s=t_arr,
        deck_origin_in_base_m=deck_origin_in_base_m,
    )

