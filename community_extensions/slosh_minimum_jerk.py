"""Idea #19: Open-Meniscus Pendulum-Slosh Minimum-Jerk Trajectory Time-Scaler.

Inspired by UniCT-ARSLab/DMDTE and high-speed open-plate cobot transport threads.
Models each open 96-well meniscus via its fundamental cylindrical slosh mode:
    omega_n = sqrt(1.841 * g / R) * tanh(1.841 * h / R)
with fill headspace `h_free = H_well - h`, accounting for rigid-plate rotational
Euler tangential (`alpha_12 x r_well`) and centripetal (`-omega_12^2 * r_well`)
accelerations across all outer corner wells (`A1`, `A12`, `H1`, `H12`), and solves
for the minimal trajectory duration `T*` under a 5th-order minimum-jerk law
    s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
so worst-case meniscus rise stays strictly below `0.7 * h_free` (`0` spills)
while maintaining `>= 5 mm` exact continuous capsule-AABB obstacle clearance on a
planar 2-link laboratory arm (`L1 = 0.30 m`, `L2 = 0.25 m`, `|q2| <= 2.60 rad`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lab_hands.collision import BenchScene, batch_min_clearance, min_clearance, path_min_clearance
from lab_hands.kinematics import Arm2D, wrap_angle

GRAVITY_M_S2: float = 9.80665
BESSEL_XI_11: float = 1.841
LINK_1_M: float = 0.30
LINK_2_M: float = 0.25
MAX_ABS_Q2_RAD: float = 2.60
MIN_CLEARANCE_MM: float = 5.0

# Outer well-center offsets (m) from plate centroid on a standard 8x12 9 mm-pitch SBS 96-well plate:
# (+-5.5 * 9 mm, +-3.5 * 9 mm) = (+-49.5 mm, +-31.5 mm)
DEFAULT_CORNER_OFFSETS_XY: tuple[tuple[float, float], ...] = (
    (0.0495, 0.0315),
    (0.0495, -0.0315),
    (-0.0495, 0.0315),
    (-0.0495, -0.0315),
)

# Standard benchtop obstacle boxes (xmin, ymin, xmax, ymax) in meters from lab-hands:
# tube rack wall + reagent bottle between pickup zone (-0.05, 0.40) and plate zone (0.20, 0.05)
BENCH_OBSTACLES_M: tuple[tuple[float, float, float, float], ...] = (
    (0.33, -0.08, 0.36, 0.12),
    (0.15, 0.27, 0.24, 0.34),
)


def _slosh_arm(capsule_radius_m: float = 0.0) -> Arm2D:
    return Arm2D(
        l1=LINK_1_M,
        l2=LINK_2_M,
        joint_limits=((-math.pi, math.pi), (-MAX_ABS_Q2_RAD, MAX_ABS_Q2_RAD)),
        clearance_m=float(capsule_radius_m),
        margin_m=MIN_CLEARANCE_MM / 1000.0,
    )


def _slosh_scene() -> BenchScene:
    return BenchScene(obstacles=tuple(BENCH_OBSTACLES_M))


@dataclass(frozen=True)
class OpenWellPlateSpec:
    """Geometry, mounting, and liquid state of an open 96-well assay plate during transport."""

    well_radius_m: float = 0.00345      # 6.9 mm diameter standard 96-well flat bottom
    well_depth_m: float = 0.01067       # 10.67 mm total well depth (~399 uL brim)
    fill_height_m: float = 0.00920      # liquid fill height `h`
    damping_ratio: float = 0.028        # viscous boundary-layer damping ratio `zeta`
    corner_offsets_xy: tuple[tuple[float, float], ...] = DEFAULT_CORNER_OFFSETS_XY
    rigid_forearm_mount: bool = True    # True: plate rotates with q1 + q2; False: locked-yaw wrist gimbal

    @property
    def free_headspace_m(self) -> float:
        return float(self.well_depth_m - self.fill_height_m)


@dataclass(frozen=True)
class SloshTrajectoryResult:
    """Evaluated open-plate transport trajectory metrics across center and corner wells."""

    method: str
    duration_s: float
    omega_n_rad_s: float
    free_headspace_mm: float
    peak_meniscus_rise_mm: float
    meniscus_to_headspace_ratio: float
    spilled: bool
    peak_lateral_accel_m_s2: float
    min_obstacle_clearance_mm: float
    max_abs_q2_rad: float


def compute_fundamental_slosh_omega(well_radius_m: float, fill_height_m: float) -> float:
    """Fundamental cylindrical antisymmetric slosh frequency `omega_n` (`rad/s`):

        omega_n = sqrt(1.841 * g / R) * tanh(1.841 * h / R)
    """
    if well_radius_m <= 0.0 or fill_height_m <= 0.0:
        raise ValueError("well_radius_m and fill_height_m must be strictly positive")
    arg = BESSEL_XI_11 * fill_height_m / well_radius_m
    return float(np.sqrt(BESSEL_XI_11 * GRAVITY_M_S2 / well_radius_m) * np.tanh(arg))


def _fk_positions(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(elbow_xy, ee_xy)` for joint configuration(s) `q` of shape `(..., 2)`."""
    return _slosh_arm().batch_fk(q)


def _config_clearance_m(q: np.ndarray, capsule_radius_m: float = 0.0) -> float:
    """Exact continuous capsule-surface-to-AABB distance (m) via `lab_hands.collision`."""
    return min_clearance(q, arm=_slosh_arm(capsule_radius_m), scene=_slosh_scene())


def _trajectory_min_clearance_mm(q_traj: np.ndarray, capsule_radius_m: float = 0.0) -> float:
    """Exact continuous swept capsule-surface-to-AABB clearance (mm) across `q_traj`."""
    arr = np.asarray(q_traj, dtype=np.float64)
    arm = _slosh_arm(capsule_radius_m)
    scene = _slosh_scene()
    if arr.ndim == 1:
        return 1000.0 * min_clearance(arr, arm=arm, scene=scene)
    if arr.shape[0] == 1:
        return 1000.0 * float(batch_min_clearance(arr, arm=arm, scene=scene)[0])
    return 1000.0 * path_min_clearance(list(arr), arm=arm, scene=scene)


def _ik_elbow_down(
    xy: tuple[float, float],
    prev_q: np.ndarray | None = None,
    capsule_radius_m: float = 0.0,
) -> np.ndarray:
    """Analytical 2-link IK wrapping `q1` into `[-pi, pi)` and rejecting `< 5 mm` clearance branches."""
    x, y = float(xy[0]), float(xy[1])
    r2 = x * x + y * y
    raw_cos_q2 = (r2 - LINK_1_M**2 - LINK_2_M**2) / (2.0 * LINK_1_M * LINK_2_M)
    if abs(raw_cos_q2) > 1.0 + 1e-9:
        raise ValueError(f"Target {xy} is outside arm reach [{abs(LINK_1_M - LINK_2_M)}, {LINK_1_M + LINK_2_M}] m")
    cos_q2 = float(np.clip(raw_cos_q2, -1.0, 1.0))

    min_clear_m = MIN_CLEARANCE_MM / 1000.0
    safe_candidates: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        q2 = float(wrap_angle(sign * math.acos(cos_q2)))
        if abs(q2) > MAX_ABS_Q2_RAD:
            continue
        k1 = LINK_1_M + LINK_2_M * math.cos(q2)
        k2 = LINK_2_M * math.sin(q2)
        q1 = float(wrap_angle(math.atan2(y, x) - math.atan2(k2, k1)))
        q = np.array([q1, q2], dtype=np.float64)
        if _config_clearance_m(q, capsule_radius_m=capsule_radius_m) >= min_clear_m:
            safe_candidates.append(q)

    if not safe_candidates:
        raise ValueError(
            f"All IK branches for target {xy} violate >= {MIN_CLEARANCE_MM} mm clearance or |q2| <= {MAX_ABS_Q2_RAD} rad"
        )
    if prev_q is not None and len(safe_candidates) > 1:
        q_ref = np.asarray(prev_q, dtype=np.float64)
        return min(safe_candidates, key=lambda q: float(np.linalg.norm(q - q_ref)))
    return safe_candidates[0]


def _plan_safe_joint_spline(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    safety_margin_m: float = 0.0075,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Plan a smooth collision-free joint path (`>= 5 mm` continuous segment-AABB clearance)."""
    s_grid = np.linspace(0.0, 1.0, 65)[:, None]
    chord = (1.0 - s_grid) * q_start[None, :] + s_grid * q_goal[None, :]
    if _trajectory_min_clearance_mm(chord) >= safety_margin_m * 1000.0:
        return q_start, None

    arm = _slosh_arm()
    scene = _slosh_scene()
    q1_grid, q2_grid = np.meshgrid(np.linspace(-0.8, 2.2, 13), np.linspace(-2.45, 2.45, 15), indexing="ij")
    cands = np.column_stack([q1_grid.ravel(), q2_grid.ravel()])
    s1 = s_grid[None, :, :]
    curves = (
        ((1.0 - s1) ** 2) * q_start[None, None, :]
        + 2.0 * (1.0 - s1) * s1 * cands[:, None, :]
        + (s1**2) * q_goal[None, None, :]
    )
    in_q2 = np.max(np.abs(curves[:, :, 1]), axis=1) <= MAX_ABS_Q2_RAD
    if not bool(np.any(in_q2)):
        raise RuntimeError("No candidate spline satisfies |q2| <= 2.60 rad")

    valid_cands = cands[in_q2]
    valid_curves = curves[in_q2]
    sample_mins_mm = 1000.0 * np.min(batch_min_clearance(valid_curves, arm=arm, scene=scene), axis=1)
    order = np.argsort(-sample_mins_mm)

    best_via: np.ndarray | None = None
    best_clear = -1e9
    for idx in order[:4]:
        if float(sample_mins_mm[idx]) <= best_clear or float(sample_mins_mm[idx]) < MIN_CLEARANCE_MM:
            break
        exact_clear_mm = 1000.0 * path_min_clearance(valid_curves[idx], arm=arm, scene=scene)
        if exact_clear_mm > best_clear:
            best_clear = exact_clear_mm
            best_via = valid_cands[idx].copy()
    if best_via is None or best_clear < MIN_CLEARANCE_MM:
        raise RuntimeError(f"Could not find >= 5 mm clearance path (best={best_clear:.2f} mm)")
    return q_start, best_via


def _eval_joint_kinematics(
    q_start: np.ndarray,
    q_via: np.ndarray | None,
    q_goal: np.ndarray,
    s: np.ndarray,
    ds_dt: np.ndarray,
    d2s_dt2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(q, q_dot, q_ddot, acc_center)` along the linear or quadratic Bezier joint curve."""
    if q_via is None:
        dq_ds = np.broadcast_to(q_goal - q_start, (s.size, 2))
        d2q_ds2 = np.zeros((s.size, 2), dtype=np.float64)
        q = (1.0 - s[:, None]) * q_start[None, :] + s[:, None] * q_goal[None, :]
    else:
        q = (
            ((1.0 - s[:, None]) ** 2) * q_start[None, :]
            + 2.0 * (1.0 - s[:, None]) * s[:, None] * q_via[None, :]
            + (s[:, None] ** 2) * q_goal[None, :]
        )
        dq_ds = (
            2.0 * (1.0 - s[:, None]) * (q_via - q_start)[None, :]
            + 2.0 * s[:, None] * (q_goal - q_via)[None, :]
        )
        d2q_ds2 = np.broadcast_to(2.0 * (q_goal - 2.0 * q_via + q_start), (s.size, 2))

    q_dot = dq_ds * ds_dt[:, None]
    q_ddot = dq_ds * d2s_dt2[:, None] + d2q_ds2 * (ds_dt[:, None] ** 2)

    q1 = q[:, 0]
    q12 = q[:, 0] + q[:, 1]
    q1_dot = q_dot[:, 0]
    q12_dot = q_dot[:, 0] + q_dot[:, 1]
    q1_ddot = q_ddot[:, 0]
    q12_ddot = q_ddot[:, 0] + q_ddot[:, 1]

    ax = (
        -LINK_1_M * np.sin(q1) * q1_ddot
        - LINK_1_M * np.cos(q1) * (q1_dot**2)
        - LINK_2_M * np.sin(q12) * q12_ddot
        - LINK_2_M * np.cos(q12) * (q12_dot**2)
    )
    ay = (
        LINK_1_M * np.cos(q1) * q1_ddot
        - LINK_1_M * np.sin(q1) * (q1_dot**2)
        + LINK_2_M * np.cos(q12) * q12_ddot
        - LINK_2_M * np.sin(q12) * (q12_dot**2)
    )
    return q, q_dot, q_ddot, np.stack([ax, ay], axis=-1)


def _eval_joint_curve(
    q_start: np.ndarray,
    q_via: np.ndarray | None,
    q_goal: np.ndarray,
    s: np.ndarray,
    ds_dt: np.ndarray,
    d2s_dt2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `q(t)` and exact Cartesian end-effector center acceleration `a_ee(t)`."""
    q, _, _, acc_center = _eval_joint_kinematics(q_start, q_via, q_goal, s, ds_dt, d2s_dt2)
    return q, acc_center


def _eval_plate_well_accelerations(
    q_start: np.ndarray,
    q_via: np.ndarray | None,
    q_goal: np.ndarray,
    s: np.ndarray,
    ds_dt: np.ndarray,
    d2s_dt2: np.ndarray,
    plate: OpenWellPlateSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `q(t)` and well accelerations `(T, K, 2)` across plate center + peripheral corner wells."""
    q, q_dot, q_ddot, acc_center = _eval_joint_kinematics(q_start, q_via, q_goal, s, ds_dt, d2s_dt2)
    if not plate.rigid_forearm_mount or not plate.corner_offsets_xy:
        return q, acc_center[:, None, :]

    offsets = [(0.0, 0.0), *plate.corner_offsets_xy]
    q12 = q[:, 0] + q[:, 1]
    w12 = q_dot[:, 0] + q_dot[:, 1]
    a12 = q_ddot[:, 0] + q_ddot[:, 1]
    c12 = np.cos(q12)
    s12 = np.sin(q12)

    well_accels = np.zeros((s.size, len(offsets), 2), dtype=np.float64)
    well_accels[:, 0, :] = acc_center
    for idx_w, (rxb, ryb) in enumerate(offsets[1:], start=1):
        rx = c12 * rxb - s12 * ryb
        ry = s12 * rxb + c12 * ryb
        well_accels[:, idx_w, 0] = acc_center[:, 0] - a12 * ry - (w12**2) * rx
        well_accels[:, idx_w, 1] = acc_center[:, 1] + a12 * rx - (w12**2) * ry
    return q, well_accels


def _simulate_multi_well_slosh(
    well_accels_m_s2: np.ndarray,
    dt_s: float,
    plate: OpenWellPlateSpec,
) -> float:
    """Vectorized RK4 integration of fundamental cylindrical slosh mode across `K` wells `(T, K, 2)`."""
    omega_n = compute_fundamental_slosh_omega(plate.well_radius_m, plate.fill_height_m)
    zeta = plate.damping_ratio
    n_steps, k_wells, _ = well_accels_m_s2.shape
    n_ring = max(10, int(np.ceil(0.12 / max(dt_s, 1e-4))))
    a_pad = np.concatenate(
        [well_accels_m_s2, np.zeros((n_ring, k_wells, 2), dtype=np.float64)],
        axis=0,
    )

    n_sub = max(1, int(np.ceil(omega_n * dt_s / 0.15)))
    h = dt_s / n_sub
    gain = (plate.well_radius_m / GRAVITY_M_S2) * (omega_n**2)

    eta = np.zeros((k_wells, 2), dtype=np.float64)
    eta_dot = np.zeros((k_wells, 2), dtype=np.float64)
    peak_r = 0.0
    w2 = omega_n * omega_n
    two_zw = 2.0 * zeta * omega_n

    for i in range(n_steps + n_ring):
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

        r = float(np.max(np.hypot(eta[:, 0], eta[:, 1])))
        if r > peak_r:
            peak_r = r

    return peak_r


def simulate_meniscus_slosh(
    accel_xy_m_s2: np.ndarray,
    dt_s: float,
    plate: OpenWellPlateSpec,
) -> float:
    """Integrate the 2D fundamental cylindrical pendulum-slosh mode (+ post-stop ringdown).

    Accepts either a single-well acceleration array `(T, 2)` or multi-well `(T, K, 2)`.
    """
    arr = np.asarray(accel_xy_m_s2, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    return _simulate_multi_well_slosh(arr, dt_s, plate)


def _min_jerk_law(tau: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
    ds_dt = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / duration_s
    d2s_dt2 = (60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)) / (duration_s**2)
    return s, ds_dt, d2s_dt2


def _trapezoidal_law(
    n_steps: int,
    duration_s: float,
    accel_frac: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard bang-coast-bang trapezoidal velocity profile on `s in [0, 1]`."""
    tau = np.linspace(0.0, 1.0, n_steps)
    ta = accel_frac
    v_peak = 1.0 / (1.0 - ta)
    a_peak = v_peak / ta

    s = np.zeros_like(tau)
    ds_dtau = np.zeros_like(tau)
    d2s_dtau2 = np.zeros_like(tau)

    m1 = tau < ta
    m2 = (tau >= ta) & (tau <= (1.0 - ta))
    m3 = tau > (1.0 - ta)

    s[m1] = 0.5 * a_peak * (tau[m1] ** 2)
    ds_dtau[m1] = a_peak * tau[m1]
    d2s_dtau2[m1] = a_peak

    s[m2] = 0.5 * a_peak * (ta**2) + v_peak * (tau[m2] - ta)
    ds_dtau[m2] = v_peak
    d2s_dtau2[m2] = 0.0

    rem = 1.0 - tau[m3]
    s[m3] = 1.0 - 0.5 * a_peak * (rem**2)
    ds_dtau[m3] = a_peak * rem
    d2s_dtau2[m3] = -a_peak

    return s, ds_dtau / duration_s, d2s_dtau2 / (duration_s**2)


def plan_unscaled_trapezoidal(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    plate: OpenWellPlateSpec,
    nominal_cartesian_amax_m_s2: float = 5.4,
) -> SloshTrajectoryResult:
    """Baseline cobot trapezoidal profile evaluated across plate center + all corner wells."""
    q_start = _ik_elbow_down(start_xy)
    q_goal = _ik_elbow_down(goal_xy, prev_q=q_start)
    _, q_via = _plan_safe_joint_spline(q_start, q_goal)

    s_probe = np.linspace(0.0, 1.0, 50)
    q_probe, _ = _eval_joint_curve(q_start, q_via, q_goal, s_probe, np.zeros_like(s_probe), np.zeros_like(s_probe))
    _, ee_probe = _fk_positions(q_probe)
    arc_len = float(np.sum(np.linalg.norm(np.diff(ee_probe, axis=0), axis=1)))

    ta = 0.22
    a_peak_norm = 1.0 / (ta * (1.0 - ta))
    duration_s = max(0.26, float(np.sqrt(a_peak_norm * max(arc_len, 0.05) / nominal_cartesian_amax_m_s2)))

    n_steps = 161
    dt_s = duration_s / (n_steps - 1)
    s, ds_dt, d2s_dt2 = _trapezoidal_law(n_steps, duration_s, accel_frac=ta)
    q_traj, well_accels = _eval_plate_well_accelerations(q_start, q_via, q_goal, s, ds_dt, d2s_dt2, plate)

    peak_rise_m = _simulate_multi_well_slosh(well_accels, dt_s, plate)
    omega_n = compute_fundamental_slosh_omega(plate.well_radius_m, plate.fill_height_m)
    h_free_m = plate.free_headspace_m
    ratio = peak_rise_m / h_free_m
    clear_mm = _trajectory_min_clearance_mm(q_traj)
    peak_lat_acc = float(np.max(np.linalg.norm(well_accels, axis=-1)))

    return SloshTrajectoryResult(
        method="unscaled_trapezoidal",
        duration_s=round(duration_s, 4),
        omega_n_rad_s=round(omega_n, 4),
        free_headspace_mm=round(1000.0 * h_free_m, 4),
        peak_meniscus_rise_mm=round(1000.0 * peak_rise_m, 4),
        meniscus_to_headspace_ratio=round(ratio, 4),
        spilled=bool(peak_rise_m >= h_free_m),
        peak_lateral_accel_m_s2=round(peak_lat_acc, 4),
        min_obstacle_clearance_mm=round(clear_mm, 4),
        max_abs_q2_rad=round(float(np.max(np.abs(q_traj[:, 1]))), 4),
    )


def plan_slosh_bounded_min_jerk(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    plate: OpenWellPlateSpec,
    headspace_safety_fraction: float = 0.70,
) -> SloshTrajectoryResult:
    """Solve for minimal duration `T*` keeping worst-case corner-well slosh `< 0.70 * h_free`."""
    if plate.free_headspace_m <= 0.0:
        raise ValueError("Plate has non-positive free headspace")
    q_start = _ik_elbow_down(start_xy)
    q_goal = _ik_elbow_down(goal_xy, prev_q=q_start)
    _, q_via = _plan_safe_joint_spline(q_start, q_goal)

    omega_n = compute_fundamental_slosh_omega(plate.well_radius_m, plate.fill_height_m)
    h_free_m = plate.free_headspace_m
    strict_cap_m = (headspace_safety_fraction - 0.025) * h_free_m

    n_steps = 161
    tau = np.linspace(0.0, 1.0, n_steps)
    s_unit, ds_unit, d2s_unit = _min_jerk_law(tau, 1.0)
    q_traj, well_acc_unit = _eval_plate_well_accelerations(
        q_start, q_via, q_goal, s_unit, ds_unit, d2s_unit, plate
    )

    def _rise_at_duration(dur_s: float) -> tuple[float, np.ndarray]:
        dt = dur_s / (n_steps - 1)
        well_acc = well_acc_unit / (dur_s * dur_s)
        rise = _simulate_multi_well_slosh(well_acc, dt, plate)
        return rise, well_acc

    t_lo = 0.18
    t_hi = 0.90
    rise_hi, _ = _rise_at_duration(t_hi)
    while rise_hi > strict_cap_m and t_hi < 16.0:
        t_hi *= 1.6
        rise_hi, _ = _rise_at_duration(t_hi)

    rise_lo, _ = _rise_at_duration(t_lo)
    if rise_lo <= strict_cap_m:
        t_star = t_lo
    else:
        for _ in range(16):
            t_mid = 0.5 * (t_lo + t_hi)
            r_mid, _ = _rise_at_duration(t_mid)
            if r_mid <= strict_cap_m:
                t_hi = t_mid
            else:
                t_lo = t_mid
        t_star = t_hi

    # Round t_star up to 4 decimals (matching reported duration_s) so evaluating at `res.duration_s`
    # also strictly satisfies `< headspace_safety_fraction * h_free` across all 4 corner wells.
    t_star = math.ceil(t_star * 10000.0) / 10000.0
    peak_rise_m, well_accels = _rise_at_duration(t_star)
    while peak_rise_m >= headspace_safety_fraction * h_free_m:
        t_star = math.ceil((t_star * 1.05) * 10000.0) / 10000.0
        peak_rise_m, well_accels = _rise_at_duration(t_star)

    clear_mm = _trajectory_min_clearance_mm(q_traj)
    assert clear_mm >= MIN_CLEARANCE_MM, f"Clearance {clear_mm} mm < 5.0 mm"
    ratio = peak_rise_m / h_free_m
    peak_lat_acc = float(np.max(np.linalg.norm(well_accels, axis=-1)))

    return SloshTrajectoryResult(
        method="slosh_bounded_min_jerk",
        duration_s=round(t_star, 4),
        omega_n_rad_s=round(omega_n, 4),
        free_headspace_mm=round(1000.0 * h_free_m, 4),
        peak_meniscus_rise_mm=round(1000.0 * peak_rise_m, 4),
        meniscus_to_headspace_ratio=round(ratio, 4),
        spilled=bool(peak_rise_m >= h_free_m),
        peak_lateral_accel_m_s2=round(peak_lat_acc, 4),
        min_obstacle_clearance_mm=round(clear_mm, 4),
        max_abs_q2_rad=round(float(np.max(np.abs(q_traj[:, 1]))), 4),
    )


def run_benchmark(seed: int = 20261123) -> dict[str, Any]:
    """Benchmark 60 open-96-well plate transport tasks around benchtop obstacles.

    Compares `unscaled_trapezoidal` vs `slosh_bounded_min_jerk`, reporting
    `well_spill_rate_pct` and `mean_transport_time_sec` across all plate corner wells.
    """
    rng = np.random.default_rng(seed)
    n_tasks = 60

    trap_spills = 0
    trap_times: list[float] = []
    trap_ratios: list[float] = []

    mj_spills = 0
    mj_times: list[float] = []
    mj_ratios: list[float] = []
    mj_clearances: list[float] = []

    for _ in range(n_tasks):
        start_xy = (float(rng.uniform(-0.09, 0.02)), float(rng.uniform(0.36, 0.44)))
        goal_xy = (float(rng.uniform(0.20, 0.30)), float(rng.uniform(-0.02, 0.06)))
        fill_h = float(rng.uniform(0.00885, 0.00965))
        zeta = float(rng.uniform(0.020, 0.038))
        plate = OpenWellPlateSpec(fill_height_m=fill_h, damping_ratio=zeta)

        trap_res = plan_unscaled_trapezoidal(start_xy, goal_xy, plate)
        mj_res = plan_slosh_bounded_min_jerk(start_xy, goal_xy, plate)

        trap_spills += int(trap_res.spilled)
        trap_times.append(trap_res.duration_s)
        trap_ratios.append(trap_res.meniscus_to_headspace_ratio)

        mj_spills += int(mj_res.spilled)
        mj_times.append(mj_res.duration_s)
        mj_ratios.append(mj_res.meniscus_to_headspace_ratio)
        mj_clearances.append(mj_res.min_obstacle_clearance_mm)

    return {
        "idea_id": "idea_19_open_meniscus_slosh_min_jerk",
        "n_tasks": n_tasks,
        "min_obstacle_clearance_mm": round(min(mj_clearances), 4),
        "unscaled_trapezoidal": {
            "well_spill_rate_pct": round(100.0 * trap_spills / n_tasks, 4),
            "spilled_tasks": trap_spills,
            "mean_transport_time_sec": round(float(np.mean(trap_times)), 4),
            "max_meniscus_to_headspace_ratio": round(max(trap_ratios), 4),
        },
        "slosh_bounded_min_jerk": {
            "well_spill_rate_pct": round(100.0 * mj_spills / n_tasks, 4),
            "spilled_tasks": mj_spills,
            "mean_transport_time_sec": round(float(np.mean(mj_times)), 4),
            "max_meniscus_to_headspace_ratio": round(max(mj_ratios), 4),
            "min_clearance_mm": round(min(mj_clearances), 4),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2))
