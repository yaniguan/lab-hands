"""Idea #19: Open-Meniscus Pendulum-Slosh Minimum-Jerk Trajectory Time-Scaler.

Inspired by UniCT-ARSLab/DMDTE and high-speed open-plate cobot transport threads.
Models each open 96-well meniscus via its fundamental cylindrical slosh mode:
    omega_n = sqrt(1.841 * g / R) * tanh(1.841 * h / R)
with fill headspace `h_free = H_well - h`, and solves for the minimal trajectory
duration `T*` under a 5th-order minimum-jerk time-scaling law
    s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
so peak dynamic meniscus rise stays strictly below `0.7 * h_free` (`0` spills)
while maintaining `>= 5 mm` continuous segment-AABB obstacle clearance on a
planar 2-link laboratory arm (`L1 = 0.30 m`, `L2 = 0.25 m`, `|q2| <= 2.60 rad`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GRAVITY_M_S2: float = 9.80665
BESSEL_XI_11: float = 1.841
LINK_1_M: float = 0.30
LINK_2_M: float = 0.25
MAX_ABS_Q2_RAD: float = 2.60
MIN_CLEARANCE_MM: float = 5.0

# Standard benchtop obstacle boxes (xmin, ymin, xmax, ymax) in meters from lab-hands:
# tube rack wall + reagent bottle between pickup zone (-0.05, 0.40) and plate zone (0.20, 0.05)
BENCH_OBSTACLES_M: tuple[tuple[float, float, float, float], ...] = (
    (0.33, -0.08, 0.36, 0.12),
    (0.15, 0.27, 0.24, 0.34),
)


@dataclass(frozen=True)
class OpenWellPlateSpec:
    """Geometry and liquid state of an open 96-well assay plate during transport."""

    well_radius_m: float = 0.00345      # 6.9 mm diameter standard 96-well flat bottom
    well_depth_m: float = 0.01067       # 10.67 mm total well depth (~399 uL brim)
    fill_height_m: float = 0.00920      # liquid fill height `h`
    damping_ratio: float = 0.028        # viscous boundary-layer damping ratio `zeta`

    @property
    def free_headspace_m(self) -> float:
        return float(self.well_depth_m - self.fill_height_m)


@dataclass(frozen=True)
class SloshTrajectoryResult:
    """Evaluated open-plate transport trajectory metrics."""

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
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    elbow = np.stack([LINK_1_M * np.cos(q1), LINK_1_M * np.sin(q1)], axis=-1)
    ee = elbow + np.stack([LINK_2_M * np.cos(q12), LINK_2_M * np.sin(q12)], axis=-1)
    return elbow, ee


def _ik_elbow_down(xy: tuple[float, float]) -> np.ndarray:
    """Analytical 2-link inverse kinematics preferring positive elbow (`q2 >= 0`) when safe."""
    x, y = float(xy[0]), float(xy[1])
    r2 = x * x + y * y
    cos_q2 = (r2 - LINK_1_M ** 2 - LINK_2_M ** 2) / (2.0 * LINK_1_M * LINK_2_M)
    cos_q2 = float(np.clip(cos_q2, -0.999999, 0.999999))
    candidates: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        q2 = sign * float(np.arccos(cos_q2))
        if abs(q2) > MAX_ABS_Q2_RAD:
            continue
        k1 = LINK_1_M + LINK_2_M * np.cos(q2)
        k2 = LINK_2_M * np.sin(q2)
        q1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
        q = np.array([q1, q2], dtype=np.float64)
        if _config_clearance_m(q) >= (MIN_CLEARANCE_MM / 1000.0):
            return q
        candidates.append(q)
    if not candidates:
        raise ValueError(f"Target {xy} unreachable within |q2| <= {MAX_ABS_Q2_RAD} rad")
    return max(candidates, key=lambda q: _config_clearance_m(q))


def _point_to_aabb_dist_m(pts: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = box
    dx = np.maximum(np.maximum(xmin - pts[..., 0], 0.0), pts[..., 0] - xmax)
    dy = np.maximum(np.maximum(ymin - pts[..., 1], 0.0), pts[..., 1] - ymax)
    inside = (pts[..., 0] >= xmin) & (pts[..., 0] <= xmax) & (pts[..., 1] >= ymin) & (pts[..., 1] <= ymax)
    dist = np.hypot(dx, dy)
    return np.where(inside, -1e-3, dist)


def _config_clearance_m(q: np.ndarray) -> float:
    """Minimum distance (m) from both arm links to all benchtop AABB obstacles."""
    base = np.zeros(2, dtype=np.float64)
    elbow, ee = _fk_positions(q)
    lambdas = np.linspace(0.0, 1.0, 25)[:, None]
    seg1 = base[None, :] * (1.0 - lambdas) + elbow[None, :] * lambdas
    seg2 = elbow[None, :] * (1.0 - lambdas) + ee[None, :] * lambdas
    pts = np.concatenate([seg1, seg2], axis=0)
    min_d = float("inf")
    for box in BENCH_OBSTACLES_M:
        min_d = min(min_d, float(np.min(_point_to_aabb_dist_m(pts, box))))
    return min_d


def _trajectory_min_clearance_mm(q_traj: np.ndarray) -> float:
    """Dense swept clearance check across all trajectory configurations in millimeters."""
    elbow, ee = _fk_positions(q_traj)
    base = np.zeros_like(elbow)
    lambdas = np.linspace(0.0, 1.0, 21)[None, :, None]
    seg1 = base[:, None, :] * (1.0 - lambdas) + elbow[:, None, :] * lambdas
    seg2 = elbow[:, None, :] * (1.0 - lambdas) + ee[:, None, :] * lambdas
    pts = np.concatenate([seg1, seg2], axis=1)
    min_d = float("inf")
    for box in BENCH_OBSTACLES_M:
        min_d = min(min_d, float(np.min(_point_to_aabb_dist_m(pts, box))))
    return 1000.0 * min_d


def _plan_safe_joint_spline(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    safety_margin_m: float = 0.0075,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Plan a smooth collision-free joint path (`>= 5 mm` obstacle clearance).

    If the straight joint chord violates `safety_margin_m`, selects a deterministic
    safe retracted elbow via-point `q_mid` so the quadratic Bezier curve in joint space
    cleanly clears all benchtop bottles/racks with smooth C^infinity spatial derivatives.
    """
    s_grid = np.linspace(0.0, 1.0, 65)[:, None]
    chord = (1.0 - s_grid) * q_start[None, :] + s_grid * q_goal[None, :]
    if _trajectory_min_clearance_mm(chord) >= safety_margin_m * 1000.0:
        return q_start, None

    # Search deterministic candidate via-points across safe elbow-folded corridor
    best_via: np.ndarray | None = None
    best_clear = -1e9
    for q1_cand in np.linspace(-0.8, 2.2, 13):
        for q2_cand in np.linspace(-2.45, 2.45, 15):
            cand = np.array([float(q1_cand), float(q2_cand)], dtype=np.float64)
            curve = (
                ((1.0 - s_grid) ** 2) * q_start[None, :]
                + 2.0 * (1.0 - s_grid) * s_grid * cand[None, :]
                + (s_grid ** 2) * q_goal[None, :]
            )
            if np.max(np.abs(curve[:, 1])) > MAX_ABS_Q2_RAD:
                continue
            clear_mm = _trajectory_min_clearance_mm(curve)
            if clear_mm > best_clear:
                best_clear = clear_mm
                best_via = cand
    if best_via is None or best_clear < MIN_CLEARANCE_MM:
        raise RuntimeError(f"Could not find >= 5 mm clearance path (best={best_clear:.2f} mm)")
    return q_start, best_via


def _eval_joint_curve(
    q_start: np.ndarray,
    q_via: np.ndarray | None,
    q_goal: np.ndarray,
    s: np.ndarray,
    ds_dt: np.ndarray,
    d2s_dt2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `q(t)` and exact Cartesian end-effector lateral acceleration `a_ee(t)`."""
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
        - LINK_1_M * np.cos(q1) * (q1_dot ** 2)
        - LINK_2_M * np.sin(q12) * q12_ddot
        - LINK_2_M * np.cos(q12) * (q12_dot ** 2)
    )
    ay = (
        LINK_1_M * np.cos(q1) * q1_ddot
        - LINK_1_M * np.sin(q1) * (q1_dot ** 2)
        + LINK_2_M * np.cos(q12) * q12_ddot
        - LINK_2_M * np.sin(q12) * (q12_dot ** 2)
    )
    return q, np.stack([ax, ay], axis=-1)


def simulate_meniscus_slosh(
    accel_xy_m_s2: np.ndarray,
    dt_s: float,
    plate: OpenWellPlateSpec,
) -> float:
    """Integrate the 2D fundamental cylindrical pendulum-slosh mode (+ post-stop ringdown).

    Solves `eta_ddot + 2*zeta*omega_n*eta_dot + omega_n^2*eta = (R / g) * omega_n^2 * a_lat(t)`
    using an exact unconditionally-stable sub-stepped semi-implicit / RK4 integrator and
    returns peak radial meniscus displacement `max_t ||eta(t)||` in meters.
    """
    omega_n = compute_fundamental_slosh_omega(plate.well_radius_m, plate.fill_height_m)
    zeta = plate.damping_ratio
    # Include 0.12 s post-stop free oscillation ringdown
    n_ring = max(10, int(np.ceil(0.12 / max(dt_s, 1e-4))))
    a_pad = np.vstack([accel_xy_m_s2, np.zeros((n_ring, 2), dtype=np.float64)])

    # Sub-step so omega_n * h_sub <= 0.15 for high-accuracy RK4 integration
    n_sub = max(1, int(np.ceil(omega_n * dt_s / 0.15)))
    h = dt_s / n_sub
    gain = (plate.well_radius_m / GRAVITY_M_S2) * (omega_n ** 2)

    eta = np.zeros(2, dtype=np.float64)
    eta_dot = np.zeros(2, dtype=np.float64)
    peak_r = 0.0
    w2 = omega_n * omega_n
    two_zw = 2.0 * zeta * omega_n

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

        r = float(np.hypot(eta[0], eta[1]))
        if r > peak_r:
            peak_r = r

    return peak_r


def _min_jerk_law(tau: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
    ds_dt = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / duration_s
    d2s_dt2 = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (duration_s ** 2)
    return s, ds_dt, d2s_dt2


def _trapezoidal_law(n_steps: int, duration_s: float, accel_frac: float = 0.22) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard bang-coast-bang trapezoidal velocity profile on `s in [0, 1]`."""
    tau = np.linspace(0.0, 1.0, n_steps)
    ta = accel_frac
    # Normalization: integral_0^1 v(tau) dtau = 1 => v_peak * (1 - ta) = 1
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

    s[m2] = 0.5 * a_peak * (ta ** 2) + v_peak * (tau[m2] - ta)
    ds_dtau[m2] = v_peak
    d2s_dtau2[m2] = 0.0

    rem = 1.0 - tau[m3]
    s[m3] = 1.0 - 0.5 * a_peak * (rem ** 2)
    ds_dtau[m3] = a_peak * rem
    d2s_dtau2[m3] = -a_peak

    return s, ds_dtau / duration_s, d2s_dtau2 / (duration_s ** 2)


def plan_unscaled_trapezoidal(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    plate: OpenWellPlateSpec,
    nominal_cartesian_amax_m_s2: float = 5.4,
) -> SloshTrajectoryResult:
    """Baseline cobot trapezoidal profile ignoring liquid fill headspace and slosh modes."""
    q_start = _ik_elbow_down(start_xy)
    q_goal = _ik_elbow_down(goal_xy)
    _, q_via = _plan_safe_joint_spline(q_start, q_goal)

    # Estimate path length in task space to pick standard high-speed trapezoidal duration
    s_probe = np.linspace(0.0, 1.0, 50)
    q_probe, _ = _eval_joint_curve(q_start, q_via, q_goal, s_probe, np.zeros_like(s_probe), np.zeros_like(s_probe))
    _, ee_probe = _fk_positions(q_probe)
    arc_len = float(np.sum(np.linalg.norm(np.diff(ee_probe, axis=0), axis=1)))

    # Standard trapezoidal timing for `a_peak_s / T^2 * arc_len ~= nominal_cartesian_amax`
    ta = 0.22
    a_peak_norm = 1.0 / (ta * (1.0 - ta))
    duration_s = max(0.26, float(np.sqrt(a_peak_norm * max(arc_len, 0.05) / nominal_cartesian_amax_m_s2)))

    n_steps = 161
    dt_s = duration_s / (n_steps - 1)
    s, ds_dt, d2s_dt2 = _trapezoidal_law(n_steps, duration_s, accel_frac=ta)
    q_traj, accel_xy = _eval_joint_curve(q_start, q_via, q_goal, s, ds_dt, d2s_dt2)

    peak_rise_m = simulate_meniscus_slosh(accel_xy, dt_s, plate)
    omega_n = compute_fundamental_slosh_omega(plate.well_radius_m, plate.fill_height_m)
    h_free_m = plate.free_headspace_m
    ratio = peak_rise_m / h_free_m
    clear_mm = _trajectory_min_clearance_mm(q_traj)

    return SloshTrajectoryResult(
        method="unscaled_trapezoidal",
        duration_s=round(duration_s, 4),
        omega_n_rad_s=round(omega_n, 4),
        free_headspace_mm=round(1000.0 * h_free_m, 4),
        peak_meniscus_rise_mm=round(1000.0 * peak_rise_m, 4),
        meniscus_to_headspace_ratio=round(ratio, 4),
        spilled=bool(peak_rise_m >= h_free_m),
        peak_lateral_accel_m_s2=round(float(np.max(np.linalg.norm(accel_xy, axis=1))), 4),
        min_obstacle_clearance_mm=round(clear_mm, 4),
        max_abs_q2_rad=round(float(np.max(np.abs(q_traj[:, 1]))), 4),
    )


def plan_slosh_bounded_min_jerk(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    plate: OpenWellPlateSpec,
    headspace_safety_fraction: float = 0.70,
) -> SloshTrajectoryResult:
    """Solve for minimal duration `T*` under 5th-order minimum-jerk keeping slosh `< 0.7 * h_free`."""
    if plate.free_headspace_m <= 0.0:
        raise ValueError("Plate has non-positive free headspace")
    q_start = _ik_elbow_down(start_xy)
    q_goal = _ik_elbow_down(goal_xy)
    _, q_via = _plan_safe_joint_spline(q_start, q_goal)

    omega_n = compute_fundamental_slosh_omega(plate.well_radius_m, plate.fill_height_m)
    h_free_m = plate.free_headspace_m
    # Target strictly inside `0.70 * h_free` (e.g., `0.675 * h_free`)
    strict_cap_m = (headspace_safety_fraction - 0.025) * h_free_m

    n_steps = 161
    tau = np.linspace(0.0, 1.0, n_steps)

    def _rise_at_duration(dur_s: float) -> tuple[float, np.ndarray, np.ndarray]:
        dt = dur_s / (n_steps - 1)
        s, ds_dt, d2s_dt2 = _min_jerk_law(tau, dur_s)
        q_tr, acc_xy = _eval_joint_curve(q_start, q_via, q_goal, s, ds_dt, d2s_dt2)
        rise = simulate_meniscus_slosh(acc_xy, dt, plate)
        return rise, q_tr, acc_xy

    # Bracket [t_lo, t_hi] and solve via monotonic bisection for minimal safe T*
    t_lo = 0.18
    t_hi = 0.90
    rise_hi, _, _ = _rise_at_duration(t_hi)
    while rise_hi > strict_cap_m and t_hi < 12.0:
        t_hi *= 1.6
        rise_hi, _, _ = _rise_at_duration(t_hi)

    rise_lo, _, _ = _rise_at_duration(t_lo)
    if rise_lo <= strict_cap_m:
        t_star = t_lo
    else:
        for _ in range(16):
            t_mid = 0.5 * (t_lo + t_hi)
            r_mid, _, _ = _rise_at_duration(t_mid)
            if r_mid <= strict_cap_m:
                t_hi = t_mid
            else:
                t_lo = t_mid
        t_star = t_hi

    peak_rise_m, q_traj, accel_xy = _rise_at_duration(t_star)
    # Guarantee strict inequality `< 0.70 * h_free` even if minor resonant ripples appear
    while peak_rise_m >= headspace_safety_fraction * h_free_m:
        t_star *= 1.05
        peak_rise_m, q_traj, accel_xy = _rise_at_duration(t_star)

    clear_mm = _trajectory_min_clearance_mm(q_traj)
    assert clear_mm >= MIN_CLEARANCE_MM, f"Clearance {clear_mm} mm < 5.0 mm"
    ratio = peak_rise_m / h_free_m

    return SloshTrajectoryResult(
        method="slosh_bounded_min_jerk",
        duration_s=round(t_star, 4),
        omega_n_rad_s=round(omega_n, 4),
        free_headspace_mm=round(1000.0 * h_free_m, 4),
        peak_meniscus_rise_mm=round(1000.0 * peak_rise_m, 4),
        meniscus_to_headspace_ratio=round(ratio, 4),
        spilled=bool(peak_rise_m >= h_free_m),
        peak_lateral_accel_m_s2=round(float(np.max(np.linalg.norm(accel_xy, axis=1))), 4),
        min_obstacle_clearance_mm=round(clear_mm, 4),
        max_abs_q2_rad=round(float(np.max(np.abs(q_traj[:, 1]))), 4),
    )


def run_benchmark(seed: int = 20261123) -> dict[str, Any]:
    """Benchmark 60 open-96-well plate transport tasks around benchtop obstacles.

    Compares `unscaled_trapezoidal` vs `slosh_bounded_min_jerk`, reporting
    `well_spill_rate_pct` and `mean_transport_time_sec`.
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
        # Sample pickup near benchtop hotel zone (-0.05, 0.40) and dropoff across 96-well deck (0.20..0.30, -0.02..0.06)
        start_xy = (float(rng.uniform(-0.09, 0.02)), float(rng.uniform(0.36, 0.44)))
        goal_xy = (float(rng.uniform(0.20, 0.30)), float(rng.uniform(-0.02, 0.06)))
        # High-fill open 96-well assay plates (fill 8.85..9.65 mm inside 10.67 mm well => h_free 1.02..1.82 mm)
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
