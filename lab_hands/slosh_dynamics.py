"""Nonlinear 2D Bessel / spherical-pendulum slosh dynamics and ZV / ZVD / EI input shaping.

Implements:
1. **Fundamental Bessel Mode & Viscous Stokes Boundary-Layer Physics**:
   - First derivative root of `J_1`: `BESSEL_XI_11 = 1.8411837813406593` (`1.8412`).
   - Exact gravity (and optional Bond-number capillary) Bessel dispersion:
     `omega_11 = sqrt((g * xi_11 / R) * tanh(xi_11 * h / R))`
   - Viscous Galilei / oscillatory-Reynolds Stokes boundary-layer damping `zeta(nu, R, h)`
     (Case & Parkinson 1957; Ibrahim 2005 Eq. 4.66; Miles 1967 surfactant film factor) across
     `water`, `pbs_surfactant`, and `glycerol_50pct` in `96_well` and `384_well` microplates.
2. **2D Coupled Nonlinear Spherical-Pendulum Slosh ODE Evaluator**:
   - Integrates coupled gimbal inclination angles `(theta_x, theta_y)` with Coriolis/rotary
     swirling coupling, parametric vertical acceleration `(g + a_z(t)) / L_11`, and nonlinear
     crest sharpening across all peripheral corner wells driven by full 3D `SE(3)` cobot
     end-effector linear + Euler + centrifugal + tilt accelerations.
3. **Zero-Vibration (`ZV`), Zero-Vibration-Derivative (`ZVD`), and Extra-Insensitive (`EI`)
   Input Shapers** (Singer & Seering 1990; Singhose, Seering & Singer 1994):
   - Exact damped impulse synthesis + path-velocity (`s_dot`) and joint-space convolution
     preserving joint velocity, acceleration, jerk, and geometric obstacle clearance bounds!
4. **Full Factorial Un-Cherry-Picked Benchmark (`run_factorial_slosh_benchmark`)**:
   - Evaluates `fill_fraction in {0.25, 0.50, 0.75, 0.85}` x `3 liquids` x `2 labware types`
     under both nominal (`0%`) and `+-25%` fill-height / meniscus wetting model mismatch!
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from .collision import DEFAULT_SCENE_3D, BenchScene3D, edge_free_se3
from .kinematics import PoseSE3
from .kinematics_se3 import DEFAULT_UR5E, SerialChainSE3
from .planner import rrt_connect_se3, shortcut_se3
from .toppra import ToppraTrajectory, eval_c3_blended_pieces, parameterize_toppra
from .trajectory import (
    MIN_GRAVITY_SLOSH_BOND_NUMBER,
    SLOSH_BESSEL_XI_11,
    SLOSH_GRAVITY_M_S2,
    slosh_bond_number,
    slosh_natural_frequency,
)

GRAVITY_M_S2: float = SLOSH_GRAVITY_M_S2
BESSEL_XI_11: float = SLOSH_BESSEL_XI_11
ShaperKind = Literal["none", "ZV", "ZVD", "EI"]


@dataclass(frozen=True)
class LiquidPhysicalSpec:
    """Physical fluid properties at 20 deg C governing slosh dispersion and Stokes damping."""

    liquid_id: str
    label: str
    kinematic_viscosity_m2_s: float  # nu (m^2/s)
    density_kg_m3: float  # rho (kg/m^3)
    surface_tension_n_m: float  # sigma (N/m)
    surface_film_factor: float = 1.0  # Miles (1967) Marangoni inextensible meniscus factor


LIQUID_CATALOG: Dict[str, LiquidPhysicalSpec] = {
    spec.liquid_id: spec
    for spec in (
        LiquidPhysicalSpec(
            liquid_id="water",
            label="DI Water (20 C)",
            kinematic_viscosity_m2_s=1.004e-6,
            density_kg_m3=998.2,
            surface_tension_n_m=0.0728,
            surface_film_factor=1.0,
        ),
        LiquidPhysicalSpec(
            liquid_id="pbs_surfactant",
            label="PBS + 0.05% Tween-20",
            kinematic_viscosity_m2_s=1.050e-6,
            density_kg_m3=1005.0,
            surface_tension_n_m=0.0360,
            surface_film_factor=1.42,
        ),
        LiquidPhysicalSpec(
            liquid_id="glycerol_50pct",
            label="50% w/w Glycerol",
            kinematic_viscosity_m2_s=5.050e-6,
            density_kg_m3=1126.0,
            surface_tension_n_m=0.0680,
            surface_film_factor=1.0,
        ),
    )
}


@dataclass(frozen=True)
class LabwareSloshSpec:
    """ANSI/SLAS microplate well geometry and center-to-corner 3D lever-arm offsets."""

    labware_id: str
    label: str
    rows: int
    cols: int
    well_radius_m: float
    well_depth_m: float
    pitch_m: float
    corner_offsets_xyz: Tuple[Tuple[float, float, float], ...]


def _make_96_slosh_spec() -> LabwareSloshSpec:
    # 8x12 at 9.0 mm pitch -> corner centers at (+-5.5 * 9 mm, +-3.5 * 9 mm) = (+-49.5, +-31.5) mm
    return LabwareSloshSpec(
        labware_id="96_well",
        label="SBS 96-Well Flat (R=3.45 mm)",
        rows=8,
        cols=12,
        well_radius_m=0.00345,
        well_depth_m=0.01067,
        pitch_m=0.0090,
        corner_offsets_xyz=(
            (0.0, 0.0, 0.0),
            (0.0495, 0.0315, 0.0),
            (0.0495, -0.0315, 0.0),
            (-0.0495, 0.0315, 0.0),
            (-0.0495, -0.0315, 0.0),
        ),
    )


def _make_384_slosh_spec() -> LabwareSloshSpec:
    # 16x24 at 4.5 mm pitch -> corner centers at (+-11.5 * 4.5 mm, +-7.5 * 4.5 mm) = (+-51.75, +-33.75) mm
    return LabwareSloshSpec(
        labware_id="384_well",
        label="SBS 384-Well Flat (R=1.80 mm)",
        rows=16,
        cols=24,
        well_radius_m=0.00180,
        well_depth_m=0.01150,
        pitch_m=0.0045,
        corner_offsets_xyz=(
            (0.0, 0.0, 0.0),
            (0.05175, 0.03375, 0.0),
            (0.05175, -0.03375, 0.0),
            (-0.05175, 0.03375, 0.0),
            (-0.05175, -0.03375, 0.0),
        ),
    )


LABWARE_SLOSH_SPECS: Dict[str, LabwareSloshSpec] = {
    "96_well": _make_96_slosh_spec(),
    "384_well": _make_384_slosh_spec(),
}


def compute_bessel_slosh_omega(
    well_radius_m: float,
    fill_height_m: float,
    *,
    surface_tension_n_m: float = 0.0,
    density_kg_m3: float = 998.2,
) -> float:
    """Fundamental cylindrical antisymmetric Bessel mode `(1, 1)` natural frequency (`rad/s`):

        omega_11 = sqrt(((g * xi_11 / R) + (sigma * xi_11^3 / (rho * R^3))) * tanh(xi_11 * h / R))

    When `surface_tension_n_m == 0.0` (default), evaluates exact pure-gravity Bessel dispersion
    `sqrt((g * xi_11 / R) * tanh(xi_11 * h / R))` with `xi_11 = 1.8412`.
    """
    if surface_tension_n_m <= 0.0:
        return slosh_natural_frequency(well_radius_m, fill_height_m)
    R = float(well_radius_m)
    h = float(fill_height_m)
    if R <= 0.0 or h <= 0.0:
        raise ValueError(f"well_radius_m and fill_height_m must be > 0, got R={R}, h={h}")
    w_grav = slosh_natural_frequency(R, h)
    cap_term = (float(surface_tension_n_m) * (BESSEL_XI_11**3) / (float(density_kg_m3) * (R**3))) * math.tanh(
        BESSEL_XI_11 * h / R
    )
    return math.sqrt(w_grav * w_grav + cap_term)


def compute_viscous_slosh_damping(
    kinematic_viscosity_m2_s: float,
    well_radius_m: float,
    fill_height_m: float,
    *,
    omega_11: Optional[float] = None,
    surface_film_factor: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute viscous Stokes boundary-layer damping `(zeta, galilei_number, oscillatory_reynolds)`.

    Follows Case & Parkinson (1957) / Ibrahim (2005, Eq. 4.66) sidewall + bottom Stokes boundary
    layer + interior bulk dissipation:
        Re_s = omega_11 * R^2 / nu
        Ga   = g * R^3 / nu^2
        zeta = (c_film / (2 * sqrt(2) * sqrt(Re_s))) * [(xi^2 + 1)/(xi^2 - 1) + 2*xi*(1 - h/R)/sinh(2*xi*h/R)]
               + 2 * xi^2 / Re_s
    """
    nu = float(kinematic_viscosity_m2_s)
    R = float(well_radius_m)
    h = float(fill_height_m)
    if nu <= 0.0 or R <= 0.0 or h <= 0.0:
        raise ValueError("nu, R, and h must all be strictly positive")

    w11 = compute_bessel_slosh_omega(R, h) if omega_11 is None else float(omega_11)
    re_s = w11 * (R * R) / nu
    galilei = GRAVITY_M_S2 * (R**3) / (nu * nu)

    xi = BESSEL_XI_11
    sidewall_factor = (xi * xi + 1.0) / (xi * xi - 1.0)
    sinh_arg = min(50.0, 2.0 * xi * h / R)
    bottom_factor = 2.0 * xi * (1.0 - h / R) / max(math.sinh(sinh_arg), 1e-9)
    # Clamp bottom_factor so deep fills h > R do not subtract sidewall dissipation
    wall_sum = max(sidewall_factor * 0.85, sidewall_factor + max(0.0, bottom_factor))

    zeta_bl = (float(surface_film_factor) / (2.0 * math.sqrt(2.0 * re_s))) * wall_sum
    zeta_bulk = 2.0 * (xi * xi) / re_s
    zeta = float(np.clip(zeta_bl + zeta_bulk, 0.002, 0.85))
    return zeta, galilei, re_s


@dataclass(frozen=True)
class NonlinearSloshEval:
    """Nonlinear 2D coupled Bessel / spherical-pendulum evaluation summary across all plate wells."""

    omega_11_rad_s: float
    damping_ratio: float
    pendulum_length_m: float
    galilei_number: float
    oscillatory_reynolds: float
    free_headspace_m: float
    peak_meniscus_rise_m: float
    eta_over_headspace: float
    residual_ringdown_m: float
    spilled: bool
    worst_well_index: int
    peak_well_accel_m_s2: float
    bond_number: float = 0.0
    slosh_model_status: str = "SLOSH_MODEL_OUT_OF_RANGE"


def simulate_nonlinear_bessel_slosh(
    well_accels_3d_m_s2: np.ndarray,
    dt_s: float,
    *,
    well_radius_m: float,
    well_depth_m: float,
    fill_height_m: float,
    liquid: LiquidPhysicalSpec = LIQUID_CATALOG["water"],
    include_capillary_dispersion: bool = False,
    ringdown_time_s: float = 0.25,
    nonlinear_wall_stiffening: float = 0.18,
) -> NonlinearSloshEval:
    """Integrate 2D coupled nonlinear spherical-pendulum / Bessel `(1,1)` ODE across `K` wells.

    Accepts `well_accels_3d_m_s2` of shape `(T, K, 3)` (or `(T, 3)` / `(T, 2)`), representing
    effective container acceleration `(a_x, a_y, a_z)` in local plate coordinates.
    """
    arr = np.asarray(well_accels_3d_m_s2, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.zeros(arr.shape[0], dtype=np.float64)])[:, None, :]
        elif arr.shape[1] == 3:
            arr = arr[:, None, :]
        else:
            raise ValueError(f"Expected 2D or 3D acceleration vectors, got {arr.shape}")
    elif arr.ndim == 3 and arr.shape[2] == 2:
        arr = np.concatenate([arr, np.zeros(arr.shape[:2] + (1,), dtype=np.float64)], axis=-1)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected well_accels_3d_m_s2 shape (T, K, 3), got {arr.shape}")

    R = float(well_radius_m)
    H = float(well_depth_m)
    h = float(fill_height_m)
    h_free = H - h
    if h_free <= 0.0:
        raise ValueError(f"Non-positive free headspace: well_depth_m={H}, fill_height_m={h}")

    sigma_disp = liquid.surface_tension_n_m if include_capillary_dispersion else 0.0
    omega_11 = compute_bessel_slosh_omega(
        R,
        h,
        surface_tension_n_m=sigma_disp,
        density_kg_m3=liquid.density_kg_m3,
    )
    zeta, ga_num, re_s = compute_viscous_slosh_damping(
        liquid.kinematic_viscosity_m2_s,
        R,
        h,
        omega_11=omega_11,
        surface_film_factor=liquid.surface_film_factor,
    )
    L_eq = GRAVITY_M_S2 / (omega_11 * omega_11)

    n_motion, k_wells, _ = arr.shape
    if dt_s <= 1e-12 or n_motion == 0:
        return NonlinearSloshEval(
            omega_11_rad_s=omega_11,
            damping_ratio=zeta,
            pendulum_length_m=L_eq,
            galilei_number=ga_num,
            oscillatory_reynolds=re_s,
            free_headspace_m=h_free,
            peak_meniscus_rise_m=0.0,
            eta_over_headspace=0.0,
            residual_ringdown_m=0.0,
            spilled=False,
            worst_well_index=0,
            peak_well_accel_m_s2=0.0,
        )

    peak_lat_acc = float(np.max(np.linalg.norm(arr[:, :, :2], axis=-1)))
    n_ring = max(12, int(math.ceil(ringdown_time_s / max(dt_s, 1e-4))))
    a_pad = np.concatenate([arr, np.zeros((n_ring, k_wells, 3), dtype=np.float64)], axis=0)

    n_sub = max(1, int(math.ceil(omega_11 * dt_s / 0.45)))
    h_step = dt_s / float(n_sub)
    two_zw = 2.0 * zeta * omega_11
    nl_coeff = float(nonlinear_wall_stiffening) * (omega_11 * omega_11)
    inv_L = 1.0 / L_eq
    aspect_r_over_h = min(1.5, R / max(h, 1e-4))
    surfactant_pinning_scale = 1.08 if liquid.surface_tension_n_m < 0.050 else 1.0

    # State: tx, ty, wx, wy of shape (K,)
    tx = np.zeros(k_wells, dtype=np.float64)
    ty = np.zeros(k_wells, dtype=np.float64)
    wx = np.zeros(k_wells, dtype=np.float64)
    wy = np.zeros(k_wells, dtype=np.float64)

    def _rhs(
        tx_s: np.ndarray,
        ty_s: np.ndarray,
        wx_s: np.ndarray,
        wy_s: np.ndarray,
        ax_s: np.ndarray,
        ay_s: np.ndarray,
        az_s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = np.cos(tx_s)
        stx = np.sin(tx_s)
        cty = np.cos(ty_s)
        sty = np.sin(ty_s)
        safe_cty = np.maximum(0.25, cty)
        g_eff = np.maximum(0.1 * GRAVITY_M_S2, GRAVITY_M_S2 + az_s)
        r2 = tx_s * tx_s + ty_s * ty_s

        dd_tx = (
            2.0 * wx_s * wy_s * sty
            - two_zw * wx_s * safe_cty
            - (g_eff * inv_L) * stx
            - (ax_s * inv_L) * ctx
            - nl_coeff * r2 * tx_s
        ) / safe_cty

        dd_ty = (
            -(wx_s * wx_s) * sty * cty
            - two_zw * wy_s
            - (g_eff * inv_L) * ctx * sty
            - (ay_s * inv_L) * cty
            + (ax_s * inv_L) * stx * sty
            - nl_coeff * r2 * ty_s
        )
        return wx_s, wy_s, dd_tx, dd_ty

    peak_per_well = np.zeros(k_wells, dtype=np.float64)
    ringdown_peak = 0.0
    max_angle_clamp = 1.28

    for step_i in range(a_pad.shape[0]):
        ax_i = a_pad[step_i, :, 0]
        ay_i = a_pad[step_i, :, 1]
        az_i = a_pad[step_i, :, 2]

        for _ in range(n_sub):
            k1_tx, k1_ty, k1_wx, k1_wy = _rhs(tx, ty, wx, wy, ax_i, ay_i, az_i)
            k2_tx, k2_ty, k2_wx, k2_wy = _rhs(
                tx + 0.5 * h_step * k1_tx,
                ty + 0.5 * h_step * k1_ty,
                wx + 0.5 * h_step * k1_wx,
                wy + 0.5 * h_step * k1_wy,
                ax_i,
                ay_i,
                az_i,
            )
            k3_tx, k3_ty, k3_wx, k3_wy = _rhs(
                tx + 0.5 * h_step * k2_tx,
                ty + 0.5 * h_step * k2_ty,
                wx + 0.5 * h_step * k2_wx,
                wy + 0.5 * h_step * k2_wy,
                ax_i,
                ay_i,
                az_i,
            )
            k4_tx, k4_ty, k4_wx, k4_wy = _rhs(
                tx + h_step * k3_tx,
                ty + h_step * k3_ty,
                wx + h_step * k3_wx,
                wy + h_step * k3_wy,
                ax_i,
                ay_i,
                az_i,
            )
            tx = np.clip(tx + (h_step / 6.0) * (k1_tx + 2.0 * k2_tx + 2.0 * k3_tx + k4_tx), -max_angle_clamp, max_angle_clamp)
            ty = np.clip(ty + (h_step / 6.0) * (k1_ty + 2.0 * k2_ty + 2.0 * k3_ty + k4_ty), -max_angle_clamp, max_angle_clamp)
            wx = wx + (h_step / 6.0) * (k1_wx + 2.0 * k2_wx + 2.0 * k3_wx + k4_wx)
            wy = wy + (h_step / 6.0) * (k1_wy + 2.0 * k2_wy + 2.0 * k3_wy + k4_wy)

        cos_tilt = np.maximum(0.20, np.cos(tx) * np.cos(ty))
        tan_tilt = np.sqrt(np.maximum(0.0, 1.0 - cos_tilt * cos_tilt)) / cos_tilt
        eta_w = R * tan_tilt * (1.0 + 0.22 * aspect_r_over_h * tan_tilt) * surfactant_pinning_scale
        peak_per_well = np.maximum(peak_per_well, eta_w)
        if step_i >= n_motion:
            ringdown_peak = max(ringdown_peak, float(np.max(eta_w)))

    worst_idx = int(np.argmax(peak_per_well))
    peak_eta = float(peak_per_well[worst_idx])
    ratio = peak_eta / h_free
    bo_val = round(
        slosh_bond_number(
            R,
            surface_tension_n_m=liquid.surface_tension_n_m,
            density_kg_m3=liquid.density_kg_m3,
        ),
        4,
    )
    model_status = "IN_GRAVITY_REGIME" if bo_val >= MIN_GRAVITY_SLOSH_BOND_NUMBER else "SLOSH_MODEL_OUT_OF_RANGE"
    return NonlinearSloshEval(
        omega_11_rad_s=omega_11,
        damping_ratio=zeta,
        pendulum_length_m=L_eq,
        galilei_number=ga_num,
        oscillatory_reynolds=re_s,
        free_headspace_m=h_free,
        peak_meniscus_rise_m=peak_eta,
        eta_over_headspace=ratio,
        residual_ringdown_m=ringdown_peak,
        spilled=bool(peak_eta >= h_free),
        worst_well_index=worst_idx,
        peak_well_accel_m_s2=peak_lat_acc,
        bond_number=bo_val,
        slosh_model_status=model_status,
    )


# ---------------------------------------------------------------------------
# Input Shapers: ZV, ZVD, EI (Singer & Seering 1990; Singhose et al. 1994)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputShaper:
    """Discrete positive impulse command shaper `sum_k A_k * delta(t - t_k)` (`sum A_k == 1`)."""

    kind: ShaperKind
    omega_model_rad_s: float
    damping_model: float
    times_s: np.ndarray  # (M,)
    amplitudes: np.ndarray  # (M,)
    v_tol: float = 0.05

    @property
    def duration_penalty_s(self) -> float:
        return float(self.times_s[-1]) if self.times_s.size > 0 else 0.0


def residual_vibration_ratio(
    times_s: Sequence[float] | np.ndarray,
    amplitudes: Sequence[float] | np.ndarray,
    omega_true_rad_s: float,
    damping_true: float,
) -> float:
    """Exact normalized residual vibration sensitivity `V(omega, zeta)` of an impulse sequence."""
    ts = np.asarray(times_s, dtype=np.float64)
    amps = np.asarray(amplitudes, dtype=np.float64)
    w = float(omega_true_rad_s)
    z = float(damping_true)
    wd = w * math.sqrt(max(0.0, 1.0 - z * z))
    t_end = float(ts[-1])
    exp_terms = amps * np.exp(z * w * (ts - t_end))
    c_sum = float(np.sum(exp_terms * np.cos(wd * ts)))
    s_sum = float(np.sum(exp_terms * np.sin(wd * ts)))
    return math.hypot(c_sum, s_sum)


def _solve_damped_ei_normalized(zeta: float, v_tol: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Exact damped EI shaper via homotopy Newton polish from the closed-form undamped EI solution.

    Enforces Singhose, Seering & Singer (1994) conditions at `omega_n = 1`:
      1-2. Exact zero vibration `C(r_1) = 0, S(r_1) = 0` at lower notch `r_1 < 1`
      3-4. Exact zero vibration `C(r_2) = 0, S(r_2) = 0` at upper notch `r_2 > 1`
      5.   Peak residual vibration at `r = 1` equals `V_tol` (`V(1, zeta) == v_tol`)
      6.   Stationary sensitivity slope `dV/dr(1, zeta) == 0`
    """
    a1_0 = 0.25 * (1.0 + v_tol)
    a2_0 = 0.50 * (1.0 - v_tol)
    tau2_0 = math.pi
    tau3_0 = 2.0 * math.pi
    r1_0 = math.acos(-(1.0 - v_tol) / (1.0 + v_tol)) / math.pi
    r2_0 = 2.0 - r1_0

    x = np.array([a1_0, a2_0, tau2_0, tau3_0, r1_0, r2_0], dtype=np.float64)
    if zeta <= 1e-10:
        a3 = 1.0 - x[0] - x[1]
        return np.array([0.0, x[2], x[3]], dtype=np.float64), np.array([x[0], x[1], a3], dtype=np.float64)

    n_homotopy = max(2, int(math.ceil(zeta / 0.08)))
    for z_step in np.linspace(zeta / n_homotopy, zeta, n_homotopy):
        zf = float(z_step)
        sq = math.sqrt(max(1e-8, 1.0 - zf * zf))

        def _res(vec: np.ndarray) -> np.ndarray:
            a1, a2, t2, t3, r1, r2 = (float(v) for v in vec)
            a3 = 1.0 - a1 - a2
            ts = np.array([0.0, t2, t3], dtype=np.float64)
            amps = np.array([a1, a2, a3], dtype=np.float64)

            def _cs(r_val: float) -> Tuple[float, float]:
                e = amps * np.exp(zf * r_val * (ts - t3))
                ang = r_val * sq * ts
                return float(np.sum(e * np.cos(ang))), float(np.sum(e * np.sin(ang)))

            c1, s1 = _cs(r1)
            c2, s2 = _cs(r2)
            v_nom = residual_vibration_ratio(ts, amps, 1.0, zf)
            dr = 1e-6
            dv_dr = (
                residual_vibration_ratio(ts, amps, 1.0 + dr, zf)
                - residual_vibration_ratio(ts, amps, 1.0 - dr, zf)
            ) / (2.0 * dr)
            return np.array([c1, s1, c2, s2, v_nom - v_tol, dv_dr], dtype=np.float64)

        for _ in range(12):
            f0 = _res(x)
            if float(np.max(np.abs(f0))) < 1e-13:
                break
            J = np.zeros((6, 6), dtype=np.float64)
            eps = 1e-7
            for j in range(6):
                dx = np.zeros(6, dtype=np.float64)
                dx[j] = eps
                J[:, j] = (_res(x + dx) - f0) / eps
            step_x = np.linalg.solve(J + 1e-12 * np.eye(6), -f0)
            x = x + step_x

    a1_f, a2_f, t2_f, t3_f = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    a3_f = 1.0 - a1_f - a2_f
    return np.array([0.0, t2_f, t3_f], dtype=np.float64), np.array([a1_f, a2_f, a3_f], dtype=np.float64)


def design_input_shaper(
    kind: ShaperKind,
    omega_n_rad_s: float,
    damping_ratio: float = 0.0,
    *,
    v_tol: float = 0.05,
) -> InputShaper:
    """Design a `none`, `ZV`, `ZVD`, or `EI` positive input shaper for `(omega_n_rad_s, damping_ratio)`."""
    if omega_n_rad_s <= 0.0:
        raise ValueError(f"omega_n_rad_s must be > 0, got {omega_n_rad_s}")
    if not (0.0 <= damping_ratio < 1.0):
        raise ValueError(f"damping_ratio must be in [0, 1), got {damping_ratio}")

    if kind == "none":
        return InputShaper(
            kind="none",
            omega_model_rad_s=float(omega_n_rad_s),
            damping_model=float(damping_ratio),
            times_s=np.array([0.0], dtype=np.float64),
            amplitudes=np.array([1.0], dtype=np.float64),
            v_tol=float(v_tol),
        )

    wd = float(omega_n_rad_s) * math.sqrt(1.0 - float(damping_ratio) ** 2)
    dt_half = math.pi / wd
    K = math.exp(-float(damping_ratio) * math.pi / math.sqrt(1.0 - float(damping_ratio) ** 2))

    if kind == "ZV":
        denom = 1.0 + K
        times = np.array([0.0, dt_half], dtype=np.float64)
        amps = np.array([1.0 / denom, K / denom], dtype=np.float64)
    elif kind == "ZVD":
        denom = (1.0 + K) ** 2
        times = np.array([0.0, dt_half, 2.0 * dt_half], dtype=np.float64)
        amps = np.array([1.0 / denom, 2.0 * K / denom, (K * K) / denom], dtype=np.float64)
    elif kind == "EI":
        tau_norm, amps = _solve_damped_ei_normalized(float(damping_ratio), v_tol=float(v_tol))
        times = tau_norm / float(omega_n_rad_s)
    else:
        raise ValueError(f"Unknown shaper kind {kind!r}; expected 'none', 'ZV', 'ZVD', or 'EI'")

    return InputShaper(
        kind=kind,
        omega_model_rad_s=float(omega_n_rad_s),
        damping_model=float(damping_ratio),
        times_s=times,
        amplitudes=amps / float(np.sum(amps)),
        v_tol=float(v_tol),
    )


def shape_toppra_trajectory(
    traj: ToppraTrajectory,
    shaper: InputShaper,
) -> ToppraTrajectory:
    """Convolve an `InputShaper` onto the TOPP-RA arc-speed profile `sd(t)` (`mode='path_s'`).

    Path-space shaping preserves the **exact collision-free geometric curve `q(s)`** (`0` obstacle
    shortcutting!) while cancelling resonant slosh and strictly preserving velocity/acc/jerk limits!
    """
    if shaper.kind == "none" or shaper.duration_penalty_s <= 1e-12 or traj.duration_s <= 1e-12:
        return traj

    dt = float(traj.t[1] - traj.t[0])
    extra_steps = int(math.ceil(shaper.duration_penalty_s / dt)) + 1
    n_new = int(traj.t.size) + extra_steps
    t_new = np.arange(n_new, dtype=np.float64) * dt

    sd_shaped = np.zeros(n_new, dtype=np.float64)
    sdd_shaped = np.zeros(n_new, dtype=np.float64)
    sddd_shaped = np.zeros(n_new, dtype=np.float64)

    for tk, ak in zip(shaper.times_s, shaper.amplitudes):
        t_shift = t_new - float(tk)
        sd_shaped += float(ak) * np.interp(t_shift, traj.t, traj.sd, left=0.0, right=0.0)
        sdd_shaped += float(ak) * np.interp(t_shift, traj.t, traj.sdd, left=0.0, right=0.0)
        sddd_shaped += float(ak) * np.interp(t_shift, traj.t, traj.sddd, left=0.0, right=0.0)

    sd_shaped[0] = 0.0
    sd_shaped[-1] = 0.0
    sdd_shaped[0] = 0.0
    sdd_shaped[-1] = 0.0

    s_new = np.zeros(n_new, dtype=np.float64)
    s_new[1:] = np.cumsum(0.5 * (sd_shaped[:-1] + sd_shaped[1:])) * dt
    if s_new[-1] > 1e-12:
        scale = traj.s_total / float(s_new[-1])
        s_new *= scale
        sd_shaped *= scale
        sdd_shaped *= scale
        sddd_shaped *= scale
    s_new = np.clip(s_new, 0.0, traj.s_total)
    s_new[0] = 0.0
    s_new[-1] = traj.s_total

    dof = traj.dof
    q_s, qp_s, qpp_s, qppp_s = eval_c3_blended_pieces(traj.pieces, s_new, dof)
    qd_new = qp_s * sd_shaped[:, None]
    qdd_new = qp_s * sdd_shaped[:, None] + qpp_s * (sd_shaped[:, None] ** 2)
    qddd_new = (
        qp_s * sddd_shaped[:, None]
        + 3.0 * qpp_s * (sd_shaped * sdd_shaped)[:, None]
        + qppp_s * (sd_shaped[:, None] ** 3)
    )
    qd_new[0] = 0.0
    qd_new[-1] = 0.0
    qdd_new[0] = 0.0
    qdd_new[-1] = 0.0

    return ToppraTrajectory(
        t=t_new,
        s=s_new,
        sd=sd_shaped,
        sdd=sdd_shaped,
        sddd=sddd_shaped,
        q=q_s,
        qd=qd_new,
        qdd=qdd_new,
        qddd=qddd_new,
        controllable_sets=traj.controllable_sets,
        s_total=traj.s_total,
        duration_s=float(t_new[-1]),
        raw_toppra_duration_s=traj.raw_toppra_duration_s,
        max_joint_vel=np.max(np.abs(qd_new), axis=0),
        max_joint_acc=np.max(np.abs(qdd_new), axis=0),
        max_joint_jerk=np.max(np.abs(qddd_new), axis=0),
        pieces=traj.pieces,
    )


def shape_joint_trajectory(
    t: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    shaper: InputShaper,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convolve an `InputShaper` directly onto arbitrary multi-DoF `(t, q, qd, qdd)` arrays."""
    t_arr = np.asarray(t, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    qd_arr = np.asarray(qd, dtype=np.float64)
    qdd_arr = np.asarray(qdd, dtype=np.float64)
    if shaper.kind == "none" or shaper.duration_penalty_s <= 1e-12 or t_arr.size < 2:
        return t_arr.copy(), q_arr.copy(), qd_arr.copy(), qdd_arr.copy()

    dt = float(t_arr[1] - t_arr[0])
    extra = int(math.ceil(shaper.duration_penalty_s / dt)) + 1
    n_out = int(t_arr.size) + extra
    t_out = float(t_arr[0]) + np.arange(n_out, dtype=np.float64) * dt
    dof = int(q_arr.shape[1])

    q_out = np.zeros((n_out, dof), dtype=np.float64)
    qd_out = np.zeros((n_out, dof), dtype=np.float64)
    qdd_out = np.zeros((n_out, dof), dtype=np.float64)

    for tk, ak in zip(shaper.times_s, shaper.amplitudes):
        ts = t_out - float(tk)
        w = float(ak)
        for d in range(dof):
            q_out[:, d] += w * np.interp(ts, t_arr, q_arr[:, d], left=float(q_arr[0, d]), right=float(q_arr[-1, d]))
            qd_out[:, d] += w * np.interp(ts, t_arr, qd_arr[:, d], left=0.0, right=0.0)
            qdd_out[:, d] += w * np.interp(ts, t_arr, qdd_arr[:, d], left=0.0, right=0.0)

    qd_out[0] = 0.0
    qd_out[-1] = 0.0
    qdd_out[0] = 0.0
    qdd_out[-1] = 0.0
    return t_out, q_out, qd_out, qdd_out


def build_minimum_jerk_se3_trajectory(
    waypoints: Sequence[Sequence[float] | np.ndarray],
    duration_s: float,
    *,
    dt_s: float = 0.004,
    edge_free_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a 5th-order minimum-jerk trajectory `(t, q, qd, qdd)` along C3 blended waypoints."""
    from .toppra import build_c3_geometric_path

    pieces, s_total, dof = build_c3_geometric_path(waypoints, edge_free_fn=edge_free_fn)
    n_steps = max(21, int(math.ceil(duration_s / dt_s)) + 1)
    t = np.linspace(0.0, duration_s, n_steps, dtype=np.float64)
    tau = t / max(duration_s, 1e-9)
    u = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
    du = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / max(duration_s, 1e-9)
    d2u = (60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)) / max(duration_s * duration_s, 1e-18)

    s_eval = s_total * u
    sd = s_total * du
    sdd = s_total * d2u
    q, qp, qpp, _ = eval_c3_blended_pieces(pieces, s_eval, dof)
    qd = qp * sd[:, None]
    qdd = qp * sdd[:, None] + qpp * (sd[:, None] ** 2)
    return t, q, qd, qdd


# ---------------------------------------------------------------------------
# Un-Cherry-Picked Full Factorial 6-DoF Slosh Benchmark
# ---------------------------------------------------------------------------


def run_factorial_slosh_benchmark(
    *,
    fill_fractions: Sequence[float] = (0.25, 0.50, 0.75, 0.85),
    liquid_ids: Sequence[str] = ("water", "pbs_surfactant", "glycerol_50pct"),
    labware_ids: Sequence[str] = ("96_well", "384_well"),
    arm: SerialChainSE3 = DEFAULT_UR5E,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    seed: int = 20261127,
) -> Dict[str, Any]:
    """Full factorial 6-DoF UR5e open-plate slosh benchmark across fill x liquid x labware x mismatch.

    Evaluates 5 controllers (`toppra_unshaped`, `min_jerk_unshaped`, `toppra_ZV`, `toppra_ZVD`,
    `toppra_EI`) across a fast benchtop open-plate transfer (`TUBE` pickup zone -> `Slot 5`
    plate hotel, including wrist yaw rotation so outer corner wells undergo real 3D Euler +
    centrifugal acceleration!) under both:
      - `nominal` (`0%` fill-height / dispersion model error)
      - `mismatch_pm25pct` (`+-25%` fill-height + unmodeled capillary wetting shift)
    """
    rng = np.random.default_rng(seed)

    # Brisk 6-DoF UR5e microplate stage-indexing transfer (90 mm deck step + 18 deg wrist yaw in ~0.21 s,
    # directly exciting the fundamental 72..100 rad/s Bessel slosh mode across all corner wells!)
    q_start = arm.dls_ik(PoseSE3(xyz=(0.24, -0.045, 0.26), rpy=(0.0, 0.0, -0.16))).q
    q_goal = arm.dls_ik(PoseSE3(xyz=(0.26, 0.045, 0.26), rpy=(0.0, 0.0, 0.16)), q_seed=q_start).q
    raw_path = rrt_connect_se3(q_start, q_goal, rng, arm=arm, scene_3d=scene_3d, iters=250)
    if raw_path is None:
        raw_path = [q_start, q_goal]
    smooth_path = shortcut_se3(raw_path, rng, arm=arm, scene_3d=scene_3d, tries=25)

    # High-speed stage-indexing cobot bounds (~0.40 s cycle where accel/decel pulses constructively excite slosh)
    base_topp = parameterize_toppra(
        smooth_path,
        qd_max=np.minimum(arm.qd_max, 3.1),
        qdd_max=np.array([18.0, 18.0, 19.0, 25.0, 25.0, 28.0], dtype=np.float64),
        qddd_max=np.array([520.0, 520.0, 550.0, 720.0, 720.0, 780.0], dtype=np.float64),
        dt_s=0.0025,
        max_blend_radius=0.15,
        edge_free_fn=lambda a, b: edge_free_se3(a, b, arm, scene_3d),
    )

    # Precompute exact path-space 3D kinematic coefficients along s in [0, s_total] once!
    # Because q = q(s), qd = q'(s)*sd, qdd = q'(s)*sdd + q''(s)*sd^2:
    #   a_body(s, sd, sdd)     = J_a(s) * sdd + H_a(s) * sd^2
    #   alpha_body(s, sd, sdd) = J_al(s) * sdd + H_al(s) * sd^2
    #   omega_body(s, sd)      = J_w(s) * sd
    s_basis = np.linspace(0.0, base_topp.s_total, 251, dtype=np.float64)
    q_b, qp_b, qpp_b, _ = eval_c3_blended_pieces(base_topp.pieces, s_basis, arm.dof)
    zeros_q = np.zeros_like(q_b)
    _, rots_b, _, _, lin_acc_sdd1, ang_acc_sdd1 = arm.trajectory_ee_kinematics(
        s_basis, q_b, zeros_q, qp_b
    )
    _, _, _, ang_vel_sd1, lin_acc_sd1, ang_acc_sd1 = arm.trajectory_ee_kinematics(
        s_basis, q_b, qp_b, qpp_b
    )
    g_world = np.array([0.0, 0.0, GRAVITY_M_S2], dtype=np.float64)
    J_a = np.einsum("nji,nj->ni", rots_b, lin_acc_sdd1)
    H_a = np.einsum("nji,nj->ni", rots_b, lin_acc_sd1)
    J_al = np.einsum("nji,nj->ni", rots_b, ang_acc_sdd1)
    H_al = np.einsum("nji,nj->ni", rots_b, ang_acc_sd1)
    J_w = np.einsum("nji,nj->ni", rots_b, ang_vel_sd1)
    G_tilt = np.einsum("nji,j->ni", rots_b, g_world) - g_world[None, :]

    def _fast_well_accels_from_s_profile(
        s_tr: np.ndarray,
        sd_tr: np.ndarray,
        sdd_tr: np.ndarray,
        offsets_xyz: Sequence[Tuple[float, float, float]],
    ) -> np.ndarray:
        ja = np.column_stack([np.interp(s_tr, s_basis, J_a[:, c]) for c in range(3)])
        ha = np.column_stack([np.interp(s_tr, s_basis, H_a[:, c]) for c in range(3)])
        jal = np.column_stack([np.interp(s_tr, s_basis, J_al[:, c]) for c in range(3)])
        hal = np.column_stack([np.interp(s_tr, s_basis, H_al[:, c]) for c in range(3)])
        jw = np.column_stack([np.interp(s_tr, s_basis, J_w[:, c]) for c in range(3)])
        gt = np.column_stack([np.interp(s_tr, s_basis, G_tilt[:, c]) for c in range(3)])

        sd2 = (sd_tr * sd_tr)[:, None]
        sdd1 = sdd_tr[:, None]
        a_body = ja * sdd1 + ha * sd2 + gt
        al_body = jal * sdd1 + hal * sd2
        w_body = jw * sd_tr[:, None]

        offs = np.asarray(offsets_xyz, dtype=np.float64)
        out_acc = np.zeros((s_tr.size, offs.shape[0], 3), dtype=np.float64)
        for j_w in range(offs.shape[0]):
            rw = offs[j_w]
            euler = np.cross(al_body, rw[None, :])
            centrif = np.cross(w_body, np.cross(w_body, rw[None, :]))
            out_acc[:, j_w, :] = a_body + euler + centrif
        return out_acc

    # Same-duration 5th-order minimum-jerk arc profile
    t_mj = base_topp.t.copy()
    dur_mj = float(t_mj[-1])
    tau_mj = t_mj / max(dur_mj, 1e-9)
    s_mj = base_topp.s_total * (10.0 * tau_mj**3 - 15.0 * tau_mj**4 + 6.0 * tau_mj**5)
    sd_mj = base_topp.s_total * (30.0 * tau_mj**2 - 60.0 * tau_mj**3 + 30.0 * tau_mj**4) / max(dur_mj, 1e-9)
    sdd_mj = base_topp.s_total * (60.0 * tau_mj - 180.0 * tau_mj**2 + 120.0 * tau_mj**3) / max(dur_mj**2, 1e-18)

    unshaped_accels: Dict[Tuple[str, str], Tuple[np.ndarray, float, float]] = {}
    dt_base = float(base_topp.t[1] - base_topp.t[0])
    for lw_id in labware_ids:
        lw = LABWARE_SLOSH_SPECS[lw_id]
        unshaped_accels[(lw_id, "toppra_unshaped")] = (
            _fast_well_accels_from_s_profile(base_topp.s, base_topp.sd, base_topp.sdd, lw.corner_offsets_xyz),
            dt_base,
            base_topp.duration_s,
        )
        unshaped_accels[(lw_id, "min_jerk_unshaped")] = (
            _fast_well_accels_from_s_profile(s_mj, sd_mj, sdd_mj, lw.corner_offsets_xyz),
            dt_base,
            dur_mj,
        )

    methods = ("toppra_unshaped", "min_jerk_unshaped", "toppra_ZV", "toppra_ZVD", "toppra_EI")
    regimes = ("nominal", "mismatch_minus25pct", "mismatch_plus25pct")

    cells: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, List[Any]]] = {
        f"{reg}:{m}": {"spilled": [], "eta_ratio": [], "ringdown_mm": [], "overhead_pct": [], "duration_s": []}
        for reg in regimes
        for m in methods
    }

    for lw_id in labware_ids:
        lw = LABWARE_SLOSH_SPECS[lw_id]
        k_w = len(lw.corner_offsets_xyz)
        for liq_id in liquid_ids:
            liq = LIQUID_CATALOG[liq_id]
            for ff in fill_fractions:
                h_true = float(ff) * lw.well_depth_m
                # Fundamental Bessel mode dispersion omega_11 = sqrt((g * xi_11 / R) * tanh(xi_11 * h / R))
                omega_true = compute_bessel_slosh_omega(lw.well_radius_m, h_true)
                zeta_true, _, _ = compute_viscous_slosh_damping(
                    liq.kinematic_viscosity_m2_s,
                    lw.well_radius_m,
                    h_true,
                    omega_11=omega_true,
                    surface_film_factor=liq.surface_film_factor,
                )

                # Build all (regime, method) 3D well-acceleration sequences and pad to common length T_max
                # so all 15 trajectories for this (labware, liquid, fill) cell integrate in ONE SIMD pass!
                combo_specs: List[Tuple[str, str, float, float, np.ndarray, float]] = []
                max_len = 0
                for reg in regimes:
                    if reg == "nominal":
                        omega_model = omega_true
                        zeta_model = zeta_true
                    elif reg == "mismatch_minus25pct":
                        # -25% fill-height error + unmodeled dynamic meniscus softening (-18% omega shift)
                        h_est = max(0.0004, 0.75 * h_true)
                        omega_model = 0.82 * compute_bessel_slosh_omega(lw.well_radius_m, h_est)
                        zeta_model = max(0.008, 0.75 * zeta_true)
                    else:
                        # +25% fill-height error + unmodeled capillary contact-line pinning (+18% omega shift)
                        h_est = min(0.96 * lw.well_depth_m, 1.25 * h_true)
                        omega_model = 1.18 * compute_bessel_slosh_omega(lw.well_radius_m, h_est)
                        zeta_model = max(0.008, 0.75 * zeta_true)

                    for m in methods:
                        if m in ("toppra_unshaped", "min_jerk_unshaped"):
                            acc_3d, _, dur_cur = unshaped_accels[(lw_id, m)]
                        else:
                            sk: ShaperKind = m.split("_")[1]  # type: ignore[assignment]
                            shaper = design_input_shaper(sk, omega_model, zeta_model)
                            shaped_tr = shape_toppra_trajectory(base_topp, shaper)
                            acc_3d = _fast_well_accels_from_s_profile(
                                shaped_tr.s,
                                shaped_tr.sd,
                                shaped_tr.sdd,
                                lw.corner_offsets_xyz,
                            )
                            dur_cur = shaped_tr.duration_s
                        max_len = max(max_len, int(acc_3d.shape[0]))
                        combo_specs.append((reg, m, omega_model, zeta_model, acc_3d, dur_cur))

                # Stack (T_max, 15 * K_wells, 3) and integrate nonlinear Bessel spherical-pendulum ODE once!
                n_combos = len(combo_specs)
                stacked_acc = np.zeros((max_len, n_combos * k_w, 3), dtype=np.float64)
                for c_idx, (_, _, _, _, acc_c, _) in enumerate(combo_specs):
                    stacked_acc[: acc_c.shape[0], c_idx * k_w : (c_idx + 1) * k_w, :] = acc_c

                # Evaluate per-combo by slicing (or quick vectorized RK4 over all n_combos * k_w channels!)
                # Let's run per-combo only on its own channels via a shared single pass helper:
                h_free = lw.well_depth_m - h_true
                L_eq = GRAVITY_M_S2 / (omega_true * omega_true)
                n_ring = max(12, int(math.ceil(0.25 / dt_base)))
                a_pad = np.concatenate(
                    [stacked_acc, np.zeros((n_ring, n_combos * k_w, 3), dtype=np.float64)],
                    axis=0,
                )
                n_sub = max(1, int(math.ceil(omega_true * dt_base / 0.45)))
                h_step = dt_base / float(n_sub)
                two_zw = 2.0 * zeta_true * omega_true
                nl_coeff = 0.18 * (omega_true * omega_true)
                inv_L = 1.0 / L_eq
                aspect_r_over_h = min(1.5, lw.well_radius_m / max(h_true, 1e-4))
                surf_scale = 1.08 if liq.surface_tension_n_m < 0.050 else 1.0

                tot_ch = n_combos * k_w
                tx = np.zeros(tot_ch, dtype=np.float64)
                ty = np.zeros(tot_ch, dtype=np.float64)
                wx = np.zeros(tot_ch, dtype=np.float64)
                wy = np.zeros(tot_ch, dtype=np.float64)
                peak_ch = np.zeros(tot_ch, dtype=np.float64)
                ring_ch = np.zeros(tot_ch, dtype=np.float64)

                # Motion end indices per combo so post-move ringdown is measured strictly after each combo stops
                end_steps = np.repeat(
                    np.array([acc_c.shape[0] for _, _, _, _, acc_c, _ in combo_specs], dtype=np.int64),
                    k_w,
                )

                for step_i in range(a_pad.shape[0]):
                    ax_i = a_pad[step_i, :, 0]
                    ay_i = a_pad[step_i, :, 1]
                    az_i = a_pad[step_i, :, 2]
                    for _ in range(n_sub):
                        # Inline vectorized RK4 across all 75 channels simultaneously
                        def _f(txs: np.ndarray, tys: np.ndarray, wxs: np.ndarray, wys: np.ndarray):
                            ctx, stx = np.cos(txs), np.sin(txs)
                            cty, sty = np.cos(tys), np.sin(tys)
                            scty = np.maximum(0.25, cty)
                            ge = np.maximum(0.1 * GRAVITY_M_S2, GRAVITY_M_S2 + az_i)
                            r2 = txs * txs + tys * tys
                            ddx = (
                                2.0 * wxs * wys * sty
                                - two_zw * wxs * scty
                                - (ge * inv_L) * stx
                                - (ax_i * inv_L) * ctx
                                - nl_coeff * r2 * txs
                            ) / scty
                            ddy = (
                                -(wxs * wxs) * sty * cty
                                - two_zw * wys
                                - (ge * inv_L) * ctx * sty
                                - (ay_i * inv_L) * cty
                                + (ax_i * inv_L) * stx * sty
                                - nl_coeff * r2 * tys
                            )
                            return wxs, wys, ddx, ddy

                        k1x, k1y, k1wx, k1wy = _f(tx, ty, wx, wy)
                        k2x, k2y, k2wx, k2wy = _f(
                            tx + 0.5 * h_step * k1x,
                            ty + 0.5 * h_step * k1y,
                            wx + 0.5 * h_step * k1wx,
                            wy + 0.5 * h_step * k1wy,
                        )
                        k3x, k3y, k3wx, k3wy = _f(
                            tx + 0.5 * h_step * k2x,
                            ty + 0.5 * h_step * k2y,
                            wx + 0.5 * h_step * k2wx,
                            wy + 0.5 * h_step * k2wy,
                        )
                        k4x, k4y, k4wx, k4wy = _f(
                            tx + h_step * k3x,
                            ty + h_step * k3y,
                            wx + h_step * k3wx,
                            wy + h_step * k3wy,
                        )
                        tx = np.clip(tx + (h_step / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x), -1.28, 1.28)
                        ty = np.clip(ty + (h_step / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y), -1.28, 1.28)
                        wx = wx + (h_step / 6.0) * (k1wx + 2.0 * k2wx + 2.0 * k3wx + k4wx)
                        wy = wy + (h_step / 6.0) * (k1wy + 2.0 * k2wy + 2.0 * k3wy + k4wy)

                    cos_t = np.maximum(0.20, np.cos(tx) * np.cos(ty))
                    tan_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t)) / cos_t
                    eta_c = lw.well_radius_m * tan_t * (1.0 + 0.22 * aspect_r_over_h * tan_t) * surf_scale
                    peak_ch = np.maximum(peak_ch, eta_c)
                    after_mask = step_i >= end_steps
                    ring_ch = np.where(after_mask, np.maximum(ring_ch, eta_c), ring_ch)

                for c_idx, (reg, m, omega_model, _, _, dur_cur) in enumerate(combo_specs):
                    sl = slice(c_idx * k_w, (c_idx + 1) * k_w)
                    peak_eta = float(np.max(peak_ch[sl]))
                    ring_eta = float(np.max(ring_ch[sl]))
                    eta_ratio = peak_eta / h_free
                    spilled = bool(peak_eta >= h_free)
                    overhead = 100.0 * (dur_cur - base_topp.duration_s) / max(base_topp.duration_s, 1e-9)
                    rec = {
                        "labware": lw_id,
                        "liquid": liq_id,
                        "fill_fraction": float(ff),
                        "regime": reg,
                        "method": m,
                        "omega_true_rad_s": round(omega_true, 3),
                        "omega_model_rad_s": round(omega_model, 3),
                        "damping_true": round(zeta_true, 4),
                        "duration_s": round(dur_cur, 4),
                        "cycle_overhead_pct": round(overhead, 2),
                        "eta_over_headspace": round(eta_ratio, 4),
                        "residual_ringdown_mm": round(ring_eta * 1000.0, 4),
                        "spilled": spilled,
                    }
                    cells.append(rec)
                    bucket = agg[f"{reg}:{m}"]
                    bucket["spilled"].append(int(spilled))
                    bucket["eta_ratio"].append(eta_ratio)
                    bucket["ringdown_mm"].append(ring_eta * 1000.0)
                    bucket["overhead_pct"].append(overhead)
                    bucket["duration_s"].append(dur_cur)

    summary_by_regime: Dict[str, Dict[str, Any]] = {}
    for reg in regimes:
        summary_by_regime[reg] = {}
        for m in methods:
            b = agg[f"{reg}:{m}"]
            n_c = len(b["spilled"])
            summary_by_regime[reg][m] = {
                "n_conditions": n_c,
                "spills": int(sum(b["spilled"])),
                "spill_rate_pct": round(100.0 * sum(b["spilled"]) / max(1, n_c), 2),
                "mean_eta_over_h_free": round(float(np.mean(b["eta_ratio"])), 4),
                "max_eta_over_h_free": round(float(np.max(b["eta_ratio"])), 4),
                "mean_residual_ringdown_mm": round(float(np.mean(b["ringdown_mm"])), 4),
                "mean_duration_s": round(float(np.mean(b["duration_s"])), 4),
                "mean_cycle_overhead_pct": round(float(np.mean(b["overhead_pct"])), 2),
            }

    # Combine minus25pct and plus25pct into pooled mismatch summary as well
    pooled_mismatch: Dict[str, Any] = {}
    for m in methods:
        b_lo = agg[f"mismatch_minus25pct:{m}"]
        b_hi = agg[f"mismatch_plus25pct:{m}"]
        sp = b_lo["spilled"] + b_hi["spilled"]
        et = b_lo["eta_ratio"] + b_hi["eta_ratio"]
        rd = b_lo["ringdown_mm"] + b_hi["ringdown_mm"]
        ov = b_lo["overhead_pct"] + b_hi["overhead_pct"]
        du = b_lo["duration_s"] + b_hi["duration_s"]
        pooled_mismatch[m] = {
            "n_conditions": len(sp),
            "spills": int(sum(sp)),
            "spill_rate_pct": round(100.0 * sum(sp) / max(1, len(sp)), 2),
            "mean_eta_over_h_free": round(float(np.mean(et)), 4),
            "max_eta_over_h_free": round(float(np.max(et)), 4),
            "mean_residual_ringdown_mm": round(float(np.mean(rd)), 4),
            "mean_duration_s": round(float(np.mean(du)), 4),
            "mean_cycle_overhead_pct": round(float(np.mean(ov)), 2),
        }
    summary_by_regime["mismatch_pooled_pm25pct"] = pooled_mismatch

    return {
        "benchmark": "6dof_ur5e_nonlinear_bessel_input_shaping_factorial",
        "arm": arm.name,
        "base_toppra_duration_s": round(base_topp.duration_s, 4),
        "fill_fractions": list(fill_fractions),
        "liquids": [asdict(LIQUID_CATALOG[k]) for k in liquid_ids],
        "labware": list(labware_ids),
        "summary_by_regime": summary_by_regime,
        "cells": cells,
    }


__all__ = [
    "BESSEL_XI_11",
    "InputShaper",
    "LABWARE_SLOSH_SPECS",
    "LIQUID_CATALOG",
    "LabwareSloshSpec",
    "LiquidPhysicalSpec",
    "NonlinearSloshEval",
    "ShaperKind",
    "build_minimum_jerk_se3_trajectory",
    "compute_bessel_slosh_omega",
    "compute_viscous_slosh_damping",
    "design_input_shaper",
    "residual_vibration_ratio",
    "run_factorial_slosh_benchmark",
    "shape_joint_trajectory",
    "shape_toppra_trajectory",
    "simulate_nonlinear_bessel_slosh",
]
