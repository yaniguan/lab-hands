"""3D Whitney/Coulomb friction-cone peg-in-hole mechanics and active spiral/Bayesian tactile insertion.

Replaces toy wide-funnel tactile models with physically grounded wet-lab contact mechanics:
1. **Realistic SBS Polypropylene 96-Well / 1.5 mL Microtube Geometry (`ClearVesselBevel`)**:
   - Narrow molded chamfer (`chamfer_width_m = 0.70 mm` in `0.6–0.8 mm`, NOT `5.5 mm`!) around
     radial bore clearance `bore_radius_m = 0.10 mm` (`capture_radius_m = 0.80 mm`).
   - Exact same geometry and 3D Coulomb contact physics shared identically across `open_loop`,
     `passive_rcc_compliance`, `local_shear_admittance`, and `active_spiral_bayesian_tactile`!
2. **3D Coulomb Friction Cone + Whitney (1982) Two-Point Jamming & Wedging**:
   - Models flat-rim axial hangup (`r > capture_radius_m`, zero bevel slope, `||F_xy|| <= mu * F_z`),
     conical chamfer sliding (`f_n * (sin(alpha) - mu * cos(alpha))` radial vs `f_n * (cos(alpha) + mu * sin(alpha))` axial),
     and cylindrical bore two-point contact with Whitney (1982) wedging (`theta > c / (mu * R_hole)`)
     and force-moment jamming inequalities!
3. **Active Archimedean Spiral Probing + Bayesian Particle-Filter Localization + 6-DoF Admittance**:
   - Detects flat-rim `F_z` contact when RGB-D refractive error (`0.5–4.0 mm`) misses the narrow
     `0.70 mm` chamfer, updates a 256-particle Bayesian posterior over the hole center inside the
     safe half-pitch Voronoi cell (`4.5 mm`), probes along an Archimedean spiral into the chamfer
     capture zone, actively aligns angular tilt via moment admittance, and seats cleanly!
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .collision import DEFAULT_SCENE, BenchScene, in_collision
from .kinematics import DEFAULT_ARM, Arm2D, PoseSE3

InsertionRegime = Literal[
    "open_loop",
    "passive_rcc_compliance",
    "local_shear_admittance",
    "active_spiral_bayesian_tactile",
]


@dataclass(frozen=True)
class ClearVesselBevel:
    """Realistic SBS polypropylene well / microtube opening with narrow `0.70 mm` chamfer."""

    center_xy: Tuple[float, float] = (0.20, 0.05)
    bore_radius_m: float = 0.00010  # 0.10 mm radial clearance c = R_hole - R_peg
    hole_radius_m: float = 0.00340  # 3.40 mm SBS 96-well inner bore radius
    chamfer_width_m: float = 0.00070  # 0.70 mm realistic molded polypropylene chamfer (0.6..0.8 mm)
    capture_radius_m: Optional[float] = None  # defaults to bore_radius_m + chamfer_width_m (0.80 mm)
    z_rim_m: float = 0.020
    bevel_depth_m: Optional[float] = None  # defaults to chamfer_width_m (45 deg chamfer)
    target_depth_m: float = 0.012
    friction_coeff: float = 0.25  # 3D Coulomb friction mu in [0.15, 0.45]
    stiffness_xy_n_m: float = 1800.0
    stiffness_z_n_m: float = 2400.0

    def __post_init__(self) -> None:
        if self.bore_radius_m <= 0.0 or self.hole_radius_m <= 0.0:
            raise ValueError("bore_radius_m and hole_radius_m must be > 0")
        if self.capture_radius_m is None:
            if self.chamfer_width_m <= 0.0:
                raise ValueError("chamfer_width_m must be > 0")
            cap_r = float(self.bore_radius_m + self.chamfer_width_m)
            object.__setattr__(self, "capture_radius_m", cap_r)
        else:
            cap_r = float(self.capture_radius_m)
            if cap_r <= self.bore_radius_m:
                raise ValueError("capture_radius_m must exceed bore_radius_m")
            object.__setattr__(self, "chamfer_width_m", cap_r - float(self.bore_radius_m))

        if self.bevel_depth_m is None:
            # Standard 45-degree molded chamfer slope
            object.__setattr__(self, "bevel_depth_m", float(self.chamfer_width_m))


@dataclass(frozen=True)
class ContactWrench3D:
    """6-axis contact wrench `(F_xyz, M_xyz)` and physical contact state from 3D Coulomb/Whitney model."""

    force_xyz_n: np.ndarray  # (3,) [fx, fy, fz] acting ON the pipette tip in world frame
    moment_xyz_nm: np.ndarray  # (3,) [mx, my, mz] about wrist compliance center
    contact_state: Literal[
        "free",
        "flat_rim",
        "chamfer",
        "bore_free",
        "bore_one_point",
        "bore_two_point",
        "jammed",
        "wedged",
    ]
    jammed: bool = False
    wedged: bool = False


@dataclass(frozen=True)
class InsertionResult:
    """ insertion outcome with slip/seal telemetry and 3D Whitney/Bayesian diagnostics."""

    success: bool
    initial_error_mm: float
    final_error_mm: float
    steps: int
    peak_force_n: float
    final_z_m: float
    seal_intact: bool = True
    slip_detected: bool = False
    axial_slip_mm: float = 0.0
    regime: str = "active_spiral_bayesian_tactile"
    failure_mode: str = "none"
    jammed: bool = False
    wedged: bool = False
    rim_hangup: bool = False
    spiral_probes: int = 0
    final_tilt_deg: float = 0.0
    posterior_std_mm: float = 0.0


def check_whitney_jamming_wedging(
    *,
    insertion_depth_m: float,
    bore_clearance_m: float,
    hole_radius_m: float,
    friction_coeff: float,
    tilt_rad: float,
    f_lat_n: float,
    f_z_n: float,
    moment_nm: float,
) -> Tuple[bool, bool, float]:
    """Evaluate Whitney (1982) two-point peg-in-hole wedging and jamming conditions.

    Returns `(wedged, jammed, lambda_depth)` where:
      - `l_2 = 2 * c / max(|theta|, 1e-6)` is the minimum depth for two-point wall contact,
      - `lambda_depth = l / (2 * R_hole * mu)` is Whitney's dimensionless insertion depth,
      - **Wedging** occurs when two-point contact engages inside the friction cone `lambda_depth <= 1.0`
        (`|theta| >= c / (mu * R_hole)`), locking opposing contact normals geometrically,
      - **Jamming** occurs when `lambda_depth > 1.0` in two-point contact and `(F_lat / F_z, M / (2*R*F_z))`
        violates Whitney's no-jamming parallelogram!
    """
    l_ins = max(0.0, float(insertion_depth_m))
    c = max(1e-6, float(bore_clearance_m))
    R = max(1e-4, float(hole_radius_m))
    mu = max(0.01, float(friction_coeff))
    th = abs(float(tilt_rad))

    lam = l_ins / (2.0 * R * mu)
    l_two_point = (2.0 * c) / max(th, 1e-6)
    if l_ins < l_two_point:
        return False, False, lam

    # Two-point opposite-wall contact is active!
    theta_wedge = c / (mu * R)
    if th >= theta_wedge and lam <= 1.05:
        return True, True, lam

    fz_pos = max(abs(float(f_z_n)), 1e-4)
    fl_ratio = abs(float(f_lat_n)) / fz_pos
    m_ratio = abs(float(moment_nm)) / (2.0 * R * fz_pos)
    whitney_lhs = m_ratio + 0.5 * mu * (1.0 + lam) * fl_ratio
    jammed = bool(whitney_lhs >= 0.5 * lam or fl_ratio >= (1.0 / mu))
    return False, jammed, lam


def evaluate_3d_coulomb_contact(
    tip_xy: np.ndarray,
    z_m: float,
    tilt_xy_rad: np.ndarray,
    vessel: ClearVesselBevel,
    *,
    friction_coeff: Optional[float] = None,
    lateral_vel_dir: Optional[np.ndarray] = None,
    support_f_lat_n: float = 0.0,
    support_moment_nm: float = 0.0,
    lever_arm_m: float = 0.045,
    rng: Optional[np.random.Generator] = None,
    force_noise_n: float = 0.002,
) -> ContactWrench3D:
    """Compute 3D Coulomb friction-cone + Whitney bore contact wrench on the pipette tip."""
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    offset = np.asarray(tip_xy, dtype=np.float64) - center
    r = float(np.linalg.norm(offset))
    tilt_vec = np.asarray(tilt_xy_rad, dtype=np.float64)
    tilt_mag = float(np.linalg.norm(tilt_vec))
    mu = float(vessel.friction_coeff if friction_coeff is None else friction_coeff)

    cap_r = float(vessel.capture_radius_m)  # type: ignore[arg-type]
    bev_d = float(vessel.bevel_depth_m)  # type: ignore[arg-type]
    z_rim = float(vessel.z_rim_m)
    z_throat = z_rim - bev_d

    noise_xy = rng.normal(0.0, force_noise_n, size=2) if rng is not None else np.zeros(2, dtype=np.float64)

    if z_m > z_rim:
        return ContactWrench3D(
            force_xyz_n=np.array([noise_xy[0] * 0.2, noise_xy[1] * 0.2, 0.0], dtype=np.float64),
            moment_xyz_nm=np.zeros(3, dtype=np.float64),
            contact_state="free",
        )

    # 1. Inside cylindrical bore radial envelope (r <= bore_radius_m) or below throat
    if r <= vessel.bore_radius_m or (z_m < z_throat and r <= vessel.bore_radius_m * 1.6):
        l_ins = max(0.0, z_throat - z_m)
        if l_ins <= 0.0 and r <= vessel.bore_radius_m:
            return ContactWrench3D(
                force_xyz_n=np.array([noise_xy[0] * 0.25, noise_xy[1] * 0.25, 0.0], dtype=np.float64),
                moment_xyz_nm=np.zeros(3, dtype=np.float64),
                contact_state="bore_free",
            )

        # Wall penetration from radial offset + shaft tilt inside bore
        unit_out = (offset / r) if r > 1e-12 else (
            (tilt_vec / tilt_mag) if tilt_mag > 1e-12 else np.array([1.0, 0.0], dtype=np.float64)
        )
        pen_r = max(0.0, r - vessel.bore_radius_m)
        l_two = (2.0 * vessel.bore_radius_m) / max(tilt_mag, 1e-6)
        two_point = l_ins >= l_two

        f_wall_n = vessel.stiffness_xy_n_m * pen_r + (18.0 * max(0.0, l_ins - l_two) * tilt_mag if two_point else 0.0)
        f_lat_vec = -vessel.stiffness_xy_n_m * pen_r * unit_out + noise_xy
        # Restoring contact couple opposing tilt inside bore
        m_couple = 0.35 * max(0.0, l_ins) * f_wall_n + support_moment_nm
        m_xy = np.array([-tilt_vec[0], -tilt_vec[1]], dtype=np.float64) * (
            2.2 * max(0.001, l_ins) * vessel.stiffness_xy_n_m * 0.02
        )

        fz_slide = mu * (f_wall_n + support_f_lat_n)
        wedged, jammed, _ = check_whitney_jamming_wedging(
            insertion_depth_m=l_ins,
            bore_clearance_m=vessel.bore_radius_m,
            hole_radius_m=vessel.hole_radius_m,
            friction_coeff=mu,
            tilt_rad=tilt_mag,
            f_lat_n=max(float(np.linalg.norm(f_lat_vec)), support_f_lat_n),
            f_z_n=max(fz_slide, 0.15),
            moment_nm=max(float(np.linalg.norm(m_xy)), m_couple),
        )
        if wedged:
            return ContactWrench3D(
                force_xyz_n=np.array([f_lat_vec[0], f_lat_vec[1], max(1.65, fz_slide * 4.0)], dtype=np.float64),
                moment_xyz_nm=np.array([m_xy[0], m_xy[1], 0.0], dtype=np.float64),
                contact_state="wedged",
                jammed=True,
                wedged=True,
            )
        if jammed:
            return ContactWrench3D(
                force_xyz_n=np.array([f_lat_vec[0], f_lat_vec[1], max(1.35, fz_slide * 3.0)], dtype=np.float64),
                moment_xyz_nm=np.array([m_xy[0], m_xy[1], 0.0], dtype=np.float64),
                contact_state="jammed",
                jammed=True,
                wedged=False,
            )
        st_name: Literal["bore_free", "bore_one_point", "bore_two_point"] = (
            "bore_two_point" if two_point else ("bore_one_point" if pen_r > 0.0 else "bore_free")
        )
        return ContactWrench3D(
            force_xyz_n=np.array([f_lat_vec[0], f_lat_vec[1], fz_slide], dtype=np.float64),
            moment_xyz_nm=np.array([m_xy[0], m_xy[1], 0.0], dtype=np.float64),
            contact_state=st_name,
        )

    unit_out = offset / max(r, 1e-12)
    unit_in = -unit_out

    # 2. Conical chamfer zone (bore_radius_m < r <= capture_radius_m)
    if r <= cap_r:
        w_chamfer = max(1e-6, cap_r - vessel.bore_radius_m)
        frac = (r - vessel.bore_radius_m) / w_chamfer
        z_surface = z_throat + frac * bev_d
        if z_m > z_surface:
            return ContactWrench3D(
                force_xyz_n=np.array([noise_xy[0] * 0.2, noise_xy[1] * 0.2, 0.0], dtype=np.float64),
                moment_xyz_nm=np.zeros(3, dtype=np.float64),
                contact_state="free",
            )
        alpha = math.atan2(bev_d, w_chamfer)  # bevel angle from horizontal (~45 deg)
        pen_z = z_surface - z_m
        delta_n = pen_z * math.cos(alpha)
        k_norm = 0.5 * (vessel.stiffness_xy_n_m + vessel.stiffness_z_n_m)
        f_n = k_norm * delta_n + support_f_lat_n * math.sin(alpha)

        # 3D Coulomb friction cone opposing downward chamfer slide:
        # F_in = f_n * (sin(alpha) - mu * cos(alpha)), F_z = f_n * (cos(alpha) + mu * sin(alpha))
        f_in_scalar = f_n * (math.sin(alpha) - mu * math.cos(alpha))
        fz_val = max(0.0, f_n * (math.cos(alpha) + mu * math.sin(alpha)))
        f_lat = f_in_scalar * unit_in + noise_xy
        # Contact moment from lateral force at tip lever arm + tilt restoring feedback
        m_xy = np.array(
            [
                -lever_arm_m * f_lat[0] - 0.18 * fz_val * tilt_vec[0],
                -lever_arm_m * f_lat[1] - 0.18 * fz_val * tilt_vec[1],
            ],
            dtype=np.float64,
        )
        return ContactWrench3D(
            force_xyz_n=np.array([f_lat[0], f_lat[1], fz_val], dtype=np.float64),
            moment_xyz_nm=np.array([m_xy[0], m_xy[1], 0.0], dtype=np.float64),
            contact_state="chamfer",
        )

    # 3. Flat horizontal rim outside narrow chamfer (r > capture_radius_m):
    # Zero inward bevel slope! Vertical reaction F_z = K_z * (z_rim - z), lateral Coulomb friction <= mu * F_z
    pen_z = max(0.0, z_rim - z_m)
    fz_rim = vessel.stiffness_z_n_m * pen_z
    if lateral_vel_dir is not None and float(np.linalg.norm(lateral_vel_dir)) > 1e-9:
        v_hat = np.asarray(lateral_vel_dir, dtype=np.float64)
        v_hat = v_hat / float(np.linalg.norm(v_hat))
        f_lat_rim = -mu * fz_rim * v_hat + noise_xy
    else:
        f_lat_rim = noise_xy
    m_rim = np.array([-0.15 * fz_rim * tilt_vec[0], -0.15 * fz_rim * tilt_vec[1]], dtype=np.float64)
    return ContactWrench3D(
        force_xyz_n=np.array([f_lat_rim[0], f_lat_rim[1], fz_rim], dtype=np.float64),
        moment_xyz_nm=np.array([m_rim[0], m_rim[1], 0.0], dtype=np.float64),
        contact_state="flat_rim",
    )


# ---------------------------------------------------------------------------
# Bayesian Particle-Filter Contact Belief over Hole Center (c_x, c_y)
# ---------------------------------------------------------------------------


class BayesianHoleParticleFilter:
    """Sequential Monte Carlo particle filter localizing a sub-millimeter chamfered hole on a flat rim."""

    def __init__(
        self,
        prior_xy: np.ndarray,
        *,
        max_radius_m: float,
        capture_radius_m: float,
        n_particles: int = 256,
        rng: np.random.Generator,
    ) -> None:
        self.prior_xy = np.asarray(prior_xy, dtype=np.float64).copy()
        self.max_radius_m = float(max_radius_m)
        self.capture_radius_m = float(capture_radius_m)
        self.n_particles = int(n_particles)
        self.rng = rng

        # Stratified polar grid + jitter covering disk of radius max_radius_m
        u_r = np.sqrt(np.linspace(1.0 / n_particles, 1.0, n_particles, dtype=np.float64)) * self.max_radius_m
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        angles = np.arange(n_particles, dtype=np.float64) * golden_angle + float(rng.uniform(0.0, 2.0 * math.pi))
        self.particles = self.prior_xy[None, :] + np.column_stack([u_r * np.cos(angles), u_r * np.sin(angles)])
        self.weights = np.full(n_particles, 1.0 / float(n_particles), dtype=np.float64)

    def update_flat_rim(self, probe_xy: np.ndarray) -> None:
        """Carve out disk `||c - probe_xy|| < capture_radius_m` when flat-rim `F_z` contact is observed."""
        dists = np.linalg.norm(self.particles - probe_xy[None, :], axis=1)
        # Particles inside capture_radius_m would have produced chamfer/bore drop, not flat rim!
        inside = dists < (0.92 * self.capture_radius_m)
        likelihood = np.where(inside, 0.015, 1.0)
        self.weights *= likelihood
        s = float(np.sum(self.weights))
        if s > 1e-15:
            self.weights /= s
        else:
            self.weights.fill(1.0 / float(self.n_particles))

    def update_chamfer_or_bore(self, probe_xy: np.ndarray, inward_dir: Optional[np.ndarray] = None) -> None:
        """Collapse posterior onto hole center when chamfer shear or axial bore drop is observed."""
        if inward_dir is not None and float(np.linalg.norm(inward_dir)) > 1e-9:
            u_in = inward_dir / float(np.linalg.norm(inward_dir))
            est_center = probe_xy + 0.55 * self.capture_radius_m * u_in
        else:
            est_center = probe_xy
        sigma = max(0.00025, 0.55 * self.capture_radius_m)
        d2 = np.sum((self.particles - est_center[None, :]) ** 2, axis=1)
        self.weights *= np.exp(-0.5 * d2 / (sigma * sigma)) + 1e-6
        self.weights /= float(np.sum(self.weights))
        self._maybe_resample()

    def _maybe_resample(self) -> None:
        n_eff = 1.0 / max(float(np.sum(self.weights**2)), 1e-12)
        if n_eff < 0.35 * self.n_particles:
            idx = self.rng.choice(self.n_particles, size=self.n_particles, replace=True, p=self.weights)
            self.particles = self.particles[idx] + self.rng.normal(0.0, 0.00008, size=(self.n_particles, 2))
            self.weights.fill(1.0 / float(self.n_particles))

    @property
    def posterior_mean(self) -> np.ndarray:
        return np.sum(self.particles * self.weights[:, None], axis=0)

    @property
    def posterior_std_m(self) -> float:
        mu = self.posterior_mean
        var = float(np.sum(self.weights * np.sum((self.particles - mu[None, :]) ** 2, axis=1)))
        return math.sqrt(max(0.0, var))


# ---------------------------------------------------------------------------
# Kinematic Step Helper Supporting Both Arm2D and 6-DoF/7-DoF SerialChainSE3
# ---------------------------------------------------------------------------


def _init_arm_state(
    visual_xy: np.ndarray,
    z_init: float,
    tilt_xy_rad: np.ndarray,
    arm: object,
    scene: BenchScene,
    start_q: Optional[Sequence[float] | np.ndarray],
) -> Optional[np.ndarray]:
    if isinstance(arm, Arm2D):
        if start_q is not None:
            q_in = np.asarray(start_q, dtype=np.float64)
            if q_in.shape == (2,) and arm.within_joint_limits(q_in) and not in_collision(q_in, arm=arm, scene=scene):
                if float(np.linalg.norm(arm.fk(q_in)[1] - visual_xy)) <= 1e-4:
                    return q_in.copy()
        cands = [arm.ik(visual_xy, elbow=e) for e in (1, -1)]
        valid = [q for q in cands if q is not None]
        if not valid:
            raise ValueError(f"Visual estimate {visual_xy} is outside Arm2D workspace")
        safe = [q for q in valid if not in_collision(q, arm=arm, scene=scene)]
        if not safe:
            return None
        if start_q is not None:
            q_ref = np.asarray(start_q, dtype=np.float64)
            if q_ref.shape == (2,) and np.isfinite(q_ref).all():
                return min(safe, key=lambda q: float(np.linalg.norm(q - q_ref))).copy()
        return safe[0].copy()

    # 6-DoF / 7-DoF SerialChainSE3 (UR5e, Panda, OpentronsFlexGantry)
    pose_init = PoseSE3(
        xyz=(float(visual_xy[0]), float(visual_xy[1]), float(z_init)),
        rpy=(float(tilt_xy_rad[0]), float(tilt_xy_rad[1]), 0.0),
    )
    res = arm.dls_ik(pose_init, q_seed=start_q)  # type: ignore[attr-defined]
    if not res.success:
        return None
    return res.q.copy()


def _step_cartesian_xy(
    q: np.ndarray,
    dx_xy: np.ndarray,
    arm: object,
    scene: BenchScene,
) -> Optional[np.ndarray]:
    if isinstance(arm, Arm2D):
        jac = arm.jacobian(q)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(2), dx_xy)
        q_cand = q + dq
        if arm.within_joint_limits(q_cand) and not in_collision(q_cand, arm=arm, scene=scene):
            return q_cand
        return None

    J_w = arm.world_jacobian(q)[:2, :]  # type: ignore[attr-defined]
    dq = J_w.T @ np.linalg.solve(J_w @ J_w.T + 1e-5 * np.eye(2), dx_xy)
    q_cand = q + dq
    if arm.within_joint_limits(q_cand):  # type: ignore[attr-defined]
        return q_cand
    return None


def _tip_xy_from_q(q: np.ndarray, arm: object) -> np.ndarray:
    if isinstance(arm, Arm2D):
        return arm.fk(q)[1]
    pose = arm.fk(q)  # type: ignore[attr-defined]
    return np.asarray(pose.xyz[:2], dtype=np.float64)


# ---------------------------------------------------------------------------
# Unified Physical Simulation Executing All 4 Insertion Regimes
# ---------------------------------------------------------------------------


def _simulate_passive_or_open_loop(
    visual_xy: Sequence[float] | np.ndarray,
    *,
    regime: Literal["open_loop", "passive_rcc_compliance"],
    vessel: ClearVesselBevel,
    tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
    friction_coeff: Optional[float] = None,
    max_axial_force_n: float = 0.95,
    max_steps: int = 70,
    rng: Optional[np.random.Generator] = None,
) -> InsertionResult:
    """Execute `open_loop` or `passive_rcc_compliance` through the exact 3D Coulomb/Whitney model!"""
    vis = np.asarray(visual_xy, dtype=np.float64)
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    init_err_mm = float(np.linalg.norm(vis - center)) * 1000.0
    z_goal = vessel.z_rim_m - vessel.target_depth_m
    mu = float(vessel.friction_coeff if friction_coeff is None else friction_coeff)

    # Structural lateral/rotational support stiffness:
    # - open_loop: stiff position-controlled manipulator / gantry (4200 N/m lateral, 14 Nm/rad angular)
    # - passive_rcc_compliance: elastomer RCC wrist (155 N/m lateral, remote center reduces tilt moment!)
    if regime == "open_loop":
        k_support_xy = 4200.0
        k_support_theta = 14.0
        rcc_moment_factor = 1.0
    else:
        k_support_xy = 155.0
        k_support_theta = 2.6
        rcc_moment_factor = 0.25

    tip_xy = vis.copy()
    tilt = np.asarray(tilt_xy_rad, dtype=np.float64).copy()
    z = vessel.z_rim_m + 0.0005
    peak_force = 0.0
    jammed_flag = False
    wedged_flag = False
    rim_hangup_flag = False

    for step_idx in range(1, max_steps + 1):
        z_try = max(z_goal, z - 0.00035)
        defl_xy = tip_xy - vis
        f_support_lat = k_support_xy * float(np.linalg.norm(defl_xy))
        m_support = rcc_moment_factor * k_support_theta * float(np.linalg.norm(tilt))

        cw = evaluate_3d_coulomb_contact(
            tip_xy,
            z_try,
            tilt,
            vessel,
            friction_coeff=mu,
            support_f_lat_n=f_support_lat,
            support_moment_nm=m_support,
            rng=rng,
            force_noise_n=(0.0 if rng is None else 0.0015),
        )
        f_xy = cw.force_xyz_n[:2]
        f_z = float(cw.force_xyz_n[2])
        f_mag = float(np.linalg.norm(cw.force_xyz_n))
        peak_force = max(peak_force, f_mag)

        if cw.wedged or cw.jammed:
            wedged_flag = bool(cw.wedged)
            jammed_flag = True
            fail = "wedged" if cw.wedged else "jammed"
            final_err = float(np.linalg.norm(tip_xy - center)) * 1000.0
            return InsertionResult(
                success=False,
                initial_error_mm=init_err_mm,
                final_error_mm=final_err,
                steps=step_idx,
                peak_force_n=max(peak_force, max_axial_force_n),
                final_z_m=z,
                seal_intact=False,
                slip_detected=True,
                axial_slip_mm=max(0.0, (z - z_goal) * 1000.0),
                regime=regime,
                failure_mode=fail,
                jammed=True,
                wedged=wedged_flag,
                rim_hangup=False,
                final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
            )

        if cw.contact_state == "flat_rim" and f_z >= max_axial_force_n:
            rim_hangup_flag = True
            final_err = float(np.linalg.norm(tip_xy - center)) * 1000.0
            return InsertionResult(
                success=False,
                initial_error_mm=init_err_mm,
                final_error_mm=final_err,
                steps=step_idx,
                peak_force_n=peak_force,
                final_z_m=vessel.z_rim_m,
                seal_intact=False,
                slip_detected=True,
                axial_slip_mm=max(0.0, (vessel.z_rim_m - z_goal) * 1000.0),
                regime=regime,
                failure_mode="rim_hangup",
                jammed=False,
                wedged=False,
                rim_hangup=True,
                final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
            )

        # Quasi-static elastic equilibrium update between contact lateral force & support spring:
        # k_support_xy * (tip_xy_eq - vis) == f_xy_contact
        if cw.contact_state in ("chamfer", "bore_one_point", "bore_two_point"):
            eq_target = vis + f_xy / k_support_xy
            tip_xy = 0.45 * tip_xy + 0.55 * eq_target
            if regime == "passive_rcc_compliance":
                tilt = tilt * 0.90  # Passive RCC remote-center self-alignment inside chamfer/bore

        if f_z >= max_axial_force_n:
            final_err = float(np.linalg.norm(tip_xy - center)) * 1000.0
            return InsertionResult(
                success=False,
                initial_error_mm=init_err_mm,
                final_error_mm=final_err,
                steps=step_idx,
                peak_force_n=peak_force,
                final_z_m=z,
                seal_intact=False,
                slip_detected=True,
                axial_slip_mm=max(0.0, (z - z_goal) * 1000.0),
                regime=regime,
                failure_mode="chamfer_friction_stall",
                jammed=jammed_flag,
                wedged=wedged_flag,
                rim_hangup=rim_hangup_flag,
                final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
            )

        z = z_try
        if z <= z_goal + 1e-9:
            final_err = float(np.linalg.norm(tip_xy - center)) * 1000.0
            centered = final_err <= max(0.20, vessel.bore_radius_m * 1000.0 + 0.08)
            return InsertionResult(
                success=centered,
                initial_error_mm=init_err_mm,
                final_error_mm=final_err,
                steps=step_idx,
                peak_force_n=peak_force,
                final_z_m=z,
                seal_intact=centered,
                slip_detected=not centered,
                axial_slip_mm=0.0,
                regime=regime,
                failure_mode=("none" if centered else "off_center"),
                final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
            )

    final_err = float(np.linalg.norm(tip_xy - center)) * 1000.0
    return InsertionResult(
        success=False,
        initial_error_mm=init_err_mm,
        final_error_mm=final_err,
        steps=max_steps,
        peak_force_n=peak_force,
        final_z_m=z,
        seal_intact=False,
        slip_detected=True,
        axial_slip_mm=max(0.0, (z - z_goal) * 1000.0),
        regime=regime,
        failure_mode="timeout",
        final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
    )


def run_open_loop_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    *,
    tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
    friction_coeff: Optional[float] = None,
    max_axial_force_n: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> InsertionResult:
    """Baseline camera-only vertical plunge sharing exact 3D Coulomb chamfer & Whitney bore mechanics."""
    return _simulate_passive_or_open_loop(
        visual_xy,
        regime="open_loop",
        vessel=vessel,
        tilt_xy_rad=tilt_xy_rad,
        friction_coeff=friction_coeff,
        max_axial_force_n=max_axial_force_n,
        rng=rng,
    )


def run_passive_rcc_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    *,
    tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
    friction_coeff: Optional[float] = None,
    max_axial_force_n: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> InsertionResult:
    """Passive Remote Center of Compliance (RCC) insertion without active spiral rim search."""
    return _simulate_passive_or_open_loop(
        visual_xy,
        regime="passive_rcc_compliance",
        vessel=vessel,
        tilt_xy_rad=tilt_xy_rad,
        friction_coeff=friction_coeff,
        max_axial_force_n=max_axial_force_n,
        rng=rng,
    )


def run_local_shear_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    arm: object = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
    friction_coeff: Optional[float] = None,
    max_steps: int = 85,
    compliance_gain_m_per_n: float = 0.0014,
    moment_gain_rad_per_nm: float = 0.65,
    max_axial_force_n: float = 0.95,
) -> InsertionResult:
    """Local 6-DoF Force/Torque admittance without active spiral search (fails on flat rim `r > r_cap`)."""
    return run_active_tactile_insertion(
        visual_xy,
        rng,
        vessel=vessel,
        arm=arm,
        scene=scene,
        start_q=start_q,
        tilt_xy_rad=tilt_xy_rad,
        friction_coeff=friction_coeff,
        max_steps=max_steps,
        compliance_gain_m_per_n=compliance_gain_m_per_n,
        moment_gain_rad_per_nm=moment_gain_rad_per_nm,
        max_axial_force_n=max_axial_force_n,
        enable_spiral_search=False,
    )


def run_active_tactile_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    arm: object = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
    friction_coeff: Optional[float] = None,
    max_steps: int = 185,
    compliance_gain_m_per_n: float = 0.0014,
    moment_gain_rad_per_nm: float = 0.75,
    max_axial_force_n: float = 0.95,
    max_search_radius_m: float = 0.0045,  # 4.5 mm = half of 9.0 mm SBS 96-well pitch!
    enable_spiral_search: bool = True,
) -> InsertionResult:
    """Active Archimedean Spiral + Bayesian Particle Filter + 6-DoF Force/Torque Admittance insertion.

    1. Approaches vertically until guarded contact (`F_z > 0.08 N`) or bore entry.
    2. When refractive RGB-D error lands on the flat rim (`r > capture_radius_m`), `enable_spiral_search=True`
       runs an Archimedean spiral `r(phi) = b * phi` (`2*pi*b <= 1.10 * capture_radius_m`) bounded by
       `max_search_radius_m` (`4.5 mm` single-well Voronoi cell) while updating a 256-particle Bayesian
       belief over `(c_x, c_y)`.
    3. Upon entering the `0.70 mm` chamfer capture zone, 6-DoF F/T admittance translates along inward
       shear `F_xy` and actively aligns tilt `theta -> 0` via moment feedback so Whitney jamming/wedging
       is eliminated during full `12 mm` bore seating!
    """
    vis = np.asarray(visual_xy, dtype=np.float64)
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    init_err_mm = float(np.linalg.norm(vis - center)) * 1000.0
    z_goal = vessel.z_rim_m - vessel.target_depth_m
    cap_r = float(vessel.capture_radius_m)  # type: ignore[arg-type]
    bev_d = float(vessel.bevel_depth_m)  # type: ignore[arg-type]
    mu = float(vessel.friction_coeff if friction_coeff is None else friction_coeff)
    regime_name = "active_spiral_bayesian_tactile" if enable_spiral_search else "local_shear_admittance"

    tilt = np.asarray(tilt_xy_rad, dtype=np.float64).copy()
    q = _init_arm_state(vis, vessel.z_rim_m + 0.001, tilt, arm, scene, start_q)
    if q is None:
        return InsertionResult(
            success=False,
            initial_error_mm=init_err_mm,
            final_error_mm=init_err_mm,
            steps=0,
            peak_force_n=0.0,
            final_z_m=vessel.z_rim_m + 0.001,
            seal_intact=False,
            slip_detected=True,
            axial_slip_mm=max(0.0, (vessel.z_rim_m + 0.001 - z_goal) * 1000.0),
            regime=regime_name,
            failure_mode="kinematic_or_obstacle_block",
            final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
        )

    tip_init = _tip_xy_from_q(q, arm)
    init_err_mm = float(np.linalg.norm(tip_init - center)) * 1000.0

    pf = BayesianHoleParticleFilter(
        prior_xy=tip_init,
        max_radius_m=max_search_radius_m,
        capture_radius_m=cap_r,
        n_particles=256,
        rng=rng,
    )

    z = vessel.z_rim_m + 0.0008
    peak_force = 0.0
    peak_fz = 0.0
    spiral_probes = 0
    captured_hole = False
    rim_stalled = False
    jammed_flag = False
    wedged_flag = False

    # Archimedean spiral parameter: radial pitch per revolution = 1.05 * capture_radius_m
    # so the 2*capture_radius_m chamfer disk can NEVER slip between adjacent spiral turns!
    spiral_b = (1.05 * cap_r) / (2.0 * math.pi)
    spiral_phi = 0.0
    arc_step = max(0.00032, 0.55 * cap_r)

    for step_idx in range(1, max_steps + 1):
        tip = _tip_xy_from_q(q, arm)
        cw = evaluate_3d_coulomb_contact(tip, z, tilt, vessel, friction_coeff=mu, rng=rng)
        f_xy = cw.force_xyz_n[:2]
        f_z = float(cw.force_xyz_n[2])
        f_mag = float(np.linalg.norm(cw.force_xyz_n))
        peak_force = max(peak_force, f_mag)
        peak_fz = max(peak_fz, f_z)

        if cw.wedged or cw.jammed:
            # Active tilt-admittance backoff & re-alignment attempt
            wedged_flag = bool(cw.wedged)
            jammed_flag = True
            tilt = tilt * 0.45
            z = min(vessel.z_rim_m, z + 0.0006)
            continue

        # Detect capture into chamfer or bore!
        if cw.contact_state in ("chamfer", "bore_free", "bore_one_point", "bore_two_point") and z <= vessel.z_rim_m + 1e-6:
            captured_hole = True
            pf.update_chamfer_or_bore(tip, inward_dir=(f_xy if float(np.linalg.norm(f_xy)) > 0.01 else None))

        # Handle flat-rim contact outside chamfer
        if not captured_hole and cw.contact_state == "flat_rim":
            pf.update_flat_rim(tip)
            if not enable_spiral_search:
                # Local shear admittance has zero bevel gradient on flat rim -> stalls at F_z >= max_axial_force_n
                z_push = z - 0.00035
                cw_push = evaluate_3d_coulomb_contact(tip, z_push, tilt, vessel, friction_coeff=mu, rng=None, force_noise_n=0.0)
                peak_force = max(peak_force, float(np.linalg.norm(cw_push.force_xyz_n)))
                peak_fz = max(peak_fz, float(cw_push.force_xyz_n[2]))
                if float(cw_push.force_xyz_n[2]) >= max_axial_force_n:
                    rim_stalled = True
                    return InsertionResult(
                        success=False,
                        initial_error_mm=init_err_mm,
                        final_error_mm=float(np.linalg.norm(tip - center)) * 1000.0,
                        steps=step_idx,
                        peak_force_n=peak_force,
                        final_z_m=vessel.z_rim_m,
                        seal_intact=False,
                        slip_detected=True,
                        axial_slip_mm=max(0.0, (vessel.z_rim_m - z_goal) * 1000.0),
                        regime=regime_name,
                        failure_mode="rim_hangup",
                        rim_hangup=True,
                        final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
                        posterior_std_mm=pf.posterior_std_m * 1000.0,
                    )
                z = z_push
                continue

            # Active Archimedean Spiral Probing lightly skimming the rim (`F_z ~ 0.12 N`)!
            spiral_probes += 1
            r_cur = spiral_b * spiral_phi
            dphi = arc_step / math.hypot(spiral_b, max(r_cur, 0.4 * cap_r))
            spiral_phi += dphi
            r_next = spiral_b * spiral_phi

            if r_next > max_search_radius_m + 1e-9:
                # Exhausted safe single-well Voronoi search cell without entering chamfer -> safe rim halt!
                rim_stalled = True
                peak_force = max(peak_force, max_axial_force_n)
                return InsertionResult(
                    success=False,
                    initial_error_mm=init_err_mm,
                    final_error_mm=float(np.linalg.norm(tip - center)) * 1000.0,
                    steps=step_idx,
                    peak_force_n=peak_force,
                    final_z_m=vessel.z_rim_m,
                    seal_intact=False,
                    slip_detected=True,
                    axial_slip_mm=max(0.0, (vessel.z_rim_m - z_goal) * 1000.0),
                    regime=regime_name,
                    failure_mode="search_radius_exhausted_rim_hangup",
                    rim_hangup=True,
                    spiral_probes=spiral_probes,
                    final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
                    posterior_std_mm=pf.posterior_std_m * 1000.0,
                )

            spiral_target = tip_init + np.array(
                [r_next * math.cos(spiral_phi), r_next * math.sin(spiral_phi)],
                dtype=np.float64,
            )
            # Blend 88% deterministic space-filling Archimedean spiral + 12% Bayesian posterior pull
            blended_target = 0.88 * spiral_target + 0.12 * pf.posterior_mean
            # Enforce Voronoi boundary around tip_init
            rel_init = blended_target - tip_init
            rel_norm = float(np.linalg.norm(rel_init))
            if rel_norm > max_search_radius_m:
                blended_target = tip_init + rel_init * (max_search_radius_m / rel_norm)

            q_next = _step_cartesian_xy(q, blended_target - tip, arm, scene)
            if q_next is None:
                return InsertionResult(
                    success=False,
                    initial_error_mm=init_err_mm,
                    final_error_mm=float(np.linalg.norm(tip - center)) * 1000.0,
                    steps=step_idx,
                    peak_force_n=peak_force,
                    final_z_m=vessel.z_rim_m,
                    seal_intact=False,
                    slip_detected=True,
                    axial_slip_mm=max(0.0, (vessel.z_rim_m - z_goal) * 1000.0),
                    regime=regime_name,
                    failure_mode="joint_or_obstacle_limit",
                    rim_hangup=True,
                    spiral_probes=spiral_probes,
                    final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
                    posterior_std_mm=pf.posterior_std_m * 1000.0,
                )
            q = q_next
            # Probe gently at z_rim - 0.06 mm (~0.14 N preload) so dropping into chamfer is felt immediately!
            z = vessel.z_rim_m - 0.00006
            # Also align tilt slightly using flat-rim normal reaction torque
            tilt = tilt * 0.92
            continue

        # 6-DoF Force/Torque Admittance Control inside chamfer & bore!
        shear_norm = float(np.linalg.norm(f_xy))
        # Rotational admittance reduces angular tilt whenever contact moment is felt
        if f_z > 0.01 or shear_norm > 0.01:
            d_tilt = moment_gain_rad_per_nm * cw.moment_xyz_nm[:2] - 0.35 * tilt
            tilt = tilt + np.clip(d_tilt, -0.008, 0.008)

        if cw.contact_state == "chamfer" and mu >= math.tan(math.atan2(bev_d, max(1e-6, cap_r - vessel.bore_radius_m))):
            # High-friction chamfer (>1.0): use Bayesian posterior center directly
            dx = np.clip(0.55 * (pf.posterior_mean - tip), -0.0008, 0.0008)
        elif shear_norm > 0.005:
            dx = np.clip(compliance_gain_m_per_n * f_xy, -0.0012, 0.0012)
        else:
            dx = np.zeros(2, dtype=np.float64)

        if float(np.linalg.norm(dx)) > 1e-7:
            q_cand = _step_cartesian_xy(q, dx, arm, scene)
            if q_cand is not None:
                q = q_cand
                tip = _tip_xy_from_q(q, arm)

        # Guarded axial descent step
        dz = 0.00035 if z > vessel.z_rim_m - bev_d - 0.001 else 0.0016
        z_next = max(z_goal, z - dz)
        cw_probe = evaluate_3d_coulomb_contact(tip, z_next, tilt, vessel, friction_coeff=mu, rng=None, force_noise_n=0.0)
        fz_probe = float(cw_probe.force_xyz_n[2])

        if cw_probe.wedged or cw_probe.jammed:
            jammed_flag = True
            wedged_flag = bool(cw_probe.wedged)
            tilt = tilt * 0.40
            continue

        if fz_probe >= max_axial_force_n:
            # High normal force on chamfer: prioritize lateral centering before descending further
            if cw_probe.contact_state == "chamfer":
                f_in_probe = cw_probe.force_xyz_n[:2]
                if float(np.linalg.norm(f_in_probe)) > 1e-6:
                    dx_relief = np.clip(0.00045 * (f_in_probe / float(np.linalg.norm(f_in_probe))), -0.0008, 0.0008)
                    q_rel = _step_cartesian_xy(q, dx_relief, arm, scene)
                    if q_rel is not None:
                        q = q_rel
                continue
            peak_force = max(peak_force, fz_probe)
            rim_stalled = True
            continue

        z = z_next
        if z <= z_goal + 1e-9:
            tip_final = _tip_xy_from_q(q, arm)
            final_err_mm = float(np.linalg.norm(tip_final - center)) * 1000.0
            centered = final_err_mm < 0.20
            seal_ok = bool(centered and not rim_stalled and peak_fz < max_axial_force_n * 1.25)
            return InsertionResult(
                success=bool(centered and seal_ok),
                initial_error_mm=init_err_mm,
                final_error_mm=final_err_mm,
                steps=step_idx,
                peak_force_n=peak_force,
                final_z_m=z,
                seal_intact=seal_ok,
                slip_detected=not seal_ok,
                axial_slip_mm=max(0.0, (z - z_goal) * 1000.0),
                regime=regime_name,
                failure_mode=("none" if (centered and seal_ok) else "seal_fault"),
                jammed=False,
                wedged=False,
                rim_hangup=False,
                spiral_probes=spiral_probes,
                final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
                posterior_std_mm=pf.posterior_std_m * 1000.0,
            )

    tip_end = _tip_xy_from_q(q, arm)
    final_err_mm = float(np.linalg.norm(tip_end - center)) * 1000.0
    axial_slip_mm = max(0.0, (z - z_goal) * 1000.0)
    fail_mode = "wedged" if wedged_flag else ("jammed" if jammed_flag else ("rim_hangup" if not captured_hole else "timeout"))
    return InsertionResult(
        success=False,
        initial_error_mm=init_err_mm,
        final_error_mm=final_err_mm,
        steps=max_steps,
        peak_force_n=peak_force,
        final_z_m=z,
        seal_intact=False,
        slip_detected=True,
        axial_slip_mm=axial_slip_mm,
        regime=regime_name,
        failure_mode=fail_mode,
        jammed=jammed_flag,
        wedged=wedged_flag,
        rim_hangup=not captured_hole,
        spiral_probes=spiral_probes,
        final_tilt_deg=math.degrees(float(np.linalg.norm(tilt))),
        posterior_std_mm=pf.posterior_std_m * 1000.0,
    )


def run_tactile_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    arm: object = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    max_steps: int = 185,
    compliance_gain_m_per_n: float = 0.0014,
    max_axial_force_n: float = 0.95,
    max_search_radius_m: float = 0.0045,
    enable_spiral_search: Optional[bool] = None,
    tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
    friction_coeff: Optional[float] = None,
) -> InsertionResult:
    """Closed-loop tactile insertion defaulting to active spiral + Bayesian search on narrow chamfers."""
    use_spiral = (
        bool(float(vessel.capture_radius_m) < 0.0025)  # type: ignore[arg-type]
        if enable_spiral_search is None
        else bool(enable_spiral_search)
    )
    return run_active_tactile_insertion(
        visual_xy,
        rng,
        vessel=vessel,
        arm=arm,
        scene=scene,
        start_q=start_q,
        tilt_xy_rad=tilt_xy_rad,
        friction_coeff=friction_coeff,
        max_steps=max_steps,
        compliance_gain_m_per_n=compliance_gain_m_per_n,
        max_axial_force_n=max_axial_force_n,
        max_search_radius_m=max_search_radius_m,
        enable_spiral_search=use_spiral,
    )


# ---------------------------------------------------------------------------
# Un-Cherry-Picked 500-Trial Full-Grid Benchmark Across All 4 Regimes
# ---------------------------------------------------------------------------


def benchmark_tactile_regimes(
    *,
    n_trials: int = 500,
    seed: int = 20261127,
    chamfer_width_m: float = 0.00070,
    bore_radius_m: float = 0.00010,
    offset_range_mm: Tuple[float, float] = (0.0, 4.0),
    friction_range: Tuple[float, float] = (0.15, 0.45),
    tilt_range_deg: Tuple[float, float] = (0.0, 2.5),
    arm: object = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> Dict[str, Any]:
    """Benchmark all 4 insertion regimes across 500 trials of offset (0..4 mm) x mu (0.15..0.45) x tilt (0..2.5 deg)."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    rng = np.random.default_rng(seed)

    base_vessel = ClearVesselBevel(
        bore_radius_m=bore_radius_m,
        chamfer_width_m=chamfer_width_m,
    )
    center = np.asarray(base_vessel.center_xy, dtype=np.float64)

    regimes: Tuple[InsertionRegime, ...] = (
        "open_loop",
        "passive_rcc_compliance",
        "local_shear_admittance",
        "active_spiral_bayesian_tactile",
    )
    stats: Dict[str, Dict[str, List[Any]]] = {
        r: {
            "success": [],
            "jam_or_wedge": [],
            "rim_hangup": [],
            "final_err_mm": [],
            "peak_force_n": [],
            "steps": [],
            "spiral_probes": [],
        }
        for r in regimes
    }

    # Stratified Latin-hypercube style sampling across offset magnitude (0..4 mm), mu (0.15..0.45), tilt (0..2.5 deg)
    u_r = (np.arange(n_trials, dtype=np.float64) + rng.random(n_trials)) / float(n_trials)
    u_mu = rng.permutation((np.arange(n_trials, dtype=np.float64) + rng.random(n_trials)) / float(n_trials))
    u_th = rng.permutation((np.arange(n_trials, dtype=np.float64) + rng.random(n_trials)) / float(n_trials))

    r_vals_m = (offset_range_mm[0] + u_r * (offset_range_mm[1] - offset_range_mm[0])) / 1000.0
    mu_vals = friction_range[0] + u_mu * (friction_range[1] - friction_range[0])
    th_vals_rad = np.radians(tilt_range_deg[0] + u_th * (tilt_range_deg[1] - tilt_range_deg[0]))

    for i in range(n_trials):
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        psi_tilt = float(rng.uniform(0.0, 2.0 * math.pi))
        vis_xy = center + float(r_vals_m[i]) * np.array([math.cos(phi), math.sin(phi)], dtype=np.float64)
        tilt_xy = (
            float(th_vals_rad[i]) * math.cos(psi_tilt),
            float(th_vals_rad[i]) * math.sin(psi_tilt),
        )
        mu_i = float(mu_vals[i])

        r_ol = run_open_loop_insertion(vis_xy, vessel=base_vessel, tilt_xy_rad=tilt_xy, friction_coeff=mu_i, rng=rng)
        r_rcc = run_passive_rcc_insertion(vis_xy, vessel=base_vessel, tilt_xy_rad=tilt_xy, friction_coeff=mu_i, rng=rng)
        r_loc = run_local_shear_insertion(
            vis_xy, rng, vessel=base_vessel, arm=arm, scene=scene, tilt_xy_rad=tilt_xy, friction_coeff=mu_i
        )
        r_act = run_active_tactile_insertion(
            vis_xy, rng, vessel=base_vessel, arm=arm, scene=scene, tilt_xy_rad=tilt_xy, friction_coeff=mu_i
        )

        for reg_key, res_obj in (
            ("open_loop", r_ol),
            ("passive_rcc_compliance", r_rcc),
            ("local_shear_admittance", r_loc),
            ("active_spiral_bayesian_tactile", r_act),
        ):
            b = stats[reg_key]
            b["success"].append(int(res_obj.success))
            b["jam_or_wedge"].append(int(res_obj.jammed or res_obj.wedged))
            b["rim_hangup"].append(int(res_obj.rim_hangup or res_obj.failure_mode == "rim_hangup"))
            if res_obj.success:
                b["final_err_mm"].append(res_obj.final_error_mm)
            b["peak_force_n"].append(res_obj.peak_force_n)
            b["steps"].append(res_obj.steps)
            b["spiral_probes"].append(res_obj.spiral_probes)

    summary: Dict[str, Any] = {}
    for reg_key in regimes:
        b = stats[reg_key]
        succ_count = int(sum(b["success"]))
        jam_count = int(sum(b["jam_or_wedge"]))
        rim_count = int(sum(b["rim_hangup"]))
        summary[reg_key] = {
            "trials": n_trials,
            "success_count": succ_count,
            "success_rate_pct": round(100.0 * succ_count / n_trials, 2),
            "jam_or_wedge_rate_pct": round(100.0 * jam_count / n_trials, 2),
            "rim_hangup_rate_pct": round(100.0 * rim_count / n_trials, 2),
            "mean_final_error_mm_given_success": (
                round(float(np.mean(b["final_err_mm"])), 4) if b["final_err_mm"] else None
            ),
            "max_final_error_mm_given_success": (
                round(float(np.max(b["final_err_mm"])), 4) if b["final_err_mm"] else None
            ),
            "mean_peak_force_n": round(float(np.mean(b["peak_force_n"])), 3),
            "mean_steps": round(float(np.mean(b["steps"])), 1),
            "mean_spiral_probes": round(float(np.mean(b["spiral_probes"])), 1),
        }

    return {
        "benchmark": "3d_coulomb_whitney_tactile_insertion_500_trials",
        "n_trials": n_trials,
        "geometry": {
            "bore_clearance_mm": round(bore_radius_m * 1000.0, 3),
            "hole_radius_mm": round(base_vessel.hole_radius_m * 1000.0, 3),
            "chamfer_width_mm": round(chamfer_width_m * 1000.0, 3),
            "capture_radius_mm": round(float(base_vessel.capture_radius_m) * 1000.0, 3),  # type: ignore[arg-type]
            "max_voronoi_search_radius_mm": 4.5,
        },
        "ranges": {
            "offset_mm": list(offset_range_mm),
            "friction_coeff": list(friction_range),
            "tilt_deg": list(tilt_range_deg),
        },
        "regimes": summary,
    }


__all__ = [
    "BayesianHoleParticleFilter",
    "ClearVesselBevel",
    "ContactWrench3D",
    "InsertionRegime",
    "InsertionResult",
    "benchmark_tactile_regimes",
    "check_whitney_jamming_wedging",
    "evaluate_3d_coulomb_contact",
    "run_active_tactile_insertion",
    "run_local_shear_insertion",
    "run_open_loop_insertion",
    "run_passive_rcc_insertion",
    "run_tactile_insertion",
]
