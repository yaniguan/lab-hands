"""Closed-loop tactile 'last-centimeter' compliant insertion for transparent vessels.

Simulates optical tactile pad shear/normal force feedback on a chamfered clear
polypropylene well or borosilicate glass tube where RGB-D depth cameras suffer
up to +-3 mm refractive pose error, surfacing axial slip and tip-seal telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .collision import DEFAULT_SCENE, BenchScene, in_collision
from .kinematics import DEFAULT_ARM, Arm2D


@dataclass(frozen=True)
class ClearVesselBevel:
    """Chamfered transparent tube/well opening geometry."""

    center_xy: Tuple[float, float] = (0.20, 0.05)
    bore_radius_m: float = 0.00005  # 0.05 mm radial clearance around tip shaft
    capture_radius_m: float = 0.0055  # 5.5 mm chamfer capture radius (> sqrt(2)*3 mm)
    z_rim_m: float = 0.020
    bevel_depth_m: float = 0.004
    target_depth_m: float = 0.012
    stiffness_xy_n_m: float = 300.0
    stiffness_z_n_m: float = 350.0


@dataclass(frozen=True)
class InsertionResult:
    success: bool
    initial_error_mm: float
    final_error_mm: float
    steps: int
    peak_force_n: float
    final_z_m: float
    seal_intact: bool = True
    slip_detected: bool = False
    axial_slip_mm: float = 0.0


def _contact_wrench(
    tip_xy: np.ndarray,
    z: float,
    vessel: ClearVesselBevel,
    rng: np.random.Generator,
    shear_noise_n: float = 0.001,
) -> Tuple[np.ndarray, float]:
    """Simulated 2D lateral shear force (fx, fy) and vertical reaction force fz on tip."""
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    offset = tip_xy - center
    r = float(np.linalg.norm(offset))
    if z > vessel.z_rim_m:
        return np.zeros(2, dtype=np.float64), 0.0

    if r <= vessel.bore_radius_m:
        return rng.normal(0.0, shear_noise_n * 0.25, size=2), 0.0

    unit_out = offset / max(r, 1e-12)
    z_throat = vessel.z_rim_m - vessel.bevel_depth_m
    penetration_r = r - vessel.bore_radius_m

    if r <= vessel.capture_radius_m:
        frac = penetration_r / (vessel.capture_radius_m - vessel.bore_radius_m)
        z_surface = z_throat + frac * vessel.bevel_depth_m
        if z > z_surface:
            return np.zeros(2, dtype=np.float64), 0.0
        penetration_z = z_surface - z
        # Restoring lateral shear vanishes smoothly as r -> bore_radius_m (no step discontinuity!)
        f_lat = -vessel.stiffness_xy_n_m * penetration_r * unit_out
        f_lat = f_lat + rng.normal(0.0, shear_noise_n, size=2)
        f_z = vessel.stiffness_z_n_m * min(penetration_z, 0.0015)
        return f_lat, float(f_z)

    # Outside chamfer capture radius: flat horizontal rim has zero inward bevel slope,
    # generating pure axial reaction force f_z (rim hangup / axial slip).
    penetration_z = max(0.0, vessel.z_rim_m - z)
    f_lat = rng.normal(0.0, shear_noise_n, size=2)
    f_z = vessel.stiffness_z_n_m * penetration_z
    return f_lat, float(f_z)


def run_open_loop_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    vessel: ClearVesselBevel = ClearVesselBevel(),
) -> InsertionResult:
    """Baseline camera-only vertical plunge at noisy visual estimate `visual_xy`."""
    vis = np.asarray(visual_xy, dtype=np.float64)
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    err_m = float(np.linalg.norm(vis - center))
    ok = err_m <= vessel.bore_radius_m
    z_goal = vessel.z_rim_m - vessel.target_depth_m
    final_z = z_goal if ok else vessel.z_rim_m
    peak_f = 0.0 if ok else vessel.stiffness_z_n_m * 0.003
    axial_slip_mm = 0.0 if ok else max(0.0, (final_z - z_goal) * 1000.0)
    return InsertionResult(
        success=ok,
        initial_error_mm=err_m * 1000.0,
        final_error_mm=err_m * 1000.0,
        steps=1,
        peak_force_n=peak_f,
        final_z_m=final_z,
        seal_intact=ok,
        slip_detected=not ok,
        axial_slip_mm=axial_slip_mm,
    )


def _select_tactile_start_q(
    visual_xy: np.ndarray,
    arm: Arm2D,
    scene: BenchScene,
    start_q: Optional[Sequence[float] | np.ndarray],
) -> Optional[np.ndarray]:
    if start_q is not None:
        q_in = np.asarray(start_q, dtype=np.float64)
        if q_in.shape == (2,) and arm.within_joint_limits(q_in) and not in_collision(q_in, arm=arm, scene=scene):
            _, tip_in = arm.fk(q_in)
            if float(np.linalg.norm(tip_in - visual_xy)) <= 1e-4:
                return q_in.copy()

    candidates = []
    for elbow in (1, -1):
        q_cand = arm.ik(visual_xy, elbow=elbow)
        if q_cand is not None:
            candidates.append(q_cand)
    if not candidates:
        raise ValueError(f"Visual estimate {visual_xy} is outside arm workspace")

    safe_candidates = [q for q in candidates if not in_collision(q, arm=arm, scene=scene)]
    if not safe_candidates:
        return None
    if start_q is not None:
        q_ref = np.asarray(start_q, dtype=np.float64)
        if q_ref.shape == (2,) and np.isfinite(q_ref).all():
            return min(safe_candidates, key=lambda q: float(np.linalg.norm(q - q_ref))).copy()
    return safe_candidates[0].copy()


def run_tactile_insertion(
    visual_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    max_steps: int = 80,
    compliance_gain_m_per_n: float = 0.0018,
    max_axial_force_n: float = 0.95,
) -> InsertionResult:
    """Closed-loop compliant tactile insertion driven by 2D contact shear feedback via Arm2D Jacobian.

    Note: 0 < compliance_gain * stiffness_xy = 0.54 < 1 guarantees monotonic non-overshooting convergence
    inside `capture_radius_m`, while enforcing joint limits, obstacle clearance, and axial rim-hangup detection.
    """
    vis = np.asarray(visual_xy, dtype=np.float64)
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    init_err_mm = float(np.linalg.norm(vis - center)) * 1000.0
    z_goal = vessel.z_rim_m - vessel.target_depth_m

    q = _select_tactile_start_q(vis, arm=arm, scene=scene, start_q=start_q)
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
        )

    _, tip_init = arm.fk(q)
    init_err_mm = float(np.linalg.norm(tip_init - center)) * 1000.0

    z = vessel.z_rim_m + 0.001
    peak_force = 0.0
    peak_fz = 0.0
    slip_detected = False
    shear_threshold_n = 0.004  # ~0.013 mm radial deflection above bore_radius_m

    for step_idx in range(1, max_steps + 1):
        _, tip = arm.fk(q)
        f_shear, f_z = _contact_wrench(tip, z, vessel, rng)
        f_mag = float(np.hypot(float(np.linalg.norm(f_shear)), f_z))
        peak_force = max(peak_force, f_mag)
        peak_fz = max(peak_fz, f_z)
        shear_norm = float(np.linalg.norm(f_shear))

        if f_z >= max_axial_force_n:
            # Flat-rim hangup or axial overload: spring compliance stalls on rim without punching through
            slip_detected = True
            z = vessel.z_rim_m
            continue

        if shear_norm > shear_threshold_n:
            dx = np.clip(compliance_gain_m_per_n * f_shear, -0.0020, 0.0020)
            jac = arm.jacobian(q)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(2), dx)
            q_cand = q + dq
            if arm.within_joint_limits(q_cand) and not in_collision(q_cand, arm=arm, scene=scene):
                q = q_cand
                if f_z < 0.15:
                    z = max(z_goal, z - 0.0004)
            else:
                slip_detected = True
        else:
            if slip_detected:
                z = vessel.z_rim_m
                continue
            dz = 0.0006 if z > vessel.z_rim_m - vessel.bevel_depth_m else 0.0020
            z_next = max(z_goal, z - dz)
            _, f_z_probe = _contact_wrench(tip, z_next, vessel, rng, shear_noise_n=0.0)
            if f_z_probe >= max_axial_force_n:
                peak_force = max(peak_force, float(f_z_probe))
                peak_fz = max(peak_fz, float(f_z_probe))
                slip_detected = True
                z = vessel.z_rim_m
                continue
            z = z_next
            if z <= z_goal + 1e-9:
                _, tip_final = arm.fk(q)
                final_err_mm = float(np.linalg.norm(tip_final - center)) * 1000.0
                centered = final_err_mm < 0.2
                seal_ok = bool(centered and not slip_detected and peak_fz < max_axial_force_n)
                return InsertionResult(
                    success=bool(centered and seal_ok),
                    initial_error_mm=init_err_mm,
                    final_error_mm=final_err_mm,
                    steps=step_idx,
                    peak_force_n=peak_force,
                    final_z_m=z,
                    seal_intact=seal_ok,
                    slip_detected=slip_detected,
                    axial_slip_mm=max(0.0, (z - z_goal) * 1000.0),
                )

    _, tip_end = arm.fk(q)
    final_err_mm = float(np.linalg.norm(tip_end - center)) * 1000.0
    axial_slip_mm = max(0.0, (z - z_goal) * 1000.0)
    return InsertionResult(
        success=False,
        initial_error_mm=init_err_mm,
        final_error_mm=final_err_mm,
        steps=max_steps,
        peak_force_n=peak_force,
        final_z_m=z,
        seal_intact=False,
        slip_detected=bool(slip_detected or axial_slip_mm > 0.1),
        axial_slip_mm=axial_slip_mm,
    )
