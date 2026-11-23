"""Closed-loop tactile 'last-centimeter' compliant insertion for transparent vessels.

Simulates optical tactile pad shear/normal force feedback on a chamfered clear
polypropylene well or borosilicate glass tube where RGB-D depth cameras suffer
up to +-3 mm refractive pose error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

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

    penetration_z = max(0.0, vessel.z_rim_m - z)
    f_lat = -vessel.stiffness_xy_n_m * penetration_r * unit_out
    f_lat = f_lat + rng.normal(0.0, shear_noise_n, size=2)
    f_z = vessel.stiffness_z_n_m * penetration_z
    return f_lat, float(f_z)


def run_open_loop_insertion(
    visual_xy: np.ndarray,
    vessel: ClearVesselBevel = ClearVesselBevel(),
) -> InsertionResult:
    """Baseline camera-only vertical plunge at noisy visual estimate `visual_xy`."""
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    err_m = float(np.linalg.norm(visual_xy - center))
    ok = err_m <= vessel.bore_radius_m
    final_z = vessel.z_rim_m - vessel.target_depth_m if ok else vessel.z_rim_m
    peak_f = 0.0 if ok else vessel.stiffness_z_n_m * 0.003
    return InsertionResult(
        success=ok,
        initial_error_mm=err_m * 1000.0,
        final_error_mm=err_m * 1000.0,
        steps=1,
        peak_force_n=peak_f,
        final_z_m=final_z,
    )


def run_tactile_insertion(
    visual_xy: np.ndarray,
    rng: np.random.Generator,
    *,
    vessel: ClearVesselBevel = ClearVesselBevel(),
    arm: Arm2D = DEFAULT_ARM,
    max_steps: int = 80,
    compliance_gain_m_per_n: float = 0.0018,
) -> InsertionResult:
    """Closed-loop compliant tactile insertion driven by 2D contact shear feedback via Arm2D Jacobian.

    Note: 0 < compliance_gain * stiffness_xy = 0.54 < 1 guarantees monotonic non-overshooting convergence.
    """
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    q = arm.ik(visual_xy, elbow=1)
    if q is None:
        q = arm.ik(visual_xy, elbow=-1)
    if q is None:
        raise ValueError(f"Visual estimate {visual_xy} is outside arm workspace")

    _, tip_init = arm.fk(q)
    init_err_mm = float(np.linalg.norm(tip_init - center)) * 1000.0

    z = vessel.z_rim_m + 0.001
    z_goal = vessel.z_rim_m - vessel.target_depth_m
    peak_force = 0.0
    shear_threshold_n = 0.004  # ~0.013 mm radial deflection above bore_radius_m

    for step_idx in range(1, max_steps + 1):
        _, tip = arm.fk(q)
        f_shear, f_z = _contact_wrench(tip, z, vessel, rng)
        f_mag = float(np.hypot(float(np.linalg.norm(f_shear)), f_z))
        peak_force = max(peak_force, f_mag)
        shear_norm = float(np.linalg.norm(f_shear))

        if shear_norm > shear_threshold_n:
            dx = np.clip(compliance_gain_m_per_n * f_shear, -0.0020, 0.0020)
            J = arm.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(2), dx)
            q = q + dq
            if f_z < 0.15:
                z = max(z_goal, z - 0.0004)
        else:
            dz = 0.0006 if z > vessel.z_rim_m - vessel.bevel_depth_m else 0.0020
            z = max(z_goal, z - dz)
            if z <= z_goal + 1e-9:
                _, tip_final = arm.fk(q)
                final_err_mm = float(np.linalg.norm(tip_final - center)) * 1000.0
                return InsertionResult(
                    success=final_err_mm < 0.2,
                    initial_error_mm=init_err_mm,
                    final_error_mm=final_err_mm,
                    steps=step_idx,
                    peak_force_n=peak_force,
                    final_z_m=z,
                )

    _, tip_end = arm.fk(q)
    final_err_mm = float(np.linalg.norm(tip_end - center)) * 1000.0
    return InsertionResult(
        success=False,
        initial_error_mm=init_err_mm,
        final_error_mm=final_err_mm,
        steps=max_steps,
        peak_force_n=peak_force,
        final_z_m=z,
    )
