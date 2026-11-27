"""Unit and benchmark tests for lab-hands community_extensions (Idea #19 & issues #26..#28)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from community_extensions import (  # noqa: E402
    DEFAULT_CORNER_OFFSETS_XY,
    OpenWellPlateSpec,
    compute_fundamental_slosh_omega,
    plan_slosh_bounded_min_jerk,
    plan_unscaled_trapezoidal,
    run_benchmark,
    simulate_meniscus_slosh,
)
from community_extensions.slosh_minimum_jerk import (  # noqa: E402
    _eval_joint_curve,
    _ik_elbow_down,
    _min_jerk_law,
    _plan_safe_joint_spline,
    _trajectory_min_clearance_mm,
)
from lab_hands import in_collision, min_clearance  # noqa: E402


def test_fundamental_cylindrical_slosh_dispersion() -> None:
    from lab_hands.trajectory import SLOSH_BESSEL_XI_11, SLOSH_GRAVITY_M_S2, slosh_bond_number, slosh_natural_frequency

    r_96 = 0.00345
    for fill_mm in (0.5, 2.0, 9.2):
        h_m = fill_mm / 1000.0
        omega = slosh_natural_frequency(r_96, h_m)
        omega_ext = compute_fundamental_slosh_omega(r_96, h_m)
        expected = math.sqrt((SLOSH_BESSEL_XI_11 * SLOSH_GRAVITY_M_S2 / r_96) * math.tanh(SLOSH_BESSEL_XI_11 * h_m / r_96))
        assert omega == pytest.approx(expected, rel=1e-12)
        assert omega_ext == pytest.approx(expected, rel=1e-12)

    # At 0.5 mm fill (R = 3.45 mm), standard linear Bessel dispersion gives ~36.9 rad/s (not 18.9 rad/s!)
    assert slosh_natural_frequency(0.00345, 0.0005) == pytest.approx(36.9383, abs=0.05)
    assert 70.0 < compute_fundamental_slosh_omega(0.00345, 0.0090) < 75.0
    assert slosh_bond_number(0.00345) == pytest.approx(1.621, abs=0.05)

    with pytest.raises(ValueError, match="strictly positive"):
        compute_fundamental_slosh_omega(0.0, 0.005)


def test_continuous_segment_clearance_matches_core_issue_26() -> None:
    # Reagent bottle corner (0.15, 0.34) grazing config where legacy 21-pt sampler reported 6.27 mm
    q_graze = np.array([1.270833, -0.520516])
    exact_mm = min_clearance(q_graze) * 1000.0
    ext_mm = _trajectory_min_clearance_mm(q_graze[None, :])
    assert in_collision(q_graze) is True
    assert exact_mm < 3.0
    assert ext_mm == pytest.approx(exact_mm, abs=1e-6)

    # Capsule radius subtracted identically in extension helper
    ext_caps_mm = _trajectory_min_clearance_mm(q_graze[None, :], capsule_radius_m=0.002)
    assert ext_caps_mm == pytest.approx(max(0.0, exact_mm - 2.0), abs=1e-6)


def test_ik_elbow_down_wraps_q1_and_rejects_colliding_targets_issue_27() -> None:
    # Negative-x quadrant target must wrap q1 into [-pi, pi)
    q_neg = _ik_elbow_down((-0.25, -0.25))
    assert -math.pi <= float(q_neg[0]) < math.pi

    # Target inside reagent bottle obstacle (0.15..0.24, 0.27..0.34) must raise ValueError
    with pytest.raises(ValueError, match="violate >= 5.0 mm clearance"):
        _ik_elbow_down((0.20, 0.30))


def test_corner_well_rotational_slosh_worst_case_task_issue_28() -> None:
    # Benchmark task #34 from run_benchmark(seed=20261123) where ignoring Euler + centripetal
    # rotational acceleration at corner wells previously reached 0.8271 >= 0.70 * h_free
    start_xy = (0.001243421432287256, 0.3769753850959561)
    goal_xy = (0.2511232755051319, 0.028173808106823576)
    plate = OpenWellPlateSpec(fill_height_m=0.009463602010843192, damping_ratio=0.03265264103313463)

    mj = plan_slosh_bounded_min_jerk(start_xy, goal_xy, plate)
    q_s = _ik_elbow_down(start_xy)
    q_g = _ik_elbow_down(goal_xy, prev_q=q_s)
    _, q_v = _plan_safe_joint_spline(q_s, q_g)

    tau = np.linspace(0.0, 1.0, 161)
    dt = mj.duration_s / 160.0
    s, ds_dt, d2s_dt2 = _min_jerk_law(tau, mj.duration_s)
    q_tr, acc_center = _eval_joint_curve(q_s, q_v, q_g, s, ds_dt, d2s_dt2)

    if q_v is None:
        dq_ds = np.broadcast_to(q_g - q_s, (161, 2))
        d2q_ds2 = np.zeros((161, 2), dtype=np.float64)
    else:
        dq_ds = 2.0 * (1.0 - s[:, None]) * (q_v - q_s)[None, :] + 2.0 * s[:, None] * (q_g - q_v)[None, :]
        d2q_ds2 = np.broadcast_to(2.0 * (q_g - 2.0 * q_v + q_s), (161, 2))
    qd = dq_ds * ds_dt[:, None]
    qdd = dq_ds * d2s_dt2[:, None] + d2q_ds2 * (ds_dt[:, None] ** 2)
    q12, w12, a12 = q_tr[:, 0] + q_tr[:, 1], qd[:, 0] + qd[:, 1], qdd[:, 0] + qdd[:, 1]
    c12, s12 = np.cos(q12), np.sin(q12)

    corner_ratios = []
    for rxb, ryb in DEFAULT_CORNER_OFFSETS_XY:
        rx, ry = c12 * rxb - s12 * ryb, s12 * rxb + c12 * ryb
        acc_corner = acc_center + np.column_stack([-a12 * ry - (w12**2) * rx, a12 * rx - (w12**2) * ry])
        corner_ratios.append(simulate_meniscus_slosh(acc_corner, dt, plate) / plate.free_headspace_m)

    assert max(corner_ratios) < 0.70
    assert mj.meniscus_to_headspace_ratio < 0.70

    # Locked-yaw wrist gimbal eliminates rotational terms and runs faster than rigid forearm mount
    gimbal_plate = OpenWellPlateSpec(
        fill_height_m=plate.fill_height_m,
        damping_ratio=plate.damping_ratio,
        rigid_forearm_mount=False,
    )
    mj_gimbal = plan_slosh_bounded_min_jerk(start_xy, goal_xy, gimbal_plate)
    assert mj_gimbal.duration_s <= mj.duration_s


def test_slosh_scaler_scales_longer_for_tighter_headspace() -> None:
    start = (-0.05, 0.40)
    goal = (0.24, 0.02)
    roomy = OpenWellPlateSpec(fill_height_m=0.0075)   # 3.17 mm headspace
    tight = OpenWellPlateSpec(fill_height_m=0.0098)   # 0.87 mm headspace

    res_roomy = plan_slosh_bounded_min_jerk(start, goal, roomy)
    res_tight = plan_slosh_bounded_min_jerk(start, goal, tight)

    assert res_tight.duration_s > res_roomy.duration_s
    assert res_roomy.meniscus_to_headspace_ratio < 0.70
    assert res_tight.meniscus_to_headspace_ratio < 0.70
    assert res_roomy.spilled is False
    assert res_tight.spilled is False
    assert res_tight.min_obstacle_clearance_mm >= 5.0
    assert res_tight.max_abs_q2_rad <= 2.60


def test_unscaled_trapezoidal_spills_tight_well_while_min_jerk_succeeds() -> None:
    start = (-0.05, 0.40)
    goal = (0.27, 0.00)
    tight = OpenWellPlateSpec(fill_height_m=0.0095, damping_ratio=0.022)

    trap = plan_unscaled_trapezoidal(start, goal, tight)
    mj = plan_slosh_bounded_min_jerk(start, goal, tight)

    assert trap.spilled is True
    assert mj.spilled is False
    assert mj.meniscus_to_headspace_ratio < 0.70
    assert mj.min_obstacle_clearance_mm >= 5.0


def test_rejects_overfilled_well_with_zero_headspace() -> None:
    overfilled = OpenWellPlateSpec(well_depth_m=0.010, fill_height_m=0.010)
    with pytest.raises(ValueError, match="non-positive free headspace"):
        plan_slosh_bounded_min_jerk((-0.05, 0.40), (0.24, 0.02), overfilled)


def test_benchmark_60_open_plate_transport_tasks() -> None:
    report = run_benchmark()
    assert report["n_tasks"] == 60
    assert report["min_obstacle_clearance_mm"] >= 5.0

    trap = report["unscaled_trapezoidal"]
    mj = report["slosh_bounded_min_jerk"]

    assert trap["well_spill_rate_pct"] > 50.0
    assert mj["well_spill_rate_pct"] == 0.0
    assert mj["spilled_tasks"] == 0
    assert mj["max_meniscus_to_headspace_ratio"] < 0.70
    assert mj["min_clearance_mm"] >= 5.0
    assert 0.25 < mj["mean_transport_time_sec"] < 2.5
