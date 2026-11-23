"""Unit and benchmark tests for lab-hands community_extensions (Idea #19)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from community_extensions import (  # noqa: E402
    OpenWellPlateSpec,
    compute_fundamental_slosh_omega,
    plan_slosh_bounded_min_jerk,
    plan_unscaled_trapezoidal,
    run_benchmark,
)


def test_fundamental_cylindrical_slosh_dispersion() -> None:
    omega = compute_fundamental_slosh_omega(well_radius_m=0.00345, fill_height_m=0.0090)
    expected = math.sqrt(1.841 * 9.80665 / 0.00345) * math.tanh(1.841 * 0.0090 / 0.00345)
    assert abs(omega - expected) < 1e-9
    assert 70.0 < omega < 75.0

    with pytest.raises(ValueError, match="strictly positive"):
        compute_fundamental_slosh_omega(0.0, 0.005)


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
