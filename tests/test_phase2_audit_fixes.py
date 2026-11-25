"""Regression test suite for Phase-2 robotics & formal audit issues #26..#31."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from community_extensions import OpenWellPlateSpec
from lab_hands import (
    DEFAULT_ARM,
    DEFAULT_ARM_SE3,
    DEFAULT_SCENE,
    STEPBOT_LABWARE_CATALOG,
    Arm2D,
    ArmSE3,
    BenchScene,
    ClearVesselBevel,
    DeckLayout,
    PoseSE3,
    batch_min_clearance,
    build_trajectory_record,
    edge_free,
    edge_free_3d,
    get_plate_geometry,
    ik_solutions,
    in_collision,
    in_collision_3d,
    kinematic_tip_sweep_bound,
    min_clearance,
    parameterize_path,
    path_length,
    path_min_clearance,
    plan_3d_pick_place,
    plan_transfer_3d,
    plan_well_tour,
    rrt_connect,
    run_tactile_insertion,
    segment_aabb_clearance,
    segment_aabb_distance,
    select_ik_branch,
    shortcut,
)


# --- Issue #26: Continuous capsule-to-AABB clearance & thin splash-shield partitions ---
def test_segment_aabb_clearance_and_capsule_radius_consistency_issue_26() -> None:
    divider = np.array([[0.1415, -0.002, 0.1435, 0.002]])
    assert segment_aabb_distance(np.array([0.0, 0.0]), np.array([0.30, 0.0]), divider)[0] == 0.0
    assert segment_aabb_clearance(np.array([0.0, 0.0]), np.array([0.30, 0.0]), divider, capsule_radius=0.01)[0] == 0.0

    arm_caps = Arm2D(clearance_m=0.015, margin_m=0.005)
    scene_18mm = BenchScene(obstacles=((0.10, 0.018, 0.20, 0.08),))
    # Surface clearance = 18 mm - 15 mm = 3 mm (< 5 mm margin -> in_collision is True)
    assert in_collision([0.0, 0.0], arm=arm_caps, scene=scene_18mm) is True
    assert min_clearance([0.0, 0.0], arm=arm_caps, scene=scene_18mm) == pytest.approx(0.003, abs=1e-9)
    assert batch_min_clearance(np.array([[0.0, 0.0]]), arm=arm_caps, scene=scene_18mm)[0] == pytest.approx(
        0.003, abs=1e-9
    )

    # Safe 25 mm centerline distance with 15 mm capsule -> 10 mm surface clearance (>= 5 mm -> collision-free)
    scene_25mm = BenchScene(obstacles=((0.10, 0.025, 0.20, 0.08),))
    assert in_collision([0.0, 0.0], arm=arm_caps, scene=scene_25mm) is False
    assert min_clearance([0.0, 0.0], arm=arm_caps, scene=scene_25mm) == pytest.approx(0.010, abs=1e-9)
    rec = build_trajectory_record(
        trial=1,
        seed=0,
        planner="rrt_connect",
        goal_well="A1",
        raw_path=[np.array([0.0, 0.0]), np.array([0.0, -0.05])],
        smooth_path=[np.array([0.0, 0.0]), np.array([0.0, -0.05])],
        plan_time_s=0.001,
        arm=arm_caps,
        scene=scene_25mm,
    )
    assert rec.min_clearance_m == pytest.approx(0.010, abs=1e-5)


# --- Issue #27: Stateful IK branch continuity & Viterbi multi-well tour optimizer ---
def test_viterbi_well_tour_eliminates_slot11_elbow_flip_issue_27() -> None:
    deck = DeckLayout()
    plate11 = deck.add_labware("p11", "corning_96_wellplate_360ul_flat", slot=11)

    # Verify Slot 11 A5 has 2 branches while A6 only permits elbow = -1
    sa = ik_solutions(plate11.well_xy("A5"))
    sb = ik_solutions(plate11.well_xy("A6"))
    assert len(sa) == 2
    assert len(sb) == 1 and sb[0][1] < 0.0

    # Viterbi multi-well tour across A1..A12 selects elbow = -1 from A1 so A5 -> A6 has zero flip
    row_wells = [f"p11:A{col}" for col in range(1, 13)]
    tour = plan_well_tour(row_wells, deck=deck, rng=np.random.default_rng(0))
    assert tour is not None
    assert tour.elbow_branches == (-1,) * 12
    assert tour.elbow_flips == 0
    # Leg 4 corresponds to A5 -> A6 (9 mm step): < 0.06 rad instead of 3.89 rad!
    leg_a5_a6 = tour.legs[4]
    assert path_length(leg_a5_a6) < 0.06
    for leg_a, leg_b in zip(tour.legs, tour.legs[1:]):
        assert np.allclose(leg_a[-1], leg_b[0], atol=1e-9)
    tp = parameterize_path(tour.waypoints)
    assert tp.duration_s < 1.5

    # Continuity-aware select_ik_branch preserves prior elbow sign
    q_prev_neg = tour.goal_configs[0]
    q_next = select_ik_branch(plate11.well_xy("A2"), prev_q=q_prev_neg)
    assert q_next is not None and q_next[1] < 0.0
    assert float(np.linalg.norm(q_next - q_prev_neg)) < 0.10


# --- Issue #28: Analytical C2 curve derivatives & cylindrical slosh scaling in parameterize_path ---
def test_parameterize_path_analytical_derivatives_and_slosh_scaling_issue_28() -> None:
    start = ik_solutions(DEFAULT_SCENE.tube_xy)[0]
    goals = ik_solutions(DEFAULT_SCENE.plate.well_xy("H12"), prev_q=start)
    raw = rrt_connect(start, goals, np.random.default_rng(2))
    assert raw is not None
    sp = shortcut(raw, np.random.default_rng(2))
    assert len(sp) >= 3  # exercises corner blends

    tp = parameterize_path(sp, n_samples=2001)
    qd_fd = np.gradient(tp.q, tp.t, axis=0, edge_order=2)
    assert float(np.max(np.abs(qd_fd - tp.qd))) < 1e-4

    # Tight open-plate fill scales duration up to keep corner-well slosh < 0.70 * h_free
    tight_plate = OpenWellPlateSpec(fill_height_m=0.01055, damping_ratio=0.025)
    tp_slosh = parameterize_path(sp, n_samples=2001, open_plate=tight_plate)
    assert tp_slosh.duration_s > tp.duration_s


# --- Issue #29: Lipschitz swept-volume segment subdivision soundness ---
def test_lipschitz_edge_free_rejects_interior_grazing_chord_issue_29() -> None:
    # Closest approach (4.99995 mm < 5.0 mm) occurs strictly between grid samples at s = 0.625 (q1 = 0.0)
    scene = BenchScene(obstacles=((0.30499995, -1e-5, 0.320, 1e-5),))
    qa = np.array([-0.0025, 2.5])
    qb = np.array([+0.0015, 2.5])
    q_mid = qa + 0.625 * (qb - qa)

    assert in_collision(qa, scene=scene) is False
    assert in_collision(qb, scene=scene) is False
    assert in_collision(q_mid, scene=scene) is True
    assert min_clearance(q_mid, scene=scene) * 1000.0 < 5.0

    assert kinematic_tip_sweep_bound(qa, qb) == pytest.approx(0.0022, abs=1e-12)
    assert edge_free(qa, qb, scene=scene) is False
    assert path_min_clearance([qa, qb], scene=scene) * 1000.0 < 5.0

    # Verify Lean 4 proof source defines the formal Lipschitz theorems with zero sorry
    lean_src = (Path(__file__).resolve().parents[1] / "lean" / "LabHandsLean.lean").read_text(encoding="utf-8")
    assert "def tipSweepBoundUM" in lean_src
    assert "def halfStepSweepUM" in lean_src
    assert "theorem sweptSegment_continuous_clearance_sound" in lean_src
    assert "sorry" not in lean_src


# --- Issue #30: Tactile insertion joint-limit / obstacle guards and slip/seal flags ---
def test_tactile_insertion_obstacle_and_rim_hangup_guards_issue_30() -> None:
    vessel = ClearVesselBevel(bore_radius_m=0.00008, capture_radius_m=0.0055)
    visual_xy = np.asarray(vessel.center_xy) + np.array([0.0042, 0.0040])  # 5.80 mm > 5.50 mm capture
    ins = run_tactile_insertion(visual_xy, np.random.default_rng(0), vessel=vessel)
    assert ins.success is False
    assert ins.seal_intact is False
    assert ins.slip_detected is True
    assert ins.axial_slip_mm == pytest.approx(12.0, abs=0.5)
    assert ins.final_z_m == pytest.approx(vessel.z_rim_m, abs=1e-6)

    # Obstacle blocking visual_xy returns safe failure instead of executing colliding servo steps
    blocked_scene = BenchScene(obstacles=((0.19, 0.04, 0.21, 0.06),))
    ins_blocked = run_tactile_insertion(
        np.asarray(vessel.center_xy),
        np.random.default_rng(0),
        vessel=vessel,
        scene=blocked_scene,
    )
    assert ins_blocked.success is False
    assert ins_blocked.seal_intact is False
    assert ins_blocked.slip_detected is True


# --- Issue #31: Per-labware Z-rim, attached-tip lift clearance, and 6-DOF SE(3) kinematics ---
def test_falcon_120mm_rim_and_all_24_stepbot_labware_3d_transfers_issue_31() -> None:
    falcon = get_plate_geometry("opentrons_15_tuberack_falcon_15ml_conical")
    assert falcon.z_rim_m == pytest.approx(0.120)
    scene_falcon = BenchScene(plate=falcon)
    assert scene_falcon.z_safe_travel > falcon.z_rim_m

    traj = plan_transfer_3d("A1", np.random.default_rng(0), scene=scene_falcon)
    assert traj is not None
    assert all(pt[2] > falcon.z_rim_m for pt in traj[1:-1])

    # Horizontal transit at legacy 95 mm over Falcon rack (outside A1 bore, e.g. at B2) MUST collide!
    q_b2 = ik_solutions(falcon.well_xy("B2"), scene=scene_falcon)[0]
    assert (
        in_collision_3d(
            [q_b2[0], q_b2[1], 0.095],
            scene=scene_falcon,
            allowed_well_xy=falcon.well_xy("A1"),
        )
        is True
    )

    # Free-space point outside both source tube and labware footprints at z = 50 mm is collision-free
    q_free = ik_solutions((0.05, -0.30), scene=scene_falcon)[0]
    assert in_collision_3d([q_free[0], q_free[1], 0.050], scene=scene_falcon) is False

    # Regression across all 24 STEPBOT_LABWARE_CATALOG entries
    assert len(STEPBOT_LABWARE_CATALOG) == 24
    for load_name, geom in STEPBOT_LABWARE_CATALOG.items():
        sc = BenchScene(plate=geom)
        assert sc.z_safe_travel > max(sc.z_tube_rim, geom.z_rim_m)
        first_well = geom.well_name(1, "row")
        tr = plan_transfer_3d(first_well, np.random.default_rng(0), scene=sc)
        assert tr is not None, f"3D transfer failed for {load_name}:{first_well}"
        assert all(wp[2] > geom.z_rim_m for wp in tr[1:-1])


def test_se3_kinematics_attached_1000ul_tip_and_verticality_guard_issue_31() -> None:
    # 6R ArmSE3 closed-form FK/IK round-trip across tilted and vertical poses
    arm6 = ArmSE3()
    target_pose = PoseSE3(xyz=(0.24, 0.04, 0.22), rpy=(0.01, -0.01, 0.35))
    q6 = arm6.ik_se3(target_pose, elbow=1)
    assert q6 is not None
    recovered = arm6.fk_se3(q6)
    assert np.allclose(recovered.homogeneous_matrix(), target_pose.homogeneous_matrix(), atol=1e-9)

    # Multi-labware deck: pick up 1000 uL filter tip (L_tip = 95 mm, z_rim = 85 mm) and dispense
    # into 2 mL deep-well block (z_rim = 42 mm) with Falcon rack (z_rim = 120 mm) also on deck!
    deck = DeckLayout()
    deck.add_labware("tips1000", "opentrons_96_filtertiprack_1000ul", slot=4)
    deck.add_labware("deep96", "nest_96_wellplate_2ml_deep", slot=5)
    deck.add_labware("falcon15", "opentrons_15_tuberack_falcon_15ml_conical", slot=1)
    scene_deck = BenchScene(deck=deck)

    plan = plan_3d_pick_place(
        "tips1000:A1",
        "deep96:H12",
        np.random.default_rng(0),
        arm=DEFAULT_ARM,
        arm_se3=DEFAULT_ARM_SE3,
        scene=scene_deck,
        deck=deck,
    )
    assert plan is not None
    assert plan.collision_free is True
    assert plan.attached_tip_length_m == pytest.approx(0.095)
    # Must clear tallest deck labware (120 mm Falcon) + 95 mm mounted tip = 215 mm + margin!
    assert plan.z_safe_transit_m >= 0.120 + 0.095 + 0.005
    assert plan.z_safe_transit_m <= scene_deck.z_max
    assert plan.q6_waypoints is not None and len(plan.q6_waypoints) == len(plan.waypoints_3d)
    assert plan.max_tool_tilt_rad == pytest.approx(0.0, abs=1e-12)

    # Excessive tool tilt (2.2 deg > 1.5 deg) into narrow 4.1 mm deep-well block is rejected!
    tilted_plan = plan_3d_pick_place(
        "tips1000:A1",
        "deep96:H12",
        np.random.default_rng(0),
        scene=scene_deck,
        deck=deck,
        tool_rpy=(math.radians(2.2), 0.0, 0.0),
    )
    assert tilted_plan is None
