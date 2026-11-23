"""Comprehensive pytest suite for lab-hands covering issues #1..#24."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from demo.plan_to_well import (
    L1,
    L2,
    PITCH,
    PLATE_ORIGIN,
    TUBE,
    edge_free,
    fk,
    ik,
    in_collision,
    length,
    main as plan_main,
    min_clearance,
    rrt,
    rrt_connect,
    shortcut,
    well_name,
    well_xy,
)
from demo.tactile_insert import main as tactile_main
from lab_hands import (
    STEPBOT_LABWARE_CATALOG,
    TRAJECTORY_SCHEMA,
    Arm2D,
    BenchScene,
    ClearVesselBevel,
    DeckLayout,
    PlateGeometry,
    TrajectoryRecord,
    edge_free_3d,
    edge_subdivision_count,
    get_plate_geometry,
    ik_solutions,
    in_collision_3d,
    parameterize_path,
    parse_well_ref,
    path_min_clearance,
    plan_transfer_3d,
    run_open_loop_insertion,
    run_tactile_insertion,
    segment_aabb_distance,
    wrap_angle,
)


# --- Issues #1 & #2: SBS 96-well bounds and formatting ---
def test_well_name_bounds_and_orders():
    assert well_name(1, "row") == "A1"
    assert well_name(12, "row") == "A12"
    assert well_name(13, "row") == "B1"
    assert well_name(65, "row") == "F5"
    assert well_name(96, "row") == "H12"

    assert well_name(1, "col") == "A1"
    assert well_name(8, "col") == "H1"
    assert well_name(9, "col") == "A2"
    assert well_name(65, "col") == "A9"
    assert well_name(96, "col") == "H12"

    assert len({well_name(i, "row") for i in range(1, 97)}) == 96
    assert {well_name(i, "row") for i in range(1, 97)} == {well_name(i, "col") for i in range(1, 97)}

    for bad_idx in (0, -1, 97, True, False, 1.5):  # type: ignore[list-item]
        with pytest.raises(ValueError):
            well_name(bad_idx, "row")  # type: ignore[arg-type]

    for bad_order in ("row_major", "column", "", "ROW"):
        with pytest.raises(ValueError):
            well_name(1, bad_order)


def test_well_xy_normalization_and_validation():
    assert np.allclose(well_xy("A1"), PLATE_ORIGIN)
    assert np.allclose(well_xy(" a1 "), PLATE_ORIGIN)
    assert np.allclose(well_xy("h12"), PLATE_ORIGIN + np.array([11 * PITCH, -7 * PITCH]))

    for bad_well in ("A0", "A13", "A99", "B-2", "I1", "A01", "", "12"):
        with pytest.raises(ValueError):
            well_xy(bad_well)


# --- Issues #3 & #4: Continuous segment-AABB collision and closed-box margin ---
def test_corner_grazing_detected_where_15_point_sampling_tunnels():
    # Construct a tiny obstacle corner between the 7th and 8th of 15 discrete samples along Link 1
    # Sample 6 is at x = 6/14 * 0.30 = 0.12857 m; Sample 7 is at x = 7/14 * 0.30 = 0.15000 m
    arm_zero = Arm2D(l1=0.30, l2=0.25, clearance_m=0.0, margin_m=0.0)
    thin_corner_scene = BenchScene(obstacles=((0.136, -0.002, 0.142, 0.002),))
    q_zero = np.array([0.0, 0.0])

    # Verify 15-point discrete sampling misses (0.136..0.142), whereas exact segment-AABB catches it
    pts_15 = np.linspace(0.0, 0.30, 15)
    assert not np.any((pts_15 > 0.136) & (pts_15 < 0.142))
    assert segment_aabb_distance(np.zeros(2), np.array([0.30, 0.0]), thin_corner_scene.obstacle_array)[0] == 0.0
    assert in_collision(q_zero, margin_m=0.0) or _check_scene_collision(q_zero, arm_zero, thin_corner_scene)


def _check_scene_collision(q, arm, scene):
    from lab_hands.collision import in_collision as scene_in_collision

    return scene_in_collision(q, arm=arm, scene=scene)


def test_closed_box_boundary_and_safety_margin():
    # Segment lying directly on top edge y = 0.10 of box [0.10, 0.00, 0.20, 0.10] has distance 0 -> collision
    box = np.array([[0.10, 0.00, 0.20, 0.10]])
    d_touch = segment_aabb_distance(np.array([0.05, 0.10]), np.array([0.25, 0.10]), box)
    assert d_touch[0] == 0.0

    arm_tight = Arm2D(l1=0.30, l2=0.25, clearance_m=0.0, margin_m=0.0)
    scene_touch = BenchScene(obstacles=((0.10, -0.10, 0.20, 0.00),))
    # At q = (0, 0), Link 1 lies along y = 0.0, touching ymax = 0.00 of obstacle
    assert _check_scene_collision([0.0, 0.0], arm_tight, scene_touch) is True

    # With margin_m = 0.005, a pass 4 mm away from obstacle is rejected as unsafe
    scene_4mm = BenchScene(obstacles=((0.10, 0.004, 0.20, 0.10),))
    arm_5mm = Arm2D(l1=0.30, l2=0.25, clearance_m=0.0, margin_m=0.005)
    assert _check_scene_collision([0.0, 0.0], arm_5mm, scene_4mm) is True


# --- Issues #5, #6, #11: IK boundary clamping, list/tuple inputs, angle wrapping, joint limits ---
def test_ik_boundary_reachability_tuples_and_wrapping():
    # Tuple and list inputs must work without TypeError
    q_tup = ik((0.20, 0.05))
    q_lst = ik([0.20, 0.05])
    assert q_tup is not None and q_lst is not None
    assert np.allclose(q_tup, q_lst)

    # Exact outer boundary ||p|| = L1 + L2 = 0.55 m (subject to IEEE-754 rounding)
    for angle in np.linspace(-2.5, 2.5, 17):
        target = ((L1 + L2) * math.cos(angle), (L1 + L2) * math.sin(angle))
        sol = ik(target, elbow=1)
        assert sol is not None
        assert -math.pi <= sol[0] <= math.pi
        _, tip = fk(sol)
        assert np.linalg.norm(tip - np.asarray(target)) < 1e-12

    # Unreachable target returns None
    assert ik((0.80, 0.0)) is None
    assert ik((0.01, 0.0)) is None

    # Inner workspace folding beyond |q2| > 2.6 rad is rejected by joint limits
    # At r = 0.06 m, cos(q2) = (0.0036 - 0.09 - 0.0625) / 0.15 = -0.9927 -> |q2| = 3.02 rad > 2.6 rad
    assert ik((0.06, 0.0)) is None
    assert in_collision([0.0, 2.9]) is True

    # Negative-x quadrant targets wrap q1 cleanly into [-pi, pi)
    q_negx = ik((-0.25, -0.25), elbow=1)
    assert q_negx is not None
    assert -math.pi <= q_negx[0] < math.pi
    assert np.linalg.norm(fk(q_negx)[1] - np.array([-0.25, -0.25])) < 1e-12
    assert -math.pi <= wrap_angle(-3.5 * math.pi) < math.pi


def test_fk_ik_roundtrip_all_96_wells():
    for idx in range(1, 97):
        p = well_xy(well_name(idx, "row"))
        for elbow in (1, -1):
            q = ik(p, elbow=elbow)
            assert q is not None
            _, tip = fk(q)
            assert float(np.linalg.norm(tip - p)) < 1e-12


# --- Issues #7, #8, #9, #12, #13, #14, #19: RRT, RRT-Connect, shortcut, Jacobian step bound ---
def test_edge_subdivision_jacobian_lipschitz_bound():
    a = np.array([0.0, 0.0])
    b = np.array([0.02, 0.02])  # joint norm = 0.0283 rad, tip sweep upper bound = 0.55*0.02 + 0.25*0.02 = 16 mm
    n = edge_subdivision_count(a, b, max_tip_step_m=0.002)
    assert n >= 9  # 16 mm / 2 mm + 1 = 9 subdivisions


def test_rrt_and_rrt_connect_guards_and_no_duplicate_goal():
    rng = np.random.default_rng(0)
    # Invalid colliding start or goal immediately returns None
    colliding_q = np.array([0.0, 3.0])  # violates |q2| <= 2.6
    valid_q = ik(TUBE, elbow=1)
    assert valid_q is not None
    assert rrt(colliding_q, valid_q, rng) is None
    assert rrt(valid_q, colliding_q, rng) is None
    assert rrt_connect(valid_q, [], rng) is None

    # Goal within step radius with unobstructed edge returns direct 2-node path without duplicate goal
    nearby_q = valid_q + np.array([0.04, 0.02])
    assert edge_free(valid_q, nearby_q)
    p_direct = rrt(valid_q, nearby_q, rng, goal_bias=1.0)
    assert p_direct is not None
    assert len(p_direct) == 2
    assert not np.allclose(p_direct[-1], p_direct[-2])

    # Multi-branch goal planning + shortcutting on obstructed well A12
    goals = ik_solutions(well_xy("A12"))
    assert len(goals) >= 1
    for solver in (rrt, rrt_connect):
        path = solver(valid_q, goals, np.random.default_rng(3))
        assert path is not None
        # Verify zero consecutive duplicate nodes
        assert all(not np.allclose(x, y, atol=1e-9) for x, y in zip(path, path[1:]))
        sp = shortcut(path, np.random.default_rng(3))
        assert length(sp) <= length(path) + 1e-9
        assert all(edge_free(x, y) for x, y in zip(sp, sp[1:]))
        assert min_clearance(sp[-1]) > 0.005


def test_rrt_connect_beats_forward_rrt_tail_latency():
    import time

    start = ik_solutions(TUBE)[0]
    goals = ik_solutions(well_xy("H12"))
    t_rrt, t_conn = [], []
    for seed in range(8):
        t0 = time.perf_counter()
        p_f = rrt(start, goals, np.random.default_rng(seed))
        t_rrt.append(time.perf_counter() - t0)
        assert p_f is not None

        t0 = time.perf_counter()
        p_c = rrt_connect(start, goals, np.random.default_rng(seed))
        t_conn.append(time.perf_counter() - t0)
        assert p_c is not None

    assert np.mean(t_conn) < np.mean(t_rrt)


# --- Issue #17: 2.5D safe-retract Z clearance over tube rims and plates ---
def test_scara_3d_retract_transit_plunge_safety():
    q_tube = ik_solutions(TUBE)[0]
    w_xy = well_xy("D6")
    # Translating horizontally while tip is lowered inside the source tube (z = 0.02 m < z_tube_rim = 0.075 m) must fail!
    a3_low = np.array([q_tube[0], q_tube[1], 0.020])
    b3_low = np.array([q_tube[0] + 0.05, q_tube[1], 0.020])
    assert in_collision_3d(a3_low, allowed_well_xy=w_xy) is False
    assert edge_free_3d(a3_low, b3_low, allowed_well_xy=w_xy) is False

    # Full 3-phase pick-and-place trajectory succeeds and every 3D segment is rim-safe
    traj_3d = plan_transfer_3d("D6", np.random.default_rng(0))
    assert traj_3d is not None and len(traj_3d) >= 4
    assert traj_3d[0][2] < 0.030  # starts inside source tube
    assert all(pt[2] >= 0.095 - 1e-9 for pt in traj_3d[1:-1])  # transits at safe Z
    assert traj_3d[-1][2] < 0.015  # plunges into target well
    assert all(edge_free_3d(x, y, allowed_well_xy=w_xy) for x, y in zip(traj_3d, traj_3d[1:]))


# --- Issue #18: Time-parameterized minimum-jerk / bounded liquid-slosh acceleration ---
def test_time_parameterization_slosh_and_velocity_bounds():
    start = ik_solutions(TUBE)[0]
    goals = ik_solutions(well_xy("F5"))
    raw = rrt_connect(start, goals, np.random.default_rng(1))
    assert raw is not None
    sp = shortcut(raw, np.random.default_rng(1))

    tp = parameterize_path(sp, qd_max=1.2, qdd_max=3.5, a_max_liquid=0.8)
    assert tp.duration_s > 0.0
    assert tp.max_joint_vel_rad_s <= 1.2 + 1e-3
    assert tp.max_joint_acc_rad_s2 <= 3.5 + 1e-2
    assert tp.max_tip_acc_m_s2 <= 0.8 + 1e-2
    # Zero velocity and zero acceleration at rest endpoints
    assert np.allclose(tp.qd[0], 0.0, atol=1e-9)
    assert np.allclose(tp.qd[-1], 0.0, atol=1e-9)
    assert np.allclose(tp.qdd[0], 0.0, atol=1e-9)
    assert np.allclose(tp.qdd[-1], 0.0, atol=1e-9)


# --- Issue #20: Tactile last-centimeter compliant insertion ---
def test_tactile_compliant_insertion_resolves_3mm_visual_error():
    vessel = ClearVesselBevel(bore_radius_m=0.00008, capture_radius_m=0.0055)
    noisy_xy = np.asarray(vessel.center_xy) + np.array([0.0028, -0.0025])
    assert run_open_loop_insertion(noisy_xy, vessel=vessel).success is False
    res = run_tactile_insertion(noisy_xy, np.random.default_rng(0), vessel=vessel)
    assert res.success is True
    assert res.final_error_mm < 0.2
    assert tactile_main(["--trials", "30", "--seed", "0"]) == 0


# --- Issue #21 & #22: Configurable PlateGeometry and 24-labware stepbot catalog ---
def test_stepbot_24_labware_catalog_and_deck_refs():
    assert len(STEPBOT_LABWARE_CATALOG) == 24
    for n_expected, load_name in [
        (6, "corning_6_wellplate_16.8ml_flat"),
        (12, "corning_12_wellplate_6.9ml_flat"),
        (24, "corning_24_wellplate_3.4ml_flat"),
        (96, "corning_96_wellplate_360ul_flat"),
        (384, "corning_384_wellplate_112ul_flat"),
        (10, "opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical"),
    ]:
        geom: PlateGeometry = get_plate_geometry(load_name)
        assert geom.n_wells == n_expected
        first = geom.well_name(1, "row")
        last = geom.well_name(n_expected, "row")
        assert geom.well_xy(first).shape == (2,)
        assert geom.well_xy(last).shape == (2,)

    assert parse_well_ref("plate:p24") == ("plate", "P24")
    deck = DeckLayout()
    deck.add_labware("plate", "corning_384_wellplate_112ul_flat", slot=5)
    deck.add_labware("reservoir", "nest_12_reservoir_15ml", slot=2)
    p_p24 = deck.well_xy("plate:P24")
    p_a1 = deck.well_xy("plate:A1")
    assert np.allclose(p_p24 - p_a1, [23 * 0.0045, -15 * 0.0045])
    assert deck.well_xy("reservoir:A12").shape == (2,)


# --- Issues #10, #23, #24: CLI JSONL export, clearance reporting, and non-zero failure exit ---
def test_cli_jsonl_export_and_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_jsonl = tmp_path / "trajectories.jsonl"
    rc = plan_main(["--planner", "rrt_connect", "--seeds", "2", "--out-jsonl", str(out_jsonl)])
    assert rc == 0
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20  # 10 target wells x 2 seeds
    recs = [TrajectoryRecord.from_json(line) for line in lines]
    assert all(r.schema == TRAJECTORY_SCHEMA for r in recs)
    assert all(r.collision_free and r.joint_limits_satisfied for r in recs)
    assert all(r.min_clearance_m > 0.005 for r in recs)
    assert path_min_clearance([np.asarray(q) for q in recs[0].joint_waypoints]) > 0.005

    # Verify CLI exits non-zero if start configuration is unreachable
    import demo.plan_to_well as ptw

    monkeypatch.setattr(ptw, "TUBE", np.array([2.0, 2.0]))
    with pytest.raises(SystemExit) as exc:
        ptw.main(["--seeds", "1"])
    assert exc.value.code == 1
