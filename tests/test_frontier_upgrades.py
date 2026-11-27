"""Comprehensive regression & physics tests for Frontier Upgrades 1..4 in `lab-hands`."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lab_hands import (
    BESSEL_XI_11,
    DEFAULT_OPENTRONS_FLEX,
    DEFAULT_PANDA,
    DEFAULT_SCENE_3D,
    DEFAULT_UR5E,
    LABWARE_SLOSH_SPECS,
    LIQUID_CATALOG,
    BenchScene3D,
    ClearVesselBevel,
    PoseSE3,
    benchmark_tactile_regimes,
    capsule_aabb_clearance_3d,
    check_whitney_jamming_wedging,
    compute_bessel_slosh_omega,
    compute_viscous_slosh_damping,
    convert_finebio_to_se3_plans,
    design_input_shaper,
    edge_free_se3,
    extract_opentrons_runlog_trajectory,
    in_collision_se3,
    informed_rrt_star,
    inverse_se3,
    load_finebio_annotations,
    load_lerobot_episode,
    log_se3,
    log_so3,
    min_clearance_se3,
    parameterize_toppra,
    path_min_clearance_se3,
    plan_3d_pick_place,
    plan_to_opentrons_trajectory,
    residual_vibration_ratio,
    rrt_connect_se3,
    run_active_tactile_insertion,
    run_factorial_slosh_benchmark,
    run_local_shear_insertion,
    run_open_loop_insertion,
    run_passive_rcc_insertion,
    sample_informed_ellipsoid,
    segment_aabb_distance_3d,
    shape_toppra_trajectory,
    shortcut_se3,
    simulate_nonlinear_bessel_slosh,
    trajectory_to_lerobot_tensors,
    validate_lerobot_frame_tensors,
    validate_opentrons_trajectory,
)


# ---------------------------------------------------------------------------
# Upgrade 1: 6-DoF UR5e, 7-DoF Panda, Opentrons Flex, 3D Collision & TOPP-RA
# ---------------------------------------------------------------------------


def test_ur5e_and_panda_poe_matches_dh_and_mdh_bit_exact() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(25):
        q_u = DEFAULT_UR5E.sample_configuration(rng)
        assert np.allclose(DEFAULT_UR5E.fk_poe(q_u), DEFAULT_UR5E.fk_dh(q_u), atol=1e-13)

        q_p = DEFAULT_PANDA.sample_configuration(rng)
        assert np.allclose(DEFAULT_PANDA.fk_poe(q_p), DEFAULT_PANDA.fk_mdh(q_p), atol=1e-13)


def test_space_body_and_world_jacobians_and_lie_accelerations() -> None:
    rng = np.random.default_rng(101)
    eps = 1e-7
    for arm in (DEFAULT_UR5E, DEFAULT_PANDA, DEFAULT_OPENTRONS_FLEX):
        q = arm.sample_configuration(rng)
        J_w = arm.world_jacobian(q)
        J_b = arm.body_jacobian(q)
        T0 = arm.fk_poe(q)

        Jw_fd = np.zeros_like(J_w)
        Jb_fd = np.zeros_like(J_b)
        for j in range(arm.dof):
            dq = np.zeros(arm.dof, dtype=np.float64)
            dq[j] = eps
            T1 = arm.fk_poe(q + dq)
            dp = (T1[:3, 3] - T0[:3, 3]) / eps
            dw = (T0[:3, :3] @ log_so3(T0[:3, :3].T @ T1[:3, :3])) / eps
            Jw_fd[:, j] = np.concatenate([dp, dw])
            Jb_fd[:, j] = log_se3(inverse_se3(T0) @ T1) / eps

        assert np.allclose(J_w, Jw_fd, atol=1e-6)
        assert np.allclose(J_b, Jb_fd, atol=1e-6)

    # Verify analytical Lie-bracket 3D end-effector acceleration against finite differences
    t = np.linspace(0.0, 0.8, 321)
    freq = 2.0 * math.pi / 0.8
    dir6 = np.array([0.8, -0.5, 0.7, -0.4, 0.6, 0.9], dtype=np.float64)
    q_tr = DEFAULT_UR5E.home_q[None, :] + 0.18 * np.sin(freq * t)[:, None] * dir6[None, :]
    qd_tr = 0.18 * freq * np.cos(freq * t)[:, None] * dir6[None, :]
    qdd_tr = -0.18 * (freq**2) * np.sin(freq * t)[:, None] * dir6[None, :]

    _, _, lv, av, la, aa = DEFAULT_UR5E.trajectory_ee_kinematics(
        t, q_tr, qd_tr, qdd_tr, attached_tip_length_m=0.058
    )
    la_fd = np.gradient(lv, t, axis=0, edge_order=2)
    aa_fd = np.gradient(av, t, axis=0, edge_order=2)
    assert float(np.max(np.abs(la[5:-5] - la_fd[5:-5]))) < 5e-3
    assert float(np.max(np.abs(aa[5:-5] - aa_fd[5:-5]))) < 5e-3


def test_dls_ik_nullspace_optimization_on_ur5e_and_7dof_panda() -> None:
    from lab_hands.kinematics_se3 import make_panda_arm

    bench_panda = make_panda_arm(base_offset_xyz=(-0.16, 0.0, 0.0))
    targets = [
        PoseSE3(xyz=(0.24, 0.04, 0.18), rpy=(0.0, 0.0, 0.0)),
        PoseSE3(xyz=(-0.05, 0.40, 0.16), rpy=(0.0, 0.0, 0.25)),
        PoseSE3(xyz=(0.30, -0.02, 0.20), rpy=(0.01, -0.01, -0.35)),
    ]
    for tgt in targets:
        ru = DEFAULT_UR5E.dls_ik(tgt, attached_tip_length_m=0.058)
        assert ru.success is True
        assert ru.pos_error_m <= 1e-5
        assert ru.rot_error_rad <= 1e-4
        tip_u = DEFAULT_UR5E.fk(ru.q).tip_position(attached_tip_length_m=0.058)
        assert np.allclose(tip_u, tgt.xyz, atol=2e-5)

        # Compare redundant 7-DoF Panda IK with vs without null-space manipulability optimization
        rp_no_ns = bench_panda.dls_ik(
            tgt, attached_tip_length_m=0.058, nullspace_limit_gain=0.0, nullspace_manip_gain=0.0
        )
        rp_opt = bench_panda.dls_ik(
            tgt, attached_tip_length_m=0.058, nullspace_limit_gain=0.35, nullspace_manip_gain=0.06
        )
        assert rp_no_ns.success and rp_opt.success
        assert bench_panda.within_joint_limits(rp_opt.q)
        assert rp_opt.manipulability >= rp_no_ns.manipulability - 1e-4


def test_3d_capsule_aabb_collision_and_swept_se3_lipschitz_check() -> None:
    # 1. Exact 3D thin splash-shield piercing test (2 mm thick partition in X)
    thin_box = np.array([[0.150, -0.05, 0.00, 0.152, 0.05, 0.30]], dtype=np.float64)
    d_pierce = segment_aabb_distance_3d([0.0, 0.0, 0.15], [0.30, 0.0, 0.15], thin_box)
    assert float(d_pierce[0]) == 0.0

    # 2. 3D capsule surface clearance subtracts capsule radius accurately
    box_above = np.array([[0.10, -0.05, 0.25, 0.20, 0.05, 0.35]], dtype=np.float64)
    cl = capsule_aabb_clearance_3d([0.0, 0.0, 0.20], [0.30, 0.0, 0.20], 0.018, box_above)
    assert float(cl[0]) == pytest.approx(0.050 - 0.018, abs=1e-9)

    # 3. Continuous swept 6-DoF UR5e chord blocked by interior 3D obstacle
    q_a = DEFAULT_UR5E.dls_ik(PoseSE3(xyz=(0.10, 0.32, 0.14), rpy=(0.0, 0.0, 0.0))).q
    q_b = DEFAULT_UR5E.dls_ik(PoseSE3(xyz=(0.28, 0.30, 0.14), rpy=(0.0, 0.0, 0.0)), q_seed=q_a).q
    # Reagent bottle sits at (0.15..0.24, 0.27..0.34, 0.00..0.23) directly between q_a and q_b!
    assert in_collision_se3(q_a, DEFAULT_UR5E, DEFAULT_SCENE_3D) is False
    assert in_collision_se3(q_b, DEFAULT_UR5E, DEFAULT_SCENE_3D) is False
    assert edge_free_se3(q_a, q_b, DEFAULT_UR5E, DEFAULT_SCENE_3D) is False


def test_6dof_informed_rrt_star_and_toppra_kinematic_bounds() -> None:
    rng = np.random.default_rng(19)
    q_s = DEFAULT_UR5E.dls_ik(PoseSE3(xyz=(-0.05, 0.40, 0.16), rpy=(0.0, 0.0, 0.0))).q
    q_g = DEFAULT_UR5E.dls_ik(PoseSE3(xyz=(0.299, -0.013, 0.16), rpy=(0.0, 0.0, 0.0)), q_seed=q_s).q

    # Verify admissible prolate hyperellipsoidal sampler invariant ||q - q_s|| + ||q_g - q|| <= c_best
    c_min = float(np.linalg.norm(q_g - q_s))
    c_best = c_min * 1.35
    for _ in range(40):
        q_samp = sample_informed_ellipsoid(
            q_s, q_g, c_best, DEFAULT_UR5E.q_lower, DEFAULT_UR5E.q_upper, rng
        )
        assert float(np.linalg.norm(q_samp - q_s) + np.linalg.norm(q_g - q_samp)) <= c_best + 1e-7

    # Run Informed RRT* in 6-DoF UR5e joint space
    ir = informed_rrt_star(q_s, q_g, rng, arm=DEFAULT_UR5E, scene_3d=DEFAULT_SCENE_3D, iters=180)
    assert ir is not None
    assert ir.best_cost_rad <= ir.initial_cost_rad + 1e-9
    assert ir.ellipsoid_samples > 50
    assert path_min_clearance_se3(ir.path, DEFAULT_UR5E, DEFAULT_SCENE_3D) >= 0.005

    # Run TOPP-RA and verify velocity, acceleration, AND jerk bounds
    tp = parameterize_toppra(
        ir.path,
        qd_max=DEFAULT_UR5E.qd_max,
        qdd_max=DEFAULT_UR5E.qdd_max,
        qddd_max=DEFAULT_UR5E.qddd_max,
        edge_free_fn=lambda a, b: edge_free_se3(a, b, DEFAULT_UR5E, DEFAULT_SCENE_3D),
    )
    assert tp.duration_s > 0.0
    assert np.all(tp.max_joint_vel <= DEFAULT_UR5E.qd_max + 1e-6)
    assert np.all(tp.max_joint_acc <= DEFAULT_UR5E.qdd_max + 1e-6)
    assert np.all(tp.max_joint_jerk <= DEFAULT_UR5E.qddd_max + 1e-6)
    assert np.allclose(tp.qd[0], 0.0, atol=1e-9) and np.allclose(tp.qd[-1], 0.0, atol=1e-9)
    assert np.allclose(tp.qdd[0], 0.0, atol=1e-9) and np.allclose(tp.qdd[-1], 0.0, atol=1e-9)
    assert np.allclose(tp.q[0], q_s, atol=1e-9) and np.allclose(tp.q[-1], q_g, atol=1e-9)


# ---------------------------------------------------------------------------
# Upgrade 2: Nonlinear Bessel Slosh Dynamics & ZV / ZVD / EI Input Shaping
# ---------------------------------------------------------------------------


def test_bessel_dispersion_and_viscous_stokes_damping_laws() -> None:
    assert round(BESSEL_XI_11, 4) == 1.8412
    R96 = LABWARE_SLOSH_SPECS["96_well"].well_radius_m
    R384 = LABWARE_SLOSH_SPECS["384_well"].well_radius_m
    h = 0.006

    w96 = compute_bessel_slosh_omega(R96, h)
    expected = math.sqrt((9.80665 * BESSEL_XI_11 / R96) * math.tanh(BESSEL_XI_11 * h / R96))
    assert w96 == pytest.approx(expected, rel=1e-12)

    z_water_96, ga_w, re_w = compute_viscous_slosh_damping(
        LIQUID_CATALOG["water"].kinematic_viscosity_m2_s, R96, h
    )
    z_pbs_96, _, _ = compute_viscous_slosh_damping(
        LIQUID_CATALOG["pbs_surfactant"].kinematic_viscosity_m2_s,
        R96,
        h,
        surface_film_factor=LIQUID_CATALOG["pbs_surfactant"].surface_film_factor,
    )
    z_gly_96, ga_g, re_g = compute_viscous_slosh_damping(
        LIQUID_CATALOG["glycerol_50pct"].kinematic_viscosity_m2_s, R96, h
    )
    z_water_384, _, _ = compute_viscous_slosh_damping(
        LIQUID_CATALOG["water"].kinematic_viscosity_m2_s, R384, h
    )

    assert 0.015 < z_water_96 < z_pbs_96 < z_gly_96 < 0.25
    assert z_water_384 > z_water_96  # narrower 384-well bore increases Stokes boundary-layer damping
    assert ga_w > ga_g and re_w > re_g


def test_zv_zvd_ei_input_shapers_and_mismatch_insensitivity() -> None:
    wn = 74.0
    zeta = 0.032
    zv = design_input_shaper("ZV", wn, zeta)
    zvd = design_input_shaper("ZVD", wn, zeta)
    ei = design_input_shaper("EI", wn, zeta, v_tol=0.05)

    # Exact impulse normalization & positivity
    for sh in (zv, zvd, ei):
        assert np.all(sh.amplitudes > 0.0)
        assert float(np.sum(sh.amplitudes)) == pytest.approx(1.0, abs=1e-12)

    # Nominal frequency properties
    assert residual_vibration_ratio(zv.times_s, zv.amplitudes, wn, zeta) < 1e-12
    assert residual_vibration_ratio(zvd.times_s, zvd.amplitudes, wn, zeta) < 1e-12
    assert residual_vibration_ratio(ei.times_s, ei.amplitudes, wn, zeta) == pytest.approx(0.05, abs=1e-6)

    # At +-15% natural frequency mismatch, EI beats ZVD and both crush ZV!
    for scale in (0.85, 1.15):
        v_zv = residual_vibration_ratio(zv.times_s, zv.amplitudes, wn * scale, zeta)
        v_zvd = residual_vibration_ratio(zvd.times_s, zvd.amplitudes, wn * scale, zeta)
        v_ei = residual_vibration_ratio(ei.times_s, ei.amplitudes, wn * scale, zeta)
        assert v_ei < v_zvd < v_zv
        assert v_ei <= 0.05 + 1e-4


def test_factorial_slosh_benchmark_zvd_ei_outperform_under_pm25pct_mismatch() -> None:
    report = run_factorial_slosh_benchmark()
    nom = report["summary_by_regime"]["nominal"]
    mis = report["summary_by_regime"]["mismatch_pooled_pm25pct"]

    # Nominal high-speed indexing: ZVD & EI cut spills vs ZV, min-jerk, and unshaped TOPP-RA
    assert nom["toppra_unshaped"]["spills"] >= nom["min_jerk_unshaped"]["spills"] > 0
    assert nom["toppra_ZVD"]["spills"] < nom["toppra_ZV"]["spills"]
    assert nom["toppra_EI"]["spills"] < nom["min_jerk_unshaped"]["spills"]
    assert nom["toppra_EI"]["mean_residual_ringdown_mm"] < 0.1 * nom["min_jerk_unshaped"]["mean_residual_ringdown_mm"]

    # Under +-25% model mismatch, ZVD and EI beat ZV and unshaped minimum-jerk on spill rate,
    # peak meniscus ratio eta_max / h_free, mean meniscus ratio, and post-move ringdown!
    assert mis["toppra_EI"]["spills"] <= mis["toppra_ZVD"]["spills"] < mis["toppra_ZV"]["spills"]
    assert mis["toppra_EI"]["spills"] < mis["min_jerk_unshaped"]["spills"] < mis["toppra_unshaped"]["spills"]
    assert (
        mis["toppra_EI"]["max_eta_over_h_free"]
        < mis["toppra_ZVD"]["max_eta_over_h_free"]
        < mis["min_jerk_unshaped"]["max_eta_over_h_free"]
        < mis["toppra_ZV"]["max_eta_over_h_free"]
        < mis["toppra_unshaped"]["max_eta_over_h_free"]
    )
    assert mis["toppra_EI"]["mean_eta_over_h_free"] < mis["toppra_ZV"]["mean_eta_over_h_free"]
    assert mis["toppra_EI"]["mean_eta_over_h_free"] < mis["min_jerk_unshaped"]["mean_eta_over_h_free"]
    assert mis["toppra_EI"]["mean_residual_ringdown_mm"] < 0.15 * mis["min_jerk_unshaped"]["mean_residual_ringdown_mm"]


# ---------------------------------------------------------------------------
# Upgrade 3: 3D Whitney/Coulomb Contact & Active Spiral + Bayesian Tactile
# ---------------------------------------------------------------------------


def test_realistic_chamfer_fair_open_loop_and_flat_rim_spiral_recovery() -> None:
    vessel = ClearVesselBevel()
    assert vessel.chamfer_width_m == pytest.approx(0.00070)
    assert vessel.capture_radius_m == pytest.approx(0.00080)  # 0.80 mm, NOT 5.5 mm!

    center = np.asarray(vessel.center_xy, dtype=np.float64)

    # 1. Fair physics: tiny offset (0.14 mm > bore_radius=0.10 mm) slides down chamfer even in open-loop!
    ol_inner = run_open_loop_insertion(center + np.array([0.00014, 0.0]), vessel=vessel, friction_coeff=0.18)
    assert ol_inner.success is True

    # 2. Flat rim outside narrow 0.70 mm chamfer (2.6 mm refractive offset, 2.0 deg tilt, mu=0.35):
    # open_loop, passive_rcc, and naive local_shear_admittance ALL fail on the flat rim,
    # whereas active_spiral_bayesian_tactile searches into the chamfer, aligns tilt, and inserts!
    vis_flat = center + np.array([0.0019, -0.0018])  # ~2.62 mm > 0.80 mm capture radius
    tilt_2deg = (math.radians(1.5), math.radians(-1.3))

    r_ol = run_open_loop_insertion(vis_flat, vessel=vessel, tilt_xy_rad=tilt_2deg, friction_coeff=0.35)
    r_rcc = run_passive_rcc_insertion(vis_flat, vessel=vessel, tilt_xy_rad=tilt_2deg, friction_coeff=0.35)
    r_loc = run_local_shear_insertion(
        vis_flat, np.random.default_rng(0), vessel=vessel, tilt_xy_rad=tilt_2deg, friction_coeff=0.35
    )
    r_act = run_active_tactile_insertion(
        vis_flat, np.random.default_rng(0), vessel=vessel, tilt_xy_rad=tilt_2deg, friction_coeff=0.35
    )

    assert r_ol.success is False and r_ol.rim_hangup is True
    assert r_rcc.success is False and r_rcc.rim_hangup is True
    assert r_loc.success is False and r_loc.rim_hangup is True
    assert r_act.success is True
    assert r_act.spiral_probes > 5
    assert r_act.final_error_mm < 0.15
    assert r_act.final_tilt_deg < math.degrees(math.hypot(*tilt_2deg))

    # Also verify active tactile insertion runs natively on 6-DoF UR5eArm!
    r_ur5e = run_active_tactile_insertion(
        vis_flat,
        np.random.default_rng(1),
        vessel=vessel,
        arm=DEFAULT_UR5E,
        tilt_xy_rad=tilt_2deg,
        friction_coeff=0.30,
    )
    assert r_ur5e.success is True
    assert r_ur5e.final_error_mm < 0.15


def test_whitney_wedging_and_4_regime_benchmark_hierarchy() -> None:
    # Direct Whitney (1982) wedging check: c = 0.05 mm, R = 3.4 mm, mu = 0.45 -> theta_wedge = 0.0327 rad (1.87 deg)
    wedged, jammed, _ = check_whitney_jamming_wedging(
        insertion_depth_m=0.0025,
        bore_clearance_m=0.00005,
        hole_radius_m=0.0034,
        friction_coeff=0.45,
        tilt_rad=math.radians(2.4),
        f_lat_n=0.4,
        f_z_n=0.5,
        moment_nm=0.01,
    )
    assert wedged is True and jammed is True

    rep = benchmark_tactile_regimes(n_trials=120, seed=7)
    regs = rep["regimes"]
    assert regs["open_loop"]["success_rate_pct"] < regs["local_shear_admittance"]["success_rate_pct"]
    assert regs["passive_rcc_compliance"]["success_rate_pct"] < regs["local_shear_admittance"]["success_rate_pct"]
    assert regs["local_shear_admittance"]["success_rate_pct"] < 30.0  # bounded by ~20% chamfer area ratio
    assert regs["active_spiral_bayesian_tactile"]["success_rate_pct"] >= 92.0


# ---------------------------------------------------------------------------
# Upgrade 4: Canonical LeRobot & Opentrons Data Interface Shapes
# ---------------------------------------------------------------------------


def test_canonical_lerobot_and_opentrons_tensor_shapes_and_guards() -> None:
    # 1. Plan 3D pick & place on 6-DoF UR5eArm -> Opentrons -> LeRobot (T, 7)
    plan = plan_3d_pick_place("A1", "H12", np.random.default_rng(0), arm_se3=DEFAULT_UR5E)
    assert plan is not None and plan.q6_waypoints is not None

    ot_traj = plan_to_opentrons_trajectory(plan, aspirate_volume_ul=150.0, samples_per_segment=8)
    ot_dict = ot_traj.to_dict()
    validate_opentrons_trajectory(ot_dict)

    T = ot_traj.n_frames
    assert ot_traj.deck_tip_mm.shape == (T, 3) and ot_traj.deck_tip_mm.dtype == np.float64
    assert ot_traj.base_tip_m.shape == (T, 3) and ot_traj.base_tip_m.dtype == np.float64
    assert ot_traj.base_pose_se3.shape == (T, 6) and ot_traj.base_pose_se3.dtype == np.float64
    assert ot_traj.plunger_ul.shape == (T,) and ot_traj.plunger_ul.dtype == np.float64
    assert np.all(ot_traj.deck_tip_mm[:, 0] >= 0.0) and np.all(ot_traj.deck_tip_mm[:, 0] <= 418.0)
    assert np.all(ot_traj.deck_tip_mm[:, 1] >= 0.0) and np.all(ot_traj.deck_tip_mm[:, 1] <= 353.0)
    assert np.all(ot_traj.deck_tip_mm[:, 2] >= 0.0) and np.all(ot_traj.deck_tip_mm[:, 2] <= 218.0)

    lr_table = ot_traj.to_lerobot_ur5e(episode_index=5)
    lr_dict = lr_table.to_dict()
    validate_lerobot_frame_tensors(lr_dict)

    assert lr_dict["observation.state"].shape == (T, 7) and lr_dict["observation.state"].dtype == np.float32
    assert lr_dict["action"].shape == (T, 7) and lr_dict["action"].dtype == np.float32
    assert lr_dict["timestamp"].shape == (T,) and lr_dict["timestamp"].dtype == np.float32
    assert lr_dict["frame_index"].shape == (T,) and lr_dict["frame_index"].dtype == np.int64
    assert lr_dict["episode_index"].shape == (T,) and lr_dict["episode_index"].dtype == np.int64
    assert lr_dict["next.done"].shape == (T,) and lr_dict["next.done"].dtype == np.bool_
    assert bool(lr_dict["next.done"][-1]) is True and not bool(np.any(lr_dict["next.done"][:-1]))

    # Negative guards: wrong dtypes or out-of-deck coordinates must fail closed!
    bad_lr = dict(lr_dict)
    bad_lr["observation.state"] = bad_lr["observation.state"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        validate_lerobot_frame_tensors(bad_lr)

    bad_ot = dict(ot_dict)
    bad_ot["deck_tip_mm"] = bad_ot["deck_tip_mm"].copy()
    bad_ot["deck_tip_mm"][0, 0] = 425.0  # > 418 mm deck X limit
    with pytest.raises(ValueError, match="Opentrons deck envelope"):
        validate_opentrons_trajectory(bad_ot)


def test_committed_frontier_benchmark_summary_artifact() -> None:
    summary_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "frontier_benchmark_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["schema"] == "lab-hands/frontier-benchmarks/v1"
    assert "ur5e" in data["kinematics_planning_toppra"]
    assert "panda" in data["kinematics_planning_toppra"]
    assert data["tactile_coulomb_whitney_500"]["n_trials"] == 500


def test_live_opentrons_simulate_8_8_runlog_shapes() -> None:
    sim = pytest.importorskip("opentrons.simulate")
    proto = """
metadata = {'apiLevel': '2.19'}
def run(protocol):
    tips = protocol.load_labware('opentrons_96_tiprack_300ul', '1')
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', '5')
    p300 = protocol.load_instrument('p300_single_gen2', 'right', tip_racks=[tips])
    p300.pick_up_tip(tips['A1'])
    p300.aspirate(125, plate['A1'])
    p300.dispense(125, plate['H12'])
    p300.drop_tip(tips['A1'])
"""
    runlog, _ = sim.simulate(io.StringIO(proto))
    ot_traj = extract_opentrons_runlog_trajectory(runlog)
    validate_opentrons_trajectory(ot_traj.to_dict())
    assert ot_traj.deck_tip_mm.shape[1] == 3 and ot_traj.deck_tip_mm.dtype == np.float64
    assert ot_traj.base_pose_se3.shape == (ot_traj.n_frames, 6)
    assert float(np.max(ot_traj.plunger_ul)) == pytest.approx(125.0)
    lr = ot_traj.to_lerobot_ur5e(episode_index=0).to_dict()
    validate_lerobot_frame_tensors(lr)
    assert lr["observation.state"].shape == (ot_traj.n_frames, 7)
