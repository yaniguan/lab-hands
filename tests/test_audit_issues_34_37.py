"""Acceptance-criteria regression tests for open issues #34, #35, #36, and #37."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from autolab_wire import HALT_REASON, process_packet
from community_extensions import OpenWellPlateSpec, compute_fundamental_slosh_omega, plan_slosh_bounded_min_jerk, run_benchmark
from lab_hands import (
    DEFAULT_FRANKA_MODEL,
    DEFAULT_UR5E_MODEL,
    MIN_GRAVITY_SLOSH_BOND_NUMBER,
    OPENTRONS_LABWARE_V1,
    OPENTRONS_LABWARE_V1_BUNDLE,
    SLOSH_BESSEL_XI_11,
    SLOSH_GRAVITY_M_S2,
    STEPBOT_LABWARE_CATALOG,
    BenchScene,
    DeckLayout,
    audit_lerobot_episode,
    audit_se3_trajectory,
    get_plate_geometry,
    load_lerobot_episode,
    plan_3d_pick_place,
    slosh_bond_number,
    slosh_natural_frequency,
)
from scripts.sync_opentrons_labware import LOAD_NAMES_24, extract_labware_entry

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Issue #34: Official Opentrons labware Z heights, well bottoms & tip lengths
# ---------------------------------------------------------------------------


def test_catalog_matches_opentrons_labware_v1_json_offline_issue_34() -> None:
    """Offline check (no network, no extra package): all 24 catalog entries match opentrons_labware_v1.json."""
    assert OPENTRONS_LABWARE_V1_BUNDLE["source_package"] == "opentrons-shared-data"
    assert OPENTRONS_LABWARE_V1_BUNDLE["spdx_license"] == "Apache-2.0"
    assert "Apache License" in OPENTRONS_LABWARE_V1_BUNDLE["license_notice"]
    assert len(STEPBOT_LABWARE_CATALOG) == 24
    assert set(STEPBOT_LABWARE_CATALOG.keys()) == set(OPENTRONS_LABWARE_V1.keys())

    for load_name, geom in STEPBOT_LABWARE_CATALOG.items():
        ref = OPENTRONS_LABWARE_V1[load_name]
        assert geom.z_rim_m * 1000.0 == pytest.approx(ref["zDimension_mm"], abs=0.01)
        assert geom.z_bottom_m * 1000.0 == pytest.approx(ref["a1"]["z_mm"], abs=0.01)
        assert geom.well_radius_m * 1000.0 == pytest.approx(ref["well_radius_mm"], abs=0.01)
        assert geom.is_tiprack is bool(ref["isTiprack"])
        if geom.is_tiprack:
            expected_tip_mm = float(ref["tipLength_mm"]) - float(ref["tipOverlap_mm"])
            assert geom.default_tip_length_m() * 1000.0 == pytest.approx(expected_tip_mm, abs=0.01)


def test_opentrons_labware_v1_json_matches_installed_opentrons_shared_data_issue_34() -> None:
    """Compare opentrons_labware_v1.json with installed opentrons-shared-data for all 24 load names.

    NOTE: Do NOT skip this test — CI installs opentrons-shared-data so this always runs.
    """
    import opentrons_shared_data.labware as ot_lw  # type: ignore[import-not-found]

    assert len(LOAD_NAMES_24) == 24
    for load_name in LOAD_NAMES_24:
        defn = ot_lw.load_definition(load_name, 1)
        live_entry = extract_labware_entry(defn)
        cached_entry = OPENTRONS_LABWARE_V1[load_name]
        assert cached_entry["zDimension_mm"] == pytest.approx(live_entry["zDimension_mm"], abs=0.01)
        assert cached_entry["a1"]["z_mm"] == pytest.approx(live_entry["a1"]["z_mm"], abs=0.01)
        assert cached_entry["a1"]["depth_mm"] == pytest.approx(live_entry["a1"]["depth_mm"], abs=0.01)
        assert cached_entry["well_radius_mm"] == pytest.approx(live_entry["well_radius_mm"], abs=0.01)
        if live_entry["isTiprack"]:
            assert cached_entry["tipLength_mm"] == pytest.approx(live_entry["tipLength_mm"], abs=0.01)
            assert cached_entry["tipOverlap_mm"] == pytest.approx(live_entry["tipOverlap_mm"], abs=0.01)
            assert cached_entry["effectiveTipLength_mm"] == pytest.approx(
                live_entry["effectiveTipLength_mm"], abs=0.01
            )


def test_1000ul_tiprack_safe_travel_and_eppendorf_bottom_height_issue_34() -> None:
    # 1. With 1000 uL tiprack on deck and no tip mounted, z_safe_travel > 97.47 mm
    tip1000 = get_plate_geometry("opentrons_96_tiprack_1000ul")
    assert tip1000.z_rim_m * 1000.0 == pytest.approx(97.47, abs=0.01)
    deck_tip = DeckLayout()
    deck_tip.add_labware("tips1000", "opentrons_96_tiprack_1000ul", slot=4)
    scene_no_tip = BenchScene(deck=deck_tip, attached_tip_length_m=0.0)
    assert float(scene_no_tip.z_safe_travel) * 1000.0 > 97.47  # type: ignore[arg-type]

    # 2. With Eppendorf 1.5 mL rack on deck (alias `epp`), lowest planned tip Z >= 42.05 mm
    deck_epp = DeckLayout()
    deck_epp.add_labware("epp", "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap", slot=5)
    scene_epp = BenchScene(deck=deck_epp, attached_tip_length_m=0.05183)
    plan_epp = plan_3d_pick_place(
        "TUBE",
        "epp:A1",
        np.random.default_rng(0),
        scene=scene_epp,
        deck=deck_epp,
        attached_tip_length_m=0.05183,
    )
    assert plan_epp is not None
    # Check tip Z at destination epp:A1 (and all waypoints over epp): z_tip = z_flange - attached_tip_length_m
    tip_z_at_epp_mm = (plan_epp.poses_se3[-1].tip_position(attached_tip_length_m=plan_epp.attached_tip_length_m)[2]) * 1000.0
    assert tip_z_at_epp_mm >= 42.05


# ---------------------------------------------------------------------------
# Issue #35: Wire gate requested-clearance planning & numeric input validation
# ---------------------------------------------------------------------------


def test_wire_gate_clearance_sweep_and_field_validation_issue_35() -> None:
    for req in (5.0, 8.5, 25.0, 50.0, 500.0):
        out = process_packet(
            {
                "start_well": "TUBE",
                "target_well": "H12",
                "clearance_mm": req,
                "seed": 0,
                "require_tactile_insert": False,
            }
        )
        if not out["halted"]:
            assert out["manipulation"]["measured_min_clearance_mm"] >= out["manipulation"]["requested_min_clearance_mm"]
        else:
            assert out["halt_reason"] == HALT_REASON


# ---------------------------------------------------------------------------
# Issue #36: Published UR5e & Franka DH models + real LeRobot episode audits
# ---------------------------------------------------------------------------


def test_ur5e_and_franka_hand_computed_dh_vectors_issue_36() -> None:
    # UR5e hand-computed poses from Universal Robots published Standard DH table:
    # d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996], a = [0, -0.425, -0.3922, 0, 0, 0]
    p_ur_zero = DEFAULT_UR5E_MODEL.fk_se3([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(p_ur_zero.xyz, [-0.8172, -0.2329, 0.0628], atol=1e-6)

    p_ur_j1_90 = DEFAULT_UR5E_MODEL.fk_se3([math.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(p_ur_j1_90.xyz, [0.2329, -0.8172, 0.0628], atol=1e-6)

    p_ur_up = DEFAULT_UR5E_MODEL.fk_se3([0.0, -math.pi / 2.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(p_ur_up.xyz, [-0.0997, -0.2329, 0.9797], atol=1e-6)

    # Franka Emika Panda hand-computed poses from published Craig Modified DH table:
    # d1=0.333, d3=0.316, d5=0.384, a4=0.0825, a5=-0.0825, a7=0.088, d_flange=0.107
    p_fr_zero = DEFAULT_FRANKA_MODEL.fk_se3([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(p_fr_zero.xyz, [0.088, 0.0, 0.926], atol=1e-6)

    p_fr_j1_90 = DEFAULT_FRANKA_MODEL.fk_se3([math.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(p_fr_j1_90.xyz, [0.0, 0.088, 0.926], atol=1e-6)

    p_fr_ready = DEFAULT_FRANKA_MODEL.fk_se3([0.0, 0.0, 0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, 0.0])
    assert np.allclose(p_fr_ready.xyz, [0.5545, 0.0, 0.6245], atol=1e-6)


def test_real_autobio_ur5e_and_droid_franka_fixtures_and_unknown_robot_issue_36() -> None:
    # 1. Real AutoBio pipette-mujoco episode 0 (100 frames, MIT)
    autobio_path = FIXTURES_DIR / "autobio_pipette_mujoco_ep0_100.json"
    ep_ab = load_lerobot_episode(autobio_path, dataset_id="autobio_pipette_mujoco")
    assert ep_ab.robot_type == "ur5e"
    assert ep_ab.n_arm_joints == 6
    assert ep_ab.arm_joints.shape == (100, 6)

    # Check UR5e flange stays above robot base plane (z > 0; ~+0.480 m at frame 0)
    poses_ab = [DEFAULT_UR5E_MODEL.fk_se3(ep_ab.arm_joints[i]) for i in range(ep_ab.n_frames)]
    assert poses_ab[0].xyz[2] == pytest.approx(0.4801, abs=1e-3)
    assert math.hypot(poses_ab[0].xyz[0], poses_ab[0].xyz[1]) == pytest.approx(0.5313, abs=1e-3)
    assert all(p.xyz[2] > 0.0 for p in poses_ab)

    # Without base_pose_in_deck, obstacle & rim checks are "not_checked", NOT failures!
    audit_ab = audit_lerobot_episode(ep_ab)
    assert audit_ab.robot_type == "ur5e"
    assert audit_ab.n_arm_joints == 6
    assert audit_ab.flange_above_base_ok is True
    assert audit_ab.min_flange_z_m > 0.40
    assert audit_ab.obstacle_check_status == "not_checked"
    assert audit_ab.rim_check_status == "not_checked"
    assert audit_ab.obstacle_clearance_ok is None
    assert audit_ab.rim_clearance_ok is None
    assert "OBSTACLE_CLEARANCE_BELOW_5MM" not in audit_ab.violations
    assert "WELL_RIM_COLLISION" not in audit_ab.violations
    assert audit_ab.safe is True

    # 2. Real DROID 1.0.1 7-joint Franka episode 0 (15 frames, CC-BY-4.0)
    droid_path = FIXTURES_DIR / "droid_franka_ep0_15.json"
    ep_dr = load_lerobot_episode(droid_path, dataset_id="droid")
    assert ep_dr.robot_type == "Franka"
    assert ep_dr.n_arm_joints == 7
    assert ep_dr.arm_joints.shape == (15, 7)
    assert ep_dr.cartesian_positions is not None and ep_dr.cartesian_positions.shape == (15, 6)

    # Published Franka MDH FK agrees with recorded observation.state.cartesian_position within < 5 mm!
    for i in range(ep_dr.n_frames):
        fk_xyz = np.asarray(DEFAULT_FRANKA_MODEL.fk_se3(ep_dr.arm_joints[i]).xyz, dtype=np.float64)
        rec_xyz = ep_dr.cartesian_positions[i, :3]
        assert float(np.linalg.norm(fk_xyz - rec_xyz)) * 1000.0 < 5.0

    audit_dr = audit_lerobot_episode(ep_dr)
    assert audit_dr.robot_type == "franka"
    assert audit_dr.n_arm_joints == 7
    assert audit_dr.obstacle_check_status == "not_checked"
    assert audit_dr.rim_check_status == "not_checked"
    assert audit_dr.safe is True

    # 3. Unknown robot_type gets ROBOT_MODEL_UNKNOWN and safe is None (never safe: True or safe: False)
    ep_unknown = load_lerobot_episode(
        [{"episode_index": 0, "frame_index": 0, "timestamp": 0.0, "observation.state": [0.0] * 6}],
        robot_type="so_arm100_unmodeled",
    )
    audit_unk = audit_lerobot_episode(ep_unknown)
    assert "ROBOT_MODEL_UNKNOWN" in audit_unk.violations
    assert audit_unk.safe is None
    assert audit_unk.safe is not True
    assert audit_unk.safe is not False


# ---------------------------------------------------------------------------
# Issue #37: Unified Bessel dispersion, Bond number flag, and honest wording
# ---------------------------------------------------------------------------


def test_unified_slosh_dispersion_bond_number_and_repo_wording_issue_37() -> None:
    # 1. Exact formula sqrt((xi*g/R) * tanh(xi*h/R)) at 0.5, 2.0, and 9.2 mm fill (R = 3.45 mm)
    r_m = 0.00345
    for fill_mm in (0.5, 2.0, 9.2):
        h_m = fill_mm / 1000.0
        ref = math.sqrt((SLOSH_BESSEL_XI_11 * SLOSH_GRAVITY_M_S2 / r_m) * math.tanh(SLOSH_BESSEL_XI_11 * h_m / r_m))
        assert slosh_natural_frequency(r_m, h_m) == pytest.approx(ref, rel=1e-9)
        assert compute_fundamental_slosh_omega(r_m, h_m) == pytest.approx(ref, rel=1e-9)
    assert slosh_natural_frequency(r_m, 0.0005) == pytest.approx(36.9383, abs=0.1)

    # 2. Single canonical implementation: datasets.py and slosh_minimum_jerk.py do not copy sqrt(...)*tanh(...)
    for rel_path in ("lab_hands/datasets.py", "community_extensions/slosh_minimum_jerk.py"):
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "slosh_natural_frequency(" in src
        assert "np.tanh(" not in src and "math.tanh(" not in src

    # 3. 96-well Bond number (~1.62 < 5.0) is included and marked SLOSH_MODEL_OUT_OF_RANGE
    bo_96 = slosh_bond_number(0.00345)
    assert bo_96 < MIN_GRAVITY_SLOSH_BOND_NUMBER
    mj = plan_slosh_bounded_min_jerk((-0.05, 0.40), (0.24, 0.02), OpenWellPlateSpec())
    assert mj.bond_number == pytest.approx(bo_96, abs=1e-3)
    assert mj.slosh_model_status == "SLOSH_MODEL_OUT_OF_RANGE"

    bench60 = run_benchmark()
    assert bench60["bond_number"] == pytest.approx(bo_96, abs=1e-3)
    assert bench60["slosh_model_status"] == "SLOSH_MODEL_OUT_OF_RANGE"

    q6_zero = np.zeros((5, 6), dtype=np.float64)
    aud = audit_se3_trajectory(q6_zero, np.linspace(0.0, 1.0, 5))
    assert aud.bond_number == pytest.approx(bo_96, abs=1e-3)
    assert aud.slosh_model_status == "SLOSH_MODEL_OUT_OF_RANGE"

    # 4. No documentation or Lean file claims the model or Lean file "guarantees zero spills"
    banned_phrases = (
        "guarantees zero well spills",
        "guaranteeing zero well spills",
        "guaranteeing zero well-to-well spills",
        "guarantees zero spills",
        "zero-spill corner-well",
    )
    for doc_rel in ("README.md", "docs/ARCHITECTURE_AND_HISTORY.md", "lean/README.md", "lean/LabHandsLean.lean"):
        txt = (REPO_ROOT / doc_rel).read_text(encoding="utf-8").lower()
        for phrase in banned_phrases:
            assert phrase not in txt, f"Found banned overclaim {phrase!r} in {doc_rel}"
