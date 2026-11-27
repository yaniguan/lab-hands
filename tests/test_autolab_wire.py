"""Tests for `autolab_wire.py` (`autolab-wire/v1` Layer L1 `lab-hands` wire node)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import autolab_wire  # noqa: E402
from autolab_wire import LAYER, REPO_DIR, REPO_ID, process_packet  # noqa: E402
from lab_hands import DEFAULT_ARM, DEFAULT_SCENE, edge_free, path_min_clearance  # noqa: E402


def test_wire_metadata_and_cwd_independence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert REPO_ID == "lab-hands"
    assert LAYER == "L1"
    assert REPO_DIR == _ROOT.resolve()

    monkeypatch.chdir(tmp_path)
    out = process_packet({})
    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["manipulation"]["collision_free"] is True


def test_collision_free_rrt_a1_to_h12() -> None:
    packet = {
        "start_well": "A1",
        "target_well": "H12",
        "clearance_mm": 8.5,
    }
    original = copy.deepcopy(packet)

    out1 = process_packet(packet)
    out2 = process_packet(packet)

    # Input immutability and deterministic seeded planning
    assert packet == original
    assert out1 == out2
    assert json.loads(json.dumps(out1)) == out1

    assert out1["schema_version"] == "autolab-wire/v1"
    assert out1["halted"] is False
    assert out1["halt_reason"] is None

    manip = out1["manipulation"]
    assert manip["start_well"] == "A1"
    assert manip["target_well"] == "H12"
    assert manip["collision_free"] is True
    assert isinstance(manip["waypoints"], int) and manip["waypoints"] >= 2
    assert manip["path_length_rad"] > 0.0
    assert manip["requested_min_clearance_mm"] == 8.5
    assert manip["measured_min_clearance_mm"] >= 5.0
    assert manip["clearance_mm"] == manip["measured_min_clearance_mm"]
    assert len(manip["start_q"]) == 2
    assert len(manip["end_q"]) == 2
    assert manip["elbow_branch"] in (1, -1)
    assert manip["tactile"]["inserted"] is True
    assert manip["tactile"]["seal_intact"] is True
    assert manip["tactile"]["slip_detected"] is False
    assert manip["tactile"]["attribution_hint"] == "NOMINAL"

    assert len(out1["trace"]) == 1
    assert out1["trace"][-1]["repo"] == "lab-hands"
    assert out1["trace"][-1]["layer"] == "L1"
    assert out1["trace"][-1]["status"] == "PASS"

    assert len(out1["bottom_reasoning_traces"]) == 1
    brt = out1["bottom_reasoning_traces"][-1]
    assert brt["repo"] == "lab-hands"
    assert brt["invariant"] == "arm_path_collision_free_with_5mm_margin"
    assert "5.0 mm" in brt["first_principles"]


def test_requested_clearance_enforced_and_measured_truthfully_issue_30_35() -> None:
    # Check requests of 5, 8.5, 25, 50, and 500 mm for TUBE -> H12, seed 0 (#35 acceptance criteria)
    for req_mm in (5.0, 8.5, 25.0, 50.0, 500.0):
        pkt = {
            "start_well": "TUBE",
            "target_well": "H12",
            "clearance_mm": req_mm,
            "seed": 0,
            "require_tactile_insert": False,
        }
        out = process_packet(pkt)
        manip = out["manipulation"]
        assert manip["requested_min_clearance_mm"] == req_mm
        if not out["halted"]:
            assert out["trace"][-1]["status"] == "PASS"
            assert manip["measured_min_clearance_mm"] >= manip["requested_min_clearance_mm"]
            assert manip["clearance_mm"] == manip["measured_min_clearance_mm"]
        else:
            assert out["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"
            assert out["trace"][-1]["status"] == "BLOCK"
            assert "requested_clearance_mm=" in out["trace"][-1]["summary"]
            assert "best_measured_clearance_mm=" in out["trace"][-1]["summary"]

    # Default 8.5 mm request for TUBE -> H12, seed 0 passes with >= 8.5 mm measured clearance
    out_85 = process_packet(
        {"start_well": "TUBE", "target_well": "H12", "clearance_mm": 8.5, "seed": 0, "require_tactile_insert": False}
    )
    assert out_85["halted"] is False
    assert out_85["manipulation"]["measured_min_clearance_mm"] >= 8.5

    # Impossible 500 mm request halts cleanly with HALT_REASON
    out_500 = process_packet(
        {"start_well": "TUBE", "target_well": "H12", "clearance_mm": 500.0, "seed": 0, "require_tactile_insert": False}
    )
    assert out_500["halted"] is True
    assert out_500["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"


@pytest.mark.parametrize("field_name", ["seed", "clearance_mm", "capsule_radius_m", "bevel_capture_radius_m"])
@pytest.mark.parametrize("bad_value", ["abc", None, float("nan"), [1]])
def test_bad_numeric_packet_fields_halt_without_raising_issue_35(field_name: str, bad_value: object) -> None:
    out = process_packet({"start_well": "A1", "target_well": "H12", field_name: bad_value})
    assert out["halted"] is True
    assert out["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"
    assert field_name in out["trace"][-1]["summary"]


def test_tactile_bevel_recovery_and_rim_hangup_seal_block_issue_30() -> None:
    # Within 5.50 mm bevel capture radius (~3.20 mm error) -> compliant centering succeeds
    ok_pkt = {
        "start_well": "A1",
        "target_well": "H12",
        "visual_offset_mm": [2.5, -2.0],
        "seed": 0,
    }
    ok_out = process_packet(ok_pkt)
    assert ok_out["halted"] is False
    assert ok_out["manipulation"]["tactile"]["inserted"] is True
    assert ok_out["manipulation"]["tactile"]["seal_intact"] is True
    assert ok_out["manipulation"]["tactile"]["slip_detected"] is False
    assert ok_out["manipulation"]["tactile"]["final_error_mm"] < 0.2
    assert ok_out["manipulation"]["tactile"]["attribution_hint"] == "NOMINAL"

    # Outside 5.50 mm bevel capture radius (5.80 mm error) -> flat-rim hangup / seal block
    bad_pkt = {
        "start_well": "A1",
        "target_well": "H12",
        "visual_offset_mm": [4.2, 4.0],
        "seed": 0,
    }
    bad_out = process_packet(bad_pkt)
    assert bad_out["halted"] is True
    assert bad_out["halt_reason"] == "L1_HANDS_TACTILE_INSERTION_OR_SEAL_BLOCK"
    tac = bad_out["manipulation"]["tactile"]
    assert tac["inserted"] is False
    assert tac["seal_intact"] is False
    assert tac["slip_detected"] is True
    assert tac["axial_slip_mm"] > 5.0
    assert tac["peak_force_n"] >= 0.95
    assert tac["execution_artifact"] is True
    assert tac["attribution_hint"] == "EXECUTION_ARTIFACT"


def test_stateful_ik_continuity_across_chained_packets_issue_27() -> None:
    leg1 = process_packet({"start_well": "TUBE", "target_well": "H12", "seed": 3})
    assert leg1["halted"] is False
    end_q1 = np.asarray(leg1["manipulation"]["end_q"], dtype=np.float64)

    # Explicit start_q chaining
    leg2 = process_packet(
        {
            "start_well": "H12",
            "target_well": "G12",
            "start_q": leg1["manipulation"]["end_q"],
            "seed": 3,
        }
    )
    assert leg2["halted"] is False
    start_q2 = np.asarray(leg2["manipulation"]["start_q"], dtype=np.float64)
    end_q2 = np.asarray(leg2["manipulation"]["end_q"], dtype=np.float64)
    assert np.allclose(start_q2, end_q1, atol=1e-6)
    assert float(np.linalg.norm(end_q2 - start_q2)) < 0.20
    assert edge_free(end_q1, start_q2)
    assert leg2["manipulation"]["elbow_branch"] == leg1["manipulation"]["elbow_branch"]

    # Automatic packet-forwarding chaining via out["manipulation"]["end_q"]
    chained = dict(leg2)
    chained["start_well"] = "G12"
    chained["target_well"] = "F12"
    leg3 = process_packet(chained)
    assert leg3["halted"] is False
    assert np.allclose(leg3["manipulation"]["start_q"], leg2["manipulation"]["end_q"], atol=1e-6)
    assert leg3["manipulation"]["elbow_branch"] == leg2["manipulation"]["elbow_branch"]


def test_non_convex_obstacle_detour_tube_to_h12() -> None:
    out = process_packet({"start_well": "TUBE", "target_well": "H12", "clearance_mm": 7.5, "seed": 0})
    assert out["halted"] is False
    assert out["halt_reason"] is None
    assert out["manipulation"]["collision_free"] is True
    assert out["manipulation"]["waypoints"] >= 3
    assert out["manipulation"]["path_length_rad"] > 1.5
    assert out["manipulation"]["measured_min_clearance_mm"] >= 5.0


def test_clearance_below_5mm_blocks_packet() -> None:
    out = process_packet({"start_well": "A1", "target_well": "H12", "clearance_mm": 3.2})
    assert out["halted"] is True
    assert out["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"
    assert out["manipulation"]["collision_free"] is False
    assert out["manipulation"]["waypoints"] == 0
    assert out["manipulation"]["path_length_rad"] == 0.0
    assert out["manipulation"]["clearance_mm"] == 3.2
    assert out["manipulation"]["measured_min_clearance_mm"] == 0.0
    assert out["trace"][-1]["repo"] == "lab-hands"
    assert out["trace"][-1]["status"] == "BLOCK"
    assert out["bottom_reasoning_traces"][-1]["invariant"] == "arm_path_collision_free_with_5mm_margin"

    # Boundary check: exact 5.0 mm passes, 4.99 mm blocks
    out_499 = process_packet({"start_well": "A1", "target_well": "H12", "clearance_mm": 4.99})
    assert out_499["halted"] is True
    assert out_499["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"

    out_500 = process_packet({"start_well": "A1", "target_well": "H12", "clearance_mm": 5.0})
    assert out_500["halted"] is False
    assert out_500["manipulation"]["collision_free"] is True


def test_invalid_well_or_occluded_target_blocks_packet() -> None:
    bad_well = process_packet({"start_well": "A1", "target_well": "Z99", "clearance_mm": 8.5})
    assert bad_well["halted"] is True
    assert bad_well["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"
    assert bad_well["manipulation"]["collision_free"] is False

    # Obstacle directly covering target well H12 (~0.299, -0.013)
    occluded = process_packet(
        {
            "start_well": "A1",
            "target_well": "H12",
            "clearance_mm": 8.5,
            "obstacles": [[0.28, -0.03, 0.32, 0.01]],
        }
    )
    assert occluded["halted"] is True
    assert occluded["halt_reason"] == "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"
    assert occluded["manipulation"]["collision_free"] is False
    assert occluded["manipulation"]["clearance_mm"] == 0.0


def test_halted_packet_passthrough() -> None:
    upstream_halted = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L4_VERIFY_SQL_BLOCK",
        "start_well": "A1",
        "target_well": "H12",
        "clearance_mm": 8.5,
        "trace": [
            {
                "repo": "lab-verify",
                "layer": "L4",
                "status": "BLOCK",
                "summary": "Mutating SQL rejected.",
            }
        ],
        "bottom_reasoning_traces": [
            {
                "repo": "lab-verify",
                "invariant": "autonomous_wire_envelope_soundness",
                "first_principles": "Default-deny SQL AST check.",
            }
        ],
    }
    snapshot = copy.deepcopy(upstream_halted)
    out = autolab_wire.process_packet(upstream_halted)

    assert upstream_halted == snapshot
    assert out == snapshot
    assert "manipulation" not in out
    assert out["halt_reason"] == "L4_VERIFY_SQL_BLOCK"
    assert len(out["trace"]) == 1
