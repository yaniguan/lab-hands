"""Auto-Lab 15-repository wire node (`autolab-wire/v1`) for `lab-hands` (Layer L1).

Bridges upstream L4 verification (`lab-verify`) to downstream L0 edge execution (`lab-feet`)
and L4 causal attribution (`lab-attrib`) by solving continuity-aware 2-link SBS inverse
kinematics, deterministic joint-space RRT + shortcut smoothing, certified continuous
capsule-AABB clearance measurement, and closed-loop tactile bevel insertion with slip/seal telemetry.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from lab_hands.collision import (  # noqa: E402
    DEFAULT_SCENE,
    BenchScene,
    edge_free,
    in_collision,
    path_min_clearance,
)
from lab_hands.kinematics import DEFAULT_ARM, Arm2D  # noqa: E402
from lab_hands.labware import get_plate_geometry, parse_well_ref  # noqa: E402
from lab_hands.planner import (  # noqa: E402
    ik_solutions,
    path_length,
    rrt,
    rrt_connect,
    shortcut,
)
from lab_hands.tactile import ClearVesselBevel, InsertionResult, run_tactile_insertion  # noqa: E402

SCHEMA_VERSION = "autolab-wire/v1"
REPO_ID = "lab-hands"
LAYER = "L1"
HALT_REASON = "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"
TACTILE_HALT_REASON = "L1_HANDS_TACTILE_INSERTION_OR_SEAL_BLOCK"
INVARIANT_ID = "arm_path_collision_free_with_5mm_margin"
MIN_CLEARANCE_MM = 5.0


def _resolve_target_xy(well_ref: object, scene: BenchScene) -> Optional[np.ndarray]:
    """Resolve an SBS well reference ('A1'..'H12', 'slot:A1', 1..N) or 'TUBE' to XY (m)."""
    try:
        if isinstance(well_ref, int) and not isinstance(well_ref, bool):
            name = scene.plate.well_name(well_ref, order="row")
            return scene.plate.well_xy(name)
        if not isinstance(well_ref, str) or not well_ref.strip():
            return None
        _, token = parse_well_ref(well_ref)
        if token == "TUBE":
            return np.asarray(scene.tube_xy, dtype=np.float64)
        return scene.plate.well_xy(token)
    except (ValueError, KeyError, TypeError):
        return None


def _build_arm(packet: Dict[str, Any]) -> Optional[Arm2D]:
    raw_radius = packet.get("capsule_radius_m", 0.0)
    try:
        cap_r = float(raw_radius)
        if not math.isfinite(cap_r) or cap_r < 0.0:
            return None
        if cap_r == 0.0:
            return DEFAULT_ARM
        return Arm2D(
            l1=DEFAULT_ARM.l1,
            l2=DEFAULT_ARM.l2,
            joint_limits=DEFAULT_ARM.joint_limits,
            clearance_m=cap_r,
            margin_m=DEFAULT_ARM.margin_m,
        )
    except (ValueError, TypeError):
        return None


def _build_scene(packet: Dict[str, Any]) -> Optional[BenchScene]:
    """Build workspace scene from optional custom obstacles, labware load_name, and tip length."""
    custom_obstacles = packet.get("obstacles")
    load_name = packet.get("labware")
    raw_tip_len = packet.get("attached_tip_length_m", 0.0)
    if custom_obstacles is None and load_name is None and raw_tip_len in (0, 0.0, None):
        return DEFAULT_SCENE
    try:
        boxes = (
            DEFAULT_SCENE.obstacles
            if custom_obstacles is None
            else tuple((float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in custom_obstacles)
        )
        plate = DEFAULT_SCENE.plate if load_name is None else get_plate_geometry(str(load_name))
        tip_len = 0.0 if raw_tip_len is None else float(raw_tip_len)
        return BenchScene(
            obstacles=boxes,
            tube_xy=DEFAULT_SCENE.tube_xy,
            attached_tip_length_m=tip_len,
            plate=plate,
        )
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def _parse_start_q(
    packet: Dict[str, Any],
    start_xy: np.ndarray,
    arm: Arm2D,
    scene: BenchScene,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Extract and validate optional explicit `start_q` or chained `manipulation.end_q`."""
    if "start_q" in packet and packet["start_q"] is not None:
        try:
            arr = np.asarray(packet["start_q"], dtype=np.float64)
        except (ValueError, TypeError):
            return False, None
        if arr.shape != (2,) or not np.isfinite(arr).all():
            return False, None
        if not arm.within_joint_limits(arr) or in_collision(arr, arm=arm, scene=scene):
            return False, None
        return True, arr.copy()

    prev_manip = packet.get("manipulation")
    if isinstance(prev_manip, dict) and prev_manip.get("end_q") is not None:
        try:
            arr = np.asarray(prev_manip["end_q"], dtype=np.float64)
            if (
                arr.shape == (2,)
                and np.isfinite(arr).all()
                and arm.within_joint_limits(arr)
                and not in_collision(arr, arm=arm, scene=scene)
                and float(np.linalg.norm(arm.fk(arr)[1] - start_xy)) <= 2e-3
            ):
                return True, arr.copy()
        except (ValueError, TypeError):
            pass
    return True, None


def _plan_collision_free_move(
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    seed: int,
    arm: Arm2D,
    scene: BenchScene,
    start_q: Optional[Sequence[float] | np.ndarray] = None,
) -> Optional[Tuple[List[np.ndarray], float]]:
    """Solve continuity-aware 2-link IK + seeded RRT + shortcutting and return `(path, clearance_mm)`."""
    if start_q is not None:
        q_ref = np.asarray(start_q, dtype=np.float64)
        if q_ref.shape != (2,) or not arm.within_joint_limits(q_ref) or in_collision(q_ref, arm=arm, scene=scene):
            return None
        if float(np.linalg.norm(arm.fk(q_ref)[1] - start_xy)) <= 2e-3:
            starts = [q_ref.copy()]
        else:
            starts = ik_solutions(start_xy, arm=arm, scene=scene, prev_q=q_ref)
    else:
        starts = ik_solutions(start_xy, arm=arm, scene=scene)

    if not starts:
        return None

    for branch_offset, q_start in enumerate(starts):
        goals = ik_solutions(target_xy, arm=arm, scene=scene, prev_q=q_start)
        if not goals:
            continue
        rng = np.random.default_rng(seed + branch_offset)
        raw_path = rrt(q_start, goals, rng, arm=arm, scene=scene)
        if raw_path is None:
            raw_path = rrt_connect(q_start, goals, rng, arm=arm, scene=scene)
        if raw_path is None:
            continue

        smooth_path = shortcut(raw_path, rng, arm=arm, scene=scene)
        if not smooth_path:
            continue
        if not all(arm.within_joint_limits(q) for q in smooth_path):
            continue
        if not all(edge_free(a, b, arm=arm, scene=scene) for a, b in zip(smooth_path, smooth_path[1:])):
            continue
        measured_mm = float(path_min_clearance(smooth_path, arm=arm, scene=scene) * 1000.0)
        if measured_mm < MIN_CLEARANCE_MM - 1e-9:
            continue
        return smooth_path, measured_mm

    return None


def _parse_visual_offset_m(raw_offset: Any) -> Optional[np.ndarray]:
    if raw_offset is None:
        return np.zeros(2, dtype=np.float64)
    if isinstance(raw_offset, (int, float)) and not isinstance(raw_offset, bool):
        val = float(raw_offset)
        if not math.isfinite(val):
            return None
        return np.array([val / 1000.0, 0.0], dtype=np.float64)
    try:
        arr = np.asarray(raw_offset, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if arr.shape != (2,) or not np.isfinite(arr).all():
        return None
    return arr / 1000.0


def _build_tactile_payload(ins: InsertionResult) -> Dict[str, Any]:
    ok = bool(ins.success and ins.seal_intact and not ins.slip_detected)
    return {
        "inserted": ok,
        "seal_intact": bool(ins.seal_intact),
        "slip_detected": bool(ins.slip_detected),
        "initial_error_mm": round(float(ins.initial_error_mm), 4),
        "final_error_mm": round(float(ins.final_error_mm), 4),
        "axial_slip_mm": round(float(ins.axial_slip_mm), 4),
        "peak_force_n": round(float(ins.peak_force_n), 4),
        "steps": int(ins.steps),
        "execution_artifact": not ok,
        "attribution_hint": "NOMINAL" if ok else "EXECUTION_ARTIFACT",
    }


def process_packet(packet: dict) -> dict:
    """Process an `autolab-wire/v1` packet through L1 `lab-hands` planning, clearance & tactile gates."""
    out: Dict[str, Any] = copy.deepcopy(packet)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if bool(out["halted"]):
        return out

    raw_start = out.get("start_well", "A1")
    raw_target = out.get("target_well", "H12")
    start_well = str(raw_start)
    target_well = str(raw_target)
    try:
        requested_clearance_mm = float(out.get("clearance_mm", 8.5))
    except (ValueError, TypeError):
        requested_clearance_mm = float("nan")
    seed = int(out.get("seed", 0))
    require_tactile = bool(out.get("require_tactile_insert", True))

    arm = _build_arm(out)
    scene = _build_scene(out)
    smooth_path: Optional[List[np.ndarray]] = None
    measured_clearance_mm: float = 0.0
    target_xy: Optional[np.ndarray] = None

    clearance_ok = math.isfinite(requested_clearance_mm) and requested_clearance_mm >= MIN_CLEARANCE_MM
    if clearance_ok and arm is not None and scene is not None:
        start_xy = _resolve_target_xy(raw_start, scene)
        target_xy = _resolve_target_xy(raw_target, scene)
        if start_xy is not None and target_xy is not None:
            start_q_valid, q_init = _parse_start_q(out, start_xy, arm, scene)
            if start_q_valid:
                plan_res = _plan_collision_free_move(
                    start_xy,
                    target_xy,
                    seed=seed,
                    arm=arm,
                    scene=scene,
                    start_q=q_init,
                )
                if plan_res is not None:
                    smooth_path, measured_clearance_mm = plan_res

    tactile_payload: Optional[Dict[str, Any]] = None
    if not clearance_ok or smooth_path is None:
        out["halted"] = True
        out["halt_reason"] = HALT_REASON
        len_waypoints = 0
        path_len = 0.0
        collision_free = False
        reported_clearance_mm = (
            requested_clearance_mm
            if (math.isfinite(requested_clearance_mm) and requested_clearance_mm < MIN_CLEARANCE_MM)
            else 0.0
        )
        start_q_out: Optional[List[float]] = None
        end_q_out: Optional[List[float]] = None
        elbow_branch: Optional[int] = None
    else:
        len_waypoints = len(smooth_path)
        path_len = float(path_length(smooth_path))
        collision_free = True
        reported_clearance_mm = round(measured_clearance_mm, 3)
        start_q_out = [round(float(smooth_path[0][0]), 6), round(float(smooth_path[0][1]), 6)]
        end_q_out = [round(float(smooth_path[-1][0]), 6), round(float(smooth_path[-1][1]), 6)]
        elbow_branch = 1 if float(smooth_path[-1][1]) >= 0.0 else -1

        if require_tactile and target_xy is not None and arm is not None and scene is not None:
            offset_m = _parse_visual_offset_m(out.get("visual_offset_mm"))
            capture_r = float(out.get("bevel_capture_radius_m", 0.0055))
            vessel = ClearVesselBevel(
                center_xy=(float(target_xy[0]), float(target_xy[1])),
                bore_radius_m=0.00008,
                capture_radius_m=capture_r,
            )
            if offset_m is None:
                failed_ins = InsertionResult(
                    success=False,
                    initial_error_mm=0.0,
                    final_error_mm=0.0,
                    steps=0,
                    peak_force_n=0.0,
                    final_z_m=vessel.z_rim_m,
                    seal_intact=False,
                    slip_detected=True,
                    axial_slip_mm=vessel.target_depth_m * 1000.0,
                )
                tactile_payload = _build_tactile_payload(failed_ins)
            else:
                vis_xy = target_xy + offset_m
                q_tactile_seed = smooth_path[-1] if float(np.linalg.norm(offset_m)) <= 1e-6 else smooth_path[-1]
                try:
                    ins_res = run_tactile_insertion(
                        vis_xy,
                        np.random.default_rng(seed),
                        vessel=vessel,
                        arm=arm,
                        scene=scene,
                        start_q=q_tactile_seed,
                    )
                except ValueError:
                    err_mm = float(np.linalg.norm(offset_m)) * 1000.0
                    ins_res = InsertionResult(
                        success=False,
                        initial_error_mm=err_mm,
                        final_error_mm=err_mm,
                        steps=0,
                        peak_force_n=0.0,
                        final_z_m=vessel.z_rim_m,
                        seal_intact=False,
                        slip_detected=True,
                        axial_slip_mm=vessel.target_depth_m * 1000.0,
                    )
                tactile_payload = _build_tactile_payload(ins_res)

            if not tactile_payload["inserted"] or not tactile_payload["seal_intact"]:
                out["halted"] = True
                out["halt_reason"] = TACTILE_HALT_REASON

    manip_dict: Dict[str, Any] = {
        "start_well": start_well,
        "target_well": target_well,
        "collision_free": collision_free,
        "waypoints": len_waypoints,
        "path_length_rad": round(path_len, 4),
        "clearance_mm": reported_clearance_mm,
        "measured_min_clearance_mm": round(measured_clearance_mm, 3) if collision_free else 0.0,
        "requested_min_clearance_mm": (
            round(requested_clearance_mm, 3) if math.isfinite(requested_clearance_mm) else 0.0
        ),
        "start_q": start_q_out,
        "end_q": end_q_out,
        "elbow_branch": elbow_branch,
    }
    if tactile_payload is not None:
        manip_dict["tactile"] = tactile_payload
    out["manipulation"] = manip_dict

    status = "BLOCK" if out["halted"] else "PASS"
    if out["halted"] and out["halt_reason"] == TACTILE_HALT_REASON and tactile_payload is not None:
        summary = (
            f"Blocked tactile insertion at {target_well}: "
            f"final_error_mm={tactile_payload['final_error_mm']}, "
            f"peak_force_n={tactile_payload['peak_force_n']}, "
            f"seal_intact={tactile_payload['seal_intact']} (EXECUTION_ARTIFACT)."
        )
    elif out["halted"]:
        summary = (
            f"Blocked manipulation {start_well} -> {target_well}: "
            f"clearance_mm={out['manipulation']['clearance_mm']} (< 5.0 mm) or no collision-free IK/RRT path."
        )
    else:
        summary = (
            f"Planned collision-free 2-link RRT+shortcut path {start_well} -> {target_well} "
            f"({len_waypoints} waypoints, {round(path_len, 4)} rad, "
            f"measured clearance {round(measured_clearance_mm, 3)} mm >= 5.0 mm)."
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": status,
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": INVARIANT_ID,
            "first_principles": (
                "Analytical 2-link SBS 96-well inverse kinematics (|q2| <= 2.60 rad), Lipschitz-certified "
                "continuous capsule-AABB clearance checking (>= 5.0 mm margin), and closed-loop tactile "
                "shear compliance verify collision-free transport and tip-seal seating before actuation."
            ),
        }
    )
    return out


if __name__ == "__main__":
    print(json.dumps(process_packet({}), indent=2))
