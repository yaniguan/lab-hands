"""Generate reproducible frontier robotics benchmark summary in `benchmarks/results/`.

Runs:
1. 6-DoF `UR5e` & 7-DoF `Panda` SE(3) PoE/MDH kinematics, DLS IK, 3D capsule-AABB clearance,
   `RRT-Connect` vs `Informed RRT*`, and `TOPP-RA` trajectory timing.
2. Full factorial nonlinear Bessel / spherical-pendulum slosh + `ZV`/`ZVD`/`EI` input shaping
   across `fill in {0.25, 0.50, 0.75, 0.85}` x `3 liquids` x `{96_well, 384_well}` x
   `{nominal, -25% mismatch, +25% mismatch}`.
3. 500-trial 3D Whitney/Coulomb peg-in-hole & active Archimedean spiral + Bayesian tactile
   insertion benchmark across `open_loop`, `passive_rcc_compliance`, `local_shear_admittance`,
   and `active_spiral_bayesian_tactile` on realistic `0.70 mm` polypropylene chamfers.
4. Canonical HuggingFace LeRobot v2.1/v3.0 & Opentrons 3D gantry tensor shape audits across
   cached `autobio_pipette_mujoco`, `lerobot`, and `finebio` episodes in gitignored `data/`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_hands.collision import DEFAULT_SCENE_3D, path_min_clearance_se3  # noqa: E402
from lab_hands.datasets import (  # noqa: E402
    convert_finebio_to_se3_plans,
    load_finebio_annotations,
    load_lerobot_episode,
)
from lab_hands.kinematics import PoseSE3  # noqa: E402
from lab_hands.kinematics_se3 import DEFAULT_OPENTRONS_FLEX, DEFAULT_PANDA, DEFAULT_UR5E  # noqa: E402
from lab_hands.labware import DEFAULT_PLATE_96  # noqa: E402
from lab_hands.planner import informed_rrt_star, path_length, rrt_connect_se3, shortcut_se3  # noqa: E402
from lab_hands.slosh_dynamics import run_factorial_slosh_benchmark  # noqa: E402
from lab_hands.tactile import benchmark_tactile_regimes  # noqa: E402
from lab_hands.toppra import parameterize_toppra  # noqa: E402
from lab_hands.trajectory import (  # noqa: E402
    plan_to_opentrons_trajectory,
    validate_lerobot_frame_tensors,
    validate_opentrons_trajectory,
)


def _benchmark_se3_planning_and_toppra() -> Dict[str, Any]:
    from lab_hands.collision import in_collision_se3
    from lab_hands.kinematics_se3 import make_panda_arm

    wells = ("A1", "A12", "D6", "F5", "H1", "H12")
    tip_len_m = 0.095
    z_transit = 0.18
    summary: Dict[str, Any] = {}
    bench_panda = make_panda_arm(base_offset_xyz=(-0.16, 0.0, 0.0))

    for arm in (DEFAULT_UR5E, bench_panda):
        free_fn = lambda q_cand, _a=arm: not in_collision_se3(
            q_cand, _a, DEFAULT_SCENE_3D, attached_tip_length_m=tip_len_m
        )
        start_pose = PoseSE3(xyz=(-0.05, 0.40, z_transit), rpy=(0.0, 0.0, 0.0))
        ik_s = arm.dls_ik(start_pose, attached_tip_length_m=tip_len_m, collision_free_fn=free_fn)
        assert ik_s.success

        conn_ms: List[float] = []
        conn_costs: List[float] = []
        star_init_costs: List[float] = []
        star_best_costs: List[float] = []
        topp_durs: List[float] = []
        min_clears_mm: List[float] = []
        ik_pos_errs_um: List[float] = []
        manip_vals: List[float] = []

        for idx, w in enumerate(wells):
            w_xy = DEFAULT_PLATE_96.well_xy(w)
            g_pose = PoseSE3(xyz=(float(w_xy[0]), float(w_xy[1]), z_transit), rpy=(0.0, 0.0, 0.0))
            ik_g = arm.dls_ik(
                g_pose, q_seed=ik_s.q, attached_tip_length_m=tip_len_m, collision_free_fn=free_fn
            )
            assert ik_g.success
            ik_pos_errs_um.append(ik_g.pos_error_m * 1e6)
            manip_vals.append(ik_g.manipulability)

            rng = np.random.default_rng(idx)
            t0 = time.perf_counter()
            raw = rrt_connect_se3(
                ik_s.q, ik_g.q, rng, arm=arm, scene_3d=DEFAULT_SCENE_3D, attached_tip_length_m=tip_len_m
            )
            assert raw is not None
            sp = shortcut_se3(
                raw, rng, arm=arm, scene_3d=DEFAULT_SCENE_3D, attached_tip_length_m=tip_len_m
            )
            conn_ms.append((time.perf_counter() - t0) * 1000.0)
            conn_costs.append(path_length(sp))
            min_clears_mm.append(
                path_min_clearance_se3(sp, arm, DEFAULT_SCENE_3D, attached_tip_length_m=tip_len_m) * 1000.0
            )

            ir = informed_rrt_star(
                ik_s.q,
                ik_g.q,
                np.random.default_rng(idx + 100),
                arm=arm,
                scene_3d=DEFAULT_SCENE_3D,
                iters=90,
                attached_tip_length_m=tip_len_m,
            )
            assert ir is not None
            star_init_costs.append(ir.initial_cost_rad)
            star_best_costs.append(ir.best_cost_rad)

            tp = parameterize_toppra(
                ir.path,
                qd_max=arm.qd_max,
                qdd_max=arm.qdd_max,
                qddd_max=arm.qddd_max,
            )
            topp_durs.append(tp.duration_s)

        summary[arm.name] = {
            "dof": arm.dof,
            "wells_tested": len(wells),
            "success_rate_pct": 100.0,
            "mean_ik_pos_error_um": round(float(np.mean(ik_pos_errs_um)), 3),
            "mean_manipulability": round(float(np.mean(manip_vals)), 4),
            "rrt_connect_mean_ms": round(float(np.mean(conn_ms)), 2),
            "rrt_connect_shortcut_mean_rad": round(float(np.mean(conn_costs)), 3),
            "informed_rrt_star_initial_mean_rad": round(float(np.mean(star_init_costs)), 3),
            "informed_rrt_star_best_mean_rad": round(float(np.mean(star_best_costs)), 3),
            "toppra_mean_duration_s": round(float(np.mean(topp_durs)), 4),
            "min_3d_capsule_clearance_mm": round(float(np.min(min_clears_mm)), 2),
        }

    # Verify Opentrons Flex 4-DoF gantry kinematics
    flex_pose = PoseSE3(xyz=(0.24588, 0.10174, 0.045), rpy=(0.0, 0.0, 0.25))
    q_flex = DEFAULT_OPENTRONS_FLEX.analytical_ik(flex_pose)
    assert q_flex is not None
    summary["opentrons_flex"] = {
        "dof": DEFAULT_OPENTRONS_FLEX.dof,
        "deck_bounds_mm": [[0.0, 418.0], [0.0, 353.0], [0.0, 218.0]],
        "fk_ik_exact_roundtrip": bool(
            np.allclose(DEFAULT_OPENTRONS_FLEX.fk(q_flex).homogeneous_matrix(), flex_pose.homogeneous_matrix())
        ),
    }
    return summary


def _audit_cached_real_datasets() -> Dict[str, Any]:
    data_dir = REPO_ROOT / "data" / "open_datasets"
    out: Dict[str, Any] = {}

    for ds_id in ("autobio_pipette_mujoco", "lerobot"):
        rows_file = data_dir / ds_id / "sample_rows.json"
        if rows_file.exists():
            ep = load_lerobot_episode(rows_file, dataset_id=ds_id)
            t_raw = ep.to_lerobot_tensors(single_arm_7d=False)
            t_7d = ep.to_lerobot_tensors(single_arm_7d=True)
            validate_lerobot_frame_tensors(t_raw)
            validate_lerobot_frame_tensors(t_7d)
            out[ds_id] = {
                "episode_index": ep.episode_index,
                "fps": ep.fps,
                "n_frames": ep.n_frames,
                "raw_observation_state_shape": list(t_raw["observation.state"].shape),
                "single_arm_observation_state_shape": list(t_7d["observation.state"].shape),
                "single_arm_action_shape": list(t_7d["action"].shape),
                "dtypes": {k: str(v.dtype) for k, v in t_7d.items()},
            }

    fb_file = data_dir / "finebio" / "P01_01_01.txt"
    if fb_file.exists():
        anns = load_finebio_annotations(fb_file)
        compiled = convert_finebio_to_se3_plans(anns[:4], arm_se3=DEFAULT_UR5E)
        ot_frames = 0
        for _, pl in compiled:
            assert pl is not None
            ot = plan_to_opentrons_trajectory(pl)
            validate_opentrons_trajectory(ot.to_dict())
            ot_frames += ot.n_frames
        out["finebio_P01_01_01"] = {
            "total_protocol_steps": len(anns),
            "compiled_ur5e_steps": len(compiled),
            "total_opentrons_gantry_frames": ot_frames,
            "deck_mm_bounds_verified": True,
        }
    return out


def main() -> int:
    se3_summary = _benchmark_se3_planning_and_toppra()
    slosh_full = run_factorial_slosh_benchmark()
    tactile_500 = benchmark_tactile_regimes(n_trials=500, seed=20261127)
    real_ds = _audit_cached_real_datasets()

    compact = {
        "schema": "lab-hands/frontier-benchmarks/v1",
        "kinematics_planning_toppra": se3_summary,
        "slosh_factorial_6dof": {
            "benchmark": slosh_full["benchmark"],
            "base_toppra_duration_s": slosh_full["base_toppra_duration_s"],
            "fill_fractions": slosh_full["fill_fractions"],
            "labware": slosh_full["labware"],
            "liquids": [x["liquid_id"] for x in slosh_full["liquids"]],
            "summary_by_regime": slosh_full["summary_by_regime"],
        },
        "tactile_coulomb_whitney_500": tactile_500,
        "real_dataset_and_opentrons_interfaces": real_ds,
    }

    out_path = REPO_ROOT / "benchmarks" / "results" / "frontier_benchmark_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
