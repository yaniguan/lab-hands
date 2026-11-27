"""Benchtop motion-planning benchmark for 6-DoF UR5e, 7-DoF Franka Panda, and 2-DoF arms.

Moves a pipette tip from a reagent pickup tube to named SBS 96-well plate wells around
3D benchtop obstacles (reagent bottle + rack partition), supporting:
  - 6-DoF `UR5eArm` & 7-DoF `PandaArm` PoE/MDH kinematics, DLS null-space IK, 3D multi-link
    capsule-vs-AABB continuous swept collision checking, `rrt_connect_se3`, `informed_rrt_star`,
    and `TOPP-RA` reachability path parameterization (`--arm ur5e` / `--arm panda`),
  - Backward-compatible 2-DoF `Arm2D` planar RRT / RRT-Connect and `lab-hands/trajectory/v1`
    JSONL export.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lab_hands.collision import (  # noqa: E402
    DEFAULT_OBSTACLES,
    DEFAULT_SCENE_3D,
    DEFAULT_TUBE_XY,
    BenchScene,
    edge_free as _edge_free,
    edge_free_se3,
    in_collision as _in_collision,
    min_clearance as _min_clearance,
    path_min_clearance,
    path_min_clearance_se3,
)
from lab_hands.kinematics import DEFAULT_JOINT_LIMITS, Arm2D, PoseSE3  # noqa: E402
from lab_hands.kinematics_se3 import DEFAULT_PANDA, DEFAULT_UR5E  # noqa: E402
from lab_hands.labware import PlateGeometry  # noqa: E402
from lab_hands.planner import (  # noqa: E402
    ik_solutions,
    informed_rrt_star,
    path_length as length,
    plan_well_tour,
    rrt as _rrt,
    rrt_connect as _rrt_connect,
    rrt_connect_se3,
    select_ik_branch,
    shortcut as _shortcut,
    shortcut_se3,
)
from lab_hands.toppra import parameterize_toppra  # noqa: E402
from lab_hands.trajectory import TrajectoryRecord, build_trajectory_record  # noqa: E402

L1, L2 = 0.30, 0.25  # link lengths (m)
JOINT_LIMITS = DEFAULT_JOINT_LIMITS  # ((-pi, pi), (-2.6, 2.6))
CLEARANCE_M = 0.0  # link radius (m)
MARGIN_M = 0.005  # closed-box safety margin (m)
PLATE_ORIGIN = np.array([0.20, 0.05])  # center of well A1 (m)
PITCH = 0.009  # SBS 96-well pitch: 9 mm
ROWS = "ABCDEFGH"
OBSTACLES = list(DEFAULT_OBSTACLES)  # axis-aligned boxes: xmin, ymin, xmax, ymax
TUBE = np.array(DEFAULT_TUBE_XY)  # tube position to aspirate from

_EXPORTED_HELPERS = (plan_well_tour, select_ik_branch, edge_free_se3)


def _current_arm(clearance_m: Optional[float] = None, margin_m: Optional[float] = None) -> Arm2D:
    return Arm2D(
        l1=L1,
        l2=L2,
        joint_limits=JOINT_LIMITS,
        clearance_m=CLEARANCE_M if clearance_m is None else float(clearance_m),
        margin_m=MARGIN_M if margin_m is None else float(margin_m),
    )


def _current_plate() -> PlateGeometry:
    return PlateGeometry(
        rows=len(ROWS),
        cols=12,
        pitch_x=PITCH,
        pitch_y=PITCH,
        origin_xy=(float(PLATE_ORIGIN[0]), float(PLATE_ORIGIN[1])),
    )


def _current_scene() -> BenchScene:
    return BenchScene(
        obstacles=tuple(tuple(float(v) for v in box) for box in OBSTACLES),
        tube_xy=(float(TUBE[0]), float(TUBE[1])),
        plate=_current_plate(),
    )


def well_name(index: int, order: str = "row") -> str:
    """1-based well index -> name. 'row': A1..A12,B1..; 'col': A1..H1,A2.."""
    return _current_plate().well_name(index, order=order)


def well_xy(name: str) -> np.ndarray:
    return _current_plate().well_xy(name)


def fk(q: Sequence[float] | np.ndarray):
    return _current_arm().fk(q)


def ik(p: Sequence[float] | np.ndarray, elbow: int = 1) -> Optional[np.ndarray]:
    return _current_arm().ik(p, elbow=elbow)


def in_collision(
    q: Sequence[float] | np.ndarray,
    *,
    clearance_m: Optional[float] = None,
    margin_m: Optional[float] = None,
) -> bool:
    arm = _current_arm(clearance_m=clearance_m, margin_m=margin_m)
    return _in_collision(q, arm=arm, scene=_current_scene())


def min_clearance(q: Sequence[float] | np.ndarray) -> float:
    return _min_clearance(q, arm=_current_arm(), scene=_current_scene())


def edge_free(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    step: float = 0.02,
    *,
    max_tip_step_m: float = 0.002,
) -> bool:
    return _edge_free(a, b, arm=_current_arm(), scene=_current_scene(), step=step, max_tip_step_m=max_tip_step_m)


def rrt(
    start,
    goal,
    rng: np.random.Generator,
    iters: int = 3000,
    step: float = 0.15,
    goal_bias: float = 0.1,
) -> Optional[List[np.ndarray]]:
    return _rrt(start, goal, rng, iters=iters, step=step, goal_bias=goal_bias, arm=_current_arm(), scene=_current_scene())


def rrt_connect(
    start,
    goal,
    rng: np.random.Generator,
    iters: int = 3000,
    step: float = 0.20,
) -> Optional[List[np.ndarray]]:
    return _rrt_connect(start, goal, rng, iters=iters, step=step, arm=_current_arm(), scene=_current_scene())


def shortcut(path: Sequence[np.ndarray], rng: np.random.Generator, tries: int = 100) -> List[np.ndarray]:
    return _shortcut(path, rng, tries=tries, arm=_current_arm(), scene=_current_scene())


def _run_se3_cobot_showcase(arm_name: str = "ur5e", seeds_per_well: int = 2) -> int:
    from lab_hands.collision import in_collision_se3
    from lab_hands.kinematics_se3 import make_panda_arm

    # Benchtop cell mount: Panda base sits at (-0.16, 0, 0) m behind deck so all 96 wells + TUBE
    # lie in Panda's 0.36..0.46 m sweet-spot reach outside its 88 mm J7 inner dead-zone!
    arm_se3 = DEFAULT_UR5E if arm_name == "ur5e" else make_panda_arm(base_offset_xyz=(-0.16, 0.0, 0.0))
    tip_len_m = 0.095  # Mounted 1000 uL micropipette tip clearing 240 mm rack partition
    scene_3d = DEFAULT_SCENE_3D
    z_transit = 0.18
    free_fn = lambda q_cand: not in_collision_se3(
        q_cand, arm_se3, scene_3d, attached_tip_length_m=tip_len_m
    )

    start_ik = arm_se3.dls_ik(
        PoseSE3(xyz=(float(TUBE[0]), float(TUBE[1]), z_transit), rpy=(0.0, 0.0, 0.0)),
        attached_tip_length_m=tip_len_m,
        collision_free_fn=free_fn,
    )
    if not start_ik.success:
        print(f"error: {arm_se3.name} start DLS IK failed at {TUBE}", file=sys.stderr)
        return 1
    q_start = start_ik.q

    targets = ["A1", "A12", "D6", "F5", "H1", "H12"]
    conn_times, star_gains, topp_durs, clear_mms = [], [], [], []
    total = 0
    ok = 0

    for w in targets:
        txy = well_xy(w)
        goal_ik = arm_se3.dls_ik(
            PoseSE3(xyz=(float(txy[0]), float(txy[1]), z_transit), rpy=(0.0, 0.0, 0.0)),
            q_seed=q_start,
            attached_tip_length_m=tip_len_m,
            collision_free_fn=free_fn,
        )
        if not goal_ik.success:
            return 1
        q_goal = goal_ik.q
        for sd in range(seeds_per_well):
            total += 1
            rng = np.random.default_rng(sd)
            t0 = time.perf_counter()
            raw = rrt_connect_se3(
                q_start, q_goal, rng, arm=arm_se3, scene_3d=scene_3d, attached_tip_length_m=tip_len_m
            )
            if raw is None:
                continue
            sp = shortcut_se3(
                raw, rng, arm=arm_se3, scene_3d=scene_3d, attached_tip_length_m=tip_len_m
            )
            conn_times.append((time.perf_counter() - t0) * 1000.0)
            cl_mm = path_min_clearance_se3(sp, arm_se3, scene_3d, attached_tip_length_m=tip_len_m) * 1000.0
            if cl_mm < 5.0 - 1e-6:
                return 1
            clear_mms.append(cl_mm)
            topp = parameterize_toppra(
                sp,
                qd_max=arm_se3.qd_max,
                qdd_max=arm_se3.qdd_max,
                qddd_max=arm_se3.qddd_max,
            )
            topp_durs.append(topp.duration_s)
            if sd == 0 and w in ("A12", "H12"):
                ir = informed_rrt_star(
                    q_start,
                    q_goal,
                    rng,
                    arm=arm_se3,
                    scene_3d=scene_3d,
                    iters=70,
                    attached_tip_length_m=tip_len_m,
                )
                if ir is not None and ir.initial_cost_rad > 1e-6:
                    star_gains.append(100.0 * (ir.initial_cost_rad - ir.best_cost_rad) / ir.initial_cost_rad)
            ok += 1

    if ok != total:
        return 1
    mean_gain = float(np.mean(star_gains)) if star_gains else 0.0
    print(
        f"6-DoF/7-DoF SE(3) cobot ({arm_se3.name.upper()}, {arm_se3.dof}-DoF): {ok}/{total} (100.0%), "
        f"RRT-Connect mean {np.mean(conn_times):.1f} ms, Informed-RRT* cost reduction {mean_gain:.1f}%, "
        f"TOPP-RA mean cycle {np.mean(topp_durs):.2f} s, min 3D capsule clearance {np.min(clear_mms):.1f} mm"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", choices=["rrt", "rrt_connect", "informed_rrt_star"], default="rrt")
    ap.add_argument("--arm", choices=["2d_and_ur5e", "ur5e", "panda", "2d"], default="2d_and_ur5e")
    ap.add_argument("--seeds", type=int, default=10, help="Number of RNG seeds per target well")
    ap.add_argument("--out-jsonl", type=Path, default=None, help="Optional JSONL path for lab-hands/trajectory/v1 records")
    args = ap.parse_args(argv)

    print("well 65 ->", well_name(65, "row"), "(row-major),", well_name(65, "col"), "(column-major)")

    if args.arm in ("ur5e", "panda") and args.out_jsonl is None:
        return _run_se3_cobot_showcase(arm_name=args.arm, seeds_per_well=max(1, min(args.seeds, 5)))

    arm = _current_arm()
    scene = _current_scene()

    starts = ik_solutions(TUBE, arm=arm, scene=scene)
    if not starts:
        print("error: start configuration at TUBE is unreachable or in collision", file=sys.stderr)
        raise SystemExit(1)
    start = starts[0]

    blocked = 0
    ok, times, raw_lens, smooth_lens, clearances = 0, [], [], [], []
    records: List[TrajectoryRecord] = []
    targets = ["A1", "A12", "D6", "E5", "F5", "H1", "H12", "C9", "G3", "B7"]
    trials = 0

    for w in targets:
        target_xy = well_xy(w)
        goals = ik_solutions(target_xy, arm=arm, scene=scene)
        if not goals:
            print(f"error: no collision-free IK solution for target well {w}", file=sys.stderr)
            raise SystemExit(1)
        blocked += int(not any(_edge_free(start, g, arm=arm, scene=scene) for g in goals))
        for seed in range(args.seeds):
            trials += 1
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            if args.planner == "informed_rrt_star":
                ir = informed_rrt_star(start, goals[0], rng, arm=arm, scene=scene, iters=300)
                path = None if ir is None else ir.path
            elif args.planner == "rrt_connect":
                path = _rrt_connect(start, goals, rng, arm=arm, scene=scene)
            else:
                path = _rrt(start, goals, rng, arm=arm, scene=scene)

            if path is None:
                continue
            sp = _shortcut(path, rng, arm=arm, scene=scene)
            dt = time.perf_counter() - t0

            if not all(_edge_free(a, b, arm=arm, scene=scene) for a, b in zip(sp, sp[1:])):
                print(f"error: collision detected on smoothed path for {w} (seed={seed})", file=sys.stderr)
                raise SystemExit(1)
            _, tip_end = arm.fk(sp[-1])
            if float(np.linalg.norm(tip_end - target_xy)) > 1e-6:
                print(f"error: endpoint FK error for {w} (seed={seed})", file=sys.stderr)
                raise SystemExit(1)

            times.append(dt)
            ok += 1
            raw_lens.append(length(path))
            smooth_lens.append(length(sp))
            clearances.append(path_min_clearance(sp, arm=arm, scene=scene))
            if args.out_jsonl is not None:
                records.append(
                    build_trajectory_record(
                        trial=trials,
                        seed=seed,
                        planner=args.planner,
                        goal_well=w,
                        raw_path=path,
                        smooth_path=sp,
                        plan_time_s=dt,
                        arm=arm,
                        scene=scene,
                    )
                )

    if not ok or ok != trials:
        print(f"error: planner succeeded on {ok}/{trials} trials (100% required)", file=sys.stderr)
        raise SystemExit(1)

    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.out_jsonl.write_text("\n".join(r.to_json() for r in records) + "\n", encoding="utf-8")

    print(f"straight joint-space move blocked by obstacles for {blocked} of {len(targets)} wells")
    print(f"trials: {trials}, success: {ok} ({100 * ok / trials:.1f}%)")
    print(f"plan time ({args.planner}): mean {np.mean(times) * 1000:.1f} ms, max {np.max(times) * 1000:.1f} ms")
    print(
        f"joint path length: raw {np.mean(raw_lens):.2f} rad -> shortcut mean {np.mean(smooth_lens):.2f} rad "
        f"(p95 {np.percentile(smooth_lens, 95):.2f} rad, max {np.max(smooth_lens):.2f} rad)"
    )
    print(f"minimum link-obstacle clearance: min {np.min(clearances) * 1000:.1f} mm, mean {np.mean(clearances) * 1000:.1f} mm")

    if args.arm == "2d_and_ur5e":
        rc6 = _run_se3_cobot_showcase(arm_name="ur5e", seeds_per_well=1)
        if rc6 != 0:
            raise SystemExit(rc6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
