"""Benchtop motion-planning benchmark: move a pipette tip held by a planar 2-link arm
from a tube rack to named wells of an SBS 96-well plate, avoiding a rack wall and a
reagent bottle. Supports forward RRT and bidirectional RRT-Connect, shortcut smoothing,
minimum-clearance verification, and `lab-hands/trajectory/v1` JSONL export.
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
    DEFAULT_TUBE_XY,
    BenchScene,
    edge_free as _edge_free,
    in_collision as _in_collision,
    min_clearance as _min_clearance,
    path_min_clearance,
)
from lab_hands.kinematics import DEFAULT_JOINT_LIMITS, Arm2D  # noqa: E402
from lab_hands.labware import PlateGeometry  # noqa: E402
from lab_hands.planner import (  # noqa: E402
    ik_solutions,
    path_length as length,
    rrt as _rrt,
    rrt_connect as _rrt_connect,
    shortcut as _shortcut,
)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", choices=["rrt", "rrt_connect"], default="rrt")
    ap.add_argument("--seeds", type=int, default=10, help="Number of RNG seeds per target well")
    ap.add_argument("--out-jsonl", type=Path, default=None, help="Optional JSONL path for lab-hands/trajectory/v1 records")
    args = ap.parse_args(argv)

    arm = _current_arm()
    scene = _current_scene()

    print("well 65 ->", well_name(65, "row"), "(row-major),", well_name(65, "col"), "(column-major)")
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
    solve_fn = _rrt_connect if args.planner == "rrt_connect" else _rrt

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
            path = solve_fn(start, goals, rng, arm=arm, scene=scene)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
