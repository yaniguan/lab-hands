"""Sampling-based joint-space motion planners (RRT, RRT-Connect), IK branch continuity, and tour DP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .collision import (
    DEFAULT_SCENE,
    DEFAULT_SCENE_3D,
    BenchScene,
    BenchScene3D,
    edge_free,
    edge_free_se3,
    in_collision,
    in_collision_se3,
    path_min_clearance_se3,
)
from .kinematics import DEFAULT_ARM, Arm2D, PoseSE3
from .labware import DeckLayout, PlateGeometry

GoalInput = Union[np.ndarray, Sequence[float], Sequence[np.ndarray]]
WellRefInput = Union[str, Sequence[float], np.ndarray]


def ik_solutions(
    p: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    prev_q: Optional[Sequence[float] | np.ndarray] = None,
) -> List[np.ndarray]:
    """Return all collision-free, joint-limit-valid IK configurations for target `p`.

    When `prev_q` is provided, valid branches are ordered by joint-space proximity to `prev_q`
    so sequential moves preserve elbow continuity whenever possible.
    """
    out: List[np.ndarray] = []
    for elbow in (1, -1):
        q = arm.ik(p, elbow=elbow)
        if q is not None and not in_collision(q, arm=arm, scene=scene):
            if not any(np.allclose(q, prev, atol=1e-9) for prev in out):
                out.append(q)
    if prev_q is not None and len(out) > 1:
        q_ref = np.asarray(prev_q, dtype=np.float64)
        if q_ref.shape == (2,) and np.isfinite(q_ref).all():
            out.sort(key=lambda q: float(np.linalg.norm(q - q_ref)))
    return out


def select_ik_branch(
    p: Sequence[float] | np.ndarray,
    prev_q: Optional[Sequence[float] | np.ndarray] = None,
    *,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    prefer_elbow: int = 1,
) -> Optional[np.ndarray]:
    """Select a collision-free IK branch minimizing joint displacement from `prev_q`."""
    if prefer_elbow not in (1, -1):
        raise ValueError(f"prefer_elbow must be +1 or -1, got {prefer_elbow!r}")
    sols = ik_solutions(p, arm=arm, scene=scene, prev_q=prev_q)
    if not sols:
        return None
    if prev_q is not None:
        return sols[0].copy()
    for q in sols:
        branch = 1 if q[1] >= 0.0 else -1
        if branch == prefer_elbow:
            return q.copy()
    return sols[0].copy()


def _normalize_goals(
    goal: GoalInput,
    arm: Arm2D,
    scene: BenchScene,
) -> List[np.ndarray]:
    if goal is None:
        return []
    if isinstance(goal, np.ndarray):
        candidates = [goal] if goal.ndim == 1 else [goal[i] for i in range(len(goal))]
    elif isinstance(goal, (list, tuple)) and len(goal) > 0 and isinstance(goal[0], (np.ndarray, list, tuple)):
        candidates = [np.asarray(g, dtype=np.float64) for g in goal if g is not None]
    else:
        candidates = [np.asarray(goal, dtype=np.float64)]

    valid: List[np.ndarray] = []
    for g in candidates:
        arr = np.asarray(g, dtype=np.float64)
        if arr.shape == (2,) and not in_collision(arr, arm=arm, scene=scene):
            if not any(np.allclose(arr, prev, atol=1e-9) for prev in valid):
                valid.append(arr)
    return valid


def _direct_shortcut_goals(
    start: np.ndarray,
    goals: List[np.ndarray],
    arm: Arm2D,
    scene: BenchScene,
) -> Optional[List[np.ndarray]]:
    for g in sorted(goals, key=lambda q: float(np.linalg.norm(q - start))):
        if np.allclose(start, g, atol=1e-9):
            return [start.copy()]
        if edge_free(start, g, arm=arm, scene=scene):
            return [start.copy(), g.copy()]
    return None


def _trace_path(nodes: List[np.ndarray], parent: List[int], idx: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    k = idx
    while k != -1:
        out.append(nodes[k].copy())
        k = parent[k]
    out.reverse()
    return out


def rrt(
    start: Sequence[float] | np.ndarray,
    goal: GoalInput,
    rng: np.random.Generator,
    iters: int = 3000,
    step: float = 0.15,
    goal_bias: float = 0.1,
    *,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> Optional[List[np.ndarray]]:
    """Forward joint-space RRT prioritizing same-elbow goal continuity before cross-branch fallback."""
    if start is None:
        return None
    q_start = np.asarray(start, dtype=np.float64)
    if q_start.shape != (2,) or in_collision(q_start, arm=arm, scene=scene):
        return None
    goals = _normalize_goals(goal, arm=arm, scene=scene)
    if not goals:
        return None

    direct = _direct_shortcut_goals(q_start, goals, arm=arm, scene=scene)
    if direct is not None:
        return direct

    goals_sorted = sorted(goals, key=lambda g: float(np.linalg.norm(g - q_start)))
    same_branch_warmup = min(400, max(1, iters // 4))

    nodes: List[np.ndarray] = [q_start.copy()]
    parent: List[int] = [-1]
    arr = np.zeros((iters + 2, 2), dtype=np.float64)
    arr[0] = q_start

    for it in range(iters):
        active_goals = goals_sorted[:1] if (len(goals_sorted) > 1 and it < same_branch_warmup) else goals_sorted
        if rng.random() < goal_bias:
            if len(active_goals) > 1 and rng.random() < 0.8:
                x = active_goals[0]
            else:
                x = active_goals[int(rng.integers(0, len(active_goals)))]
        else:
            x = arm.sample_configuration(rng)

        i = int(np.argmin(np.linalg.norm(arr[: len(nodes)] - x, axis=1)))
        d = x - nodes[i]
        norm_d = float(np.linalg.norm(d))
        if norm_d < 1e-9:
            continue
        eff_step = step
        new = nodes[i] + d * min(1.0, eff_step / norm_d)
        if not edge_free(nodes[i], new, arm=arm, scene=scene):
            if len(nodes) <= 10 and norm_d > 0.04:
                new = nodes[i] + d * min(1.0, 0.04 / norm_d)
                if not edge_free(nodes[i], new, arm=arm, scene=scene):
                    continue
            else:
                continue
        if any(np.allclose(new, existing, atol=1e-12) for existing in nodes[-4:]):
            continue
        arr[len(nodes)] = new
        nodes.append(new)
        parent.append(i)
        new_idx = len(nodes) - 1

        for g in sorted(active_goals, key=lambda q: float(np.linalg.norm(q - new))):
            dist_g = float(np.linalg.norm(new - g))
            if dist_g <= 1e-9:
                nodes[new_idx] = g.copy()
                return _trace_path(nodes, parent, new_idx)
            if dist_g < step and edge_free(new, g, arm=arm, scene=scene):
                nodes.append(g.copy())
                parent.append(new_idx)
                return _trace_path(nodes, parent, len(nodes) - 1)
    return None


class _Tree:
    def __init__(self, roots: Sequence[np.ndarray], capacity: int):
        self.nodes: List[np.ndarray] = [r.copy() for r in roots]
        self.parent: List[int] = [-1] * len(roots)
        self.arr = np.zeros((capacity + len(roots) + 4, 2), dtype=np.float64)
        for idx, r in enumerate(self.nodes):
            self.arr[idx] = r

    def nearest(self, x: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.arr[: len(self.nodes)] - x, axis=1)))

    def append(self, q: np.ndarray, parent_idx: int) -> int:
        idx = len(self.nodes)
        if idx >= len(self.arr):
            grown = np.zeros((len(self.arr) * 2, 2), dtype=np.float64)
            grown[:idx] = self.arr[:idx]
            self.arr = grown
        self.arr[idx] = q
        self.nodes.append(q)
        self.parent.append(parent_idx)
        return idx


def _extend_tree(
    tree: _Tree,
    target: np.ndarray,
    step: float,
    arm: Arm2D,
    scene: BenchScene,
    *,
    allow_micro_step: bool = True,
) -> Tuple[str, int]:
    near_idx = tree.nearest(target)
    d = target - tree.nodes[near_idx]
    norm_d = float(np.linalg.norm(d))
    if norm_d < 1e-9:
        return "reached", near_idx
    if norm_d <= step:
        q_new = target.copy()
        reached = True
    else:
        q_new = tree.nodes[near_idx] + d * (step / norm_d)
        reached = False
    if not edge_free(tree.nodes[near_idx], q_new, arm=arm, scene=scene):
        if allow_micro_step and len(tree.nodes) <= 10:
            u = d / norm_d
            perp1 = np.array([-u[1], u[0]], dtype=np.float64)
            perp2 = np.array([u[1], -u[0]], dtype=np.float64)
            for cand_vec in (0.05 * u, 0.12 * perp1, 0.12 * perp2):
                q_cand = tree.nodes[near_idx] + cand_vec
                if edge_free(tree.nodes[near_idx], q_cand, arm=arm, scene=scene):
                    return "advanced", tree.append(q_cand, near_idx)
        return "trapped", near_idx
    new_idx = tree.append(q_new, near_idx)
    return ("reached" if reached else "advanced"), new_idx


def _connect_tree(
    tree: _Tree,
    target: np.ndarray,
    step: float,
    arm: Arm2D,
    scene: BenchScene,
) -> Tuple[str, int]:
    first = True
    while True:
        status, idx = _extend_tree(tree, target, step=step, arm=arm, scene=scene, allow_micro_step=first)
        first = False
        if status != "advanced":
            return status, idx


def rrt_connect(
    start: Sequence[float] | np.ndarray,
    goal: GoalInput,
    rng: np.random.Generator,
    iters: int = 3000,
    step: float = 0.20,
    *,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> Optional[List[np.ndarray]]:
    """Bidirectional RRT-Connect growing alternating trees from start and valid IK goals."""
    if start is None:
        return None
    q_start = np.asarray(start, dtype=np.float64)
    if q_start.shape != (2,) or in_collision(q_start, arm=arm, scene=scene):
        return None
    goals = _normalize_goals(goal, arm=arm, scene=scene)
    if not goals:
        return None

    direct = _direct_shortcut_goals(q_start, goals, arm=arm, scene=scene)
    if direct is not None:
        return direct

    goals_sorted = sorted(goals, key=lambda g: float(np.linalg.norm(g - q_start)))
    tree_start = _Tree([q_start], capacity=iters * 2)
    tree_goal = _Tree([goals_sorted[0]], capacity=iters * 2)
    extra_goals = goals_sorted[1:]
    a_is_start = True

    for it in range(iters):
        if it == 250 and extra_goals:
            for eg in extra_goals:
                tree_goal.append(eg.copy(), -1)
            extra_goals = []
        q_rand = arm.sample_configuration(rng)
        tree_a = tree_start if a_is_start else tree_goal
        tree_b = tree_goal if a_is_start else tree_start

        status_a, idx_a = _extend_tree(tree_a, q_rand, step=step, arm=arm, scene=scene)
        if status_a != "trapped":
            q_new = tree_a.nodes[idx_a]
            status_b, idx_b = _connect_tree(tree_b, q_new, step=step, arm=arm, scene=scene)
            if status_b == "reached":
                path_a = _trace_path(tree_a.nodes, tree_a.parent, idx_a)
                path_b = _trace_path(tree_b.nodes, tree_b.parent, idx_b)
                if a_is_start:
                    return path_a + path_b[-2::-1]
                return path_b + path_a[-2::-1]
        a_is_start = not a_is_start
    return None


def shortcut(
    path: Sequence[np.ndarray],
    rng: np.random.Generator,
    tries: int = 100,
    *,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> List[np.ndarray]:
    """Smooth joint-space path by sampling strictly non-adjacent waypoint pairs + greedy pass."""
    cleaned: List[Tuple[int, np.ndarray]] = []
    for uid, pt in enumerate(path):
        arr = np.asarray(pt, dtype=np.float64).copy()
        if not cleaned or not np.allclose(arr, cleaned[-1][1], atol=1e-12):
            cleaned.append((uid, arr))
    if len(cleaned) < 3:
        return [arr for _, arr in cleaned]

    failed_pairs: set[Tuple[int, int]] = set()
    for _ in range(tries):
        if len(cleaned) < 3:
            break
        i = int(rng.integers(0, len(cleaned) - 2))
        j = int(rng.integers(i + 2, len(cleaned)))
        key = (cleaned[i][0], cleaned[j][0])
        if key in failed_pairs:
            continue
        if edge_free(cleaned[i][1], cleaned[j][1], arm=arm, scene=scene):
            cleaned = cleaned[: i + 1] + cleaned[j:]
        else:
            failed_pairs.add(key)

    # Final deterministic greedy tightening pass
    i = 0
    while i < len(cleaned) - 2:
        for j in range(len(cleaned) - 1, i + 1, -1):
            key = (cleaned[i][0], cleaned[j][0])
            if key in failed_pairs:
                continue
            if edge_free(cleaned[i][1], cleaned[j][1], arm=arm, scene=scene):
                cleaned = cleaned[: i + 1] + cleaned[j:]
                break
            failed_pairs.add(key)
        i += 1
    return [arr for _, arr in cleaned]


def path_length(path: Sequence[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path, path[1:])))


@dataclass(frozen=True)
class WellTourResult:
    """Multi-well tour optimized via Viterbi dynamic programming over IK elbow branches."""

    wells: Tuple[str, ...]
    goal_configs: List[np.ndarray]
    elbow_branches: Tuple[int, ...]
    legs: List[List[np.ndarray]]
    waypoints: List[np.ndarray]
    total_path_length_rad: float
    elbow_flips: int


def _resolve_tour_point(
    item: WellRefInput,
    idx: int,
    *,
    plate: Optional[PlateGeometry],
    deck: Optional[DeckLayout],
    scene: BenchScene,
) -> Tuple[str, np.ndarray]:
    if isinstance(item, str):
        s = item.strip()
        if s.upper() == "TUBE":
            return "TUBE", np.asarray(scene.tube_xy, dtype=np.float64)
        if deck is not None:
            return s, deck.well_xy(s)
        if plate is not None:
            return s, plate.well_xy(s)
        if scene.deck is not None and ":" in s:
            return s, scene.deck.well_xy(s)
        return s, scene.plate.well_xy(s)
    arr = np.asarray(item, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"Tour target[{idx}] must be a well name or 2D coordinate, got {item!r}")
    return f"pt_{idx}", arr


def _tour_transition_cost(
    qa: np.ndarray,
    qb: np.ndarray,
    arm: Arm2D,
    scene: BenchScene,
) -> float:
    dist = float(np.linalg.norm(qb - qa))
    flip_penalty = math.pi if (qa[1] >= 0.0) != (qb[1] >= 0.0) else 0.0
    obs_penalty = 0.0 if edge_free(qa, qb, arm=arm, scene=scene) else 1.5
    return dist + flip_penalty + obs_penalty


def plan_well_tour(
    wells: Sequence[WellRefInput],
    rng: Optional[np.random.Generator] = None,
    *,
    start_q: Optional[Sequence[float] | np.ndarray] = None,
    plate: Optional[PlateGeometry] = None,
    deck: Optional[DeckLayout] = None,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    planner: str = "rrt_connect",
) -> Optional[WellTourResult]:
    """Plan a continuous multi-well tour using Viterbi DP over valid IK branches to avoid elbow flips."""
    if not wells:
        raise ValueError("wells must be non-empty")
    eff_rng = np.random.default_rng(0) if rng is None else rng

    q_init: Optional[np.ndarray] = None
    if start_q is not None:
        q_init = np.asarray(start_q, dtype=np.float64)
        if q_init.shape != (2,) or in_collision(q_init, arm=arm, scene=scene):
            return None

    labels: List[str] = []
    candidates: List[List[np.ndarray]] = []
    for idx, w in enumerate(wells):
        label, xy = _resolve_tour_point(w, idx, plate=plate, deck=deck, scene=scene)
        sols = ik_solutions(xy, arm=arm, scene=scene)
        if not sols:
            return None
        labels.append(label)
        candidates.append(sols)

    m = len(candidates)
    dp: List[List[float]] = []
    backptr: List[List[int]] = []

    first_costs: List[float] = []
    for q0 in candidates[0]:
        if q_init is not None:
            first_costs.append(_tour_transition_cost(q_init, q0, arm, scene))
        else:
            # Tiny deterministic tie-breaker preferring elbow=+1 only when total tour costs match
            first_costs.append(0.0 if q0[1] >= 0.0 else 1e-6)
    dp.append(first_costs)
    backptr.append([-1] * len(candidates[0]))

    for t in range(1, m):
        cur_costs: List[float] = []
        cur_back: List[int] = []
        for q_cur in candidates[t]:
            best_c = math.inf
            best_prev = 0
            for prev_idx, q_prev in enumerate(candidates[t - 1]):
                c = dp[t - 1][prev_idx] + _tour_transition_cost(q_prev, q_cur, arm, scene)
                if c < best_c:
                    best_c = c
                    best_prev = prev_idx
            cur_costs.append(best_c)
            cur_back.append(best_prev)
        dp.append(cur_costs)
        backptr.append(cur_back)

    best_last = int(np.argmin(dp[-1]))
    chosen_indices = [0] * m
    chosen_indices[-1] = best_last
    for t in range(m - 1, 0, -1):
        chosen_indices[t - 1] = backptr[t][chosen_indices[t]]

    goal_configs = [candidates[t][chosen_indices[t]].copy() for t in range(m)]
    elbow_branches = tuple(1 if q[1] >= 0.0 else -1 for q in goal_configs)
    flips = sum(1 for a, b in zip(elbow_branches, elbow_branches[1:]) if a != b)

    solve_primary = rrt_connect if planner == "rrt_connect" else rrt
    solve_backup = rrt if planner == "rrt_connect" else rrt_connect

    node_seq: List[np.ndarray] = ([q_init.copy()] if q_init is not None else []) + goal_configs
    legs: List[List[np.ndarray]] = []
    if len(node_seq) == 1:
        waypoints = [node_seq[0].copy()]
        return WellTourResult(
            wells=tuple(labels),
            goal_configs=goal_configs,
            elbow_branches=elbow_branches,
            legs=[],
            waypoints=waypoints,
            total_path_length_rad=0.0,
            elbow_flips=0,
        )

    for qa, qb in zip(node_seq, node_seq[1:]):
        if np.allclose(qa, qb, atol=1e-9):
            legs.append([qa.copy(), qb.copy()])
        elif edge_free(qa, qb, arm=arm, scene=scene):
            legs.append([qa.copy(), qb.copy()])
        else:
            raw = solve_primary(qa, [qb], eff_rng, arm=arm, scene=scene)
            if raw is None:
                raw = solve_backup(qa, [qb], eff_rng, arm=arm, scene=scene)
            if raw is None:
                return None
            sp = shortcut(raw, eff_rng, arm=arm, scene=scene)
            sp[0] = qa.copy()
            sp[-1] = qb.copy()
            legs.append(sp)

    waypoints: List[np.ndarray] = [legs[0][0].copy()]
    for leg in legs:
        for pt in leg[1:]:
            waypoints.append(pt.copy())

    return WellTourResult(
        wells=tuple(labels),
        goal_configs=goal_configs,
        elbow_branches=elbow_branches,
        legs=legs,
        waypoints=waypoints,
        total_path_length_rad=path_length(waypoints),
        elbow_flips=flips,
    )


# ---------------------------------------------------------------------------
# Full N-DoF (6-DoF UR5e / 7-DoF Panda / 2-DoF) RRT-Connect & Informed RRT*
# ---------------------------------------------------------------------------


class _TreeND:
    """Dimension-agnostic RRT tree supporting arbitrary D-DoF joint spaces."""

    def __init__(self, roots: Sequence[np.ndarray], dof: int, capacity: int):
        self.dof = int(dof)
        self.nodes: List[np.ndarray] = [np.asarray(r, dtype=np.float64).copy() for r in roots]
        self.parent: List[int] = [-1] * len(roots)
        self.arr = np.zeros((max(16, capacity + len(roots) + 4), self.dof), dtype=np.float64)
        for idx, r in enumerate(self.nodes):
            self.arr[idx] = r

    def nearest(self, x: np.ndarray) -> int:
        diffs = self.arr[: len(self.nodes)] - x[None, :]
        return int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))

    def append(self, q: np.ndarray, parent_idx: int) -> int:
        idx = len(self.nodes)
        if idx >= len(self.arr):
            grown = np.zeros((len(self.arr) * 2, self.dof), dtype=np.float64)
            grown[:idx] = self.arr[:idx]
            self.arr = grown
        self.arr[idx] = q
        self.nodes.append(q)
        self.parent.append(parent_idx)
        return idx


def _extend_tree_nd(
    tree: _TreeND,
    target: np.ndarray,
    step: float,
    edge_ok_fn,
) -> Tuple[str, int]:
    near_idx = tree.nearest(target)
    d = target - tree.nodes[near_idx]
    norm_d = float(np.linalg.norm(d))
    if norm_d < 1e-9:
        return "reached", near_idx
    if norm_d <= step:
        q_new = target.copy()
        reached = True
    else:
        q_new = tree.nodes[near_idx] + d * (step / norm_d)
        reached = False
    if not edge_ok_fn(tree.nodes[near_idx], q_new):
        return "trapped", near_idx
    new_idx = tree.append(q_new, near_idx)
    return ("reached" if reached else "advanced"), new_idx


def _connect_tree_nd(
    tree: _TreeND,
    target: np.ndarray,
    step: float,
    edge_ok_fn,
) -> Tuple[str, int]:
    while True:
        status, idx = _extend_tree_nd(tree, target, step=step, edge_ok_fn=edge_ok_fn)
        if status != "advanced":
            return status, idx


def rrt_connect_se3(
    start_q: Sequence[float] | np.ndarray,
    goal_q: Sequence[float] | np.ndarray | Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    arm: object,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    iters: int = 2000,
    step: float = 0.35,
    attached_tip_length_m: Optional[float] = None,
) -> Optional[List[np.ndarray]]:
    """Bidirectional RRT-Connect in full 6-DoF / 7-DoF SE(3) cobot joint space."""
    q_s = np.asarray(start_q, dtype=np.float64)
    dof = int(getattr(arm, "dof", q_s.size))
    if q_s.shape != (dof,):
        return None
    if in_collision_se3(q_s, arm, scene_3d, attached_tip_length_m=attached_tip_length_m):
        return None

    if isinstance(goal_q, np.ndarray):
        raw_goals = [goal_q] if goal_q.ndim == 1 else [goal_q[i] for i in range(len(goal_q))]
    elif isinstance(goal_q, (list, tuple)) and len(goal_q) > 0 and isinstance(goal_q[0], (np.ndarray, list, tuple)):
        raw_goals = [np.asarray(g, dtype=np.float64) for g in goal_q]
    else:
        raw_goals = [np.asarray(goal_q, dtype=np.float64)]

    goals = [
        g
        for g in raw_goals
        if g.shape == (dof,)
        and not in_collision_se3(g, arm, scene_3d, attached_tip_length_m=attached_tip_length_m)
    ]
    if not goals:
        return None

    def _edge_ok(qa: np.ndarray, qb: np.ndarray) -> bool:
        return edge_free_se3(qa, qb, arm, scene_3d, attached_tip_length_m=attached_tip_length_m)

    goals_sorted = sorted(goals, key=lambda g: float(np.linalg.norm(g - q_s)))
    for g in goals_sorted:
        if np.allclose(q_s, g, atol=1e-9):
            return [q_s.copy()]
        if _edge_ok(q_s, g):
            return [q_s.copy(), g.copy()]

    tree_start = _TreeND([q_s], dof=dof, capacity=iters * 2)
    tree_goal = _TreeND(goals_sorted[:1], dof=dof, capacity=iters * 2)
    extra_goals = goals_sorted[1:]
    a_is_start = True

    for it in range(iters):
        if it == 150 and extra_goals:
            for eg in extra_goals:
                tree_goal.append(eg.copy(), -1)
            extra_goals = []

        # Focused Gaussian-corridor + uniform C-space sampling for fast high-DoF narrow clearance
        if rng.random() < 0.25:
            alpha = float(rng.random())
            mid = (1.0 - alpha) * q_s + alpha * goals_sorted[0]
            q_rand = arm.clamp_joints(mid + rng.normal(0.0, 0.45, size=dof))  # type: ignore[attr-defined]
        else:
            q_rand = arm.sample_configuration(rng)  # type: ignore[attr-defined]

        tree_a = tree_start if a_is_start else tree_goal
        tree_b = tree_goal if a_is_start else tree_start

        status_a, idx_a = _extend_tree_nd(tree_a, q_rand, step=step, edge_ok_fn=_edge_ok)
        if status_a != "trapped":
            q_new = tree_a.nodes[idx_a]
            status_b, idx_b = _connect_tree_nd(tree_b, q_new, step=step, edge_ok_fn=_edge_ok)
            if status_b == "reached":
                path_a = _trace_path(tree_a.nodes, tree_a.parent, idx_a)
                path_b = _trace_path(tree_b.nodes, tree_b.parent, idx_b)
                return (path_a + path_b[-2::-1]) if a_is_start else (path_b + path_a[-2::-1])
        a_is_start = not a_is_start
    return None


def shortcut_se3(
    path: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    arm: object,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    tries: int = 80,
    attached_tip_length_m: Optional[float] = None,
) -> List[np.ndarray]:
    """Smooth an N-DoF SE(3) joint path via non-adjacent shortcutting + deterministic greedy pass."""
    cleaned: List[Tuple[int, np.ndarray]] = []
    for uid, pt in enumerate(path):
        arr = np.asarray(pt, dtype=np.float64).copy()
        if not cleaned or not np.allclose(arr, cleaned[-1][1], atol=1e-12):
            cleaned.append((uid, arr))
    if len(cleaned) < 3:
        return [arr for _, arr in cleaned]

    def _edge_ok(qa: np.ndarray, qb: np.ndarray) -> bool:
        return edge_free_se3(qa, qb, arm, scene_3d, attached_tip_length_m=attached_tip_length_m)

    failed: set[Tuple[int, int]] = set()
    for _ in range(tries):
        if len(cleaned) < 3:
            break
        i = int(rng.integers(0, len(cleaned) - 2))
        j = int(rng.integers(i + 2, len(cleaned)))
        key = (cleaned[i][0], cleaned[j][0])
        if key in failed:
            continue
        if _edge_ok(cleaned[i][1], cleaned[j][1]):
            cleaned = cleaned[: i + 1] + cleaned[j:]
        else:
            failed.add(key)

    i = 0
    while i < len(cleaned) - 2:
        for j in range(len(cleaned) - 1, i + 1, -1):
            key = (cleaned[i][0], cleaned[j][0])
            if key in failed:
                continue
            if _edge_ok(cleaned[i][1], cleaned[j][1]):
                cleaned = cleaned[: i + 1] + cleaned[j:]
                break
            failed.add(key)
        i += 1
    return [arr for _, arr in cleaned]


def _prolate_ellipsoid_rotation(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    """Compute SO(d) rotation aligning e_1 with `(q_goal - q_start) / ||q_goal - q_start||`."""
    d = q_start.size
    diff = q_goal - q_start
    c_min = float(np.linalg.norm(diff))
    if c_min <= 1e-12:
        return np.eye(d, dtype=np.float64)
    a1 = (diff / c_min)[:, None]
    e1 = np.zeros((1, d), dtype=np.float64)
    e1[0, 0] = 1.0
    M = a1 @ e1
    U, _, Vt = np.linalg.svd(M, full_matrices=True)
    diag = np.ones(d, dtype=np.float64)
    diag[-1] = float(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag(diag) @ Vt


def sample_informed_ellipsoid(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    c_best: float,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    rng: np.random.Generator,
    rot_world_from_ellipsoid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Direct sampling inside Gammell et al. (2014) admissible prolate hyperellipsoid `X_{f_hat}`."""
    d = q_start.size
    c_min = float(np.linalg.norm(q_goal - q_start))
    if not math.isfinite(c_best) or c_best <= c_min + 1e-12:
        u = float(rng.random())
        return np.clip((1.0 - u) * q_start + u * q_goal, q_lower, q_upper)

    C = (
        _prolate_ellipsoid_rotation(q_start, q_goal)
        if rot_world_from_ellipsoid is None
        else rot_world_from_ellipsoid
    )
    r1 = 0.5 * c_best
    r_perp = 0.5 * math.sqrt(max(0.0, c_best * c_best - c_min * c_min))
    radii = np.full(d, r_perp, dtype=np.float64)
    radii[0] = r1
    center = 0.5 * (q_start + q_goal)

    for _ in range(24):
        z = rng.normal(0.0, 1.0, size=d)
        nz = float(np.linalg.norm(z))
        if nz <= 1e-12:
            continue
        x_ball = (float(rng.random()) ** (1.0 / float(d))) * (z / nz)
        cand = center + C @ (radii * x_ball)
        if bool(np.all(cand >= q_lower) and np.all(cand <= q_upper)):
            return cand

    z = rng.normal(0.0, 1.0, size=d)
    nz = max(1e-12, float(np.linalg.norm(z)))
    x_ball = (float(rng.random()) ** (1.0 / float(d))) * (z / nz)
    return np.clip(center + C @ (radii * x_ball), q_lower, q_upper)


@dataclass(frozen=True)
class InformedRRTStarResult:
    """Optimal sampling-based planning result from Informed RRT* (Gammell et al., 2014)."""

    path: List[np.ndarray]
    best_cost_rad: float
    initial_cost_rad: float
    c_min_rad: float
    cost_history: Tuple[Tuple[int, float], ...]
    ellipsoid_samples: int
    total_nodes: int


def informed_rrt_star(
    start_q: Sequence[float] | np.ndarray,
    goal_q: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    arm: object = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    scene_3d: Optional[BenchScene3D] = None,
    iters: int = 600,
    step: float = 0.35,
    goal_bias: float = 0.12,
    rewire_radius: float = 0.75,
    attached_tip_length_m: Optional[float] = None,
    warm_start_connect: bool = True,
) -> Optional[InformedRRTStarResult]:
    """Asymptotically optimal Informed RRT* in 2-DoF (`Arm2D`) or 6/7-DoF (`UR5eArm`, `PandaArm`).

    Uses parent cost selection, k-nearest/ball rewiring, and direct prolate hyperellipsoidal
    sampling `||q - q_start|| + ||q_goal - q|| <= c_best` once an initial feasible path is found.
    """
    q_s = np.asarray(start_q, dtype=np.float64)
    q_g = np.asarray(goal_q, dtype=np.float64)
    is_2d = isinstance(arm, Arm2D)
    dof = 2 if is_2d else int(getattr(arm, "dof", q_s.size))
    if q_s.shape != (dof,) or q_g.shape != (dof,):
        return None

    eff_scene3d = DEFAULT_SCENE_3D if scene_3d is None else scene_3d

    if is_2d:
        q_lower = np.array([lo for lo, _ in arm.joint_limits], dtype=np.float64)
        q_upper = np.array([hi for _, hi in arm.joint_limits], dtype=np.float64)

        def _in_coll(q: np.ndarray) -> bool:
            return in_collision(q, arm=arm, scene=scene)

        def _edge_ok(qa: np.ndarray, qb: np.ndarray) -> bool:
            return edge_free(qa, qb, arm=arm, scene=scene)

    else:
        q_lower = np.asarray(arm.q_lower, dtype=np.float64)  # type: ignore[attr-defined]
        q_upper = np.asarray(arm.q_upper, dtype=np.float64)  # type: ignore[attr-defined]

        def _in_coll(q: np.ndarray) -> bool:
            return in_collision_se3(q, arm, eff_scene3d, attached_tip_length_m=attached_tip_length_m)

        def _edge_ok(qa: np.ndarray, qb: np.ndarray) -> bool:
            return edge_free_se3(qa, qb, arm, eff_scene3d, attached_tip_length_m=attached_tip_length_m)

    if _in_coll(q_s) or _in_coll(q_g):
        return None

    c_min = float(np.linalg.norm(q_g - q_s))
    if c_min <= 1e-9:
        return InformedRRTStarResult(
            path=[q_s.copy()],
            best_cost_rad=0.0,
            initial_cost_rad=0.0,
            c_min_rad=0.0,
            cost_history=((0, 0.0),),
            ellipsoid_samples=0,
            total_nodes=1,
        )

    if _edge_ok(q_s, q_g):
        return InformedRRTStarResult(
            path=[q_s.copy(), q_g.copy()],
            best_cost_rad=c_min,
            initial_cost_rad=c_min,
            c_min_rad=c_min,
            cost_history=((0, c_min),),
            ellipsoid_samples=0,
            total_nodes=2,
        )

    nodes: List[np.ndarray] = [q_s.copy()]
    parent: List[int] = [-1]
    cost: List[float] = [0.0]
    children: List[List[int]] = [[]]
    arr = np.zeros((iters + 64, dof), dtype=np.float64)
    arr[0] = q_s

    def _add_node(q_pt: np.ndarray, par_idx: int, c_val: float) -> int:
        nonlocal arr
        idx = len(nodes)
        if idx >= len(arr):
            grown = np.zeros((len(arr) * 2, dof), dtype=np.float64)
            grown[:idx] = arr[:idx]
            arr = grown
        arr[idx] = q_pt
        nodes.append(q_pt)
        parent.append(par_idx)
        cost.append(c_val)
        children.append([])
        if par_idx >= 0:
            children[par_idx].append(idx)
        return idx

    def _propagate_cost_delta(root_idx: int, delta: float) -> None:
        stack = list(children[root_idx])
        while stack:
            u = stack.pop()
            cost[u] += delta
            stack.extend(children[u])

    def _is_descendant(cand_anc: int, node_idx: int) -> bool:
        cur = node_idx
        while cur != -1:
            if cur == cand_anc:
                return True
            cur = parent[cur]
        return False

    goal_parents: List[int] = []
    c_best = math.inf
    initial_cost = math.inf
    cost_history: List[Tuple[int, float]] = []
    ellipsoid_samples = 0
    C_rot = _prolate_ellipsoid_rotation(q_s, q_g)

    # Optional fast bidirectional RRT-Connect seed to initialize tree & finite c_best immediately
    if warm_start_connect:
        seed_path = (
            rrt_connect(q_s, q_g, rng, iters=max(200, iters // 2), arm=arm, scene=scene)
            if is_2d
            else rrt_connect_se3(
                q_s,
                q_g,
                rng,
                arm=arm,
                scene_3d=eff_scene3d,
                iters=max(200, iters // 2),
                step=step,
                attached_tip_length_m=attached_tip_length_m,
            )
        )
        if seed_path is not None and len(seed_path) >= 2:
            prev_idx = 0
            for wp in seed_path[1:-1]:
                d_seg = float(np.linalg.norm(wp - nodes[prev_idx]))
                if d_seg > 1e-9 and _edge_ok(nodes[prev_idx], wp):
                    prev_idx = _add_node(wp.copy(), prev_idx, cost[prev_idx] + d_seg)
            d_last = float(np.linalg.norm(q_g - nodes[prev_idx]))
            if _edge_ok(nodes[prev_idx], q_g):
                goal_parents.append(prev_idx)
                c_best = cost[prev_idx] + d_last
                initial_cost = c_best
                cost_history.append((0, c_best))

    gamma_rrt = max(1.5, 2.2 * float(np.linalg.norm(q_upper - q_lower)) / math.sqrt(float(dof)))

    for it in range(1, iters + 1):
        if math.isfinite(c_best):
            q_rand = sample_informed_ellipsoid(q_s, q_g, c_best, q_lower, q_upper, rng, C_rot)
            ellipsoid_samples += 1
        elif rng.random() < goal_bias:
            q_rand = q_g
        else:
            q_rand = rng.uniform(q_lower, q_upper).astype(np.float64)

        n_cur = len(nodes)
        diffs = arr[:n_cur] - q_rand[None, :]
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        near_idx = int(np.argmin(dists))
        d_near = float(dists[near_idx])
        if d_near <= 1e-9:
            continue

        q_new = q_rand.copy() if d_near <= step else (nodes[near_idx] + (q_rand - nodes[near_idx]) * (step / d_near))
        # Prune immediately if q_new cannot possibly improve current c_best
        if math.isfinite(c_best):
            f_hat = float(np.linalg.norm(q_new - q_s) + np.linalg.norm(q_g - q_new))
            if f_hat >= c_best - 1e-9:
                continue
        if _in_coll(q_new):
            continue

        # Shrinking RRT* neighborhood ball
        r_ball = min(
            rewire_radius,
            gamma_rrt * ((math.log(float(n_cur + 1)) / float(n_cur + 1)) ** (1.0 / float(dof))),
        )
        r_ball = max(r_ball, step * 1.25)
        diffs_new = arr[:n_cur] - q_new[None, :]
        dists_new = np.sqrt(np.einsum("ij,ij->i", diffs_new, diffs_new))
        cand_ids = np.flatnonzero(dists_new <= r_ball)
        if near_idx not in cand_ids:
            cand_ids = np.append(cand_ids, near_idx)

        # Keep at most 6 best parent candidates sorted by cost-to-come through candidate
        cand_Order = sorted(cand_ids.tolist(), key=lambda idx: cost[idx] + float(dists_new[idx]))[:6]
        best_par = -1
        best_c = math.inf
        for cid in cand_Order:
            c_try = cost[cid] + float(dists_new[cid])
            if c_try >= best_c - 1e-12:
                continue
            if _edge_ok(nodes[cid], q_new):
                best_par = cid
                best_c = c_try
                break

        if best_par == -1:
            continue

        new_idx = _add_node(q_new, best_par, best_c)

        # Rewire neighborhood through new_idx when strictly cheaper
        for cid in cand_Order:
            if cid == best_par or cid == 0:
                continue
            d_c = float(dists_new[cid])
            new_c = best_c + d_c
            if new_c < cost[cid] - 1e-7 and not _is_descendant(cid, new_idx):
                if _edge_ok(q_new, nodes[cid]):
                    old_par = parent[cid]
                    if old_par >= 0 and cid in children[old_par]:
                        children[old_par].remove(cid)
                    parent[cid] = new_idx
                    children[new_idx].append(cid)
                    delta = new_c - cost[cid]
                    cost[cid] = new_c
                    _propagate_cost_delta(cid, delta)

        # Check direct or step-bounded goal connection
        d_goal = float(np.linalg.norm(q_g - q_new))
        if d_goal <= max(step * 1.8, rewire_radius) and (cost[new_idx] + d_goal < c_best - 1e-7):
            if _edge_ok(q_new, q_g):
                goal_parents.append(new_idx)

        # Re-evaluate best cost across all goal-connected parents (rewiring can improve earlier goal parents!)
        if goal_parents:
            cur_best_par = min(
                goal_parents,
                key=lambda pid: cost[pid] + float(np.linalg.norm(q_g - nodes[pid])),
            )
            cur_total = cost[cur_best_par] + float(np.linalg.norm(q_g - nodes[cur_best_par]))
            if cur_total < c_best - 1e-9:
                c_best = cur_total
                if not math.isfinite(initial_cost):
                    initial_cost = c_best
                cost_history.append((it, c_best))

    if not goal_parents:
        return None

    best_goal_par = min(
        goal_parents,
        key=lambda pid: cost[pid] + float(np.linalg.norm(q_g - nodes[pid])),
    )
    opt_path = _trace_path(nodes, parent, best_goal_par)
    if not np.allclose(opt_path[-1], q_g, atol=1e-9):
        opt_path.append(q_g.copy())
    final_cost = path_length(opt_path)
    return InformedRRTStarResult(
        path=opt_path,
        best_cost_rad=final_cost,
        initial_cost_rad=(initial_cost if math.isfinite(initial_cost) else final_cost),
        c_min_rad=c_min,
        cost_history=tuple(cost_history),
        ellipsoid_samples=ellipsoid_samples,
        total_nodes=len(nodes),
    )

