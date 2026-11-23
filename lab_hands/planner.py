"""Sampling-based joint-space motion planners (RRT, RRT-Connect) and shortcut smoothing."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .collision import DEFAULT_SCENE, BenchScene, edge_free, in_collision
from .kinematics import DEFAULT_ARM, Arm2D

GoalInput = Union[np.ndarray, Sequence[float], Sequence[np.ndarray]]


def ik_solutions(
    p: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> List[np.ndarray]:
    """Return all collision-free, joint-limit-valid IK configurations for target `p`."""
    out: List[np.ndarray] = []
    for elbow in (1, -1):
        q = arm.ik(p, elbow=elbow)
        if q is not None and not in_collision(q, arm=arm, scene=scene):
            if not any(np.allclose(q, prev, atol=1e-9) for prev in out):
                out.append(q)
    return out


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
    """Forward joint-space RRT supporting single or multi-branch IK goals."""
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

    nodes: List[np.ndarray] = [q_start.copy()]
    parent: List[int] = [-1]
    arr = np.zeros((iters + 2, 2), dtype=np.float64)
    arr[0] = q_start

    for _ in range(iters):
        if rng.random() < goal_bias:
            # Prefer nearest valid IK branch most of the time while still exploring all branches
            if len(goals) > 1 and rng.random() < 0.8:
                x = min(goals, key=lambda g: float(np.linalg.norm(g - q_start)))
            else:
                x = goals[int(rng.integers(0, len(goals)))]
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

        for g in sorted(goals, key=lambda q: float(np.linalg.norm(q - new))):
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
    """Bidirectional RRT-Connect growing alternating trees from start and all valid IK goals."""
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
