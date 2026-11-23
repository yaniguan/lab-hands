"""Exact vectorized 2D segment-vs-AABB clearance, collision checking, and 2.5D Z safety."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .kinematics import DEFAULT_ARM, Arm2D
from .labware import DEFAULT_PLATE_96, PlateGeometry

DEFAULT_OBSTACLES: Tuple[Tuple[float, float, float, float], ...] = (
    (0.33, -0.08, 0.36, 0.12),  # tube rack wall
    (0.15, 0.27, 0.24, 0.34),  # reagent bottle
)
DEFAULT_TUBE_XY: Tuple[float, float] = (-0.05, 0.40)


@dataclass(frozen=True)
class BenchScene:
    """Cluttered benchtop workspace with AABB obstacles and 2.5D vessel rim heights."""

    obstacles: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_OBSTACLES
    tube_xy: Tuple[float, float] = DEFAULT_TUBE_XY
    tube_radius_m: float = 0.0055
    z_tube_bottom: float = 0.010
    z_tube_rim: float = 0.075
    z_plate_rim: float = 0.015
    z_safe_travel: float = 0.095
    z_max: float = 0.180
    plate: PlateGeometry = field(default_factory=lambda: DEFAULT_PLATE_96)
    _boxes: np.ndarray = field(init=False, repr=False, compare=False)
    _corners: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for idx, (x0, y0, x1, y1) in enumerate(self.obstacles):
            if x0 > x1 or y0 > y1:
                raise ValueError(f"Invalid obstacle[{idx}] bounds ({x0}, {y0}, {x1}, {y1})")
        if self.z_safe_travel <= max(self.z_tube_rim, self.z_plate_rim):
            raise ValueError("z_safe_travel must strictly exceed both z_tube_rim and z_plate_rim")
        if self.obstacles:
            arr = np.asarray(self.obstacles, dtype=np.float64)
            xmin, ymin, xmax, ymax = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            corners = np.stack(
                [
                    np.column_stack([xmin, ymin]),
                    np.column_stack([xmin, ymax]),
                    np.column_stack([xmax, ymin]),
                    np.column_stack([xmax, ymax]),
                ],
                axis=1,
            )
        else:
            arr = np.zeros((0, 4), dtype=np.float64)
            corners = np.zeros((0, 4, 2), dtype=np.float64)
        object.__setattr__(self, "_boxes", arr)
        object.__setattr__(self, "_corners", corners)

    @property
    def obstacle_array(self) -> np.ndarray:
        return self._boxes


DEFAULT_SCENE = BenchScene()


def _liang_barsky_hit(p0: np.ndarray, p1: np.ndarray, boxes: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Fast vectorized closed-box Liang-Barsky intersection of segment(s) p0->p1 with inflated boxes."""
    xmin = boxes[:, 0] - pad
    ymin = boxes[:, 1] - pad
    xmax = boxes[:, 2] + pad
    ymax = boxes[:, 3] + pad

    dx = (p1[..., 0] - p0[..., 0])[..., None]
    dy = (p1[..., 1] - p0[..., 1])[..., None]
    px = p0[..., 0][..., None]
    py = p0[..., 1][..., None]

    with np.errstate(divide="ignore", invalid="ignore"):
        tx1 = (xmin - px) / dx
        tx2 = (xmax - px) / dx
        ty1 = (ymin - py) / dy
        ty2 = (ymax - py) / dy

        tmin_x = np.where(dx == 0.0, np.where((px >= xmin) & (px <= xmax), -np.inf, 2.0), np.minimum(tx1, tx2))
        tmax_x = np.where(dx == 0.0, np.where((px >= xmin) & (px <= xmax), np.inf, -1.0), np.maximum(tx1, tx2))
        tmin_y = np.where(dy == 0.0, np.where((py >= ymin) & (py <= ymax), -np.inf, 2.0), np.minimum(ty1, ty2))
        tmax_y = np.where(dy == 0.0, np.where((py >= ymin) & (py <= ymax), np.inf, -1.0), np.maximum(ty1, ty2))

    tmin = np.maximum(0.0, np.maximum(tmin_x, tmin_y))
    tmax = np.minimum(1.0, np.minimum(tmax_x, tmax_y))
    return tmin <= tmax


def segment_aabb_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    boxes: np.ndarray,
    *,
    _corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact 2D Euclidean distance from line segment(s) p0->p1 to closed AABB(s) [x0, y0, x1, y1]."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.full(p0.shape[:-1] + (0,), np.inf, dtype=np.float64)

    hit = _liang_barsky_hit(p0, p1, boxes, pad=0.0)
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    px0, py0 = p0[..., None, 0], p0[..., None, 1]
    dx0 = np.maximum(0.0, np.maximum(xmin - px0, px0 - xmax))
    dy0 = np.maximum(0.0, np.maximum(ymin - py0, py0 - ymax))

    px1, py1 = p1[..., None, 0], p1[..., None, 1]
    dx1 = np.maximum(0.0, np.maximum(xmin - px1, px1 - xmax))
    dy1 = np.maximum(0.0, np.maximum(ymin - py1, py1 - ymax))
    d2_endpoints = np.minimum(dx0 * dx0 + dy0 * dy0, dx1 * dx1 + dy1 * dy1)

    if _corners is None:
        corners = np.stack(
            [
                np.column_stack([xmin, ymin]),
                np.column_stack([xmin, ymax]),
                np.column_stack([xmax, ymin]),
                np.column_stack([xmax, ymax]),
            ],
            axis=1,
        )
    else:
        corners = _corners

    p0c = p0[..., None, None, :]
    dc = (p1 - p0)[..., None, None, :]
    dd = dc[..., 0] * dc[..., 0] + dc[..., 1] * dc[..., 1]
    valid = dd > 1e-24
    safe_dd = np.where(valid, dd, 1.0)
    rel = corners - p0c
    u = np.where(valid, (rel[..., 0] * dc[..., 0] + rel[..., 1] * dc[..., 1]) / safe_dd, 0.0)
    u = np.clip(u, 0.0, 1.0)
    diff = rel - u[..., None] * dc
    d2_corners = np.min(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1], axis=-1)

    return np.where(hit, 0.0, np.sqrt(np.minimum(d2_endpoints, d2_corners)))


def batch_min_clearance(
    qs: np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> np.ndarray:
    """Exact minimum Euclidean link-to-obstacle distance for each configuration in `qs` (..., 2)."""
    boxes = scene._boxes
    if boxes.shape[0] == 0:
        return np.full(qs.shape[:-1], np.inf, dtype=np.float64)
    e, t = arm.batch_fk(qs)
    o = np.zeros_like(e)
    d1 = np.min(segment_aabb_distance(o, e, boxes, _corners=scene._corners), axis=-1)
    d2 = np.min(segment_aabb_distance(e, t, boxes, _corners=scene._corners), axis=-1)
    return np.minimum(d1, d2)


def min_clearance(
    q: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
) -> float:
    arr = np.asarray(q, dtype=np.float64)
    return float(batch_min_clearance(arr[None, :], arm=arm, scene=scene)[0])


def batch_in_collision(
    qs: np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    clearance_m: Optional[float] = None,
    margin_m: Optional[float] = None,
) -> np.ndarray:
    arr = np.asarray(qs, dtype=np.float64)
    in_limits = arm.batch_within_joint_limits(arr)
    boxes = scene._boxes
    if boxes.shape[0] == 0:
        return ~in_limits

    pad = (
        (arm.clearance_m if clearance_m is None else float(clearance_m))
        + (arm.margin_m if margin_m is None else float(margin_m))
    )
    q1 = arr[..., 0]
    q12 = arr[..., 0] + arr[..., 1]
    ex = arm.l1 * np.cos(q1)
    ey = arm.l1 * np.sin(q1)
    e = np.stack([ex, ey], axis=-1)
    t = np.stack([ex + arm.l2 * np.cos(q12), ey + arm.l2 * np.sin(q12)], axis=-1)
    o = np.zeros_like(e)

    broad_hit = np.any(_liang_barsky_hit(o, e, boxes, pad=pad) | _liang_barsky_hit(e, t, boxes, pad=pad), axis=-1)
    if pad <= 0.0 or not np.any(broad_hit):
        return (~in_limits) | broad_hit

    out = ~in_limits
    flat_broad = broad_hit & in_limits
    if np.any(flat_broad):
        e_sub = e[flat_broad]
        t_sub = t[flat_broad]
        o_sub = np.zeros_like(e_sub)
        d1 = np.min(segment_aabb_distance(o_sub, e_sub, boxes, _corners=scene._corners), axis=-1)
        d2 = np.min(segment_aabb_distance(e_sub, t_sub, boxes, _corners=scene._corners), axis=-1)
        out = out.copy()
        out[flat_broad] = np.minimum(d1, d2) <= pad
    return out


def in_collision(
    q: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    clearance_m: Optional[float] = None,
    margin_m: Optional[float] = None,
) -> bool:
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"Expected joint configuration of shape (2,), got {arr.shape}")
    return bool(
        batch_in_collision(
            arr[None, :],
            arm=arm,
            scene=scene,
            clearance_m=clearance_m,
            margin_m=margin_m,
        )[0]
    )


def edge_subdivision_count(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    *,
    step: float = 0.02,
    max_tip_step_m: float = 0.002,
) -> int:
    """Bound subdivision count by both joint norm step and Jacobian Lipschitz tip displacement."""
    if step <= 0.0 or max_tip_step_m <= 0.0:
        raise ValueError("step and max_tip_step_m must be > 0")
    dq0 = abs(float(b[0]) - float(a[0]))
    dq1 = abs(float(b[1]) - float(a[1]))
    n_joint = int(math.ceil(math.hypot(dq0, dq1) / step)) + 1
    tip_sweep_upper = (arm.l1 + arm.l2) * dq0 + arm.l2 * dq1
    n_tip = int(math.ceil(tip_sweep_upper / max_tip_step_m)) + 1
    return max(2, n_joint, n_tip)


_COARSE_S = np.array([[0.0], [0.5], [1.0], [0.25], [0.75]], dtype=np.float64)


def edge_free(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    step: float = 0.02,
    max_tip_step_m: float = 0.002,
    clearance_m: Optional[float] = None,
    margin_m: Optional[float] = None,
) -> bool:
    """Vectorized continuous-step collision check along joint-space segment a -> b."""
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    if qa.shape != (2,) or qb.shape != (2,):
        raise ValueError("edge_free endpoints must have shape (2,)")
    if not (np.isfinite(qa).all() and np.isfinite(qb).all()):
        return False
    n = edge_subdivision_count(qa, qb, arm=arm, step=step, max_tip_step_m=max_tip_step_m)
    diff = qb - qa
    # Fast 5-point coarse reject for long chords before evaluating dense grid
    if n > 16:
        q_coarse = qa[None, :] + _COARSE_S * diff[None, :]
        if bool(
            np.any(
                batch_in_collision(
                    q_coarse,
                    arm=arm,
                    scene=scene,
                    clearance_m=clearance_m,
                    margin_m=margin_m,
                )
            )
        ):
            return False
    s = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    qs = qa[None, :] + s * diff[None, :]
    return not bool(
        np.any(
            batch_in_collision(
                qs,
                arm=arm,
                scene=scene,
                clearance_m=clearance_m,
                margin_m=margin_m,
            )
        )
    )


def path_min_clearance(
    path: Sequence[np.ndarray],
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    step: float = 0.02,
    max_tip_step_m: float = 0.002,
) -> float:
    """Minimum link-to-obstacle Euclidean clearance along a piecewise-linear joint path."""
    if not path:
        raise ValueError("path must be non-empty")
    if len(path) == 1:
        return min_clearance(path[0], arm=arm, scene=scene)
    best = math.inf
    for a, b in zip(path, path[1:]):
        qa = np.asarray(a, dtype=np.float64)
        qb = np.asarray(b, dtype=np.float64)
        n = edge_subdivision_count(qa, qb, arm=arm, step=step, max_tip_step_m=max_tip_step_m)
        s = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
        qs = qa[None, :] + s * (qb - qa)[None, :]
        best = min(best, float(np.min(batch_min_clearance(qs, arm=arm, scene=scene))))
    return best


def in_collision_3d(
    state_3d: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    allowed_well_xy: Optional[np.ndarray] = None,
    z_margin_m: float = 0.003,
) -> bool:
    """2.5D SCARA collision check on state (q1, q2, z) enforcing vessel rim clearances."""
    arr = np.asarray(state_3d, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return True
    q = arr[:2]
    z = float(arr[2])
    if z < 0.0 or z > scene.z_max:
        return True
    if in_collision(q, arm=arm, scene=scene):
        return True
    if z >= scene.z_safe_travel - 1e-12:
        return False

    _, tip = arm.fk(q)
    tube_pt = np.asarray(scene.tube_xy, dtype=np.float64)
    if float(np.linalg.norm(tip - tube_pt)) <= scene.tube_radius_m and z >= scene.z_tube_bottom - 1e-9:
        return False
    if allowed_well_xy is not None:
        well_pt = np.asarray(allowed_well_xy, dtype=np.float64)
        if float(np.linalg.norm(tip - well_pt)) <= scene.plate.well_radius_m and z >= scene.plate.z_bottom_m - 1e-9:
            return False

    if z < scene.z_tube_rim + z_margin_m and float(np.linalg.norm(tip - tube_pt)) <= 0.03:
        return True
    if z < scene.z_plate_rim + z_margin_m:
        ox, oy = scene.plate.origin_xy
        px_min = ox - 0.015
        px_max = ox + (scene.plate.cols - 1) * scene.plate.pitch_x + 0.015
        py_min = oy - (scene.plate.rows - 1) * scene.plate.pitch_y - 0.015
        py_max = oy + 0.015
        if px_min <= tip[0] <= px_max and py_min <= tip[1] <= py_max:
            return True
    return True


def edge_free_3d(
    a_3d: Sequence[float] | np.ndarray,
    b_3d: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    allowed_well_xy: Optional[np.ndarray] = None,
) -> bool:
    """Verify 3-DoF SCARA segment (q1, q2, z) avoids planar obstacles and rim tip-shearing."""
    sa = np.asarray(a_3d, dtype=np.float64)
    sb = np.asarray(b_3d, dtype=np.float64)
    if sa.shape != (3,) or sb.shape != (3,):
        raise ValueError("edge_free_3d endpoints must have shape (3,)")
    dq_norm = float(np.linalg.norm(sb[:2] - sa[:2]))
    min_z = min(float(sa[2]), float(sb[2]))
    if dq_norm > 1e-9 and min_z < scene.z_safe_travel - 1e-9:
        return False
    n_xy = edge_subdivision_count(sa[:2], sb[:2], arm=arm)
    n_z = int(math.ceil(abs(float(sb[2] - sa[2])) / 0.002)) + 1
    n = max(2, n_xy, n_z)
    for alpha in np.linspace(0.0, 1.0, n):
        st = sa + float(alpha) * (sb - sa)
        if in_collision_3d(st, arm=arm, scene=scene, allowed_well_xy=allowed_well_xy):
            return False
    return True
