"""Exact vectorized 2D capsule-vs-AABB clearance, Lipschitz swept-segment checks, and 3D Z safety."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import DEFAULT_ARM, Arm2D
from .labware import DEFAULT_PLATE_96, DeckLayout, PlateGeometry

DEFAULT_OBSTACLES: Tuple[Tuple[float, float, float, float], ...] = (
    (0.33, -0.08, 0.36, 0.12),  # tube rack wall
    (0.15, 0.27, 0.24, 0.34),  # reagent bottle
)
DEFAULT_TUBE_XY: Tuple[float, float] = (-0.05, 0.40)


@dataclass(frozen=True)
class BenchScene:
    """Cluttered benchtop workspace with AABB obstacles, per-labware Z rims, and mounted tip length."""

    obstacles: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_OBSTACLES
    tube_xy: Tuple[float, float] = DEFAULT_TUBE_XY
    tube_radius_m: float = 0.0055
    z_tube_bottom: float = 0.010
    z_tube_rim: float = 0.075
    z_plate_rim: float = 0.015
    z_safe_travel: Optional[float] = None
    z_max: float = 0.300
    z_margin_m: float = 0.005
    attached_tip_length_m: float = 0.0
    plate: PlateGeometry = field(default_factory=lambda: DEFAULT_PLATE_96)
    deck: Optional[DeckLayout] = None
    _boxes: np.ndarray = field(init=False, repr=False, compare=False)
    _corners: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for idx, (x0, y0, x1, y1) in enumerate(self.obstacles):
            if x0 > x1 or y0 > y1:
                raise ValueError(f"Invalid obstacle[{idx}] bounds ({x0}, {y0}, {x1}, {y1})")
        if self.attached_tip_length_m < 0.0:
            raise ValueError(f"attached_tip_length_m must be >= 0, got {self.attached_tip_length_m}")
        if self.z_margin_m < 0.0:
            raise ValueError(f"z_margin_m must be >= 0, got {self.z_margin_m}")

        eff_plate_rim = max(float(self.z_plate_rim), float(self.plate.z_rim_m))
        object.__setattr__(self, "z_plate_rim", eff_plate_rim)

        deck_rim = self.deck.max_rim_m if self.deck is not None else 0.0
        max_rim = max(float(self.z_tube_rim), eff_plate_rim, deck_rim)
        min_safe_flange = max_rim + float(self.attached_tip_length_m) + float(self.z_margin_m)

        if self.z_safe_travel is None:
            resolved_safe = max(0.095, round(min_safe_flange + 0.005, 6))
            object.__setattr__(self, "z_safe_travel", resolved_safe)
        else:
            explicit_safe = float(self.z_safe_travel)
            if explicit_safe <= max_rim or explicit_safe < min_safe_flange - 1e-12:
                raise ValueError(
                    f"z_safe_travel ({explicit_safe:.4f} m) must satisfy >= max_rim + attached_tip_length_m "
                    f"+ z_margin_m ({min_safe_flange:.4f} m)"
                )
            object.__setattr__(self, "z_safe_travel", explicit_safe)

        if float(self.z_safe_travel) > float(self.z_max) + 1e-12:  # type: ignore[arg-type]
            raise ValueError(
                f"z_safe_travel ({self.z_safe_travel} m) exceeds z_max ({self.z_max} m)"
            )

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

    @property
    def max_vessel_rim_m(self) -> float:
        deck_rim = self.deck.max_rim_m if self.deck is not None else 0.0
        return max(float(self.z_tube_rim), float(self.z_plate_rim), float(self.plate.z_rim_m), deck_rim)

    def with_attached_tip(
        self,
        attached_tip_length_m: float,
        *,
        plate: Optional[PlateGeometry] = None,
        deck: Optional[DeckLayout] = None,
    ) -> "BenchScene":
        """Return a copy of this scene updated for a newly mounted/ejected pipette tip or deck."""
        eff_plate = self.plate if plate is None else plate
        eff_deck = self.deck if deck is None else deck
        return BenchScene(
            obstacles=self.obstacles,
            tube_xy=self.tube_xy,
            tube_radius_m=self.tube_radius_m,
            z_tube_bottom=self.z_tube_bottom,
            z_tube_rim=self.z_tube_rim,
            z_plate_rim=max(float(self.z_plate_rim), float(eff_plate.z_rim_m)),
            z_safe_travel=None,
            z_max=max(float(self.z_max), 0.300),
            z_margin_m=self.z_margin_m,
            attached_tip_length_m=float(attached_tip_length_m),
            plate=eff_plate,
            deck=eff_deck,
        )


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


def segment_aabb_clearance(
    p0: np.ndarray,
    p1: np.ndarray,
    boxes: np.ndarray,
    capsule_radius: float = 0.0,
    *,
    _corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact 2D surface-to-AABB Euclidean clearance for closed link capsule(s) of radius `capsule_radius`."""
    if capsule_radius < 0.0:
        raise ValueError(f"capsule_radius must be >= 0, got {capsule_radius}")
    center_dist = segment_aabb_distance(p0, p1, boxes, _corners=_corners)
    if capsule_radius == 0.0:
        return center_dist
    return np.maximum(0.0, center_dist - float(capsule_radius))


def batch_min_clearance(
    qs: np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    clearance_m: Optional[float] = None,
) -> np.ndarray:
    """Exact minimum capsule-surface-to-obstacle clearance for each configuration in `qs` (..., 2)."""
    boxes = scene._boxes
    qs_arr = np.asarray(qs, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.full(qs_arr.shape[:-1], np.inf, dtype=np.float64)
    cap_r = arm.clearance_m if clearance_m is None else float(clearance_m)
    e, t = arm.batch_fk(qs_arr)
    o = np.zeros_like(e)
    d1 = np.min(segment_aabb_clearance(o, e, boxes, capsule_radius=cap_r, _corners=scene._corners), axis=-1)
    d2 = np.min(segment_aabb_clearance(e, t, boxes, capsule_radius=cap_r, _corners=scene._corners), axis=-1)
    return np.minimum(d1, d2)


def min_clearance(
    q: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    clearance_m: Optional[float] = None,
) -> float:
    arr = np.asarray(q, dtype=np.float64)
    return float(batch_min_clearance(arr[None, :], arm=arm, scene=scene, clearance_m=clearance_m)[0])


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

    cap_r = arm.clearance_m if clearance_m is None else float(clearance_m)
    marg = arm.margin_m if margin_m is None else float(margin_m)
    pad = cap_r + marg
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


def kinematic_tip_sweep_bound(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
) -> float:
    """Exact kinematic Lipschitz upper bound (L1 + L2)*|dq1| + L2*|dq2| on link displacement (m)."""
    dq0 = abs(float(b[0]) - float(a[0]))
    dq1 = abs(float(b[1]) - float(a[1]))
    return float((arm.l1 + arm.l2) * dq0 + arm.l2 * dq1)


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
    tip_sweep_upper = kinematic_tip_sweep_bound(a, b, arm=arm)
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
    """Continuous Lipschitz-certified collision check along joint-space segment a -> b."""
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    if qa.shape != (2,) or qb.shape != (2,):
        raise ValueError("edge_free endpoints must have shape (2,)")
    if not (np.isfinite(qa).all() and np.isfinite(qb).all()):
        return False
    if not (arm.within_joint_limits(qa) and arm.within_joint_limits(qb)):
        return False

    cap_r = arm.clearance_m if clearance_m is None else float(clearance_m)
    base_margin = arm.margin_m if margin_m is None else float(margin_m)

    # Exact zero-discretization check at endpoints s=0 and s=1
    q_ends = np.stack([qa, qb], axis=0)
    if bool(np.any(batch_in_collision(q_ends, arm=arm, scene=scene, clearance_m=cap_r, margin_m=base_margin))):
        return False
    if scene._boxes.shape[0] == 0:
        return True

    sweep = kinematic_tip_sweep_bound(qa, qb, arm=arm)
    if sweep <= 1e-15:
        return True

    n = edge_subdivision_count(qa, qb, arm=arm, step=step, max_tip_step_m=max_tip_step_m)
    diff = qb - qa
    if n > 16:
        q_coarse = qa[None, :] + _COARSE_S * diff[None, :]
        if bool(
            np.any(
                batch_in_collision(
                    q_coarse,
                    arm=arm,
                    scene=scene,
                    clearance_m=cap_r,
                    margin_m=base_margin,
                )
            )
        ):
            return False

    s = np.linspace(0.0, 1.0, n, dtype=np.float64)
    qs = qa[None, :] + s[:, None] * diff[None, :]
    half_step_m = 0.5 * sweep / float(n - 1)

    # Sound fast-accept via LabHandsLean Lipschitz swept-segment margin inflation
    hits_inflated = batch_in_collision(
        qs,
        arm=arm,
        scene=scene,
        clearance_m=cap_r,
        margin_m=base_margin + half_step_m,
    )
    if not bool(np.any(hits_inflated)):
        return True

    # Fast-reject if any grid sample already violates base_margin
    flagged_qs = qs[hits_inflated]
    if bool(np.any(batch_in_collision(flagged_qs, arm=arm, scene=scene, clearance_m=cap_r, margin_m=base_margin))):
        return False

    # Adaptive Lipschitz midpoint refinement on only the narrow-band sub-intervals
    active_mask = hits_inflated[:-1] | hits_inflated[1:]
    s_lo = s[:-1][active_mask]
    s_hi = s[1:][active_mask]

    for _ in range(14):
        if s_lo.size == 0:
            return True
        s_mid = 0.5 * (s_lo + s_hi)
        rad = 0.5 * (s_hi - s_lo) * sweep
        q_mid = qa[None, :] + s_mid[:, None] * diff[None, :]
        cl_mid = batch_min_clearance(q_mid, arm=arm, scene=scene, clearance_m=cap_r)
        if bool(np.any(cl_mid <= base_margin)):
            return False
        uncertain = cl_mid <= (base_margin + rad)
        if not bool(np.any(uncertain)):
            return True
        lo_u = s_lo[uncertain]
        mid_u = s_mid[uncertain]
        hi_u = s_hi[uncertain]
        s_lo = np.concatenate([lo_u, mid_u])
        s_hi = np.concatenate([mid_u, hi_u])

    # Conservative fail-closed guarantee if any sub-interval remains within Lipschitz uncertainty
    return False


def path_min_clearance(
    path: Sequence[np.ndarray],
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    step: float = 0.02,
    max_tip_step_m: float = 0.002,
    clearance_m: Optional[float] = None,
) -> float:
    """Certified continuous capsule-to-obstacle Euclidean clearance along a piecewise-linear joint path."""
    if path is None or len(path) == 0:
        raise ValueError("path must be non-empty")
    if step <= 0.0 or max_tip_step_m <= 0.0:
        raise ValueError("step and max_tip_step_m must be > 0")
    if scene._boxes.shape[0] == 0:
        return math.inf

    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        pts = np.stack([np.asarray(p, dtype=np.float64) for p in path], axis=0)
    cap_r = arm.clearance_m if clearance_m is None else float(clearance_m)
    end_cl = batch_min_clearance(pts, arm=arm, scene=scene, clearance_m=cap_r)
    best_ub = float(np.min(end_cl))
    if pts.shape[0] == 1 or best_ub <= 0.0:
        return max(0.0, best_ub)

    qa_all = pts[:-1]
    diff_all = pts[1:] - qa_all
    dq_all = np.abs(diff_all)
    sweeps = (arm.l1 + arm.l2) * dq_all[:, 0] + arm.l2 * dq_all[:, 1]
    active = sweeps > 1e-15
    if not bool(np.any(active)):
        return max(0.0, best_ub)

    qa_act = qa_all[active]
    diff_act = diff_all[active]
    sweeps_act = sweeps[active]
    dq_act = dq_all[active]

    n_joint = np.ceil(np.hypot(dq_act[:, 0], dq_act[:, 1]) / step).astype(np.int64)
    n_tip = np.ceil(sweeps_act / max_tip_step_m).astype(np.int64)
    m_subs = np.maximum(1, np.maximum(n_joint, n_tip))

    seg_ids = np.repeat(np.arange(m_subs.size, dtype=np.int64), m_subs)
    offsets = np.concatenate([[0], np.cumsum(m_subs[:-1])])
    local_k = np.arange(seg_ids.size, dtype=np.float64) - np.repeat(offsets, m_subs)
    inv_m = 1.0 / np.repeat(m_subs.astype(np.float64), m_subs)
    s_lo = local_k * inv_m
    s_hi = (local_k + 1.0) * inv_m

    certified_floor = best_ub
    for _ in range(8):
        if s_lo.size == 0:
            break
        s_mid = 0.5 * (s_lo + s_hi)
        rad = 0.5 * (s_hi - s_lo) * sweeps_act[seg_ids]
        q_mid = qa_act[seg_ids] + s_mid[:, None] * diff_act[seg_ids]
        cl_mid = batch_min_clearance(q_mid, arm=arm, scene=scene, clearance_m=cap_r)
        mid_min = float(np.min(cl_mid))
        if mid_min < best_ub:
            best_ub = mid_min
        if best_ub <= 0.0:
            return 0.0

        lb = cl_mid - rad
        safe = lb >= (best_ub - 1e-6)
        if bool(np.any(safe)):
            certified_floor = min(certified_floor, float(np.min(lb[safe])))
        refine = ~safe
        if not bool(np.any(refine)):
            s_lo = np.empty(0, dtype=np.float64)
            break
        seg_r = seg_ids[refine]
        lo_r = s_lo[refine]
        mid_r = s_mid[refine]
        hi_r = s_hi[refine]
        seg_ids = np.concatenate([seg_r, seg_r])
        s_lo = np.concatenate([lo_r, mid_r])
        s_hi = np.concatenate([mid_r, hi_r])

    if s_lo.size > 0:
        s_mid = 0.5 * (s_lo + s_hi)
        rad = 0.5 * (s_hi - s_lo) * sweeps_act[seg_ids]
        q_mid = qa_act[seg_ids] + s_mid[:, None] * diff_act[seg_ids]
        cl_mid = batch_min_clearance(q_mid, arm=arm, scene=scene, clearance_m=cap_r)
        certified_floor = min(certified_floor, float(np.min(cl_mid - rad)))

    return max(0.0, min(best_ub, certified_floor))


def _collect_allowed_wells(
    allowed_well_xy: Optional[np.ndarray | Sequence[float]],
    allowed_wells_xy: Optional[Iterable[Sequence[float] | np.ndarray]],
) -> List[np.ndarray]:
    pts: List[np.ndarray] = []
    if allowed_well_xy is not None:
        pts.append(np.asarray(allowed_well_xy, dtype=np.float64))
    if allowed_wells_xy is not None:
        for w in allowed_wells_xy:
            pts.append(np.asarray(w, dtype=np.float64))
    return pts


def in_collision_3d(
    state_3d: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    allowed_well_xy: Optional[np.ndarray | Sequence[float]] = None,
    allowed_wells_xy: Optional[Iterable[Sequence[float] | np.ndarray]] = None,
    z_margin_m: float = 0.003,
) -> bool:
    """2.5D/3D collision check on (q1, q2, z_flange) enforcing per-labware rims and attached tip length."""
    arr = np.asarray(state_3d, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return True
    q = arr[:2]
    z_flange = float(arr[2])
    z_tip = z_flange - float(scene.attached_tip_length_m)
    if z_tip < -1e-12 or z_flange > float(scene.z_max) + 1e-12:
        return True
    if in_collision(q, arm=arm, scene=scene):
        return True

    _, tip = arm.fk(q)
    eff_margin = max(float(z_margin_m), 0.0)
    allowed_pts = _collect_allowed_wells(allowed_well_xy, allowed_wells_xy)

    # 1. Source tube bore and collar check
    tube_pt = np.asarray(scene.tube_xy, dtype=np.float64)
    dist_tube = float(np.linalg.norm(tip - tube_pt))
    if dist_tube <= scene.tube_radius_m:
        if z_tip < scene.z_tube_bottom - 1e-9:
            return True
        return False
    if dist_tube <= 0.03 and z_tip < scene.z_tube_rim + eff_margin - 1e-12:
        return True

    # 2. Check primary plate + all multi-labware deck footprints using true per-labware z_rim_m
    labware_items: List[Tuple[PlateGeometry, float]] = [
        (scene.plate, max(float(scene.z_plate_rim), float(scene.plate.z_rim_m)))
    ]
    if scene.deck is not None:
        for geom in scene.deck.labware_by_alias.values():
            labware_items.append((geom, float(geom.z_rim_m)))

    for geom, rim_z in labware_items:
        px_min, px_max, py_min, py_max = geom.footprint_bounds()
        if px_min <= float(tip[0]) <= px_max and py_min <= float(tip[1]) <= py_max:
            in_allowed_bore = False
            for w_pt in allowed_pts:
                if (
                    px_min <= float(w_pt[0]) <= px_max
                    and py_min <= float(w_pt[1]) <= py_max
                    and float(np.linalg.norm(tip - w_pt)) <= geom.well_radius_m + 1e-9
                ):
                    if z_tip < geom.z_bottom_m - 1e-9:
                        return True
                    in_allowed_bore = True
                    break
            if not in_allowed_bore and z_tip < rim_z + eff_margin - 1e-12:
                return True

    # 3. Free space outside tube and labware footprints is collision-free for 0 <= z_tip and z_flange <= z_max
    return False


def edge_free_3d(
    a_3d: Sequence[float] | np.ndarray,
    b_3d: Sequence[float] | np.ndarray,
    arm: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    *,
    allowed_well_xy: Optional[np.ndarray | Sequence[float]] = None,
    allowed_wells_xy: Optional[Iterable[Sequence[float] | np.ndarray]] = None,
    z_margin_m: float = 0.003,
) -> bool:
    """Verify 3D segment (q1, q2, z_flange) avoids planar obstacles and vessel/tiprack rim shearing."""
    sa = np.asarray(a_3d, dtype=np.float64)
    sb = np.asarray(b_3d, dtype=np.float64)
    if sa.shape != (3,) or sb.shape != (3,):
        raise ValueError("edge_free_3d endpoints must have shape (3,)")
    dq_norm = float(np.linalg.norm(sb[:2] - sa[:2]))
    min_z_flange = min(float(sa[2]), float(sb[2]))
    req_safe_flange = max(
        float(scene.z_safe_travel),  # type: ignore[arg-type]
        scene.max_vessel_rim_m + float(scene.attached_tip_length_m) + max(float(z_margin_m), float(scene.z_margin_m)),
    )
    if dq_norm > 1e-9 and min_z_flange < req_safe_flange - 1e-9:
        return False
    if dq_norm > 1e-9 and not edge_free(sa[:2], sb[:2], arm=arm, scene=scene):
        return False
    n_xy = edge_subdivision_count(sa[:2], sb[:2], arm=arm)
    n_z = int(math.ceil(abs(float(sb[2] - sa[2])) / 0.002)) + 1
    n = max(2, n_xy, n_z)
    for alpha in np.linspace(0.0, 1.0, n):
        st = sa + float(alpha) * (sb - sa)
        if in_collision_3d(
            st,
            arm=arm,
            scene=scene,
            allowed_well_xy=allowed_well_xy,
            allowed_wells_xy=allowed_wells_xy,
            z_margin_m=z_margin_m,
        ):
            return False
    return True
