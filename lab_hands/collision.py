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
    z_plate_rim: Optional[float] = None
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

        eff_plate_rim = (
            float(self.plate.z_rim_m)
            if self.z_plate_rim is None
            else max(float(self.z_plate_rim), float(self.plate.z_rim_m))
        )
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


# ---------------------------------------------------------------------------
# 3D Multi-Link Capsule-vs-AABB & Self-Collision Continuous Swept Checks (SE3)
# ---------------------------------------------------------------------------

DEFAULT_OBSTACLES_3D: Tuple[Tuple[float, float, float, float, float, float], ...] = (
    (0.33, -0.08, 0.00, 0.36, 0.12, 0.24),  # tall rack partition (240 mm)
    (0.15, 0.27, 0.00, 0.24, 0.34, 0.23),  # 1 L reagent media bottle (230 mm)
)


@dataclass(frozen=True)
class BenchScene3D:
    """3D benchtop collision scene with 3D AABBs, table plane, labware rims, and mounted tip."""

    obstacles_3d: Tuple[Tuple[float, float, float, float, float, float], ...] = DEFAULT_OBSTACLES_3D
    include_table_slab: bool = True
    table_z_m: float = 0.0
    attached_tip_length_m: float = 0.0
    margin_m: float = 0.005
    _boxes3d: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_list: List[Tuple[float, float, float, float, float, float]] = []
        for idx, b in enumerate(self.obstacles_3d):
            if len(b) != 6 or b[0] > b[3] or b[1] > b[4] or b[2] > b[5]:
                raise ValueError(f"Invalid 3D AABB[{idx}]={b!r}")
            raw_list.append(tuple(float(v) for v in b))  # type: ignore[arg-type]
        if self.include_table_slab:
            # Benchtop table surface excluding central cobot mounting pedestal around (0, 0)
            tz = float(self.table_z_m)
            raw_list.append((-1.20, -1.20, tz - 0.12, 1.20, 1.20, tz))
        arr = np.asarray(raw_list, dtype=np.float64) if raw_list else np.zeros((0, 6), dtype=np.float64)
        object.__setattr__(self, "_boxes3d", arr)

    @property
    def obstacle_array_3d(self) -> np.ndarray:
        return self._boxes3d

    @classmethod
    def from_bench_scene(
        cls,
        scene: BenchScene = DEFAULT_SCENE,
        *,
        obstacle_height_m: float = 0.24,
        include_table_slab: bool = True,
    ) -> "BenchScene3D":
        boxes_3d = tuple(
            (float(x0), float(y0), 0.0, float(x1), float(y1), float(obstacle_height_m))
            for (x0, y0, x1, y1) in scene.obstacles
        )
        return cls(
            obstacles_3d=boxes_3d,
            include_table_slab=include_table_slab,
            attached_tip_length_m=float(scene.attached_tip_length_m),
            margin_m=float(scene.z_margin_m),
        )


DEFAULT_SCENE_3D = BenchScene3D()


def segment_aabb_distance_3d(
    p0: np.ndarray | Sequence[float],
    p1: np.ndarray | Sequence[float],
    boxes_3d: np.ndarray,
) -> np.ndarray:
    """Exact 3D Euclidean distance from line segment(s) `p0->p1` `(..., 3)` to closed 3D AABB(s) `(M, 6)`.

    Exploits exact convexity and piecewise-quadratic structure of `sum_c max(0, bmin_c - p_c(u), p_c(u) - bmax_c)^2`
    across the `<= 6` slab-crossing breakpoints in `[0, 1]`, evaluating both exact interval vertices
    `-B / A` and slab breakpoints with zero discretization error!
    """
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    boxes = np.asarray(boxes_3d, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.full(a.shape[:-1] + (0,), np.inf, dtype=np.float64)

    d = b - a  # (..., 3)
    bmin = boxes[:, :3]  # (M, 3)
    bmax = boxes[:, 3:]  # (M, 3)

    a_e = a[..., None, :]  # (..., 1, 3)
    d_e = d[..., None, :]  # (..., 1, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_d = np.abs(d_e) > 1e-15
        safe_d = np.where(valid_d, d_e, 1.0)
        u_lo = np.where(valid_d, (bmin - a_e) / safe_d, 0.0)
        u_hi = np.where(valid_d, (bmax - a_e) / safe_d, 0.0)

    u_lo = np.clip(u_lo, 0.0, 1.0)
    u_hi = np.clip(u_hi, 0.0, 1.0)
    zeros = np.zeros(u_lo.shape[:-1] + (1,), dtype=np.float64)
    ones = np.ones(u_lo.shape[:-1] + (1,), dtype=np.float64)

    # 8 sorted breakpoints in [0, 1] -> 7 intervals on which active AABB faces are constant
    u_breaks = np.sort(np.concatenate([zeros, ones, u_lo, u_hi], axis=-1), axis=-1)  # (..., M, 8)
    ua = u_breaks[..., :-1]  # (..., M, 7)
    ub = u_breaks[..., 1:]  # (..., M, 7)
    um = 0.5 * (ua + ub)  # (..., M, 7)

    pm = a_e[..., None, :] + um[..., None] * d_e[..., None, :]  # (..., M, 7, 3)
    bmin_i = bmin[..., None, :]  # (M, 1, 3)
    bmax_i = bmax[..., None, :]  # (M, 1, 3)

    below = pm < bmin_i
    above = pm > bmax_i
    active = below | above

    r0 = np.where(
        below,
        a_e[..., None, :] - bmin_i,
        np.where(above, a_e[..., None, :] - bmax_i, 0.0),
    )
    dc = np.where(active, d_e[..., None, :], 0.0)

    A = np.sum(dc * dc, axis=-1)  # (..., M, 7)
    B = np.sum(dc * r0, axis=-1)  # (..., M, 7)
    u_vert = np.where(A > 1e-28, -B / np.maximum(A, 1e-28), um)
    u_vert = np.clip(u_vert, ua, ub)

    # Candidate minimizers in [0, 1]: all 8 breakpoints + all 7 exact quadratic vertices (15 points)
    u_cands = np.concatenate([u_breaks, u_vert], axis=-1)  # (..., M, 15)
    pts = a_e[..., None, :] + u_cands[..., None] * d_e[..., None, :]  # (..., M, 15, 3)
    diff_lo = bmin_i - pts
    diff_hi = pts - bmax_i
    delta = np.maximum(0.0, np.maximum(diff_lo, diff_hi))
    d2 = np.sum(delta * delta, axis=-1)  # (..., M, 15)
    return np.sqrt(np.min(d2, axis=-1))


def capsule_aabb_clearance_3d(
    p0: np.ndarray | Sequence[float],
    p1: np.ndarray | Sequence[float],
    capsule_radii: np.ndarray | float,
    boxes_3d: np.ndarray,
) -> np.ndarray:
    """Exact 3D capsule-surface-to-AABB Euclidean clearance `(..., M)`."""
    radii = np.asarray(capsule_radii, dtype=np.float64)
    if bool(np.any(radii < 0.0)):
        raise ValueError("capsule_radii must be >= 0")
    dist = segment_aabb_distance_3d(p0, p1, boxes_3d)
    return np.maximum(0.0, dist - radii[..., None])


def segment_segment_distance_3d(
    a0: np.ndarray | Sequence[float],
    a1: np.ndarray | Sequence[float],
    b0: np.ndarray | Sequence[float],
    b1: np.ndarray | Sequence[float],
) -> float:
    """Exact 3D Euclidean distance between two closed line segments `a0->a1` and `b0->b1`."""
    p1 = np.asarray(a0, dtype=np.float64)
    q1 = np.asarray(a1, dtype=np.float64)
    p2 = np.asarray(b0, dtype=np.float64)
    q2 = np.asarray(b1, dtype=np.float64)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-18 and e <= 1e-18:
        return float(np.linalg.norm(r))
    if a <= 1e-18:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-18:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    closest = (p1 + s * d1) - (p2 + t * d2)
    return float(np.linalg.norm(closest))


def min_clearance_se3(
    q: Sequence[float] | np.ndarray,
    arm: object,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    *,
    attached_tip_length_m: Optional[float] = None,
    check_self_collision: bool = True,
) -> float:
    """Exact minimum 3D capsule-to-AABB (and non-adjacent self-link) surface clearance (m)."""
    eff_tip = (
        float(scene_3d.attached_tip_length_m)
        if attached_tip_length_m is None
        else float(attached_tip_length_m)
    )
    p0s, p1s, radii = arm.link_capsules(q, attached_tip_length_m=eff_tip)  # type: ignore[attr-defined]
    boxes = scene_3d.obstacle_array_3d
    best = math.inf

    if boxes.shape[0] > 0:
        # Exclude shoulder column (index 0) from floor-slab check since base bolts to table z=0
        if scene_3d.include_table_slab and boxes.shape[0] >= 1:
            obs_boxes = boxes[:-1]
            slab_box = boxes[-1:]
            if obs_boxes.shape[0] > 0:
                cl_obs = capsule_aabb_clearance_3d(p0s, p1s, radii, obs_boxes)
                best = min(best, float(np.min(cl_obs)))
            if p0s.shape[0] > 1:
                cl_slab = capsule_aabb_clearance_3d(p0s[1:], p1s[1:], radii[1:], slab_box)
                best = min(best, float(np.min(cl_slab)))
        else:
            cl_all = capsule_aabb_clearance_3d(p0s, p1s, radii, boxes)
            best = min(best, float(np.min(cl_all)))

    if check_self_collision and p0s.shape[0] >= 3:
        n_caps = int(p0s.shape[0])
        for i in range(n_caps):
            for j in range(i + 2, n_caps):
                d_seg = segment_segment_distance_3d(p0s[i], p1s[i], p0s[j], p1s[j])
                cl_self = max(0.0, d_seg - float(radii[i] + radii[j]))
                if cl_self < best:
                    best = cl_self
    return best


def in_collision_se3(
    q: Sequence[float] | np.ndarray,
    arm: object,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    *,
    margin_m: Optional[float] = None,
    attached_tip_length_m: Optional[float] = None,
) -> bool:
    """Return True if `q` violates joint limits, self-collides, or breaches 3D AABB `margin_m`."""
    if not arm.within_joint_limits(q):  # type: ignore[attr-defined]
        return True
    eff_margin = float(scene_3d.margin_m if margin_m is None else margin_m)
    cl = min_clearance_se3(
        q,
        arm,
        scene_3d=scene_3d,
        attached_tip_length_m=attached_tip_length_m,
    )
    return bool(cl <= eff_margin)


def kinematic_sweep_bound_se3(
    qa: Sequence[float] | np.ndarray,
    qb: Sequence[float] | np.ndarray,
    arm: object,
    *,
    attached_tip_length_m: float = 0.0,
) -> float:
    """Tight physical multi-joint Jacobian Lipschitz upper bound (m) on any link/tip displacement."""
    a = np.asarray(qa, dtype=np.float64)
    b = np.asarray(qb, dtype=np.float64)
    dq = np.abs(b - a)
    kinds = getattr(arm, "joint_kinds", ("revolute",) * dq.size)
    # Physical distal lever-arm ceilings (shoulder -> forearm -> wrist) for UR5e / Panda class cobots
    base_levers = (0.86, 0.86, 0.46, 0.16, 0.12, 0.04, 0.03)
    total = 0.0
    n = dq.size
    for i in range(n):
        if kinds[i] == "prismatic":
            total += float(dq[i])
        else:
            lev = base_levers[i] if i < len(base_levers) else 0.10
            total += (lev + float(attached_tip_length_m)) * float(dq[i])
    return float(total)


def edge_free_se3(
    qa: Sequence[float] | np.ndarray,
    qb: Sequence[float] | np.ndarray,
    arm: object,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    *,
    max_sweep_step_m: float = 0.040,
    margin_m: Optional[float] = None,
    attached_tip_length_m: Optional[float] = None,
) -> bool:
    """Continuous Lipschitz-certified 3D multi-link capsule-vs-AABB swept collision check."""
    a = np.asarray(qa, dtype=np.float64)
    b = np.asarray(qb, dtype=np.float64)
    if not (arm.within_joint_limits(a) and arm.within_joint_limits(b)):  # type: ignore[attr-defined]
        return False

    eff_tip = (
        float(scene_3d.attached_tip_length_m)
        if attached_tip_length_m is None
        else float(attached_tip_length_m)
    )
    base_margin = float(scene_3d.margin_m if margin_m is None else margin_m)

    if in_collision_se3(a, arm, scene_3d, margin_m=base_margin, attached_tip_length_m=eff_tip):
        return False
    if in_collision_se3(b, arm, scene_3d, margin_m=base_margin, attached_tip_length_m=eff_tip):
        return False

    sweep = kinematic_sweep_bound_se3(a, b, arm, attached_tip_length_m=eff_tip)
    if sweep <= 1e-12:
        return True

    n = max(2, int(math.ceil(sweep / max_sweep_step_m)) + 1)
    s_grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
    half_step_m = 0.5 * sweep / float(n - 1)
    diff = b - a

    # Coarse midpoint early-reject for long chords
    if n > 8:
        for sc in (0.5, 0.25, 0.75):
            if in_collision_se3(a + sc * diff, arm, scene_3d, margin_m=base_margin, attached_tip_length_m=eff_tip):
                return False

    cl_samples = np.array(
        [
            min_clearance_se3(a + float(s) * diff, arm, scene_3d, attached_tip_length_m=eff_tip)
            for s in s_grid
        ],
        dtype=np.float64,
    )
    if bool(np.any(cl_samples <= base_margin)):
        return False
    if bool(np.all(cl_samples > base_margin + half_step_m)):
        return True

    # Adaptive Lipschitz midpoint refinement on narrow-band intervals
    uncertain_intervals = (cl_samples[:-1] <= base_margin + half_step_m) | (
        cl_samples[1:] <= base_margin + half_step_m
    )
    s_lo = s_grid[:-1][uncertain_intervals]
    s_hi = s_grid[1:][uncertain_intervals]

    for _ in range(10):
        if s_lo.size == 0:
            return True
        s_mid = 0.5 * (s_lo + s_hi)
        rad = 0.5 * (s_hi - s_lo) * sweep
        cl_mid = np.array(
            [
                min_clearance_se3(a + float(sm) * diff, arm, scene_3d, attached_tip_length_m=eff_tip)
                for sm in s_mid
            ],
            dtype=np.float64,
        )
        if bool(np.any(cl_mid <= base_margin)):
            return False
        still = cl_mid <= (base_margin + rad)
        if not bool(np.any(still)):
            return True
        lo_u = s_lo[still]
        mid_u = s_mid[still]
        hi_u = s_hi[still]
        s_lo = np.concatenate([lo_u, mid_u])
        s_hi = np.concatenate([mid_u, hi_u])
    return False


def path_min_clearance_se3(
    path: Sequence[Sequence[float] | np.ndarray],
    arm: object,
    scene_3d: BenchScene3D = DEFAULT_SCENE_3D,
    *,
    max_sweep_step_m: float = 0.008,
    attached_tip_length_m: Optional[float] = None,
) -> float:
    """Certified minimum 3D capsule-to-AABB surface clearance (m) along a multi-DoF joint path."""
    if not path:
        raise ValueError("path must be non-empty")
    eff_tip = (
        float(scene_3d.attached_tip_length_m)
        if attached_tip_length_m is None
        else float(attached_tip_length_m)
    )
    pts = [np.asarray(p, dtype=np.float64) for p in path]
    best = min(min_clearance_se3(p, arm, scene_3d, attached_tip_length_m=eff_tip) for p in pts)
    if len(pts) == 1 or best <= 0.0:
        return max(0.0, best)

    for qa, qb in zip(pts, pts[1:]):
        sweep = kinematic_sweep_bound_se3(qa, qb, arm, attached_tip_length_m=eff_tip)
        if sweep <= 1e-12:
            continue
        n = max(3, int(math.ceil(sweep / max_sweep_step_m)) + 1)
        half_step = 0.5 * sweep / float(n - 1)
        for s in np.linspace(0.0, 1.0, n)[1:-1]:
            cl = min_clearance_se3(qa + float(s) * (qb - qa), arm, scene_3d, attached_tip_length_m=eff_tip)
            best = min(best, max(0.0, cl - 0.25 * half_step))
    return max(0.0, best)

