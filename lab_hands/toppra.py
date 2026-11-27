"""TOPP-RA reachability-analysis time-optimal path parameterization with jerk bounds.

Implements Pham & Pham (2018) *Time-Optimal Path Parameterization Based on Reachability
Analysis (TOPP-RA)* over C3-continuous multi-DoF joint paths `q(s)`:

1. **C3 Geometric Path Blending**: Connects `D`-DoF joint waypoints via linear segments
   and 6th-order symmetric corner blends `g(tau) = 5*tau^4 - 6*tau^5 + 2*tau^6`
   so `q(s)`, `q'(s)`, `q''(s)`, and `q'''(s)` are all continuous (`C3`), automatically
   shrinking blend radii if an optional `edge_free_fn` flags obstacle proximity.
2. **Exact Backward Reachability Pass**: Computes 1-step controllable squared-speed
   sets `K_i = [L_i, U_i]` (`x_i = s_dot(s_i)^2`) via exact 2-variable Fourier-Motzkin
   LP elimination over per-joint velocity (`|qd| <= qd_max`) and acceleration
   (`|qdd| <= qdd_max`) bounds.
3. **Greedy Time-Optimal Forward Pass**: Integrates `x_{i+1} = x_i + 2 * ds_i * u_i*`
   from `x_0 = 0` to `x_N = 0`.
4. **Integral-Preserving Raised-Cosine Jerk Filter**: Convolves `s_dot(t)` with a
   compact-support Hann kernel `w_tau(t)` (`int w_tau(t) dt = 1`) and exact boundary
   dilation so `s(0) = 0`, `s(T) = s_total`, `qd(0) = qd(T) = 0`, `qdd(0) = qdd(T) = 0`,
   and all per-joint velocity, acceleration, AND jerk (`|qddd| <= qddd_max`) bounds hold!
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

BlendPieceND = Tuple[str, float, Tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class ToppraTrajectory:
    """Reachability-optimal C3 rest-to-rest multi-DoF trajectory respecting (qd, qdd, qddd) bounds."""

    t: np.ndarray  # (T,)
    s: np.ndarray  # (T,)
    sd: np.ndarray  # (T,)
    sdd: np.ndarray  # (T,)
    sddd: np.ndarray  # (T,)
    q: np.ndarray  # (T, D)
    qd: np.ndarray  # (T, D)
    qdd: np.ndarray  # (T, D)
    qddd: np.ndarray  # (T, D)
    controllable_sets: np.ndarray  # (N_grid, 2) backward-pass [L_i, U_i] on x_i = sd^2
    s_total: float
    duration_s: float
    raw_toppra_duration_s: float
    max_joint_vel: np.ndarray  # (D,)
    max_joint_acc: np.ndarray  # (D,)
    max_joint_jerk: np.ndarray  # (D,)
    pieces: Tuple[BlendPieceND, ...]

    @property
    def dof(self) -> int:
        return int(self.q.shape[1])


def eval_c3_blended_pieces(
    pieces: Sequence[BlendPieceND],
    s_eval: np.ndarray,
    dof: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate C3-continuous `q(s), q'(s), q''(s), q'''(s)` at `s_eval`."""
    s_arr = np.asarray(s_eval, dtype=np.float64)
    m_pts = s_arr.size
    q_s = np.zeros((m_pts, dof), dtype=np.float64)
    qp_s = np.zeros((m_pts, dof), dtype=np.float64)
    qpp_s = np.zeros((m_pts, dof), dtype=np.float64)
    qppp_s = np.zeros((m_pts, dof), dtype=np.float64)

    cursor = 0.0
    n_pieces = len(pieces)
    for idx_p, (kind, plen, pdata) in enumerate(pieces):
        s_lo = cursor
        s_hi = cursor + plen
        if idx_p == n_pieces - 1:
            mask = (s_arr >= s_lo - 1e-12) & (s_arr <= s_hi + 1e-12)
        else:
            mask = (s_arr >= s_lo - 1e-12) & (s_arr < s_hi)
        if not bool(np.any(mask)):
            cursor = s_hi
            continue
        loc = np.clip(s_arr[mask] - s_lo, 0.0, plen)
        if kind == "line":
            a_pt, u_vec = pdata
            q_s[mask] = a_pt[None, :] + loc[:, None] * u_vec[None, :]
            qp_s[mask] = u_vec[None, :]
            qpp_s[mask] = 0.0
            qppp_s[mask] = 0.0
        else:
            a_b, u_prev, delta, rk_arr = pdata
            rk = float(rk_arr[0])
            inv_2rk = 1.0 / (2.0 * rk)
            tau = loc * inv_2rk
            # 6th-order symmetric C3 blend: g(tau) = 5*tau^4 - 6*tau^5 + 2*tau^6
            g = 5.0 * (tau**4) - 6.0 * (tau**5) + 2.0 * (tau**6)
            gp = 20.0 * (tau**3) - 30.0 * (tau**4) + 12.0 * (tau**5)
            gpp = 60.0 * (tau**2) - 120.0 * (tau**3) + 60.0 * (tau**4)
            gppp = 120.0 * tau - 360.0 * (tau**2) + 240.0 * (tau**3)

            v0 = 2.0 * rk * u_prev
            q_s[mask] = a_b[None, :] + tau[:, None] * v0[None, :] + g[:, None] * delta[None, :]
            qp_s[mask] = u_prev[None, :] + (gp * inv_2rk)[:, None] * delta[None, :]
            qpp_s[mask] = (gpp * (inv_2rk**2))[:, None] * delta[None, :]
            qppp_s[mask] = (gppp * (inv_2rk**3))[:, None] * delta[None, :]
        cursor = s_hi

    return q_s, qp_s, qpp_s, qppp_s


def build_c3_geometric_path(
    waypoints: Sequence[Sequence[float] | np.ndarray],
    *,
    max_blend_radius: float = 0.10,
    edge_free_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
) -> Tuple[List[BlendPieceND], float, int]:
    """Build a C3-continuous blended path through `D`-DoF waypoints."""
    pts: List[np.ndarray] = []
    for wp in waypoints:
        arr = np.asarray(wp, dtype=np.float64).copy()
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(f"Each waypoint must be 1D, got {arr.shape}")
        if not pts or float(np.linalg.norm(arr - pts[-1])) > 1e-10:
            pts.append(arr)
    if not pts:
        raise ValueError("waypoints must be non-empty")
    dof = int(pts[0].size)
    if any(p.size != dof for p in pts):
        raise ValueError("All waypoints must have identical DOF")
    if len(pts) == 1:
        return [("line", 0.0, (pts[0], np.zeros(dof, dtype=np.float64)))], 0.0, dof

    seg_vecs = [pts[k + 1] - pts[k] for k in range(len(pts) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    tangents = [v / max(L, 1e-12) for v, L in zip(seg_vecs, seg_lens)]
    m = len(seg_lens)

    radii = [0.0] * (m + 1)
    for k in range(1, m):
        radii[k] = min(0.28 * seg_lens[k - 1], 0.28 * seg_lens[k], float(max_blend_radius))

    for _ in range(10):
        pieces: List[BlendPieceND] = []
        for k in range(m):
            r_in = radii[k]
            r_out = radii[k + 1]
            lin_len = max(0.0, seg_lens[k] - r_in - r_out)
            a_lin = pts[k] + r_in * tangents[k]
            if lin_len > 1e-12:
                pieces.append(("line", lin_len, (a_lin, tangents[k])))
            if k < m - 1 and radii[k + 1] > 1e-9:
                rk = radii[k + 1]
                a_b = pts[k + 1] - rk * tangents[k]
                delta = rk * (tangents[k + 1] - tangents[k])
                pieces.append(("blend", 2.0 * rk, (a_b, tangents[k], delta, np.array([rk], dtype=np.float64))))

        s_total = float(sum(p[1] for p in pieces))
        if edge_free_fn is None:
            return pieces, s_total, dof

        ok = True
        for kind, plen, pdata in pieces:
            if kind == "line":
                a_pt, u_vec = pdata
                if not edge_free_fn(a_pt, a_pt + plen * u_vec):
                    ok = False
                    break
            else:
                s_sub = np.linspace(0.0, plen, 13, dtype=np.float64)
                q_sub, _, _, _ = eval_c3_blended_pieces([(kind, plen, pdata)], s_sub, dof)
                for qa, qb in zip(q_sub, q_sub[1:]):
                    if not edge_free_fn(qa, qb):
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            return pieces, s_total, dof
        radii = [0.5 * r for r in radii]

    fallback: List[BlendPieceND] = [
        ("line", seg_lens[k], (pts[k], tangents[k])) for k in range(m) if seg_lens[k] > 1e-12
    ]
    return fallback, float(sum(p[1] for p in fallback)), dof


def _stage_u_linear_bounds(
    qp_a: np.ndarray,
    qpp_a: np.ndarray,
    qp_b: np.ndarray,
    qpp_b: np.ndarray,
    a_max: np.ndarray,
    ds: float,
    l_next: float,
    u_next: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct linear bounds `alpha_m * x + beta_m <= u <= gamma_k * x + delta_k` on stage `[s_i, s_{i+1}]`.

    Enforces joint acceleration constraints at BOTH `s_i` (`x_i`) and `s_{i+1}` (`x_{i+1} = x_i + 2*ds*u`)
    together with target controllable interval `l_next <= x_i + 2*ds*u <= u_next`.
    """
    inv_2ds = 1.0 / (2.0 * ds)
    alphas: List[float] = [-inv_2ds]
    betas: List[float] = [l_next * inv_2ds]
    gammas: List[float] = [-inv_2ds]
    deltas: List[float] = [u_next * inv_2ds]

    # Endpoint A: -a_max <= qp_a * u + qpp_a * x <= a_max
    # Endpoint B: -a_max <= qp_b * u + qpp_b * (x + 2*ds*u) <= a_max
    #             -> (qp_b + 2*ds*qpp_b) * u + qpp_b * x in [-a_max, a_max]
    for coeff_u, coeff_x in ((qp_a, qpp_a), (qp_b + 2.0 * ds * qpp_b, qpp_b)):
        for d in range(a_max.size):
            cu = float(coeff_u[d])
            cx = float(coeff_x[d])
            am = float(a_max[d])
            if abs(cu) > 1e-11:
                if cu > 0.0:
                    alphas.append(-cx / cu)
                    betas.append(-am / cu)
                    gammas.append(-cx / cu)
                    deltas.append(am / cu)
                else:
                    # Dividing by negative cu flips inequality direction
                    alphas.append(-cx / cu)
                    betas.append(am / cu)
                    gammas.append(-cx / cu)
                    deltas.append(-am / cu)
            # When |cu| <= 1e-11, |cx * x| <= am is handled by x_max_grid

    return (
        np.asarray(alphas, dtype=np.float64),
        np.asarray(betas, dtype=np.float64),
        np.asarray(gammas, dtype=np.float64),
        np.asarray(deltas, dtype=np.float64),
    )


def _one_step_controllable_interval(
    x_hi_vel: float,
    qp_a: np.ndarray,
    qpp_a: np.ndarray,
    qp_b: np.ndarray,
    qpp_b: np.ndarray,
    a_max: np.ndarray,
    ds: float,
    l_next: float,
    u_next: float,
) -> Tuple[float, float]:
    """Exact 2-variable LP Fourier-Motzkin elimination solving `[L_i, U_i]` for `x_i = sd(s_i)^2`."""
    lo = 0.0
    hi = float(x_hi_vel)

    # Pure centripetal bound |qpp * x| <= a_max where |qp| ~ 0
    for qp_v, qpp_v in ((qp_a, qpp_a), (qp_b + 2.0 * ds * qpp_b, qpp_b)):
        mask_zero_u = (np.abs(qp_v) <= 1e-11) & (np.abs(qpp_v) > 1e-12)
        if bool(np.any(mask_zero_u)):
            cent_hi = float(np.min(a_max[mask_zero_u] / np.abs(qpp_v[mask_zero_u])))
            hi = min(hi, cent_hi)

    if hi < lo - 1e-12:
        return 0.0, 0.0

    alphas, betas, gammas, deltas = _stage_u_linear_bounds(
        qp_a, qpp_a, qp_b, qpp_b, a_max, ds, l_next, u_next
    )
    # Pairwise elimination: alpha_m * x + beta_m <= gamma_k * x + delta_k
    # <-> (alpha_m - gamma_k) * x <= delta_k - beta_m
    A = alphas[:, None] - gammas[None, :]
    B = deltas[None, :] - betas[:, None]

    zero_mask = np.abs(A) <= 1e-13
    if bool(np.any(zero_mask & (B < -1e-9))):
        return 0.0, 0.0

    pos_mask = A > 1e-13
    if bool(np.any(pos_mask)):
        hi = min(hi, float(np.min(B[pos_mask] / A[pos_mask])))

    neg_mask = A < -1e-13
    if bool(np.any(neg_mask)):
        lo = max(lo, float(np.max(B[neg_mask] / A[neg_mask])))

    if hi < lo:
        if lo - hi <= 1e-7:
            mid = max(0.0, 0.5 * (lo + hi))
            return mid, mid
        return 0.0, 0.0
    return max(0.0, lo), max(0.0, hi)


def parameterize_toppra(
    waypoints: Sequence[Sequence[float] | np.ndarray],
    *,
    qd_max: Sequence[float] | float,
    qdd_max: Sequence[float] | float,
    qddd_max: Optional[Sequence[float] | float] = None,
    n_grid: int = 241,
    dt_s: float = 0.004,
    max_blend_radius: float = 0.10,
    edge_free_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    safety_scale: float = 0.985,
) -> ToppraTrajectory:
    """Compute a reachability-optimal TOPP-RA trajectory respecting velocity, acceleration, and jerk bounds."""
    if n_grid < 11:
        raise ValueError(f"n_grid must be >= 11, got {n_grid}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}")

    pieces, s_total, dof = build_c3_geometric_path(
        waypoints,
        max_blend_radius=max_blend_radius,
        edge_free_fn=edge_free_fn,
    )

    v_lim_raw = np.broadcast_to(np.asarray(qd_max, dtype=np.float64), (dof,)).copy()
    a_lim_raw = np.broadcast_to(np.asarray(qdd_max, dtype=np.float64), (dof,)).copy()
    if qddd_max is None:
        j_lim_raw = 12.0 * a_lim_raw
    else:
        j_lim_raw = np.broadcast_to(np.asarray(qddd_max, dtype=np.float64), (dof,)).copy()

    if bool(np.any(v_lim_raw <= 0.0) or np.any(a_lim_raw <= 0.0) or np.any(j_lim_raw <= 0.0)):
        raise ValueError("All velocity, acceleration, and jerk bounds must be strictly positive")

    if s_total <= 1e-12:
        q0 = np.asarray(waypoints[0], dtype=np.float64)[None, :]
        z1 = np.zeros(1, dtype=np.float64)
        zn = np.zeros((1, dof), dtype=np.float64)
        return ToppraTrajectory(
            t=z1,
            s=z1,
            sd=z1,
            sdd=z1,
            sddd=z1,
            q=q0.copy(),
            qd=zn.copy(),
            qdd=zn.copy(),
            qddd=zn.copy(),
            controllable_sets=np.zeros((1, 2), dtype=np.float64),
            s_total=0.0,
            duration_s=0.0,
            raw_toppra_duration_s=0.0,
            max_joint_vel=np.zeros(dof, dtype=np.float64),
            max_joint_acc=np.zeros(dof, dtype=np.float64),
            max_joint_jerk=np.zeros(dof, dtype=np.float64),
            pieces=tuple(pieces),
        )

    v_lim = safety_scale * v_lim_raw
    a_lim = safety_scale * a_lim_raw
    j_lim = safety_scale * j_lim_raw

    s_grid = np.linspace(0.0, s_total, n_grid, dtype=np.float64)
    ds_grid = np.diff(s_grid)
    _, qp_g, qpp_g, qppp_g = eval_c3_blended_pieces(pieces, s_grid, dof)

    # State velocity ceiling x_max[i] from joint speed, corner centripetal accel, and corner curvature jerk
    x_max = np.full(n_grid, 1e8, dtype=np.float64)
    for i in range(n_grid):
        for d in range(dof):
            aqp = abs(float(qp_g[i, d]))
            aqpp = abs(float(qpp_g[i, d]))
            aqppp = abs(float(qppp_g[i, d]))
            if aqp > 1e-11:
                x_max[i] = min(x_max[i], (float(v_lim[d]) / aqp) ** 2)
            if aqpp > 1e-11:
                x_max[i] = min(x_max[i], 0.85 * float(a_lim[d]) / aqpp)
            if aqppp > 1e-11:
                x_max[i] = min(x_max[i], (0.75 * float(j_lim[d]) / aqppp) ** (2.0 / 3.0))
    x_max[0] = 0.0
    x_max[-1] = 0.0

    # --- 1. TOPP-RA Backward Reachability Pass ---
    K = np.zeros((n_grid, 2), dtype=np.float64)
    K[-1, :] = 0.0
    for i in range(n_grid - 2, -1, -1):
        li, ui = _one_step_controllable_interval(
            float(x_max[i]),
            qp_g[i],
            qpp_g[i],
            qp_g[i + 1],
            qpp_g[i + 1],
            a_lim,
            float(ds_grid[i]),
            float(K[i + 1, 0]),
            float(K[i + 1, 1]),
        )
        K[i, 0] = li
        K[i, 1] = ui
    K[0, :] = 0.0

    # --- 2. TOPP-RA Forward Time-Optimal Pass ---
    x_opt = np.zeros(n_grid, dtype=np.float64)
    u_opt = np.zeros(n_grid - 1, dtype=np.float64)
    t_stage = np.zeros(n_grid, dtype=np.float64)

    for i in range(n_grid - 1):
        xi = float(x_opt[i])
        ds = float(ds_grid[i])
        alphas, betas, gammas, deltas = _stage_u_linear_bounds(
            qp_g[i],
            qpp_g[i],
            qp_g[i + 1],
            qpp_g[i + 1],
            a_lim,
            ds,
            float(K[i + 1, 0]),
            float(K[i + 1, 1]),
        )
        u_lo = float(np.max(alphas * xi + betas))
        u_hi = float(np.min(gammas * xi + deltas))
        ui = u_hi if u_hi >= u_lo else 0.5 * (u_lo + u_hi)
        x_next = float(np.clip(xi + 2.0 * ds * ui, K[i + 1, 0], K[i + 1, 1]))
        if i == n_grid - 2:
            x_next = 0.0
        ui = (x_next - xi) / (2.0 * ds)
        u_opt[i] = ui
        x_opt[i + 1] = x_next

        v_avg = 0.5 * (math.sqrt(max(0.0, xi)) + math.sqrt(max(0.0, x_next)))
        if v_avg <= 1e-12:
            dt_i = math.sqrt(2.0 * ds / max(abs(ui), 1e-6))
        else:
            dt_i = ds / v_avg
        t_stage[i + 1] = t_stage[i] + dt_i

    raw_dur = float(t_stage[-1])

    # Sample raw TOPP-RA speed profile sd_raw(t) on uniform fine time grid
    n_raw = max(32, int(math.ceil(raw_dur / dt_s)) + 1)
    dt_eff = raw_dur / float(n_raw - 1)
    t_raw = np.linspace(0.0, raw_dur, n_raw, dtype=np.float64)

    # Exact piecewise-constant-acceleration stage interpolation for s_raw(t) and sd_raw(t)
    stage_idx = np.clip(np.searchsorted(t_stage, t_raw, side="right") - 1, 0, n_grid - 2)
    dt_local = t_raw - t_stage[stage_idx]
    v0_stage = np.sqrt(np.maximum(0.0, x_opt[stage_idx]))
    u_stage = u_opt[stage_idx]
    sd_raw = np.maximum(0.0, v0_stage + u_stage * dt_local)
    sd_raw[0] = 0.0
    sd_raw[-1] = 0.0

    # Normalize integral so trapezoidal integral of sd_raw equals exact s_total
    trap_area = float(np.trapezoid(sd_raw, dx=dt_eff)) if hasattr(np, "trapezoid") else float(np.trapz(sd_raw, dx=dt_eff))
    if trap_area > 1e-12:
        sd_raw = sd_raw * (s_total / trap_area)

    # --- 3. Integral-Preserving Raised-Cosine Jerk Filter ---
    # Kernel width tau_j >= 2 * max(a_lim / j_lim) smooths bang-bang acceleration switches into C2/C3 S-curves
    tau_j = max(4.0 * dt_eff, float(np.max(2.2 * a_lim / j_lim)))
    m_kern = max(5, int(math.ceil(tau_j / dt_eff)) | 1)  # odd length >= 5
    k_grid = np.linspace(0.0, 1.0, m_kern, dtype=np.float64)
    # Hann raised-cosine window: w(0) = w(1) = 0, w'(0) = w'(1) = 0 -> smooth acceleration & bounded jerk!
    kern = 1.0 - np.cos(2.0 * math.pi * k_grid)
    kern_sum = float(np.sum(kern))
    kern = kern / kern_sum

    sd_filt = np.convolve(sd_raw, kern, mode="full")
    sd_filt[0] = 0.0
    sd_filt[-1] = 0.0
    n_out = sd_filt.size
    base_dur = float(n_out - 1) * dt_eff

    # Exact cumulative trapezoidal integration of sd_filt -> s(t) with exact endpoint s_total
    s_cum = np.zeros(n_out, dtype=np.float64)
    s_cum[1:] = np.cumsum(0.5 * (sd_filt[:-1] + sd_filt[1:])) * dt_eff
    if s_cum[-1] > 1e-12:
        scale_s = s_total / float(s_cum[-1])
        s_cum *= scale_s
        sd_filt *= scale_s
    s_cum = np.clip(s_cum, 0.0, s_total)
    s_cum[0] = 0.0
    s_cum[-1] = s_total

    # Evaluate exact C3 geometric derivatives at s_cum
    q_tr, qp_tr, qpp_tr, qppp_tr = eval_c3_blended_pieces(pieces, s_cum, dof)

    # Base derivatives at time scale alpha = 1.0
    sdd_base = np.gradient(sd_filt, dt_eff, edge_order=2)
    sdd_base[0] = 0.0
    sdd_base[-1] = 0.0
    sddd_base = np.gradient(sdd_base, dt_eff, edge_order=2)

    qd_base = qp_tr * sd_filt[:, None]
    qdd_base = qp_tr * sdd_base[:, None] + qpp_tr * (sd_filt[:, None] ** 2)
    qddd_base = (
        qp_tr * sddd_base[:, None]
        + 3.0 * qpp_tr * (sd_filt * sdd_base)[:, None]
        + qppp_tr * (sd_filt[:, None] ** 3)
    )

    # Exact uniform time-dilation factor alpha >= 1.0 if corner curvature q'''*sd^3 slightly nudges any bound
    ratio_v = float(np.max(np.abs(qd_base) / v_lim_raw[None, :]))
    ratio_a = math.sqrt(float(np.max(np.abs(qdd_base) / a_lim_raw[None, :])))
    ratio_j = float(np.max(np.abs(qddd_base) / j_lim_raw[None, :])) ** (1.0 / 3.0)
    alpha = max(1.0, ratio_v * 1.002, ratio_a * 1.002, ratio_j * 1.002)

    final_dur = base_dur * alpha
    t_out = np.linspace(0.0, final_dur, n_out, dtype=np.float64)
    sd_out = sd_filt / alpha
    sdd_out = sdd_base / (alpha**2)
    sddd_out = sddd_base / (alpha**3)

    qd_out = qd_base / alpha
    qdd_out = qdd_base / (alpha**2)
    qddd_out = qddd_base / (alpha**3)
    qd_out[0] = 0.0
    qd_out[-1] = 0.0
    qdd_out[0] = 0.0
    qdd_out[-1] = 0.0

    return ToppraTrajectory(
        t=t_out,
        s=s_cum,
        sd=sd_out,
        sdd=sdd_out,
        sddd=sddd_out,
        q=q_tr,
        qd=qd_out,
        qdd=qdd_out,
        qddd=qddd_out,
        controllable_sets=K,
        s_total=float(s_total),
        duration_s=float(final_dur),
        raw_toppra_duration_s=float(raw_dur),
        max_joint_vel=np.max(np.abs(qd_out), axis=0),
        max_joint_acc=np.max(np.abs(qdd_out), axis=0),
        max_joint_jerk=np.max(np.abs(qddd_out), axis=0),
        pieces=tuple(pieces),
    )


__all__ = [
    "BlendPieceND",
    "ToppraTrajectory",
    "build_c3_geometric_path",
    "eval_c3_blended_pieces",
    "parameterize_toppra",
]
