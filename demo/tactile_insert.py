"""3D Whitney/Coulomb peg-in-hole & active spiral/Bayesian tactile insertion benchmark.

Evaluates all 4 insertion regimes (`open_loop`, `passive_rcc_compliance`,
`local_shear_admittance`, and `active_spiral_bayesian_tactile`) on an un-cherry-picked
SBS polypropylene 96-well / microtube opening (`chamfer_width = 0.70 mm`,
`bore_clearance = 0.10 mm`, `capture_radius = 0.80 mm`) across refractive RGB-D pose
offsets (`0..4.0 mm`), 3D Coulomb friction (`mu in [0.15, 0.45]`), and angular tilt
misalignment (`theta in [0.0, 2.5] deg`).

Usage:
    python demo/tactile_insert.py [--trials 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lab_hands.tactile import benchmark_tactile_regimes  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--visual-noise-mm", type=float, default=4.0, help="Max radial refractive offset in mm")
    args = ap.parse_args(argv)

    report = benchmark_tactile_regimes(
        n_trials=args.trials,
        seed=args.seed,
        chamfer_width_m=0.00070,
        bore_radius_m=0.00010,
        offset_range_mm=(0.0, float(args.visual_noise_mm)),
        friction_range=(0.15, 0.45),
        tilt_range_deg=(0.0, 2.5),
    )

    geom = report["geometry"]
    regs = report["regimes"]
    print(
        f"geometry: bore_clearance={geom['bore_clearance_mm']:.2f} mm, "
        f"chamfer_width={geom['chamfer_width_mm']:.2f} mm, "
        f"capture_radius={geom['capture_radius_mm']:.2f} mm, "
        f"voronoi_search_limit={geom['max_voronoi_search_radius_mm']:.1f} mm"
    )
    for key in (
        "open_loop",
        "passive_rcc_compliance",
        "local_shear_admittance",
        "active_spiral_bayesian_tactile",
    ):
        r = regs[key]
        err_str = (
            f"{r['mean_final_error_mm_given_success']:.3f} mm"
            if r["mean_final_error_mm_given_success"] is not None
            else "n/a"
        )
        print(
            f"{key:30s}: success {r['success_count']:3d}/{r['trials']} ({r['success_rate_pct']:5.1f}%), "
            f"jam/wedge {r['jam_or_wedge_rate_pct']:4.1f}%, rim_hangup {r['rim_hangup_rate_pct']:5.1f}%, "
            f"mean_err {err_str}, peak_F {r['mean_peak_force_n']:.2f} N"
        )

    act = regs["active_spiral_bayesian_tactile"]
    loc = regs["local_shear_admittance"]
    ol = regs["open_loop"]
    max_err = act["max_final_error_mm_given_success"] or 999.0

    if (
        act["success_rate_pct"] < 90.0
        or act["success_rate_pct"] <= loc["success_rate_pct"]
        or loc["success_rate_pct"] <= ol["success_rate_pct"]
        or max_err >= 0.20
    ):
        print(
            f"error: tactile benchmark regression: active={act['success_rate_pct']}%, "
            f"local={loc['success_rate_pct']}%, max_err={max_err:.3f} mm",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
