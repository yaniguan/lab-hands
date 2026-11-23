"""Closed-loop tactile 'last-centimeter' insertion benchmark on a clear vessel bevel.

Demonstrates resolving +-3 mm camera depth/refraction pose error on a transparent
tube or microplate well down to < 0.2 mm centering error via 2D contact shear
compliance feedback.

Usage:
    python demo/tactile_insert.py [--trials 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lab_hands.tactile import ClearVesselBevel, run_open_loop_insertion, run_tactile_insertion  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visual-noise-mm", type=float, default=3.0, help="Max per-axis visual pose error in mm")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    # Bore radius 0.08 mm around pipette tip shaft so contact shear stays active down to < 0.2 mm centering
    vessel = ClearVesselBevel(bore_radius_m=0.00008, capture_radius_m=0.0055)
    center = np.asarray(vessel.center_xy, dtype=np.float64)
    noise_m = args.visual_noise_mm / 1000.0

    open_ok = 0
    tac_ok = 0
    init_errs = []
    tac_errs = []
    tac_forces = []
    tac_steps = []

    for _ in range(args.trials):
        offset = rng.uniform(-noise_m, noise_m, size=2)
        visual_xy = center + offset
        ol = run_open_loop_insertion(visual_xy, vessel=vessel)
        tc = run_tactile_insertion(visual_xy, rng, vessel=vessel)
        open_ok += int(ol.success)
        tac_ok += int(tc.success)
        init_errs.append(tc.initial_error_mm)
        tac_errs.append(tc.final_error_mm)
        tac_forces.append(tc.peak_force_n)
        tac_steps.append(tc.steps)

    max_err = float(np.max(tac_errs))
    print(
        f"trials: {args.trials}, visual pose error: mean {np.mean(init_errs):.2f} mm, max {np.max(init_errs):.2f} mm"
    )
    print(f"open-loop camera plunge success: {open_ok}/{args.trials} ({100.0 * open_ok / args.trials:.1f}%)")
    print(
        f"tactile compliant insertion success: {tac_ok}/{args.trials} ({100.0 * tac_ok / args.trials:.1f}%), "
        f"final centering error: mean {np.mean(tac_errs):.3f} mm, max {max_err:.3f} mm "
        f"(mean {np.mean(tac_steps):.1f} steps, peak contact {np.mean(tac_forces):.2f} N)"
    )

    if tac_ok != args.trials or max_err >= 0.2:
        print(
            f"error: tactile insertion regression: success={tac_ok}/{args.trials}, max_err={max_err:.3f} mm >= 0.200 mm",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
