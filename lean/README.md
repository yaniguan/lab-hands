# `LabHandsLean` -- Formal L1 Obstacle Clearance, Lipschitz Swept-Segment & Slosh Proofs

`LabHandsLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the spatial collision, kinematic swept-volume subdivision, tactile centering, and open-meniscus slosh time-scaling invariants of `lab-hands`:

- **Kinematic Lipschitz Swept-Segment Soundness (`tipSweepBoundUM`, `halfStepSweepUM`, `sweptSegment_continuous_clearance_sound`, `checkSweptSegment_sound`)**: Proves that bounding finite `N`-subdivision grid samples by `reqMarginUM + halfStepSweepUM` under the 2-link Jacobian bound `(L1 + L2) * |dq1| + L2 * |dq2|` guarantees continuous obstacle clearance `>= reqMarginUM` (`5000 um` = `5.0 mm`) at every interior point along `[0, 1]`.
- **Core Manipulation Gate (`checkLabHands`, `checkLabHands_guarantees_5mm_clearance`)**: Verifies `>= 5 mm` continuous segment-AABB obstacle clearance, `|q2| <= 2600 mrad` elbow self-collision limit, non-negative rim Z clearance (`checkLabHands_rim_and_tactile_safe`), tactile insertion radial error `<= 200 um`, and slosh-bounded lateral acceleration.
- **Open-Meniscus Minimum-Jerk Time-Scaler (`checkSloshScaledTrajectory`)**: Proves that keeping predicted cylindrical slosh meniscus rise strictly below `0.7 * h_free` (`sloshScaled_meniscus_strictly_below_70pct_headspace`) guarantees zero well spills (`sloshScaled_zero_well_spills`) while preserving `>= 5 mm` obstacle clearance (`sloshScaled_preserves_5mm_clearance_and_duration`).

## Build

```bash
cd lean
lake build
```
