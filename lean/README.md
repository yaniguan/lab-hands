# `LabHandsLean` -- Formal L1 Obstacle Clearance & Open-Meniscus Slosh Proofs

`LabHandsLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the spatial collision, tactile centering, and open-meniscus slosh time-scaling invariants of `lab-hands`:

- **Core Manipulation Gate (`checkLabHands`, `checkLabHands_guarantees_5mm_clearance`)**: Verifies `>= 5 mm` continuous segment-AABB obstacle clearance, `|q2| <= 2600 mrad` elbow self-collision limit, non-negative rim Z clearance (`checkLabHands_rim_and_tactile_safe`), tactile insertion radial error `<= 200 um`, and slosh-bounded lateral acceleration.
- **Open-Meniscus Minimum-Jerk Time-Scaler (`checkSloshScaledTrajectory`)**: Proves that keeping predicted cylindrical slosh meniscus rise strictly below `0.7 * h_free` (`sloshScaled_meniscus_strictly_below_70pct_headspace`) guarantees zero well spills (`sloshScaled_zero_well_spills`) while preserving `>= 5 mm` obstacle clearance (`sloshScaled_preserves_5mm_clearance_and_duration`).

## Build

```bash
cd lean
lake build
```
