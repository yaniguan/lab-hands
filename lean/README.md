# `LabHandsLean` -- Formal L1 6-DoF Swept-Volume, Input-Shaper, Whitney & Slosh Proofs

`LabHandsLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the spatial collision, 2-DoF & 6-DoF kinematic swept-volume subdivision, Whitney peg-in-hole non-wedging, input-shaper bound convexity, and surrogate slosh headspace inequalities of `lab-hands`:

- **2-DoF & 6-DoF Kinematic Lipschitz Swept-Segment Soundness (`tipSweepBoundUM`, `tipSweepBound6DoFUM`, `halfStepSweepUM`, `sweptSegment_continuous_clearance_sound`, `sweptSegment6DoF_continuous_clearance_sound`, `checkSweptSegment_sound`)**: Proves that bounding finite `N`-subdivision grid samples by `reqMarginUM + halfStepSweepUM` guarantees continuous obstacle clearance `>= reqMarginUM` (`5000 um` = `5.0 mm`) at every interior point along `[0, 1]`.
- **Whitney (1982) Two-Point Peg-in-Hole Non-Wedging (`checkWhitneyNonWedging`, `checkWhitneyNonWedging_sound`)**: Formalizes the exact Coulomb friction-cone non-wedging inequality `mu * R_hole * theta < c` (`frictionMilli * holeRadiusUM * tiltUrad < 10^9 * boreClearanceUM`).
- **Positive Input-Shaper Convexity (`positiveShaper_preserves_kinematic_bound`)**: Proves that positive 3-impulse `ZV`/`ZVD`/`EI` shapers summing to unity never amplify joint velocity, acceleration, or jerk bounds.
- **Core Manipulation & Surrogate Slosh Headspace Gates (`checkLabHands`, `checkSloshScaledTrajectory`)**: Proves `>= 5 mm` obstacle clearance (`checkLabHands_guarantees_5mm_clearance`), rim transit & tactile centering (`checkLabHands_rim_and_tactile_safe`), and that the predicted meniscus rise under the flat-surface linear gravity surrogate stays strictly below `0.7 * h_free` (`sloshScaled_meniscus_strictly_below_70pct_headspace`, `sloshScaled_predicted_rise_below_headspace`; `#37`: bounds the surrogate prediction rather than guaranteeing physical spill absence in low-Bond-number microplates).

## Build

```bash
cd lean
lake build
```
