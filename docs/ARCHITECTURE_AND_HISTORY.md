# `lab-hands` -- Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L1 · Protocol Compiler & Visuo-Tactile Manipulation`  
> **License**: MIT (repository code; external datasets and repositories retain their respective licenses)

---

## 1. Object

`lab-hands` is the **6-DoF / 7-DoF Manipulation, Open-Well Slosh Control & Visuo-Tactile Insertion (`L1`)** component of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`-`L6` Brain/Body stack, `L1` bridges symbolic wet-lab protocol steps to collision-free, dynamically feasible physical execution across robotic cobot arms (`UR5e`, `Franka Emika Panda`) and liquid-handling gantries (`Opentrons Flex` / `OT-2`), while closing the last-millimeter force/torque loop where RGB-D cameras fail on transparent borosilicate glass, clear polypropylene tubes, and refractive menisci.

### Inputs and Outputs (`lab_hands/`, `demo/`, `benchmarks/`)

- **Inputs**:
  - 1-based well indices (`1..N`, row-major or column-major) or alphanumeric SBS well references (`A1`-`P24`, `slot:well`) across the 24-entry `STEPBOT_LABWARE_CATALOG` (`6`..`384`-well microplates, reservoirs, `120 mm` Falcon conical racks, and `10`..`1000 uL` tipracks).
  - Exact `SE(3)` serial manipulator chains (`UR5eArm` 6-DoF, `PandaArm` 7-DoF, `OpentronsFlexGantry` 4-DoF, plus legacy `Arm2D` / `ArmSE3`), 3D benchtop obstacle scenes (`BenchScene3D`), liquid physical properties (`LIQUID_CATALOG`), SBS polypropylene chamfer geometry (`ClearVesselBevel`), and deterministic RNG seeds.
- **Outputs**:
  - Null-space-optimized Damped Least-Squares IK solutions (`dls_ik`), Lipschitz-certified 3D multi-link capsule-vs-AABB collision-free joint paths (`rrt_connect_se3`, `informed_rrt_star`, `shortcut_se3`), jerk-bounded time-optimal trajectories (`parameterize_toppra`), `ZV` / `ZVD` / `EI` input-shaped open-plate transport trajectories (`design_input_shaper`, `shape_toppra_trajectory`), active spiral + Bayesian particle-filter tactile insertions (`run_active_tactile_insertion`), canonical HuggingFace LeRobot `(T, 7)` `float32` frame tables (`LeRobotFrameTable`), canonical Opentrons `(T, 3)` `mm` deck waypoint trajectories (`OpentronsWaypointTrajectory`), and `autolab-wire/v1` / `lab-hands/trajectory/v1` JSON artifacts.

---

## 2. Data & Canonical Tensor Contracts

### 2.1 Built-in Geometric, Labware & Deck Specification (`lab_hands/labware.py`)

- **SBS 96/384-well & 24-entry `stepbot` catalog**:
  - Default `8` rows (`ROWS = "ABCDEFGH"`) x `12` columns (`1..12`), `PITCH = 0.009 m` (`9.0 mm`), `PLATE_ORIGIN = (0.20, 0.05) m`, plus full 24-entry `STEPBOT_LABWARE_CATALOG` carrying per-labware `z_bottom_m`, `z_rim_m` (up to `0.120 m` for 15/50 mL Falcon racks), `well_radius_m`, and `is_tiprack`.
  - Exact **Opentrons Deck <-> Robot-Base Coordinate Transform**: `DEFAULT_DECK_ORIGIN_IN_BASE_M = (0.14562, -0.17424, 0.0) m` aligns `DEFAULT_SLOT_ORIGINS` (`Slot 1 A1 = (0.16, -0.10) m`, `Slot 5 A1 = (0.2925, -0.0095) m`) to `0.00 mm` with `opentrons.simulate` 8.8.2 (`Slot 1 A1 = (14.38, 74.24) mm`, `Slot 5 A1 = (146.88, 164.74) mm`) inside the `(0..418, 0..353, 0..218) mm` deck envelope.

### 2.2 Canonical LeRobot v2.1/v3.0 & Opentrons Gantry Interfaces (`lab_hands/trajectory.py`, `lab_hands/datasets.py`)

- **`LeRobotFrameTable` (`validate_lerobot_frame_tensors`)**:
  - `observation.state`: `np.ndarray` `float32` shape `(T, D_state)` (default `(T, 7)` = 6 UR5e joints + 1 gripper/plunger).
  - `action`: `np.ndarray` `float32` shape `(T, D_action)` (`(T, 7)`).
  - `timestamp`: `float32 (T,)`, `frame_index`: `int64 (T,)`, `episode_index`: `int64 (T,)`, `next.done`: `bool (T,)` (`True` on terminal frame).
- **`OpentronsWaypointTrajectory` (`validate_opentrons_trajectory`, `extract_opentrons_runlog_trajectory`)**:
  - `deck_tip_mm`: `float64 (T, 3)` in millimeters bounded inside `(0..418, 0..353, 0..218) mm`.
  - `base_tip_m`: `float64 (T, 3)` SI meters, `base_pose_se3`: `float64 (T, 6)` `(x, y, z, roll, pitch, yaw)`, `plunger_ul`: `float64 (T,)` in `uL`.
- **12-Entry Typed Open Manipulation Registry (`OPEN_MANIPULATION_DATASETS`)**:
  - Spans `wetlab_video` (`finebio`, real 1-column `P01_01_01.txt` step segmentation), `kinematics` (`droid`, `lerobot`, `autobio_pipette_mujoco`), `transparent_depth` (`transcg`, `transproteus`, `clearpose`, `layered_depth`), and `visuo_tactile` (`tvl`, `tactx`, `feelsight`, `octopi_physiclear`) with fail-closed SPDX `commercial_use_ok` checks.

---

## 3. Model

### 3.1 6-DoF `UR5e`, 7-DoF `Panda` & `OpentronsFlexGantry` Kinematics (`lab_hands/kinematics_se3.py`)

- **Exact PoE + DH / Craig MDH Cross-Verification**:
  - `UR5eArm` (Universal Robots UR5e Standard DH: `d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996] m`, `a = [0, -0.425, -0.3922, 0, 0, 0] m`) and `PandaArm` (Franka FCI Craig Modified DH: `d1 = 0.333`, `d3 = 0.316`, `d5 = 0.384`, `a4 = 0.0825`, `a5 = -0.0825`, `a7 = 0.088`, `d7 = 0.107 m`) implement both `fk_poe` ($\prod_{i=1}^n \exp([\mathcal{S}_i] q_i) M$) and `fk_dh` / `fk_mdh`, agreeing to `< 1e-13` across random joint configurations.
- **Analytical Jacobians, Null-Space DLS IK & Lie-Bracket Spatial Accelerations**:
  - `space_jacobian`, `body_jacobian`, and `world_jacobian` match central finite differences to `< 3e-8`.
  - `dls_ik` uses adaptive Levenberg-Marquardt damping ($\lambda^2 = \max(\lambda_{\text{sing}}^2, \min(\lambda_0^2, \frac{1}{2}\|e\|^2))$) with null-space projection $(I - J^\# J)$ optimizing joint-limit margins and log-det Yoshikawa manipulability $w(q) = \sqrt{\det(J J^\top)}$.
  - `trajectory_ee_kinematics` propagates exact spatial twists $V_i$ and accelerations $\dot{V}_i = \text{Ad}_{T_{i-1,i}}\dot{V}_{i-1} + \text{ad}_{V_i}\mathcal{S}_i \dot{q}_i + \mathcal{S}_i \ddot{q}_i$ to compute 3D corner-well linear, Euler, and centrifugal accelerations (`multi_well_accelerations_3d`).

### 3.2 3D Capsule-AABB Continuous Swept Collision, Informed RRT\* & TOPP-RA (`collision.py`, `planner.py`, `toppra.py`)

- **Exact 3D Segment-to-AABB Distance (`segment_aabb_distance_3d`, `capsule_aabb_clearance_3d`, `edge_free_se3`)**: Evaluates exact piecewise-quadratic minimum distance over slab entry/exit intervals on $[0, 1]$ for all 3D link capsules + mounted pipette tip, certified continuously along joint chords via `kinematic_sweep_bound_se3` (formally proven in `lean/LabHandsLean.lean`: `sweptSegment6DoF_continuous_clearance_sound`).
- **6-DoF / 7-DoF `RRT-Connect` & `Informed RRT*`**: Combines bidirectional greedy extension with Gammell et al. (2014) prolate hyper-ellipsoid informed sampling (`sample_informed_ellipsoid`) once an initial solution $c_{\text{best}}$ is found, reducing 7-DoF Panda joint path cost by `23.6%` (`2.235 -> 1.708 rad`).
- **TOPP-RA Reachability Path Parameterization (`parameterize_toppra`)**: Smooths waypoints with C3-continuous `5*tau^4 - 6*tau^5 + 2*tau^6` corner blends, solves backward controllable sets $[x_{\min}[k], x_{\max}[k]]$ ($x = \dot{s}^2$) via exact 1D Fourier-Motzkin LP elimination and greedy forward pass, and applies an integral-preserving raised-cosine filter to satisfy per-joint velocity, acceleration, and jerk bounds.

### 3.3 2D Coupled Nonlinear Bessel Slosh Dynamics & `ZV` / `ZVD` / `EI` Input Shaping (`slosh_dynamics.py`)

- **Nonlinear Bessel Spherical-Pendulum ODE**: Models the fundamental cylindrical mode ($\xi_{11} = 1.8412$, $\omega_{11} = \sqrt{(g \xi_{11}/R)\tanh(\xi_{11} h / R)}$) with Stokes viscous boundary-layer damping $\zeta(\nu, R, h)$ and cubic rotary-slosh hardening under full 3D container acceleration $(a_x, a_y, g + a_z)$ in the tilted end-effector frame.
- **Command Shapers (`design_input_shaper`)**: Implements exact damped 2-impulse `ZV`, 3-impulse `ZVD`, and homotopy-Newton 3-impulse `EI` ($V(\omega_n, \zeta) = V_{\text{tol}} = 0.05$, $\partial V/\partial \omega|_{\omega_n} = 0$, $V(\omega_{1,2}, \zeta) = 0$). Positive impulse weights ($\sum A_k = 1, A_k > 0$) preserve all `TOPP-RA` joint velocity/acceleration/jerk bounds (`positiveShaper_preserves_kinematic_bound` in `lean/LabHandsLean.lean`).

### 3.4 3D Whitney/Coulomb Contact Mechanics & Active Spiral/Bayesian Tactile Insertion (`tactile.py`)

- **Symmetric Physical Contact Model (`ClearVesselBevel`, `evaluate_3d_coulomb_contact`, `check_whitney_jamming_wedging`)**: Applies the exact same narrow SBS polypropylene chamfer (`0.70 mm` width, `0.10 mm` radial clearance, `0.80 mm` capture radius), 3D Coulomb friction cone ($\mu \in [0.15, 0.45]$), angular tilt ($\theta \in [0^\circ, 2.5^\circ]$), and Whitney (1982) jamming/wedging laws across open-loop, passive RCC, local shear admittance, and active tactile regimes.
- **Active Spiral Probing + Bayesian Particle Filter (`BayesianHoleParticleFilter`, `run_active_tactile_insertion`)**: Maintains 256 weighted hypotheses inside the `4.5 mm` SBS half-pitch Voronoi disk; flat-rim hits ($f_z > 0, \|\mathbf{f}_{xy}\| \approx 0$) down-weight nearby hole centers while Archimedean spiral micro-hop probes search outward until chamfer shear wrenches localize the bore and 6-DoF admittance seats the tip (`99.0%` success across 500 trials).

---

## 4. Benchmarks (`benchmarks/results/frontier_benchmark_summary.json`)

| Domain | Metric | Measured Result |
|---|---|---|
| **6-DoF UR5e / 7-DoF Panda Planning** | Cluttered 3D pick-and-place success | `100.0%` (`< 2.2 um` IK error, `>= 5.0 mm` continuous 3D clearance) |
| **Informed RRT\* + TOPP-RA (Panda)** | Joint path cost / mean cycle time | `2.235 -> 1.708 rad` (`-23.6%`) / `0.873 s` |
| **6-DoF Slosh (`+-25%` Mismatch, `n=48`)** | Spill rate (`toppra_unshaped` -> `ZV` -> `EI`) | `16.67%` -> `12.50%` -> **`8.33%`** (`0.264 -> 0.026 mm` ringdown) |
| **3D Tactile Insertion (`500` trials, `0.70 mm` chamfer)** | `open_loop` / `passive_rcc` / `local_shear` / **`active_spiral_bayesian`** | `1.2%` / `4.0%` / `19.0%` / **`99.0%`** (`0.045 mm` mean error, `0.602 N` force) |
| **LeRobot & Opentrons Interfaces** | Real `pipette-mujoco`, `LeRobot v3`, `FineBio` & `opentrons.simulate` 8.8.2 | `(100, 7)` `float32` state/action & `(T, 3)` `float64` `mm` deck path verified |

---

## 5. History

| Date (ISO-8601) | Commit / Ref | Summary of Changes |
|---|---|---|
| `2026-11-18` | `eb4d967` (`main`) | Initial 2D SBS 96-well RRT + shortcut demo & curated visuo-tactile resource map. |
| `2026-11-24` | `a6300e7` (`main`) | Closed `#26..#31`: exact continuous `segment_aabb_clearance`, Lean 4 Lipschitz swept-segment proof, Viterbi `plan_well_tour`, corner-well slosh scaling, per-labware Z-rim/tip `plan_3d_pick_place`, and `autolab-wire/v1` telemetry. |
| `2026-11-25` | `53d7ac4` (`main`) | Added 12-entry `OPEN_MANIPULATION_DATASETS` registry (`lab_hands/datasets.py`), SPDX commercial-use compliance gates, and LeRobot/FineBio safety auditors. |
| `2026-11-27` | `feat/frontier-6dof-zvd-slosh-coulomb-tactile` | Upgraded to exact PoE/DH/MDH 6-DoF `UR5e` & 7-DoF `Panda` + `OpentronsFlexGantry`, 3D capsule-AABB swept collision, `Informed RRT*` + `TOPP-RA`, 2D nonlinear Bessel slosh ODE + `ZV`/`ZVD`/`EI` input shaping, 3D Whitney/Coulomb + active spiral/Bayesian tactile insertion (`99.0%` on `0.70 mm` chamfer), and canonical LeRobot `(T, 7)` / Opentrons `(T, 3)` `mm` interfaces. |

---

## 6. Unique Properties & Lean 4 Guarantees

1. **Machine-Checked Lean 4 Guardrails (`lean/LabHandsLean.lean`)**: Proves `sweptSegment_continuous_clearance_sound`, `sweptSegment6DoF_continuous_clearance_sound`, `checkLabHands_guarantees_5mm_clearance`, `sloshScaled_predicted_rise_below_headspace` (bounding the surrogate linear-gravity prediction `< 0.7 * h_free` rather than claiming physical spill immunity in low-Bond-number microplates), `checkWhitneyNonWedging_sound`, and `positiveShaper_preserves_kinematic_bound` with zero `sorry` axioms.
2. **Un-Cherry-Picked Physical Symmetry**: Identical `0.70 mm` SBS chamfer geometry and 3D Coulomb friction mechanics across open-loop and tactile controllers; full factorial slosh evaluation including `+-25%` dispersion mismatch.
3. **Pure-NumPy Cross-OS Reproducibility**: Runs identically across Linux, Windows, and macOS on CPU.

---

## 7. 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` (`REPO_ID = "lab-hands"`, `LAYER = "L1"`) implements `process_packet(packet: dict) -> dict` for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) packet bus (`autolab-wire/v1`):

- **Upstream**: [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4` SQL & `lean-mcp/v1` gate, paired with `lab-rehearsal` and sibling `L1` [`stepbot`](https://github.com/yaniguan/stepbot)).
- **Node Execution (`lab-hands`)**: Executes continuity-aware IK, seeded RRT + `shortcut`, certified continuous `path_min_clearance` (`>= 5.0 mm`, `halt_reason = "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"`), and active spiral + Bayesian tactile insertion on `0.70 mm` SBS chamfers (`halt_reason = "L1_HANDS_TACTILE_INSERTION_OR_SEAL_BLOCK"` outside the `4.5 mm` Voronoi cell).
- **Downstream**: [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L0` INT8 / 4-bit bench runtime) and [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4` post-execution causal blame).
