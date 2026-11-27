# lab-hands

Give a lab agent *hands and touch*: move pipettes, tubes and well plates safely with **6-DoF UR5e**, **7-DoF Franka Emika Panda**, and **Opentrons Flex** kinematics, suppress open-well liquid slosh with **nonlinear Bessel-mode dynamics + ZV/ZVD/EI input shaping**, and use **active visuo-tactile spiral search + 3D Whitney/Coulomb compliance** for the last millimeter where depth cameras fail on clear glass and polypropylene.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), `lab-hands` owns **Layer `L1 · 6-DoF Manipulation, Slosh Control & Visuo-Tactile Insertion`**: robotic cobot arms, liquid-handler gantries, open-meniscus plate transport, and contact-guided tip/tube insertion. Cameras refract and specularly drop out on transparent vessels (`0.5..4.0 mm` localization error), while standard ANSI/SLAS polypropylene wells only offer a narrow `0.70 mm` rim chamfer (`0.80 mm` total capture radius). Solving manipulation honestly requires multi-DoF `SE(3)` collision-free motion generation, reachability-optimal jerk-bounded timing, robust input shaping against fill-height dispersion mismatch, and active force/torque search when a pipette tip lands on the flat rim outside the chamfer.

## Core Engineering Stack (`lab_hands/`)

Pure-`numpy` CPU implementation across Linux, Windows, and macOS (with optional live `opentrons.simulate` 8.8+ verification and machine-checked Lean 4 safety theorems in `lean/LabHandsLean.lean`):

1. **6-DoF / 7-DoF `SE(3)` Cobot Kinematics, Informed RRT\* & TOPP-RA (`kinematics_se3.py`, `collision.py`, `planner.py`, `toppra.py`)**:
   - Exact Product-of-Exponentials (`PoE`) cross-verified to `< 1e-13` against Standard DH for **6-DoF Universal Robots UR5e** (`UR5eArm`) and Craig Modified DH for **7-DoF Franka Emika Panda** (`PandaArm`), plus **4-DoF Opentrons Flex** (`OpentronsFlexGantry`, 3-axis Cartesian gantry + rotary gripper) and legacy planar `Arm2D` backward compatibility.
   - Analytical **Space, Body, and World Jacobians**, exact Lie-bracket spatial link/end-effector acceleration recursion (`trajectory_ee_kinematics`), and **adaptive Levenberg-Marquardt Damped Least-Squares (`DLS`) IK** (`dls_ik`) with null-space joint-limit centering and Yoshikawa manipulability ascent (`< 2.5 um` pose convergence).
   - **3D Multi-Link Capsule-vs-AABB Continuous Swept Collision Checking** (`BenchScene3D`, `capsule_aabb_clearance_3d`, `edge_free_se3`) using per-joint Jacobian-Lipschitz link sweep bounds formally verified in `lean/LabHandsLean.lean` (`sweptSegment6DoF_continuous_clearance_sound`).
   - **6-DoF / 7-DoF Bidirectional `RRT-Connect` + `Informed RRT*`** (prolate hyper-ellipsoid sampling) paired with **TOPP-RA** (`parameterize_toppra`: C3-continuous `5*tau^4 - 6*tau^5 + 2*tau^6` corner blends, exact backward/forward reachability LP passes, and integral-preserving raised-cosine jerk filtering respecting per-joint velocity, acceleration, and jerk bounds).

2. **2D Coupled Nonlinear Bessel Pendulum Slosh ODE & `ZV` / `ZVD` / `EI` Input Shaping (`slosh_dynamics.py`)**:
   - Physically grounded **fundamental cylindrical Bessel mode** ($\xi_{11} = 1.8412$, $\omega_{11} = \sqrt{(g \xi_{11}/R)\tanh(\xi_{11} h / R)}$) with **Galilei/Reynolds Stokes sidewall + bottom boundary-layer viscous damping** $\zeta(\nu, R, h)$ and surfactant surface-film damping, integrated as a 2D coupled nonlinear spherical pendulum with cubic rotary-slosh stiffening driven by full 3D end-effector linear + angular Euler ($\boldsymbol{\alpha} \times \mathbf{r}_w$) + centrifugal ($\boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}_w)$) accelerations across all outer corner wells of 96-well and 384-well microplates.
   - Exact damped **Zero-Vibration (`ZV`)**, **Zero-Vibration-Derivative (`ZVD`)**, and homotopy-Newton **Extra-Insensitive (`EI`, $V_{\text{tol}} = 5\%$)** positive impulse command shapers (`design_input_shaper`, `shape_toppra_trajectory`).

3. **3D Whitney/Coulomb Friction-Cone Contact & Active Visuo-Tactile Spiral Insertion (`tactile.py`)**:
   - Identical physical **SBS polypropylene `0.70 mm` chamfer** (`bore_clearance = 0.10 mm`, `capture_radius = 0.80 mm`, **not** an artificial `5.5 mm` funnel) and **3D Coulomb friction cone** ($\mu \in [0.15, 0.45]$, angular tilt $\theta \in [0^\circ, 2.5^\circ]$) shared fairly across `open_loop`, `passive_rcc_compliance`, `local_shear_admittance`, and `active_spiral_bayesian_tactile`.
   - Enforces **Whitney (1982) two-point peg-in-hole jamming & geometric wedging** (`insertion_depth < 2 * r_peg * mu` two-point friction-cone overlap).
   - **Active Archimedean Spiral Probing + 256-Particle Bayesian Contact Filter + 6-DoF F/T Admittance** (`run_active_tactile_insertion`): detects zero-shear flat-rim contact ($r > 0.80\text{ mm}$), executes bounded micro-hop spiral probes inside the `4.5 mm` SBS half-pitch Voronoi cell, fuses flat-rim exclusions and inward chamfer normal wrenches into a Bayesian posterior over hole center, aligns angular tilt, and seats cleanly.

4. **Canonical HuggingFace LeRobot v2.1/v3.0 & Opentrons 3D Gantry Tensor Interfaces (`trajectory.py`, `datasets.py`)**:
   - **LeRobot frame-tabular contract (`LeRobotFrameTable`, `validate_lerobot_frame_tensors`)**: `observation.state` `float32 (T, 7)` (`6` UR5e joints + `1` gripper/plunger), `action` `float32 (T, 7)`, `timestamp` `float32 (T,)`, `frame_index` `int64 (T,)`, `episode_index` `int64 (T,)`, `next.done` `bool (T,)`.
   - **Opentrons 3D Gantry contract (`OpentronsWaypointTrajectory`, `validate_opentrons_trajectory`)**: deck-frame pipette tip path `deck_tip_mm` `float64 (T, 3)` in millimeters within `(0..418, 0..353, 0..218) mm` (exact `0.00 mm` round-trip alignment with `opentrons.simulate` 8.8.2 via `DEFAULT_DECK_ORIGIN_IN_BASE_M = (0.14562, -0.17424, 0.0) m`), robot-base SI `base_tip_m` `float64 (T, 3)`, `base_pose_se3` `float64 (T, 6)` `(x, y, z, roll, pitch, yaw)`, and `plunger_ul` `float64 (T,)` in `uL`.

## Quickstart & CLI Demos

```bash
python demo/plan_to_well.py
python demo/plan_to_well.py --arm ur5e --seeds 3
python demo/plan_to_well.py --arm panda --seeds 3
python demo/tactile_insert.py --trials 500
python benchmarks/run_frontier_benchmarks.py
python -m pytest -q
```

Reference output (`demo/output.txt`, CPU):

```text
well 65 -> F5 (row-major), A9 (column-major)
straight joint-space move blocked by obstacles for 10 of 10 wells
trials: 100, success: 100 (100.0%)
plan time (rrt): mean 328.9 ms, max 747.1 ms
joint path length: raw 2.47 rad -> shortcut mean 2.13 rad (p95 2.42 rad, max 3.01 rad)
minimum link-obstacle clearance: min 5.0 mm, mean 6.5 mm
6-DoF/7-DoF SE(3) cobot (UR5E, 6-DoF): 6/6 (100.0%), RRT-Connect mean 1327.1 ms, Informed-RRT* cost reduction 1.8%, TOPP-RA mean cycle 2.08 s, min 3D capsule clearance 6.2 mm
```

With `--arm panda --seeds 2` (`7-DoF Franka Emika Panda`, benchtop mount `(-0.16, 0.0, 0.0) m`), `12/12` (`100.0%`) cluttered 3D transfers succeed with `RRT-Connect` mean `600.2 ms`, `Informed RRT*` path-cost reduction `8.2%`, `TOPP-RA` mean cycle `0.76 s`, and `8.5 mm` minimum 3D multi-link capsule clearance.

## Frontier Robotics Benchmarks (`benchmarks/results/frontier_benchmark_summary.json`)

### 1. 6-DoF / 7-DoF `SE(3)` Kinematics, Informed RRT\* & TOPP-RA

| Robot Chain | DoF | Cluttered 3D Success | Mean DLS IK Error | RRT-Connect Shortcut Cost | Informed RRT\* Cost (`init -> best`) | TOPP-RA Mean Duration | Min 3D Capsule Clearance |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Universal Robots UR5e** (`UR5eArm`) | 6 | `100.0%` | `0.92 um` | `3.060 rad` | `3.282 -> 3.065 rad` | `2.934 s` | `4.74 mm` (swept lower bound; `>= 5.0 mm` exact) |
| **Franka Emika Panda** (`PandaArm`) | 7 | `100.0%` | `2.17 um` | `1.754 rad` | `2.235 -> 1.708 rad` (`-23.6%`) | `0.873 s` | `5.03 mm` |
| **Opentrons Flex Gantry** (`OpentronsFlexGantry`) | 4 | Exact analytical FK/IK | `0.00 um` | Deck `(0..418, 0..353, 0..218) mm` | Verified vs `opentrons.simulate` 8.8.2 | - | - |

### 2. Full Factorial 6-DoF Nonlinear Bessel Slosh & Input Shaping Benchmark

Evaluated on `UR5eArm` high-speed stage indexing (`0.4112 s` base `TOPP-RA` move with simultaneous `90 mm` translation + `18 deg` wrist yaw exciting 3D Euler & centrifugal corner-well modes) across **4 fill fractions (`0.25, 0.50, 0.75, 0.85`) x 3 liquids (`water`, `PBS + 0.1% Tween-20 surfactant`, `50% glycerol`) x 2 labware formats (`96-well`, `384-well`) = 24 physical conditions** under both **nominal dispersion** and **`+-25%` fill-height + capillary wetting mismatch** (`48` pooled mismatch trials per controller):

| Regime | Controller | Spill Rate (`eta_max >= h_free`) | Mean `eta_max / h_free` | Peak `eta_max / h_free` | Mean Post-Move Ringdown | Mean Cycle Overhead |
|---|---|---:|---:|---:|---:|---:|
| **Nominal (`0%` error, `n=24`)** | `toppra_unshaped` | `16.67%` (`4/24`) | `0.6236` | `1.7615` | `0.1615 mm` | `+0.00%` (`0.411 s`) |
| **Nominal (`0%` error, `n=24`)** | `min_jerk_unshaped` | `12.50%` (`3/24`) | `0.4138` | `1.1734` | `0.2642 mm` | `+0.00%` (`0.411 s`) |
| **Nominal (`0%` error, `n=24`)** | `toppra_ZV` | `12.50%` (`3/24`) | `0.4154` | `1.1207` | `0.0228 mm` | `+10.24%` (`0.453 s`) |
| **Nominal (`0%` error, `n=24`)** | `toppra_ZVD` | **`4.17%` (`1/24`)** | **`0.4009`** | **`1.0633`** | `0.0166 mm` | `+19.69%` (`0.492 s`) |
| **Nominal (`0%` error, `n=24`)** | `toppra_EI` ($V_{\text{tol}}=5\%$) | **`4.17%` (`1/24`)** | `0.4028` | `1.0763` | **`0.0158 mm`** | `+19.66%` (`0.492 s`) |
| **`+-25%` Fill/Dispersion Mismatch (`n=48`)** | `toppra_unshaped` | `16.67%` (`8/48`) | `0.6236` | `1.7615` | `0.1615 mm` | `+0.00%` (`0.411 s`) |
| **`+-25%` Fill/Dispersion Mismatch (`n=48`)** | `min_jerk_unshaped` | `12.50%` (`6/48`) | `0.4138` | `1.1734` | `0.2642 mm` | `+0.00%` (`0.411 s`) |
| **`+-25%` Fill/Dispersion Mismatch (`n=48`)** | `toppra_ZV` | `12.50%` (`6/48`) | `0.4515` | `1.2586` | `0.0543 mm` | `+10.61%` (`0.455 s`) |
| **`+-25%` Fill/Dispersion Mismatch (`n=48`)** | `toppra_ZVD` | **`8.33%` (`4/48`)** | `0.4043` | `1.1210` | `0.0304 mm` | `+20.17%` (`0.494 s`) |
| **`+-25%` Fill/Dispersion Mismatch (`n=48`)** | `toppra_EI` ($V_{\text{tol}}=5\%$) | **`8.33%` (`4/48`)** | **`0.3996`** | **`1.1008`** | **`0.0260 mm`** | `+20.08%` (`0.494 s`) |

Under `+-25%` fill-height mismatch, 2-impulse `ZV` loses phase cancellation and overshoots unshaped minimum-jerk in peak meniscus rise (`1.2586` vs `1.1734`), whereas 3-impulse **`ZVD` and `EI` cut spills by `50%` vs unshaped `TOPP-RA` and `33%` vs `ZV`/minimum-jerk**, achieve `0%` spills across all `0.25..0.75` fills (only extreme `85%`-fill low-surface-tension surfactant corner wells exceed `h_free = 1.69 mm` due to quasi-static inertial tilt), and reduce post-move slosh ringdown **`10.2x`** (`0.2642 mm -> 0.0260 mm`).

### 3. 500-Trial 3D Whitney/Coulomb Tactile Insertion Benchmark (`0.70 mm` SBS Chamfer)

Stratified 500-trial full grid across **refractive RGB-D offset $r \in [0.0, 4.0]\text{ mm}$ x Coulomb friction $\mu \in [0.15, 0.45]$ x angular tilt $\theta \in [0.0^\circ, 2.5^\circ]$** on identical SBS polypropylene geometry (`bore_clearance = 0.10 mm`, `chamfer_width = 0.70 mm`, `capture_radius = 0.80 mm`):

| Insertion Regime | Success Rate (`500` trials) | Jam / Wedge Rate | Flat-Rim Hangup Rate | Mean Error (given success) | Mean Peak Axial Force | Mean Spiral Probes |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop` (fair `0.70 mm` chamfer) | `1.20%` (`6/500`) | `3.40%` | `80.00%` | `0.0428 mm` | `1.325 N` | `0.0` |
| `passive_rcc_compliance` | `4.00%` (`20/500`) | `4.00%` | `83.80%` | `0.0426 mm` | `1.364 N` | `0.0` |
| `local_shear_admittance` (no active search) | `19.00%` (`95/500`) | `0.00%` | `80.00%` | `0.0588 mm` | `1.273 N` | `0.0` |
| **`active_spiral_bayesian_tactile`** | **`99.00%` (`495/500`)** | **`0.00%`** | **`0.00%`** | **`0.0450 mm`** (`max 0.100 mm`) | **`0.602 N`** | `41.6` |

Because `(4.0 - 0.8) / 4.0 = 80.0%` of refractive camera offsets land on the flat polypropylene rim outside the `0.70 mm` chamfer where lateral shear slope vanishes, naive local shear admittance caps at `19.0%` (`80.0%` rim hangup). Active Archimedean spiral probing + 256-particle Bayesian contact localization recovers flat-rim landings to **`99.0%` (`495/500`)** at less than half the peak contact force (`0.602 N` vs `1.325 N`).

### 4. Verified Real Dataset & Live `opentrons.simulate` 8.8.2 Shapes & Robot-Specific Audits

- **`autobio-bench/pipette-mujoco`** (`100` frames @ `50 Hz` `ur5e`, `tests/fixtures/autobio_pipette_mujoco_ep0_100.json`), **`lerobot/droid_1.0.1`** (`15` frames @ `15 Hz` 7-DoF `Franka`, `tests/fixtures/droid_franka_ep0_15.json`, FK matches `observation.state.cartesian_position` to `0.00003 mm`), & **`lerobot/aloha_mobile_cabinet`**: `audit_lerobot_episode` dispatches published `UR5eRobotModel` (6-DoF) and `FrankaRobotModel` (7-DoF) from `meta/info.json` `robot_type` (`ROBOT_MODEL_UNKNOWN` / `safe: None` on unmodeled robots, `obstacle_check_status = "not_checked"` without `base_pose_in_deck`).
- **2D Planar 60-Task Minimum-Jerk Slosh Baseline (`community_extensions/slosh_minimum_jerk.py`)**: unified `slosh_natural_frequency` (`sqrt((xi_11*g/R)*tanh(xi_11*h/R))`) + Bond-number regime flag (`Bo = 1.6212 < 5.0 -> SLOSH_MODEL_OUT_OF_RANGE` on 96-well water): `slosh_bounded_min_jerk` achieves `0/60` (`0.0%`) predicted surrogate spills (`max_meniscus_to_headspace_ratio = 0.675 < 0.70`, mean `1.5939 s`, `min_clearance = 5.50 mm`) vs `60/60` (`100.0%`, mean `0.7904 s`) for unscaled trapezoidal profiles.
- **`FineBio` (`P01_01_01.txt`), `opentrons_labware_v1.json` & `opentrons.simulate` 8.8.2**: all 24 `STEPBOT_LABWARE_CATALOG` Z heights, well bottoms, and `tipLength - tipOverlap` overhangs are synced from `opentrons-shared-data` v1 (`scripts/sync_opentrons_labware.py`) and cross-checked against live `opentrons.simulate` `p300_single_gen2` runlogs (`deck_tip_mm` `float64 (T, 3)` strictly inside `(0..418, 0..353, 0..218) mm`).

## Datasets

Typed dataset registry (`lab_hands/datasets.py` + `scripts/fetch_open_datasets.py`, keeping downloaded metadata and episode shards gitignored under `data/open_datasets/`) with machine-checked SPDX licensing, `commercial_use_ok` guardrails, canonical LeRobot `(T, 7)` tensor projection, and robot-specific `UR5e` / `Franka` (`robot_type`) kinematic & Bond-number-flagged slosh episode auditing:

| Dataset ID | Dataset | Modality | Venue / Year | SPDX License | Commercial OK | What |
|---|---|---|---|---|---|---|
| `finebio` | [FineBio](https://github.com/aistairc/FineBio) | `wetlab_video` | CVPR 2024 / IJCV 2025 | `LicenseRef-FineBio-NC` | **No** (**non-commercial**) | Multi-view (egocentric + 5 benchtop views) PCR & micropipetting videos with hierarchical step, atomic-operation, and manipulated/affected labware annotations |
| `droid` | [DROID](https://droid-dataset.github.io/) | `kinematics` | RSS 2024 | `CC-BY-4.0` | Yes | 76k+ distributed 7-DoF Franka Panda teleoperated trajectories across 564 scenes ([`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)) |
| `lerobot` | [LeRobot v3.0](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet) | `kinematics` | HuggingFace 2025 | `Apache-2.0` | Yes | Standardized 50 Hz Apache Parquet + AV1 robot imitation episodes (`observation.state`, `action`, `timestamp`) for ALOHA & SO-ARM100 |
| `autobio_pipette_mujoco` | [AutoBio Pipette MuJoCo](https://huggingface.co/datasets/autobio-bench/pipette-mujoco) | `kinematics` | AutoBio-Bench 2025 | `MIT` | Yes | 100 simulated 50 Hz UR5e wet-lab micropipetting episodes (71,014 frames in LeRobot v2.1 Parquet schema) |
| `transcg` | [TransCG](https://github.com/Galaxies99/TransCG) | `transparent_depth` | RA-L / ICRA 2023 | `CC-BY-NC-SA-4.0` | **No** (**non-commercial**) | 57k real RGB-D frames of 51 transparent containers with ground-truth depth completion & grasp labels |
| `transproteus` | [TransProteus](https://github.com/sagieppel/TransProteus_Paper_Code) | `transparent_depth` | Digital Discovery 2024 | `CC-BY-4.0` | Yes | 50k+ annotated scenes of liquids, phase boundaries, and solids inside transparent chemical vessels with 3D XYZ & depth maps |
| `clearpose` | [ClearPose](https://github.com/opipari/ClearPose) | `transparent_depth` | ECCV / 2024 | `CC-BY-4.0` | Yes | 350k+ labeled real RGB-D frames covering 63 transparent glassware & reagent vessels under liquid fills and heavy clutter |
| `layered_depth` | [LayeredDepth](https://huggingface.co/datasets/princeton-vl/LayeredDepth) | `transparent_depth` | ICCV 2025 | `CC0-1.0` | Yes | Multi-layer depth benchmark resolving front-surface glass geometry alongside transmitted interior/background surfaces |
| `tvl` | [TVL (Touch-Vision-Language)](https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset) | `visuo_tactile` | ICML 2024 | `Apache-2.0` | Yes | 44k paired in-the-wild DIGIT optical tactile and RGB frames aligned with contact-property language annotations |
| `tactx` | [TactX](https://huggingface.co/datasets/TactX/TactX) | `visuo_tactile` | TactX 2025 | `CC-BY-4.0` | Yes | 544 teleoperated Franka episodes across 3 tactile sensors (`Daimon`, `eFlesh`, `FlexiTac`) and 4 contact-rich insertion/manipulation tasks |
| `feelsight` | [FeelSight / NeuralFeels](https://huggingface.co/datasets/suddhu/Feelsight) | `visuo_tactile` | Meta AI 2024 | `MIT` | Yes | 70 real & simulated DIGIT visuo-tactile + proprioception contact sequences for sub-millimeter pose tracking under occlusion |
| `octopi_physiclear` | [Octopi & PhysiCLeAR](https://github.com/clear-nus/octopi-1.5) | `visuo_tactile` | RSS 2024 / RSS 2025 | `LicenseRef-Octopi-Research` | **No** (**non-commercial**) | GelSight tactile video suite with physical compliance, roughness, and hardness annotations for tactile-LLM reasoning |

Additional foundational references: [Open X-Embodiment](https://robotics-transformer-x.github.io/) (mixed per sub-dataset), [ClearGrasp](https://sites.google.com/view/cleargrasp), [Vector-LabPics](https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net), [Touch-Slide (DIGIT)](https://huggingface.co/datasets/facebook/touch-slide) (`CC-BY-NC-4.0`, **non-commercial**), [Touch and Go](https://touch-and-go.github.io/), and [ObjectFolder](https://objectfolder.stanford.edu/).

```bash
python scripts/fetch_open_datasets.py --list --commercial-only --min-year 2024
python scripts/fetch_open_datasets.py --verify-urls
python scripts/fetch_open_datasets.py --dataset autobio_pipette_mujoco --max-rows 100
```

Licenses marked **non-commercial** are fine for research but blocked automatically when `--commercial-only` or `check_commercial_compliance()` is active.

## Open-source repos

Stars as of 2026-11-18.

| Repo | What | Stars | License |
|---|---|---:|---|
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | LeRobot: Making AI for Robotics more accessible with end-to-end learning | 28,954 | Apache-2.0 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | - | 14,691 | Apache-2.0 |
| [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Standard Open Arm 100 | 8,156 | Apache-2.0 |
| [openvla/openvla](https://github.com/openvla/openvla) | OpenVLA: An open-source vision-language-action model for robotic manipulation. | 7,317 | MIT |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | [RSS 2023] Diffusion Policy Visuomotor Policy Learning via Action Diffusion | 4,736 | MIT |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | - | 2,347 | MIT |
| [ompl/ompl](https://github.com/ompl/ompl) | The Open Motion Planning Library (OMPL) | 2,207 | custom (see repo) |
| [moveit/moveit2](https://github.com/moveit/moveit2) | :robot: MoveIt for ROS 2 | 2,123 | BSD-3-Clause |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | CUDA Accelerated Robot Library | 2,027 | Apache-2.0 |
| [octo-models/octo](https://github.com/octo-models/octo) | Octo is a transformer-based robot policy trained on a diverse mix of 800k robot... | 1,846 | MIT |
| [PyLabRobot/pylabrobot](https://github.com/PyLabRobot/pylabrobot) | interactive & hardware agnostic SDK for lab automation | 657 | MIT |
| [Opentrons/opentrons](https://github.com/Opentrons/opentrons) | Software for writing protocols and running them on the Opentrons Flex and Opentrons OT-2 | 537 | Apache-2.0 |
| [facebookresearch/tacto](https://github.com/facebookresearch/tacto) | Simulator of vision-based tactile sensors. (archived) | 471 | MIT |
| [Shreeyak/cleargrasp](https://github.com/Shreeyak/cleargrasp) | Official repository for the paper "ClearGrasp: 3D Shape Estimation of Transparent... | 332 | Apache-2.0 |
| [facebookresearch/sparsh](https://github.com/facebookresearch/sparsh) | Sparsh Self-supervised touch representations for vision-based tactile sensing (archived) | 225 | CC BY-NC 4.0 (**non-commercial**) |
| [gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics) | GelSight SDK for robotic sensors | 207 | GPL-3.0 |
| [Galaxies99/TransCG](https://github.com/Galaxies99/TransCG) | [RAL 2022 & ICRA 2023] TransCG: A Large-Scale Real-World Dataset for Transparent... | 122 | CC BY-NC-SA 4.0 (**non-commercial**) |
| [facebookresearch/digit-interface](https://github.com/facebookresearch/digit-interface) | Python interface for the DIGIT tactile sensor (archived) | 94 | CC BY-NC 4.0 (**non-commercial**) |
| [sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net](https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net) | Detecting and segmenting and classifying materials inside mostly transparent vessels... | 25 | MIT |

## Papers

| Paper | Venue | Link |
|---|---|---|
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | RSS 2023 | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation | arXiv 2024 | [arXiv:2401.02117](https://arxiv.org/abs/2401.02117) |
| Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | RSS 2023 | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) |
| SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | arXiv 2025 | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) |
| $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control | arXiv 2024 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| OpenVLA: An Open-Source Vision-Language-Action Model | CoRL 2024 | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| Octo: An Open-Source Generalist Robot Policy | RSS 2024 | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | ICRA 2024 | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) |
| cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation | arXiv 2023 | [arXiv:2310.17274](https://arxiv.org/abs/2310.17274) |
| DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation | RA-L 2020 | [arXiv:2005.14679](https://arxiv.org/abs/2005.14679) |
| Sparsh: Self-supervised touch representations for vision-based tactile sensing | CoRL 2024 | [arXiv:2410.24090](https://arxiv.org/abs/2410.24090) |
| Touch and Go: Learning from Human-Collected Vision and Touch | NeurIPS 2022 D&B | [arXiv:2211.12498](https://arxiv.org/abs/2211.12498) |
| ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation | ICRA 2020 | [arXiv:1910.02550](https://arxiv.org/abs/1910.02550) |
| TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline | RA-L 2022 | [arXiv:2202.08471](https://arxiv.org/abs/2202.08471) |
| Computer Vision for Recognition of Materials and Vessels in Chemistry Lab Settings and the Vector-LabPics Data Set | ACS Central Science 2020 | [DOI](https://doi.org/10.1021/acscentsci.0c00460) |

## Conferences and workshops

- [CoRL - Conference on Robot Learning](https://www.corl.org/)
- [RSS - Robotics: Science and Systems](https://roboticsconference.org/)
- [ICRA - IEEE Intl. Conference on Robotics and Automation](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra)
- [NeurIPS (main track, Datasets and Benchmarks, workshops)](https://neurips.cc/)
- [SLAS - Society for Laboratory Automation and Screening](https://www.slas.org/)

## Progress

Recent results, newest first. Date = first arXiv version; each line is a short summary of the abstract.

| Date | Paper | In one line |
|---|---|---|
| 2026-11-09 | [Robo-MLT: Dual-System Robotic Manipulation for Long-Horizon Medical Laboratory Tasks](https://arxiv.org/abs/2611.13039) | Robo-MLT: couples workflow monitoring with learned actions for long-horizon medical lab tasks. |
| 2026-10-03 | [AgenticTactileVLA: Contact-Guided Execution-Time Supervision for Generalizable Dexterous Manipulation without VLA Retraining](https://arxiv.org/abs/2610.04391) | AgenticTactileVLA: touch-guided supervision at execution time, without retraining the policy. |
| 2026-09-30 | [Tool-Policy Co-Design for Powder Weighing in Laboratory Automation](https://arxiv.org/abs/2609.39797) | Tool-policy co-design for powder weighing: optimizes the dispensing tool shape and its control policy together. |
| 2026-07-22 | [Scalable Low-Cost Laboratory Automation: A Digital Twin-Integrated Robotic Platform for Autonomous Liquid Handling (RAINBOT)](https://arxiv.org/abs/2607.20662) | RAINBOT: a low-cost liquid-handling robot built from a consumer 3D printer, with a digital twin. |
| 2026-06-11 | [Pipette: An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Robotics](https://arxiv.org/abs/2606.12936) | Pipette: a simulation platform, benchmark and data augmentation pipeline for wet-lab robot learning. |
| 2025-07-20 | [Touch in the Wild: Learning Fine-Grained Manipulation with a Portable Visuo-Tactile Gripper](https://arxiv.org/abs/2507.15062) | Touch in the Wild: a portable visuo-tactile gripper for collecting fine manipulation demos. |

## System Profile: Object · Data · Model · Training · History · Unique Properties

`lab-hands` is part of the [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ecosystem ([Project Studio](https://yaniguan.github.io/auto-lab/)) at **Layer `L1 · Protocol Compiler & Visuo-Tactile Manipulation`**.

- **Object**: Physical 6-DoF/7-DoF cobot and gantry manipulation, open-meniscus slosh suppression, and active visuo-tactile insertion layer moving pipettes, tubes, and microplates safely around cluttered decks (`lab_hands/`).
- **Data**: Parameterized ANSI/SLAS labware footprints (`PlateGeometry` + 24-entry `STEPBOT_LABWARE_CATALOG`), 3D AABB + vessel-rim `BenchScene3D`, canonical HuggingFace LeRobot v2.1/v3.0 `(T, 7)` `float32` tensors (`LeRobotFrameTable`), canonical Opentrons `(T, 3)` `mm` deck-gantry trajectories (`OpentronsWaypointTrajectory`), `lab-hands/trajectory/v1` JSONL artifacts, and the 12-entry typed `OPEN_MANIPULATION_DATASETS` registry (`lab_hands/datasets.py`).
- **Model**: Product-of-Exponentials / DH / Modified-DH `UR5eArm` (6-DoF), `PandaArm` (7-DoF), and `OpentronsFlexGantry` (4-DoF) chains with analytical Space/Body/World Jacobians & null-space DLS IK (`kinematics_se3.py`), 3D multi-link capsule-AABB continuous swept collision checking (`collision.py`), 6-DoF/7-DoF `RRT-Connect` & `Informed RRT*` (`planner.py`), reachability-optimal jerk-bounded `TOPP-RA` (`toppra.py`), 2D coupled nonlinear Bessel slosh ODE + `ZV`/`ZVD`/`EI` input shapers (`slosh_dynamics.py`), and 3D Whitney/Coulomb contact mechanics + active Archimedean spiral / 256-particle Bayesian tactile insertion (`tactile.py`).
- **Training & Benchmarks**: Deterministic analytical, sampling-based, input-shaped, and Bayesian force/torque baselines: `100.0%` cluttered 3D planning on `UR5e` & `Panda`, `8.33%` vs `16.67%` spill rate (`10.2x` ringdown suppression) with `EI` input shaping under `+-25%` fill-height mismatch, and `99.0%` (`495/500`) active tactile insertion across `0..4 mm` offsets on narrow `0.70 mm` SBS chamfers (vs `1.2%` open-loop and `19.0%` local shear admittance).
- **History**: `eb4d967` (2026-11-18, planar 2-DoF RRT baseline) -> `a6300e7` (continuous capsule clearance, Lean 4 Lipschitz proofs, Viterbi IK tours, `autolab-wire/v1`) -> `53d7ac4` (12-entry open manipulation dataset registry) -> frontier 6-DoF/7-DoF `UR5e`/`Panda` PoE + `Informed RRT*` + `TOPP-RA`, nonlinear Bessel slosh + `ZV`/`ZVD`/`EI` input shaping, 3D Whitney/Coulomb + active spiral/Bayesian tactile insertion, and canonical LeRobot/Opentrons tensor contracts.
- **Unique Properties**: Pure-`numpy` CPU reproducibility across Linux/Windows/macOS, machine-checked Lean 4 safety theorems (`lean/LabHandsLean.lean` with zero `sorry` axioms), exact `0.00 mm` alignment with `opentrons.simulate` 8.8+, and direct `autolab-wire/v1` integration with [`lab-verify`](https://github.com/yaniguan/lab-verify), [`stepbot`](https://github.com/yaniguan/stepbot), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), [`lab-feet`](https://github.com/yaniguan/lab-feet), and [`lab-attrib`](https://github.com/yaniguan/lab-attrib).

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` (`REPO_ID = "lab-hands"`, `LAYER = "L1"`) exposes `process_packet(packet: dict) -> dict` for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) wire bus (`autolab-wire/v1`):

- **Wire Topology**: Upstream [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4` SQL & `lean-mcp/v1` gate, alongside `lab-rehearsal` and sibling `L1` [`stepbot`](https://github.com/yaniguan/stepbot)) -> **`lab-hands`** (`L1` continuity-aware IK, seeded RRT & shortcutting, certified continuous clearance, and active tactile bevel insertion) -> downstream [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L0` INT8 / 4-bit bench runtime) -> [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4` post-execution causal blame, NNLS XRD deconvolution & SHA-256 certified ledger).
- **Packet Gate & Payload**: Validates all numeric packet fields (`seed`, `clearance_mm`, `capsule_radius_m`, `bevel_capture_radius_m`) without raising, plans to `max(5.0 mm, clearance_mm)` so every `PASS` packet satisfies `measured_min_clearance_mm >= requested_min_clearance_mm` (`halt_reason = "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"`), enforces tactile seal seating inside the `4.5 mm` SBS Voronoi search limit (`halt_reason = "L1_HANDS_TACTILE_INSERTION_OR_SEAL_BLOCK"`), emits `measured_min_clearance_mm`, `start_q`/`end_q`, and `tactile` slip/force telemetry in `out["manipulation"]`, and records `invariant = "arm_path_collision_free_with_5mm_margin"` in `bottom_reasoning_traces`.
- **15-Repository Stack (`L0`-`L6`)**: `lab-ears` (`L2`) · `lab-ask-why`, `lab-decide` (`L6`) · `lab-automation-atlas`, `buy-or-make`, `labshift` (`L5`) · `lab-rehearsal`, `lab-verify`, `lab-attrib` (`L4`) · `stepbot`, `lab-hands` (`L1`) · `lab-feet` (`L0`) · `twinstep`, `labfill`, `labbench-posttrain` (`L3`).

## Repository Layout

```text
lab-hands/
├── .github/workflows/
│   └── demo.yml                  # Cross-OS (Linux/Windows/macOS) pytest + 6-DoF/7-DoF demo CI
├── benchmarks/
│   ├── results/
│   │   └── frontier_benchmark_summary.json  # Committed 6-DoF/7-DoF, slosh, tactile & dataset metrics
│   └── run_frontier_benchmarks.py           # Reproducible end-to-end benchmark runner
├── community_extensions/
│   └── slosh_minimum_jerk.py     # Open-meniscus pendulum-slosh minimum-jerk time-scaler
├── demo/
│   ├── plan_to_well.py           # CLI benchmark (arm2d / ur5e / panda, Informed RRT*, TOPP-RA)
│   ├── tactile_insert.py         # 4-regime 3D Whitney/Coulomb & active spiral tactile benchmark CLI
│   └── output.txt                # Reference CPU benchmark metrics
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── lab_hands/
│   ├── data/
│   │   └── opentrons_labware_v1.json # Synced opentrons-shared-data v1 Z/bottom/tip definitions
│   ├── __init__.py               # Public package exports
│   ├── collision.py              # 2D & 3D multi-link capsule-AABB clearance & Lipschitz swept checks
│   ├── datasets.py               # OpenManipulationDataset registry, UR5e/Franka LeRobot & FineBio audits
│   ├── kinematics.py             # UR5e/Franka published DH models, Arm2D/ArmSE3 & joint limits
│   ├── kinematics_se3.py         # Exact PoE/DH/MDH UR5e (6-DoF), Panda (7-DoF), Opentrons Flex & DLS IK
│   ├── labware.py                # PlateGeometry, DeckLayout & 24-entry Opentrons-synced labware catalog
│   ├── planner.py                # 2D & 6/7-DoF RRT, RRT-Connect, Informed RRT* & shortcutting
│   ├── slosh_dynamics.py         # 2D nonlinear Bessel spherical-pendulum ODE & ZV/ZVD/EI input shaping
│   ├── tactile.py                # 3D Whitney/Coulomb mechanics & active spiral + Bayesian tactile control
│   ├── toppra.py                 # TOPP-RA reachability path parameterization with C3 blends & jerk filtering
│   └── trajectory.py             # Shared slosh dispersion/Bond number, LeRobotFrameTable & Opentrons paths
├── lean/
│   ├── LabHandsLean.lean         # Lean 4 proofs for 6-DoF swept clearance, Whitney non-wedging & shapers
│   └── README.md
├── scripts/
│   ├── fetch_open_datasets.py    # CLI for dataset listing, SPDX commercial checks & HF/FineBio sample pulls
│   └── sync_opentrons_labware.py # Syncs 24 catalog entries from opentrons-shared-data v1
├── tests/
│   ├── fixtures/                 # Real AutoBio UR5e (MIT) & DROID Franka (CC-BY-4.0) episode fixtures
│   ├── test_audit_issues_34_37.py# Acceptance-criteria tests for #34..#37
│   ├── test_autolab_wire.py      # autolab-wire/v1 L1 requested-clearance & validation tests
│   ├── test_community_extensions.py
│   ├── test_frontier_upgrades.py # 6/7-DoF PoE/Jacobian/IK/TOPP-RA, Bessel ZV/ZVD/EI, Whitney & shape tests
│   ├── test_open_datasets_2026.py# Open dataset registry, non-commercial flags & SE(3)/slosh audit tests
│   ├── test_phase2_audit_fixes.py# Regression suite for issues #26..#31 & #34
│   └── test_plan_to_well.py      # Core kinematics, collision, planning & CLI export tests
├── autolab_wire.py               # 15-repo Auto-Lab wire node (REPO_ID="lab-hands", LAYER="L1")
├── LICENSE                       # MIT License
├── pyproject.toml                # Pytest configuration
├── README.md                     # Overview, frontier benchmarks, and curated manipulation/tactile resources
└── requirements.txt              # numpy>=1.26, pytest>=8
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
