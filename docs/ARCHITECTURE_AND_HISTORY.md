# `lab-hands` -- Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L1 · Protocol Compiler & Visuo-Tactile Manipulation`  
> **License**: MIT (repository code; external datasets and repositories retain their respective licenses)

---

## 1. Object

`lab-hands` is the **Visuo-Tactile Manipulation & Motion Planning (`L1`)** component of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`-`L6` Brain/Body stack, `L1` bridges symbolic benchtop actions to collision-free physical execution: moving pipettes, tubes, and microplates across dense lab decks while framing visuo-tactile feedback for the "last centimeter" of manipulation, where depth cameras fail on transparent borosilicate glass, clear polypropylene tubes, and refractive liquid menisci.

### Inputs and Outputs (`lab_hands/` & `demo/plan_to_well.py`)

- **Inputs**:
  - 1-based well indices (`1..N`, row-major or column-major) or alphanumeric SBS well references (`A1`-`H12`, `alias:A1`) across the 24-entry `STEPBOT_LABWARE_CATALOG` (`6`..`384`-well microplates, reservoirs, `120 mm` Falcon conical racks, and `10`..`1000 uL` tipracks).
  - Planar 2-link arm (`Arm2D`, `L1 = 0.30 m`, `L2 = 0.25 m`, configurable `clearance_m` capsule radius and `margin_m = 0.005 m`) and 6-DOF spherical-wrist cobot kinematics (`ArmSE3`, `PoseSE3`), aspiration source (`TUBE = (-0.05, 0.40) m`), AABB deck obstacles (`OBSTACLES`), and deterministic RNG seeds.
- **Outputs**:
  - Continuity-aware inverse-kinematics joint configurations (`select_ik_branch`) and Viterbi-optimal multi-well tours (`plan_well_tour`), Lipschitz-certified collision-free waypoint trajectories (`rrt`, `rrt_connect`, `shortcut`), 3D / 6-DOF `SE(3)` safe-retract vertical-horizontal-vertical transfers (`plan_transfer_3d`, `plan_3d_pick_place`), corner-well slosh-bounded trajectories (`parameterize_path`, `plan_slosh_bounded_min_jerk`), closed-loop tactile bevel insertions with slip/seal telemetry (`run_tactile_insertion`), and `autolab-wire/v1` / `lab-hands/trajectory/v1` JSON artifacts.

### Scope Boundaries

- **Implemented today**: Pure-`numpy` exact continuous 2D capsule-vs-AABB clearance (`segment_aabb_clearance`, `batch_min_clearance`), Jacobian-Lipschitz swept-segment verification (`edge_free`, `path_min_clearance`, formally verified in `lean/LabHandsLean.lean`), continuity-aware IK & Viterbi multi-well tour planning (`select_ik_branch`, `plan_well_tour`), per-labware Z-rim + attached-tip 3D / 6-DOF `SE(3)` safe transit (`ArmSE3`, `PoseSE3`, `plan_3d_pick_place`), rigid-plate corner-well cylindrical slosh time-scaling, closed-loop compliant tactile bevel insertion with axial slip/seal detection, and `autolab-wire/v1` packet execution.
- **Out of scope / next stages**: Full mesh-vs-mesh 3D physics contact dynamics in MuJoCo (`SO-ARM100` / `ALOHA`) and demonstration-trained visuomotor or visuo-tactile policies (`ACT`, `Diffusion Policy`, `OpenVLA`, `SmolVLA`, `Sparsh`).

---

## 2. Data

### 2.1 Built-in Geometric & Labware Specification (`lab_hands/labware.py`)

All geometry is deterministic, self-contained, and anchored to ANSI/SLAS (SBS) labware dimensions:

- **SBS 96-well & 24-entry `stepbot` catalog**:
  - Default `8` rows (`ROWS = "ABCDEFGH"`) x `12` columns (`1..12`), `PITCH = 0.009 m` (`9.0 mm`), `PLATE_ORIGIN = (0.20, 0.05) m`, plus full 24-entry `STEPBOT_LABWARE_CATALOG` carrying per-labware `z_bottom_m`, `z_rim_m` (up to `0.120 m` for 15/50 mL Falcon racks and `0.085 m` for `1000 uL` filter tipracks), `well_radius_m`, and `is_tiprack`.
  - Dual index-to-well mapping (`well_name(index, order)`): supports both `'row'` (`65 -> F5`) and `'col'` (`65 -> A9`) 1-based conventions.
- **Arm & cluttered benchtop layout**:
  - Base shoulder joint at `(0.0, 0.0) m`; planar link lengths `L1 = 0.30 m`, `L2 = 0.25 m` (`|q2| <= 2.60 rad`); 6R `ArmSE3` cobot chain (`z_base = 0.10 m`, `l_upper = 0.35 m`, `l_forearm = 0.32 m`, `z_max = 0.300 m`).
  - Static AABB obstacles `(xmin, ymin, xmax, ymax)`:
    1. Tube rack wall: `(0.33, -0.08, 0.36, 0.12)`.
    2. Reagent bottle: `(0.15, 0.27, 0.24, 0.34)`.

### 2.2 Curated External Datasets & Licenses

`README.md` organizes nine foundational datasets across robot manipulation (Open X-Embodiment, LeRobot, AutoBio pipette MuJoCo), transparent vessels (TransCG, ClearGrasp, Vector-LabPics), and visuo-tactile sensing (Touch-Slide / DIGIT, Touch and Go, ObjectFolder), flagging non-commercial restrictions explicitly.

---

## 3. Model

### 3.1 Planar 2R & 6-DOF `SE(3)` Kinematics (`Arm2D`, `ArmSE3`, `PoseSE3`)

- **Planar 2R (`Arm2D.fk`, `Arm2D.ik`)**: Analytical forward and two-branch (`elbow in {+1, -1}`) inverse kinematics with `[-pi, pi)` angle wrapping and `|q2| <= 2.60 rad` self-collision limits.
- **Continuity-Aware IK & Viterbi Tour Planning (`select_ik_branch`, `plan_well_tour`)**: Orders valid IK branches by proximity to `prev_q` and solves a Viterbi dynamic program over multi-well tours so partially constrained deck slots select globally consistent elbow branches without unnecessary `> pi` mid-plate elbow flips.
- **6-DOF `SE(3)` Cobot Chain (`PoseSE3`, `ArmSE3`)**: Closed-form 6R spherical-wrist `fk_se3` / `ik_se3` and verticality guard `acos(R_zz) <= max_tilt_rad` preventing pipette-shaft wall binding inside `42 mm` deep-well blocks and `120 mm` Falcon conical tubes.

### 3.2 Continuous Capsule-AABB Clearance & Lipschitz Swept-Segment Verification (`lab_hands/collision.py`)

- **Exact 2D Capsule-AABB Distance (`segment_aabb_distance`, `segment_aabb_clearance`, `batch_min_clearance`)**: Vectorized Liang-Barsky closed-box hit test combined with exact endpoint-to-box and box-corner-to-segment projection minus link `clearance_m` (`capsule_radius`), eliminating discrete link-point corner tunneling across both `lab_hands` and `community_extensions/slosh_minimum_jerk.py`.
- **Certified Continuous Swept-Segment Check (`edge_free`, `path_min_clearance`)**: Bounds link displacement along joint chord `q_a -> q_b` by the kinematic Jacobian Lipschitz constant `(L1 + L2) * |dq1| + L2 * |dq2|`, inflates grid checks by half-step sweep radius `0.5 * sweep / (n - 1)` with adaptive midpoint bisection on narrow-band intervals, and pairs directly with the Lean 4 `sweptSegment_continuous_clearance_sound` proof in `lean/LabHandsLean.lean`.
- **Per-Labware 3D Rim & Tip Safety (`BenchScene`, `in_collision_3d`, `plan_3d_pick_place`)**: Synchronizes `z_safe_travel` with `plate.z_rim_m`, multi-labware `DeckLayout` heights, and mounted disposable tip length (`attached_tip_length_m`) so horizontal transits clear tall racks while free-space vertical motion outside labware footprints remains unobstructed.

### 3.3 RRT / RRT-Connect, Corner-Well Slosh Time-Scaling & Tactile Insertion

- **Motion Planning (`rrt`, `rrt_connect`, `shortcut`)**: Same-branch warmup prioritization avoids gratuitous homotopy flips while preserving cross-branch reachability around non-convex obstacles.
- **Analytical C2 Curve & Corner-Well Slosh Scaling (`parameterize_path`, `plan_slosh_bounded_min_jerk`)**: Evaluates C2-blended geometric curves analytically at minimum-jerk warped arc coordinates (`||dq/dt - qd||_inf < 1e-4`) and integrates cylindrical meniscus slosh across all four outer corner wells (`+-49.5 mm`, `+-31.5 mm`) under translational + Euler (`alpha_12 x r`) + centripetal (`-omega_12^2 r`) accelerations to keep meniscus rise `< 0.70 * h_free`.
- **Closed-Loop Tactile Insertion (`run_tactile_insertion`)**: Uses Jacobian-transpose 2D shear compliance within joint and obstacle limits, detecting flat-rim axial overload, lateral stalls, `axial_slip_mm`, and `seal_intact`.

---

## 4. Training & Benchmarks

`lab-hands` provides deterministic analytical, sampling-based, and compliant-control baselines on CPU:

| Benchmark | Result | Notes |
|---|---|---|
| Well index sanity check (`65`) | `F5` (row-major) / `A9` (column-major) | Verifies dual SBS 96-well conventions |
| Cluttered RRT / RRT-Connect (`100` trials) | `100 / 100` (`100.0%`) | Verified `>= 5.0 mm` continuous capsule-AABB clearance |
| Open-plate corner-well slosh (`60` tasks) | `0 / 60` spills (`< 0.70 * h_free`) | Includes Euler + centripetal acceleration at all 4 corner wells |
| Tactile bevel insertion (`100` trials) | `100 / 100` (`100.0%`, `< 0.10 mm` error) | Resolves `+-3 mm` transparent-vessel refractive camera offset |

---

## 5. History

| Date (ISO-8601) | Commit / Ref | Summary of Changes |
|---|---|---|
| `2026-11-18` | `eb4d967` (`main`) | Initial 2D SBS 96-well RRT + shortcut demo & curated visuo-tactile resource map. |
| `2026-11-24` | `lab_hands/` modularization & Phase-2 audit fixes | Added exact continuous `segment_aabb_clearance`, Lean 4 Lipschitz swept-segment proof (`LabHandsLean`), stateful IK & Viterbi `plan_well_tour`, corner-well rotational slosh scaling, 6-DOF `SE(3)` + per-labware Z-rim/tip `plan_3d_pick_place`, and `autolab-wire/v1` measured clearance & tactile telemetry. |

---

## 6. Unique Properties

### Distinctive Invariants & Guarantees

1. **Zero Corner-Tunneling Continuous Clearance**: Exact capsule-vs-AABB distance paired with kinematic Lipschitz segment subdivision guarantees no thin partition or obstacle corner can slip between discrete samples.
2. **Machine-Checked Lean 4 Guardrails (`lean/LabHandsLean.lean`)**: Proves `sweptSegment_continuous_clearance_sound`, `checkLabHands_guarantees_5mm_clearance`, and `sloshScaled_zero_well_spills` with zero `sorry` axioms.
3. **Pure-NumPy Cross-OS Reproducibility**: Runs identically across Linux, Windows, and macOS on CPU.

### Remaining Scope Boundaries

- **Analytical contact model**: Tactile bevel compliance uses an analytical spring-shear contact model rather than finite-element elastomer deformation or real-time GelSight/DIGIT optical rendering.
- **Quasi-rigid link assumption**: Structural link deflection is neglected relative to the `5.0 mm` obstacle margin.

### Ecosystem Interfaces Across Auto-Lab

- **Sibling `L1` (`stepbot`)**: [`stepbot`](https://github.com/yaniguan/stepbot) compiles liquid-handling steps for Cartesian gantries, sharing the 24-entry `STEPBOT_LABWARE_CATALOG` with `lab-hands`.
- **Upstream from `L2` (`lab-ears`) & `L4` (`lab-rehearsal`, `lab-verify`)**: Target wells, reagent vessels, and verified execution envelopes flow into `lab-hands`.
- **Downstream to `L0` (`lab-feet`) & `L4` (`lab-attrib`)**: Collision-free `manipulation` packets carrying measured path clearance and tactile seal/slip telemetry feed [`lab-feet`](https://github.com/yaniguan/lab-feet) and [`lab-attrib`](https://github.com/yaniguan/lab-attrib).

---

## 7. 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` (`REPO_ID = "lab-hands"`, `LAYER = "L1"`) implements `process_packet(packet: dict) -> dict` for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) packet bus (`autolab-wire/v1`):

- **Path Resolution & Isolation**: Resolves repository imports strictly via `REPO_DIR = Path(__file__).resolve().parent` without relying on `os.getcwd()`, deep-copies incoming packets, and passes upstream `halted=True` packets through unchanged.
- **Upstream & Downstream Dataflow**:
  - **Upstream**: [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4` default-deny SQL AST gate and `lean-mcp/v1` capability token minting, paired with `lab-rehearsal` and sibling `L1` [`stepbot`](https://github.com/yaniguan/stepbot)).
  - **Node Execution (`lab-hands`)**: Reads `start_well` (default `"A1"`), `target_well` (default `"H12"`), `clearance_mm` (default `8.5`), optional `start_q` (or chained `manipulation.end_q`), and optional `visual_offset_mm`. Executes continuity-aware 2-link IK, seeded RRT + `shortcut` smoothing, certified continuous `path_min_clearance`, and closed-loop tactile bevel insertion (`run_tactile_insertion`). Enforces `>= 5.0 mm` clearance (`halt_reason = "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"`) and tactile tip-seating integrity (`halt_reason = "L1_HANDS_TACTILE_INSERTION_OR_SEAL_BLOCK"` with `attribution_hint = "EXECUTION_ARTIFACT"` on bevel hangup or seal slip). Emits `out["manipulation"]` with `"measured_min_clearance_mm"`, `"requested_min_clearance_mm"`, `"clearance_mm"`, `"start_q"`, `"end_q"`, `"elbow_branch"`, and `"tactile"`.
  - **Downstream**: [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L0` INT8 / 4-bit bench runtime) and [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4` post-execution causal blame, separating `EXECUTION_ARTIFACT` tip/seal faults from scientific `CERTIFIED_NEGATIVE` results).
- **15-Repository Auto-Lab Stack**:
  - `L2`: `lab-ears`
  - `L6`: `lab-ask-why`, `lab-decide`
  - `L5`: `lab-automation-atlas`, `buy-or-make`, `labshift`
  - `L4`: `lab-rehearsal`, `lab-verify`, `lab-attrib`
  - `L1`: `stepbot`, `lab-hands`
  - `L0`: `lab-feet`
  - `L3`: `twinstep`, `labfill`, `labbench-posttrain`
