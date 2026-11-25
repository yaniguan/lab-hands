# lab-hands

Give a lab agent *hands and touch*: move pipettes, tubes and well plates safely, and use touch for the last centimeter, where cameras fail on clear glass and plastic.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the hands are planned: a robot arm, pipettes and well plates, and liquid-handler scripts. Touch lives in this repo too. Cameras struggle with transparent vessels, so the last centimeter needs touch, and it uses the same arms, tasks and data. This repo collects open hardware, policies, simulators and tactile datasets, and starts with a toy planner that reaches named wells.

## First attempt

A benchtop manipulation baseline (`lab_hands/`): a planar 2-link arm holds a pipette tip, starts at a reagent tube, and reaches named wells on standard ANSI/SLAS labware (6..384-well footprints aligned with `stepbot`'s 24-labware catalog) without clipping a reagent bottle or rack wall (`>= 5 mm` continuous segment-AABB clearance and joint limits). It solves both elbow IK branches, plans in joint space with forward `rrt` or bidirectional `rrt_connect`, shortcuts and C2-time-parameterizes paths under liquid-slosh acceleration bounds, verifies 2.5D safe-retract Z clearances, and exports `lab-hands/trajectory/v1` JSONL artifacts. `demo/tactile_insert.py` adds closed-loop compliant tactile bevel insertion resolving `+-3 mm` transparent-vessel visual error to `< 0.1 mm`. Pure numpy on CPU.

```bash
python demo/plan_to_well.py
python demo/plan_to_well.py --planner rrt_connect --out-jsonl runs/trajectories.jsonl
python demo/tactile_insert.py
python -m pytest -q
```

Output (CPU only, re-measured with continuous 5 mm safety margin):

```
well 65 -> F5 (row-major), A9 (column-major)
straight joint-space move blocked by obstacles for 10 of 10 wells
trials: 100, success: 100 (100.0%)
plan time (rrt): mean 63.7 ms, max 112.8 ms
joint path length: raw 2.56 rad -> shortcut mean 2.16 rad (p95 2.47 rad, max 5.14 rad)
minimum link-obstacle clearance: min 5.0 mm, mean 6.6 mm
```

With `--planner rrt_connect`, mean plan + shortcut time drops to `29.0 ms` (`max 88.1 ms`, `p95 2.29 rad`, `max 2.74 rad` with zero homotopy flips). In `demo/tactile_insert.py`, closed-loop 2D contact shear compliance reaches `100/100` (`100.0%`) insertion success (`mean 0.085 mm`, `max 0.093 mm` centering error vs. `0/100` for open-loop camera plunge). Next: porting to a 3D physics simulator with a multi-DoF URDF arm, then learning visuomotor/tactile policies (ACT or Diffusion Policy) from demonstrations.

## Datasets

Typed dataset registry (`lab_hands/datasets.py` + `scripts/fetch_open_datasets.py`, keeping downloaded metadata and episode shards gitignored under `data/open_datasets/`) with machine-checked SPDX licensing, `commercial_use_ok` guardrails, and `ArmSE3` / 96-well clearance & meniscus-slosh episode auditing:

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
python scripts/fetch_open_datasets.py --dataset autobio_pipette_mujoco --max-rows 64
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
| $π_0$: A Vision-Language-Action Flow Model for General Robot Control | arXiv 2024 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
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

- **Object**: Physical manipulation and contact-sensing layer moving pipettes, tubes, and microplates safely around benchtop obstacles (`lab_hands/`), paired with closed-loop compliant tactile insertion for transparent glass and plastic vessels (`demo/tactile_insert.py`).
- **Data**: Parameterized ANSI/SLAS labware footprints (`PlateGeometry` + 24-entry `STEPBOT_LABWARE_CATALOG` covering `6`..`384`-well plates, reservoirs, Falcon racks, and tipracks), AABB + 2.5D vessel-rim `BenchScene`, `lab-hands/trajectory/v1` JSONL artifacts, and a 12-entry typed `OPEN_MANIPULATION_DATASETS` registry (`lab_hands/datasets.py`) spanning `kinematics`, `transparent_depth`, `visuo_tactile`, and `wetlab_video` with LeRobot Parquet/JSONL and FineBio 3D benchtop episode loaders.
- **Model**: Analytical 2-link (`Arm2D`) and 6-DOF `SE(3)` spherical-wrist (`ArmSE3`, `PoseSE3`) kinematics, continuity-aware IK & Viterbi multi-well tour planning (`select_ik_branch`, `plan_well_tour`), exact continuous capsule-vs-AABB clearance (`segment_aabb_clearance`) & Jacobian-Lipschitz `edge_free`, per-labware Z-rim + attached-tip 3D safe transit (`plan_transfer_3d`, `plan_3d_pick_place`), analytical C2 corner-well slosh time parameterization (`parameterize_path`, `plan_slosh_bounded_min_jerk`), and Jacobian-transpose tactile servoing with slip/seal detection (`run_tactile_insertion`).
- **Training**: Classical sampling-based & compliant control baselines: `100/100` (`100.0%`) obstacle-blocked planning success (`>= 5.0 mm` verified continuous obstacle clearance), `60/60` zero-spill corner-well open-plate transports (`< 0.70 * h_free`), and `100/100` (`100.0%`) tactile insertion resolving `+-3 mm` visual error to `< 0.10 mm`.
- **History**: `eb4d967` (2026-11-18, `main` 2D SBS 96-well RRT + shortcut demo & curated visuo-tactile map) -> modular `lab_hands/` package adding exact continuous capsule-AABB clearance, Lean 4 Lipschitz swept-segment soundness, stateful IK & Viterbi tours, 6-DOF `SE(3)` per-labware Z-rim/tip planning, corner-well slosh scaling, tactile slip/seal telemetry, and `autolab-wire/v1` integration.
- **Unique Properties**: Pure-`numpy` CPU reproducibility across Linux/Windows/macOS, exact continuous segment-box clearance (`0` corner-tunneling gap), dual-order SBS well addressing (`65 -> F5` row-major vs `A9` column-major), and direct `autolab-wire/v1` integration with [`lab-verify`](https://github.com/yaniguan/lab-verify), [`stepbot`](https://github.com/yaniguan/stepbot), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), [`lab-feet`](https://github.com/yaniguan/lab-feet), and [`lab-attrib`](https://github.com/yaniguan/lab-attrib).

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` (`REPO_ID = "lab-hands"`, `LAYER = "L1"`) exposes `process_packet(packet: dict) -> dict` for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) wire bus (`autolab-wire/v1`):

- **Wire Topology**: Upstream [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4` SQL & `lean-mcp/v1` gate, alongside `lab-rehearsal` and sibling `L1` [`stepbot`](https://github.com/yaniguan/stepbot)) -> **`lab-hands`** (`L1` continuity-aware 2-link IK, seeded RRT & shortcutting, certified continuous clearance, and tactile bevel insertion) -> downstream [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L0` INT8 / 4-bit bench runtime) -> [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4` post-execution causal blame, NNLS XRD deconvolution & SHA-256 certified ledger).
- **Packet Gate & Payload**: Reads `start_well` (default `"A1"`), `target_well` (default `"H12"`), `clearance_mm` (default `8.5`), optional `start_q`, and optional `visual_offset_mm`; enforces `LabHandsLean` `>= 5.0 mm` continuous clearance (`halt_reason = "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK"`) and tactile seal seating (`halt_reason = "L1_HANDS_TACTILE_INSERTION_OR_SEAL_BLOCK"`), emits `measured_min_clearance_mm`, `start_q`/`end_q`, and `tactile` slip/force telemetry in `out["manipulation"]`, and records `invariant = "arm_path_collision_free_with_5mm_margin"` in `bottom_reasoning_traces`.
- **15-Repository Stack (`L0`-`L6`)**: `lab-ears` (`L2`) · `lab-ask-why`, `lab-decide` (`L6`) · `lab-automation-atlas`, `buy-or-make`, `labshift` (`L5`) · `lab-rehearsal`, `lab-verify`, `lab-attrib` (`L4`) · `stepbot`, `lab-hands` (`L1`) · `lab-feet` (`L0`) · `twinstep`, `labfill`, `labbench-posttrain` (`L3`).

## Repository Layout

```text
lab-hands/
├── .github/workflows/
│   └── demo.yml                  # Cross-OS (Linux/Windows/macOS) pytest + benchmark CI
├── community_extensions/
│   └── slosh_minimum_jerk.py     # Open-meniscus pendulum-slosh minimum-jerk time-scaler
├── demo/
│   ├── plan_to_well.py           # CLI benchmark (rrt / rrt_connect, JSONL export, clearance gates)
│   ├── tactile_insert.py         # Closed-loop compliant tactile bevel insertion demo
│   └── output.txt                # Reference 100-trial CPU metrics
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── lab_hands/
│   ├── __init__.py               # Public package exports
│   ├── collision.py              # Exact segment-AABB clearance, vectorized edge_free & 2.5D Z safety
│   ├── datasets.py               # OpenManipulationDataset registry, LeRobot/FineBio loaders & SE(3)/slosh audits
│   ├── kinematics.py             # Arm2D/ArmSE3 FK/IK, boundary clamping, wrapping & joint limits
│   ├── labware.py                # PlateGeometry, DeckLayout & 24-entry stepbot labware catalog
│   ├── planner.py                # Multi-branch RRT, bidirectional RRT-Connect & shortcutting
│   ├── tactile.py                # Simulated 2D contact shear & compliant bevel insertion controller
│   └── trajectory.py             # C2 slosh-safe time parameterization & lab-hands/trajectory/v1 JSONL
├── lean/
│   └── LabHandsLean.lean         # Lean 4 proofs for >= 5 mm clearance & open-meniscus slosh bounds
├── scripts/
│   └── fetch_open_datasets.py    # CLI for dataset listing, SPDX commercial checks, live URL checks & sample pulls
├── tests/
│   ├── test_autolab_wire.py      # autolab-wire/v1 L1 packet & clearance-gate tests
│   ├── test_community_extensions.py
│   ├── test_open_datasets_2026.py# Open dataset registry, non-commercial flags & SE(3)/slosh audit tests
│   ├── test_phase2_audit_fixes.py
│   └── test_plan_to_well.py      # Pytest suite covering kinematics, collision, planning & tactile
├── autolab_wire.py               # 15-repo Auto-Lab wire node (REPO_ID="lab-hands", LAYER="L1")
├── LICENSE                       # MIT License
├── pyproject.toml                # Pytest configuration
├── README.md                     # Overview, demos, and curated manipulation/tactile resources
└── requirements.txt              # numpy>=1.26, pytest>=8
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
