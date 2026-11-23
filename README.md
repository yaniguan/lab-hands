# lab-hands

Give a lab agent *hands and touch*: move pipettes, tubes and well plates safely, and use touch for the last centimeter, where cameras fail on clear glass and plastic.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the hands are planned: a robot arm, pipettes and well plates, and liquid-handler scripts. Touch lives in this repo too. Cameras struggle with transparent vessels, so the last centimeter needs touch, and it uses the same arms, tasks and data. This repo collects open hardware, policies, simulators and tactile datasets, and starts with a toy planner that reaches named wells.

## First attempt

A top-down 2D toy: a planar 2-link arm holds a pipette tip, starts at a tube and must reach a named well of an SBS 96-well plate (9 mm pitch) without hitting a reagent bottle or a rack wall. It maps well index to name (row-major and column-major), solves inverse kinematics, plans in joint space with RRT and then shortcuts the path. 10 target wells x 10 seeds. Pure numpy on CPU.

```bash
python demo/plan_to_well.py
```

Output (Linux, CPU only, 2026-11-18):

```
well 65 -> F5 (row-major), A9 (column-major)
straight joint-space move blocked by obstacles for 10 of 10 wells
trials: 100, success: 100 (100.0%)
plan time: mean 70.6 ms, max 278.4 ms
joint path length: raw 2.44 rad -> shortcut 2.08 rad
```

The straight joint move is blocked for every target, and RRT finds a path in all 100 trials. This is geometry only: no dynamics, no liquid, no camera. Next: the same task in a 3D simulator with a real arm model, then a learned policy (ACT or Diffusion Policy) from demonstrations.

## Datasets

| Dataset | What | License |
|---|---|---|
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | Large multi-robot trajectory collection in one format | Mixed per sub-dataset - check each |
| [LeRobot datasets](https://huggingface.co/lerobot) | Many robot datasets in one format (e.g. ALOHA sim) | Per dataset (e.g. aloha_sim_insertion_human: MIT) |
| [AutoBio pipette task](https://huggingface.co/datasets/autobio-bench/pipette-mujoco) | Simulated lab pipetting demos | MIT |
| [TransCG](https://graspnet.net/transcg) | Real depth data for transparent objects | CC BY-NC-SA 4.0 (**non-commercial**) |
| [ClearGrasp](https://sites.google.com/view/cleargrasp) | Synthetic + real transparent object depth | See project site |
| [Vector-LabPics](https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net) | Lab images of vessels and materials (liquid, foam, solid) | Code MIT; data see source |
| [Touch-Slide (DIGIT)](https://huggingface.co/datasets/facebook/touch-slide) | Tactile images for pretraining | CC BY-NC 4.0 (**non-commercial**) |
| [Touch and Go](https://touch-and-go.github.io/) | Human-collected vision + touch | See project site |
| [ObjectFolder](https://objectfolder.stanford.edu/) | Multisensory object data incl. tactile | See project site |

Licenses marked **non-commercial** are fine for research but not for products.

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

- **Object**: Physical manipulation and contact-sensing layer moving pipettes, tubes, and microplates safely around benchtop obstacles, while framing visuo-tactile control for transparent glass and plastic vessels.
- **Data**: Synthetic ANSI/SLAS (SBS) 96-well plate geometry (`9 mm` pitch, row-major and column-major indexing) with tube-rack and reagent-bottle AABB obstacles (`10` wells x `10` seeds), plus 9 curated robot, transparent-object, and tactile datasets.
- **Model**: Analytical 2-link planar kinematics (`fk`, `ik` with elbow-branch fallback), discrete 30-point link-AABB collision checking, joint-space RRT (`iters=3000`, `step=0.15 rad`, `goal_bias=0.1`), and 100-iteration randomized shortcut smoothing.
- **Training**: Classical sampling-based planner (zero learned weights yet): resolves `10/10` obstacle-blocked straight moves with `100/100` (`100.0%`) RRT trial success in `70.6 ms` mean CPU time, shortening paths from `2.44 rad` to `2.08 rad` (`-14.8%`).
- **History**: `eb4d967` (2026-11-18, `main` 2D SBS 96-well RRT + shortcut planner, reference benchmark, CI workflow, and visuo-tactile literature map).
- **Unique Properties**: Pure-`numpy` single-file reproducibility with zero simulator/GPU dependencies, explicit dual-order SBS 96-well addressing (`65 -> F5` row-major vs `A9` column-major), and direct `L1` pairing with [`stepbot`](https://github.com/yaniguan/stepbot).

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## Repository Layout

```text
lab-hands/
├── .github/workflows/
│   └── demo.yml                  # CI workflow running the 100-trial RRT well-planning benchmark
├── demo/
│   ├── plan_to_well.py           # 2-link IK, AABB collision checker, RRT, and shortcut smoother
│   └── output.txt                # Reference 100-trial CPU metrics
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── LICENSE                       # MIT License
├── README.md                     # Overview, planner demo, and curated manipulation/tactile resources
└── requirements.txt              # numpy only
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
