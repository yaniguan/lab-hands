"""Open-source wet-lab manipulation, transparent-vessel depth, and visuo-tactile dataset registry.

Provides:
1. Typed `OpenManipulationDataset` registry (`OPEN_MANIPULATION_DATASETS`) with explicit SPDX
   licensing and fail-closed `commercial_use_ok` enforcement across four modalities:
   `kinematics`, `transparent_depth`, `visuo_tactile`, and `wetlab_video`.
2. Episode and action loaders for HuggingFace LeRobot (`.parquet`, `.jsonl`, `.json`, and
   `datasets-server` `/rows` payloads) and FineBio hierarchical wet-lab action annotations
   (ActionFormer JSON database and tabular take transcripts).
3. Physical trajectory safety auditing (`audit_se3_trajectory`, `audit_lerobot_episode`,
   `audit_finebio_annotations`) converting demonstrations and benchtop annotations into
   6-DOF `ArmSE3` poses, continuous `>= 5.0 mm` capsule-AABB obstacle clearance checks,
   96-well rim & plunge verticality checks, and cylindrical corner-well meniscus slosh bounds.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from .collision import (
    DEFAULT_SCENE,
    BenchScene,
    in_collision_3d,
    min_clearance,
    path_min_clearance,
)
from .kinematics import DEFAULT_ARM, DEFAULT_ARM_SE3, Arm2D, ArmSE3, PoseSE3, wrap_angle
from .labware import DeckLayout, PlateGeometry
from .planner import select_ik_branch
from .trajectory import PickPlacePlan3D, parameterize_path, plan_3d_pick_place

DatasetModality = Literal["kinematics", "transparent_depth", "visuo_tactile", "wetlab_video"]
VALID_MODALITIES: Tuple[DatasetModality, ...] = (
    "kinematics",
    "transparent_depth",
    "visuo_tactile",
    "wetlab_video",
)

_GRAVITY_M_S2: float = 9.80665
_BESSEL_XI_11: float = 1.841
_MIN_OBSTACLE_CLEARANCE_MM: float = 5.0
_DEFAULT_96_CORNER_OFFSETS_XY: Tuple[Tuple[float, float], ...] = (
    (0.0495, 0.0315),
    (0.0495, -0.0315),
    (-0.0495, 0.0315),
    (-0.0495, -0.0315),
)


def _is_noncommercial_spdx(spdx_license: str) -> bool:
    token = spdx_license.strip().upper()
    parts = token.replace("_", "-").split("-")
    return (
        "NC" in parts
        or "NONCOMMERCIAL" in token
        or token.startswith("LICENSEREF-")
        or token in {"NOASSERTION", "UNLICENSED", "CUSTOM"}
    )


@dataclass(frozen=True)
class OpenManipulationDataset:
    """Metadata and license guardrails for an external robotics, depth, or tactile dataset."""

    dataset_id: str
    title: str
    modality: DatasetModality
    url: str
    spdx_license: str
    commercial_use_ok: bool
    year: int
    venue: str
    summary: str
    hf_repo_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError(f"dataset_id must be a non-empty string, got {self.dataset_id!r}")
        if self.dataset_id != self.dataset_id.strip().lower():
            raise ValueError(f"dataset_id must be lowercase canonical slug, got {self.dataset_id!r}")
        if self.modality not in VALID_MODALITIES:
            raise ValueError(
                f"Invalid modality {self.modality!r}; expected one of {VALID_MODALITIES}"
            )
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ValueError(f"Dataset url must be a valid https:// URL, got {self.url!r}")
        if not isinstance(self.spdx_license, str) or not self.spdx_license.strip():
            raise ValueError("spdx_license must be a non-empty SPDX identifier")
        if not isinstance(self.commercial_use_ok, bool):
            raise ValueError("commercial_use_ok must be a boolean")
        if _is_noncommercial_spdx(self.spdx_license) and self.commercial_use_ok:
            raise ValueError(
                f"Dataset {self.dataset_id!r} carries non-commercial/restricted license "
                f"{self.spdx_license!r} and cannot set commercial_use_ok=True"
            )
        if self.year < 2018 or self.year > 2030:
            raise ValueError(f"Unexpected dataset publication year {self.year}")


OPEN_MANIPULATION_DATASETS: Dict[str, OpenManipulationDataset] = {
    spec.dataset_id: spec
    for spec in (
        OpenManipulationDataset(
            dataset_id="finebio",
            title="FineBio: Fine-Grained Video Dataset of Biological Experiments",
            modality="wetlab_video",
            url="https://github.com/aistairc/FineBio",
            spdx_license="LicenseRef-FineBio-NC",
            commercial_use_ok=False,
            year=2024,
            venue="CVPR 2024 / IJCV 2025",
            summary=(
                "Multi-view wet-lab biology benchmark (egocentric + 5 benchtop views, 32 participants, "
                "7 PCR and micropipetting protocols) with hierarchical step, atomic-operation, and "
                "manipulated/affected labware annotations."
            ),
        ),
        OpenManipulationDataset(
            dataset_id="droid",
            title="DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset",
            modality="kinematics",
            url="https://droid-dataset.github.io/",
            spdx_license="CC-BY-4.0",
            commercial_use_ok=True,
            year=2024,
            venue="RSS 2024",
            summary=(
                "Distributed 76k+ teleoperated 7-DoF Franka Panda manipulation trajectories across "
                "564 scenes with calibrated stereo wrist/exterior RGB-D and SE(3)/joint states."
            ),
            hf_repo_id="lerobot/droid_1.0.1",
        ),
        OpenManipulationDataset(
            dataset_id="lerobot",
            title="HuggingFace LeRobot Standardized Manipulation Datasets (v3.0)",
            modality="kinematics",
            url="https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet",
            spdx_license="Apache-2.0",
            commercial_use_ok=True,
            year=2025,
            venue="HuggingFace Robotics",
            summary=(
                "Standardized 50 Hz Apache Parquet + AV1 robot demonstration episodes "
                "(observation.state, action, timestamp, episode_index) for ALOHA and SO-ARM100."
            ),
            hf_repo_id="lerobot/aloha_mobile_cabinet",
        ),
        OpenManipulationDataset(
            dataset_id="autobio_pipette_mujoco",
            title="AutoBio Wet-Lab Pipette MuJoCo Trajectories (LeRobot v2.1)",
            modality="kinematics",
            url="https://huggingface.co/datasets/autobio-bench/pipette-mujoco",
            spdx_license="MIT",
            commercial_use_ok=True,
            year=2025,
            venue="AutoBio-Bench",
            summary=(
                "100 simulated 50 Hz UR5e wet-lab micropipette episodes (71,014 frames in LeRobot "
                "Parquet format) transferring reagents across benchtop tubes and microplates."
            ),
            hf_repo_id="autobio-bench/pipette-mujoco",
        ),
        OpenManipulationDataset(
            dataset_id="transcg",
            title="TransCG: Large-Scale Real-World Transparent Object Depth Completion",
            modality="transparent_depth",
            url="https://github.com/Galaxies99/TransCG",
            spdx_license="CC-BY-NC-SA-4.0",
            commercial_use_ok=False,
            year=2023,
            venue="IEEE RA-L / ICRA 2023",
            summary=(
                "57k real RGB-D frames of 51 transparent containers and glassware items in cluttered "
                "scenes with ground-truth depth completion maps and robotic grasp poses."
            ),
        ),
        OpenManipulationDataset(
            dataset_id="transproteus",
            title="TransProteus: 3D Shape & Depth of Liquids and Solids in Transparent Vessels",
            modality="transparent_depth",
            url="https://github.com/sagieppel/TransProteus_Paper_Code",
            spdx_license="CC-BY-4.0",
            commercial_use_ok=True,
            year=2024,
            venue="RSC Digital Discovery 2024",
            summary=(
                "50k+ scenes of liquids, phase boundaries, and solids inside transparent chemical "
                "vessels with 3D XYZ surface maps, depth meshes, and optical material properties."
            ),
        ),
        OpenManipulationDataset(
            dataset_id="clearpose",
            title="ClearPose: Large-Scale Transparent Object Segmentation, Depth & 6D Pose",
            modality="transparent_depth",
            url="https://github.com/opipari/ClearPose",
            spdx_license="CC-BY-4.0",
            commercial_use_ok=True,
            year=2024,
            venue="ECCV / Progress Lab",
            summary=(
                "350k+ labeled real RGB-D frames covering 63 transparent household and reagent "
                "containers under heavy clutter, liquid fills, and refractive backdrop shifts."
            ),
        ),
        OpenManipulationDataset(
            dataset_id="layered_depth",
            title="LayeredDepth: Seeing and Seeing Through the Glass Multi-Layer Depth",
            modality="transparent_depth",
            url="https://huggingface.co/datasets/princeton-vl/LayeredDepth",
            spdx_license="CC0-1.0",
            commercial_use_ok=True,
            year=2025,
            venue="ICCV 2025",
            summary=(
                "Multi-layer depth benchmark resolving front-surface glass geometry simultaneously "
                "with interior/transmitted surfaces behind transparent vessels."
            ),
            hf_repo_id="princeton-vl/LayeredDepth",
        ),
        OpenManipulationDataset(
            dataset_id="tvl",
            title="TVL: A Touch, Vision, and Language Dataset for Multimodal Alignment",
            modality="visuo_tactile",
            url="https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset",
            spdx_license="Apache-2.0",
            commercial_use_ok=True,
            year=2024,
            venue="ICML 2024",
            summary=(
                "44k paired in-the-wild DIGIT optical tactile and RGB observations aligned with "
                "contact-surface language descriptions for visuo-tactile encoder pretraining."
            ),
            hf_repo_id="mlfu7/Touch-Vision-Language-Dataset",
        ),
        OpenManipulationDataset(
            dataset_id="tactx",
            title="TactX: Cross-Sensor Visuo-Tactile Manipulation Dataset",
            modality="visuo_tactile",
            url="https://huggingface.co/datasets/TactX/TactX",
            spdx_license="CC-BY-4.0",
            commercial_use_ok=True,
            year=2025,
            venue="TactX Project 2025",
            summary=(
                "544 teleoperated Franka episodes across 3 tactile sensors (Daimon, eFlesh, FlexiTac) "
                "and 4 contact-rich precision insertion, pick-and-place, and surface-wiping tasks."
            ),
            hf_repo_id="TactX/TactX",
        ),
        OpenManipulationDataset(
            dataset_id="feelsight",
            title="FeelSight / NeuralFeels: Visuo-Tactile In-Hand Manipulation Dataset",
            modality="visuo_tactile",
            url="https://huggingface.co/datasets/suddhu/Feelsight",
            spdx_license="MIT",
            commercial_use_ok=True,
            year=2024,
            venue="Science Robotics / Meta AI 2024",
            summary=(
                "70 real and simulated DIGIT visuo-tactile + proprioception contact sequences "
                "tracking sub-millimeter 3D object pose under severe visual occlusion."
            ),
            hf_repo_id="suddhu/Feelsight",
        ),
        OpenManipulationDataset(
            dataset_id="octopi_physiclear",
            title="Octopi & PhysiCLeAR: GelSight Tactile Property Reasoning Suite",
            modality="visuo_tactile",
            url="https://github.com/clear-nus/octopi-1.5",
            spdx_license="LicenseRef-Octopi-Research",
            commercial_use_ok=False,
            year=2025,
            venue="RSS 2024 / RSS 2025",
            summary=(
                "GelSight optical tactile video suite with physical compliance, roughness, and "
                "hardness annotations paired with multimodal tactile-language reasoning."
            ),
        ),
    )
}

_DATASET_ALIASES: Dict[str, str] = {
    "aloha": "lerobot",
    "lerobot_aloha_mobile_cabinet": "lerobot",
    "autobio": "autobio_pipette_mujoco",
    "pipette_mujoco": "autobio_pipette_mujoco",
    "octopi": "octopi_physiclear",
    "physiclear": "octopi_physiclear",
    "touch_vision_language": "tvl",
    "neuralfeels": "feelsight",
    "layereddepth": "layered_depth",
}


def get_dataset(dataset_id: str) -> OpenManipulationDataset:
    """Resolve canonical or alias `dataset_id` (case-insensitive) from `OPEN_MANIPULATION_DATASETS`."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise KeyError(f"Invalid dataset_id {dataset_id!r}")
    norm = dataset_id.strip().lower().replace("-", "_")
    norm = _DATASET_ALIASES.get(norm, norm)
    if norm not in OPEN_MANIPULATION_DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset_id!r}; available: {sorted(OPEN_MANIPULATION_DATASETS)}"
        )
    return OPEN_MANIPULATION_DATASETS[norm]


def list_datasets(
    *,
    modality: Optional[DatasetModality] = None,
    commercial_only: bool = False,
    min_year: Optional[int] = None,
) -> List[OpenManipulationDataset]:
    """Filter registered datasets by modality, commercial-use eligibility, and publication year."""
    if modality is not None and modality not in VALID_MODALITIES:
        raise ValueError(f"Invalid modality filter {modality!r}; expected one of {VALID_MODALITIES}")
    items = list(OPEN_MANIPULATION_DATASETS.values())
    if modality is not None:
        items = [d for d in items if d.modality == modality]
    if commercial_only:
        items = [d for d in items if d.commercial_use_ok]
    if min_year is not None:
        items = [d for d in items if d.year >= int(min_year)]
    return items


def check_commercial_compliance(dataset_ids: Iterable[str]) -> Tuple[bool, List[str]]:
    """Return `(all_commercial_ok, blocked_dataset_ids)` for production training/deployment gates."""
    blocked: List[str] = []
    for raw_id in dataset_ids:
        spec = get_dataset(raw_id)
        if not spec.commercial_use_ok and spec.dataset_id not in blocked:
            blocked.append(spec.dataset_id)
    return (len(blocked) == 0), blocked


@dataclass(frozen=True)
class LeRobotEpisode:
    """Normalized LeRobot trajectory episode mapped onto a 6-DOF `ArmSE3` joint configuration array."""

    dataset_id: str
    episode_index: int
    fps: float
    timestamps_s: np.ndarray
    joint_positions_rad: np.ndarray
    actions: Optional[np.ndarray]
    q6_se3: np.ndarray
    gripper: Optional[np.ndarray] = None
    task_index: int = 0

    @property
    def n_frames(self) -> int:
        return int(self.q6_se3.shape[0])

    @property
    def duration_s(self) -> float:
        if self.timestamps_s.size < 2:
            return 0.0
        return float(self.timestamps_s[-1] - self.timestamps_s[0])


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    """Load rows from an Apache Parquet episode file via optional `pyarrow` or `pandas`."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        table = pq.read_table(path)
        return list(table.to_pylist())
    except ImportError:
        pass
    try:
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_parquet(path)
        return list(df.to_dict(orient="records"))
    except ImportError as exc:
        raise ImportError(
            f"Reading binary Parquet file {path} requires optional 'pyarrow' or 'pandas'; "
            "use JSONL / HuggingFace datasets-server JSON for pure-numpy environments."
        ) from exc


def _normalize_lerobot_rows(
    source: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return _read_parquet_rows(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty LeRobot episode file: {path}")
        if suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        payload = json.loads(text)
        return _normalize_lerobot_rows(payload)

    if isinstance(source, Mapping):
        if "rows" in source and isinstance(source["rows"], list):
            # HuggingFace datasets-server /rows format: {"rows": [{"row_idx": 0, "row": {...}}, ...]}
            unwrapped: List[Dict[str, Any]] = []
            for item in source["rows"]:
                if isinstance(item, Mapping) and "row" in item and isinstance(item["row"], Mapping):
                    unwrapped.append(dict(item["row"]))
                elif isinstance(item, Mapping):
                    unwrapped.append(dict(item))
            return unwrapped
        # Column-oriented dict {"observation.state": [...], "timestamp": [...]}
        keys = list(source.keys())
        if not keys:
            return []
        n = len(source[keys[0]])
        return [{k: source[k][i] for k in keys} for i in range(n)]

    return [dict(r) for r in source]


def load_lerobot_episode(
    source: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    episode_index: Optional[int] = None,
    dataset_id: str = "lerobot",
    arm_joint_slice: Tuple[int, int] = (0, 6),
    gripper_index: Optional[int] = 6,
    default_fps: float = 50.0,
    joint_scale: float = 1.0,
    joint_offsets_rad: Optional[Sequence[float]] = None,
) -> LeRobotEpisode:
    """Load a LeRobot v2/v3 `.parquet`, `.jsonl`, `.json`, or HF `/rows` trajectory into `LeRobotEpisode`."""
    if default_fps <= 0.0:
        raise ValueError(f"default_fps must be > 0, got {default_fps}")
    s_lo, s_hi = int(arm_joint_slice[0]), int(arm_joint_slice[1])
    if s_lo < 0 or s_hi <= s_lo or (s_hi - s_lo) > 6:
        raise ValueError(f"arm_joint_slice must select 1..6 joints, got {arm_joint_slice!r}")

    rows = _normalize_lerobot_rows(source)
    if not rows:
        raise ValueError("LeRobot source contains zero rows")

    if episode_index is None:
        first_ep = rows[0].get("episode_index", 0)
        target_ep = int(first_ep) if first_ep is not None else 0
    else:
        target_ep = int(episode_index)

    if any("episode_index" in r for r in rows):
        ep_rows = [r for r in rows if int(r.get("episode_index", target_ep)) == target_ep]
    else:
        ep_rows = rows
    if not ep_rows:
        raise ValueError(f"Episode index {target_ep} not found in LeRobot source")

    if all("frame_index" in r for r in ep_rows):
        ep_rows = sorted(ep_rows, key=lambda r: int(r["frame_index"]))

    state_key: Optional[str] = None
    for cand in ("observation.state", "state", "qpos", "joint_positions"):
        if cand in ep_rows[0]:
            state_key = cand
            break
    if state_key is None:
        raise ValueError(
            f"LeRobot row missing state vector ('observation.state' or 'state'); keys={sorted(ep_rows[0])}"
        )

    act_key: Optional[str] = None
    for cand in ("action", "actions"):
        if cand in ep_rows[0] and ep_rows[0][cand] is not None:
            act_key = cand
            break

    states = np.asarray([r[state_key] for r in ep_rows], dtype=np.float64) * float(joint_scale)
    if states.ndim == 1:
        states = states[:, None]
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError(f"Invalid LeRobot state tensor shape={states.shape} or non-finite values")

    actions: Optional[np.ndarray] = None
    if act_key is not None:
        actions = np.asarray([r[act_key] for r in ep_rows], dtype=np.float64) * float(joint_scale)
        if actions.ndim == 1:
            actions = actions[:, None]

    n_frames, state_dim = states.shape
    if "timestamp" in ep_rows[0] or "time_s" in ep_rows[0]:
        t_key = "timestamp" if "timestamp" in ep_rows[0] else "time_s"
        timestamps = np.asarray([float(r[t_key]) for r in ep_rows], dtype=np.float64)
        timestamps = timestamps - float(timestamps[0])
    else:
        frame_indices = np.asarray(
            [float(r.get("frame_index", idx)) for idx, r in enumerate(ep_rows)],
            dtype=np.float64,
        )
        frame_indices = frame_indices - float(frame_indices[0])
        timestamps = frame_indices / float(default_fps)

    if n_frames >= 2:
        diffs = np.diff(timestamps)
        pos_diffs = diffs[diffs > 1e-9]
        fps = float(1.0 / np.median(pos_diffs)) if pos_diffs.size > 0 else float(default_fps)
    else:
        fps = float(default_fps)

    eff_hi = min(state_dim, s_hi)
    if s_lo >= state_dim:
        raise ValueError(f"arm_joint_slice {arm_joint_slice} out of bounds for state_dim={state_dim}")
    sliced = states[:, s_lo:eff_hi]
    q6 = np.zeros((n_frames, 6), dtype=np.float64)
    q6[:, : sliced.shape[1]] = sliced

    if joint_offsets_rad is not None:
        offsets = np.asarray(joint_offsets_rad, dtype=np.float64)
        if offsets.shape != (6,):
            raise ValueError(f"joint_offsets_rad must have shape (6,), got {offsets.shape}")
        q6 = q6 + offsets[None, :]

    gripper: Optional[np.ndarray] = None
    if gripper_index is not None and 0 <= int(gripper_index) < state_dim:
        gripper = states[:, int(gripper_index)].copy()

    task_idx = int(ep_rows[0].get("task_index", 0) or 0)
    return LeRobotEpisode(
        dataset_id=dataset_id,
        episode_index=target_ep,
        fps=round(fps, 4),
        timestamps_s=timestamps,
        joint_positions_rad=states,
        actions=actions,
        q6_se3=q6,
        gripper=gripper,
        task_index=task_idx,
    )


@dataclass(frozen=True)
class FineBioActionAnnotation:
    """Single wet-lab atomic manipulation annotation mapped onto `lab_hands` benchtop references."""

    take_id: str
    start_s: float
    end_s: float
    hand: str
    verb: str
    manipulated_object: str
    affected_object: Optional[str]
    source_ref: str
    target_ref: str
    source_xyz_m: Optional[Tuple[float, float, float]] = None
    target_xyz_m: Optional[Tuple[float, float, float]] = None
    attached_tip_length_m: float = 0.0
    carrying_open_plate: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s - self.start_s))


def build_finebio_benchtop_deck() -> DeckLayout:
    """Standard multi-labware wet-lab deck layout matching FineBio PCR & pipetting protocols."""
    deck = DeckLayout()
    deck.add_labware("falcon15", "opentrons_15_tuberack_falcon_15ml_conical", slot=1)
    deck.add_labware("eppendorf", "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap", slot=2)
    deck.add_labware("tips", "opentrons_96_filtertiprack_200ul", slot=4)
    deck.add_labware("plate96", "corning_96_wellplate_360ul_flat", slot=5)
    deck.add_labware("pcr96", "nest_96_wellplate_100ul_pcr_full_skirt", slot=8)
    return deck


def _map_finebio_object_to_ref(obj_name: Optional[str], well_cursor: int = 1) -> Optional[str]:
    if obj_name is None:
        return None
    tok = obj_name.strip().lower().replace("-", "_").replace(" ", "_")
    if not tok or tok in {"none", "null", "na"}:
        return None
    if ":" in obj_name or tok == "tube":
        return obj_name.strip() if ":" in obj_name else "TUBE"
    col = ((well_cursor - 1) % 12) + 1
    row_ch = chr(ord("A") + (((well_cursor - 1) // 12) % 8))
    well_96 = f"{row_ch}{col}"
    if "tip" in tok:
        return f"tips:{well_96}"
    if "pcr" in tok:
        return f"pcr96:{well_96}"
    if "plate" in tok or "well" in tok:
        return f"plate96:{well_96}"
    if "50ml" in tok or "15ml" in tok or "falcon" in tok or "conical" in tok:
        return "falcon15:A1"
    if "microtube" in tok or "eppendorf" in tok or "sample_tube" in tok or "reagent" in tok:
        return "eppendorf:A1"
    return "TUBE"


def _parse_xyz_tuple(raw: Any) -> Optional[Tuple[float, float, float]]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite 3D coordinate [x, y, z] in meters, got {raw!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _snap_xyz_to_plate_ref(
    xyz_m: Tuple[float, float, float],
    deck: DeckLayout,
) -> Optional[str]:
    xy = np.array([xyz_m[0], xyz_m[1]], dtype=np.float64)
    best_ref: Optional[str] = None
    best_dist = 0.015  # 15 mm capture tolerance around labware wells
    for alias, geom in deck.labware_by_alias.items():
        px0, px1, py0, py1 = geom.footprint_bounds(pad_m=0.015)
        if not (px0 <= xy[0] <= px1 and py0 <= xy[1] <= py1):
            continue
        for well in geom.ordered_wells("row"):
            d = float(np.linalg.norm(xy - geom.well_xy(well)))
            if d < best_dist:
                best_dist = d
                best_ref = f"{alias}:{well}"
    return best_ref


def load_finebio_annotations(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    take_id: Optional[str] = None,
    deck: Optional[DeckLayout] = None,
) -> List[FineBioActionAnnotation]:
    """Load FineBio atomic wet-lab manipulation annotations from ActionFormer JSON, JSONL, or TXT/TSV."""
    eff_deck = build_finebio_benchtop_deck() if deck is None else deck
    raw_items: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty FineBio annotation file: {path}")
        if suffix in {".txt", ".tsv", ".csv"}:
            default_take = take_id or path.stem
            delim = "\t" if "\t" in text.splitlines()[0] else ("," if suffix == ".csv" else None)
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [p.strip() for p in (stripped.split(delim) if delim else stripped.split())]
                if parts[0].lower() in {"start", "start_s", "start_sec"}:
                    continue
                if len(parts) < 4:
                    raise ValueError(f"Expected >= 4 columns in FineBio transcript line: {line!r}")
                entry: Dict[str, Any] = {
                    "segment": [float(parts[0]), float(parts[1])],
                    "verb_label": parts[2],
                    "manipulated_label": parts[3],
                    "affected_label": parts[4] if len(parts) >= 5 else None,
                    "hand_label": parts[5] if len(parts) >= 6 else "right",
                }
                if len(parts) >= 7 and parts[6]:
                    entry["source_ref"] = parts[6]
                if len(parts) >= 8 and parts[7]:
                    entry["target_ref"] = parts[7]
                raw_items.append((default_take, entry))
        elif suffix == ".jsonl":
            default_take = take_id or path.stem
            for line in text.splitlines():
                if line.strip():
                    obj = json.loads(line)
                    raw_items.append((str(obj.get("take_id", default_take)), obj))
        else:
            return load_finebio_annotations(json.loads(text), take_id=take_id, deck=eff_deck)
    elif isinstance(source, Mapping):
        if "database" in source and isinstance(source["database"], Mapping):
            db = source["database"]
            selected_keys = [take_id] if take_id is not None else sorted(db.keys())
            for k in selected_keys:
                if k not in db:
                    raise KeyError(f"Take {k!r} not found in FineBio database")
                take_entry = db[k]
                for ann in take_entry.get("annotations", []):
                    raw_items.append((str(k), dict(ann)))
        elif "annotations" in source and isinstance(source["annotations"], list):
            t_id = str(take_id or source.get("take_id", "P01_01_01"))
            for ann in source["annotations"]:
                raw_items.append((t_id, dict(ann)))
        else:
            raise ValueError("Unsupported FineBio mapping format; expected 'database' or 'annotations'")
    else:
        default_take = take_id or "P01_01_01"
        for ann in source:
            d = dict(ann)
            raw_items.append((str(d.get("take_id", default_take)), d))

    out: List[FineBioActionAnnotation] = []
    tip_len_by_take: Dict[str, float] = {}
    last_ref_by_take: Dict[str, str] = {}
    well_idx_by_take: Dict[str, int] = {}

    for cur_take, rec in raw_items:
        seg = rec.get("segment")
        if seg is not None and len(seg) == 2:
            t0, t1 = float(seg[0]), float(seg[1])
        else:
            t0 = float(rec.get("start_s", rec.get("start_sec", 0.0)))
            t1 = float(rec.get("end_s", rec.get("end_sec", t0)))
        if t1 < t0 or not (math.isfinite(t0) and math.isfinite(t1)):
            raise ValueError(f"Invalid FineBio segment [{t0}, {t1}] in {cur_take}")

        verb = str(rec.get("verb_label", rec.get("verb", "transfer"))).strip().lower()
        manip = str(rec.get("manipulated_label", rec.get("manipulated", "micropipette"))).strip().lower()
        aff_raw = rec.get("affected_label", rec.get("affected"))
        affected = None if aff_raw is None else str(aff_raw).strip().lower()
        hand = str(rec.get("hand_label", rec.get("hand", "right"))).strip().lower()

        src_xyz = _parse_xyz_tuple(rec.get("source_xyz_m"))
        dst_xyz = _parse_xyz_tuple(rec.get("target_xyz_m", rec.get("xyz_m")))

        well_cursor = well_idx_by_take.get(cur_take, 1)
        prev_ref = last_ref_by_take.get(cur_take, "TUBE")

        explicit_src = rec.get("source_ref")
        explicit_dst = rec.get("target_ref", rec.get("well_ref"))

        src_ref = (
            str(explicit_src)
            if explicit_src
            else (_snap_xyz_to_plate_ref(src_xyz, eff_deck) if src_xyz is not None else None)
            or prev_ref
        )
        dst_ref = (
            str(explicit_dst)
            if explicit_dst
            else (_snap_xyz_to_plate_ref(dst_xyz, eff_deck) if dst_xyz is not None else None)
            or _map_finebio_object_to_ref(affected or manip, well_cursor=well_cursor)
            or "plate96:A1"
        )

        if dst_ref == src_ref and explicit_src is None:
            src_ref = "TUBE" if dst_ref != "TUBE" else "plate96:A1"

        cur_tip = tip_len_by_take.get(cur_take, 0.0)
        if "attached_tip_length_m" in rec and rec["attached_tip_length_m"] is not None:
            eff_tip = float(rec["attached_tip_length_m"])
            tip_len_by_take[cur_take] = eff_tip
        else:
            if verb in {"insert", "attach", "mount", "take_tip"} and (
                (affected and "tip" in affected) or dst_ref.startswith("tips:")
            ):
                cur_tip = 0.058
                tip_len_by_take[cur_take] = cur_tip
            eff_tip = cur_tip
            if verb in {"eject", "detach", "discard_tip"}:
                tip_len_by_take[cur_take] = 0.0

        carrying_plate = bool(
            rec.get(
                "carrying_open_plate",
                ("plate" in manip or "pcr" in manip) and verb in {"move", "place", "put", "transfer", "carry"},
            )
        )

        if dst_ref.startswith(("plate96:", "pcr96:", "tips:")):
            well_idx_by_take[cur_take] = well_cursor + 1
        last_ref_by_take[cur_take] = dst_ref

        out.append(
            FineBioActionAnnotation(
                take_id=cur_take,
                start_s=t0,
                end_s=t1,
                hand=hand,
                verb=verb,
                manipulated_object=manip,
                affected_object=affected,
                source_ref=src_ref,
                target_ref=dst_ref,
                source_xyz_m=src_xyz,
                target_xyz_m=dst_xyz,
                attached_tip_length_m=eff_tip,
                carrying_open_plate=carrying_plate,
            )
        )

    return out


@dataclass(frozen=True)
class TrajectorySafetyAudit:
    """Unified physical safety audit report across `ArmSE3`, 96-well clearance, and meniscus slosh."""

    source_id: str
    safe: bool
    n_frames: int
    duration_s: float
    se3_joint_limits_ok: bool
    obstacle_clearance_ok: bool
    min_obstacle_clearance_mm: float
    rim_clearance_ok: bool
    plunge_verticality_ok: bool
    max_plunge_tilt_deg: float
    slosh_safe: bool
    spilled: bool
    peak_lateral_accel_m_s2: float
    peak_meniscus_rise_mm: float
    free_headspace_mm: float
    meniscus_to_headspace_ratio: float
    recommended_min_duration_s: float
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _integrate_cylindrical_meniscus_slosh_m(
    accel_xy_m_s2: np.ndarray,
    dt_s: float,
    *,
    well_radius_m: float = 0.00345,
    fill_height_m: float = 0.00920,
    damping_ratio: float = 0.028,
    yaw_vel_rad_s: Optional[np.ndarray] = None,
    yaw_acc_rad_s2: Optional[np.ndarray] = None,
    yaw_rad: Optional[np.ndarray] = None,
    corner_offsets_xy: Sequence[Tuple[float, float]] = _DEFAULT_96_CORNER_OFFSETS_XY,
    include_corner_rotation: bool = False,
) -> Tuple[float, float]:
    """Integrate fundamental cylindrical slosh mode (`xi_11 = 1.841`) + 120 ms ringdown via RK4."""
    acc = np.asarray(accel_xy_m_s2, dtype=np.float64)
    if acc.shape[0] == 0 or dt_s <= 1e-9:
        return 0.0, 0.0

    omega_n = math.sqrt(_BESSEL_XI_11 * _GRAVITY_M_S2 / well_radius_m) * math.tanh(
        _BESSEL_XI_11 * fill_height_m / well_radius_m
    )

    if (
        include_corner_rotation
        and corner_offsets_xy
        and yaw_rad is not None
        and yaw_vel_rad_s is not None
        and yaw_acc_rad_s2 is not None
    ):
        pts = [(0.0, 0.0), *corner_offsets_xy]
        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)
        well_acc = np.zeros((acc.shape[0], len(pts), 2), dtype=np.float64)
        well_acc[:, 0, :] = acc
        for idx_w, (rxb, ryb) in enumerate(pts[1:], start=1):
            rx = c * rxb - s * ryb
            ry = s * rxb + c * ryb
            well_acc[:, idx_w, 0] = acc[:, 0] - yaw_acc_rad_s2 * ry - (yaw_vel_rad_s**2) * rx
            well_acc[:, idx_w, 1] = acc[:, 1] + yaw_acc_rad_s2 * rx - (yaw_vel_rad_s**2) * ry
    else:
        well_acc = acc[:, None, :]

    peak_lat_acc = float(np.max(np.linalg.norm(well_acc, axis=-1)))
    n_ring = max(10, int(math.ceil(0.12 / max(dt_s, 1e-4))))
    k_wells = well_acc.shape[1]
    a_pad = np.concatenate([well_acc, np.zeros((n_ring, k_wells, 2), dtype=np.float64)], axis=0)

    n_sub = max(1, int(math.ceil(omega_n * dt_s / 0.15)))
    h = dt_s / n_sub
    gain = (well_radius_m / _GRAVITY_M_S2) * (omega_n * omega_n)
    w2 = omega_n * omega_n
    two_zw = 2.0 * damping_ratio * omega_n

    eta = np.zeros((k_wells, 2), dtype=np.float64)
    eta_dot = np.zeros((k_wells, 2), dtype=np.float64)
    peak_r = 0.0
    for i in range(a_pad.shape[0]):
        drive = gain * a_pad[i]
        for _ in range(n_sub):
            k1_x = eta_dot
            k1_v = drive - two_zw * eta_dot - w2 * eta
            x2 = eta + 0.5 * h * k1_x
            v2 = eta_dot + 0.5 * h * k1_v
            k2_x = v2
            k2_v = drive - two_zw * v2 - w2 * x2
            x3 = eta + 0.5 * h * k2_x
            v3 = eta_dot + 0.5 * h * k2_v
            k3_x = v3
            k3_v = drive - two_zw * v3 - w2 * x3
            x4 = eta + h * k3_x
            v4 = eta_dot + h * k3_v
            k4_x = v4
            k4_v = drive - two_zw * v4 - w2 * x4
            eta = eta + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            eta_dot = eta_dot + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        r_cur = float(np.max(np.hypot(eta[:, 0], eta[:, 1])))
        if r_cur > peak_r:
            peak_r = r_cur
    return peak_r, peak_lat_acc


def _check_se3_tip_rim_and_plunge(
    poses: Sequence[PoseSE3],
    *,
    scene: BenchScene,
    attached_tip_length_m: float,
    allowed_wells_xy: Optional[Sequence[np.ndarray]],
    max_plunge_tilt_rad: float,
    z_margin_m: float = 0.003,
) -> Tuple[bool, bool, float, List[np.ndarray]]:
    """Check 3D vessel-rim clearance and pipette verticality across `PoseSE3` frames."""
    rim_ok = True
    verticality_ok = True
    max_plunge_tilt = 0.0

    labware_items: List[Tuple[PlateGeometry, float]] = [
        (scene.plate, max(float(scene.z_plate_rim), float(scene.plate.z_rim_m)))
    ]
    if scene.deck is not None:
        for geom in scene.deck.labware_by_alias.values():
            labware_items.append((geom, float(geom.z_rim_m)))

    tube_xy = np.asarray(scene.tube_xy, dtype=np.float64)
    allowed_pts = [np.asarray(w, dtype=np.float64) for w in (allowed_wells_xy or ())]
    effective_allowed: List[np.ndarray] = list(allowed_pts)

    for pose in poses:
        flange_z = float(pose.xyz[2])
        tip_xyz = pose.tip_position(attached_tip_length_m=attached_tip_length_m)
        tip_xy = tip_xyz[:2]
        z_tip = float(tip_xyz[2])
        tilt = pose.tool_tilt_rad()

        if z_tip < -1e-9 or flange_z > float(scene.z_max) + 1e-9:
            rim_ok = False

        # Check source reagent tube
        dist_tube = float(np.linalg.norm(tip_xy - tube_xy))
        if dist_tube <= scene.tube_radius_m + 1e-9:
            if z_tip < scene.z_tube_bottom - 1e-9:
                rim_ok = False
            if z_tip <= scene.z_tube_rim + 1e-9:
                if not any(float(np.linalg.norm(tube_xy - p)) <= 1e-6 for p in effective_allowed):
                    effective_allowed.append(tube_xy.copy())
                max_plunge_tilt = max(max_plunge_tilt, tilt)
                plunge_depth = max(0.0, scene.z_tube_rim - z_tip)
                if not pose.check_plunge_verticality(
                    scene.tube_radius_m,
                    plunge_depth,
                    attached_tip_length_m=attached_tip_length_m,
                    max_tilt_rad=max_plunge_tilt_rad,
                ):
                    verticality_ok = False
            continue
        if dist_tube <= 0.030 and z_tip < scene.z_tube_rim + z_margin_m - 1e-9:
            rim_ok = False

        # Check 96-well plate and all deck labware footprints
        for geom, rim_z in labware_items:
            px0, px1, py0, py1 = geom.footprint_bounds()
            if px0 <= float(tip_xy[0]) <= px1 and py0 <= float(tip_xy[1]) <= py1:
                matched_bore: Optional[np.ndarray] = None
                if allowed_pts:
                    for w_pt in allowed_pts:
                        if (
                            px0 <= float(w_pt[0]) <= px1
                            and py0 <= float(w_pt[1]) <= py1
                            and float(np.linalg.norm(tip_xy - w_pt)) <= geom.well_radius_m + 1e-9
                        ):
                            matched_bore = w_pt
                            break
                else:
                    # Auto-detect alignment with any catalog well bore on this plate
                    for well_name in geom.ordered_wells("row"):
                        w_pt = geom.well_xy(well_name)
                        if float(np.linalg.norm(tip_xy - w_pt)) <= geom.well_radius_m + 1e-9:
                            matched_bore = w_pt
                            break

                if matched_bore is not None:
                    if not any(float(np.linalg.norm(matched_bore - p)) <= 1e-6 for p in effective_allowed):
                        effective_allowed.append(matched_bore.copy())
                    if z_tip < geom.z_bottom_m - 1e-9:
                        rim_ok = False
                    if z_tip <= rim_z + 1e-9:
                        max_plunge_tilt = max(max_plunge_tilt, tilt)
                        plunge_depth = max(0.0, rim_z - z_tip)
                        if not pose.check_plunge_verticality(
                            geom.well_radius_m,
                            plunge_depth,
                            attached_tip_length_m=attached_tip_length_m,
                            max_tilt_rad=max_plunge_tilt_rad,
                        ):
                            verticality_ok = False
                elif z_tip < rim_z + z_margin_m - 1e-9:
                    rim_ok = False

    return rim_ok, verticality_ok, max_plunge_tilt, effective_allowed


def audit_se3_trajectory(
    q6_trajectory: Sequence[Sequence[float]] | np.ndarray,
    timestamps_s: Sequence[float] | np.ndarray,
    *,
    source_id: str = "se3_trajectory",
    arm_se3: ArmSE3 = DEFAULT_ARM_SE3,
    arm_2d: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    attached_tip_length_m: Optional[float] = None,
    allowed_wells_xy: Optional[Sequence[Sequence[float] | np.ndarray]] = None,
    well_radius_m: float = 0.00345,
    well_depth_m: float = 0.01067,
    fill_height_m: float = 0.00920,
    damping_ratio: float = 0.028,
    headspace_safety_fraction: float = 0.70,
    carrying_open_plate: bool = False,
) -> TrajectorySafetyAudit:
    """Audit a 6-DOF `ArmSE3` joint trajectory for limits, `>= 5 mm` clearance, 96-well rims, and slosh."""
    q6_arr = np.asarray(q6_trajectory, dtype=np.float64)
    t_arr = np.asarray(timestamps_s, dtype=np.float64)
    if q6_arr.ndim != 2 or q6_arr.shape[1] != 6 or q6_arr.shape[0] < 1:
        raise ValueError(f"q6_trajectory must have shape (T, 6) with T >= 1, got {q6_arr.shape}")
    if t_arr.shape != (q6_arr.shape[0],):
        raise ValueError(f"timestamps_s shape {t_arr.shape} must match T={q6_arr.shape[0]}")

    h_free_m = float(well_depth_m - fill_height_m)
    if h_free_m <= 0.0:
        raise ValueError("96-well slosh specification has non-positive free headspace")

    eff_tip_len = (
        float(scene.attached_tip_length_m) if attached_tip_length_m is None else float(attached_tip_length_m)
    )
    eff_scene = scene if abs(eff_tip_len - scene.attached_tip_length_m) <= 1e-12 else scene.with_attached_tip(eff_tip_len)

    n_frames = int(q6_arr.shape[0])
    duration_s = float(t_arr[-1] - t_arr[0]) if n_frames >= 2 else 0.0

    se3_limits_ok = all(arm_se3.within_joint_limits(q6_arr[i]) for i in range(n_frames))
    poses: List[PoseSE3] = [arm_se3.fk_se3(q6_arr[i]) for i in range(n_frames)]
    tip_xyz = np.stack([p.tip_position(attached_tip_length_m=eff_tip_len) for p in poses], axis=0)
    tip_xy = tip_xyz[:, :2]

    # Project SE(3) Cartesian tip path onto continuity-consistent planar 2-link chain for benchtop AABB check
    q2_path: List[np.ndarray] = []
    planar_reachable = True
    prev_q2: Optional[np.ndarray] = None
    for xy in tip_xy:
        q2_cand: Optional[np.ndarray] = None
        for elbow in (1, -1):
            sol = arm_2d.ik(xy, elbow=elbow)
            if sol is not None:
                if prev_q2 is None or q2_cand is None:
                    q2_cand = sol
                elif float(np.linalg.norm(sol - prev_q2)) < float(np.linalg.norm(q2_cand - prev_q2)):
                    q2_cand = sol
        if q2_cand is None:
            planar_reachable = False
            break
        q2_path.append(q2_cand)
        prev_q2 = q2_cand

    if planar_reachable and q2_path:
        if len(q2_path) == 1:
            min_clear_mm = 1000.0 * min_clearance(q2_path[0], arm=arm_2d, scene=eff_scene)
        else:
            min_clear_mm = 1000.0 * path_min_clearance(q2_path, arm=arm_2d, scene=eff_scene)
        obstacle_ok = min_clear_mm >= (_MIN_OBSTACLE_CLEARANCE_MM - 1e-6)
    else:
        min_clear_mm = 0.0
        obstacle_ok = False

    allowed_arrs = None if allowed_wells_xy is None else [np.asarray(w, dtype=np.float64) for w in allowed_wells_xy]
    rim_ok, verticality_ok, max_plunge_tilt, resolved_allowed = _check_se3_tip_rim_and_plunge(
        poses,
        scene=eff_scene,
        attached_tip_length_m=eff_tip_len,
        allowed_wells_xy=allowed_arrs,
        max_plunge_tilt_rad=arm_se3.max_plunge_tilt_rad,
    )

    # Also verify dense segments do not tunnel across rims between waypoints
    if rim_ok and planar_reachable and n_frames >= 2:
        for i in range(n_frames - 1):
            dist_xy = float(np.linalg.norm(tip_xy[i + 1] - tip_xy[i]))
            dist_z = abs(float(poses[i + 1].xyz[2] - poses[i].xyz[2]))
            n_sub = max(2, int(math.ceil(max(dist_xy / 0.004, dist_z / 0.004))) + 1)
            if n_sub > 2:
                for alpha in np.linspace(0.0, 1.0, n_sub)[1:-1]:
                    q2_interp = (1.0 - float(alpha)) * q2_path[i] + float(alpha) * q2_path[i + 1]
                    z_flange_interp = (1.0 - float(alpha)) * float(poses[i].xyz[2]) + float(alpha) * float(
                        poses[i + 1].xyz[2]
                    )
                    if in_collision_3d(
                        [float(q2_interp[0]), float(q2_interp[1]), z_flange_interp],
                        arm=arm_2d,
                        scene=eff_scene,
                        allowed_wells_xy=resolved_allowed,
                    ):
                        rim_ok = False
                        break
            if not rim_ok:
                break

    if n_frames >= 3 and duration_s > 1e-6:
        t_uniform = np.linspace(float(t_arr[0]), float(t_arr[-1]), max(n_frames, 81), dtype=np.float64)
        dt_s = float(t_uniform[1] - t_uniform[0])
        x_u = np.interp(t_uniform, t_arr, tip_xy[:, 0])
        y_u = np.interp(t_uniform, t_arr, tip_xy[:, 1])
        vx = np.gradient(x_u, dt_s, edge_order=2)
        vy = np.gradient(y_u, dt_s, edge_order=2)
        ax = np.gradient(vx, dt_s, edge_order=2)
        ay = np.gradient(vy, dt_s, edge_order=2)
        acc_xy = np.column_stack([ax, ay])

        yaws = np.unwrap([p.rpy[2] for p in poses])
        yaw_u = np.interp(t_uniform, t_arr, yaws)
        yaw_vel = np.gradient(yaw_u, dt_s, edge_order=2)
        yaw_acc = np.gradient(yaw_vel, dt_s, edge_order=2)

        peak_rise_m, peak_lat_acc = _integrate_cylindrical_meniscus_slosh_m(
            acc_xy,
            dt_s,
            well_radius_m=well_radius_m,
            fill_height_m=fill_height_m,
            damping_ratio=damping_ratio,
            yaw_rad=yaw_u,
            yaw_vel_rad_s=yaw_vel,
            yaw_acc_rad_s2=yaw_acc,
            include_corner_rotation=carrying_open_plate,
        )
    else:
        peak_rise_m = 0.0
        peak_lat_acc = 0.0

    ratio = peak_rise_m / h_free_m
    spilled = bool(peak_rise_m >= h_free_m)
    slosh_ok = bool(ratio < headspace_safety_fraction and not spilled)

    if slosh_ok:
        rec_min_dur = duration_s
    else:
        scale = math.sqrt(max(ratio / max(headspace_safety_fraction - 0.02, 0.1), 1.05))
        rec_min_dur = duration_s * scale

    violations: List[str] = []
    if not se3_limits_ok:
        violations.append("SE3_JOINT_LIMIT_EXCEEDED")
    if not obstacle_ok:
        violations.append("OBSTACLE_CLEARANCE_BELOW_5MM")
    if not rim_ok:
        violations.append("WELL_RIM_COLLISION")
    if not verticality_ok:
        violations.append("PLUNGE_TILT_EXCEEDED")
    if not slosh_ok:
        violations.append("SLOSH_HEADSPACE_EXCEEDED")

    return TrajectorySafetyAudit(
        source_id=source_id,
        safe=(len(violations) == 0),
        n_frames=n_frames,
        duration_s=round(duration_s, 4),
        se3_joint_limits_ok=bool(se3_limits_ok),
        obstacle_clearance_ok=bool(obstacle_ok),
        min_obstacle_clearance_mm=round(float(min_clear_mm), 4),
        rim_clearance_ok=bool(rim_ok),
        plunge_verticality_ok=bool(verticality_ok),
        max_plunge_tilt_deg=round(math.degrees(max_plunge_tilt), 4),
        slosh_safe=bool(slosh_ok),
        spilled=spilled,
        peak_lateral_accel_m_s2=round(float(peak_lat_acc), 4),
        peak_meniscus_rise_mm=round(1000.0 * peak_rise_m, 4),
        free_headspace_mm=round(1000.0 * h_free_m, 4),
        meniscus_to_headspace_ratio=round(float(ratio), 4),
        recommended_min_duration_s=round(float(rec_min_dur), 4),
        violations=tuple(violations),
    )


def audit_lerobot_episode(
    episode: LeRobotEpisode,
    *,
    arm_se3: ArmSE3 = DEFAULT_ARM_SE3,
    arm_2d: Arm2D = DEFAULT_ARM,
    scene: BenchScene = DEFAULT_SCENE,
    attached_tip_length_m: float = 0.0,
    allowed_wells_xy: Optional[Sequence[Sequence[float] | np.ndarray]] = None,
    carrying_open_plate: bool = False,
    headspace_safety_fraction: float = 0.70,
) -> TrajectorySafetyAudit:
    """Run `ArmSE3`, 96-well rim, `>= 5 mm` obstacle, and meniscus slosh checks on a `LeRobotEpisode`."""
    return audit_se3_trajectory(
        episode.q6_se3,
        episode.timestamps_s,
        source_id=f"{episode.dataset_id}:ep{episode.episode_index:04d}",
        arm_se3=arm_se3,
        arm_2d=arm_2d,
        scene=scene,
        attached_tip_length_m=attached_tip_length_m,
        allowed_wells_xy=allowed_wells_xy,
        carrying_open_plate=carrying_open_plate,
        headspace_safety_fraction=headspace_safety_fraction,
    )


def convert_finebio_to_se3_plans(
    annotations: Sequence[FineBioActionAnnotation],
    *,
    seed: int = 0,
    arm_2d: Arm2D = DEFAULT_ARM,
    arm_se3: ArmSE3 = DEFAULT_ARM_SE3,
    deck: Optional[DeckLayout] = None,
    scene: Optional[BenchScene] = None,
) -> List[Tuple[FineBioActionAnnotation, Optional[PickPlacePlan3D]]]:
    """Compile FineBio benchtop atomic manipulation steps into 6-DOF `ArmSE3` `PickPlacePlan3D` plans."""
    eff_deck = build_finebio_benchtop_deck() if deck is None else deck
    eff_scene = BenchScene(deck=eff_deck) if scene is None else scene
    rng = np.random.default_rng(seed)

    results: List[Tuple[FineBioActionAnnotation, Optional[PickPlacePlan3D]]] = []
    prev_q: Optional[np.ndarray] = None
    for ann in annotations:
        is_pickup = ann.verb in {"insert", "attach", "mount", "take_tip"} and ann.target_ref.startswith("tips:")
        plan = plan_3d_pick_place(
            ann.source_ref,
            ann.target_ref,
            rng,
            arm=arm_2d,
            arm_se3=arm_se3,
            scene=eff_scene,
            deck=eff_deck,
            attached_tip_length_m=ann.attached_tip_length_m,
            pickup_tip_from_source=is_pickup,
            start_q=prev_q,
        )
        if plan is not None and plan.waypoints_3d:
            prev_q = plan.waypoints_3d[-1][:2].copy()
        else:
            prev_q = select_ik_branch(eff_scene.tube_xy, arm=arm_2d, scene=eff_scene)
        results.append((ann, plan))
    return results


def audit_finebio_annotations(
    annotations: Sequence[FineBioActionAnnotation],
    *,
    seed: int = 0,
    arm_2d: Arm2D = DEFAULT_ARM,
    arm_se3: ArmSE3 = DEFAULT_ARM_SE3,
    deck: Optional[DeckLayout] = None,
    scene: Optional[BenchScene] = None,
    headspace_safety_fraction: float = 0.70,
) -> List[TrajectorySafetyAudit]:
    """Audit each FineBio 3D benchtop action annotation against `ArmSE3`, 96-well clearance, and slosh."""
    eff_deck = build_finebio_benchtop_deck() if deck is None else deck
    eff_scene = BenchScene(deck=eff_deck) if scene is None else scene
    compiled = convert_finebio_to_se3_plans(
        annotations,
        seed=seed,
        arm_2d=arm_2d,
        arm_se3=arm_se3,
        deck=eff_deck,
        scene=eff_scene,
    )

    audits: List[TrajectorySafetyAudit] = []
    for idx, (ann, plan) in enumerate(compiled):
        src_id = f"{ann.take_id}:step{idx:02d}:{ann.verb}:{ann.source_ref}->{ann.target_ref}"
        if plan is None or plan.q6_waypoints is None or len(plan.q6_waypoints) < 2:
            audits.append(
                TrajectorySafetyAudit(
                    source_id=src_id,
                    safe=False,
                    n_frames=0,
                    duration_s=round(ann.duration_s, 4),
                    se3_joint_limits_ok=False,
                    obstacle_clearance_ok=False,
                    min_obstacle_clearance_mm=0.0,
                    rim_clearance_ok=False,
                    plunge_verticality_ok=False,
                    max_plunge_tilt_deg=0.0,
                    slosh_safe=False,
                    spilled=False,
                    peak_lateral_accel_m_s2=0.0,
                    peak_meniscus_rise_mm=0.0,
                    free_headspace_mm=1.47,
                    meniscus_to_headspace_ratio=0.0,
                    recommended_min_duration_s=0.0,
                    violations=("PLAN_INFEASIBLE",),
                )
            )
            continue

        # Interpolate C2 minimum-jerk profile over the annotated segment duration [start_s, end_s]
        planar_wps = [wp[:2] for wp in plan.waypoints_3d]
        tp_ref = parameterize_path(
            planar_wps,
            arm=arm_2d,
            scene=eff_scene.with_attached_tip(plan.attached_tip_length_m, deck=eff_deck),
            well_radius_m=0.00345,
            fill_height_m=0.00920,
            well_depth_m=0.01067,
            damping_ratio=0.028,
            headspace_safety_fraction=headspace_safety_fraction,
            rigid_forearm_mount=ann.carrying_open_plate,
            n_samples=121,
        )

        seg_dur = max(ann.duration_s, 0.05)
        n_dense = tp_ref.t.size
        tau = np.linspace(0.0, 1.0, n_dense, dtype=np.float64)

        # Construct dense 6-DOF SE(3) trajectory: plunge retract -> C2 minimum-jerk transit -> plunge
        start_yaw = float(wrap_angle(tp_ref.q[0, 0] + tp_ref.q[0, 1])) if ann.carrying_open_plate else 0.0
        end_yaw = float(wrap_angle(tp_ref.q[-1, 0] + tp_ref.q[-1, 1])) if ann.carrying_open_plate else 0.0
        pose_start = PoseSE3(xyz=plan.poses_se3[0].xyz, rpy=(0.0, 0.0, start_yaw))
        pose_end = PoseSE3(xyz=plan.poses_se3[-1].xyz, rpy=(0.0, 0.0, end_yaw))
        q6_start = arm_se3.ik_se3(pose_start, elbow=1)
        if q6_start is None:
            q6_start = arm_se3.ik_se3(pose_start, elbow=-1)
        if q6_start is None:
            q6_start = plan.q6_waypoints[0]
        q6_end = arm_se3.ik_se3(pose_end, elbow=1)
        if q6_end is None:
            q6_end = arm_se3.ik_se3(pose_end, elbow=-1)
        if q6_end is None:
            q6_end = plan.q6_waypoints[-1]

        q6_dense: List[np.ndarray] = [q6_start]
        t_dense: List[float] = [0.0]
        vertical_frac = 0.12
        transit_dur = seg_dur * (1.0 - 2.0 * vertical_frac)

        for k in range(n_dense):
            _, xy_k = arm_2d.fk(tp_ref.q[k])
            yaw_k = float(wrap_angle(tp_ref.q[k, 0] + tp_ref.q[k, 1])) if ann.carrying_open_plate else 0.0
            pose_k = PoseSE3(xyz=(float(xy_k[0]), float(xy_k[1]), plan.z_safe_transit_m), rpy=(0.0, 0.0, yaw_k))
            q6_k = arm_se3.ik_se3(pose_k, elbow=1)
            if q6_k is None:
                q6_k = arm_se3.ik_se3(pose_k, elbow=-1)
            if q6_k is None:
                q6_k = plan.q6_waypoints[1]
            q6_dense.append(q6_k)
            t_dense.append(seg_dur * vertical_frac + float(tau[k]) * transit_dur)

        q6_dense.append(q6_end)
        t_dense.append(seg_dur)

        src_pose_xy = np.asarray(plan.poses_se3[0].xyz[:2], dtype=np.float64)
        dst_pose_xy = np.asarray(plan.poses_se3[-1].xyz[:2], dtype=np.float64)
        audit = audit_se3_trajectory(
            np.stack(q6_dense, axis=0),
            np.asarray(t_dense, dtype=np.float64),
            source_id=src_id,
            arm_se3=arm_se3,
            arm_2d=arm_2d,
            scene=eff_scene,
            attached_tip_length_m=plan.attached_tip_length_m,
            allowed_wells_xy=[src_pose_xy, dst_pose_xy],
            carrying_open_plate=ann.carrying_open_plate,
            headspace_safety_fraction=headspace_safety_fraction,
        )
        rec_dur = max(audit.recommended_min_duration_s, tp_ref.duration_s / max(1.0 - 2.0 * vertical_frac, 0.5))
        if not audit.slosh_safe:
            audit = TrajectorySafetyAudit(
                **{
                    **audit.to_dict(),
                    "recommended_min_duration_s": round(float(rec_dur), 4),
                }
            )
        audits.append(audit)

    return audits


__all__ = [
    "DatasetModality",
    "FineBioActionAnnotation",
    "LeRobotEpisode",
    "OPEN_MANIPULATION_DATASETS",
    "OpenManipulationDataset",
    "TrajectorySafetyAudit",
    "VALID_MODALITIES",
    "audit_finebio_annotations",
    "audit_lerobot_episode",
    "audit_se3_trajectory",
    "build_finebio_benchtop_deck",
    "check_commercial_compliance",
    "convert_finebio_to_se3_plans",
    "get_dataset",
    "list_datasets",
    "load_finebio_annotations",
    "load_lerobot_episode",
]
