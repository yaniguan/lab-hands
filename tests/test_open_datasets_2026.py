"""Tests for 2024-2026 open wet-lab, transparent-vessel depth & visuo-tactile datasets integration."""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from lab_hands import (
    DEFAULT_ARM,
    DEFAULT_ARM_SE3,
    DEFAULT_SCENE,
    OPEN_MANIPULATION_DATASETS,
    VALID_MODALITIES,
    BenchScene,
    OpenManipulationDataset,
    PoseSE3,
    audit_finebio_annotations,
    audit_lerobot_episode,
    audit_se3_trajectory,
    build_finebio_benchtop_deck,
    check_commercial_compliance,
    convert_finebio_to_se3_plans,
    get_dataset,
    list_datasets,
    load_finebio_annotations,
    load_lerobot_episode,
    parameterize_path,
    plan_3d_pick_place,
)
from scripts import fetch_open_datasets as fetch_cli


def _build_safe_se3_episode_rows(
    *,
    source_ref: str = "TUBE",
    target_ref: str = "A1",
    fps: float = 50.0,
    duration_scale: float = 1.35,
) -> List[Dict[str, Any]]:
    """Synthesize a smooth 50 Hz 6-DOF `ArmSE3` episode in LeRobot schema from `lab_hands` primitives."""
    plan = plan_3d_pick_place(
        source_ref,
        target_ref,
        np.random.default_rng(7),
        arm=DEFAULT_ARM,
        arm_se3=DEFAULT_ARM_SE3,
        scene=DEFAULT_SCENE,
    )
    assert plan is not None and plan.q6_waypoints is not None

    planar_wps = [wp[:2] for wp in plan.waypoints_3d]
    tp = parameterize_path(
        planar_wps,
        arm=DEFAULT_ARM,
        scene=DEFAULT_SCENE,
        well_radius_m=0.00345,
        fill_height_m=0.00920,
        well_depth_m=0.01067,
        damping_ratio=0.028,
        headspace_safety_fraction=0.70,
        rigid_forearm_mount=False,
        n_samples=101,
    )
    transit_dur = max(1.6, tp.duration_s * duration_scale)
    plunge_dur = 0.30
    total_dur = 2.0 * plunge_dur + transit_dur
    n_frames = int(round(total_dur * fps)) + 1
    times = np.linspace(0.0, total_dur, n_frames, dtype=np.float64)

    _, xy_src = DEFAULT_ARM.fk(planar_wps[0])
    _, xy_dst = DEFAULT_ARM.fk(planar_wps[-1])

    rows: List[Dict[str, Any]] = []
    for idx, t_cur in enumerate(times):
        if t_cur <= plunge_dur:
            u = t_cur / plunge_dur
            s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            z_cur = (1.0 - s) * plan.z_pick_m + s * plan.z_safe_transit_m
            xy_cur = xy_src
        elif t_cur >= plunge_dur + transit_dur:
            u = (t_cur - plunge_dur - transit_dur) / plunge_dur
            s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            z_cur = (1.0 - s) * plan.z_safe_transit_m + s * plan.z_place_m
            xy_cur = xy_dst
        else:
            tau = (t_cur - plunge_dur) / transit_dur
            x_cur = float(np.interp(tau * tp.duration_s, tp.t, tp.tip_xy[:, 0]))
            y_cur = float(np.interp(tau * tp.duration_s, tp.t, tp.tip_xy[:, 1]))
            xy_cur = np.array([x_cur, y_cur], dtype=np.float64)
            z_cur = plan.z_safe_transit_m

        pose = PoseSE3(xyz=(float(xy_cur[0]), float(xy_cur[1]), float(z_cur)), rpy=(0.0, 0.0, 0.0))
        q6 = DEFAULT_ARM_SE3.ik_se3(pose, elbow=1)
        assert q6 is not None
        state7 = [round(float(v), 7) for v in q6] + [0.02]
        rows.append(
            {
                "episode_index": 0,
                "frame_index": idx,
                "timestamp": round(float(t_cur), 6),
                "observation.state": state7,
                "action": state7,
                "task_index": 0,
                "robot_type": "arm_se3",
            }
        )
    return rows


def test_open_datasets_registry_modalities_and_years() -> None:
    assert len(OPEN_MANIPULATION_DATASETS) >= 10
    recent_2024_2026 = list_datasets(min_year=2024)
    assert len(recent_2024_2026) >= 8

    seen_modalities = {spec.modality for spec in OPEN_MANIPULATION_DATASETS.values()}
    assert seen_modalities == set(VALID_MODALITIES)

    for ds_id, spec in OPEN_MANIPULATION_DATASETS.items():
        assert ds_id == spec.dataset_id
        assert spec.url.startswith("https://")
        assert spec.modality in VALID_MODALITIES
        assert len(spec.spdx_license) >= 3
        assert len(spec.summary) >= 30


def test_non_commercial_vs_permissive_license_guardrails() -> None:
    # Restricted / non-commercial datasets
    transcg = get_dataset("TransCG")
    finebio = get_dataset("FineBio")
    octopi = get_dataset("Octopi")
    assert transcg.commercial_use_ok is False
    assert transcg.spdx_license == "CC-BY-NC-SA-4.0"
    assert finebio.commercial_use_ok is False
    assert finebio.modality == "wetlab_video"
    assert octopi.commercial_use_ok is False

    # Commercially permissive datasets
    droid = get_dataset("DROID")
    lerobot = get_dataset("LeRobot")
    autobio = get_dataset("autobio_pipette_mujoco")
    transproteus = get_dataset("TransProteus")
    clearpose = get_dataset("ClearPose")
    layered = get_dataset("LayeredDepth")
    tvl = get_dataset("TVL")
    tactx = get_dataset("TactX")
    feelsight = get_dataset("FeelSight")

    for permissive_ds in (droid, lerobot, autobio, transproteus, clearpose, layered, tvl, tactx, feelsight):
        assert permissive_ds.commercial_use_ok is True

    assert droid.spdx_license == "CC-BY-4.0"
    assert lerobot.spdx_license == "Apache-2.0"

    ok_perm, blocked_perm = check_commercial_compliance(["droid", "lerobot", "transproteus", "tvl", "tactx"])
    assert ok_perm is True and blocked_perm == []

    ok_mix, blocked_mix = check_commercial_compliance(["droid", "transcg", "lerobot", "finebio"])
    assert ok_mix is False
    assert blocked_mix == ["transcg", "finebio"]

    comm_list = list_datasets(commercial_only=True)
    assert all(d.commercial_use_ok for d in comm_list)
    assert "transcg" not in {d.dataset_id for d in comm_list}
    assert "finebio" not in {d.dataset_id for d in comm_list}

    # Fail-closed dataclass invariant: NC or LicenseRef cannot set commercial_use_ok=True
    with pytest.raises(ValueError, match="non-commercial"):
        OpenManipulationDataset(
            dataset_id="bad_nc",
            title="Invalid NC Commercial Flag",
            modality="transparent_depth",
            url="https://example.org/dataset",
            spdx_license="CC-BY-NC-4.0",
            commercial_use_ok=True,
            year=2025,
            venue="Test",
            summary="Should fail validation when marking CC-BY-NC-4.0 as commercially permissive.",
        )

    with pytest.raises(ValueError, match="Invalid modality"):
        OpenManipulationDataset(
            dataset_id="bad_mod",
            title="Invalid Modality",
            modality="audio_only",  # type: ignore[arg-type]
            url="https://example.org/dataset",
            spdx_license="MIT",
            commercial_use_ok=True,
            year=2025,
            venue="Test",
            summary="Should fail validation when modality is outside VALID_MODALITIES.",
        )


def test_lerobot_jsonl_hf_rows_and_parquet_loaders_and_safe_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _build_safe_se3_episode_rows(source_ref="TUBE", target_ref="A1")

    # 1. JSONL trajectory loader + safety audit
    jsonl_path = tmp_path / "episode_000000.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ep_jsonl = load_lerobot_episode(jsonl_path, dataset_id="lerobot", robot_type="arm_se3")
    assert ep_jsonl.n_frames == len(rows)
    assert ep_jsonl.fps == pytest.approx(50.0, abs=0.2)
    assert ep_jsonl.q6_se3.shape == (len(rows), 6)
    assert ep_jsonl.gripper is not None and ep_jsonl.gripper.shape == (len(rows),)

    audit_ok = audit_lerobot_episode(
        ep_jsonl,
        base_pose_in_deck=PoseSE3(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)),
        allowed_wells_xy=[DEFAULT_SCENE.tube_xy, DEFAULT_SCENE.plate.well_xy("A1")],
    )
    assert audit_ok.safe is True
    assert audit_ok.se3_joint_limits_ok is True
    assert audit_ok.obstacle_clearance_ok is True
    assert audit_ok.min_obstacle_clearance_mm >= 5.0
    assert audit_ok.rim_clearance_ok is True
    assert audit_ok.plunge_verticality_ok is True
    assert audit_ok.max_plunge_tilt_deg == pytest.approx(0.0, abs=1e-4)
    assert audit_ok.slosh_safe is True
    assert audit_ok.spilled is False
    assert audit_ok.meniscus_to_headspace_ratio < 0.70
    assert audit_ok.violations == ()

    # 2. HuggingFace datasets-server /rows JSON wrapper (14-DoF bimanual AutoBio UR5e schema)
    hf_rows_payload = {
        "features": [{"name": "state"}, {"name": "actions"}, {"name": "timestamp"}],
        "rows": [
            {
                "row_idx": i,
                "row": {
                    "episode_index": 2,
                    "frame_index": i,
                    "timestamp": r["timestamp"],
                    "state": r["observation.state"] + r["observation.state"],
                    "actions": r["action"] + r["action"],
                },
            }
            for i, r in enumerate(rows)
        ],
    }
    hf_json_path = tmp_path / "sample_rows.json"
    hf_json_path.write_text(json.dumps(hf_rows_payload), encoding="utf-8")
    ep_hf = load_lerobot_episode(
        hf_json_path,
        episode_index=2,
        dataset_id="autobio_pipette_mujoco",
        arm_joint_slice=(7, 13),
        gripper_index=13,
    )
    assert np.allclose(ep_hf.q6_se3, ep_jsonl.q6_se3)

    # 3. Binary Apache Parquet round-trip when pyarrow is available + graceful ImportError fallback
    parquet_path = tmp_path / "episode_000000.parquet"
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        pq.write_table(pa.Table.from_pylist(rows), parquet_path)
        ep_pq = load_lerobot_episode(parquet_path, dataset_id="lerobot")
        assert np.allclose(ep_pq.q6_se3, ep_jsonl.q6_se3)
    except ImportError:
        parquet_path.write_bytes(b"PAR1")

    real_import = builtins.__import__

    def _block_tabular_imports(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("pyarrow", "pandas")):
            raise ImportError(f"Blocked optional import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_tabular_imports)
    with pytest.raises(ImportError, match="pyarrow"):
        load_lerobot_episode(parquet_path)


def test_se3_safety_auditor_catches_slosh_rim_tilt_obstacle_and_limit_faults() -> None:
    # 1. Aggressive lateral teleop shake -> triggers 96-well meniscus spill
    n = 61
    t_fast = np.linspace(0.0, 0.30, n)
    q6_shake = []
    for t_cur in t_fast:
        x = 0.22 + 0.045 * math.sin(2.0 * math.pi * t_cur / 0.15)
        pose = PoseSE3(xyz=(x, 0.02, 0.15), rpy=(0.0, 0.0, 0.0))
        q6 = DEFAULT_ARM_SE3.ik_se3(pose, elbow=1)
        assert q6 is not None
        q6_shake.append(q6)
    audit_slosh = audit_se3_trajectory(q6_shake, t_fast, source_id="teleop_shake")
    assert audit_slosh.safe is False
    assert audit_slosh.slosh_safe is False
    assert audit_slosh.spilled is True
    assert "SLOSH_HEADSPACE_EXCEEDED" in audit_slosh.violations
    assert audit_slosh.recommended_min_duration_s > audit_slosh.duration_s

    # 2. Low-Z horizontal sweep across 96-well plate below 15 mm rim -> WELL_RIM_COLLISION
    w_a1 = DEFAULT_SCENE.plate.well_xy("A1")
    w_a4 = DEFAULT_SCENE.plate.well_xy("A4")
    q6_rim = []
    for alpha in np.linspace(0.0, 1.0, 15):
        xy = (1.0 - alpha) * w_a1 + alpha * w_a4
        pose = PoseSE3(xyz=(float(xy[0]), float(xy[1]), 0.011), rpy=(0.0, 0.0, 0.0))
        q6 = DEFAULT_ARM_SE3.ik_se3(pose, elbow=1)
        assert q6 is not None
        q6_rim.append(q6)
    audit_rim = audit_se3_trajectory(
        q6_rim,
        np.linspace(0.0, 2.0, 15),
        source_id="low_z_rim_clip",
        allowed_wells_xy=[w_a1],
    )
    assert audit_rim.rim_clearance_ok is False
    assert "WELL_RIM_COLLISION" in audit_rim.violations

    # 3. Tilted pipette plunge inside well A1 bore (3.5 deg > 1.5 deg) -> PLUNGE_TILT_EXCEEDED
    pose_tilted = PoseSE3(
        xyz=(float(w_a1[0]), float(w_a1[1]), 0.008),
        rpy=(math.radians(3.5), 0.0, 0.0),
    )
    q6_tilted = DEFAULT_ARM_SE3.ik_se3(pose_tilted, elbow=1)
    assert q6_tilted is not None
    audit_tilt = audit_se3_trajectory(
        [q6_tilted, q6_tilted],
        [0.0, 0.5],
        source_id="tilted_plunge",
        allowed_wells_xy=[w_a1],
    )
    assert audit_tilt.plunge_verticality_ok is False
    assert audit_tilt.max_plunge_tilt_deg == pytest.approx(3.5, abs=1e-2)
    assert "PLUNGE_TILT_EXCEEDED" in audit_tilt.violations

    # 4. Obstacle-grazing configuration (< 5.0 mm capsule-AABB clearance) & SE3 joint limit breach
    blocked_scene = BenchScene(obstacles=((0.20, 0.048, 0.25, 0.08),))
    pose_obs = PoseSE3(xyz=(0.20, 0.045, 0.14), rpy=(0.0, 0.0, 0.0))
    q6_obs = DEFAULT_ARM_SE3.ik_se3(pose_obs, elbow=1)
    assert q6_obs is not None
    q6_bad_lim = q6_obs.copy()
    q6_bad_lim[2] = 3.05  # exceeds |q3| <= 2.80 rad limit
    audit_obs = audit_se3_trajectory(
        [q6_obs, q6_bad_lim],
        [0.0, 1.5],
        source_id="obstacle_and_limit",
        scene=blocked_scene,
    )
    assert audit_obs.obstacle_clearance_ok is False
    assert audit_obs.se3_joint_limits_ok is False
    assert "OBSTACLE_CLEARANCE_BELOW_5MM" in audit_obs.violations
    assert "SE3_JOINT_LIMIT_EXCEEDED" in audit_obs.violations


def test_finebio_action_annotations_loader_se3_plans_and_slosh_audit(tmp_path: Path) -> None:
    deck = build_finebio_benchtop_deck()
    plate96 = deck.labware_by_alias["plate96"]
    b2_xy = plate96.well_xy("B2")

    finebio_db = {
        "database": {
            "P01_01_01": {
                "subset": "validation",
                "fps": 30.0,
                "duration": 22.0,
                "annotations": [
                    {
                        "segment": [0.0, 5.2],
                        "hand_label": "right",
                        "verb_label": "insert",
                        "manipulated_label": "micropipette",
                        "affected_label": "tip_rack",
                    },
                    {
                        "segment": [5.5, 8.0],
                        "hand_label": "right",
                        "verb_label": "aspirate",
                        "manipulated_label": "micropipette",
                        "affected_label": "microtube",
                    },
                    {
                        "segment": [8.3, 11.0],
                        "hand_label": "right",
                        "verb_label": "dispense",
                        "manipulated_label": "micropipette",
                        "affected_label": "96_well_plate",
                        "target_xyz_m": [float(b2_xy[0]), float(b2_xy[1]), 0.007],
                    },
                    {
                        "segment": [11.5, 18.8],
                        "hand_label": "right",
                        "verb_label": "eject",
                        "manipulated_label": "micropipette",
                        "affected_label": "TUBE",
                    },
                ],
            }
        }
    }
    db_path = tmp_path / "finebio_p01.json"
    db_path.write_text(json.dumps(finebio_db), encoding="utf-8")

    anns = load_finebio_annotations(db_path, take_id="P01_01_01", deck=deck)
    assert len(anns) == 4
    assert anns[0].target_ref == "tips:A1"
    assert anns[0].attached_tip_length_m == pytest.approx(0.05183, abs=1e-5)
    assert anns[1].target_ref == "eppendorf:A1"
    assert anns[1].attached_tip_length_m == pytest.approx(0.05183, abs=1e-5)
    # 3D coordinate snapping resolves B2 on slot-5 96-well plate
    assert anns[2].target_ref == "plate96:B2"
    assert anns[2].attached_tip_length_m == pytest.approx(0.05183, abs=1e-5)

    plans = convert_finebio_to_se3_plans(anns, seed=0, deck=deck)
    assert all(plan is not None for _, plan in plans)
    # With 124.35 mm Falcon rack on slot 1 + 51.83 mm mounted 200 uL filter tip, safe transit > 181 mm
    for _, plan in plans[:3]:
        assert plan is not None
        assert plan.z_safe_transit_m >= 0.12435 + 0.05183 + 0.005

    audits = audit_finebio_annotations(anns, seed=0, deck=deck)
    assert len(audits) == 4
    assert all(a.safe for a in audits)
    assert all(a.min_obstacle_clearance_mm >= 5.0 for a in audits)
    assert all(a.slosh_safe for a in audits)

    # Tabular FineBio .txt take with a rushed 0.16 s open 96-well plate carry flags corner-well slosh!
    txt_path = tmp_path / "P02_03_01.txt"
    txt_path.write_text(
        "# start_s end_s verb manipulated affected hand source_ref target_ref\n"
        "0.00 3.20 move 96_well_plate pcr_plate right plate96:A1 pcr96:A1\n"
        "3.50 3.66 move 96_well_plate pcr_plate right pcr96:A1 plate96:A1\n",
        encoding="utf-8",
    )
    plate_anns = load_finebio_annotations(txt_path, deck=deck)
    assert len(plate_anns) == 2
    assert all(a.carrying_open_plate is True for a in plate_anns)

    plate_audits = audit_finebio_annotations(plate_anns, seed=0, deck=deck)
    assert plate_audits[0].safe is True
    assert plate_audits[1].safe is False
    assert plate_audits[1].slosh_safe is False
    assert "SLOSH_HEADSPACE_EXCEEDED" in plate_audits[1].violations
    assert plate_audits[1].recommended_min_duration_s > plate_audits[1].duration_s


def test_fetch_open_datasets_cli_and_gitignore_guard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gitignore_lines = {
        ln.strip() for ln in (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "data/" in gitignore_lines

    rc_list = fetch_cli.main(["--list", "--commercial-only", "--min-year", "2024"])
    assert rc_list == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) >= 7
    assert all(item["commercial_use_ok"] is True for item in listed)

    # Commercial-only mode blocks FineBio and TransCG before touching network
    rc_blocked = fetch_cli.main(
        ["--dataset", "finebio", "--commercial-only", "--out-dir", str(tmp_path)]
    )
    assert rc_blocked == 3
    err_payload = json.loads(capsys.readouterr().err)
    assert err_payload["ok"] is False
    assert "finebio" in err_payload["error"]

    # CLI --audit-sample returns 0 on safe offline LeRobot JSONL
    safe_rows = _build_safe_se3_episode_rows()
    sample_file = tmp_path / "safe_ep.jsonl"
    sample_file.write_text("\n".join(json.dumps(r) for r in safe_rows) + "\n", encoding="utf-8")
    rc_audit = fetch_cli.main(["--audit-sample", str(sample_file), "--sample-kind", "lerobot"])
    assert rc_audit == 0
    audit_json = json.loads(capsys.readouterr().out)
    assert audit_json["safe"] is True
