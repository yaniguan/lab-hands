"""CLI utility to inspect, license-check, fetch samples, and safety-audit open manipulation datasets.

All downloaded manifests and sample trajectory shards stay strictly inside gitignored `data/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from lab_hands.datasets import (  # noqa: E402
    OPEN_MANIPULATION_DATASETS,
    VALID_MODALITIES,
    audit_finebio_annotations,
    audit_lerobot_episode,
    check_commercial_compliance,
    get_dataset,
    list_datasets,
    load_finebio_annotations,
    load_lerobot_episode,
)

DEFAULT_DATA_DIR = REPO_DIR / "data" / "open_datasets"
_USER_AGENT = "lab-hands-open-datasets/2026.11"


def _http_get_bytes(url: str, *, max_bytes: Optional[int] = None, timeout_s: float = 15.0) -> TupleResult:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        status = int(resp.status)
        body = resp.read() if max_bytes is None else resp.read(max_bytes)
        return status, body


TupleResult = tuple[int, bytes]


def verify_dataset_urls(
    dataset_ids: Optional[Sequence[str]] = None,
    *,
    timeout_s: float = 12.0,
) -> Dict[str, Dict[str, Any]]:
    """Perform live HTTP reachability checks against registered dataset URLs."""
    ids = list(OPEN_MANIPULATION_DATASETS.keys()) if dataset_ids is None else [get_dataset(d).dataset_id for d in dataset_ids]
    report: Dict[str, Dict[str, Any]] = {}
    for ds_id in ids:
        spec = get_dataset(ds_id)
        try:
            status, _ = _http_get_bytes(spec.url, max_bytes=1024, timeout_s=timeout_s)
            report[ds_id] = {
                "url": spec.url,
                "status_code": status,
                "ok": 200 <= status < 400,
                "spdx_license": spec.spdx_license,
                "commercial_use_ok": spec.commercial_use_ok,
            }
        except Exception as exc:
            report[ds_id] = {
                "url": spec.url,
                "status_code": None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "spdx_license": spec.spdx_license,
                "commercial_use_ok": spec.commercial_use_ok,
            }
    return report


def fetch_dataset_sample(
    dataset_id: str,
    *,
    out_dir: Path = DEFAULT_DATA_DIR,
    commercial_only: bool = False,
    max_rows: int = 64,
    timeout_s: float = 15.0,
) -> Path:
    """Fetch lightweight dataset metadata and sample rows into gitignored `data/open_datasets/<id>/`."""
    spec = get_dataset(dataset_id)
    if commercial_only:
        ok, blocked = check_commercial_compliance([spec.dataset_id])
        if not ok:
            raise PermissionError(
                f"Blocked non-commercial dataset {blocked[0]!r} (SPDX: {spec.spdx_license}) "
                "under --commercial-only policy."
            )

    if max_rows < 1 or max_rows > 500:
        raise ValueError(f"max_rows must be in 1..500, got {max_rows}")

    target_dir = Path(out_dir) / spec.dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        **asdict(spec),
        "requested_max_rows": int(max_rows),
        "artifacts": [],
    }

    if spec.hf_repo_id is not None:
        encoded_repo = urllib.parse.quote(spec.hf_repo_id, safe="")
        info_url = f"https://huggingface.co/datasets/{spec.hf_repo_id}/resolve/main/meta/info.json"
        try:
            st_info, info_bytes = _http_get_bytes(info_url, max_bytes=262_144, timeout_s=timeout_s)
            if st_info == 200:
                info_path = target_dir / "info.json"
                info_path.write_bytes(info_bytes)
                manifest["artifacts"].append("info.json")
        except Exception:
            pass

        if spec.modality == "kinematics":
            rows_url = (
                f"https://datasets-server.huggingface.co/rows?dataset={encoded_repo}"
                f"&config=default&split=train&offset=0&length={int(max_rows)}"
            )
            st_rows, rows_bytes = _http_get_bytes(rows_url, max_bytes=2_097_152, timeout_s=timeout_s)
            if st_rows == 200:
                rows_path = target_dir / "sample_rows.json"
                rows_path.write_bytes(rows_bytes)
                manifest["artifacts"].append("sample_rows.json")
    elif spec.dataset_id == "finebio":
        gt_url = "https://raw.githubusercontent.com/aistairc/FineBio/main/step_segmentation/data/groundTruth/P01_01_01.txt"
        try:
            st_gt, gt_bytes = _http_get_bytes(gt_url, max_bytes=262_144, timeout_s=timeout_s)
            if st_gt == 200:
                (target_dir / "P01_01_01.txt").write_bytes(gt_bytes)
                manifest["artifacts"].append("P01_01_01.txt")
        except Exception:
            pass

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return target_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, license-verify, fetch samples, and safety-audit open lab-hands datasets."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered open datasets as JSON.",
    )
    parser.add_argument(
        "--modality",
        choices=list(VALID_MODALITIES),
        default=None,
        help="Filter listed datasets by modality.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Filter listed datasets by minimum publication year.",
    )
    parser.add_argument(
        "--commercial-only",
        action="store_true",
        help="Restrict listing or sample downloads to commercially permissive SPDX licenses.",
    )
    parser.add_argument(
        "--verify-urls",
        action="store_true",
        help="Perform live HTTP status checks on dataset URLs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset ID to fetch lightweight metadata/sample rows into gitignored data/.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=64,
        help="Maximum rows to fetch from HuggingFace datasets-server (1..500).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target directory for downloaded dataset samples (default: data/open_datasets).",
    )
    parser.add_argument(
        "--audit-sample",
        type=Path,
        default=None,
        help="Audit a local LeRobot (.parquet/.jsonl/.json) or FineBio (.json/.txt) file.",
    )
    parser.add_argument(
        "--sample-kind",
        choices=["lerobot", "finebio"],
        default="lerobot",
        help="Schema of --audit-sample input file.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Optional robot_type override ('ur5e', 'franka', or 'arm_se3') when auditing standalone JSONL files.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list:
        specs = list_datasets(
            modality=args.modality,
            commercial_only=args.commercial_only,
            min_year=args.min_year,
        )
        print(json.dumps([asdict(s) for s in specs], indent=2))
        return 0

    if args.verify_urls:
        target_ids: Optional[List[str]] = [args.dataset] if args.dataset else None
        report = verify_dataset_urls(target_ids)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if all(v["ok"] for v in report.values()) else 1

    if args.audit_sample is not None:
        if args.sample_kind == "lerobot":
            ep = load_lerobot_episode(args.audit_sample, robot_type=args.robot_type)
            audit = audit_lerobot_episode(ep)
            print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
            return 0 if audit.safe is True else 2
        anns = load_finebio_annotations(args.audit_sample)
        audits = audit_finebio_annotations(anns)
        print(json.dumps([a.to_dict() for a in audits], indent=2, sort_keys=True))
        return 0 if all(a.safe for a in audits) else 2

    if args.dataset is not None:
        try:
            saved_dir = fetch_dataset_sample(
                args.dataset,
                out_dir=args.out_dir,
                commercial_only=args.commercial_only,
                max_rows=args.max_rows,
            )
        except PermissionError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 3
        print(json.dumps({"ok": True, "dataset_dir": str(saved_dir)}, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
