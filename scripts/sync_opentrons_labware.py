"""Sync official Opentrons v1 labware Z dimensions, well bottoms, and tip lengths (#34).

Reads all 24 `STEPBOT_LABWARE_CATALOG` load names from `opentrons-shared-data`
(Apache-2.0, Copyright Opentrons Labworks, Inc.) and writes
`lab_hands/data/opentrons_labware_v1.json` so `lab_hands` keeps a zero-network,
pure-`numpy` runtime while matching official Opentrons labware definitions to 0.01 mm.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = REPO_ROOT / "lab_hands" / "data" / "opentrons_labware_v1.json"

LOAD_NAMES_24: Tuple[str, ...] = (
    "corning_96_wellplate_360ul_flat",
    "corning_384_wellplate_112ul_flat",
    "corning_24_wellplate_3.4ml_flat",
    "corning_12_wellplate_6.9ml_flat",
    "corning_6_wellplate_16.8ml_flat",
    "nest_96_wellplate_200ul_flat",
    "nest_96_wellplate_100ul_pcr_full_skirt",
    "nest_96_wellplate_2ml_deep",
    "nest_12_reservoir_15ml",
    "nest_1_reservoir_195ml",
    "usascientific_12_reservoir_22ml",
    "agilent_1_reservoir_290ml",
    "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap",
    "opentrons_24_tuberack_nest_1.5ml_snapcap",
    "opentrons_15_tuberack_falcon_15ml_conical",
    "opentrons_6_tuberack_falcon_50ml_conical",
    "opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical",
    "opentrons_96_tiprack_300ul",
    "opentrons_96_tiprack_20ul",
    "opentrons_96_tiprack_10ul",
    "opentrons_96_tiprack_1000ul",
    "opentrons_96_filtertiprack_200ul",
    "opentrons_96_filtertiprack_20ul",
    "opentrons_96_filtertiprack_1000ul",
)


def extract_labware_entry(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Extract official Z dimensions, A1 well geometry, and tiprack parameters in mm."""
    dims = defn["dimensions"]
    a1 = defn["wells"]["A1"]
    params = defn["parameters"]
    is_tiprack = bool(params.get("isTiprack", False))
    shape = str(a1.get("shape", "circular"))

    a1_entry: Dict[str, Any] = {
        "shape": shape,
        "z_mm": round(float(a1["z"]), 4),
        "depth_mm": round(float(a1["depth"]), 4),
    }
    if "diameter" in a1 and a1["diameter"] is not None:
        a1_entry["diameter_mm"] = round(float(a1["diameter"]), 4)
        well_radius_mm = 0.5 * float(a1["diameter"])
    else:
        x_dim = float(a1["xDimension"])
        y_dim = float(a1["yDimension"])
        a1_entry["xDimension_mm"] = round(x_dim, 4)
        a1_entry["yDimension_mm"] = round(y_dim, 4)
        well_radius_mm = 0.5 * min(x_dim, y_dim)

    entry: Dict[str, Any] = {
        "load_name": str(params["loadName"]),
        "version": int(defn.get("version", 1)),
        "xDimension_mm": round(float(dims["xDimension"]), 4),
        "yDimension_mm": round(float(dims["yDimension"]), 4),
        "zDimension_mm": round(float(dims["zDimension"]), 4),
        "a1": a1_entry,
        "well_radius_mm": round(well_radius_mm, 4),
        "isTiprack": is_tiprack,
    }
    if is_tiprack:
        tip_len = float(params["tipLength"])
        tip_ov = float(params["tipOverlap"])
        entry["tipLength_mm"] = round(tip_len, 4)
        entry["tipOverlap_mm"] = round(tip_ov, 4)
        entry["effectiveTipLength_mm"] = round(tip_len - tip_ov, 4)
    return entry


def build_opentrons_labware_bundle() -> Dict[str, Any]:
    import opentrons_shared_data.labware as ot_lw  # type: ignore[import-not-found]

    pkg_version = importlib_metadata.version("opentrons-shared-data")
    labware_map: Dict[str, Any] = {}
    for load_name in LOAD_NAMES_24:
        defn = ot_lw.load_definition(load_name, 1)
        labware_map[load_name] = extract_labware_entry(defn)

    return {
        "schema": "lab-hands/opentrons-labware-z-sync/v1",
        "source_package": "opentrons-shared-data",
        "source_version": pkg_version,
        "definition_version": 1,
        "spdx_license": "Apache-2.0",
        "license_notice": (
            "Derived from Opentrons labware definitions in opentrons-shared-data "
            "(Copyright Opentrons Labworks, Inc.), licensed under the Apache License, "
            "Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)."
        ),
        "labware": labware_map,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync 24 Opentrons labware Z/tip definitions.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    bundle = build_opentrons_labware_bundle()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Synced {len(bundle['labware'])} definitions (opentrons-shared-data {bundle['source_version']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
