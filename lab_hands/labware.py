"""ANSI/SLAS (SBS) microplate geometries and stepbot-compatible 24-labware catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

_WELL_RE = re.compile(r"^([A-Z]+)([1-9]\d*)$")


def _row_label(idx: int) -> str:
    if idx < 0:
        raise ValueError(f"Row index must be >= 0, got {idx}")
    if idx < 26:
        return chr(ord("A") + idx)
    return chr(ord("A") + idx // 26 - 1) + chr(ord("A") + idx % 26)


def _row_index(label: str) -> int:
    val = 0
    for ch in label:
        val = val * 26 + (ord(ch) - ord("A") + 1)
    return val - 1


@dataclass(frozen=True)
class PlateGeometry:
    """Generalized labware grid geometry supporting 6..384-well SBS footprints and tube racks."""

    rows: int = 8
    cols: int = 12
    pitch_x: float = 0.009
    pitch_y: float = 0.009
    origin_xy: Tuple[float, float] = (0.20, 0.05)
    load_name: str = "corning_96_wellplate_360ul_flat"
    max_volume_ul: float = 360.0
    is_tiprack: bool = False
    z_bottom_m: float = 0.005
    z_rim_m: float = 0.015
    well_radius_m: float = 0.0034
    custom_well_offsets: Optional[Mapping[str, Tuple[float, float]]] = field(default=None)

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"rows and cols must be >= 1, got ({self.rows}, {self.cols})")
        if self.pitch_x < 0 or self.pitch_y < 0:
            raise ValueError(f"pitch_x and pitch_y must be >= 0, got ({self.pitch_x}, {self.pitch_y})")

    @property
    def n_wells(self) -> int:
        if self.custom_well_offsets is not None:
            return len(self.custom_well_offsets)
        return self.rows * self.cols

    @property
    def row_names(self) -> Tuple[str, ...]:
        return tuple(_row_label(r) for r in range(self.rows))

    def ordered_wells(self, order: str = "row") -> Tuple[str, ...]:
        if order not in ("row", "col"):
            raise ValueError(f"order must be 'row' or 'col', got {order!r}")
        if self.custom_well_offsets is not None:
            keys = list(self.custom_well_offsets.keys())
            if order == "col":
                return tuple(keys)
            return tuple(
                sorted(
                    keys,
                    key=lambda w: (_row_index(_WELL_RE.match(w).group(1)), int(_WELL_RE.match(w).group(2))),  # type: ignore[union-attr]
                )
            )
        if order == "row":
            return tuple(f"{_row_label(r)}{c + 1}" for r in range(self.rows) for c in range(self.cols))
        return tuple(f"{_row_label(r)}{c + 1}" for c in range(self.cols) for r in range(self.rows))

    def well_name(self, index: int, order: str = "row") -> str:
        """1-based well index -> canonical alphanumeric well name."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"well index must be an integer, got {type(index).__name__}")
        if order not in ("row", "col"):
            raise ValueError(f"order must be 'row' or 'col', got {order!r}")
        total = self.n_wells
        if index < 1 or index > total:
            raise ValueError(f"well index must be in 1..{total}, got {index}")
        if self.custom_well_offsets is not None:
            return self.ordered_wells(order)[index - 1]
        i = index - 1
        r, c = (i // self.cols, i % self.cols) if order == "row" else (i % self.rows, i // self.rows)
        return f"{_row_label(r)}{c + 1}"

    def well_xy(self, name: str) -> np.ndarray:
        """Well identifier (e.g. 'A1', 'h12') -> 2D Cartesian center in meters."""
        if not isinstance(name, str):
            raise ValueError(f"well name must be a string, got {type(name).__name__}")
        s = name.strip().upper()
        if self.custom_well_offsets is not None:
            if s not in self.custom_well_offsets:
                raise ValueError(f"Unknown well {name!r} for {self.load_name}")
            ox, oy = self.custom_well_offsets[s]
            return np.array([self.origin_xy[0] + ox, self.origin_xy[1] + oy], dtype=np.float64)

        m = _WELL_RE.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid well name {name!r}")
        r = _row_index(m.group(1))
        c = int(m.group(2)) - 1
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            raise ValueError(
                f"Well {name!r} is out of bounds for {self.rows}x{self.cols} labware ({self.load_name})"
            )
        return np.array(
            [self.origin_xy[0] + c * self.pitch_x, self.origin_xy[1] - r * self.pitch_y],
            dtype=np.float64,
        )

    def footprint_bounds(self, pad_m: float = 0.015) -> Tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) bounding footprint of the labware in meters."""
        eff_pad = max(float(pad_m), float(self.well_radius_m) + 0.005)
        if self.custom_well_offsets is not None and len(self.custom_well_offsets) > 0:
            xs = [self.origin_xy[0] + ox for ox, _ in self.custom_well_offsets.values()]
            ys = [self.origin_xy[1] + oy for _, oy in self.custom_well_offsets.values()]
            return (min(xs) - eff_pad, max(xs) + eff_pad, min(ys) - eff_pad, max(ys) + eff_pad)
        ox, oy = self.origin_xy
        px_min = ox - eff_pad
        px_max = ox + (self.cols - 1) * self.pitch_x + eff_pad
        py_min = oy - (self.rows - 1) * self.pitch_y - eff_pad
        py_max = oy + eff_pad
        return (px_min, px_max, py_min, py_max)

    def default_tip_length_m(self) -> float:
        """Nominal disposable pipette tip length (m) when picking up from a tiprack."""
        if not self.is_tiprack:
            return 0.0
        if self.max_volume_ul >= 1000.0:
            return 0.095
        if self.max_volume_ul >= 200.0:
            return 0.058
        return 0.046

    def with_origin(self, origin_xy: Tuple[float, float]) -> "PlateGeometry":
        return PlateGeometry(
            rows=self.rows,
            cols=self.cols,
            pitch_x=self.pitch_x,
            pitch_y=self.pitch_y,
            origin_xy=(float(origin_xy[0]), float(origin_xy[1])),
            load_name=self.load_name,
            max_volume_ul=self.max_volume_ul,
            is_tiprack=self.is_tiprack,
            z_bottom_m=self.z_bottom_m,
            z_rim_m=self.z_rim_m,
            well_radius_m=self.well_radius_m,
            custom_well_offsets=self.custom_well_offsets,
        )


DEFAULT_PLATE_96 = PlateGeometry()


def _build_falcon_4x50_6x15(origin_xy: Tuple[float, float] = (0.20, 0.05)) -> PlateGeometry:
    # 6x 15mL tubes in columns 1-2 (3 rows, 25 mm pitch) + 4x 50mL tubes in columns 3-4 (2 rows, 35 mm pitch)
    offsets: Dict[str, Tuple[float, float]] = {
        "A1": (0.000, 0.000),
        "B1": (0.000, -0.025),
        "C1": (0.000, -0.050),
        "A2": (0.025, 0.000),
        "B2": (0.025, -0.025),
        "C2": (0.025, -0.050),
        "A3": (0.055, -0.0075),
        "B3": (0.055, -0.0425),
        "A4": (0.090, -0.0075),
        "B4": (0.090, -0.0425),
    }
    return PlateGeometry(
        rows=3,
        cols=4,
        pitch_x=0.025,
        pitch_y=0.025,
        origin_xy=origin_xy,
        load_name="opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical",
        max_volume_ul=15000.0,
        z_bottom_m=0.005,
        z_rim_m=0.120,
        well_radius_m=0.0075,
        custom_well_offsets=offsets,
    )


# Complete 24-entry labware catalog aligned 1-to-1 with stepbot/src/stepbot/catalog.py
STEPBOT_LABWARE_CATALOG: Dict[str, PlateGeometry] = {
    spec.load_name: spec
    for spec in [
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "corning_96_wellplate_360ul_flat", 360.0, False, 0.005, 0.015, 0.0034),
        PlateGeometry(16, 24, 0.0045, 0.0045, (0.20, 0.05), "corning_384_wellplate_112ul_flat", 112.0, False, 0.005, 0.015, 0.0018),
        PlateGeometry(4, 6, 0.0193, 0.0193, (0.20, 0.05), "corning_24_wellplate_3.4ml_flat", 3400.0, False, 0.005, 0.020, 0.0081),
        PlateGeometry(3, 4, 0.026, 0.026, (0.20, 0.05), "corning_12_wellplate_6.9ml_flat", 6900.0, False, 0.005, 0.020, 0.0110),
        PlateGeometry(2, 3, 0.03912, 0.03912, (0.20, 0.05), "corning_6_wellplate_16.8ml_flat", 16800.0, False, 0.005, 0.020, 0.0174),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "nest_96_wellplate_200ul_flat", 200.0, False, 0.005, 0.015, 0.0033),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "nest_96_wellplate_100ul_pcr_full_skirt", 100.0, False, 0.005, 0.021, 0.0027),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "nest_96_wellplate_2ml_deep", 2000.0, False, 0.005, 0.042, 0.0041),
        PlateGeometry(1, 12, 0.009, 0.009, (0.20, 0.05), "nest_12_reservoir_15ml", 15000.0, False, 0.005, 0.032, 0.0040),
        PlateGeometry(1, 1, 0.0, 0.0, (0.25, 0.02), "nest_1_reservoir_195ml", 195000.0, False, 0.005, 0.040, 0.0350),
        PlateGeometry(1, 12, 0.009, 0.009, (0.20, 0.05), "usascientific_12_reservoir_22ml", 22000.0, False, 0.005, 0.044, 0.0040),
        PlateGeometry(1, 1, 0.0, 0.0, (0.25, 0.02), "agilent_1_reservoir_290ml", 290000.0, False, 0.005, 0.044, 0.0350),
        PlateGeometry(4, 6, 0.01989, 0.01928, (0.20, 0.05), "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap", 1500.0, False, 0.010, 0.075, 0.0045),
        PlateGeometry(4, 6, 0.01989, 0.01928, (0.20, 0.05), "opentrons_24_tuberack_nest_1.5ml_snapcap", 1500.0, False, 0.010, 0.075, 0.0045),
        PlateGeometry(3, 5, 0.025, 0.025, (0.20, 0.05), "opentrons_15_tuberack_falcon_15ml_conical", 15000.0, False, 0.008, 0.120, 0.0075),
        PlateGeometry(2, 3, 0.035, 0.035, (0.20, 0.05), "opentrons_6_tuberack_falcon_50ml_conical", 50000.0, False, 0.008, 0.120, 0.0135),
        _build_falcon_4x50_6x15(),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_tiprack_300ul", 300.0, True, 0.010, 0.065, 0.0030),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_tiprack_20ul", 20.0, True, 0.010, 0.060, 0.0025),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_tiprack_10ul", 10.0, True, 0.010, 0.060, 0.0025),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_tiprack_1000ul", 1000.0, True, 0.010, 0.085, 0.0040),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_filtertiprack_200ul", 200.0, True, 0.010, 0.065, 0.0030),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_filtertiprack_20ul", 20.0, True, 0.010, 0.060, 0.0025),
        PlateGeometry(8, 12, 0.009, 0.009, (0.20, 0.05), "opentrons_96_filtertiprack_1000ul", 1000.0, True, 0.010, 0.085, 0.0040),
    ]
}

# Standard ANSI/SLAS SBS slot spacing (132.5 mm x 90.5 mm slot pitch)
DEFAULT_SLOT_ORIGINS: Dict[int, Tuple[float, float]] = {
    1: (0.16, -0.10),
    2: (0.2925, -0.10),
    3: (0.425, -0.10),
    4: (0.16, -0.0095),
    5: (0.2925, -0.0095),
    6: (0.425, -0.0095),
    7: (0.16, 0.081),
    8: (0.2925, 0.081),
    9: (0.425, 0.081),
    10: (0.16, 0.1715),
    11: (0.2925, 0.1715),
}


def get_plate_geometry(load_name: str, *, origin_xy: Optional[Tuple[float, float]] = None) -> PlateGeometry:
    if load_name not in STEPBOT_LABWARE_CATALOG:
        raise KeyError(f"Unknown stepbot labware {load_name!r}")
    spec = STEPBOT_LABWARE_CATALOG[load_name]
    return spec.with_origin(origin_xy) if origin_xy is not None else spec


def parse_well_ref(ref: str) -> Tuple[Optional[str], str]:
    """Parse 'alias:A1' or bare 'A1' well reference."""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"Invalid well reference: {ref!r}")
    s = ref.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"Expected 'labware:well' format, got {ref!r}")
        return parts[0].strip(), parts[1].strip().upper()
    return None, s.upper()


@dataclass
class DeckLayout:
    """Maps stepbot labware aliases on numbered deck slots to workspace PlateGeometry instances."""

    labware_by_alias: Dict[str, PlateGeometry] = field(default_factory=dict)
    slot_origins: Dict[int, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_SLOT_ORIGINS))

    def add_labware(
        self,
        alias: str,
        load_name: str,
        *,
        slot: Optional[int] = None,
        origin_xy: Optional[Tuple[float, float]] = None,
    ) -> PlateGeometry:
        if not alias or not alias.strip():
            raise ValueError("Labware alias must be non-empty")
        if origin_xy is None:
            if slot is None or slot not in self.slot_origins:
                raise ValueError(f"Provide a valid deck slot (1..11) or explicit origin_xy, got slot={slot!r}")
            origin_xy = self.slot_origins[slot]
        geom = get_plate_geometry(load_name, origin_xy=origin_xy)
        self.labware_by_alias[alias] = geom
        return geom

    @property
    def max_rim_m(self) -> float:
        if not self.labware_by_alias:
            return 0.0
        return max(float(g.z_rim_m) for g in self.labware_by_alias.values())

    def resolve_well(self, ref: str) -> Tuple[PlateGeometry, str, np.ndarray]:
        """Resolve 'alias:well' or unambiguous bare 'well' to (PlateGeometry, well_name, xy_m)."""
        alias, well = parse_well_ref(ref)
        if alias is None:
            if len(self.labware_by_alias) != 1:
                raise ValueError(f"Ambiguous bare well {ref!r}; qualify as '<alias>:{well}'")
            geom = next(iter(self.labware_by_alias.values()))
        else:
            if alias not in self.labware_by_alias:
                raise KeyError(f"Unknown deck labware alias {alias!r}")
            geom = self.labware_by_alias[alias]
        return geom, well, geom.well_xy(well)

    def well_xy(self, ref: str) -> np.ndarray:
        _, _, xy = self.resolve_well(ref)
        return xy
