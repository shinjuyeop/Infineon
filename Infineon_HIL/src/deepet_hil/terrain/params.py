"""Typed access to the synthetic terrain YAML source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from deepet_hil.schema import CHANNEL_NAMES


@dataclass(frozen=True)
class TerrainParameters:
    """One terrain's source-provided synthetic lookup parameters."""

    name: str
    class_id: int
    friction_mu: float
    relative_stiffness: float
    damping: float
    vibration_band_hz: tuple[float, float]
    vibration_rms_g: float
    impulsive: bool
    fsr_peak_n: tuple[float, float]
    rise_time_ms: tuple[float, float]
    micro_slip_probability: float


@dataclass(frozen=True)
class TerrainConfig:
    """Validated configuration loaded from YAML; raw holds replaceable assumptions."""

    path: Path
    raw: Mapping[str, Any]
    terrains: Mapping[str, TerrainParameters]

    @property
    def sampling_rate_hz(self) -> int:
        return int(self.raw["sampling"]["sampling_rate_hz"])

    @property
    def time_steps(self) -> int:
        return int(self.raw["sampling"]["time_steps"])


def _bounds(value: Mapping[str, Any]) -> tuple[float, float]:
    return float(value["min"]), float(value["max"])


def load_config(path: str | Path) -> TerrainConfig:
    """Load and minimally validate a terrain synthesis configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    status = str(raw.get("provenance", {}).get("status", "")).lower()
    if "not measured" not in status:
        raise ValueError("Config provenance must explicitly say parameters are not measured")
    sampling = raw["sampling"]
    rate = int(sampling["sampling_rate_hz"])
    steps = int(sampling["time_steps"])
    expected_steps = round(rate * float(sampling["window_length_ms"]) / 1000)
    if steps != expected_steps or tuple(sampling["channel_order"]) != CHANNEL_NAMES:
        raise ValueError("Sampling dimensions are inconsistent")

    terrains: dict[str, TerrainParameters] = {}
    ids: set[int] = set()
    for name, values in raw["terrains"].items():
        terrain = TerrainParameters(
            name=name,
            class_id=int(values["class_id"]),
            friction_mu=float(values["friction_mu"]),
            relative_stiffness=float(values["relative_stiffness"]),
            damping=float(values["damping"]),
            vibration_band_hz=_bounds(values["vibration_band_hz"]),
            vibration_rms_g=float(values["vibration_rms_g"]),
            impulsive=bool(values["impulsive"]),
            fsr_peak_n=_bounds(values["fsr_peak_n"]),
            rise_time_ms=_bounds(values["rise_time_ms"]),
            micro_slip_probability=float(values["micro_slip_probability"]),
        )
        if terrain.class_id in ids:
            raise ValueError(f"Duplicate class_id: {terrain.class_id}")
        if terrain.vibration_band_hz[1] > rate / 2:
            raise ValueError(f"{name} vibration band exceeds Nyquist")
        ids.add(terrain.class_id)
        terrains[name] = terrain
    if ids != set(range(len(terrains))):
        raise ValueError("class_id values must be contiguous from zero")
    return TerrainConfig(config_path, raw, terrains)
