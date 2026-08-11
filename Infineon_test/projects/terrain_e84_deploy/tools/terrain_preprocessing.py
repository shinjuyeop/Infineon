#!/usr/bin/env python3
"""Canonical physical-unit normalization and INT8 quantization for terrain HIL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
SIMULATION = REPO / "simulation"
DEPLOYMENT_METADATA = (
    SIMULATION
    / "outputs/terrain_dataset_v1_expanded_int8_seed_20260809/deployment_metadata.json"
)
MODEL = (
    SIMULATION
    / "outputs/terrain_dataset_v1_expanded_int8_seed_20260809/noisy_fusion_int8.tflite"
)
DATASET = SIMULATION / "outputs/terrain_dataset_v1_expanded/dataset_noisy.npz"


class TerrainPreprocessor:
    """Apply the immutable model metadata to one sample or a 50-sample window."""

    def __init__(self, metadata_path: Path = DEPLOYMENT_METADATA) -> None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.channel_order = tuple(str(value) for value in metadata["channel_order"])
        if len(self.channel_order) != 10:
            raise ValueError("terrain deployment metadata must contain 10 channels")
        self.mean = np.asarray(metadata["normalization"]["mean"], np.float32)
        self.std = np.asarray(metadata["normalization"]["std"], np.float32)
        model_input = metadata["tflite"]["input"]
        self.input_scale = float(model_input["scale"])
        self.input_zero_point = int(model_input["zero_point"])
        if self.mean.shape != (10,) or self.std.shape != (10,):
            raise ValueError("terrain normalization must contain 10 mean/std values")
        if np.any(self.std <= 0.0) or self.input_scale <= 0.0:
            raise ValueError("invalid terrain normalization or quantization metadata")

    def quantize(self, values: np.ndarray) -> np.ndarray:
        physical = np.asarray(values, dtype=np.float32)
        if physical.shape[-1:] != (10,) or physical.ndim not in (1, 2):
            raise ValueError(
                f"expected one (10,) sample or a (T,10) window, got {physical.shape}"
            )
        if not np.all(np.isfinite(physical)):
            raise ValueError("terrain physical input contains NaN/Inf")
        normalized = (physical - self.mean) / self.std
        transformed = np.rint(
            normalized / self.input_scale + self.input_zero_point
        )
        return np.clip(transformed, -128, 127).astype(np.int8)


def verify_canonical_model(
    model_path: Path = MODEL,
    metadata_path: Path = DEPLOYMENT_METADATA,
) -> dict[str, object]:
    """Reject a missing or substituted model before Host shadow inference."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = str(metadata["tflite"]["sha256"])
    expected_size = int(metadata["tflite"]["size_bytes"])
    payload = model_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or actual_hash != expected_hash:
        raise ValueError(
            "canonical model identity mismatch: "
            f"size={len(payload)} sha256={actual_hash}"
        )
    return {
        "path": str(model_path.resolve()),
        "size_bytes": len(payload),
        "sha256": actual_hash,
    }


DEFAULT_PREPROCESSOR = TerrainPreprocessor()


def quantize_physical(values: np.ndarray) -> np.ndarray:
    return DEFAULT_PREPROCESSOR.quantize(values)
