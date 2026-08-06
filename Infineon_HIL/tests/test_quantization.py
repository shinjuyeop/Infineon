"""Strict INT8 conversion smoke test for the optional deployment dependency."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from infineon_hil.deployment.model import build_deployment_model
from infineon_hil.deployment.quantization import export_full_int8_tflite

pytestmark = pytest.mark.deployment


def test_full_int8_conversion_has_expected_interface(tmp_path) -> None:
    model = build_deployment_model(seed=3)
    representative = np.random.default_rng(3).normal(size=(12, 50, 5)).astype(np.float32)
    output = tmp_path / "smoke_int8.tflite"
    input_spec, output_spec = export_full_int8_tflite(model, representative, output)
    assert output.is_file() and output.stat().st_size > 0
    assert input_spec.shape == (1, 50, 5)
    assert input_spec.dtype == "int8"
    assert input_spec.scale > 0.0
    assert output_spec.shape == (1, 4)
    assert output_spec.dtype == "int8"
    assert output_spec.scale > 0.0
