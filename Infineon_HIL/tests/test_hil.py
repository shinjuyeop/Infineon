"""Signal playback, channel integrity, mock device, and E2E smoke tests."""

from __future__ import annotations

import numpy as np

from infineon_hil.device.mock_e84 import MockE84Inference
from infineon_hil.hil.mock import MockHilOutput
from infineon_hil.hil.player import PlaybackSample, SignalPlayer
from infineon_hil.hil.simulation import run_mock_sample
from infineon_hil.model.inference import TerrainInferenceEngine
from infineon_hil.model.network import TerrainCNN
from infineon_hil.schema import NUM_CHANNELS
from infineon_hil.terrain.mapping import (
    fsr_n_to_normalized,
    normalized_to_output_code,
    output_code_to_physical,
    vibration_g_to_normalized,
)


def _engine() -> TerrainInferenceEngine:
    return TerrainInferenceEngine(
        TerrainCNN(seed=1),
        np.zeros(NUM_CHANNELS, dtype=np.float32),
        np.ones(NUM_CHANNELS, dtype=np.float32),
        ("concrete", "marble", "ice", "sand"),
    )


def test_player_writes_50_ordered_lossless_frames() -> None:
    sample = np.arange(50 * NUM_CHANNELS, dtype=np.float32).reshape(50, NUM_CHANNELS)
    player = SignalPlayer(PlaybackSample(9, 2, sample))
    output = MockHilOutput()
    assert player.play(output) == 50
    assert len(output.frames) == 50
    assert [frame.timestep for frame in output.frames] == list(range(50))
    np.testing.assert_array_equal(output.reconstructed_sample(), sample)
    np.testing.assert_array_equal(output.frames[7].to_array(), sample[7])


def test_mock_e84_collects_exact_inference_shape() -> None:
    sample = np.zeros((50, NUM_CHANNELS), dtype=np.float32)
    output = MockHilOutput()
    SignalPlayer(PlaybackSample(0, 0, sample)).play(output)
    device = MockE84Inference(_engine())
    prediction = device.infer_frames(output.frames)
    assert prediction.probabilities.shape == (4,)
    assert len(device._frames) == 50


def test_end_to_end_mock_pipeline_smoke() -> None:
    sample = np.zeros((50, NUM_CHANNELS), dtype=np.float32)
    result = run_mock_sample(sample, expected_class_id=0, sample_id=123, engine=_engine())
    assert result.sample_id == 123
    assert result.frames_played == 50
    assert result.prediction.probabilities.shape == (4,)


def test_conceptual_physical_mapping_round_trip() -> None:
    np.testing.assert_allclose(fsr_n_to_normalized([0.0, 100.0, 200.0]), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(vibration_g_to_normalized([-5.0, 0.0, 5.0], full_scale_g=5.0), [0.0, 0.5, 1.0])
    codes = normalized_to_output_code([0.0, 0.5, 1.0], max_code=1000)
    np.testing.assert_allclose(
        output_code_to_physical(codes, max_code=1000, physical_min=0.0, physical_max=200.0),
        [0.0, 100.0, 200.0],
    )
