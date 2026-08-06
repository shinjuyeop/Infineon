"""Host-only end-to-end mock HIL composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from deepet_hil.device.mock_e84 import MockE84Inference
from deepet_hil.model.inference import Prediction, TerrainInferenceEngine

from .mock import MockHilOutput
from .player import PlaybackSample, SignalPlayer


@dataclass(frozen=True)
class MockHilResult:
    sample_id: int
    expected_class_id: int
    prediction: Prediction
    frames_played: int

    @property
    def passed(self) -> bool:
        return self.expected_class_id == self.prediction.class_id


def run_mock_sample(
    sample: NDArray[np.floating],
    expected_class_id: int,
    sample_id: int,
    engine: TerrainInferenceEngine,
    *,
    playback_rate_hz: float = 1000.0,
) -> MockHilResult:
    """Play physical frames through the mock output and infer reconstructed frames."""

    player = SignalPlayer(
        PlaybackSample(sample_id, expected_class_id, np.asarray(sample, dtype=np.float32)),
        playback_rate_hz=playback_rate_hz,
    )
    output = MockHilOutput()
    count = player.play(output)
    device = MockE84Inference(engine)
    prediction = device.infer_frames(output.frames)
    return MockHilResult(sample_id, expected_class_id, prediction, count)

