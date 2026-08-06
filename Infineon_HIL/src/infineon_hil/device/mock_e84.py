"""Host model adapter standing in for future PSoC Edge E84 inference."""

from __future__ import annotations

import numpy as np

from infineon_hil.hil.frame import HilFrame
from infineon_hil.model.inference import Prediction, TerrainInferenceEngine


class MockE84Inference:
    """Collect chronological frames, reconstruct `(50,5)`, then infer on host."""

    def __init__(self, engine: TerrainInferenceEngine) -> None:
        self.engine = engine
        self._frames: list[HilFrame] = []

    @classmethod
    def from_model(cls, model_path: str) -> "MockE84Inference":
        return cls(TerrainInferenceEngine.load(model_path))

    def reset(self) -> None:
        self._frames.clear()

    def accept_frame(self, frame: HilFrame) -> Prediction | None:
        expected_timestep = len(self._frames)
        if frame.timestep != expected_timestep:
            raise ValueError(f"Expected timestep {expected_timestep}, got {frame.timestep}")
        if expected_timestep >= self.engine.model.input_steps:
            raise ValueError("Inference window is already full")
        self._frames.append(frame)
        if len(self._frames) == self.engine.model.input_steps:
            return self.infer_collected()
        return None

    def infer_collected(self) -> Prediction:
        if len(self._frames) != self.engine.model.input_steps:
            raise ValueError(
                f"Expected {self.engine.model.input_steps} frames, got {len(self._frames)}"
            )
        sample = np.stack([frame.to_array() for frame in self._frames]).astype(np.float32)
        return self.engine.predict(sample)

    def infer_frames(self, frames: list[HilFrame]) -> Prediction:
        self.reset()
        result: Prediction | None = None
        for frame in frames:
            result = self.accept_frame(frame)
        if result is None:
            raise ValueError("Not enough frames for inference")
        return result

