"""In-memory HIL output used for tests and host-only integration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .frame import HilFrame
from .interface import HilOutput


class MockHilOutput(HilOutput):
    """Store every frame losslessly; optionally print human-readable values."""

    def __init__(self, *, log_frames: bool = False) -> None:
        self.log_frames = log_frames
        self.frames: list[HilFrame] = []
        self.started = False

    def start(self) -> None:
        if self.started:
            raise RuntimeError("HIL output is already started")
        self.frames.clear()
        self.started = True

    def write(self, frame: HilFrame) -> None:
        if not self.started:
            raise RuntimeError("start() must be called before write()")
        if self.frames and frame.timestep != self.frames[-1].timestep + 1:
            raise ValueError("Frames must be contiguous and ordered")
        self.frames.append(frame)
        if self.log_frames:
            print(f"t={frame.timestep:02d} time={frame.timestamp_s:.6f}s values={frame.to_array()}")

    def stop(self) -> None:
        if not self.started:
            raise RuntimeError("HIL output is not started")
        self.started = False

    def reconstructed_sample(self) -> NDArray[np.float32]:
        """Reconstruct exactly what the backend received, in channel order."""

        if self.started:
            raise RuntimeError("stop() must complete before reconstruction")
        if not self.frames:
            raise ValueError("No frames have been captured")
        return np.stack([frame.to_array() for frame in self.frames]).astype(np.float32)

