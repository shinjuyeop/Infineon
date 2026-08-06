"""Dataset sample selection and transport-neutral timestep playback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from deepet_hil.model.dataset import load_npz_dataset
from deepet_hil.schema import NUM_CHANNELS

from .frame import HilFrame
from .interface import HilOutput


@dataclass(frozen=True)
class PlaybackSample:
    sample_id: int
    expected_class_id: int
    values: NDArray[np.float32]


class SignalPlayer:
    """Iterate one raw physical-unit sample and call a replaceable output backend."""

    def __init__(self, sample: PlaybackSample, *, playback_rate_hz: float = 1000.0) -> None:
        values = np.asarray(sample.values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Sample must have shape (time, {NUM_CHANNELS})")
        if playback_rate_hz <= 0:
            raise ValueError("playback_rate_hz must be positive")
        self.sample = PlaybackSample(sample.sample_id, sample.expected_class_id, values)
        self.playback_rate_hz = float(playback_rate_hz)

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        sample_index: int,
        *,
        playback_rate_hz: float = 1000.0,
    ) -> "SignalPlayer":
        x, y = load_npz_dataset(dataset_path)
        if not 0 <= sample_index < len(x):
            raise IndexError(f"sample_index must be in [0, {len(x) - 1}]")
        return cls(
            PlaybackSample(sample_index, int(y[sample_index]), x[sample_index]),
            playback_rate_hz=playback_rate_hz,
        )

    def frames(self):
        """Yield frames without side effects, useful for other transports."""

        for timestep, values in enumerate(self.sample.values):
            yield HilFrame.from_values(
                timestep=timestep,
                timestamp_s=timestep / self.playback_rate_hz,
                values=values,
            )

    def play(self, output: HilOutput, *, realtime: bool = False) -> int:
        """Play all frames and return their count; realtime sleeping is optional."""

        interval = 1.0 / self.playback_rate_hz
        output.start()
        count = 0
        try:
            for frame in self.frames():
                output.write(frame)
                count += 1
                if realtime and count < len(self.sample.values):
                    time.sleep(interval)
        finally:
            output.stop()
        return count

