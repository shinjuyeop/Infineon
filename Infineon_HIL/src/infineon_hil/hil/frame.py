"""One canonical five-channel HIL playback frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from infineon_hil.schema import Channel, NUM_CHANNELS


@dataclass(frozen=True)
class HilFrame:
    """Physical-unit frame independent of voltage, DAC, and transport details."""

    timestep: int
    timestamp_s: float
    fsr1_n: float
    fsr2_n: float
    fsr3_n: float
    fsr4_n: float
    vibration_g: float

    @classmethod
    def from_values(cls, timestep: int, timestamp_s: float, values: ArrayLike) -> "HilFrame":
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (NUM_CHANNELS,):
            raise ValueError(f"Frame values must have shape ({NUM_CHANNELS},)")
        if timestep < 0 or timestamp_s < 0:
            raise ValueError("Frame time values must be non-negative")
        return cls(
            timestep=int(timestep),
            timestamp_s=float(timestamp_s),
            fsr1_n=float(array[Channel.FSR1]),
            fsr2_n=float(array[Channel.FSR2]),
            fsr3_n=float(array[Channel.FSR3]),
            fsr4_n=float(array[Channel.FSR4]),
            vibration_g=float(array[Channel.VIBRATION]),
        )

    def to_array(self) -> NDArray[np.float32]:
        """Return values in the one canonical channel order."""

        return np.asarray(
            [self.fsr1_n, self.fsr2_n, self.fsr3_n, self.fsr4_n, self.vibration_g],
            dtype=np.float32,
        )

