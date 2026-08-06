"""Canonical sensor-channel schema shared by generation, AI, and HIL layers."""

from __future__ import annotations

from enum import IntEnum


class Channel(IntEnum):
    """Stable indices for the Scenario 1 sensor tensor."""

    FSR1 = 0
    FSR2 = 1
    FSR3 = 2
    FSR4 = 3
    VIBRATION = 4


CHANNEL_NAMES: tuple[str, ...] = ("FSR1", "FSR2", "FSR3", "FSR4", "vibration")
FSR_CHANNELS: tuple[Channel, ...] = (Channel.FSR1, Channel.FSR2, Channel.FSR3, Channel.FSR4)
NUM_CHANNELS = len(CHANNEL_NAMES)

