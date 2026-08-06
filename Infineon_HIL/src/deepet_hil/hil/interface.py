"""Replaceable HIL output lifecycle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .frame import HilFrame


class HilOutput(ABC):
    """Transport-neutral sink implemented now by mock and later by hardware."""

    @abstractmethod
    def start(self) -> None:
        """Prepare one playback transaction."""

    @abstractmethod
    def write(self, frame: HilFrame) -> None:
        """Accept one physical-unit frame in chronological order."""

    @abstractmethod
    def stop(self) -> None:
        """Finish the transaction and release output resources."""

