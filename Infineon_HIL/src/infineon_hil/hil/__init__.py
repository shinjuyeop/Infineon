"""Hardware-independent signal playback interfaces."""

from .frame import HilFrame
from .interface import HilOutput
from .mock import MockHilOutput
from .player import PlaybackSample, SignalPlayer

__all__ = ["HilFrame", "HilOutput", "MockHilOutput", "PlaybackSample", "SignalPlayer"]

