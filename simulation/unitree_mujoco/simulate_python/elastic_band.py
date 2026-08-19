"""SDK-independent elastic support force shared by offline MuJoCo datasets."""
from __future__ import annotations

import numpy as np


class ElasticBand:
    """Exact offline force law formerly imported through the SDK bridge."""

    def __init__(self) -> None:
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        delta = self.point - x
        distance = np.linalg.norm(delta)
        direction = delta / distance
        velocity = np.dot(dx, direction)
        return (self.stiffness * (distance - self.length) - self.damping * velocity) * direction
