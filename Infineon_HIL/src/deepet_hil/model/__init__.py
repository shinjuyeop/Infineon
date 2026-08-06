"""Small host-side 1D CNN training and inference package."""

from .inference import Prediction, TerrainInferenceEngine
from .network import TerrainCNN

__all__ = ["Prediction", "TerrainCNN", "TerrainInferenceEngine"]

