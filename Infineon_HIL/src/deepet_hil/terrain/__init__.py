"""Scenario 1 terrain synthetic-data pipeline."""

from .dataset import build_dataset, stratified_split_indices
from .generator import GeneratedWindow, generate_window
from .params import TerrainConfig, load_config

__all__ = [
    "GeneratedWindow",
    "TerrainConfig",
    "build_dataset",
    "generate_window",
    "load_config",
    "stratified_split_indices",
]

