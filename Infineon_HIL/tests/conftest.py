"""Test path and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.terrain.dataset import build_dataset
from infineon_hil.terrain.params import load_config


@pytest.fixture(scope="session")
def config():
    return load_config(PROJECT_ROOT / "configs/terrain_v0_1.yaml")


@pytest.fixture(scope="session")
def full_dataset(config):
    return build_dataset(config, samples_per_class=500, seed=42)

