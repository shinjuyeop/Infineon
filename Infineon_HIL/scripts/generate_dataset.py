#!/usr/bin/env python3
"""Generate the v0.1 engineering-estimate terrain dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepet_hil.terrain.dataset import build_dataset, save_dataset
from deepet_hil.terrain.params import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/terrain_v0_1.yaml")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/synthetic/terrain_v0_1.npz")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/synthetic/terrain_v0_1_metadata.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    x, y, metadata = build_dataset(config, args.samples_per_class, args.seed)
    save_dataset(x, y, metadata, args.output, args.metadata)
    counts = metadata.groupby("terrain_name").size().to_dict()
    print(f"Saved {args.output}: X={x.shape} {x.dtype}, y={y.shape}")
    print(f"Saved {args.metadata}: class_counts={counts}")
    print("WARNING: v0.1 is synthetic engineering-estimate data, not measured sensor data.")


if __name__ == "__main__":
    main()

