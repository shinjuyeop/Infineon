#!/usr/bin/env python3
"""Create waveform and canonical-feature inspection plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.terrain.features import FEATURE_NAMES, extract_features
from infineon_hil.terrain.params import load_config
from infineon_hil.schema import Channel


def _save_distribution(values: np.ndarray, y: np.ndarray, names: dict[int, str], title: str, unit: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot([values[y == class_id] for class_id in sorted(names)], tick_labels=[names[i] for i in sorted(names)])
    axis.set(title=title, ylabel=unit)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/terrain_v0_1.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    names = {terrain.class_id: terrain.name for terrain in config.terrains.values()}
    with np.load(args.dataset) as data:
        x, y = data["X"], data["y"]
    time_ms = np.arange(x.shape[1]) * 1000.0 / config.sampling_rate_hz

    fig, axes = plt.subplots(len(names), 1, figsize=(10, 10), sharex=True)
    for axis, class_id in zip(axes, sorted(names)):
        sample = x[np.flatnonzero(y == class_id)[0]]
        axis.plot(time_ms, sample[:, : Channel.VIBRATION])
        axis.set_ylabel(f"{names[class_id]}\nN")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle("Representative FSR waveforms (synthetic, not measured)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "representative_fsr.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(names), 1, figsize=(10, 9), sharex=True)
    for axis, class_id in zip(axes, sorted(names)):
        sample = x[np.flatnonzero(y == class_id)[0], :, Channel.VIBRATION]
        axis.plot(time_ms, sample)
        axis.set_ylabel(f"{names[class_id]}\ng")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle("Representative vibration waveforms (synthetic, not measured)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "representative_vibration.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0 / config.sampling_rate_hz)
    for class_id in sorted(names):
        spectra = np.abs(np.fft.rfft(x[y == class_id, :, Channel.VIBRATION], axis=1)) ** 2
        axis.semilogy(frequencies, spectra.mean(axis=0) + 1e-12, label=names[class_id])
    axis.set(title="Mean vibration PSD (synthetic, not measured)", xlabel="Frequency [Hz]", ylabel="Power [g²]")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "vibration_psd.png", dpi=150)
    plt.close(fig)

    features = extract_features(x, config.sampling_rate_hz)
    plot_info = [
        ("vibration_rms_distribution.png", "Vibration RMS distribution", "g"),
        ("spectral_centroid_distribution.png", "Spectral centroid distribution", "Hz"),
        ("peak_fsr_distribution.png", "Peak FSR distribution", "N"),
        ("max_load_drop_rate_distribution.png", "Maximum load-drop-rate distribution", "N/ms"),
    ]
    for index, (filename, title, unit) in enumerate(plot_info):
        _save_distribution(features[:, index], y, names, title, unit, args.output_dir / filename)
    print(f"Saved 7 inspection plots to {args.output_dir}")
    print(f"Canonical features: {', '.join(FEATURE_NAMES)}")


if __name__ == "__main__":
    main()
