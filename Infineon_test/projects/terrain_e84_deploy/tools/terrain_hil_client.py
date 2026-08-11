#!/usr/bin/env python3
"""Send one virtual FSR4+IMU6 window to the E84 terrain HIL UART."""

from __future__ import annotations

import argparse
import json
import os
import select
import struct
import termios
import time
import zlib
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
SIMULATION = REPO / "simulation"
DATASET = SIMULATION / "outputs/terrain_dataset_v1_expanded/dataset_noisy.npz"
NORMALIZATION = (
    SIMULATION
    / "outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/normalization.json"
)
INPUT_SCALE = 0.10959112644195557
INPUT_ZERO_POINT = 22


def configure_uart(fd: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
    attrs[3] = 0
    attrs[4] = termios.B1000000
    attrs[5] = termios.B1000000
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def quantize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (50, 10):
        raise ValueError(f"expected a (50,10) FSR4+IMU6 window, got {values.shape}")
    stats = json.loads(NORMALIZATION.read_text(encoding="utf-8"))["noisy/fusion"]
    normalized = (values - np.asarray(stats["mean"], np.float32)) / np.asarray(
        stats["std"], np.float32
    )
    return np.clip(
        np.rint(normalized / INPUT_SCALE + INPUT_ZERO_POINT), -128, 127
    ).astype(np.int8)


def load_input(args: argparse.Namespace) -> np.ndarray:
    if args.npy is not None:
        return quantize(np.load(args.npy, allow_pickle=False))
    with np.load(DATASET, allow_pickle=False) as dataset:
        values = np.asarray(dataset["X"][args.sample_index], np.float32)
        print(
            f"sample={args.sample_index} split={dataset['split'][args.sample_index]} "
            f"label={int(dataset['y'][args.sample_index])} "
            f"family={dataset['surface_family'][args.sample_index]}"
        )
    return quantize(values)


def read_until(fd: int, needle: bytes, timeout: float) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.2, deadline - time.monotonic()))
        if ready:
            chunk = os.read(fd, 4096)
            data.extend(chunk)
            if needle in data:
                return bytes(data)
    raise TimeoutError(f"UART timeout waiting for {needle!r}; received {bytes(data)!r}")


def write_all(fd: int, data: bytes, timeout: float) -> None:
    offset = 0
    deadline = time.monotonic() + timeout
    while offset < len(data):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"UART timeout after sending {offset}/{len(data)} bytes")
        _, writable, _ = select.select([], [fd], [], min(0.2, remaining))
        if writable:
            offset += os.write(fd, data[offset:])
    termios.tcdrain(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=Path,
        default=Path(
            "/dev/serial/by-id/usb-Cypress_Semiconductor_KitProg3_CMSIS-DAP_13070E98012D2400-if02"
        ),
    )
    parser.add_argument("--sample-index", type=int, default=878)
    parser.add_argument("--npy", type=Path, help="physical-unit float32 (50,10) MuJoCo window")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    quantized = load_input(args)
    payload = quantized.tobytes()
    frame = b"TRN1" + struct.pack("<H", len(payload)) + payload
    frame += struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure_uart(fd)
        termios.tcflush(fd, termios.TCIOFLUSH)
        # The board may already be waiting; the client does not require a reset.
        write_all(fd, frame, args.timeout)
        response = read_until(fd, b"HIL_RESULT", args.timeout)
        response += read_until(fd, b"\n", 1.0) if not response.endswith(b"\n") else b""
        marker = response.rfind(b"HIL_RESULT")
        result_line = response[marker:].splitlines()[0]
        print(result_line.decode("ascii", errors="replace"))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
