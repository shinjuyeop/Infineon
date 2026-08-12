#!/usr/bin/env python3
"""Send one virtual FSR4+IMU6 window to the E84 terrain HIL UART."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import struct
import termios
import time
import zlib
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
FIXED_METADATA = {
    "100hz": PROJECT / "deployment/fixed_test_metadata.json",
    "fast1000": PROJECT / "deployment/fast1000/fixed_test_metadata.json",
}
HIL_RESULT_RE = re.compile(rb"HIL_RESULT raw=\[([^]]+)\],class=(\d+)")


try:
    from terrain_preprocessing import (
        PROFILE_PATHS,
        paths_for_profile,
        preprocessor_for_profile,
    )
except ModuleNotFoundError:
    from tools.terrain_preprocessing import (
        PROFILE_PATHS,
        paths_for_profile,
        preprocessor_for_profile,
    )

# Backward-compatible canonical alias used by terrain_stream_client.py.
_, _, DATASET = paths_for_profile("100hz")


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


def quantize(values: np.ndarray, profile: str = "100hz") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (50, 10):
        raise ValueError(f"expected a (50,10) FSR4+IMU6 window, got {values.shape}")
    return preprocessor_for_profile(profile).quantize(values)


def load_input(args: argparse.Namespace) -> np.ndarray:
    if args.npy is not None:
        return quantize(np.load(args.npy, allow_pickle=False), args.profile)
    _, _, dataset_path = paths_for_profile(args.profile)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        values = np.asarray(dataset["X"][args.sample_index], np.float32)
        print(
            f"sample={args.sample_index} split={dataset['split'][args.sample_index]} "
            f"label={int(dataset['y'][args.sample_index])} "
            f"family={dataset['surface_family'][args.sample_index]}"
        )
    return quantize(values, args.profile)


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


def verify_fixed_golden(result_line: bytes, profile: str) -> dict[str, object]:
    metadata = json.loads(FIXED_METADATA[profile].read_text(encoding="utf-8"))
    match = HIL_RESULT_RE.search(result_line)
    if match is None:
        raise RuntimeError(f"cannot parse HIL_RESULT: {result_line!r}")
    raw = [int(value.strip()) for value in match.group(1).split(b",")]
    predicted_class = int(match.group(2))
    expected_raw = [int(value) for value in metadata["host_output_raw"]]
    expected_class = int(metadata["host_class"])
    if raw != expected_raw or predicted_class != expected_class:
        raise RuntimeError(
            f"Host golden mismatch: raw={raw}, class={predicted_class}, "
            f"expected_raw={expected_raw}, expected_class={expected_class}"
        )
    return {"passed": True, "raw": raw, "class": predicted_class}


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
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_PATHS), default="100hz"
    )
    parser.add_argument("--npy", type=Path, help="physical-unit float32 (50,10) MuJoCo window")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--expect-fixed-golden",
        action="store_true",
        help="require exact raw/class parity with the generated fixed metadata",
    )
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
        if args.expect_fixed_golden:
            if args.npy is not None or args.sample_index != 878:
                raise ValueError(
                    "--expect-fixed-golden requires the profile's fixed sample index 878"
                )
            print(json.dumps(verify_fixed_golden(result_line, args.profile), indent=2))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
