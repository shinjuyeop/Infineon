#!/usr/bin/env python3
"""Verify the fixed-regression E84 boot log against generated Host golden data."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import termios
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = Path(
    "/dev/serial/by-id/"
    "usb-Cypress_Semiconductor_KitProg3_CMSIS-DAP_13070E98012D2400-if02"
)
METADATA = {
    "100hz": PROJECT / "deployment/fixed_test_metadata.json",
    "fast1000": PROJECT / "deployment/fast1000/fixed_test_metadata.json",
}
RESULT_RE = re.compile(
    rb"TERRAIN_RESULT, output\[0\]_raw=\[([^]]+)\], "
    rb"host_raw=\[([^]]+)\], device_class=(\d+), host_class=(\d+)"
)


def configure_uart(fd: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def parse_raw(values: bytes) -> list[int]:
    return [int(value.strip()) for value in values.split(b",")]


def verify_log(payload: bytes, metadata: dict[str, object]) -> dict[str, object]:
    matches = list(RESULT_RE.finditer(payload))
    if not matches:
        raise RuntimeError("boot log contains no TERRAIN_RESULT line")
    match = matches[-1]
    device_raw = parse_raw(match.group(1))
    embedded_host_raw = parse_raw(match.group(2))
    device_class = int(match.group(3))
    embedded_host_class = int(match.group(4))
    expected_raw = [int(value) for value in metadata["host_output_raw"]]
    expected_class = int(metadata["host_class"])
    if b"Output[0] : PASS" not in payload:
        raise RuntimeError("boot log does not contain the fixed-regression PASS marker")
    if device_raw != expected_raw or embedded_host_raw != expected_raw:
        raise RuntimeError(
            f"raw mismatch: device={device_raw}, embedded={embedded_host_raw}, "
            f"expected={expected_raw}"
        )
    if device_class != expected_class or embedded_host_class != expected_class:
        raise RuntimeError(
            f"class mismatch: device={device_class}, embedded={embedded_host_class}, "
            f"expected={expected_class}"
        )
    return {
        "passed": True,
        "device_raw": device_raw,
        "device_class": device_class,
        "expected_class": metadata["expected_class"],
    }


def capture(port: Path, timeout: float) -> bytes:
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    data = bytearray()
    try:
        configure_uart(fd)
        termios.tcflush(fd, termios.TCIOFLUSH)
        print("UART ready; press the KIT_PSE84_AI RESET button now.", flush=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], min(0.2, deadline - time.monotonic()))
            if ready:
                data.extend(os.read(fd, 4096))
                if b"Output[0] : PASS" in data:
                    return bytes(data)
    finally:
        os.close(fd)
    raise TimeoutError(f"fixed-test UART timeout; received={bytes(data)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(METADATA), default="fast1000")
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--log",
        type=Path,
        help="verify an existing boot-log file instead of opening the board UART",
    )
    args = parser.parse_args()
    metadata_path = METADATA[args.profile]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = args.log.read_bytes() if args.log is not None else capture(args.port, args.timeout)
    print(payload.decode("ascii", errors="replace"))
    print(json.dumps(verify_log(payload, metadata), indent=2))


if __name__ == "__main__":
    main()
