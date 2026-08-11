#!/usr/bin/env python3
"""Replay FSR4+IMU6 samples to the E84 TRN2 streaming HIL endpoint."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import select
import struct
import termios
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from terrain_hil_client import DATASET, configure_uart, quantize, write_all
    from terrain_preprocessing import quantize_physical
except ModuleNotFoundError:
    from tools.terrain_hil_client import DATASET, configure_uart, quantize, write_all
    from tools.terrain_preprocessing import quantize_physical


DEFAULT_PORT = Path(
    "/dev/serial/by-id/"
    "usb-Cypress_Semiconductor_KitProg3_CMSIS-DAP_13070E98012D2400-if02"
)
RESULT_RE = re.compile(
    rb"STREAM_RESULT seq=(\d+),fill=(\d+),warmup=([01]),inferred=([01]),"
    rb"class=(-?\d+),raw=\[(-?\d+),(-?\d+),(-?\d+),(-?\d+)\],"
    rb"cpu_cyc=(\d+),npu_cyc=(\d+)"
)


class LineReader:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def read_line(self, fd: int, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[: newline + 1])
                del self.buffer[: newline + 1]
                return line
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([fd], [], [], min(0.05, remaining))
            if ready:
                self.buffer.extend(os.read(fd, 4096))
        raise TimeoutError(f"UART line timeout; buffered={bytes(self.buffer)!r}")


@dataclass(frozen=True)
class StreamExchange:
    result: dict[str, int]
    device_errors: tuple[str, ...]
    send_time: float
    receive_time: float

    @property
    def rtt_ms(self) -> float:
        return (self.receive_time - self.send_time) * 1000.0


class TerrainStreamLink:
    """Reusable synchronous TRN2 link with explicit stream-session boundaries."""

    def __init__(self, port: Path = DEFAULT_PORT, timeout: float = 1.0) -> None:
        self.port = Path(port)
        self.timeout = timeout
        self.fd: int | None = None
        self.reader = LineReader()
        self.next_sequence = 0

    def open(self) -> "TerrainStreamLink":
        if self.fd is not None:
            return self
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            configure_uart(self.fd)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        self.reader = LineReader()
        self.next_sequence = 0
        return self

    def start_session(self) -> None:
        """Make the next accepted sample sequence zero, resetting the E84 ring."""
        self.next_sequence = 0

    def send_quantized(self, sample: np.ndarray, stride: int = 1) -> StreamExchange:
        if self.fd is None:
            raise RuntimeError("TerrainStreamLink is not open")
        if not 1 <= stride <= 65535:
            raise ValueError("stride must be in [1,65535]")
        sequence = self.next_sequence
        send_time = time.monotonic()
        write_all(self.fd, build_frame(sequence, stride, sample), self.timeout)
        result, errors = read_result(
            self.fd, self.reader, sequence, self.timeout
        )
        receive_time = time.monotonic()
        self.next_sequence += 1
        return StreamExchange(result, tuple(errors), send_time, receive_time)

    def send_physical(self, sample: np.ndarray, stride: int = 1) -> StreamExchange:
        return self.send_quantized(quantize_physical(sample), stride)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "TerrainStreamLink":
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()


def load_window(args: argparse.Namespace) -> tuple[np.ndarray, int | None]:
    if args.npy is not None:
        values = np.load(args.npy, allow_pickle=False)
        return quantize(values), None
    with np.load(DATASET, allow_pickle=False) as dataset:
        values = np.asarray(dataset["X"][args.sample_index], np.float32)
        label = int(dataset["y"][args.sample_index])
        print(
            f"source_sample={args.sample_index} "
            f"split={dataset['split'][args.sample_index]} label={label} "
            f"family={dataset['surface_family'][args.sample_index]}"
        )
    return quantize(values), label


def build_frame(sequence: int, stride: int, sample: np.ndarray) -> bytes:
    sample = np.asarray(sample, dtype=np.int8)
    if sample.shape != (10,):
        raise ValueError(f"expected one (10,) sample, got {sample.shape}")
    payload = struct.pack("<IH", sequence & 0xFFFFFFFF, stride) + sample.tobytes()
    return (
        b"TRN2"
        + struct.pack("<H", len(payload))
        + payload
        + struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def read_result(
    fd: int, reader: LineReader, expected_sequence: int, timeout: float
) -> tuple[dict[str, int], list[str]]:
    errors: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = reader.read_line(fd, max(0.001, deadline - time.monotonic()))
        except TimeoutError as exc:
            if errors:
                raise RuntimeError(
                    f"device rejected sequence {expected_sequence}: {errors}"
                ) from exc
            raise
        if b"STREAM_ERROR" in line:
            errors.append(line[line.find(b"STREAM_ERROR") :].decode("ascii", "replace").strip())
            continue
        marker = line.find(b"STREAM_RESULT")
        if marker < 0:
            continue
        match = RESULT_RE.search(line[marker:])
        if match is None:
            raise RuntimeError(f"malformed STREAM_RESULT: {line!r}")
        values = [int(value) for value in match.groups()]
        result = {
            "sequence": values[0],
            "fill": values[1],
            "warmup": values[2],
            "inferred": values[3],
            "class": values[4],
            "raw0": values[5],
            "raw1": values[6],
            "raw2": values[7],
            "raw3": values[8],
            "cpu_cycles": values[9],
            "npu_cycles": values[10],
        }
        if result["sequence"] != expected_sequence:
            raise RuntimeError(
                f"sequence mismatch: sent {expected_sequence}, received {result['sequence']}"
            )
        return result, errors
    raise TimeoutError(f"UART timeout waiting for sequence {expected_sequence}")


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    if array.size == 0:
        return {name: float("nan") for name in ("mean", "std", "min", "max", "p95")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


def print_stats(name: str, unit: str, values: list[float]) -> None:
    stats = describe(values)
    print(
        f"{name}_{unit}: mean={stats['mean']:.3f} std={stats['std']:.3f} "
        f"min={stats['min']:.3f} max={stats['max']:.3f} p95={stats['p95']:.3f}"
    )


def write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT)
    parser.add_argument("--sample-index", type=int, default=878)
    parser.add_argument("--npy", type=Path, help="physical-unit float32 (50,10) window")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--no-realtime", action="store_true", help="send immediately instead of pacing"
    )
    args = parser.parse_args()
    if args.samples < 50:
        parser.error("--samples must be at least 50 to complete warm-up")
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    if not 1 <= args.stride <= 65535:
        parser.error("--stride must be in [1,65535]")

    window, label = load_window(args)
    repeats = math.ceil(args.samples / window.shape[0])
    samples = np.tile(window, (repeats, 1))[: args.samples]
    period = 1.0 / args.rate_hz
    rows: list[dict[str, int | float]] = []
    errors: list[str] = []
    send_times: list[float] = []
    rtt_ms: list[float] = []
    inferred_rtt_ms: list[float] = []
    deadline_misses = 0

    with TerrainStreamLink(args.port, args.timeout) as link:
        start = time.monotonic()
        for sequence, sample in enumerate(samples):
            target = start + sequence * period
            if not args.no_realtime:
                remaining = target - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            send_time = time.monotonic()
            if not args.no_realtime and (send_time - target) > period:
                deadline_misses += 1
            send_times.append(send_time)
            exchange = link.send_quantized(sample, args.stride)
            result = exchange.result
            receive_time = exchange.receive_time
            errors.extend(exchange.device_errors)
            expected_fill = min(sequence + 1, 50)
            expected_warmup = 1 if sequence < 49 else 0
            expected_inferred = (
                1 if sequence >= 49 and ((sequence - 49) % args.stride == 0) else 0
            )
            if (
                result["fill"] != expected_fill
                or result["warmup"] != expected_warmup
                or result["inferred"] != expected_inferred
            ):
                raise RuntimeError(
                    "ring cadence mismatch at sequence "
                    f"{sequence}: fill={result['fill']} warmup={result['warmup']} "
                    f"inferred={result['inferred']}"
                )
            latency_ms = (receive_time - send_time) * 1000.0
            rtt_ms.append(latency_ms)
            if result["inferred"]:
                inferred_rtt_ms.append(latency_ms)
            result.update(
                {
                    "send_elapsed_s": send_time - start,
                    "rtt_ms": latency_ms,
                    "send_lateness_ms": (send_time - target) * 1000.0,
                }
            )
            rows.append(result)
    inferred = [row for row in rows if row["inferred"] == 1]
    expected_inferences = 1 + (args.samples - 50) // args.stride
    if len(inferred) != expected_inferences:
        raise RuntimeError(
            f"inference count mismatch: expected {expected_inferences}, got {len(inferred)}"
        )
    first = inferred[0]
    first_raw = [first[f"raw{index}"] for index in range(4)]
    if args.npy is None and args.sample_index == 878:
        expected_raw = [35, -35, -128, -128]
        if first_raw != expected_raw or first["class"] != 0:
            raise RuntimeError(
                f"first-window parity failed: raw={first_raw}, class={first['class']}"
            )

    intervals_ms = [
        (send_times[index] - send_times[index - 1]) * 1000.0
        for index in range(1, len(send_times))
    ]
    print(
        f"stream_pass samples={len(rows)} warmup_samples=50 inferences={len(inferred)} "
        f"stride={args.stride} label={label} first_raw={first_raw} "
        f"first_class={first['class']} errors={len(errors)} "
        f"deadline_misses={deadline_misses}"
    )
    print_stats("send_period", "ms", intervals_ms)
    print_stats("round_trip", "ms", rtt_ms)
    print_stats("inferred_round_trip", "ms", inferred_rtt_ms)
    print_stats("cpu", "cycles", [float(row["cpu_cycles"]) for row in inferred])
    print_stats("npu", "cycles", [float(row["npu_cycles"]) for row in inferred])
    if errors:
        print("device_errors:")
        for error in errors:
            print(f"  {error}")
    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
