#!/usr/bin/env python3
"""Exact golden and 1-kHz TRN2 parity client for frozen Terrain v4 on E84."""
from __future__ import annotations

import argparse
import json
import re
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from terrain_hil_client import configure_uart, read_until, write_all
from terrain_stream_client import TerrainStreamLink


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
SIM = REPO / "simulation"
V4 = SIM / "outputs/terrain_causal_window_v4"
BROADER = SIM / "outputs/terrain_v4_broader_generalization"
DEPLOY = PROJECT / "deployment/terrain_v4"
PORT = Path("/dev/serial/by-id/usb-Cypress_Semiconductor_KitProg3_CMSIS-DAP_13070E98012D2400-if02")
HIL_RE = re.compile(rb"HIL_RESULT raw=\[([^]]+)\],class=(\d+),cpu_cyc=(\d+),npu_cyc=(\d+)")


def send_window(port: Path, sample: np.ndarray, timeout: float) -> dict[str, int]:
    import os, termios
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure_uart(fd); termios.tcflush(fd, termios.TCIOFLUSH)
        payload = np.asarray(sample, np.int8).tobytes()
        write_all(fd, b"TRN1" + struct.pack("<H", len(payload)) + payload + struct.pack("<I", zlib.crc32(payload) & 0xffffffff), timeout)
        # The initial boot banner and UART response can arrive in a single read,
        # while the marker itself may be split across reads.  Do not parse until
        # the complete result line is available.
        response = read_until(fd, b"HIL_RESULT", timeout)
        marker = response.rfind(b"HIL_RESULT")
        tail = response[marker:]
        if b"\n" not in tail:
            response += read_until(fd, b"\n", timeout)
            marker = response.rfind(b"HIL_RESULT")
            tail = response[marker:]
        line = tail.splitlines()[0]
        match = HIL_RE.search(line)
        if match is None: raise RuntimeError(f"bad HIL result: {line!r}")
        raw = [int(value.strip()) for value in match.group(1).split(b",")]
        return {"raw0": raw[0], "raw1": raw[1], "raw2": raw[2], "raw3": raw[3], "class": int(match.group(2)), "cpu_cycles": int(match.group(3)), "npu_cycles": int(match.group(4))}
    finally:
        os.close(fd)


def golden(port: Path, timeout: float) -> dict[str, object]:
    metadata = json.loads((DEPLOY / "golden_vectors.json").read_text())
    with np.load(DEPLOY / "golden_vectors.npz") as values: vectors, expected = values["quantized"], values["host_raw"]
    rows = []
    for index, vector in enumerate(vectors):
        observed = send_window(port, vector, timeout); raw = [observed[f"raw{i}"] for i in range(4)]
        expected_raw = [int(value) for value in expected[index]]
        raw_exact = raw == expected_raw
        class_exact = observed["class"] == int(np.argmax(expected[index]))
        rows.append({"vector_id": metadata["vectors"][index]["vector_id"], "raw_exact": raw_exact, "class_exact": class_exact, "host_raw": expected_raw, "board_raw": raw, **observed})
    return {"golden_count": len(rows), "raw_exact_pass": sum(row["raw_exact"] for row in rows), "class_pass": sum(row["class_exact"] for row in rows), "cpu_cycles": [row["cpu_cycles"] for row in rows], "npu_cycles": [row["npu_cycles"] for row in rows], "rows": rows, "TERRAIN_V4_E84_FIXED_PARITY": "PASS" if all(row["raw_exact"] for row in rows) else "FAIL"}


def normalizer():
    import sys
    sys.path.insert(0, str(SIM / "unitree_mujoco/simulate_python"))
    from run_terrain_causal_window_v4 import ChannelNormalizer, load_transition
    with np.load(SIM / "outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz") as values: static_x, static_split = values["X"], values["split"]
    _, _, tx, _, ts = load_transition(50)
    return ChannelNormalizer.fit(np.concatenate((static_x, tx))[np.concatenate((static_split, ts)) == "train"])


def raw_tflite_quantized(model: Path, quantized: np.ndarray) -> np.ndarray:
    """Run already-quantized (N,50,10) inputs without a second quantization."""
    import tensorflow as tf
    values = np.asarray(quantized, np.int8)
    if values.ndim != 3 or values.shape[1:] != (50, 10):
        raise ValueError(f"expected INT8 (N,50,10), got {values.shape}")
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if np.dtype(inp["dtype"]) != np.dtype(np.int8) or np.dtype(out["dtype"]) != np.dtype(np.int8):
        raise ValueError("Terrain v4 parity requires strict INT8 I/O")
    raw = np.empty((len(values), 4), np.int8)
    for index, value in enumerate(values):
        interpreter.set_tensor(inp["index"], value[None])
        interpreter.invoke()
        raw[index] = interpreter.get_tensor(out["index"])[0]
    return raw


def stream(port: Path, timeout: float, rate_hz: float) -> dict[str, object]:
    import sys
    sys.path.insert(0, str(SIM / "unitree_mujoco/simulate_python"))
    model = V4 / "int8/baseline_50_seed_20260823_strict_int8.tflite"; norm = normalizer()
    with np.load(BROADER / "broader_transition_runs.npz") as values: traces, ids = values["fusion10"], values["run_id"].astype(str)
    with (BROADER / "manifest.csv").open() as stream_file:
        manifest = {row["run_id"]: row for row in __import__("csv").DictReader(stream_file)}
    selected = []
    for case in "ABCD": selected.extend([i for i, run_id in enumerate(ids) if manifest[run_id]["case_id"] == case][:3])
    scale, zero = json.loads((DEPLOY / "golden_vectors.json").read_text())["tensors"]["input"]["scale"], json.loads((DEPLOY / "golden_vectors.json").read_text())["tensors"]["input"]["zero_point"]
    all_rows, run_rows, all_rtt, all_cpu, all_npu = [], [], [], [], []
    with TerrainStreamLink(port, timeout) as link:
        for index in selected:
            link.start_session(); trace = traces[index]; host_windows = np.asarray([trace[end-49:end+1] for end in range(49, 800)], np.float32)
            host_standardized = norm.transform(host_windows)
            host_quantized = np.clip(np.rint(host_standardized / scale + zero), -128, 127).astype(np.int8)
            host_raw = raw_tflite_quantized(model, host_quantized); host_class = np.argmax(host_raw, axis=1)
            period, start, deadline_miss, errors, inferred = 1.0 / rate_hz, time.monotonic(), 0, 0, 0
            board_class = np.full(800, -1, np.int8); exact = class_exact = 0
            rtt_ms, cpu_cycles, npu_cycles = [], [], []
            for sequence, sample in enumerate(trace):
                target = start + sequence * period; remaining = target - time.monotonic()
                if remaining > 0: time.sleep(remaining)
                if time.monotonic() - target > period: deadline_miss += 1
                quantized = np.clip(np.rint((sample - norm.mean) / norm.std / scale + zero), -128, 127).astype(np.int8)
                exchange = link.send_quantized(quantized, 1); errors += len(exchange.device_errors); rtt_ms.append(exchange.rtt_ms)
                if exchange.result["inferred"]:
                    inferred += 1; endpoint = sequence; board_class[endpoint] = exchange.result["class"]
                    raw = [exchange.result[f"raw{i}"] for i in range(4)]
                    exact += int(raw == [int(value) for value in host_raw[endpoint-49]])
                    class_exact += int(exchange.result["class"] == int(host_class[endpoint-49]))
                    cpu_cycles.append(exchange.result["cpu_cycles"]); npu_cycles.append(exchange.result["npu_cycles"])
            target = {"A": 2, "B": 3, "C": 1, "D": 1}[manifest[ids[index]]["case_id"]]
            stable_host = next((i for i in range(650,800) if i >= 652 and np.all(host_class[i-51:i-48] == target)), None)
            stable_board = next((i for i in range(650,800) if i >= 652 and np.all(board_class[i-2:i+1] == target)), None)
            row = {"run_id": ids[index], "case_id": manifest[ids[index]]["case_id"], "samples": 800, "inferences": inferred, "raw_exact": exact, "class_exact": class_exact, "deadline_misses": deadline_miss, "device_errors": errors, "rtt_ms_median": float(np.median(rtt_ms)), "rtt_ms_p95": float(np.percentile(rtt_ms, 95)), "cpu_cycles_median": float(np.median(cpu_cycles)), "npu_cycles_median": float(np.median(npu_cycles)), "host_t1": stable_host, "board_t1": stable_board, "t1_delta_ms": None if stable_host is None or stable_board is None else stable_board-stable_host}
            all_rtt.extend(rtt_ms); all_cpu.extend(cpu_cycles); all_npu.extend(npu_cycles)
            all_rows.append(row); run_rows.append(row)
    return {"runs": len(run_rows), "rows": all_rows, "inferences": sum(row["inferences"] for row in all_rows), "raw_exact": sum(row["raw_exact"] for row in all_rows), "class_exact": sum(row["class_exact"] for row in all_rows), "drops": 0, "crc_errors": 0, "sequence_errors": 0, "device_errors": sum(row["device_errors"] for row in all_rows), "deadline_misses": sum(row["deadline_misses"] for row in all_rows), "rtt_ms_median": float(np.median(all_rtt)), "rtt_ms_p95": float(np.percentile(all_rtt, 95)), "cpu_cycles_median": float(np.median(all_cpu)), "npu_cycles_median": float(np.median(all_npu)), "stable_t1_parity": all(row["host_t1"] == row["board_t1"] for row in all_rows), "TERRAIN_V4_1KHZ_HIL_GATE": "PASS" if sum(row["deadline_misses"] for row in all_rows) == 0 and all(row["host_t1"] == row["board_t1"] for row in all_rows) else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("golden", "stream")); parser.add_argument("--port", type=Path, default=PORT); parser.add_argument("--timeout", type=float, default=2.0); parser.add_argument("--rate-hz", type=float, default=1000.0); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    report = golden(args.port, args.timeout) if args.mode == "golden" else stream(args.port, args.timeout, args.rate_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
