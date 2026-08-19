#!/usr/bin/env python3
"""Generate guarded E84/U55 artifacts and selection-only goldens for frozen Terrain v4."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from generate_terrain_artifacts import write_model, write_regression


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
SIM = REPO / "simulation"
V4 = SIM / "outputs/terrain_causal_window_v4"
MODEL = V4 / "int8/baseline_50_seed_20260823_strict_int8.tflite"
MANIFEST = V4 / "int8/manifest.json"
OUT = PROJECT / "deployment/terrain_v4"
CPU_NAME = "TERRAIN_CAUSAL_WINDOW_V4_CPU"
U55_NAME = "TERRAIN_CAUSAL_WINDOW_V4_U55"
sys.path.insert(0, str(SIM / "unitree_mujoco/simulate_python"))
from run_terrain_causal_window_v4 import LABEL, ChannelNormalizer, load_transition, transition_split  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalizer() -> ChannelNormalizer:
    with np.load(SIM / "outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz") as values:
        static_x, static_split = values["X"], values["split"]
    _, _, transition_x, _, transition_values = load_transition(50)
    x = np.concatenate((static_x, transition_x))
    split = np.concatenate((static_split, transition_values))
    return ChannelNormalizer.fit(x[split == "train"])


def selection_vectors() -> tuple[np.ndarray, list[dict[str, object]]]:
    norm = normalizer()
    vectors, rows = [], []
    with np.load(SIM / "outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz") as values:
        x, y, split = values["X"], values["y"], values["split"]
    for label in range(4):
        index = int(np.flatnonzero((split == "architecture_selection") & (y == label))[len(np.flatnonzero((split == "architecture_selection") & (y == label))) // 2])
        vectors.append(x[index]); rows.append({"vector_id": f"static_{label}", "source": "v4_static_selection", "label": label, "class": next(name for name, value in LABEL.items() if value == label), "endpoint": None})
    transition_rows, traces, _, _, _ = load_transition(50)
    for case in "ABCD":
        index = next(i for i, row in enumerate(transition_rows) if transition_split(row) == "architecture_selection" and row["case_id"] == case)
        for endpoint in (649, 660):
            vectors.append(traces[index, endpoint - 49:endpoint + 1])
            terrain = transition_rows[index]["terrain_before"] if endpoint < 650 else transition_rows[index]["terrain_after"]
            rows.append({"vector_id": f"transition_{case}_{endpoint}", "source": "v4_transition_selection", "case": case, "label": LABEL[terrain], "class": terrain, "endpoint": endpoint})
    physical = np.asarray(vectors, np.float32)
    return physical, rows


def host(model: Path, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(model)); interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    quantized = np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
    outputs = []
    for item in quantized:
        interpreter.set_tensor(inp["index"], item[None]); interpreter.invoke(); outputs.append(interpreter.get_tensor(out["index"])[0])
    return quantized, np.asarray(outputs, np.int8), {"input": {"shape": [int(value) for value in inp["shape"]], "dtype": str(inp["dtype"].__name__), "scale": float(scale), "zero_point": int(zero)}, "output": {"shape": [int(value) for value in out["shape"]], "dtype": str(out["dtype"].__name__), "scale": float(out["quantization"][0]), "zero_point": int(out["quantization"][1])}}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--force", action="store_true"); parser.add_argument("--verify-only", action="store_true"); args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if sha(MODEL) != manifest["int8_sha256"] or manifest["candidate_specific_float_sha256"] != "c0a189cd4b01a3fb1474d89fedd8926c51f9dcc223ae0d849d7078adcafc3dc8": raise RuntimeError("frozen Terrain v4 provenance mismatch")
    physical, rows = selection_vectors(); standardized = normalizer().transform(physical); quantized, raw, tensors = host(MODEL, standardized)
    report = {"model_path": str(MODEL), "model_sha256": sha(MODEL), "float_sha256": manifest["candidate_specific_float_sha256"], "normalization": "reconstructed only from v4 train static+transition partition", "input_shape": [1, 50, 10], "class_mapping": LABEL, "sample_rate_hz": 1000, "window_ms": 50, "golden_count": len(rows), "vectors": [{**row, "input_sha256": hashlib.sha256(quantized[i].tobytes()).hexdigest(), "host_raw": raw[i].tolist(), "host_class": int(np.argmax(raw[i]))} for i, row in enumerate(rows)], "tensors": tensors}
    if args.verify_only: print(json.dumps(report, indent=2)); return
    if OUT.exists() and any(OUT.iterdir()) and not args.force: raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True); vela = OUT / "vela"; vela.mkdir(exist_ok=True)
    write_model(CPU_NAME, MODEL.read_bytes(), 8192, "frozen Terrain v4 raw strict-INT8")
    write_regression(CPU_NAME, quantized[0], raw[0], 0)
    result = subprocess.run(["vela", str(MODEL), "--output-dir", str(vela), "--accelerator-config", "ethos-u55-128", "--optimise", "Performance", "--memory-mode", "Sram_Only", "--arena-cache-size", "2936012", "--show-cpu-operations"], capture_output=True, text=True, check=True)
    (vela / "vela_stdout.txt").write_text(result.stdout + result.stderr)
    compiled = vela / f"{MODEL.stem}_vela.tflite"
    write_model(U55_NAME, compiled.read_bytes(), 4096, "frozen Terrain v4 Vela ethos-u55-128")
    write_regression(U55_NAME, quantized[0], raw[0], 0)
    report.update({"vela_model_path": str(compiled), "vela_model_sha256": sha(compiled), "source_size_bytes": MODEL.stat().st_size, "vela_size_bytes": compiled.stat().st_size, "macs": 58864, "accelerator_config": "ethos-u55-128", "vela_stdout": str(vela / "vela_stdout.txt")})
    np.savez_compressed(OUT / "golden_vectors.npz", physical=physical, standardized=standardized, quantized=quantized, host_raw=raw)
    (OUT / "golden_vectors.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "fixed_test_metadata.json").write_text(json.dumps({**report, "fixed_vector_id": rows[0]["vector_id"], "host_output_raw": raw[0].tolist(), "host_class": int(np.argmax(raw[0]))}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
