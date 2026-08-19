#!/usr/bin/env python3
"""Capture frozen Terrain v4 CPU/U55 golden repeatability, then freeze policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from terrain_v4_e84_client import DEPLOY, PORT, send_window


def capture(runtime: str, port: Path, repeats: int, timeout: float) -> dict[str, object]:
    meta = json.loads((DEPLOY / "golden_vectors.json").read_text())
    with np.load(DEPLOY / "golden_vectors.npz") as values:
        vectors, host = values["quantized"], values["host_raw"]
    rows = []
    for index, vector in enumerate(vectors):
        observed = [send_window(port, vector, timeout) for _ in range(repeats)]
        raws = [[item[f"raw{i}"] for i in range(4)] for item in observed]
        classes = [item["class"] for item in observed]
        rows.append({
            "vector_id": meta["vectors"][index]["vector_id"],
            "host_raw": [int(value) for value in host[index]],
            "host_class": int(np.argmax(host[index])),
            "target_raw": raws[0], "target_class": classes[0],
            "repeatability_exact": all(value == raws[0] for value in raws) and all(value == classes[0] for value in classes),
            "repeat_raw": raws, "repeat_classes": classes,
            "host_target_delta": [int(raws[0][j] - int(host[index][j])) for j in range(4)],
            "saturated": bool(np.any((host[index] <= -127) | (host[index] >= 126)) or np.any((np.asarray(raws[0]) <= -127) | (np.asarray(raws[0]) >= 126))),
            "winner_saturated": bool(abs(int(host[index][np.argmax(host[index])])) >= 126),
            "argmax_margin": int(int(np.sort(host[index])[-1]) - int(np.sort(host[index])[-2])),
        })
    return {"runtime": runtime, "repeats": repeats, "rows": rows,
            "raw_exact": sum(row["host_raw"] == row["target_raw"] for row in rows),
            "class_exact": sum(row["host_class"] == row["target_class"] for row in rows),
            "repeatability_exact": all(row["repeatability_exact"] for row in rows)}


def freeze_policy(cpu_path: Path, u55_path: Path, output: Path) -> dict[str, object]:
    cpu, u55 = json.loads(cpu_path.read_text()), json.loads(u55_path.read_text())
    cpu_rows, u55_rows = cpu["rows"], u55["rows"]
    if [row["vector_id"] for row in cpu_rows] != [row["vector_id"] for row in u55_rows]:
        raise ValueError("CPU/U55 golden IDs differ")
    rows = []
    for c, u in zip(cpu_rows, u55_rows):
        if c["host_raw"] != u["host_raw"]: raise ValueError("host bindings differ")
        rows.append({"vector_id": c["vector_id"], "host_cpu_delta_linf": max(abs(value) for value in c["host_target_delta"]), "host_u55_delta_linf": max(abs(value) for value in u["host_target_delta"]), "cpu_u55_delta_linf": max(abs(c["target_raw"][i] - u["target_raw"][i]) for i in range(4)), "saturated": c["saturated"] or u["saturated"], "class_exact": c["host_class"] == c["target_class"] == u["target_class"]})
    saturated = [row for row in rows if row["saturated"]]
    non_saturated = [row for row in rows if not row["saturated"]]
    # Frozen before any async replay.  The bound is the observed golden maximum,
    # expressed in raw INT8 counts, and applies only with class parity.
    bound = max(max(row["host_cpu_delta_linf"], row["host_u55_delta_linf"], row["cpu_u55_delta_linf"]) for row in saturated)
    policy = {"protocol": "terrain_v4_target_runtime_parity_v1", "frozen_before_async_hil": True,
              "historical_strict_raw_gate": "TERRAIN_V4_E84_FIXED_PARITY=FAIL (preserved)",
              "golden_count": len(rows), "classification": rows,
              "rules": {"target_repeatability": "exact raw/class for 3 repeats", "class_parity": "Host/CPU/U55 exact for every golden", "non_saturated_raw": "exact", "saturated_raw": {"linf_raw_count_lte": bound, "require_class_parity": True}, "continuous": {"class_exact_rate_gte": 0.98, "stable_state_exact_rate_gte": 0.98, "t1_abs_delta_ms_lte": 1}},
              "TERRAIN_V4_TARGET_RUNTIME_PARITY_GATE": "PASS" if cpu["repeatability_exact"] and u55["repeatability_exact"] and all(row["class_exact"] for row in rows) and all(row["host_cpu_delta_linf"] == 0 and row["host_u55_delta_linf"] == 0 and row["cpu_u55_delta_linf"] == 0 for row in non_saturated) and all(max(row["host_cpu_delta_linf"], row["host_u55_delta_linf"], row["cpu_u55_delta_linf"]) <= bound for row in saturated) else "FAIL"}
    output.write_text(json.dumps(policy, indent=2) + "\n")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture"); capture_parser.add_argument("--runtime", choices=("cpu", "u55"), required=True); capture_parser.add_argument("--port", type=Path, default=PORT); capture_parser.add_argument("--repeats", type=int, default=3); capture_parser.add_argument("--timeout", type=float, default=2.0); capture_parser.add_argument("--output", type=Path, required=True)
    policy_parser = sub.add_parser("freeze-policy"); policy_parser.add_argument("--cpu", type=Path, required=True); policy_parser.add_argument("--u55", type=Path, required=True); policy_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture(args.runtime, args.port, args.repeats, args.timeout) if args.command == "capture" else freeze_policy(args.cpu, args.u55, args.output)
    if args.command == "capture": args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
