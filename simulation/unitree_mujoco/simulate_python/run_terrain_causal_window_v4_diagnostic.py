"""Read-only replay of the preserved 12-run diagnostic after v4 fresh-test pass."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_terrain_causal_window_v4 import OUT, STATIC, ChannelNormalizer, load_transition
from run_terrain_transition_ai_replay import REFLEX, load, plots, reflex_replay, row_for, summary
from terrain_int8 import predict_tflite


SIM = Path(__file__).resolve().parents[2]
SOURCE = SIM / "outputs/terrain_transition_v1_pilot"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT / "diagnostic_replay")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("diagnostic replay is one-shot after fresh test")
    fresh = json.loads((OUT / "fresh_test/summary.json").read_text())
    if not fresh["FRESH_TRANSITION_TEST_PASS"]:
        raise RuntimeError("diagnostic replay requires a passing fresh transition test")
    int8 = json.loads((OUT / "int8/summary.json").read_text())
    selected = int8["selected_float"]
    if int(selected["window_ms"]) != 50:
        raise RuntimeError("preserved diagnostic currently supports the selected 50-ms model only")
    with np.load(STATIC) as values:
        static_x, static_split = values["X"], values["split"]
    _, _, transition_x, _, transition_split = load_transition(50)
    normalizer = ChannelNormalizer.fit(np.concatenate((static_x, transition_x))[np.concatenate((static_split, transition_split)) == "train"])
    manifest, data = load(SOURCE)
    model = Path(int8["manifest"]["int8_path"])
    windows = np.asarray([trace[end - 49 : end + 1] for trace in data["fusion10"] for end in range(49, 800)], np.float32)
    scores = predict_tflite(model, normalizer.transform(windows)).reshape(12, 751, 4)
    terrain = np.full((12, 800), -1, np.int8)
    terrain[:, 49:] = np.argmax(scores, axis=2)
    raw, firing = {}, {}
    for kind in REFLEX:
        raw[kind], firing[kind] = reflex_replay(data["fusion10"], kind)
    rows = [row_for(index, row, data, terrain, raw, firing) for index, row in enumerate(manifest)]
    report = summary(rows)
    out.mkdir(parents=True)
    with (out / "timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "replay_outputs.npz", terrain_scores=scores.astype(np.float32), terrain_prediction=terrain, **{f"{kind}_raw_int8": values for kind, values in raw.items()}, **{f"{kind}_raw_positive": values for kind, values in firing.items()})
    protocol = {"source": str(SOURCE), "source_runs": 12, "terrain_model": int8["manifest"], "fast_reflex_frozen": {kind: {"model": str(value["model"]), "sha256": digest(value["model"]), "window_ms": value["window"], "raw_threshold": value["raw_threshold"], "persistence": value["persistence"]} for kind, value in REFLEX.items()}, "no_terrain_or_fast_reflex_training_or_tuning_after_fresh_test": True}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plots(out, rows, data, scores, terrain, raw, firing)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
