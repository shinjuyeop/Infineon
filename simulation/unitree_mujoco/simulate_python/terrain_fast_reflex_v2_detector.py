"""Causal, split-safe detector datasets for the frozen Fast Reflex v2 scope."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terrain_fast_reflex_v2 import TRACE_PRE_MS, TRACE_SAMPLES

FUSION_CHANNELS = 10
WINDOWS = {"slip": (5, 10, 20), "sink": (10, 20, 30, 50)}
TRAIN_STRIDE_MS, VALIDATION_STRIDE_MS = 2, 1
TARGETS = {"slip": "confirmed_slip", "sink": "sustained_sink"}


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        if values.ndim != 3 or values.shape[-1] != FUSION_CHANNELS or not len(values):
            raise ValueError("expected non-empty (N,L,10) train windows")
        mean = values.astype(np.float64).mean((0, 1)); std = values.astype(np.float64).std((0, 1))
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (values.astype(np.float32) - self.mean) / self.std
        if not np.isfinite(result).all(): raise ValueError("normalization made NaN/Inf")
        return result.astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "provenance": "fit only on train windows"}


def load_v2(source: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("dataset_name") != "terrain_fast_reflex_v2" or protocol.get("final_test", {}).get("materialized"):
        raise ValueError("source must be non-final-test terrain_fast_reflex_v2")
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    with np.load(source / "inputs_fusion10.npz", allow_pickle=False) as data:
        sensors, timestamps, ids = data["sensors"], data["sample_time_s"], data["run_id"].astype(str)
    with np.load(source / "oracle_diagnostics.npz", allow_pickle=False) as data:
        labels = {name: np.asarray(data[name], bool) for name in ("confirmed_slip", "incipient_risk", "sustained_sink", "sustained_tilt")}
        oracle_ids = data["run_id"].astype(str)
    manifest_ids = np.asarray([row["run_id"] for row in rows])
    if not (len(rows) == len(sensors) == len(timestamps) and np.array_equal(manifest_ids, ids) and np.array_equal(ids, oracle_ids)):
        raise ValueError("manifest/input/oracle ordering mismatch")
    if sensors.shape[1:] != (TRACE_SAMPLES, FUSION_CHANNELS) or not np.isfinite(sensors).all() or not np.allclose(np.diff(timestamps, axis=1), .001, atol=1e-9):
        raise ValueError("expected finite native-1kHz Fusion10 traces")
    if any(row["split"] not in ("train", "validation") or row["valid"] != "1" for row in rows):
        raise ValueError("only valid train/validation v2 runs are allowed")
    ownership: dict[str, str] = {}
    for row in rows:
        key = f"{row['surface_family']}::{row['surface_seed']}::{row['session_id']}::{row['run_id']}"
        old = ownership.setdefault(key, row["split"])
        if old != row["split"]: raise ValueError("run ownership leakage")
    return rows, sensors.astype(np.float32), timestamps, labels


def make_windows(rows: list[dict[str, str]], sensors: np.ndarray, labels: dict[str, np.ndarray], detector: str, window_ms: int, split: str) -> dict[str, np.ndarray]:
    if detector not in TARGETS or window_ms not in WINDOWS[detector] or split not in ("train", "validation"):
        raise ValueError("unsupported detector/window/split")
    stride = TRAIN_STRIDE_MS if split == "train" else VALIDATION_STRIDE_MS
    x=[]; y=[]; run=[]; endpoint=[]; family=[]; mode=[]; incipient=[]; tilt=[]
    target=labels[TARGETS[detector]]
    for index,row in enumerate(rows):
        if row["split"] != split: continue
        for endpoint_ms in range(0, 100, stride):
            end=TRACE_PRE_MS + endpoint_ms; start=end-window_ms+1
            if start < 0: continue
            x.append(sensors[index,start:end+1]); y.append(target[index,end]); run.append(row["run_id"]); endpoint.append(endpoint_ms)
            family.append(row["surface_family"]); mode.append(row["mode"]); incipient.append(labels["incipient_risk"][index,end]); tilt.append(labels["sustained_tilt"][index,end])
    if not x: raise ValueError(f"no {split} windows")
    return {"x":np.asarray(x,np.float32),"y":np.asarray(y,np.int8),"run_id":np.asarray(run),"endpoint_ms":np.asarray(endpoint,np.int16),"family":np.asarray(family),"mode":np.asarray(mode),"incipient_diagnostic":np.asarray(incipient,bool),"tilt_diagnostic":np.asarray(tilt,bool)}


def statistics(data: dict[str, np.ndarray], split: str, detector: str, window_ms: int) -> dict[str, object]:
    y=data["y"]; positive=y.astype(bool)
    positive_ids=set(data["run_id"][positive]); negative_ids=set(data["run_id"][~positive])
    output={"detector":detector,"window_ms":window_ms,"split":split,"windows":int(len(y)),"positive_windows":int(y.sum()),"negative_windows":int((~positive).sum()),"positive_ratio":float(y.mean()),"positive_runs":int(len(positive_ids)),"negative_only_runs":int(len(negative_ids-positive_ids))}
    output["family_positive_windows"]={name:int(y[data["family"]==name].sum()) for name in sorted(set(data["family"]))}
    output["family_positive_runs"]={name:int(len(set(data["run_id"][(data["family"]==name)&y.astype(bool)]))) for name in sorted(set(data["family"]))}
    output["mode_windows"]={name:{"positive":int(y[data["mode"]==name].sum()),"negative":int((~y.astype(bool))[data["mode"]==name].sum())} for name in sorted(set(data["mode"]))}
    output["onset_relative_bins"]={"pre_onset_negative":int((~y.astype(bool)).sum()),"active_positive":int(y.sum())}
    return output


def save_dataset(output: Path, source: Path) -> dict[str, object]:
    rows,sensors,timestamps,labels=load_v2(source); output.mkdir(parents=True,exist_ok=True)
    report={"source":str(source.resolve()),"final_test_materialized":0,"window_semantics":"causal [t-L+1,t]; pre-onset endpoint remains negative","targets":TARGETS,"training_stride_ms":TRAIN_STRIDE_MS,"validation_stride_ms":VALIDATION_STRIDE_MS,"normalization":"train windows only","datasets":[]}
    for detector, windows in WINDOWS.items():
        for window_ms in windows:
            directory=output/f"{detector}_{window_ms}ms";directory.mkdir()
            sets={split:make_windows(rows,sensors,labels,detector,window_ms,split) for split in ("train","validation")}
            normalizer=Normalizer.fit(sets["train"]["x"])
            for split,data in sets.items():
                np.savez_compressed(directory/f"{split}.npz",x=normalizer.transform(data["x"]),y=data["y"],run_id=data["run_id"],endpoint_ms=data["endpoint_ms"],family=data["family"],mode=data["mode"],incipient_diagnostic=data["incipient_diagnostic"],tilt_diagnostic=data["tilt_diagnostic"])
                report["datasets"].append(statistics(data,split,detector,window_ms))
            (directory/"normalization.json").write_text(json.dumps(normalizer.as_dict(),indent=2)+"\n",encoding="utf-8")
    (output/"dataset_statistics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/"protocol.json").write_text(json.dumps({"status":"v2 causal detector dataset; no training","source":str(source.resolve()),"final_test_materialized":0,"targets":TARGETS,"causal_labeling":True,"scenario_name_labels_forbidden":True,"split_ownership":"source run/family split retained"},indent=2)+"\n",encoding="utf-8")
    return report
