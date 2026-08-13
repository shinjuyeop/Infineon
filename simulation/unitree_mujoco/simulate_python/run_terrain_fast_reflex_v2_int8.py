"""Strict train-calibrated INT8 export and validation-only parity for frozen v2 detectors."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from terrain_int8 import export_full_int8_binary, predict_tflite_binary, quantize, TensorQuantization
from run_terrain_fast_reflex_v2_final import FROZEN, sha256
from run_terrain_fast_reflex_v2_validation import endpoint_metrics, replay, subgroup_rows, write_csv

ROOT=Path("../.."); DATASET=ROOT/"outputs/terrain_fast_reflex_v2_detector_dataset"; OUT=ROOT/"outputs/terrain_fast_reflex_v2_int8"
REPRESENTATIVE_SAMPLES=128; REPRESENTATIVE_SEED=20260813

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--execute",action="store_true");p.add_argument("--goldens-only",action="store_true");return p.parse_args()
def choose(data):
 """Deterministic train-only label×family allocation; no validation feedback."""
 y,f=data["y"],data["family"].astype(str); strata=sorted({(int(a),str(b)) for a,b in zip(y,f)}); rng=np.random.default_rng(REPRESENTATIVE_SEED); base,extra=divmod(REPRESENTATIVE_SAMPLES,len(strata)); selected=[]
 for i,(label,fam) in enumerate(strata):
  candidates=np.flatnonzero((y==label)&(f==fam)); take=base+(i<extra)
  if take>len(candidates):raise ValueError(f"representative stratum too small {(label,fam)}")
  selected.extend(rng.choice(candidates,take,replace=False).tolist())
 result=np.asarray(selected,np.int64);rng.shuffle(result);return result
def tensor(spec): return {"shape":list(spec.shape),"dtype":spec.dtype,"scale":spec.scale,"zero_point":spec.zero_point}
def quant_threshold(threshold,spec):
 q=int(np.clip(np.rint(threshold/spec.scale+spec.zero_point),-128,127));effective=float(spec.scale*(q-spec.zero_point));return {"float_threshold":threshold,"quantized_threshold":q,"effective_dequantized_threshold":effective,"quantization_error":effective-threshold,"comparison":"raw_int8_output >= q_threshold (equivalent to dequantized output >= effective threshold)"}
def values_or_none(data,key): return data[key] if data[key] is not None else None
def goldens(out):
 """Create six validation-only raw/normalized/INT8 deployment vectors per detector."""
 source=ROOT/"outputs/terrain_fast_reflex_v2_final_scope_full"
 with np.load(source/"inputs_fusion10.npz",allow_pickle=False) as z: raw,ids=z["sensors"],z["run_id"].astype(str)
 lookup={name:i for i,name in enumerate(ids)}; import tensorflow as tf
 for detector,c in FROZEN.items():
  d=out/detector;window=c["window_ms"]
  if not (d/"model_int8.tflite").is_file() or (d/"golden_vectors.npz").exists():raise FileExistsError(f"missing model or existing goldens for {detector}")
  with np.load(DATASET/f"{detector}_{window}ms/validation.npz",allow_pickle=False) as z: valid={k:z[k] for k in z.files}
  model=tf.keras.models.load_model(c["model"],compile=False);score=model.predict(valid["x"],batch_size=1024,verbose=0).reshape(-1);int8,rawout=predict_tflite_binary(d/"model_int8.tflite",valid["x"],window)
  order=np.argsort(np.abs(score-c["threshold"])); chosen=[]
  for label in (0,1): chosen.extend([int(i) for i in order if valid["y"][i]==label][:3])
  chosen=np.asarray(chosen,np.int64);q=json.loads((d/"quantization.json").read_text());spec=TensorQuantization(tuple(q["input"]["shape"]),q["input"]["dtype"],q["input"]["scale"],q["input"]["zero_point"])
  raw_windows=np.asarray([raw[lookup[str(valid["run_id"][i])],50+int(valid["endpoint_ms"][i])-window+1:51+int(valid["endpoint_ms"][i])] for i in chosen],np.float32)
  np.savez_compressed(d/"golden_vectors.npz",raw_fusion10=raw_windows,normalized_fusion10=valid["x"][chosen],quantized_int8_input=np.asarray([quantize(v,spec) for v in valid["x"][chosen]]),raw_int8_output=rawout[chosen],dequantized_output=int8[chosen],binary_decision=(int8[chosen]>=c["threshold"]),label=valid["y"][chosen],run_id=valid["run_id"][chosen],endpoint_ms=valid["endpoint_ms"][chosen])
  (d/"golden_manifest.json").write_text(json.dumps({"source_split":"validation","count":len(chosen),"composition":"3 negative near-threshold + 3 positive near-threshold; no final samples","run_ids":valid["run_id"][chosen].astype(str).tolist(),"endpoints_ms":valid["endpoint_ms"][chosen].astype(int).tolist()},indent=2)+"\n")
def main():
 a=args()
 if a.goldens_only:
  goldens(a.output_dir.resolve());print(f"V2_INT8_GOLDENS_COMPLETE output={a.output_dir.resolve()}");return
 if not a.execute:print("Dry run only. Train-only representative; validation-only parity; final test forbidden.");return
 out=a.output_dir.resolve()
 if out.exists() and any(out.iterdir()):raise FileExistsError(f"refusing non-empty {out}")
 out.mkdir(parents=True); summary=[]
 import tensorflow as tf
 for detector,c in FROZEN.items():
  d=out/detector;d.mkdir(); window=c["window_ms"]
  with np.load(DATASET/f"{detector}_{window}ms/train.npz",allow_pickle=False) as z: train={k:z[k] for k in z.files}
  with np.load(DATASET/f"{detector}_{window}ms/validation.npz",allow_pickle=False) as z: valid={k:z[k] for k in z.files}
  if set(train["family"].astype(str)) & {"crosshatch","rounded_ridges"}:raise ValueError("validation family leaked into train representative")
  if sha256(c["model"])!=c["model_sha256"] or sha256(c["normalization"])!=c["normalization_sha256"]:raise ValueError("frozen model/normalization integrity mismatch")
  selected=choose(train); model=tf.keras.models.load_model(c["model"],compile=False)
  if tuple(model.input_shape)!=(None,window,10):raise ValueError("frozen input shape mismatch")
  tflite=d/"model_int8.tflite"; inp,outp,ops=export_full_int8_binary(model,train["x"][selected],tflite,window)
  float_score=model.predict(valid["x"],batch_size=1024,verbose=0).reshape(-1);int8_score,raw=predict_tflite_binary(tflite,valid["x"],window)
  error=np.abs(float_score-int8_score);threshold=quant_threshold(c["threshold"],outp); float_pred=float_score>=c["threshold"];int8_pred=int8_score>=c["threshold"]
  fr,frm=replay(valid,float_score,c["threshold"],c["persistence"]); ir,irm=replay(valid,int8_score,c["threshold"],c["persistence"])
  latency_delta=None if frm["latency_p95_ms"] is None or irm["latency_p95_ms"] is None else irm["latency_p95_ms"]-frm["latency_p95_ms"]
  checks={"int8_overall_causal_run_fpr_at_most_5pct":irm["overall_causal_run_fpr"]<=.05,"run_recall_delta_at_least_minus_1pp":irm["run_recall"]-frm["run_recall"]>=-.01,"p95_latency_regression_at_most_5ms":latency_delta is not None and latency_delta<=5}
  report={"detector":detector,"frozen":{k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in c.items()},"representative":{"source_split":"train","seed":REPRESENTATIVE_SEED,"samples":len(selected),"family_counts":{f:int((train['family'][selected].astype(str)==f).sum()) for f in sorted(set(train['family'].astype(str)))},"label_counts":{str(v):int((train['y'][selected]==v).sum()) for v in (0,1)}},"tflite":{"path":str(tflite),"size_bytes":tflite.stat().st_size,"input":tensor(inp),"output":tensor(outp),"operators":ops,"strict_full_int8":True},"threshold":threshold,"score_error":{"mean":float(error.mean()),"median":float(np.median(error)),"p95":float(np.percentile(error,95)),"max":float(error.max()),"correlation":float(np.corrcoef(float_score,int8_score)[0,1])},"agreement":{"endpoint_binary":float(np.mean(float_pred==int8_pred)),"positive_endpoint":float(np.mean(float_pred[valid['y']==1]==int8_pred[valid['y']==1])),"negative_endpoint":float(np.mean(float_pred[valid['y']==0]==int8_pred[valid['y']==0])),"stable_firing_run":float(np.mean(np.asarray([r['stable_firing_ms'] is not None for r in fr])==np.asarray([r['stable_firing_ms'] is not None for r in ir])))},"float":{"endpoint":endpoint_metrics(valid['y'],float_score,c['threshold']),"run":frm},"int8":{"endpoint":endpoint_metrics(valid['y'],int8_score,c['threshold']),"run":irm},"gate_checks":checks,"host_gate":"PASS" if all(checks.values()) else "FAIL"}
  (d/"quantization.json").write_text(json.dumps({"input":tensor(inp),"output":tensor(outp),"threshold":threshold,"normalization_path":str(c["normalization"]),"normalization_sha256":c["normalization_sha256"],"equations":{"input":"q=clip(round(x/scale+zero_point),-128,127)","output":"x=scale*(q-zero_point)"}},indent=2)+"\n")
  (d/"representative_manifest.json").write_text(json.dumps(report["representative"],indent=2)+"\n");(d/"conversion_protocol.json").write_text(json.dumps(report,indent=2)+"\n");write_csv(d/"family_metrics.csv",[{"format":"float",**x} for x in subgroup_rows(fr,"family")]+[{"format":"int8",**x} for x in subgroup_rows(ir,"family")]);write_csv(d/"mode_metrics.csv",[{"format":"float",**x} for x in subgroup_rows(fr,"mode")]+[{"format":"int8",**x} for x in subgroup_rows(ir,"mode")]);summary.append(report)
 vela_ready=all(x["host_gate"]=="PASS" for x in summary);(out/"int8_summary.md").write_text("# Fast Reflex v2 strict INT8 validation\n\n"+"\n".join(f"- {x['detector'].upper()}_INT8_HOST_GATE={x['host_gate']}" for x in summary)+f"\n- VELA_READY={'true' if vela_ready else 'false'}\n",encoding="utf-8");print(f"V2_INT8_COMPLETE vela_ready={str(vela_ready).lower()} output={out}")
if __name__=="__main__":main()
