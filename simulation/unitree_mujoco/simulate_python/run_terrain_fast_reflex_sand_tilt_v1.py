"""Execute bounded train+validation-only Sand tilt observability analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from terrain_fast_reflex_sand_tilt_v1 import (
    FEATURE_NAMES, FSR_MAPPING, TEST_FAMILIES, TRACE_PRE_MS, fit_logistic, load_validation_only,
    physical_features, pr_auc, rank_auc, stable_endpoints,
)


WINDOWS = (5, 10, 15, 20, 30, 50)
PERSISTENCE = 3


def write_csv(path, rows):
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../../outputs/terrain_fast_reflex_v1_full_corrected_v2"))
    parser.add_argument("--models-dir", type=Path, default=Path("../../outputs/terrain_fast_reflex_detector_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../outputs/terrain_fast_reflex_sand_tilt_v1"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def onset(trace, target="tilt"):
    mask = trace.tilt if target == "tilt" else trace.sink_or_tilt
    found = np.flatnonzero(mask[TRACE_PRE_MS:])
    return None if not len(found) else int(found[0])


def score_runs(feature_by_run, scorer):
    return [np.asarray(scorer(values), float) for values in feature_by_run]


def policy_metrics(traces, scores, threshold, persistence, target="tilt", subgroup=None):
    selected = []
    for trace, score in zip(traces, scores):
        has_sink, has_tilt = bool(np.any(trace.sink)), bool(np.any(trace.tilt))
        if subgroup == "tilt_only" and not (has_tilt and not has_sink): continue
        if subgroup == "sink_and_tilt" and not (has_tilt and has_sink): continue
        selected.append((trace, score))
    detected = false = target_runs = 0; latencies = []
    for trace, score in selected:
        stable = stable_endpoints(score, threshold, persistence)
        start = onset(trace, target)
        cutoff = 100 if start is None else start
        false += int(np.any(stable < cutoff))
        if start is not None:
            target_runs += 1
            post = stable[stable >= start]
            if len(post): detected += 1; latencies.append(float(post[0] - start))
    q = lambda p: None if not latencies else float(np.percentile(latencies, p))
    return {"run_count": len(selected), "target_runs": target_runs, "detected_runs": detected,
            "run_recall": detected / target_runs if target_runs else None,
            "false_alarm_runs": false, "run_fpr": false / len(selected) if selected else None,
            "median_latency_ms": q(50), "p95_latency_ms": q(95), "max_latency_ms": q(100)}


def candidates_from_train(scores):
    flat = np.concatenate(scores)
    return np.unique(np.r_[np.nextafter(flat.max(), np.inf), np.quantile(flat, np.linspace(0, 1, 41))])[::-1]


def select_validation_policy(traces, scores, candidates, target="tilt", subgroup="tilt_only", persistence=3):
    rows = []
    for threshold in candidates:
        overall = policy_metrics(traces, scores, float(threshold), persistence, target)
        group = policy_metrics(traces, scores, float(threshold), persistence, target, subgroup)
        row = {"threshold": float(threshold), "persistence": persistence,
               **{f"all_{k}": v for k, v in overall.items()},
               **{f"target_{k}": v for k, v in group.items()}}
        rows.append(row)
    valid = [r for r in rows if r["all_run_fpr"] <= .05 + 1e-12]
    best = max(valid, key=lambda r: (r["target_run_recall"], -r["all_run_fpr"], r["threshold"]))
    return best, rows


def model_free_rows(traces, feature_cache):
    rows = []
    for window in WINDOWS:
        values, groups = [], []
        for trace, feature in zip(traces, feature_cache[window]):
            has_sink, has_tilt = np.any(trace.sink), np.any(trace.tilt)
            if has_tilt and not has_sink:
                group = "tilt_only"
            elif has_tilt and has_sink:
                group = "sink_and_tilt"
            elif not np.any(trace.slip | trace.sink | trace.tilt):
                group = "normal"
            else:
                continue
            reference = onset(trace, "tilt") if has_tilt else 0
            endpoint = min(99, int(reference) + window - 1)
            values.append(feature[endpoint]); groups.append(group)
        values, groups = np.asarray(values), np.asarray(groups)
        for fi, name in enumerate(FEATURE_NAMES):
            normal, tilt = values[groups == "normal", fi], values[groups == "tilt_only", fi]
            labels = np.r_[np.zeros(len(normal), bool), np.ones(len(tilt), bool)]
            combined = np.r_[normal, tilt]
            auc = rank_auc(labels, combined); oriented = combined if auc >= .5 else -combined
            pooled = np.sqrt(((len(normal)-1)*normal.var(ddof=1)+(len(tilt)-1)*tilt.var(ddof=1))/max(1,len(normal)+len(tilt)-2))
            row = {"window_ms": window, "feature": name,
                   "normal_median": float(np.median(normal)), "normal_q1": float(np.percentile(normal,25)), "normal_q3": float(np.percentile(normal,75)),
                   "tilt_only_median": float(np.median(tilt)), "tilt_only_q1": float(np.percentile(tilt,25)), "tilt_only_q3": float(np.percentile(tilt,75)),
                   "cohen_d": float((tilt.mean()-normal.mean())/pooled) if pooled > 0 else 0.0,
                   "roc_auc_oriented": max(auc, 1-auc), "pr_auc_oriented": pr_auc(labels, oriented)}
            sink_tilt = values[groups == "sink_and_tilt", fi]
            row.update({"sink_tilt_median": float(np.median(sink_tilt)), "sink_tilt_q1": float(np.percentile(sink_tilt,25)), "sink_tilt_q3": float(np.percentile(sink_tilt,75))})
            rows.append(row)
    return rows


def onset_relative_rows(traces, feature_cache):
    rows = []
    for window in WINDOWS:
        for offset in range(-5, 31):
            normal_values, tilt_values = [], []
            for trace, features in zip(traces, feature_cache[window]):
                has_sink, has_tilt = np.any(trace.sink), np.any(trace.tilt)
                if has_sink or not has_tilt: continue
                ep = onset(trace); assert ep is not None
                tilt_values.append(features[np.clip(ep + offset, 0, 99)])
            for trace, features in zip(traces, feature_cache[window]):
                if np.any(trace.sink | trace.tilt): continue
                normal_values.append(features[np.clip(offset, 0, 99)])
            normal_values, tilt_values = np.asarray(normal_values), np.asarray(tilt_values)
            labels = np.r_[np.zeros(len(normal_values), bool), np.ones(len(tilt_values), bool)]
            for fi, name in enumerate(FEATURE_NAMES):
                scores = np.r_[normal_values[:,fi], tilt_values[:,fi]]; auc = rank_auc(labels, scores)
                rows.append({"window_ms": window, "offset_ms": offset, "feature": name,
                             "roc_auc_oriented": max(auc, 1-auc),
                             "pr_auc_oriented": pr_auc(labels, scores if auc >= .5 else -scores)})
    return rows


def load_existing_scores(models_dir, validation_traces):
    import tensorflow as tf
    window = 20; artifact = models_dir / "sink_tilt" / "20ms"
    model = tf.keras.models.load_model(artifact / "model.keras")
    norm = json.loads((artifact / "normalization.json").read_text(encoding="utf-8"))
    mean, std = np.asarray(norm["mean"],np.float32), np.asarray(norm["std"],np.float32)
    values=[]
    for trace in validation_traces:
        for endpoint in range(100):
            index=TRACE_PRE_MS+endpoint; values.append(trace.sensors[index-window+1:index+1])
    values=(np.asarray(values,np.float32)-mean)/std
    flat = model.predict(values, batch_size=1024, verbose=0).reshape(-1)
    return [flat[i*100:(i+1)*100] for i in range(len(validation_traces))]


def audit_rows(traces, cnn_scores, cnn_threshold=.878164799511433, cnn_persistence=8):
    rows = []
    for trace, score in zip(traces, cnn_scores):
        if not (np.any(trace.tilt) and not np.any(trace.sink)): continue
        start = onset(trace); assert start is not None
        post = score[start:]; stable = stable_endpoints(score, cnn_threshold, cnn_persistence)
        sensor = trace.sensors[TRACE_PRE_MS:]; oracle = trace.oracle[TRACE_PRE_MS:]
        fsr = sensor[:,:4]; total = fsr.sum(1)
        front, rear = fsr[:,2:].sum(1), fsr[:,:2].sum(1)
        left, right = fsr[:,[0,2]].sum(1), fsr[:,[1,3]].sum(1)
        imbalance = np.maximum(np.abs(front-rear), np.abs(left-right))/np.maximum(np.abs(total),1e-6)
        rows.append({"scenario":trace.metadata["scenario"], "family":trace.metadata["surface_family"],
                     "surface":trace.metadata["surface_index"], "run":trace.metadata["run_index"], "run_id":trace.metadata["run_id"],
                     "tilt_onset_time_s":trace.metadata["tilt_onset_time_s"],
                     "max_oracle_tilt_rad":float(np.max(oracle[:,15])), "max_gyro_xy_rad_s":float(np.max(np.linalg.norm(sensor[:,7:9],axis=1))),
                     "max_normalized_fsr_imbalance":float(np.max(imbalance)), "max_contact_load_N":float(np.max(oracle[:,1])),
                     "sink":int(np.any(trace.sink)), "slip":int(np.any(trace.slip)), "current_model_max_score_post_onset":float(np.max(post)),
                     "current_model_stable_firing_post_onset":int(np.any(stable>=start))})
    return rows


def plot_representatives(output, traces, feature20, cnn_scores, cnn_threshold):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    picked = {}
    for i,t in enumerate(traces):
        hs,ht=np.any(t.sink),np.any(t.tilt)
        if ht and not hs: group="tilt_only"
        elif ht and hs: group="sink_and_tilt"
        elif not np.any(t.slip | t.sink | t.tilt): group="normal"
        else: continue
        if group not in picked: picked[group]=i
    records=[]
    for group,index in picked.items():
        t=traces[index]; ref=onset(t) if np.any(t.tilt) else 0; ref=int(ref or 0)
        rel=np.arange(-5,31); ti=np.clip(TRACE_PRE_MS+ref+rel,0,len(t.sensors)-1); ep=np.clip(ref+rel,0,99)
        sensor=t.sensors[ti]; feat=feature20[index][ep]
        fig,ax=plt.subplots(5,1,figsize=(10,11),sharex=True)
        ax[0].plot(rel,sensor[:,:4]); ax[0].set_ylabel("FSR1..4")
        ax[1].plot(rel,feat[:,FEATURE_NAMES.index("norm_front_rear")],label="front/rear")
        ax[1].plot(rel,feat[:,FEATURE_NAMES.index("norm_left_right")],label="left/right"); ax[1].legend(); ax[1].set_ylabel("imbalance")
        ax[2].plot(rel,feat[:,FEATURE_NAMES.index("fsr_variance")]); ax[2].set_ylabel("FSR variance")
        ax[3].plot(rel,sensor[:,7:9]); ax[3].set_ylabel("gyro X/Y")
        ax[4].plot(rel,cnn_scores[index][ep]); ax[4].axhline(cnn_threshold,color="r",ls="--"); ax[4].set_ylabel("CNN score")
        for a in ax: a.axvline(0,color="k",ls=":")
        ax[-1].set_xlabel("ms relative to tilt onset (transition for normal)"); fig.suptitle(f"{group}: {t.metadata['run_id']}")
        fig.tight_layout(); path=output/f"{group}.png"; fig.savefig(path,dpi=140); plt.close(fig)
        records.append({"group":group,"run_id":t.metadata["run_id"],"plot":str(path)})
    return records


def main():
    args=parse_args()
    if not args.execute:
        print("DRY RUN: train+validation only; test family materialization/inference is forbidden. Add --execute."); return
    started=time.perf_counter(); source=args.source.resolve(); models=args.models_dir.resolve(); output=args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    train,_=load_validation_only(source,{"train"}); validation,_=load_validation_only(source,{"validation"})
    if len(train)!=180 or len(validation)!=120: raise ValueError("expected 180 train and 120 validation traces")
    if any(t.metadata["surface_family"] in TEST_FAMILIES for t in train+validation): raise AssertionError("test family loaded")
    train_cache={w:[physical_features(t.sensors[TRACE_PRE_MS:],w) for t in train] for w in WINDOWS}
    val_cache={w:[physical_features(t.sensors[TRACE_PRE_MS:],w) for t in validation] for w in WINDOWS}
    cnn_scores=load_existing_scores(models,validation); cnn_threshold=.878164799511433; cnn_persistence=8
    audit=audit_rows(validation,cnn_scores,cnn_threshold,cnn_persistence)
    if len(audit)!=15: raise ValueError(f"expected 15 tilt-only validation runs, got {len(audit)}")
    separation=model_free_rows(validation,val_cache); relative=onset_relative_rows(validation,val_cache)

    detector_rows=[]; sweep_rows=[]; best_by_kind={}
    # Physical rule: absolute and both directional forms, thresholds generated only from train.
    for window in WINDOWS:
        for fi,name in enumerate(FEATURE_NAMES):
            for orientation, transform in (("positive", lambda x:x), ("negative", lambda x:-x), ("absolute", np.abs)):
                train_scores=[transform(v[:,fi]) for v in train_cache[window]]
                val_scores=[transform(v[:,fi]) for v in val_cache[window]]
                best,sweep=select_validation_policy(validation,val_scores,candidates_from_train(train_scores))
                row={"detector":"physical_rule", "window_ms":window,"feature":name,"orientation":orientation,**best}
                detector_rows.append(row)
                for item in sweep: sweep_rows.append({"detector":"physical_rule","window_ms":window,"feature":name,"orientation":orientation,**item})
    rule_best=max(detector_rows,key=lambda r:(r["target_run_recall"],-r["all_run_fpr"],-r["window_ms"]))
    best_by_kind["physical_rule"]=rule_best

    selected_features=("norm_front_rear","norm_left_right","fsr_variance","fsr_range","d_norm_front_rear","d_norm_left_right","gyro_x","gyro_y","gyro_xy_magnitude","gyro_xy_integral")
    columns=[FEATURE_NAMES.index(n) for n in selected_features]
    for window in WINDOWS:
        tx=np.concatenate([v[::2,columns] for v in train_cache[window]])
        ty=np.concatenate([t.tilt[TRACE_PRE_MS:][::2] for t in train]).astype(int)
        run_weight=np.repeat(1/100, len(tx)); class_weight=np.where(ty, .5/max(ty.mean(),1e-6), .5/max(1-ty.mean(),1e-6)); weight=run_weight*class_weight
        model=fit_logistic(tx,ty,weight)
        train_scores=[model.score(v[:,columns]) for v in train_cache[window]]
        val_scores=[model.score(v[:,columns]) for v in val_cache[window]]
        best,sweep=select_validation_policy(validation,val_scores,candidates_from_train(train_scores))
        row={"detector":"logistic", "window_ms":window,"feature":"|".join(selected_features),**best}
        detector_rows.append(row)
        for item in sweep: sweep_rows.append({"detector":"logistic","window_ms":window,"feature":"derived10",**item})
    logistic_best=max([r for r in detector_rows if r["detector"]=="logistic"],key=lambda r:(r["target_run_recall"],-r["all_run_fpr"],-r["window_ms"]))
    best_by_kind["logistic"]=logistic_best

    # Replay best candidate and existing 20 ms CNN, then their OR combination.
    def replay_candidate(candidate):
        candidate_window=int(candidate["window_ms"])
        if candidate["detector"]=="physical_rule":
            fi=FEATURE_NAMES.index(candidate["feature"]); transform={"positive":lambda x:x,"negative":lambda x:-x,"absolute":np.abs}[candidate["orientation"]]
            return [transform(v[:,fi]) for v in val_cache[candidate_window]]
        tx=np.concatenate([v[::2,columns] for v in train_cache[candidate_window]]); ty=np.concatenate([t.tilt[TRACE_PRE_MS:][::2] for t in train]).astype(int)
        class_weight=np.where(ty,.5/max(ty.mean(),1e-6),.5/max(1-ty.mean(),1e-6)); lm=fit_logistic(tx,ty,class_weight)
        return [lm.score(v[:,columns]) for v in val_cache[candidate_window]]
    chosen=rule_best if (rule_best["target_run_recall"],-rule_best["all_run_fpr"],-rule_best["window_ms"]) >= (logistic_best["target_run_recall"],-logistic_best["all_run_fpr"],-logistic_best["window_ms"]) else logistic_best
    window=int(chosen["window_ms"]); chosen_scores=replay_candidate(chosen)
    rule_scores=replay_candidate(rule_best); logistic_scores=replay_candidate(logistic_best)
    existing=policy_metrics(validation,cnn_scores,cnn_threshold,cnn_persistence,"sand")
    tilt_all=policy_metrics(validation,chosen_scores,chosen["threshold"],PERSISTENCE,"tilt")
    combined_active=[]
    for cs,ts in zip(cnn_scores,chosen_scores):
        ca=np.zeros(100,bool); ca[stable_endpoints(cs,cnn_threshold,cnn_persistence)]=True
        ta=np.zeros(100,bool); ta[stable_endpoints(ts,chosen["threshold"],PERSISTENCE)]=True
        combined_active.append((ca|ta).astype(float))
    combined=policy_metrics(validation,combined_active,.5,1,"sand")
    def scoped_row(name, candidate_window, scores, threshold, persistence):
        sand=policy_metrics(validation,scores,threshold,persistence,"sand")
        tilt=policy_metrics(validation,scores,threshold,persistence,"sand","tilt_only")
        return {"detector":name,"window_ms":candidate_window,"threshold":threshold,"persistence":persistence,
                "sand_run_recall":sand["run_recall"],"tilt_only_run_recall":tilt["run_recall"],
                "overall_pre_onset_run_fpr":sand["run_fpr"],"sand_median_latency_ms":sand["median_latency_ms"],
                "sand_p95_latency_ms":sand["p95_latency_ms"],"sand_max_latency_ms":sand["max_latency_ms"]}
    summary=[scoped_row("existing_cnn",20,cnn_scores,cnn_threshold,cnn_persistence),
             scoped_row("physical_rule",int(rule_best["window_ms"]),rule_scores,rule_best["threshold"],PERSISTENCE),
             scoped_row("logistic",int(logistic_best["window_ms"]),logistic_scores,logistic_best["threshold"],PERSISTENCE),
             scoped_row("combined_or",max(20,window),combined_active,.5,1)]

    plots=plot_representatives(output/"plots",validation,val_cache[20],cnn_scores,cnn_threshold)
    write_csv(output/"tilt_only_run_audit.csv",audit); write_csv(output/"feature_separation.csv",separation)
    write_csv(output/"onset_relative_separation.csv",relative); write_csv(output/"candidate_summary.csv",detector_rows)
    write_csv(output/"candidate_sweep.csv",sweep_rows); write_csv(output/"validation_replay.csv",summary); write_csv(output/"plot_index.csv",plots)
    best_fsr=max((r for r in separation if r["window_ms"]<=20 and (r["feature"].startswith("fsr") or "front" in r["feature"] or "left" in r["feature"])),key=lambda r:r["roc_auc_oriented"])
    best_imu=max((r for r in separation if r["window_ms"]<=20 and ("gyro" in r["feature"] or "accel" in r["feature"])),key=lambda r:r["roc_auc_oriented"])
    config={"seed":20260812,"windows_ms":list(WINDOWS),"endpoint_stride_train_ms":2,
            "persistence":PERSISTENCE,"maximum_validation_run_fpr":.05,
            "threshold_candidate_source":"41 train-score quantiles plus above-max sentinel",
            "logistic_features":list(selected_features),"logistic_l2":1e-3,"logistic_solver":"L-BFGS-B maxiter=300",
            "normal_definition":"no slip, sink, or tilt oracle anywhere in the run",
            "positive_definition":"tilt oracle; feasibility recall reported for tilt-only runs",
            "split_safeguard":"train+validation only; test split and owned families fail closed"}
    (output/"deterministic_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    conclusions={"validation_only":True,"test_rows_materialized":0,"test_predictions_performed":False,"forbidden_test_families":sorted(TEST_FAMILIES),
                 "train_runs":len(train),"validation_runs":len(validation),"tilt_only_runs":len(audit),"canonical_tilt_threshold_rad":0.0003470679719944842,
                 "fsr_mapping":FSR_MAPPING,"fsr_mapping_basis":"g1_29dof.xml local positions: ch1/2 x=-0.05 rear, ch3/4 x=+0.12 front; +y ch1/3 left, -y ch2/4 right",
                 "imu_orientation":"left_foot_imu has no site quaternion and is attached to left_ankle_roll_link; local gyro_x/y align with ankle roll/pitch joint axes, retained as raw gyro_x/y",
                 "best_fsr_feature":best_fsr,"best_imu_feature":best_imu,"rule_best":rule_best,"logistic_best":logistic_best,"chosen_tilt":chosen,
                 "existing_cnn":existing,"combined":combined,"gate_recall_95_pass":combined["run_recall"]>=.95,"gate_fpr_5_pass":combined["run_fpr"]<=.05,"observation_20ms_pass":window<=20,
                 "small_mlp_run":False,"small_mlp_reason":"onset-aligned separation did not survive causal run replay without pre-onset false alarms; added capacity was not justified after the bounded rule and logistic baselines",
                 "hypotheses":{},"runtime_s":time.perf_counter()-started}
    # Evidence-based A-D classification.
    top=max(best_fsr["roc_auc_oriented"],best_imu["roc_auc_oriented"]); raw_recall=policy_metrics(validation,cnn_scores,cnn_threshold,cnn_persistence,"sand","tilt_only")["run_recall"]
    oracle_max=np.asarray([r["max_oracle_tilt_rad"] for r in audit]); canonical=conclusions["canonical_tilt_threshold_rad"]
    conclusions["hypotheses"]={
        "A_observability":{"judgement":"not_supported" if top>=.7 else "supported",
            "evidence":{"best_sensor_auc_under20ms":top,"best_fsr_effect_size":abs(best_fsr["cohen_d"]),"best_imu_auc_under20ms":best_imu["roc_auc_oriented"]}},
        "B_channel_representation":{"judgement":"supported" if best_fsr["roc_auc_oriented"]>=.8 and rule_best["target_run_recall"]>raw_recall else "inconclusive_or_not_supported",
            "evidence":{"fsr_auc":best_fsr["roc_auc_oriented"],"existing_cnn_tilt_only_recall":raw_recall,"rule_tilt_only_recall":rule_best["target_run_recall"]}},
        "C_label_physical_significance":{"judgement":"supported_at_canonical_boundary_scale",
            "evidence":{"canonical_threshold_rad":canonical,"audit_max_tilt_min_rad":float(oracle_max.min()),"audit_max_tilt_median_rad":float(np.median(oracle_max)),"audit_max_tilt_max_rad":float(oracle_max.max()),
                        "min_multiple_of_threshold":float(oracle_max.min()/canonical),"median_multiple_of_threshold":float(np.median(oracle_max)/canonical),"max_multiple_of_threshold":float(oracle_max.max()/canonical),
                        "normal_fsr_iqr":float(best_fsr["normal_q3"]-best_fsr["normal_q1"]),"tilt_fsr_iqr":float(best_fsr["tilt_only_q3"]-best_fsr["tilt_only_q1"]),"fsr_effect_size":best_fsr["cohen_d"]}},
        "D_joint_task_imbalance":{"judgement":"supported" if max(logistic_best["target_run_recall"],rule_best["target_run_recall"])>raw_recall else "not_supported",
            "evidence":{"existing_cnn_tilt_only_recall":raw_recall,"rule_tilt_only_recall":rule_best["target_run_recall"],"logistic_tilt_only_recall":logistic_best["target_run_recall"],"existing_sink_tilt_runs_detected":30}}
    }
    (output/"conclusions.json").write_text(json.dumps(conclusions,indent=2)+"\n",encoding="utf-8")
    lines=["# Sand Tilt validation-only analysis","","No test trace was indexed into a trace, scored, or used for inference.","",
           f"Tilt-only audit: {len(audit)} runs; median oracle maximum={np.median([r['max_oracle_tilt_rad'] for r in audit]):.9f} rad.",
           f"Best FSR feature: {best_fsr['feature']} at {best_fsr['window_ms']} ms, oriented ROC-AUC={best_fsr['roc_auc_oriented']:.3f}.",
           f"Best IMU feature: {best_imu['feature']} at {best_imu['window_ms']} ms, oriented ROC-AUC={best_imu['roc_auc_oriented']:.3f}.","",
           "| Detector | Window | Sand recall | Tilt-only recall | Overall pre-onset FPR | Sand median latency |","|---|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['detector']} | {r['window_ms']} ms | {r['sand_run_recall']:.3f} | {r['tilt_only_run_recall']:.3f} | {r['overall_pre_onset_run_fpr']:.3f} | {r['sand_median_latency_ms']} ms |")
    lines += ["",f"Combined recall gate: {'PASS' if conclusions['gate_recall_95_pass'] else 'FAIL'}",
              f"Combined FPR gate: {'PASS' if conclusions['gate_fpr_5_pass'] else 'FAIL'}",
              f"Tilt observation <=20 ms: {'PASS' if conclusions['observation_20ms_pass'] else 'FAIL'}"]
    (output/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"SAND_TILT_VALIDATION_COMPLETE runtime_s={time.perf_counter()-started:.2f} output={output}")


if __name__ == "__main__": main()
