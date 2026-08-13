"""Build/train frozen-scope Fast Reflex v2 causal binary detector artifacts."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

from terrain_fast_reflex_v2_detector import WINDOWS, save_dataset

DEFAULT_SOURCE=Path("../../outputs/terrain_fast_reflex_v2_final_scope_full")
DEFAULT_DATASET=Path("../../outputs/terrain_fast_reflex_v2_detector_dataset")
DEFAULT_OUTPUT=Path("../../outputs/terrain_fast_reflex_v2_detector_smoke")

def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("build","train","summary"));p.add_argument("--source",type=Path,default=DEFAULT_SOURCE);p.add_argument("--dataset-dir",type=Path,default=DEFAULT_DATASET);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--detector",choices=tuple(WINDOWS));p.add_argument("--windows-ms",nargs="+",type=int);p.add_argument("--epochs",type=int,default=40);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--seed",type=int,default=20260812);p.add_argument("--smoke",action="store_true");p.add_argument("--execute",action="store_true");return p.parse_args()

def model(window:int,pooling:str,seed:int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed); inputs=tf.keras.Input((window,10)); x=tf.keras.layers.Conv1D(12,5,padding="same",activation="relu")(inputs);x=tf.keras.layers.Conv1D(16,3,padding="same",activation="relu")(x)
    if pooling=="average_max": x=tf.keras.layers.Concatenate()([tf.keras.layers.GlobalAveragePooling1D()(x),tf.keras.layers.GlobalMaxPooling1D()(x)])
    else: x=tf.keras.layers.GlobalAveragePooling1D()(x)
    out=tf.keras.layers.Dense(1,activation="sigmoid")(x);m=tf.keras.Model(inputs,out);m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy");return m

def metrics(y:np.ndarray,s:np.ndarray)->dict[str,float]:
    pred=s>=.5;tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
    return {"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"f1":2*tp/max(2*tp+fp+fn,1),"endpoint_fpr":fp/max(fp+tn,1)}

def replay_foundation(data:dict[str,np.ndarray],scores:np.ndarray)->dict[str,int]:
    stable={};
    for run in np.unique(data["run_id"]):
        q=scores[data["run_id"]==run];stable[str(run)]=int(np.any(np.convolve((q>=.5).astype(int),np.ones(3,dtype=int),"valid")>=3))
    return {"runs":len(stable),"stable_firing_runs":sum(stable.values()),"persistence_samples":3,"note":"endpoint labels remain causal; pre-onset firing is future early-warning diagnostic"}

def main() -> None:
    a=args()
    if not a.execute: print("Dry run only. Use --execute; final test is never read.");return
    if a.action=="build":
        out=a.dataset_dir.resolve()
        if out.exists() and any(out.iterdir()):raise FileExistsError(f"refusing non-empty {out}")
        report=save_dataset(out,a.source.resolve());print(f"V2_DETECTOR_DATASET_COMPLETE datasets={len(report['datasets'])} output={out}");return
    if a.action=="summary":
        result=a.output_dir.resolve()/"smoke_results.json"
        d=json.loads(result.read_text() if result.exists() else (a.dataset_dir.resolve()/"dataset_statistics.json").read_text());print(json.dumps(d,indent=2));return
    if not a.detector:raise ValueError("--detector is required for train")
    windows=a.windows_ms or list(WINDOWS[a.detector])
    if any(w not in WINDOWS[a.detector] for w in windows):raise ValueError("unsupported detector window")
    out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f"refusing non-empty {out}")
    out.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); results=[]
    for w in windows:
        tr=np.load(a.dataset_dir.resolve()/f"{a.detector}_{w}ms/train.npz",allow_pickle=False);va=np.load(a.dataset_dir.resolve()/f"{a.detector}_{w}ms/validation.npz",allow_pickle=False)
        tx,ty,vx,vy=tr["x"],tr["y"],va["x"],va["y"]
        if a.smoke:
            chosen=np.r_[np.flatnonzero(ty==0)[:256],np.flatnonzero(ty==1)[:256]]
            tx,ty=tx[chosen],ty[chosen]
        counts=np.bincount(ty,minlength=2)
        if not np.all(counts): raise ValueError("training windows must contain both classes")
        class_weight={0:len(ty)/(2*counts[0]),1:len(ty)/(2*counts[1])}
        m=model(w,"average_max" if a.detector=="slip" else "average",a.seed);hist=m.fit(tx,ty,class_weight=class_weight,validation_data=(vx,vy),epochs=1 if a.smoke else a.epochs,batch_size=a.batch_size,verbose=0)
        score=m.predict(vx,verbose=0).reshape(-1); directory=out/f"{a.detector}_{w}ms";directory.mkdir();model_path=directory/"model.keras";m.save(model_path)
        import tensorflow as tf
        loaded=tf.keras.models.load_model(model_path)
        result={"detector":a.detector,"window_ms":w,"pooling":"average_max" if a.detector=="slip" else "average","parameters":m.count_params(),"epochs":len(hist.history["loss"]),"train_class_weight":class_weight,"loss_finite":bool(np.isfinite(hist.history["loss"]).all()),"model_save_load_verified":loaded.input_shape == m.input_shape,"validation":metrics(vy,score),"replay_foundation":replay_foundation({k:va[k] for k in va.files},score)};results.append(result)
    (out/"smoke_results.json").write_text(json.dumps({"source_dataset":str(a.dataset_dir.resolve()),"threshold":"fixed 0.5 smoke only; no selection/optimization","results":results,"runtime_s":time.perf_counter()-started},indent=2)+"\n");print(f"V2_DETECTOR_TRAIN_COMPLETE jobs={len(results)} output={out}")
if __name__=="__main__":main()
