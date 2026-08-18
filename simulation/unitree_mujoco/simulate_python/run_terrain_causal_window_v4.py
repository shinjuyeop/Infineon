"""Float architecture-selection sweep for v4 causal-window ablation."""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

from build_terrain_static_provenance_v4 import OUT as STATIC_OUT
from run_terrain_transition_aware_v2 import CASES, LABEL, SEEDS, SIM, aggregate, macro_f1, stable
from terrain_cnn import ChannelNormalizer, build_compact_1d_cnn, estimate_model_macs, estimate_model_resources

STATIC = STATIC_OUT / "dataset_noisy_provenance.npz"; TRANSITION = SIM / "outputs/terrain_transition_aware_v2_1_80_20"; REFERENCE = SIM / "outputs/terrain_static_reference_v4/summary.json"; OUT = SIM / "outputs/terrain_causal_window_v4"
WINDOWS = (20, 30, 50); FRONTENDS = (("baseline", False), ("causal", True))

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def load_transition(length: int):
    with (TRANSITION / "split_manifest.csv").open() as f: rows=list(csv.DictReader(f))
    with np.load(TRANSITION / "transition_runs.npz") as z: traces=z["fusion10"]
    ends=np.r_[np.arange(550,650,5),np.arange(650,700,2),np.arange(700,800,5)]; x=[];y=[];split=[];run=[]
    for i,row in enumerate(rows):
        x.extend(traces[i,e-length+1:e+1] for e in ends); y.extend([LABEL[row['terrain_before'] if e<650 else row['terrain_after']] for e in ends]); split.extend([row['split']]*len(ends)); run.extend([row['run_id']]*len(ends))
    # Reserve original transition train run_index=3; original test/validation remain sealed/excluded.
    mapped=np.asarray(['architecture_selection' if s=='train' and int(r.rsplit('_',1)[1])==3 else 'train' if s=='train' else 'excluded' for s,r in zip(split,run)])
    return rows,traces,np.asarray(x,np.float32),np.asarray(y),mapped

def transition_metrics(model,norm,traces,rows,length):
    full=np.full((len(traces),800),-1,np.int8)
    for i,trace in enumerate(traces):
        windows=np.asarray([trace[e-length+1:e+1] for e in range(length-1,800)],np.float32);full[i,length-1:]=np.argmax(model.predict(norm.transform(windows),batch_size=256,verbose=0),1)
    result=[]
    for i,row in enumerate(rows):
        target=LABEL[row['terrain_after']]; t1=stable(full[i],target); post=full[i,650:]; result.append({'run_id':row['run_id'],'case_id':row['case_id'],'detected':t1 is not None,'t1_ms':None if t1 is None else t1-650,'occupancy':float((post==target).mean()),'switches':int(np.count_nonzero(np.diff(post)!=0))})
    return aggregate(result)

def gates(metrics): return {case:metrics[case]['detection_rate']>=.9 and metrics[case]['occupancy_mean']>=.8 for case in CASES}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    with np.load(STATIC) as z:sx,sy,ss=z['X'],z['y'],z['split']
    reference=json.loads(REFERENCE.read_text());static_threshold=reference['candidates'][0]['selection_accuracy']-.01
    out.mkdir(parents=True,exist_ok=True); candidates=[]
    for length in WINDOWS:
        rows,traces,tx,ty,ts=load_transition(length);select_rows=[i for i,row in enumerate(rows) if row['split']=='train' and int(row['run_index'])==3];train_rows=[i for i,row in enumerate(rows) if row['split']=='train' and int(row['run_index'])!=3]
        x=np.concatenate((sx[:,-length:,:],tx));y=np.concatenate((sy,ty));split=np.concatenate((ss,ts));static_train=np.flatnonzero((split=='train')&(np.arange(len(x))<len(sx)));trans_train=np.flatnonzero((split=='train')&(np.arange(len(x))>=len(sx)));count=round(len(static_train)*.225/.775);rng=np.random.default_rng(20260930+length);labels=sorted(set(y[trans_train].tolist()));per=count//len(labels);fit=np.concatenate((static_train,*[rng.choice(trans_train[y[trans_train]==k],per,replace=False) for k in labels]));norm=ChannelNormalizer.fit(x[split=='train']);xn=norm.transform(x);static_selection=(split=='architecture_selection')&(np.arange(len(x))<len(sx))
        for name,causal in FRONTENDS:
            for seed in SEEDS:
                import tensorflow as tf
                tf.keras.backend.clear_session();path=out/f'{name}_{length}_seed_{seed}.keras'
                if path.exists(): model=tf.keras.models.load_model(path,compile=False);epochs=None
                else:
                    model=build_compact_1d_cnn(10,seed,time_steps=length,causal=causal);model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy']);h=model.fit(xn[fit],y[fit],validation_data=(xn[static_selection],y[static_selection]),epochs=60,batch_size=128,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)],verbose=0);epochs=len(h.history['loss']);model.save(path)
                p=np.argmax(model.predict(xn[static_selection],verbose=0),1);tm=transition_metrics(model,norm,traces[select_rows],[rows[i] for i in select_rows],length);candidates.append({'window_ms':length,'front_end':name,'seed':seed,'path':str(path),'sha256':sha(path),'epochs':epochs,'static_selection_accuracy':float((p==y[static_selection]).mean()),'static_selection_macro_f1':macro_f1(y[static_selection],p),'transition_selection':tm,'static_gate_float':float((p==y[static_selection]).mean())>=static_threshold,'transition_gates':gates(tm),'parameters':estimate_model_resources(10,time_steps=length).parameters,'macs':estimate_model_macs(10,length)})
    summary={'static_reference_selection_accuracy':reference['candidates'][0]['selection_accuracy'],'static_float_threshold':static_threshold,'frozen_mixture':{'requested_static_fraction':.775,'global_inverse_frequency_weighting':False},'transition_reservation':{'train_runs':108,'selection_runs':36,'directions':{case:9 for case in CASES}},'candidates':candidates};(out/'architecture_selection.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
