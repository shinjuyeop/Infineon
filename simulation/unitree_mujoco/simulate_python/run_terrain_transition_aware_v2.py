"""Train the unchanged 1-kHz CNN on static plus held-out transition runs."""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

from run_terrain_transition import CASES, run_one
from terrain_cnn import ChannelNormalizer, build_compact_1d_cnn
from terrain_int8 import export_full_int8_tflite, normalize, predict_tflite

SIM=Path(__file__).resolve().parents[2]
STATIC=SIM/"outputs/terrain_dataset_v1_expanded_1000hz_full/dataset_noisy.npz"
OUT=SIM/"outputs/terrain_transition_aware_v2"
OLD_MODEL=SIM/"outputs/terrain_dataset_v1_expanded_1000hz_int8_seed_20260807/noisy_fusion_int8.tflite"
OLD_META=SIM/"outputs/terrain_dataset_v1_expanded_1000hz_int8_seed_20260807/deployment_metadata.json"
FAMILIES={"train":("multisine","filtered_random","sparse_aggregate"),"validation":("crosshatch","rounded_ridges"),"test":("warped_multisine","smooth_random_patches")}
LABEL={"concrete":0,"marble":1,"ice":2,"sand":3}; SEEDS=(20260821,20260822,20260823); T0=650

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def macro_f1(y,p):
    vals=[]
    for k in range(4):
        tp=((y==k)&(p==k)).sum(); fp=((y!=k)&(p==k)).sum(); fn=((y==k)&(p!=k)).sum()
        vals.append(2*tp/max(2*tp+fp+fn,1))
    return float(np.mean(vals))
def stable(pred,target):
    n=0
    for i in range(T0,len(pred)):
        n=n+1 if pred[i]==target else 0
        if n==3:return i
    return None
def transition_metrics(model, norm, traces, rows):
    # Keep the large run-level validation/test evaluation memory bounded.
    full=np.full((len(traces),800),-1,np.int8)
    for i,x in enumerate(traces):
        windows=np.asarray([x[e-49:e+1] for e in range(49,800)],np.float32)
        full[i,49:]=np.argmax(model.predict(norm.transform(windows),batch_size=256,verbose=0),1)
    result=[]
    for i,r in enumerate(rows):
        target=LABEL[r['terrain_after']];t1=stable(full[i],target);post=full[i,T0:]
        result.append({"run_id":r['run_id'],"case_id":r['case_id'],"split":r.get('split','diagnostic'),"detected":t1 is not None,"t1_ms":None if t1 is None else t1-T0,"occupancy":float((post==target).mean()),"switches":int(np.count_nonzero(np.diff(post)!=0))})
    return result
def transition_metrics_int8(path, norm, traces, rows):
    """Evaluate a named strict-INT8 artifact without falling back to OUT/int8."""
    full=np.full((len(traces),800),-1,np.int8)
    for i,x in enumerate(traces):
        windows=np.asarray([x[e-49:e+1] for e in range(49,800)],np.float32)
        full[i,49:]=np.argmax(predict_tflite(path,norm.transform(windows)),1)
    result=[]
    for i,r in enumerate(rows):
        target=LABEL[r['terrain_after']];t1=stable(full[i],target);post=full[i,T0:]
        result.append({"run_id":r['run_id'],"case_id":r['case_id'],"split":r.get('split','diagnostic'),"detected":t1 is not None,"t1_ms":None if t1 is None else t1-T0,"occupancy":float((post==target).mean()),"switches":int(np.count_nonzero(np.diff(post)!=0))})
    return result
def aggregate(rows):
    out={}
    for c in CASES:
        g=[r for r in rows if r['case_id']==c];lat=[r['t1_ms'] for r in g if r['t1_ms'] is not None]
        out[c]={"runs":len(g),"detected":sum(r['detected'] for r in g),"detection_rate":sum(r['detected'] for r in g)/len(g),"latency_median_ms":None if not lat else float(np.median(lat)),"latency_p95_ms":None if not lat else float(np.percentile(lat,95)),"occupancy_mean":float(np.mean([r['occupancy'] for r in g])),"switches_total":sum(r['switches'] for r in g)}
    return out
def generate(out):
    rows=[]; traces=[]
    # 336 new runs: each direction has 36/24/24 train/validation/test runs.
    for split,fams in FAMILIES.items():
        for family in fams:
            for surface in range(3):
                for case in CASES:
                    for run in range(4):
                        item=run_one(case,run,family,surface); r={**item.metadata,"split":split};rows.append(r);traces.append(item.fusion10)
    np.savez_compressed(out/'transition_runs.npz',fusion10=np.asarray(traces,np.float32),run_id=np.asarray([r['run_id'] for r in rows]),terrain_before=np.asarray([r['terrain_before'] for r in rows]),terrain_after=np.asarray([r['terrain_after'] for r in rows]))
    with (out/'split_manifest.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows,np.asarray(traces,np.float32)
def transition_windows(traces,rows):
    # Deterministic causal regions: 20 pre, 20 boundary, 20 early-post samples.
    ends=np.r_[np.arange(550,650,5),np.arange(650,700,2),np.arange(700,800,5)]
    x=[];y=[];s=[];rid=[]
    for i,r in enumerate(rows):
        x.append(traces[i,ends-49][:,None,:]+np.zeros((len(ends),50,10),np.float32))
        # replace broadcast placeholder with causal windows; explicit endpoint labels.
        x[-1]=np.asarray([traces[i,e-49:e+1] for e in ends],np.float32)
        y.extend([LABEL[r['terrain_before'] if e<T0 else r['terrain_after']] for e in ends]);s.extend([r['split']]*len(ends));rid.extend([r['run_id']]*len(ends))
    return np.concatenate(x),np.asarray(y),np.asarray(s),np.asarray(rid)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true',help='resume after a generated-run artifact; never regenerates physics');a=p.parse_args();out=a.output_dir.resolve()
 protocol={"dataset":"terrain_transition_aware_v2","runs":336,"directions":{k:84 for k in CASES},"families":FAMILIES,"split_unit":"family + surface realization + source run","window_label":"label(t)=terrain_gt(t), causal [t-49,t]","sampling":"20 pre + 20 boundary + 20 early/late post endpoints per run","architecture":"Conv1D(12,k=5)-Conv1D(16,k=3)-GlobalAveragePooling-Dense(4)","seeds":SEEDS,"selection":"max validation transition detection rate, then occupancy, then static validation accuracy","gates":{"each_direction_detection_rate":.9,"each_direction_occupancy":.8,"static_accuracy_drop_pp":1.0},"diagnostic_benchmark_excluded":str(SIM/'outputs/terrain_transition_v1_pilot')}
 if not a.execute:print(json.dumps(protocol,indent=2));return
 if out.exists() and any(out.iterdir()) and not a.resume:raise FileExistsError(out)
 if a.resume:
  with (out/'split_manifest.csv').open() as f:rows=list(csv.DictReader(f))
  with np.load(out/'transition_runs.npz') as z:traces=z['fusion10']
 else:
  out.mkdir(parents=True);rows,traces=generate(out)
 tx,ty,ts,trid=transition_windows(traces,rows)
 with np.load(STATIC) as z:sx,sy,ss,sf=z['X'],z['y'],z['split'],z['surface_family']
 x=np.concatenate((sx,tx));y=np.concatenate((sy,ty));split=np.concatenate((ss,ts));family=np.concatenate((sf,np.asarray([r['surface_family'] for r in rows for _ in range(60)])))
 if set(trid[split[len(sx):]=='train']).intersection(set(trid[split[len(sx):]=='validation'])):raise ValueError('transition run leakage')
 train=split=='train';val=split=='validation';test=split=='test';norm=ChannelNormalizer.fit(x[train]);xn=norm.transform(x)
 # Predeclared data-level ablation: keep every static train item and draw an
 # equal number of transition items, stratified by endpoint class. This avoids
 # the new transition corpus overwhelming the established static domain.
 static_train=np.flatnonzero(train & (np.arange(len(x)) < len(sx))); transition_train=np.flatnonzero(train & (np.arange(len(x)) >= len(sx)))
 rng_fit=np.random.default_rng(20260821); labels_present=sorted(set(y[transition_train].tolist())); per=max(1,len(static_train)//len(labels_present)); fit_indices=np.concatenate((static_train,*[rng_fit.choice(transition_train[y[transition_train]==k],per,replace=False) for k in labels_present]))
 import tensorflow as tf
 candidates=[]
 for seed in SEEDS:
  tf.keras.backend.clear_session();m=build_compact_1d_cnn(10,seed);m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  counts=np.bincount(y[fit_indices],minlength=4);weights=np.asarray([len(y[fit_indices])/(4*counts[k]) for k in y[fit_indices]],np.float32)
  print(f"training seed={seed}",flush=True)
  h=m.fit(xn[fit_indices],y[fit_indices],sample_weight=weights,validation_data=(xn[val],y[val]),epochs=60,batch_size=128,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)],verbose=2)
  path=out/f'candidate_seed_{seed}.keras';m.save(path);vi=[i for i,r in enumerate(rows) if r['split']=='validation'];tm=aggregate(transition_metrics(m,norm,traces[vi],[rows[i] for i in vi]))
  sp=np.argmax(m.predict(xn[val & (np.arange(len(x))<len(sx))],verbose=0),1);syv=y[val & (np.arange(len(x))<len(sx))];candidates.append({"seed":seed,"path":str(path),"epochs":len(h.history['loss']),"transition":tm,"static_validation_accuracy":float((sp==syv).mean()),"static_validation_macro_f1":macro_f1(syv,sp)})
 def key(c):return (min(v['detection_rate'] for v in c['transition'].values()),np.mean([v['occupancy_mean'] for v in c['transition'].values()]),c['static_validation_accuracy'])
 selected=max(candidates,key=key);model=tf.keras.models.load_model(selected['path'],compile=False);model.save(out/'selected_model.keras')
 static={}
 for name,mask in (("validation",val & (np.arange(len(x))<len(sx))),("test",test & (np.arange(len(x))<len(sx)))):
  pp=np.argmax(model.predict(xn[mask],verbose=0),1);static[name]={"accuracy":float((pp==y[mask]).mean()),"macro_f1":macro_f1(y[mask],pp)}
 trans={k:aggregate(transition_metrics(model,norm,traces[[i for i,r in enumerate(rows) if r['split']==k]],[r for r in rows if r['split']==k])) for k in FAMILIES}
 # strict INT8 calibration is a deterministic train-only stratified subset.
 rng=np.random.default_rng(20260821);chosen=np.concatenate([rng.choice(np.flatnonzero(train & (y==k)),min(64,int((train&(y==k)).sum())),replace=False) for k in range(4)])
 int8=out/'int8';int8.mkdir();inp,outp=export_full_int8_tflite(model,xn[chosen],int8/'model_int8.tflite')
 static_test=test & (np.arange(len(x))<len(sx));intp=np.argmax(predict_tflite(int8/'model_int8.tflite',xn[static_test]),1);floatp=np.argmax(model.predict(xn[static_test],verbose=0),1);agreement=float((intp==floatp).mean());intstatic={"accuracy":float((intp==y[static_test]).mean()),"macro_f1":macro_f1(y[static_test],intp),"float_agreement":agreement}
 # diagnostic pilot read-only, old/new terrain comparison.
 with np.load(SIM/'outputs/terrain_transition_v1_pilot/transition_traces.npz') as z:dx=z['fusion10']; did=z['run_id'].astype(str);dgt=z['terrain_gt']
 with (SIM/'outputs/terrain_transition_v1_pilot/manifest.csv').open() as f:dr=list(csv.DictReader(f))
 diag=aggregate(transition_metrics(model,norm,dx,dr)); oldmeta=json.loads(OLD_META.read_text()); oldnorm=ChannelNormalizer(np.asarray(oldmeta['normalization']['mean'],np.float32),np.asarray(oldmeta['normalization']['std'],np.float32))
 # minimal old/new plot uses endpoint class predictions.
 def pfull(path,nm):
  w=np.asarray([t[e-49:e+1] for t in dx for e in range(49,800)],np.float32);q=np.argmax(predict_tflite(path,normalize(w,nm.mean,nm.std)),1).reshape(12,751);o=np.full((12,800),-1);o[:,49:]=q;return o
 old=pfull(OLD_MODEL,oldnorm);new=np.full((12,800),-1);w=np.asarray([t[e-49:e+1] for t in dx for e in range(49,800)],np.float32);new[:,49:]=np.argmax(predict_tflite(int8/'model_int8.tflite',norm.transform(w)),1).reshape(12,751)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plots=out/'plots';plots.mkdir();
 for case in ('A','C'):
  i=next(j for j,r in enumerate(dr) if r['case_id']==case);fig,ax=plt.subplots(2,1,sharex=True,figsize=(10,4));ax[0].step(np.arange(800),old[i],where='post',label='old');ax[0].step(np.arange(800),new[i],where='post',label='new');ax[0].legend();ax[1].step(np.arange(800),(np.asarray(dgt[i])==dr[i]['terrain_after']).astype(int),where='post');[a.axvline(650,color='k',ls='--') for a in ax];fig.savefig(plots/f'case_{case.lower()}_old_new.png',dpi=150);plt.close(fig)
 gates={c:trans['validation'][c]['detection_rate']>=.9 and trans['validation'][c]['occupancy_mean']>=.8 for c in CASES};ready=all(gates.values()) and intstatic['accuracy']>=.96098
 summary={"protocol":protocol,"candidates":candidates,"selected":selected,"static":static,"int8_static":intstatic,"transition":trans,"diagnostic_v2":diag,"gates":gates,"TERRAIN_TRANSITION_AWARE_V2_READY":ready,"TERRAIN_TRANSITION_AWARE_V2_INT8_READY":ready,"TERRAIN_ARCHITECTURE_CHANGE_NEEDED":not ready}
 (out/'dataset_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'normalization.json').write_text(json.dumps(norm.as_dict(),indent=2)+'\n');(int8/'manifest.json').write_text(json.dumps({"input":vars(inp),"output":vars(outp),"train_only_indices":chosen.tolist(),"sha256":sha(int8/'model_int8.tflite')},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
