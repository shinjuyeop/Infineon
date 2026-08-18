"""Bounded-memory evaluation/finalization for an already trained v2 candidate."""
import csv,json
from pathlib import Path
import numpy as np
import tensorflow as tf
from run_terrain_transition_aware_v2 import *

def main():
 out=OUT
 with (out/'split_manifest.csv').open() as f: rows=list(csv.DictReader(f))
 with np.load(out/'transition_runs.npz') as z: traces=z['fusion10']
 tx,ty,ts,_=transition_windows(traces,rows)
 with np.load(STATIC) as z:sx,sy,ss,sf=z['X'],z['y'],z['split'],z['surface_family']
 x=np.concatenate((sx,tx));y=np.concatenate((sy,ty));split=np.concatenate((ss,ts)); train=split=='train';test=split=='test'
 norm=ChannelNormalizer.fit(x[train]);xn=norm.transform(x);model=tf.keras.models.load_model(out/'selected_model.keras',compile=False)
 static={}
 for name,mask in (("validation",(split=='validation')&(np.arange(len(x))<len(sx))),("test",test&(np.arange(len(x))<len(sx)))):
  p=np.argmax(model.predict(xn[mask],batch_size=512,verbose=0),1);static[name]={"accuracy":float((p==y[mask]).mean()),"macro_f1":macro_f1(y[mask],p)}
 int8=out/'int8/model_int8.tflite';mask=test & (np.arange(len(x)) < len(sx));ip=np.argmax(predict_tflite(int8,xn[mask]),1);fp=np.argmax(model.predict(xn[mask],verbose=0),1);intstatic={"accuracy":float((ip==y[mask]).mean()),"macro_f1":macro_f1(y[mask],ip),"float_agreement":float((ip==fp).mean()),"model_size_bytes":int8.stat().st_size}
 trans={s:aggregate(transition_metrics(model,norm,traces[[i for i,r in enumerate(rows) if r['split']==s]],[r for r in rows if r['split']==s])) for s in FAMILIES}
 with np.load(SIM/'outputs/terrain_transition_v1_pilot/transition_traces.npz') as z: dx=z['fusion10']
 with (SIM/'outputs/terrain_transition_v1_pilot/manifest.csv').open() as f:dr=list(csv.DictReader(f))
 diag=aggregate(transition_metrics(model,norm,dx,dr));gates={c:trans['validation'][c]['detection_rate']>=.9 and trans['validation'][c]['occupancy_mean']>=.8 for c in CASES};ready=all(gates.values()) and intstatic['accuracy']>=.96098
 summary={"static":static,"int8_static":intstatic,"transition":trans,"diagnostic_v2":diag,"gates":gates,"TERRAIN_TRANSITION_AWARE_V2_READY":ready,"TERRAIN_TRANSITION_AWARE_V2_INT8_READY":ready,"TERRAIN_ARCHITECTURE_CHANGE_NEEDED":not ready}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'normalization.json').write_text(json.dumps(norm.as_dict(),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
