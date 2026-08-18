"""Bounded-memory evaluation/finalization for an already trained v2 candidate."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from run_terrain_transition_aware_v2 import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--int8-model',type=Path,required=True,help='candidate-specific strict INT8 flatbuffer');p.add_argument('--report',type=Path);p.add_argument('--transition-split',choices=tuple(FAMILIES),default='validation');a=p.parse_args();out=a.output_dir.resolve();int8=a.int8_model.resolve()
 with (out/'split_manifest.csv').open() as f: rows=list(csv.DictReader(f))
 with np.load(out/'transition_runs.npz') as z: traces=z['fusion10']
 tx,ty,ts,_=transition_windows(traces,rows)
 with np.load(STATIC) as z:sx,sy,ss,sf=z['X'],z['y'],z['split'],z['surface_family']
 x=np.concatenate((sx,tx));y=np.concatenate((sy,ty));split=np.concatenate((ss,ts)); train=split=='train';test=split=='test'
 norm=ChannelNormalizer.fit(x[train]);xn=norm.transform(x);model=tf.keras.models.load_model(out/'selected_model.keras',compile=False)
 static={}
 for name,mask in (("validation",(split=='validation')&(np.arange(len(x))<len(sx))),("test",test&(np.arange(len(x))<len(sx)))):
  p=np.argmax(model.predict(xn[mask],batch_size=512,verbose=0),1);static[name]={"accuracy":float((p==y[mask]).mean()),"macro_f1":macro_f1(y[mask],p)}
 mask=test & (np.arange(len(x)) < len(sx));ip=np.argmax(predict_tflite(int8,xn[mask]),1);fp=np.argmax(model.predict(xn[mask],verbose=0),1);intstatic={"accuracy":float((ip==y[mask]).mean()),"macro_f1":macro_f1(y[mask],ip),"float_agreement":float((ip==fp).mean()),"model_size_bytes":int8.stat().st_size,"sha256":hashlib.sha256(int8.read_bytes()).hexdigest()}
 selected_rows=[r for r in rows if r['split']==a.transition_split];selected_traces=traces[[i for i,r in enumerate(rows) if r['split']==a.transition_split]]
 trans={a.transition_split:aggregate(transition_metrics_int8(int8,norm,selected_traces,selected_rows))}
 gates={c:trans[a.transition_split][c]['detection_rate']>=.9 and trans[a.transition_split][c]['occupancy_mean']>=.8 for c in CASES};ready=all(gates.values()) and intstatic['accuracy']>=.96098
 summary={"static":static,"int8_static":intstatic,"transition":trans,"transition_split":a.transition_split,"gates":gates,"STATIC_RETENTION_GATE":intstatic['accuracy']>=.96098,"TRANSITION_VALIDATION_GATE":all(gates.values()),"TERRAIN_TRANSITION_AWARE_V2_READY":ready,"TERRAIN_TRANSITION_AWARE_V2_INT8_READY":ready,"TERRAIN_ARCHITECTURE_CHANGE_NEEDED":not ready}
 report=a.report.resolve() if a.report else out/'int8_evaluation.json';report.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
