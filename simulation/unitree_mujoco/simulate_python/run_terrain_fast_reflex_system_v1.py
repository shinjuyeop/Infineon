"""Read-only frozen 12-run Terrain v4 + Fast Reflex v2 system replay."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from run_terrain_causal_window_v4_diagnostic import SOURCE, load, reflex_replay
from run_terrain_causal_window_v4 import OUT, STATIC, ChannelNormalizer, load_transition, LABEL
from terrain_int8 import predict_tflite
from terrain_fast_reflex_system_v1 import Decision,case_for
SIM=Path(__file__).resolve().parents[2];OUTSYS=SIM/'outputs/terrain_fast_reflex_system_v1';T0=650
def main()->None:
 import argparse;p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUTSYS);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print('frozen read-only system replay; add --execute');return
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 with np.load(STATIC) as z:sx,ss=z['X'],z['split']
 _,_,tx,_,ts=load_transition(50);norm=ChannelNormalizer.fit(np.concatenate((sx,tx))[np.concatenate((ss,ts))=='train']);rows,data=load(SOURCE);model=Path(json.loads((OUT/'int8/summary.json').read_text())['manifest']['int8_path'])
 windows=np.asarray([x[e-49:e+1] for x in data['fusion10'] for e in range(49,800)],np.float32);terrain=np.full((12,800),-1,np.int8);terrain[:,49:]=np.argmax(predict_tflite(model,norm.transform(windows)),axis=1).reshape(12,751)
 raw={};fire={}
 for k in ('slip','sink'):raw[k],fire[k]=reflex_replay(data['fusion10'],k)
 result=[]
 for i,r in enumerate(rows):
  d=Decision(terrain_state=r['terrain_before']);events=[];last=None
  for t in range(800):
   stable_name=None
   if t>=T0 and terrain[i,t]>=0:
    c=int(terrain[i,t]);
    if t>=T0+2 and np.all(terrain[i,t-2:t+1]==c):stable_name=next(k for k,v in LABEL.items() if v==c)
   s=d.update(stable_name,bool(fire['slip'][i,t] and t>=T0),bool(fire['sink'][i,t] and t>=T0))
   if s!=last:events.append((t,s.copy()));last=s.copy()
  c=case_for(r['terrain_before'],r['terrain_after']);t1=next((t for t,s in events if s['transition_case']==c),None);t2=next((t for t in range(T0,800) if (data['confirmed_slip'][i,t] if c=='A' else data['sustained_sink'][i,t] if c=='B' else False)),None);t3=next((t for t,s in events if t>=T0 and (s['slip'] if c=='A' else s['sink'] if c=='B' else False)),None);dec=t1 if c in 'CD' else (None if t1 is None or t3 is None else max(t1,t3));final=events[-1][1];result.append({'run_id':r['run_id'],'case_gt':c,'T0':T0,'T1':t1,'T2':t2,'T3':t3,'T_DECISION':dec,'terrain_before':r['terrain_before'],'terrain_after':r['terrain_after'],'transition_case':final['transition_case'],'slip_oracle':bool(np.any(data['confirmed_slip'][i,T0:])),'sink_oracle':bool(np.any(data['sustained_sink'][i,T0:])),'slip_firing':bool(np.any(fire['slip'][i,T0:])),'sink_firing':bool(np.any(fire['sink'][i,T0:])),'hazard_reflex_required':any(s['hazard_reflex_required'] for _,s in events),'case_reflex_required':any(s['case_reflex_required'] for _,s in events),'recovery_required':final['recovery_required'],'unmatched_hazard':any(s['unmatched_hazard'] for _,s in events),'dual_hazard':any(s['dual_hazard'] for _,s in events),'state_changes':len(events)})
 a.output_dir.mkdir(parents=True);csv.DictWriter((a.output_dir/'timeline.csv').open('w',newline=''),fieldnames=list(result[0])).writeheader();
 with (a.output_dir/'timeline.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 cases={c:[x for x in result if x['case_gt']==c] for c in 'ABCD'};summary={'runs':12,'cases':{c:{'transition_correct':sum(x['transition_case']==c for x in g),'case_reflex_required':sum(x['case_reflex_required'] for x in g),'recovery_required':sum(x['recovery_required'] for x in g),'T1': [x['T1'] for x in g],'T2':[x['T2'] for x in g],'T3':[x['T3'] for x in g],'T_DECISION':[x['T_DECISION'] for x in g]} for c,g in cases.items()},'unmatched_hazard':sum(x['unmatched_hazard'] for x in result),'dual_hazard':sum(x['dual_hazard'] for x in result),'TERRAIN_FAST_REFLEX_SYSTEM_V1_READY':all(x['transition_case']==x['case_gt'] and (x['recovery_required'] if x['case_gt'] in 'CD' else True) for x in result)}
 (a.output_dir/'protocol.json').write_text(json.dumps({'frozen':True,'terrain':'v4 strict INT8 50ms persistence3','slip':'5ms raw121 persistence3','sink':'20ms raw124 persistence1','hard_terrain':['marble','concrete']},indent=2)+'\n');(a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output_dir/'case_matrix.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
