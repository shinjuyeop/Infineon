"""Frozen walking-domain failure audit: homogeneous policy baseline plus phase analysis.

This is analysis-only.  It neither changes deployed models/calibration nor
overwrites the spatial-transition pilot.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,time
from pathlib import Path
import mujoco,numpy as np
from controlled_excitation import find_allowed_foot_geom_ids,has_nonfoot_floor_contact
from g1_upstream_locomotion import TESTED_POLICY_SHA256,UnitreeG1PretrainedController
from hil_sensor import G1HilSensorReader,HIL_SENSOR_CHANNELS
from run_terrain_transition import FROZEN_CALIBRATION,label
from run_terrain_transition_ai_replay import REFLEX
from run_terrain_causal_window_v4 import ChannelNormalizer,STATIC,load_transition,LABEL
from terrain_fast_reflex_system_v1 import Decision
from terrain_fast_reflex_v2_detector import Normalizer
from terrain_int8 import predict_tflite,predict_tflite_binary
from terrain_profiles import TERRAIN_PROFILES,apply_terrain_profile

SIM=Path(__file__).resolve().parents[2]; SCENE=SIM/'unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml'
PILOT=SIM/'outputs/walking_terrain_transition_v1_pilot'; OUT=SIM/'outputs/walking_domain_failure_audit_v1'
POLICY=SIM/'unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx'; DT=.0005;BOUNDARY=.25
PHASES=('AIR','TOUCHDOWN','LOADING','MID_STANCE','PUSH_OFF')

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--policy-path',type=Path,default=POLICY);p.add_argument('--runs-per-terrain',type=int,default=3);p.add_argument('--duration-s',type=float,default=3);p.add_argument('--walking-speed',type=float,default=.2);p.add_argument('--execute',action='store_true');return p.parse_args()
def phase(force):
 """Deterministic FSR-only phase: load >=5 N; 10/20 ms edge regions."""
 load=np.asarray(force)>=5; out=np.full(len(load),'AIR',dtype='<U10');i=0
 while i<len(load):
  if not load[i]:i+=1;continue
  j=i
  while j<len(load) and load[j]:j+=1
  n=j-i;out[i:min(i+10,j)]='TOUCHDOWN';out[min(i+10,j):min(i+30,j)]='LOADING';out[max(i+30,j-10):j]='PUSH_OFF'
  if n>40:out[i+30:j-10]='MID_STANCE'
  i=j
 return out
def contacts(model,data,ids):
 return any(int(c.geom1) in ids or int(c.geom2) in ids for c in data.contact[:data.ncon])
def collect(terrain,index,policy,duration,speed):
 m=mujoco.MjModel.from_xml_path(str(SCENE));m.opt.timestep=DT; a,b=m.geom('ground_source').id,m.geom('ground_target').id
 for n in ('ground_source','ground_target'):apply_terrain_profile(m,TERRAIN_PROFILES[terrain],n)
 if terrain=='sand':m.geom_solref[a]=m.geom_solref[b]=TERRAIN_PROFILES['concrete'].solref
 allowed=find_allowed_foot_geom_ids(m)
 for g in range(m.ngeom):
  if g not in (a,b) and g not in allowed:m.geom_contype[g]=m.geom_conaffinity[g]=0
 d=mujoco.MjData(m);ctl=UnitreeG1PretrainedController(m,d,policy,speed); reader=G1HilSensorReader(m,d);left=frozenset(reader.left_foot_geom_ids);right=frozenset();pelvis=m.body('pelvis').id;foot=m.body('left_ankle_roll_link').id;vel=np.zeros(6);initial=float(d.qpos[0]); rows=[];fall=None;reason='';minz=9
 for step in range(1,int(duration/DT)+1):
  ctl.apply();mujoco.mj_step(m,d);ctl.update_after_step()
  if step%2:continue
  sensor=reader.read_vector();mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,foot,vel,0);fv=vel[3:].copy();mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,vel,0);pv=vel[3:].copy();bad=has_nonfoot_floor_contact(d,a,allowed) or has_nonfoot_floor_contact(d,b,allowed);minz=min(minz,float(d.qpos[2]));now= d.qpos[2]<.55 or d.xmat[pelvis,8]<.55 or bad
  if now and fall is None:fall=len(rows);reason='nonfoot_surface_contact' if bad else 'pelvis_instability'
  rows.append((d.time,sensor,contacts(m,d,left),contacts(m,d,right),d.xpos[foot].copy(),fv,d.xpos[pelvis].copy(),pv,d.xquat[foot].copy(),bad))
 x=list(zip(*rows)); fusion=np.asarray(x[1]); force=fusion[:,:4].sum(1); ph=phase(force);oracle=np.zeros((len(rows),18));oracle[:,0]=x[2];oracle[:,1]=force;oracle[:,4:7]=np.asarray(x[5]);oracle[:,12]=np.asarray(x[4])[:,2];oracle[:,13]=np.linalg.norm(oracle[:,4:6],axis=1);oracle[:,14]=np.maximum(0,oracle[0,12]-oracle[:,12]);oracle[:,16]=np.asarray(x[2]);oracle[:,17]=np.asarray(x[2]);labs=label(oracle,0)
 return {'time_s':np.asarray(x[0]),'fusion10':fusion,'left_contact':np.asarray(x[2]),'right_contact':np.asarray(x[3]),'foot_xyz':np.asarray(x[4]),'foot_vel_xyz':np.asarray(x[5]),'pelvis_xyz':np.asarray(x[6]),'pelvis_vel_xyz':np.asarray(x[7]),'foot_quat':np.asarray(x[8]),'nonfoot':np.asarray(x[9]),'gait_phase':ph,'oracle':oracle,**labs},{'run_id':f'homogeneous_{terrain}_{index:02d}','terrain':terrain,'forward_displacement_m':float(d.qpos[0]-initial),'fall_occurred':fall is not None,'first_fall_sample':fall,'first_fall_time_s':None if fall is None else float(rows[fall][0]),'fall_reason':reason,'min_pelvis_height_m':minz,'left_contact_count':int(np.count_nonzero(np.diff(np.r_[False,np.asarray(x[2]),False])==1)),'right_contact_count':int(np.count_nonzero(np.diff(np.r_[False,np.asarray(x[3]),False])==1)),'sample_count':len(rows)}
def frozen(fusion):
 with np.load(STATIC) as z:sx,ss=z['X'],z['split']
 _,_,tx,_,ts=load_transition(50);norm=ChannelNormalizer.fit(np.concatenate((sx,tx))[np.concatenate((ss,ts))=='train']);n,t,_=fusion.shape;windows=np.asarray([q[e-49:e+1] for q in fusion for e in range(49,t)],np.float32);scores=predict_tflite(SIM/'outputs/terrain_causal_window_v4/int8/baseline_50_seed_20260823_strict_int8.tflite',norm.transform(windows)).reshape(n,t-49,4);pred=np.full((n,t),-1,np.int8);pred[:,49:]=scores.argmax(2);raw={};fire={}
 for k,c in REFLEX.items():
  normal=json.loads(c['normalization'].read_text()); w=c['window'];windows=np.asarray([q[e-w+1:e+1] for q in fusion for e in range(w-1,t)],np.float32);_,v=predict_tflite_binary(c['model'],Normalizer(np.asarray(normal['mean'],np.float32),np.asarray(normal['std'],np.float32)).transform(windows),w);raw[k]=np.full((n,t),-128,np.int8);raw[k][:,w-1:]=v.reshape(n,t-w+1);fire[k]=raw[k]>=c['raw_threshold']
 return scores,pred,raw,fire
def phase_rows(run_ids,terrain,phases,pred,raw,fire,slip,sink):
 rows=[]
 for i,rid in enumerate(run_ids):
  for p in PHASES:
   ix=phases[i]==p
   for kind,oracle in (('terrain',None),('slip',slip[i]),('sink',sink[i])):
    if not ix.any():continue
    if kind=='terrain': rows.append({'run_id':rid,'terrain':terrain[i],'phase':p,'kind':kind,'samples':int(ix.sum()),'correct_rate':float(np.mean(pred[i,ix]==LABEL[terrain[i]])),'switches':int(np.count_nonzero(np.diff(pred[i,ix])!=0)),'raw_mean':float(np.mean(pred[i,ix])),'firing_count':'' ,'oracle_positive_count':''})
    else: rows.append({'run_id':rid,'terrain':terrain[i],'phase':p,'kind':kind,'samples':int(ix.sum()),'correct_rate':'','switches':'','raw_mean':float(np.mean(raw[kind][i,ix])),'firing_count':int(np.count_nonzero(fire[kind][i,ix])),'oracle_positive_count':int(np.count_nonzero(oracle[ix]))})
 return rows
def write_csv(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def plots(out,data,tr,pred,raw,fire):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 d=out/'plots';d.mkdir()
 # A/D terrain failures, preserving raw versus contact-valid interpretation.
 for case in ('A','D'):
  i=next(j for j,x in enumerate(tr['run_id'].astype(str)) if x.split('_')[1]==case);x=tr['time_s'][i]*1000;ph=phase(tr['fusion10'][i,:,:4].sum(1));fig,ax=plt.subplots(4,1,figsize=(11,8),sharex=True);ax[0].plot(x,tr['foot_xyz'][i,:,0]);ax[0].axhline(BOUNDARY,color='k',ls='--');ax[1].step(x,ph,where='post');ax[2].step(x,pred[len(data['run_id'])+i],where='post');ax[3].plot(x,tr['fusion10'][i,:,:4].sum(1));ax[3].set_xlabel('time [ms]');fig.tight_layout();fig.savefig(d/f'terrain_case_{case.lower()}.png',dpi=140);plt.close(fig)
 # Normal Marble detector traces, including frozen oracle and raw thresholds.
 i=0;x=data['time_s'][i]*1000
 for kind,oracle in (('slip',data['confirmed_slip'][i]),('sink',data['sustained_sink'][i])):
  fig,ax=plt.subplots(4,1,figsize=(11,8),sharex=True);ax[0].step(x,data['gait_phase'][i],where='post');ax[1].plot(x,data['fusion10'][i,:,:4]);ax[2].plot(x,data['fusion10'][i,:,4:]);ax[3].plot(x,raw[kind][i]);ax[3].axhline(REFLEX[kind]['raw_threshold'],color='r');ax[3].step(x,fire[kind][i]*127,where='post');ax[3].step(x,oracle*120,where='post');ax[3].set_xlabel('time [ms]');fig.tight_layout();fig.savefig(d/f'{kind}_marble_phase_audit.png',dpi=140);plt.close(fig)
def main():
 a=parse();protocol={'frozen':True,'no_retraining_or_retuning':True,'phase_rule':'AIR FSR<5N; TOUCHDOWN first 10 loaded ms; LOADING next 20 ms; PUSH_OFF final 10 loaded ms; remainder MID_STANCE','adapter':'offline only: update terrain state only in loaded-contact endpoints; AIR holds last loaded state','homogeneous_terrains':['marble','ice','sand','concrete']}
 if not a.execute:print(json.dumps(protocol,indent=2));return
 out=a.output_dir.resolve();
 if out.exists() and any(out.iterdir()):raise FileExistsError(out)
 if hashlib.sha256(a.policy_path.read_bytes()).hexdigest()!=TESTED_POLICY_SHA256:raise ValueError('unverified policy')
 out.mkdir(parents=True);runs=[];meta=[]
 for terrain in protocol['homogeneous_terrains']:
  for i in range(a.runs_per_terrain):x,m=collect(terrain,i,a.policy_path,a.duration_s,a.walking_speed);runs.append(x);meta.append(m);print(m)
 keys=runs[0];data={k:np.asarray([x[k] for x in runs]) for k in keys};data['run_id']=np.asarray([m['run_id'] for m in meta]);np.savez_compressed(out/'homogeneous_traces.npz',**data);(out/'homogeneous_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 # Append read-only spatial traces for phase/replay comparison.
 with np.load(PILOT/'transition_traces.npz') as z:tr={k:z[k] for k in z.files}
 fusion=np.concatenate((data['fusion10'],tr['fusion10']));scores,pred,raw,fire=frozen(fusion); n=len(meta); allphase=np.concatenate((data['gait_phase'],np.asarray([phase(q[:,:4].sum(1)) for q in tr['fusion10']]))); names=list(data['run_id'].astype(str))+list(tr['run_id'].astype(str)); terrains=[m['terrain'] for m in meta]+[{'A':'ice','B':'sand','C':'marble','D':'marble'}[str(x).split('_')[1]] for x in tr['run_id']]
 prows=phase_rows(names,terrains,allphase,pred,raw,fire,np.concatenate((data['confirmed_slip'],tr['confirmed_slip'])),np.concatenate((data['sustained_sink'],tr['sustained_sink'])));write_csv(out/'terrain_phase_audit.csv',[x for x in prows if x['kind']=='terrain']);write_csv(out/'slip_phase_audit.csv',[x for x in prows if x['kind']=='slip']);write_csv(out/'sink_phase_audit.csv',[x for x in prows if x['kind']=='sink'])
 # Contact-valid counterfactual is intentionally parallel to raw state semantics.
 counter=[]
 for i,rid in enumerate(tr['run_id'].astype(str)):
  case=rid.split('_')[1];before={'A':'marble','B':'marble','C':'ice','D':'sand'}[case];raw_d=Decision(before);adapt_d=Decision(before);t0=int(np.asarray(json.loads((PILOT/'manifest.json').read_text()))[i]['T0']);
  for q in range(pred.shape[1]):
   stable=None if q<2 or pred[n+i,q]<0 or not np.all(pred[n+i,q-2:q+1]==pred[n+i,q]) else next(k for k,v in LABEL.items() if v==pred[n+i,q]);raw_state=raw_d.update(stable,bool(fire['slip'][n+i,q]),bool(fire['sink'][n+i,q]));adapt_state=adapt_d.update(stable if allphase[n+i,q]!='AIR' else None,bool(fire['slip'][n+i,q] and allphase[n+i,q]!='AIR'),bool(fire['sink'][n+i,q] and allphase[n+i,q]!='AIR'))
  counter.append({'run_id':rid,'raw_system_case':raw_state['transition_case'],'adapter_system_case':adapt_state['transition_case'],'raw_correct':raw_state['transition_case']==case,'adapter_correct':adapt_state['transition_case']==case,'raw_switches':int(np.count_nonzero(np.diff(pred[n+i,49:])!=0)),'adapter_air_samples':int(np.count_nonzero(allphase[n+i]=='AIR')),'T0':t0})
 write_csv(out/'adapter_counterfactual.csv',counter)
 # Candidate windows are manifests only; no training corpus is altered.
 cand=[]
 for i,rid in enumerate(data['run_id'].astype(str)):
  for p in PHASES:
   for kind,w in (('slip',5),('sink',20),('terrain',50)):
    for end in np.flatnonzero(data['gait_phase'][i]==p):
     # A deterministic 50-ms stride keeps this a bounded candidate manifest,
     # not an accidental copy of the full training corpus.
     if end>=w-1 and end%50==0:cand.append({'run_id':rid,'terrain':terrains[i],'phase':p,'detector':kind,'window_ms':w,'end_sample':int(end),'oracle_positive':bool(data['confirmed_slip'][i,end] if kind=='slip' else data['sustained_sink'][i,end] if kind=='sink' else False)})
 cdir=out/'candidate_hard_negatives';cdir.mkdir();(cdir/'manifest.json').write_text(json.dumps(cand,indent=2)+'\n')
 compare=[]
 for case,target in {'A':'ice','B':'sand','C':'marble','D':'marble'}.items():
  trans=[json.loads((PILOT/'manifest.json').read_text())[i] for i,r in enumerate(tr['run_id'].astype(str)) if r.split('_')[1]==case]; base=[m for m in meta if m['terrain']==target];compare.append({'case':case,'target':target,'transition_fall_count':sum(x['fall_occurred'] for x in trans),'baseline_fall_count':sum(x['fall_occurred'] for x in base),'classification':'TARGET_TERRAIN_POLICY_FAILURE' if all(x['fall_occurred'] for x in base) else 'MIXED_UNRESOLVED'})
 write_csv(out/'transition_vs_homogeneous.csv',compare);write_csv(out/'homogeneous_audit.csv',meta);write_csv(out/'fall_audit.csv',meta)
 # Reference-domain comparison is descriptive only; static provenance remains read-only.
 dist=[]
 with np.load(STATIC) as z:reference=z['X']
 for name,values in [('controlled_train',reference.reshape(-1,10)),('walking_all',fusion.reshape(-1,10))]:
  for c,namec in enumerate(HIL_SENSOR_CHANNELS):
   v=values[:,c];dist.append({'domain':name,'channel':namec,'mean':float(v.mean()),'std':float(v.std()),'median':float(np.median(v)),'p05':float(np.percentile(v,5)),'p95':float(np.percentile(v,95)),'min':float(v.min()),'max':float(v.max())})
 write_csv(out/'sensor_distribution_comparison.csv',dist);plots(out,data,tr,pred,raw,fire)
 summary={'homogeneous':{t:{'runs':sum(m['terrain']==t for m in meta),'falls':sum(m['terrain']==t and m['fall_occurred'] for m in meta),'forward_displacement_median':float(np.median([m['forward_displacement_m'] for m in meta if m['terrain']==t])),'first_fall_time_s':next((m['first_fall_time_s'] for m in meta if m['terrain']==t),None)} for t in protocol['homogeneous_terrains']},'terrain_adapter_accuracy':{'raw':sum(x['raw_correct'] for x in counter),'adapter':sum(x['adapter_correct'] for x in counter)},'TERRAIN_WALKING_STATUS':'RETRAINING_RECOMMENDED','SLIP_WALKING_STATUS':'HARD_NEGATIVE_RETRAINING_RECOMMENDED','SINK_WALKING_STATUS':'HARD_NEGATIVE_RETRAINING_RECOMMENDED','WALKING_DOMAIN_FAILURE_AUDIT_READY':True};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
