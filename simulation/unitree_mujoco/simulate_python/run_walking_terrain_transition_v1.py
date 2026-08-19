"""Collect fixed-material spatial-terrain walking traces and replay frozen v4/v2/System-v1."""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import mujoco
import numpy as np

from controlled_excitation import find_allowed_foot_geom_ids, has_nonfoot_floor_contact
from g1_upstream_locomotion import TESTED_POLICY_SHA256, UnitreeG1PretrainedController
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from run_terrain_transition import FROZEN_CALIBRATION, _foot_oracle, label
from run_terrain_transition_ai_replay import REFLEX, stable_endpoint
from run_terrain_causal_window_v4 import ChannelNormalizer, STATIC, load_transition, LABEL
from terrain_fast_reflex_system_v1 import Decision, case_for
from terrain_int8 import predict_tflite, predict_tflite_binary
from terrain_fast_reflex_v2_detector import Normalizer
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile

SIM = Path(__file__).resolve().parents[2]
SCENE = SIM / "unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml"
OUT = SIM / "outputs/walking_terrain_transition_v1"
POLICY = SIM / "unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
CASES = {"A": ("marble", "ice"), "B": ("marble", "sand"), "C": ("ice", "marble"), "D": ("sand", "marble")}
BOUNDARY_X = 0.25
DT = .0005; RATE = 1000; STEPS = 2; LOAD_N = 5.0

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",type=Path,default=OUT); p.add_argument("--policy-path",type=Path,default=POLICY); p.add_argument("--runs-per-case",type=int,default=3); p.add_argument("--duration-s",type=float,default=3.0); p.add_argument("--walking-speed",type=float,default=.20); p.add_argument("--execute",action="store_true"); return p.parse_args()

def contact_contribution(model,data, foot_ids, ground_ids):
 out=np.zeros(2); wrench=np.zeros(6)
 for i in range(data.ncon):
  c=data.contact[i]; pair=(int(c.geom1),int(c.geom2));
  for j,g in enumerate(ground_ids):
   if g in pair and (pair[0] in foot_ids or pair[1] in foot_ids):
    mujoco.mj_contactForce(model,data,i,wrench); out[j]+=max(0.,float(wrench[0]))
 return out

def run_one(case, index, policy, duration, speed):
 before,after=CASES[case]; model=mujoco.MjModel.from_xml_path(str(SCENE)); model.opt.timestep=DT
 # Reposition the fixed half-planes without changing either material in time.
 model.geom_pos[model.geom("ground_source").id,0]=BOUNDARY_X-3.25; model.geom_pos[model.geom("ground_target").id,0]=BOUNDARY_X+3.25
 source,target=model.geom("ground_source").id,model.geom("ground_target").id
 apply_terrain_profile(model,TERRAIN_PROFILES[before],"ground_source"); apply_terrain_profile(model,TERRAIN_PROFILES[after],"ground_target")
 # Frozen walking-support-v1 is the pre-existing Sand-only locomotion support factorization.
 if after=="sand": model.geom_solref[target]=TERRAIN_PROFILES["concrete"].solref
 if before=="sand": model.geom_solref[source]=TERRAIN_PROFILES["concrete"].solref
 # Reuse the already validated walking foundation's foot-spheres-only runtime
 # contact isolation; the source XML and controller are untouched.
 allowed_all=find_allowed_foot_geom_ids(model)
 for geom_id in range(model.ngeom):
  if geom_id in (source,target) or geom_id in allowed_all: continue
  model.geom_contype[geom_id]=0; model.geom_conaffinity[geom_id]=0
 data=mujoco.MjData(model); controller=UnitreeG1PretrainedController(model,data,policy,speed); reader=G1HilSensorReader(model,data)
 feet=frozenset(reader.left_foot_geom_ids); grounds=(source,target); allowed=allowed_all; body=model.body("left_ankle_roll_link").id; pelvis=model.body("pelvis").id; velocity=np.zeros(6); wrench=np.zeros(6)
 series={k:[] for k in ("time_s","fusion10","left_contact","loaded_contact","foot_xyz","foot_vel_xyz","pelvis_xyz","pelvis_vel_xyz","foot_quat","source_force_N","target_force_N","terrain_gt","boundary_cross","nonfoot")}
 start_x=float(data.qpos[0]); fell=False; fall_reason=""; min_height=9.; first_cross=None; t0=None; was_loaded=False; target_touchdowns=0
 for step in range(1,int(round(duration/DT))+1):
  controller.apply(); mujoco.mj_step(model,data); controller.update_after_step()
  if step%STEPS: continue
  sensor=reader.read_vector(); contact=reader.has_left_foot_contact(frozenset(grounds)); forces=contact_contribution(model,data,feet,grounds); loaded=bool(contact and sensor[:4].sum()>=LOAD_N)
  mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,velocity,0); fv=velocity[3:].copy(); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,velocity,0); pv=velocity[3:].copy()
  # A sole contact point, rather than the ankle-body origin, is the relevant
  # foot position at a finite sole/terrain seam.
  foot_x=float(data.xpos[body,0]); cross=bool(forces[1] > 0.0)
  if cross and first_cross is None:first_cross=len(series["time_s"])
  dominant="target" if forces[1]>forces[0] else "source"
  gt=after if loaded and dominant=="target" else before
  if loaded and dominant=="target" and not was_loaded:
   target_touchdowns+=1
   if t0 is None:t0=len(series["time_s"])
  was_loaded=loaded
  min_height=min(min_height,float(data.qpos[2])); bad=has_nonfoot_floor_contact(data,source,allowed) or has_nonfoot_floor_contact(data,target,allowed)
  if data.qpos[2]<.55 or data.xmat[pelvis,8]<.55 or bad: fell=True; fall_reason=fall_reason or ("nonfoot_surface_contact" if bad else "fall")
  for k,v in (("time_s",data.time),("fusion10",sensor),("left_contact",contact),("loaded_contact",loaded),("foot_xyz",data.xpos[body].copy()),("foot_vel_xyz",fv),("pelvis_xyz",data.xpos[pelvis].copy()),("pelvis_vel_xyz",pv),("foot_quat",data.xquat[body].copy()),("source_force_N",forces[0]),("target_force_N",forces[1]),("terrain_gt",gt),("boundary_cross",cross),("nonfoot",bad)):series[k].append(v)
 arr={k:np.asarray(v) for k,v in series.items()}; base=np.zeros((len(arr["time_s"]),18),dtype=float)
 base[:,0]=arr["left_contact"]; base[:,1]=arr["fusion10"][:,:4].sum(1); base[:,4:7]=arr["foot_vel_xyz"]
 base[:,12]=arr["foot_xyz"][:,2]; base[:,13]=np.linalg.norm(base[:,4:6],axis=1)
 ref=t0 if t0 is not None else 0; base[:,14]=np.maximum(0,base[ref,12]-base[:,12]); base[:,16]=arr["source_force_N"]>0; base[:,17]=arr["target_force_N"]>0
 arr["oracle"]=base; labs=label(base,ref); arr.update(labs)
 meta={"run_id":f"walking_{case}_{index:02d}","case_id":case,"terrain_before":before,"terrain_after":after,"boundary_x_m":BOUNDARY_X,"expected_transition_step":"first monitored left-foot loaded target touchdown","T_BOUNDARY_CROSS":first_cross,"T_TOUCHDOWN":t0,"T0":t0,"target_touchdowns":target_touchdowns,"forward_displacement_m":float(data.qpos[0]-start_x),"walking_completed":not fell,"fall_occurred":fell,"fall_reason":fall_reason,"min_base_height_m":min_height,"fusion10_finite":bool(np.isfinite(arr["fusion10"]).all()),"sample_count":len(arr["time_s"])}
 return arr,meta

def replay(fusion, metas):
 with np.load(STATIC) as z:sx,ss=z['X'],z['split']
 _,_,tx,_,ts=load_transition(50); norm=ChannelNormalizer.fit(np.concatenate((sx,tx))[np.concatenate((ss,ts))=='train'])
 model=SIM/"outputs/terrain_causal_window_v4/int8/baseline_50_seed_20260823_strict_int8.tflite"; n,t,_=fusion.shape; windows=np.asarray([x[e-49:e+1] for x in fusion for e in range(49,t)],np.float32); scores=predict_tflite(model,norm.transform(windows)).reshape(n,t-49,4); pred=np.full((n,t),-1,np.int8);pred[:,49:]=scores.argmax(2)
 raw,fire={},{}
 for kind,cfg in REFLEX.items():
  normal=json.loads(cfg["normalization"].read_text()); normalizer=Normalizer(np.asarray(normal["mean"],np.float32),np.asarray(normal["std"],np.float32)); window=cfg["window"]
  rw=np.asarray([x[e-window+1:e+1] for x in fusion for e in range(window-1,t)],np.float32)
  _,values=predict_tflite_binary(cfg["model"],normalizer.transform(rw),window)
  padded=np.full((n,t),-128,np.int8); padded[:,window-1:]=values.reshape(n,t-window+1); raw[kind]=padded;fire[kind]=padded>=cfg["raw_threshold"]
 rows=[]
 for i,m in enumerate(metas):
  t0=m['T0']; d=Decision(terrain_state=m['terrain_before']); events=[]; last=None
  for q in range(t):
   stable=None
   if t0 is not None and q>=t0+2 and pred[i,q]>=0 and np.all(pred[i,q-2:q+1]==pred[i,q]): stable=next(k for k,v in LABEL.items() if v==pred[i,q])
   state=d.update(stable,bool(fire['slip'][i,q] and (t0 is not None and q>=t0)),bool(fire['sink'][i,q] and (t0 is not None and q>=t0)))
   if state!=last:events.append((q,state.copy()));last=state.copy()
  target=LABEL[m['terrain_after']]; t1=None if t0 is None else stable_endpoint(pred[i],target,t0); case=m['case_id']; kind='slip' if case=='A' else 'sink' if case=='B' else None
  t2=None if kind is None or t0 is None else next((q for q in range(t0,t) if (m['confirmed_slip'][q] if kind=='slip' else m['sustained_sink'][q])),None)
  t3=None if kind is None or t0 is None else next((q for q in range(t0,t) if fire[kind][i,q]),None); final=events[-1][1]
  scalar_meta={k:v for k,v in m.items() if k not in ('confirmed_slip','sustained_sink')}
  rows.append({**scalar_meta,"T1":t1,"T2":t2,"T3":t3,"T_DECISION":t1 if case in 'CD' else (max(t1,t3) if t1 is not None and t3 is not None else None),"transition_case":final['transition_case'],"case_correct":final['transition_case']==case,"case_reflex_required":final['case_reflex_required'],"recovery_required":final['recovery_required'],"unmatched_hazard":final['unmatched_hazard'],"dual_hazard":final['dual_hazard'],"slip_oracle":bool(np.any(m['confirmed_slip'][m['T0'] or 0:])),"sink_oracle":bool(np.any(m['sustained_sink'][m['T0'] or 0:])),"slip_firing":bool(np.any(fire['slip'][i,m['T0'] or 0:])),"sink_firing":bool(np.any(fire['sink'][i,m['T0'] or 0:]))})
 return rows,scores,pred,raw,fire

def plots(out, packed, rows, pred):
 import matplotlib; matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 directory=out/'plots';directory.mkdir()
 for case in CASES:
  i=next(j for j,r in enumerate(rows) if r['case_id']==case); r=rows[i]; x=packed['time_s'][i]*1000.; fig,ax=plt.subplots(5,1,figsize=(11,10),sharex=True)
  ax[0].plot(x,packed['foot_xyz'][i,:,0],label='left foot x');ax[0].axhline(BOUNDARY_X,color='k',ls='--',label='boundary');ax[0].legend();ax[0].set_ylabel('x [m]')
  ax[1].step(x,packed['loaded_contact'][i],where='post',label='loaded');ax[1].step(x,packed['terrain_gt'][i]==r['terrain_after'],where='post',label='target GT');ax[1].legend();ax[1].set_ylabel('contact / GT')
  ax[2].plot(x,packed['fusion10'][i,:,:4]);ax[2].set_ylabel('FSR [N]')
  ax[3].plot(x,packed['fusion10'][i,:,4:7]);ax[3].set_ylabel('accel')
  target=LABEL[r['terrain_after']];ax[4].step(x,pred[i]==target,where='post',label='target prediction');ax[4].legend();ax[4].set_ylabel('terrain')
  for q,name in ((r['T0'],'T0'),(r['T1'],'T1'),(r['T2'],'T2'),(r['T3'],'T3')):
   if q is not None:
    for a in ax:a.axvline(x[q],color={'T0':'k','T1':'g','T2':'orange','T3':'r'}[name],ls='--')
  ax[-1].set_xlabel('simulation time [ms]');fig.suptitle(f"Case {case}: {r['terrain_before']} → {r['terrain_after']}");fig.tight_layout();fig.savefig(directory/f'case_{case.lower()}_representative.png',dpi=150);plt.close(fig)

def main():
 a=args(); protocol={"dataset":"walking_terrain_transition_v1","spatial_materials_fixed_at_simulation_start":True,"boundary_x_m":BOUNDARY_X,"T_BOUNDARY_CROSS":"first monitored sole target-ground contact; uses contact point rather than ankle origin","T0":"first monitored left-foot target-dominant valid loaded contact (FSR sum >= 5 N)","air_semantics":"System receives no new stable terrain until post-T0 three-endpoint target prediction","controller":"existing Unitree G1 29-DOF ONNX wrapper; observation/action/control period unchanged","cases":CASES,"rate_hz":RATE,"fusion10":HIL_SENSOR_CHANNELS}
 if not a.execute:print(json.dumps(protocol,indent=2));return
 out=a.output_dir.resolve();
 if out.exists() and any(out.iterdir()):raise FileExistsError(out)
 if not a.policy_path.is_file():raise FileNotFoundError(a.policy_path)
 import hashlib
 if hashlib.sha256(a.policy_path.read_bytes()).hexdigest()!=TESTED_POLICY_SHA256:raise ValueError("unverified policy")
 out.mkdir(parents=True); traces=[]; metas=[]; started=time.perf_counter()
 for c in CASES:
  for i in range(a.runs_per_case):
   x,m=run_one(c,i,a.policy_path,a.duration_s,a.walking_speed);traces.append(x);metas.append(m);print(m['run_id'],m['T0'],m['forward_displacement_m'],m['fall_occurred'])
 keys=traces[0].keys(); packed={k:np.asarray([x[k] for x in traces]) for k in keys};packed['run_id']=np.asarray([m['run_id'] for m in metas]);np.savez_compressed(out/'transition_traces.npz',**packed)
 rows,scores,pred,raw,fire=replay(packed['fusion10'],[{**m,**{k:packed[k][i] for k in ('confirmed_slip','sustained_sink')}} for i,m in enumerate(metas)])
 def write(path,rs):
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 write(out/'audit.csv',rows); fr=out/'frozen_replay';fr.mkdir();write(fr/'timeline.csv',rows);np.savez_compressed(fr/'replay_outputs.npz',terrain_scores=scores,terrain_prediction=pred,**{f'{k}_raw_int8':v for k,v in raw.items()})
 physical=all(m['T0'] is not None and m['fusion10_finite'] and not m['fall_occurred'] for m in metas); summary={"pilot_runs":len(rows),"valid_target_contact_runs":sum(r['T0'] is not None for r in rows),"cases":{c:{"runs":sum(r['case_id']==c for r in rows),"target_contact_count":sum(r['target_touchdowns'] for r in rows if r['case_id']==c),"case_correct":sum(r['case_correct'] for r in rows if r['case_id']==c)} for c in CASES},"walking_specific_false_positive_runs":{"slip":sum(r['slip_firing'] and not r['slip_oracle'] for r in rows),"sink":sum(r['sink_firing'] and not r['sink_oracle'] for r in rows)},"WALKING_TERRAIN_TRANSITION_FOUNDATION_READY":physical,"WALKING_FROZEN_AI_REPLAY_COMPLETE":len(rows)==4*a.runs_per_case,"WALKING_MODEL_CHANGE_RECOMMENDED":not all(r['case_correct'] for r in rows),"wall_time_s":time.perf_counter()-started};plots(out,packed,rows,pred);(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(metas,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(fr/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
