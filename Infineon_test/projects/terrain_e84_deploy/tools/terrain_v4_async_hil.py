#!/usr/bin/env python3
"""FRV2-style batched 1 kHz replay for frozen Terrain v4 T4B1 firmware."""
from __future__ import annotations
import argparse, csv, json, os, select, struct, termios, time, zlib
from pathlib import Path
import numpy as np
from terrain_hil_client import configure_uart
from terrain_v4_e84_client import BROADER, DEPLOY, PORT, V4, normalizer, raw_tflite_quantized

BATCH=20; MAGIC=b"T4B1"; RESPONSE=b"T4R1"; COUNTERS=("rx_packets","sample_count","inference_count","drops","crc_errors","sequence_errors","ring_overflow","ring_underflow","device_deadline_miss")
def frame(seq:int, samples:np.ndarray)->bytes:
    data=np.asarray(samples,np.int8); assert data.ndim==2 and data.shape[1:]==(10,) and 1<=len(data)<=BATCH
    p=struct.pack('<IH',seq,len(data))+data.tobytes(); return MAGIC+struct.pack('<H',len(p))+p+struct.pack('<I',zlib.crc32(p)&0xffffffff)
def read_exact(fd:int,n:int,timeout:float)->bytes:
    b=bytearray(); end=time.monotonic()+timeout
    while len(b)<n and time.monotonic()<end:
        r,_,_=select.select([fd],[],[],max(.001,end-time.monotonic()))
        if r:b.extend(os.read(fd,n-len(b)))
    if len(b)!=n: raise TimeoutError(f"wanted {n}, got {len(b)}")
    return bytes(b)
def response(fd:int,timeout:float)->dict:
    b=bytearray()
    while RESPONSE not in b:
        b.extend(read_exact(fd,1,timeout)); b[:]=b[-4:]
    length=struct.unpack('<H',read_exact(fd,2,timeout))[0]; payload=read_exact(fd,length,timeout); crc=struct.unpack('<I',read_exact(fd,4,timeout))[0]
    if zlib.crc32(payload)&0xffffffff!=crc: raise ValueError('response crc')
    first,count,inferred=struct.unpack_from('<IHH',payload); values=struct.unpack_from('<9I',payload,8); rec=[]; off=44
    for _ in range(inferred):
        seq=struct.unpack_from('<I',payload,off)[0]; raw=list(struct.unpack_from('<4b',payload,off+4)); cls,stable=struct.unpack_from('<bb',payload,off+8);rec.append({'sequence':seq,'raw':raw,'class':cls,'stable':stable});off+=10
    return {'first_sequence':first,'count':count,'inferred':inferred,'counters':dict(zip(COUNTERS,values)),'records':rec}
def stable(raw:np.ndarray)->np.ndarray:
    out=np.full(len(raw),-1,np.int8); last=-2; n=0; state=-1
    for i,c in enumerate(np.argmax(raw,axis=1)):
        n=n+1 if c==last else 1;last=int(c)
        if n>=3:state=int(c)
        out[i]=state
    return out
def run(port:Path,limit:int|None,output:Path,timeout:float)->dict:
    policy=json.loads((DEPLOY/'runtime_parity_policy.json').read_text()); norm=normalizer(); tensors=json.loads((DEPLOY/'golden_vectors.json').read_text())['tensors']['input']; scale,zero=tensors['scale'],tensors['zero_point']
    with np.load(BROADER/'broader_transition_runs.npz') as z: traces,ids=z['fusion10'],z['run_id'].astype(str)
    manifest={r['run_id']:r for r in csv.DictReader((BROADER/'manifest.csv').open())}; selected=[]
    for case in 'ABCD': selected += [i for i,x in enumerate(ids) if manifest[x]['case_id']==case][:3]
    if limit:selected=selected[:limit]
    fd=os.open(port,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK);configure_uart(fd);termios.tcflush(fd,termios.TCIOFLUSH);runs=[];start_all=time.monotonic()
    try:
      for ix in selected:
        trace=traces[ix]; q=np.clip(np.rint(norm.transform(trace[None])[0]/scale+zero),-128,127).astype(np.int8); windows=np.asarray([q[e-49:e+1] for e in range(49,len(q))]); host=raw_tflite_quantized(V4/'int8/baseline_50_seed_20260823_strict_int8.tflite',windows); hs=stable(host); got=[]; latency=[];seq=0
        for begin in range(0,len(q),BATCH):
          t=time.monotonic();os.write(fd,frame(seq,q[begin:begin+BATCH]));reply=response(fd,timeout);latency.append((time.monotonic()-t)*1000);got+=reply['records'];seq+=reply['count']
        board=np.asarray([x['raw'] for x in got],np.int8); bc=np.asarray([x['class'] for x in got]); bs=np.asarray([x['stable'] for x in got]); target={'A':2,'B':3,'C':1,'D':1}[manifest[ids[ix]]['case_id']]
        ht=next((e+49 for e in range(len(hs)) if e+49>=650 and hs[e]==target),None);bt=next((x['sequence'] for x in got if x['sequence']>=650 and x['stable']==target),None)
        runs.append({'run_id':ids[ix],'case_id':manifest[ids[ix]]['case_id'],'samples':len(q),'inferences':len(got),'raw_exact':int(np.sum(np.all(board==host,axis=1))),'bounded':int(np.sum(np.max(np.abs(board.astype(int)-host.astype(int)),axis=1)<=policy['rules']['saturated_raw']['linf_raw_count_lte'])),'class_exact':int(np.sum(bc==np.argmax(host,axis=1))),'stable_exact':int(np.sum(bs==hs)),'host_t1':ht,'board_t1':bt,'t1_delta':None if ht is None or bt is None else bt-ht,'batch_latency_ms':latency,'counters':reply['counters'],'cpu':[0], 'npu':[0]})
    finally:os.close(fd)
    total_inf=sum(x['inferences'] for x in runs); elapsed=sum(v for x in runs for v in x['batch_latency_ms'])/1000; counters=runs[-1]['counters']; cls=sum(x['class_exact'] for x in runs)/total_inf; st=sum(x['stable_exact'] for x in runs)/total_inf; gate=all(counters[k]==0 for k in ('drops','crc_errors','sequence_errors','ring_overflow','device_deadline_miss')) and cls>=.98 and st>=.98 and all(abs(x['t1_delta'])<=1 for x in runs if x['t1_delta'] is not None)
    report={'protocol':'T4B1','batch_samples':BATCH,'runs':runs,'total_samples':sum(x['samples'] for x in runs),'total_inferences':total_inf,'effective_samples_per_second':sum(x['samples'] for x in runs)/elapsed,'batch_latency_ms_median':float(np.median([v for x in runs for v in x['batch_latency_ms']])), 'batch_latency_ms_p95':float(np.percentile([v for x in runs for v in x['batch_latency_ms']],95)), 'final_counters':counters,'raw_exact':sum(x['raw_exact'] for x in runs),'bounded_parity':sum(x['bounded'] for x in runs),'class_exact_rate':cls,'stable_state_exact_rate':st,'TERRAIN_V4_ASYNC_1KHZ_HIL_GATE':'PASS' if gate else 'FAIL','TERRAIN_V4_RUNTIME_DEPLOYMENT_READY':bool(gate and policy['TERRAIN_V4_TARGET_RUNTIME_PARITY_GATE']=='PASS')};output.write_text(json.dumps(report,indent=2)+'\n');return report
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--port',type=Path,default=PORT);p.add_argument('--limit',type=int);p.add_argument('--timeout',type=float,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.port,a.limit,a.output,a.timeout),indent=2))
if __name__=='__main__':main()
