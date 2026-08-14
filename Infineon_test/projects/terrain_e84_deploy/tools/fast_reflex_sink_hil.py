#!/usr/bin/env python3
"""Batched, asynchronous-by-block continuous HIL client for frozen Sink v2.

FRV2 deliberately does not use TerrainStreamLink/TRN2: a block is written in
one UART transaction and only one compact summary is read after it.  The board
processes its ordered samples at a logical 1 kHz cadence.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, select, struct, termios, time, zlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'simulation/outputs/terrain_fast_reflex_v2_final_scope_full'
OUT=ROOT/'simulation/outputs/terrain_fast_reflex_v2_sink_hil'
MEAN=np.array([31.530698776245117,31.693687438964844,13.53983211517334,13.843737602233887,-.07347263395786285,.008472549729049206,9.79791259765625,.000206427022931166,-.0031850915402173996,-.001231436850503087],np.float32)
STD=np.array([8.335794448852539,8.145730972290039,3.759648561477661,3.584739923477173,.5414703488349915,.275717556476593,.5658554434776306,.009250503964722157,.010388086549937725,.007618863135576248],np.float32)
SCALE=np.float32(.09580779820680618); ZERO=-8; THRESHOLD=124; BATCH=20
RESULT=re.compile(rb'FRV2_RESULT seq=(\d+),count=(\d+),inferred=(\d+),fires=(\d+),drops=(\d+),crc_errors=(\d+),deadline_miss=(\d+),quant_digest=([0-9a-f]+),raw_digest=([0-9a-f]+),max_cpu_cyc=(\d+)')

def guard(source:Path)->None:
    text=str(source).lower()
    if 'final_test' in text or any(x in text for x in ('9110','9111','9112','9113','9114','20260903','921000')): raise ValueError('final-test source is sealed and forbidden')
    if not (source/'inputs_fusion10.npz').is_file(): raise ValueError(f'invalid source: {source}')

def quantize(raw:np.ndarray)->np.ndarray:
    """Firmware-equivalent frozen preprocessing; numpy rint is ties-to-even."""
    raw=np.asarray(raw,np.float32)
    return np.clip(np.rint(((raw-MEAN)/STD)/SCALE+ZERO),-128,127).astype(np.int8)

def build_frame(sequence:int, samples:np.ndarray)->bytes:
    samples=np.asarray(samples,np.float32)
    if samples.ndim!=2 or samples.shape[1:]!=(10,) or not 1<=len(samples)<=BATCH: raise ValueError('FRV2 requires 1..20 Fusion10 samples')
    payload=struct.pack('<IH',sequence,len(samples))+samples.astype('<f4',copy=False).tobytes()
    return b'FRV2'+struct.pack('<H',len(payload))+payload+struct.pack('<I',zlib.crc32(payload)&0xffffffff)

def select_traces(source:Path)->list[dict]:
    guard(source); rows=list(csv.DictReader((source/'manifest.csv').open()))
    with np.load(source/'oracle_diagnostics.npz',allow_pickle=False) as z: oracle=np.asarray(z['sustained_sink']); ids=[str(x) for x in z['run_id']]
    onset={rid: bool(np.any(oracle[i])) for i,rid in enumerate(ids)}
    eligible=[r for r in rows if r['split']=='validation' and r['surface_family'] in {'crosshatch','rounded_ridges'} and r['run_id'] in onset]
    pos=[r for r in eligible if onset[r['run_id']]][:3]; neg=[r for r in eligible if not onset[r['run_id']]][:3]
    if len(pos)!=3 or len(neg)!=3: raise RuntimeError('need 3 oracle sustained_sink positives and 3 negatives')
    return [{'run_id':r['run_id'],'label':'sustained_sink' if r in pos else 'normal','family':r['surface_family'],'split':'validation'} for r in pos+neg]

def offline(source:Path, traces:list[dict])->dict:
    with np.load(source/'inputs_fusion10.npz',allow_pickle=False) as z: raw=np.asarray(z['sensors'],np.float32); ids=[str(x) for x in z['run_id']]
    index={x:i for i,x in enumerate(ids)}; result=[]
    for t in traces:
        q=quantize(raw[index[t['run_id']]])
        result.append({**t,'samples':len(q),'quant_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'warmup_samples':19})
    return {'protocol':'FRV2','sample_rate_hz':1000,'batch_samples':20,'threshold_raw':124,'persistence':1,'traces':result}

def configure(fd:int)->None:
    a=termios.tcgetattr(fd); a[0]=a[1]=a[3]=0; a[2]=termios.CS8|termios.CLOCAL|termios.CREAD; a[4]=a[5]=termios.B1000000; a[6][termios.VMIN]=0; a[6][termios.VTIME]=1; termios.tcsetattr(fd,termios.TCSANOW,a)
def line(fd:int,timeout:float)->bytes:
    d=bytearray(); end=time.monotonic()+timeout
    while time.monotonic()<end:
        r,_,_=select.select([fd],[],[],min(.1,end-time.monotonic()))
        if r:
            d.extend(os.read(fd,4096));
            if b'\n' in d:return bytes(d)
    raise TimeoutError(d.decode(errors='replace'))
def replay(port:Path,source:Path,manifest:dict,timeout:float)->dict:
    with np.load(source/'inputs_fusion10.npz',allow_pickle=False) as z: raw=np.asarray(z['sensors'],np.float32); ids={str(x):i for i,x in enumerate(z['run_id'])}
    fd=os.open(port,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK); configure(fd); termios.tcflush(fd,termios.TCIOFLUSH); runs=[]
    try:
      for run,t in enumerate(manifest['traces']):
        samples=raw[ids[t['run_id']]]; seq=0; replies=[]
        for start in range(0,len(samples),BATCH):
            frame=build_frame(seq,samples[start:start+BATCH]); os.write(fd,frame)
            end=time.monotonic()+timeout; reply=b''; m=None
            while time.monotonic()<end and m is None:
                reply=line(fd,max(.01,end-time.monotonic())); m=RESULT.search(reply)
            if m is None: raise RuntimeError('no FRV2_RESULT; verify frv2_sink_hil firmware: '+reply.decode(errors='replace'))
            replies.append(m.groupdict() if hasattr(m,'groupdict') else {'line':reply.decode()}); seq+=min(BATCH,len(samples)-start)
        runs.append({'run':run,**t,'replies':replies})
    finally: os.close(fd)
    return {'runs':runs}

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','smoke','replay']); p.add_argument('--source',type=Path,default=SOURCE); p.add_argument('--port',type=Path); p.add_argument('--output-dir',type=Path,default=OUT); p.add_argument('--timeout',type=float,default=3.0); a=p.parse_args()
 traces=select_traces(a.source); manifest=offline(a.source,traces); a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'trace_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if a.command=='prepare': print(f'FRV2_PREPARED traces=6 manifest={a.output_dir}/trace_manifest.json'); return
 if a.port is None: p.error('--port is required for hardware replay')
 if a.command=='smoke': manifest['traces']=manifest['traces'][:1]
 summary={'configuration':manifest,'hardware':replay(a.port,a.source,manifest,a.timeout)}; (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(f'FRV2_COMPLETE summary={a.output_dir}/summary.json')
if __name__=='__main__': main()
