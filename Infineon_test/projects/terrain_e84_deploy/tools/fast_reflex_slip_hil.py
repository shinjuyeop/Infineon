#!/usr/bin/env python3
"""Slip v2 continuous-HIL CLI; reuses the FRV2 batched transport client."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import fast_reflex_sink_hil as h

h.MEAN=np.array([31.594806671,31.777748108,13.546282768,13.864206314,-.086954400,.008436359,9.803998947,.000235322,-.003542150,-.001164354],np.float32)
h.STD=np.array([8.742906570,8.528360367,3.996133566,3.821703196,.583785653,.310120761,.621062100,.010392545,.010778599,.007863495],np.float32)
h.SCALE=np.float32(.0939839631319046); h.ZERO=-6; h.THRESHOLD=121
h.OUT=h.ROOT/'simulation/outputs/terrain_fast_reflex_v2_slip_hil'

def select_traces(source:Path)->list[dict]:
    h.guard(source); rows=list(csv.DictReader((source/'manifest.csv').open()))
    with np.load(source/'oracle_diagnostics.npz',allow_pickle=False) as z:
        oracle=np.asarray(z['confirmed_slip']); ids=[str(x) for x in z['run_id']]
    positive={rid:bool(np.any(oracle[i])) for i,rid in enumerate(ids)}
    eligible=[r for r in rows if r['split']=='validation' and r['surface_family'] in {'crosshatch','rounded_ridges'} and r['run_id'] in positive]
    pos=[r for r in eligible if positive[r['run_id']]][:3]; neg=[r for r in eligible if not positive[r['run_id']]][:3]
    if len(pos)!=3 or len(neg)!=3: raise RuntimeError('need three validation confirmed_slip positives and negatives')
    return [{'run_id':r['run_id'],'label':'confirmed_slip' if r in pos else 'normal','family':r['surface_family'],'split':'validation'} for r in pos+neg]
h.select_traces=select_traces

if __name__=='__main__': h.main()
