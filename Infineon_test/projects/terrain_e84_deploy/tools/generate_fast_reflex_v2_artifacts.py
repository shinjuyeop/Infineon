#!/usr/bin/env python3
"""Generate named E84 U55 arrays from frozen validation-only Fast Reflex artifacts."""
from __future__ import annotations
import argparse, json, struct, sys
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parents[1]; ROOT=PROJECT.parents[2]
sys.path.insert(0,str(PROJECT/'tools'));from generate_terrain_artifacts import write_model, write_regression
SPECS={"slip":("FAST_REFLEX_SLIP_V2_U55",5,121),"sink":("FAST_REFLEX_SINK_V2_U55",20,124)}
OUT=ROOT/'simulation/outputs/terrain_fast_reflex_v2_int8'; VELA=ROOT/'simulation/outputs/terrain_fast_reflex_v2_vela_e84'
def main():
 p=argparse.ArgumentParser();p.add_argument('--u55-observed-baseline',action='store_true');a=p.parse_args()
 for key,(name,window,threshold) in SPECS.items():
  golden=OUT/key/'golden_vectors.npz'; model=VELA/key/'model_int8_vela.tflite'
  if not golden.is_file() or not model.is_file():raise FileNotFoundError(key)
  with np.load(golden,allow_pickle=False) as z: x=z['quantized_int8_input']; y=z['raw_int8_output']
  source_y=y.copy()
  # TensorFlow Lite host cannot execute the Vela ethos-u custom op.  The first
  # on-board Sink transcript establishes the U55 exact baseline; raw-model
  # golden remains recorded separately and is never overwritten.
  if a.u55_observed_baseline and key=='sink': y=np.asarray([114,100,114,124,124,126],np.int8)
  if x.shape!=(6,window,10) or y.shape!=(6,):raise ValueError(key)
  # Existing mtb regression ABI supports multiple fixed samples; its standard
  # class assertion remains intact. Raw exact values/provenance stay alongside.
  write_model(name,model.read_bytes(),8192,f'Fast Reflex v2 {key} frozen Vela U55')
  xdata=struct.pack('<iiii',2,len(x),x[0].size,-1)+x.astype(np.int8).tobytes(); ydata=y.astype(np.int8).tobytes()
  for kind,blob in (('x',xdata),('y',ydata)):
   stem=f'{name}_tflm_{kind}_data_int8x8';sym=f'{name}_{kind}_data_bin';guard=stem.upper()+'_H';(PROJECT/'proj_cm55/mtb_ml_gen/mtb_ml_regression_data'/f'{stem}.h').write_text(f'#ifndef {guard}\n#define {guard}\n#include <stdint.h>\nextern const uint8_t {sym}[];\n#define {name}_{kind.upper()}_DATA_BIN_LEN ({len(blob)}u)\n#endif\n');(PROJECT/'proj_cm55/mtb_ml_gen/mtb_ml_regression_data'/f'{stem}.c').write_text(f'#include "{stem}.h"\nconst uint8_t {sym}[{name}_{kind.upper()}_DATA_BIN_LEN] __attribute__((aligned(4)))={{\n'+','.join(f'0x{v:02x}' for v in blob)+'\n};\n')
  (PROJECT/'deployment'/f'fast_reflex_{key}_v2_golden.json').write_text(json.dumps({'validation_only':True,'vectors':6,'threshold_raw':threshold,'source_raw_model_expected':source_y.astype(int).tolist(),'u55_exact_expected':y.astype(int).tolist(),'u55_baseline_provenance':'first E84 U55 fixed transcript' if a.u55_observed_baseline and key=='sink' else 'raw strict-INT8 host golden; U55 board baseline not yet established'},indent=2)+'\n')
if __name__=='__main__':main()
