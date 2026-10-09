#!/usr/bin/env python3
"""Render causal DPDFNet2 48 kHz over the frozen held-out weak-speech inputs."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'results/noise-rir-truth-01/weak-speech-backend-heldout-02'
OUT=ROOT/'results/noise-rir-truth-01/realtime-weak-preservation-01'
SDK=ROOT/'.tools/deepvqe-screen'
RATE=48000;HOP=480;ALIGN=1920

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 x,s=sf.read(p,dtype='float32')
 if s!=RATE or x.ndim!=1 or not np.isfinite(x).all():raise ValueError(f'bad input {p}')
 return x

def main():
 sys.path[:0]=[str(SDK),str(Path(__file__).resolve().parent)]
 from dpdfnet.stream import StreamEnhancer
 from dpdfnet_benchmark_adapter import DpdfNetBenchmarkAdapter
 inputs=sorted((SOURCE/'inputs').glob('*-input.wav'))
 expected=len(json.loads((SOURCE/'condition-manifest.json').read_text())['conditions'])
 if len(inputs)!=expected:raise SystemExit(f'expected frozen {expected} inputs from condition manifest, got {len(inputs)}')
 OUT.mkdir(parents=True,exist_ok=True); render=OUT/'renders';render.mkdir(exist_ok=True)
 load_start=time.perf_counter()
 model=StreamEnhancer('dpdfnet2_48khz_hr',onnx_path=SDK/'dpdfnet2_48khz_hr.onnx')
 model_load_s=time.perf_counter()-load_start
 if model._model_sr!=RATE:raise RuntimeError('DPDFNet2 model is not native 48 kHz')
 rows=[];t0=time.perf_counter()
 for i,path in enumerate(inputs,1):
  x=read(path); adapter=DpdfNetBenchmarkAdapter(model,sample_rate=RATE,alignment_samples=ALIGN)
  start=time.perf_counter()
  for off in range(0,len(x),HOP):adapter.process(x[off:off+HOP])
  y=adapter.finalize();elapsed=time.perf_counter()-start
  if len(y)!=len(x) or not np.isfinite(y).all():raise RuntimeError(f'invalid result {path}: {len(y)} vs {len(x)}')
  dest=render/(path.stem.replace('-input','')+'-dpdfnet2-48khz.wav')
  sf.write(dest,y,RATE,subtype='FLOAT')
  rows.append({'input':str(path.relative_to(ROOT)),'input_sha256':sha(path),'output':str(dest.relative_to(ROOT)),'output_sha256':sha(dest),'elapsed_s':elapsed,'rtf':elapsed/(len(x)/RATE),'samples':len(y),'sample_rate_hz':RATE,'peak_abs':float(np.max(np.abs(y))),'zero_drain_samples':adapter.zero_drain_samples})
  (OUT/'progress.json').write_text(json.dumps({'completed':i,'expected':len(inputs),'rows':rows},indent=2)+'\n')
  if i%10==0 or i==len(inputs):print(f'{i}/{len(inputs)} DPDFNet2 complete; elapsed {time.perf_counter()-t0:.1f}s',flush=True)
 report={'protocol':'Frozen input stems from weak-speech-backend-heldout-02; no input regeneration or parameter tuning. Causal native 48 kHz DPDFNet2, 10 ms chunks, 1920-sample sample-alignment adapter.','method':'dpdfnet2_48khz_hr','device':'CPU','input_count':len(inputs),'model_load_s':model_load_s,'total_inference_wall_s':time.perf_counter()-t0,'outputs':rows}
 (OUT/'render-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'progress.json').unlink(missing_ok=True)
 print(f'DONE {len(rows)} renders; median RTF {np.median([r["rtf"] for r in rows]):.3f}',flush=True)
if __name__=='__main__':main()
