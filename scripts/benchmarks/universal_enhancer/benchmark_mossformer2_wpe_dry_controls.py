#!/usr/bin/env python3
"""Dry-control safety screen for the frozen MossFormer2 -> WPE chain."""
from __future__ import annotations
import csv,json,os,sys,time,shutil
from pathlib import Path
import numpy as np,soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BENCH=ROOT/'results/noise-rir-truth-01/vctk-rir-dereverb-truth-01';RUN=BENCH/'mossformer2-wpe-screen-01-dry-controls';RATE=48000
sys.path.insert(0,str(ROOT/'scripts/benchmarks/universal_enhancer'))
from bench_vctk_wpe_truth import wpe

def read(p):
 x,s=sf.read(p,dtype='float64',always_2d=True)
 if s!=RATE:raise ValueError((p,s))
 return x.mean(1)
def e_db(x):return float(10*np.log10(np.mean(np.asarray(x,dtype='float64')**2)+1e-24))
def main():
 if RUN.exists():raise SystemExit(f'refusing to overwrite {RUN}')
 man=json.loads((BENCH/'condition-manifest.json').read_text())
 cases=[c for c in man['conditions'] if c['condition']=='dry' and c['rir']=='t60-0.45s']
 RUN.mkdir(parents=True);inputs=RUN/'inputs';inputs.mkdir();mossdir=RUN/'mossformer2';mossdir.mkdir();wpe_dir=RUN/'wpe-after-mossformer2';wpe_dir.mkdir()
 for c in cases:shutil.copy2(ROOT/c['input'],inputs/Path(c['input']).name)
 # Warm the same GPU batch model used in the paired RIR screen.
 os.chdir(ROOT/'.tools/clearervoice-src/clearvoice');sys.path.insert(0,str(Path.cwd()))
 import torch
 from clearvoice import ClearVoice
 torch.set_num_threads(4);t=time.perf_counter();cv=ClearVoice(task='speech_enhancement',model_names=['MossFormer2_SE_48K']);load=time.perf_counter()-t
 t=time.perf_counter();outputs=cv(str(inputs),online_write=False);infer=time.perf_counter()-t
 for name,a in outputs.items():
  y=np.asarray(a)
  while y.ndim>1:y=y[0]
  sf.write(mossdir/name,y.astype('float32'),RATE,subtype='PCM_16')
 rows=[];wpe_total=0
 for c in cases:
  src=ROOT/c['input'];dry_path=ROOT/c['dry_reference'];x=read(src);d=read(dry_path);mp=mossdir/Path(c['input']).name;m=read(mp)
  op=wpe_dir/Path(c['input']).name.replace('.wav','-wpe.wav');t=time.perf_counter();y0=wpe(m);sec=time.perf_counter()-t;wpe_total+=sec;sf.write(op,y0,RATE,subtype='PCM_16');y=read(op)
  lo=round(c['event_start_seconds']*RATE);hi=lo+round(c['event_duration_seconds']*RATE);r={'speaker':c['speaker'],'utterance':c['utterance'],'rir_tag':c['rir'],'wpe_seconds':sec,'wpe_rtf':sec/(len(y)/RATE),'clipped_samples':int(np.sum(np.abs(y)>=1.))}
  for method,z in [('input',x),('moss',m),('wpe_after_moss',y)]:
   r[f'{method}_event_energy_delta_vs_dry_db']=e_db(z[lo:hi])-e_db(d[lo:hi])
   for lab,a,b in [('onset_0_40',0,.04),('early_40_120',.04,.12)]:r[f'{method}_{lab}_gain_vs_dry_db']=e_db(z[lo+round(a*RATE):lo+round(b*RATE)])-e_db(d[lo+round(a*RATE):lo+round(b*RATE)])
   for lab,a,b in [('tail_50_150',.05,.15),('tail_150_300',.15,.30),('tail_300_600',.30,.60)]:r[f'{method}_{lab}_dbfs']=e_db(z[hi+round(a*RATE):min(len(z),hi+round(b*RATE))])
  rows.append(r)
 with (RUN/'event-metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for metric in ('moss_event_energy_delta_vs_dry_db','wpe_after_moss_event_energy_delta_vs_dry_db','moss_onset_0_40_gain_vs_dry_db','wpe_after_moss_onset_0_40_gain_vs_dry_db','moss_early_40_120_gain_vs_dry_db','wpe_after_moss_early_40_120_gain_vs_dry_db','moss_tail_50_150_dbfs','wpe_after_moss_tail_50_150_dbfs','moss_tail_150_300_dbfs','wpe_after_moss_tail_150_300_dbfs'):
  v=np.asarray([r[metric] for r in rows]);summary.append({'metric':metric,'n':len(v),'median':float(np.median(v)),'p10':float(np.percentile(v,10)),'p90':float(np.percentile(v,90)),'min':float(v.min()),'max':float(v.max())})
 with (RUN/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 rt={'device':str(cv.models[0].device),'load_seconds':load,'mossformer_inference_seconds':infer,'clips':len(rows),'mossformer_rtf_total':infer/(3*len(rows)),'wpe_seconds_total':wpe_total,'wpe_rtf_total':wpe_total/(3*len(rows)),'wpe_settings':{'sample_rate_hz':RATE,'n_fft':1024,'hop':256,'taps':32,'delay_frames':3,'iterations':3,'regularization':1e-6}}
 (RUN/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
 print(json.dumps({'runtime':rt,'summary':summary},indent=2))
if __name__=='__main__':main()
