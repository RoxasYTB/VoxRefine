#!/usr/bin/env python3
"""Analyze existing NFE/DFN renders against exact speech/noise stems and Adobe references."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/adobe-v2-pair-audit-2026-10-09'
EXPORT=OUT/'exports'
RATE=48000; FRAME=960
IDS=('emy_mixed_ambience18','remi_crowd18','stephanie_fan18')
NOISE={'emy_mixed_ambience18':'fan+crowd-stem.wav','remi_crowd18':'crowd-stem.wav','stephanie_fan18':'fan-stem.wav'}

def read(path):
 x,sr=sf.read(path,dtype='float64');
 if sr!=RATE: raise ValueError(f'{path}: expected {RATE}, got {sr}')
 if x.ndim>1: x=x.mean(axis=1)
 if not np.isfinite(x).all(): raise ValueError(f'nonfinite: {path}')
 return x

def frames(x):
 n=len(x)//FRAME
 return np.sqrt(np.mean(x[:n*FRAME].reshape(n,FRAME)**2,axis=1)+1e-20)

def db(x): return float(20*np.log10(max(float(x),1e-12)))

def main():
 rows=[]
 for sid in IDS:
  sample=ROOT/'corpus/samples/clear-noisy-mix-01'
  clean=read(sample/'clean'/f'{sid}-clean-reference.wav')
  noise=read(sample/'noise'/f'{sid}-{NOISE[sid]}')
  noisy=read(sample/'noisy'/f'{sid}-noisy.wav')
  n=min(map(len,(clean,noise,noisy))); clean,noise,noisy=(v[:n] for v in (clean,noise,noisy))
  cenv=frames(clean); mask=cenv >= np.percentile(cenv,95)*10**(-30/20)
  padded=np.pad(mask.astype(np.int8),6); mask=np.convolve(padded,np.ones(13,dtype=np.int32),mode='same')[6:-6]>0
  weak=mask & (cenv<=np.percentile(cenv[mask],25)); gap=~mask
  base=ROOT/'results/clear-noisy-mix-01/audio'/f'{sid}-resemble-nfe64-C.wav'
  variants={
   'NFE64-C':base,
   'NFE64-C→DFN6':ROOT/'results/post-denoise-finish-01/deepfilter-finished'/sid/'6'/f'{sid}-C-gentle-pre-postfilter.wav',
   'NFE64-C→DFN18':ROOT/'results/post-denoise-finish-01/deepfilter-finished'/sid/'18'/f'{sid}-C-gentle-pre-postfilter.wav',
   'Adobe v2':EXPORT/f'{sid}-adobe-v2.wav',
  }
  for name,path in variants.items():
   y=read(path)[:n]
   if len(y)!=n: raise ValueError(f'duration mismatch: {path} {len(y)} != {n}')
   yenv=frames(y); delta=20*np.log10((yenv+1e-12)/(cenv+1e-12))
   weak_delta=delta[weak]
   zero_weak_fraction=float(np.mean(weak_delta < -80.0))
   speech_level=float(np.median(yenv[mask])); clean_level=float(np.median(cenv[mask]))
   rows.append({'voice':sid,'candidate':name,'speech_envelope_corr':float(np.corrcoef(np.log(cenv[mask]+1e-12),np.log(yenv[mask]+1e-12))[0,1]),
    'speech_level_median_delta_db':float(np.median(delta[mask])),
    'weak_frame_p10_delta_db':float(np.percentile(delta[weak],10)),
    'weak_frames_below_minus80db_fraction':zero_weak_fraction,
    'gap_rms_dbfs':db(np.median(yenv[gap])) if np.any(gap) else None,
    'sample_peak_dbfs':db(np.max(np.abs(y)))})
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'cascade-threevoice-comparison.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
 names=['NFE64-C','NFE64-C→DFN6','NFE64-C→DFN18','Adobe v2']; colors=['#e76f51','#2a9d8f','#457b9d','#111827']
 for ax,metric,title,ylabel in [(axes[0],'weak_frame_p10_delta_db','Faibles trames (p10)','dB vs parole propre'),(axes[1],'weak_frames_below_minus80db_fraction','Faibles trames écrasées','fraction sous −80 dB vs propre'),(axes[2],'speech_envelope_corr','Enveloppe vocale','corrélation')]:
  for i,sid in enumerate(IDS):
   vals=[next(r[metric] for r in rows if r['voice']==sid and r['candidate']==name) for name in names]
   ax.plot(range(len(names)),vals,marker='o',label=sid,color=['#bc6c25','#457b9d','#2a9d8f'][i])
  ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xticks(range(len(names)),names,rotation=25,ha='right');ax.grid(alpha=.25)
 axes[0].legend(fontsize=7)
 fig.suptitle('Existing NFE64-C / DPDFNet cascade variants vs Adobe v2 reference · 3 controlled voices')
 fig.savefig(OUT/'cascade-threevoice-comparison.png',dpi=170)
 summary=[]
 for name in names:
  group=[r for r in rows if r['candidate']==name]
  summary.append({'candidate':name,'median_env_corr':float(np.median([r['speech_envelope_corr'] for r in group])),
   'median_weak_p10_db':float(np.median([r['weak_frame_p10_delta_db'] for r in group])),
   'median_weak_frames_below_minus80db_fraction':float(np.median([r['weak_frames_below_minus80db_fraction'] for r in group])),
   'median_gap_rms_dbfs':float(np.median([r['gap_rms_dbfs'] for r in group if r['gap_rms_dbfs'] is not None])) if any(r['gap_rms_dbfs'] is not None for r in group) else None})
 report={'protocol':'Existing 15 s exact 48 kHz clean speech/noise stems and NFE64-C, saved DPDFNet postfilter outputs (6/18 dB), and Adobe v2 exports. No new enhancement inference.','limitations':['Only three controlled LibriVox reads and stationary/CC0 backgrounds.','Weak-frame loss is defined as output RMS more than 80 dB below the aligned clean reference in the reference weak-active quartile; it detects near-silence, not phoneme identity.','DFN outputs were created earlier; this analysis does not establish them as Adobe-matched renders.','Adobe is a reference, not the clean speech truth.'],'summary':summary,'results_csv':str((OUT/'cascade-threevoice-comparison.csv').relative_to(ROOT)),'figure':str((OUT/'cascade-threevoice-comparison.png').relative_to(ROOT))}
 (OUT/'cascade-threevoice-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
