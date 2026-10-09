#!/usr/bin/env python3
"""Frozen DeepVQE-after-Moss and fixed 25/50% blend screen on VCTK RIR truth.

Exploratory only: upstream code is MIT but the local DNS3 checkpoint's
redistribution license is unclear. No parameter or blend is tuned on this set.
"""
from __future__ import annotations
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np,soundfile as sf,torch,torchaudio.functional as AF
from scipy.signal import correlate
ROOT=Path(__file__).resolve().parents[3];BENCH=ROOT/'results/noise-rir-truth-01/vctk-rir-dereverb-truth-01';RUN=BENCH/'mossformer2-wpe-screen-01';OUT=RUN/'deepvqe-blend';RATE=48000;RATE16=16000

def read(p):
 x,s=sf.read(p,dtype='float64',always_2d=True)
 if s!=RATE:raise ValueError((p,s))
 return x.mean(1)
def edb(x):return float(10*np.log10(np.mean(np.asarray(x,dtype='float64')**2)+1e-24))
def main():
 if OUT.exists():raise SystemExit(f'refusing to overwrite {OUT}')
 OUT.mkdir(parents=True);renders=OUT/'renders';renders.mkdir()
 sys.path.insert(0,str(ROOT/'.tools/deepvqe-screen/deepvqe-main'))
 from deepvqe import DeepVQE
 cp=ROOT/'.tools/deepvqe-screen/deepvqe_trained_on_DNS3.tar';weight_hash=hashlib.sha256(cp.read_bytes()).hexdigest()
 torch.set_num_threads(4);t=time.perf_counter();model=DeepVQE().eval();state=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);load=time.perf_counter()-t
 win=torch.hann_window(512);rows=[];infer_total=0.
 for c in json.loads((RUN/'selection.json').read_text())['conditions']:
  source=ROOT/c['input'];dry=read(ROOT/c['dry_reference']);moss_path=RUN/'mossformer2'/Path(c['input']).name;m=read(moss_path)
  x16=AF.resample(torch.from_numpy(m.copy()),RATE,RATE16).float();n=len(x16);spec=torch.stft(x16[None,:],n_fft=512,hop_length=256,win_length=512,window=win,center=True,return_complex=True)
  t=time.perf_counter()
  with torch.inference_mode(): z=model(torch.view_as_real(spec))
  sec=time.perf_counter()-t;infer_total+=sec
  y16=torch.istft(torch.complex(z[...,0],z[...,1]),n_fft=512,hop_length=256,win_length=512,window=win,center=True,length=n)[0]
  dv=AF.resample(y16,RATE16,RATE).numpy().astype(np.float64);dv=np.pad(dv,(0,max(0,len(m)-len(dv))))[:len(m)]
  # Characterize alignment before forming sample-aligned fixed blends.
  frame=480;count=min(len(m),len(dv))//frame;me=np.sqrt(np.mean(m[:count*frame].reshape(count,frame)**2,axis=1));de=np.sqrt(np.mean(dv[:count*frame].reshape(count,frame)**2,axis=1));me-=me.mean();de-=de.mean();cc=correlate(de,me,mode='full');lags=np.arange(-count+1,count);valid=np.abs(lags)<=5;lag=int(lags[valid][np.argmax(cc[valid])]);env_corr=float(np.max(cc[valid])/(np.linalg.norm(me)*np.linalg.norm(de)+1e-20))
  # Keep sample alignment fixed at file origin; record estimated envelope offset for diagnosis.
  sf.write(renders/(moss_path.stem+'-deepvqe.wav'),dv,RATE,subtype='PCM_16')
  variants={'deepvqe':dv,'blend25':.75*m+.25*dv,'blend50':.50*m+.50*dv}
  lo=round(c['event_start_seconds']*RATE);hi=lo+round(c['event_duration_seconds']*RATE)
  base={'speaker':c['speaker'],'utterance':c['utterance'],'rir':c['rir'],'deepvqe_seconds':sec,'deepvqe_rtf':sec/(len(m)/RATE),'envelope_lag_frames':lag,'envelope_lag_ms':lag*10,'envelope_correlation_near_zero':env_corr}
  for name,y in variants.items():
   op=renders/(moss_path.stem+f'-{name}.wav');sf.write(op,y,RATE,subtype='PCM_16');y=read(op);r=base.copy();r.update({'backend':name,'duration_s':len(y)/RATE,'clipped_samples':int(np.sum(np.abs(y)>=1.)),'event_energy_delta_vs_dry_db':edb(y[lo:hi])-edb(dry[lo:hi])})
   for lab,a,b in [('onset_0_40',0,.04),('early_40_120',.04,.12)]:r[f'{lab}_gain_vs_dry_db']=edb(y[lo+round(a*RATE):lo+round(b*RATE)])-edb(dry[lo+round(a*RATE):lo+round(b*RATE)])
   for lab,a,b in [('tail_50_150',.05,.15),('tail_150_300',.15,.30),('tail_300_600',.30,.60)]:
    q=y[hi+round(a*RATE):min(len(y),hi+round(b*RATE))];r[f'{lab}_dbfs']=edb(q)
   r['tail150_suppression_vs_input_db']=edb(read(source)[hi+round(.15*RATE):hi+round(.30*RATE)])-r['tail_150_300_dbfs'];rows.append(r)
  print(f"{c['speaker']}/{c['utterance']}/{c['rir']}: {sec:.2f}s lag {lag*10:+d}ms",flush=True)
 with (OUT/'event-metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for rir in ('t60-0.45s','t60-0.80s'):
  for method in ('deepvqe','blend25','blend50'):
   g=[r for r in rows if r['rir']==rir and r['backend']==method]
   for metric in ('event_energy_delta_vs_dry_db','onset_0_40_gain_vs_dry_db','early_40_120_gain_vs_dry_db','tail150_suppression_vs_input_db','tail_150_300_dbfs','tail_300_600_dbfs'):
    v=np.array([r[metric] for r in g]);summary.append({'rir':rir,'backend':method,'metric':metric,'n':len(v),'median':float(np.median(v)),'p10':float(np.percentile(v,10)),'p90':float(np.percentile(v,90)),'min':float(v.min()),'max':float(v.max())})
 with (OUT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (OUT/'runtime.json').write_text(json.dumps({'device':'cpu','load_seconds':load,'inference_total_seconds':infer_total,'clips':len(rows)//3,'rtf_total':infer_total/(3.*len(rows)/3),'checkpoint_sha256':weight_hash,'checkpoint_license':'unclear; do not redistribute','model_rate_hz':RATE16},indent=2)+'\n')
 print(json.dumps({'load_s':load,'infer_s':infer_total,'cases':len(rows)//3,'summary':summary},indent=2))
if __name__=='__main__':main()
