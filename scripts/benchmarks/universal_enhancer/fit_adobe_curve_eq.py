#!/usr/bin/env python3
"""Fit and cross-validate a shared, smooth IIR EQ against paired Adobe renders."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np,soundfile as sf,pyloudnorm as pyln
from scipy.signal import stft,sosfilt
from scipy.optimize import lsq_linear
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.benchmarks.universal_enhancer.tone_balance_sweep import spectral_curve,frame_rms,db
from voxrefine.toneshape import apply_peaking_eq
from scipy.signal import iirpeak
SR=48000;OUT=ROOT/'results/adobe-curve-eq-2026-10-09';OUT.mkdir(parents=True,exist_ok=True)
# All paths below are exact pairs already present in the repository.
BASE=ROOT/'results/noise-rir-truth-01/rir-tail-truth-02'
PAIRS=[
 {'id':'controlled-pauses-snr10','candidate':ROOT/'results/resemble-enhance-01/controlled-challenger-03/audio/noise-snr10-adobe-resemble-nfe16-C.wav','adobe':ROOT/'results/resemble-enhance-01/controlled-challenger-03/audio/adobe-v2-reference.wav','dry':ROOT/'results/resemble-enhance-01/controlled-challenger-03/audio/noise-snr10-adobe-dry.wav','mask':'max-35db'},
]
for speaker,event in [('christiane',1),('christiane',2),('naf',2)]:
 for condition in ['dry-control','rir-only']:
  tag=f'{speaker}-event-{event}-{condition}';stem=f'{speaker}-event-{event}'
  PAIRS.append({'id':tag,'candidate':BASE/f'renders/{tag}-resemble-nfe64-c.wav','adobe':ROOT/f'results/noise-rir-truth-01/adobe-pair-{tag}/adobe-v2-output.wav','dry':BASE/f'inputs/{stem}-dry-control.wav','mask':'max-35db'})

def read(p):
 x,sr=sf.read(p,dtype='float64');
 if x.ndim==2:x=x.mean(axis=1)
 if sr!=SR:raise ValueError(f'{p}: sample rate {sr}')
 return x

def mask_for(dry,n):
 rms=frame_rms(dry);active=rms>=np.max(rms)*10**(-35/20)
 # pad mask to all STFT frames handled by clipped indices
 return np.convolve(np.pad(active.astype(np.int8),6),np.ones(13,dtype=np.int32),mode='same')[6:-6]>0

def fixed_lufs_gain(x,target_lufs):
 meter=pyln.Meter(SR);return x*10**((target_lufs-float(meter.integrated_loudness(x)))/20)

def curve(x,mask):return spectral_curve(x,mask)[1]

def rbj_sos(center,q,gain_db):
 # RBJ peaking EQ with peak gain in dB.
 w=2*np.pi*center/SR; A=10**(gain_db/40); alpha=np.sin(w)/(2*q);c=np.cos(w)
 b=np.array([1+alpha*A,-2*c,1-alpha*A]);a=np.array([1+alpha/A,-2*c,1-alpha/A]);return np.array([[b[0]/a[0],b[1]/a[0],b[2]/a[0],1,a[1]/a[0],a[2]/a[0]]])

items=[];f=None
for spec in PAIRS:
 for key in ['candidate','adobe','dry']:
  if not spec[key].is_file():break
 else:
  c,dry,adobe=[read(spec[k]) for k in ('candidate','dry','adobe')];n=min(map(len,(c,dry,adobe)));c,dry,adobe=[x[:n] for x in(c,dry,adobe)]
  mask=mask_for(dry,n);dry_lufs=float(pyln.Meter(SR).integrated_loudness(dry));c=fixed_lufs_gain(c,dry_lufs);adobe=fixed_lufs_gain(adobe,dry_lufs)
  f,cc=spectral_curve(c,mask);_,ac=spectral_curve(adobe,mask)
  freqs=np.fft.rfftfreq(2048,1/SR);keep=(freqs>=80)&(freqs<=12000)
  # curve helper uses same 2048 FFT bins.
  group='controlled' if spec['id'].startswith('controlled-') else 'christiane' if spec['id'].startswith('christiane-') else 'naf'
  items.append({'id':spec['id'],'group':group,'candidate':c,'adobe':adobe,'dry':dry,'dry_lufs':dry_lufs,'mask':mask,'base_curve':cc,'adobe_curve':ac,'freqs':f,'mae':float(np.mean(abs(cc[keep]-ac[keep])))})
if len(items)<4:raise RuntimeError(f'Only {len(items)} valid exact pairs')
# Fit broad IIR EQ families and choose with leave-one-speaker-group-out validation.
base_index=np.where((f>=80)&(f<=12000))[0]
def eq_response(center,gain):
 sos=rbj_sos(center,q,gain)
 # evaluate on linear FFT bins matching spectral curve output
 ff=np.fft.rfftfreq(2048,1/SR);_,h=__import__('scipy.signal',fromlist=['sosfreqz']).sosfreqz(sos,worN=ff,fs=SR)
 # interpolate to f curve (same FFT bins)
 return 20*np.log10(np.maximum(abs(h),1e-10))
families=[('3-bell',np.array([250,1500,6500],float),0.55),
          ('5-bell',np.array([180,500,1500,3500,8000],float),0.65),
          ('7-bell',np.array([180,350,700,1400,2800,5200,9000],float),0.75)]
groups=sorted({it['group'] for it in items})
def apply_model(x,centers,q,gains):
 return apply_peaking_eq(x,SR,tuple((float(center),float(q),float(gain)) for center,gain in zip(centers,gains)))
def fit_family(centers,q,training,lam,mu,bound=2.5):
 response=np.column_stack([eq_response(cn,1.0)[base_index] for cn in centers])
 # Give every speaker equal weight despite multiple recordings per voice.
 targets=[]
 for group in sorted({it['group'] for it in training}):
  targets.append(np.mean([it['adobe_curve'][base_index]-it['base_curve'][base_index]
                          for it in training if it['group']==group],axis=0))
 target=np.mean(targets,axis=0)
 smooth=np.diff(np.eye(len(centers)),n=2,axis=0) if len(centers)>2 else np.diff(np.eye(len(centers)),axis=0)
 A=np.vstack([response,lam*smooth,mu*np.eye(len(centers))])
 b=np.r_[target,np.zeros(smooth.shape[0]),np.zeros(len(centers))]
 return lsq_linear(A,b,bounds=(-bound,bound),lsmr_tol='auto').x,response

cv=[]
for family,centers,q in families:
 for lam in (1.0,3.0,6.0):
  for mu in (0.25,0.75,1.5):
   fold_scores=[]
   for held in groups:
    train=[it for it in items if it['group']!=held]
    test=[it for it in items if it['group']==held]
    gains,response=fit_family(centers,q,train,lam,mu)
    errors=[]
    for it in test:
     filtered=apply_model(it['candidate'],centers,q,gains)
     filtered=fixed_lufs_gain(filtered,it['dry_lufs'])
     eq_curve=curve(filtered,it['mask'])
     errors.append(float(np.mean(abs(eq_curve[base_index]-it['adobe_curve'][base_index]))))
    fold_scores.append(float(np.mean(errors)))
   cv.append({'family':family,'centers_hz':centers.tolist(),'q':q,'smooth_penalty':lam,'magnitude_penalty':mu,'lopo_mae_db':float(np.mean(fold_scores)),'fold_mae_db':fold_scores})
best_cv=min(cv,key=lambda x:x['lopo_mae_db'])
# Prefer fewer filters when validation is effectively tied (within 0.02 dB).
near=[row for row in cv if row['lopo_mae_db']<=best_cv['lopo_mae_db']+0.02]
chosen=min(near,key=lambda row:(len(row['centers_hz']),row['lopo_mae_db']))
centers=np.array(chosen['centers_hz']);q=chosen['q']
gains,response=fit_family(centers,q,items,chosen['smooth_penalty'],chosen['magnitude_penalty'])

def process(x):
 return apply_model(x,centers,q,gains)
rows=[];curve_rows=[]
for it in items:
 out=fixed_lufs_gain(process(it['candidate']),it['dry_lufs'])
 fitted=curve(out,it['mask']);d=it['adobe_curve']-it['base_curve'];mae=float(np.mean(abs(fitted[(f>=80)&(f<=12000)]-it['adobe_curve'][(f>=80)&(f<=12000)])))
 rows.append({'pair':it['id'],'baseline_mae_db':it['mae'],'eq_mae_db':mae,'improvement_db':it['mae']-mae,'speaker_group':it['group']})
 for idx,hzv in enumerate(f):curve_rows.append({'pair':it['id'],'frequency_hz':float(hzv),'baseline_minus_adobe_db':float(it['base_curve'][idx]-it['adobe_curve'][idx]),'eq_minus_adobe_db':float(fitted[idx]-it['adobe_curve'][idx])})
 # render corrected candidate; preserve the original candidate's integrated LUFS for listening.
 target_lufs=pyln.Meter(SR).integrated_loudness(it['candidate'])
 out_path=OUT/f'{it["id"]}-curve-eq.wav';listen=out*10**((target_lufs-pyln.Meter(SR).integrated_loudness(out))/20);sf.write(out_path,listen,SR,subtype='PCM_24')
 base_listen=it['candidate']*10**((target_lufs-pyln.Meter(SR).integrated_loudness(it['candidate']))/20)
 base_path=OUT/f'{it["id"]}-baseline.wav';sf.write(base_path,base_listen,SR,subtype='PCM_24')
 ref_listen=it['adobe']*10**((target_lufs-pyln.Meter(SR).integrated_loudness(it['adobe']))/20)
 ref_path=OUT/f'{it["id"]}-adobe.wav';sf.write(ref_path,ref_listen,SR,subtype='PCM_24')
 import subprocess
 for audio_path in (out_path,base_path,ref_path):
  subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(audio_path),'-codec:a','libmp3lame','-b:a','192k',str(audio_path.with_suffix('.mp3'))],check=True)

with (OUT/'metrics.csv').open('w',newline='') as stream:
 w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'curves.csv').open('w',newline='') as stream:
 w=csv.DictWriter(stream,fieldnames=list(curve_rows[0]));w.writeheader();w.writerows(curve_rows)
summary={'method':'shared broad RBJ peaking EQ; model family/regularization selected by leave-one-speaker-group-out CV over controlled, Christiane and Naf pairs; spectra use one fixed integrated-LUFS gain to the dry reference; MAE uses active speech 1/6-octave curves' ,'cross_validation':cv,'chosen_cv_model':chosen,'centers_hz':centers.tolist(),'q':q,'fitted_gains_db':gains.tolist(),'pairs':rows,'median_baseline_mae_db':float(np.median([r['baseline_mae_db'] for r in rows])),'median_eq_mae_db':float(np.median([r['eq_mae_db'] for r in rows])),'mean_gain_db':float(np.mean(gains)),'limitations':['Few distinct speakers and synthetic/RIR conditions; NAF holdout is only one speaker.','Average spectrum proximity is not a perceptual quality measure and cannot match time-varying denoising.','The fitted EQ is diagnostic and not enabled in the product.']}
(OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
for i,it in enumerate(items):
 fitted=curve(process(it['candidate']),it['mask']);col=plt.cm.tab10(i)
 axs[0].plot(f,it['base_curve'],color=col,alpha=.45,lw=1);axs[0].plot(f,fitted,color=col,lw=1.2,label=it['id']+' (EQ)');
 axs[1].plot(f,it['base_curve']-it['adobe_curve'],color=col,alpha=.45,lw=1);axs[1].plot(f,fitted-it['adobe_curve'],color=col,lw=1.2,label=it['id']+' (EQ)')
axs[1].axhline(0,color='black',lw=.8)
for ax in axs:ax.set_xscale('log');ax.set_xlim(100,12000);ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=6,ncol=2);ax.set_xlabel('Frequency (Hz)')
axs[0].set_title('Same-source curves: baseline (faint) and shared EQ (solid)');axs[0].set_ylabel('Speech-active power after fixed LUFS match (dB)')
axs[1].set_title('Residual against Adobe v2; faded baseline, solid shared EQ');axs[1].set_ylabel('Candidate − Adobe (dB)')
fig.savefig(OUT/'shared-eq-curves.png',dpi=170)
print(json.dumps(summary,indent=2))
