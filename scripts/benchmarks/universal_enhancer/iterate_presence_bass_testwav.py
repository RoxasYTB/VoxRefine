#!/usr/bin/env python3
"""Listen-level-matched low-shelf sweep on the consumed test.wav sample only."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import lfilter,resample_poly,stft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/user-recording-test-2026-10-09'
OUT=BASE/'stupase-bass-matrix-01'; OUT.mkdir(parents=True,exist_ok=True)
SR=48000; FRAME=960; HOP=480; TAIL=(5.79,5.94)
def read(path):
 x,s=sf.read(path,dtype='float64')
 return x,s
def mono(x): return x if x.ndim==1 else np.mean(x,axis=1)
def env(x):
 if x.ndim==1:x=x[:,None]
 n=1+(len(x)-FRAME)//HOP
 return np.array([np.sqrt(np.mean(x[i*HOP:i*HOP+FRAME]**2)) for i in range(n)])
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)+1e-30))
def db(x):return float(20*np.log10(max(float(x),1e-15)))
def lowshelf(x,fc,gain_db,slope=.8):
 A=10**(gain_db/40); w=2*np.pi*fc/SR; c=np.cos(w); s=np.sin(w)
 alpha=s/2*np.sqrt((A+1/A)*(1/slope-1)+2); beta=2*np.sqrt(A)*alpha
 b=np.array([A*((A+1)-(A-1)*c+beta),2*A*((A-1)-(A+1)*c),A*((A+1)-(A-1)*c-beta)])
 a=np.array([(A+1)+(A-1)*c+beta,-2*((A-1)+(A+1)*c),(A+1)+(A-1)*c-beta])
 return lfilter(b/a[0],a/a[0],x)
def frameband(x,lo,hi,active):
 # Parseval-normalized one-sided FFT band power, then 20ms frame RMS.
 win=np.hanning(FRAME); n=min(len(active),1+(len(x)-FRAME)//HOP); out=[]
 for i in range(n):
  z=np.fft.rfft(x[i*HOP:i*HOP+FRAME]*win); f=np.fft.rfftfreq(FRAME,1/SR); sel=(f>=lo)&(f<hi)
  weights=np.full(sel.sum(),2.0); bins=np.flatnonzero(sel)
  if len(bins) and bins[0]==0: weights[0]=1.0
  if len(bins) and bins[-1]==FRAME//2: weights[-1]=1.0
  power=np.sum(weights*np.abs(z[sel])**2)/(FRAME*np.sum(win**2))
  out.append(np.sqrt(power+1e-30))
 out=np.asarray(out); m=min(len(out),len(active)); return out[:m][active[:m]]
# A common mono working sequence and reference active mask from Cap60.
cap,_=read(BASE/'cap60-dereverb-screen-02/deepfilternet-cap60-aligned.wav'); cap=mono(cap)
cap_env=env(cap); active=cap_env>.035*np.max(np.abs(cap)); target_level=rms(cap_env[active])
base,_=read(BASE/'stupase-presence-matrix-02/12.wav'); base=mono(base)
adobe,_=read(BASE/'06_adobe_v2.wav')
raw,sr=read(Path.home()/'Bureau'/'test.wav'); raw=mono(raw)
if sr!=SR: raw=resample_poly(raw,SR//np.gcd(sr,SR),sr//np.gcd(sr,SR))
n=min(map(len,(cap,base,raw,adobe))); cap,base,raw,adobe=cap[:n],base[:n],raw[:n],adobe[:n]
# constant active speech RMS matching; Adobe kept stereo in output.
def speech_env(x): return env(x)
def match(x):
 e=speech_env(x); m=min(len(e),len(active)); g=target_level/rms(e[:m][active[:m]])
 return x*g,g
variants={0:('Original (niveau égalisé)',raw),1:('Adobe V2 (niveau égalisé)',adobe),2:('StuPASE .5 + EQ doux, base',base)}
for idx,g in zip(range(3,7),[.5,1.,1.5,2.]):variants[idx]=(f'Base + shelf graves {g:+.1f} dB @180 Hz',lowshelf(base,180,g))
rows=[]; renders={}
filenames={0:'00_original_levelmatched.wav',1:'01_adobe_v2_levelmatched.wav',2:'02_stupase_eq_base.wav',3:'03_bass_plus_0p5db.wav',4:'04_bass_plus_1db.wav',5:'05_bass_plus_1p5db.wav',6:'06_bass_plus_2db.wav'}
for idx,(name,x) in variants.items():
 y,g=match(x); renders[idx]=y
 sf.write(OUT/filenames[idx],y.astype('float32'),SR,subtype='PCM_24')
 ym=mono(y); e=env(y); m=min(len(e),len(cap_env),len(active)); tail=slice(round(TAIL[0]*SR),round(TAIL[1]*SR))
 b80_250=rms(frameband(ym,80,250,active)); b250_1k=rms(frameband(ym,250,1000,active)); b1_3k=rms(frameband(ym,1000,3000,active))
 rows.append({'id':f'{idx:02d}','candidate':name,'active_speech_gain_db':db(g),'bass_80_250_db_relative_speech':db(b80_250/target_level),'lowmid_250_1k_db_relative_speech':db(b250_1k/target_level),'mid_1_3k_db_relative_speech':db(b1_3k/target_level),'tail_5.79_5.94_db_relative_speech':db(rms(ym[tail])/target_level),'peak':float(np.max(np.abs(y))),'clipped_samples':int(np.count_nonzero(np.abs(y)>=1))})
report={'scope':'single consumed user recording test.wav; descriptive iteration, not a general preset or independent benchmark','sample_rate_hz':SR,'sample_duration_s':n/SR,'level_match':'constant gain to Cap60 active-speech RMS, using 20ms frames and same speech mask','bass_eq':'RBJ low shelf, 180 Hz, S=.8; gains +0.5, +1.0, +1.5, +2.0 dB','tail_window_s':TAIL,'rows':rows}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(3,1,figsize=(12,13),constrained_layout=True)
cols=['#333333','#dd8d00','#777777','#168a69','#2e86ab','#8249a1','#ce5a20']
for idx,col in zip(renders,cols):
 x=mono(renders[idx]); f,t,z=stft(x,fs=SR,nperseg=2048,noverlap=1536,boundary=None); am=np.interp(t,np.arange(len(active))*HOP/SR,active.astype(float),left=0,right=0)>.5; p=np.mean(abs(z[:,am])**2,axis=1)
 axs[0].plot(f,10*np.log10(np.maximum(p,1e-16)),label=variants[idx][0],color=col,lw=1.3)
 axs[1].plot(f,10*np.log10(np.maximum(p,1e-16)),label=variants[idx][0],color=col,lw=1.3)
 e=env(renders[idx])
 axs[2].plot(np.arange(len(e))*HOP/SR,20*np.log10(np.maximum(e/target_level,1e-9)),label=variants[idx][0],color=col,lw=1.2)
axs[0].set(xlim=(30,5000),ylabel='PSD actif (dB)',title='Courbe spectrale active — comparaison complète');axs[0].grid(alpha=.25);axs[0].legend(fontsize=8,ncol=2)
axs[1].set(xlim=(30,500),ylabel='PSD actif (dB)',title='Zoom sur les basses (30–500 Hz)');axs[1].grid(alpha=.25);axs[1].legend(fontsize=8,ncol=2)
axs[2].axvspan(*TAIL,color='gold',alpha=.2);axs[2].set(xlim=(0,n/SR),ylim=(-80,5),xlabel='Temps (s)',ylabel='RMS / RMS parole (dB)',title='Enveloppes — fenêtre de queue terminale');axs[2].grid(alpha=.25);axs[2].legend(fontsize=8,ncol=2)
fig.savefig(OUT/'bass-sweep-comparison.png',dpi=170);plt.close(fig)
print(json.dumps(report,indent=2))
