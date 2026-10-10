#!/usr/bin/env python3
"""Small post-StuPASE presence matrix for consumed test.wav only."""
from pathlib import Path
import csv,json
import numpy as np,soundfile as sf
from scipy.signal import butter,sosfiltfilt,lfilter,stft
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]; BASE=ROOT/'results/user-recording-test-2026-10-09'; OUT=BASE/'stupase-presence-matrix-02'; OUT.mkdir(parents=True,exist_ok=True)
SR=48000; N=960; H=480; TAIL=(5.79,5.94)
def read(p):
 x,s=sf.read(p,dtype='float64'); assert s==SR
 return x.mean(axis=1) if x.ndim==2 else x
def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)+1e-30))
def db(x): return float(20*np.log10(max(float(x),1e-15)))
def env(x):
 return np.array([rms(x[i*H:i*H+N]) for i in range(1+(len(x)-N)//H)])
def filt(x,lo,hi=None):
 sos=butter(6,lo if hi is None else [lo,hi],btype='highpass' if hi is None else 'bandpass',fs=SR,output='sos')
 return sosfiltfilt(sos,x)
def peaking(x,fc,q,gdb):
 w=2*np.pi*fc/SR; A=10**(gdb/40); al=np.sin(w)/(2*q); c=np.cos(w)
 b=np.array([1+al*A,-2*c,1-al*A]); a=np.array([1+al/A,-2*c,1-al/A])
 return lfilter(b/a[0],a/a[0],x)
def levelshelf(x,fc,gdb):
 A=10**(gdb/40); w=2*np.pi*fc/SR; c=np.cos(w); s=np.sin(w); S=.8
 al=s/2*np.sqrt((A+1/A)*(1/S-1)+2); be=2*np.sqrt(A)*al
 b=np.array([A*((A+1)+(A-1)*c+be),-2*A*((A-1)+(A+1)*c),A*((A+1)+(A-1)*c-be)])
 a=np.array([(A+1)-(A-1)*c+be,2*((A-1)-(A+1)*c),(A+1)-(A-1)*c-be])
 return lfilter(b/a[0],a/a[0],x)
def match(x,active,speech_level):
 e=env(x); g=speech_level/rms(e[:len(active)][active[:len(e)]])
 return x*g,g
def blevel(x,active,lo,hi):
 e=env(filt(x,lo,hi)); m=min(len(e),len(active)); return rms(e[:m][active[:m]])
cap=read(BASE/'cap60-dereverb-screen-02/deepfilternet-cap60-aligned.wav'); stu=read(BASE/'stupase-testwav-cfg025-01/stupase-cap60-raw-48k.wav'); adobe=read(BASE/'06_adobe_v2.wav'); n=min(len(cap),len(stu),len(adobe)); cap,stu,adobe=[x[:n] for x in (cap,stu,adobe)]
ce=env(cap); active=ce>.035*np.max(np.abs(cap)); speech_level=rms(ce[active]); previous=np.r_[ce[0],ce[:-1]]; onset=active & (ce>1.5*np.maximum(previous,1e-10)); tail=slice(round(TAIL[0]*SR),round(TAIL[1]*SR));
cap_hf=filt(cap,7800); stu_base,g0=match(stu,active,speech_level)
# Frozen consonant-safe gate: open 20 ms before broadband activity and release
# over 40 ms. A 5 ms attack/release smoother avoids a hard binary edge.
lookahead=2
target=np.zeros_like(active,dtype=bool)
for ahead in range(lookahead+1):
    target[:len(target)-ahead if ahead else len(target)] |= active[ahead:]
gate_frames=np.zeros(len(target),dtype=float); prev=0.0; dt=H/SR
for i,wanted in enumerate(target):
    tau=.005 if wanted else .040
    coeff=np.exp(-dt/tau)
    prev=(1-coeff)*float(wanted)+coeff*prev
    gate_frames[i]=prev
tt=np.arange(len(gate_frames))*H/SR; gate=np.interp(np.arange(n)/SR,tt,gate_frames,left=0,right=0)
base_eq=levelshelf(peaking(stu,3500,.72,1.5),5500,2.0)
upper_eq=peaking(stu,5400,.58,3.0)
variants={
 '00 StuPASE baseline':stu,
 '01 Presence EQ +1.5 dB 3.5k +2 dB 5.5k':base_eq,
 '02 Stronger EQ +3 dB 5.4k':upper_eq,
 '03 HF Cap60 -18 dB':stu+.125*cap_hf,
 '04 HF Cap60 -12 dB':stu+.25*cap_hf,
 '05 HF -18 dB gated':stu+.125*cap_hf*gate,
 '06 HF -12 dB gated':stu+.25*cap_hf*gate,
 '07 HF -18 dB gated + soft EQ':base_eq+.125*cap_hf*gate,
 '08 HF -12 dB gated + soft EQ':base_eq+.25*cap_hf*gate,
 '09 HF -8.5 dB gated':stu+0.375*cap_hf*gate,
 '10 HF -6 dB gated':stu+.5*cap_hf*gate,
 '11 HF -8.5 dB gated + soft EQ':base_eq+.375*cap_hf*gate,
 '12 HF -6 dB gated + soft EQ':base_eq+.5*cap_hf*gate,
}
# Adobe is descriptive reference only, not a fitted training target.
rows=[]; audio={}
for name,x in variants.items():
 y,g=match(x,active,speech_level); audio[name]=y
 file=OUT/f'{int(name[:2]):02d}.wav'; sf.write(file,y.astype('float32'),SR,subtype='PCM_24')
 ee=env(y); ee=ee[:len(active)]; ratio=20*np.log10(np.maximum(ee,1e-12)/np.maximum(ce[:len(ee)],1e-12))
 hfy=filt(y,7800); hf_env=env(hfy); m=min(len(active),len(hf_env));
 active_hf=rms(hf_env[:m][active[:m]]); tail_hf=rms(hfy[tail]);
 m0=min(len(ratio),len(onset),len(ce)); weak=(active[:m0]) & (ce[:m0]<np.percentile(ce[:m0][active[:m0]],35))
 hf_cap_env=env(cap_hf)[:m0]; top_hf=hf_cap_env>=np.percentile(hf_cap_env[active[:m0]],80)
 hf_bins=np.abs(stft(y,fs=SR,nperseg=2048,noverlap=1536,boundary=None)[2])
 f_bins=stft(y,fs=SR,nperseg=2048,noverlap=1536,boundary=None)[0]
 spec_t=stft(y,fs=SR,nperseg=2048,noverlap=1536,boundary=None)[1]
 use=(f_bins>=8000)&(f_bins<=12000); act_spec=np.interp(spec_t,np.arange(len(active))*H/SR,active.astype(float),left=0,right=0)>.5
 spec_power=np.maximum(hf_bins[use][:,act_spec]**2,1e-20)
 flatness=float(np.mean(np.exp(np.mean(np.log(spec_power),axis=0))/np.mean(spec_power,axis=0))) if spec_power.size else None
 p95,p99=np.percentile(hf_env[:m0][active[:m0]],[95,99])
 rows.append({'id':name[:2],'candidate':name,'level_match_gain_db':db(g),'tail_total_db_rel_speech':db(rms(y[tail])/speech_level),'tail_hf_db_rel_speech':db(tail_hf/speech_level),'active_hf_db_rel_speech':db(active_hf/speech_level),'hf_speech_to_tail_db':db(active_hf/max(tail_hf,1e-15)),'active_p10_vs_cap60_db':float(np.percentile(ratio[active[:len(ratio)]],10)),'weak_p10_vs_cap60_db':float(np.percentile(ratio[weak],10)),'onset_p10_vs_cap60_db':float(np.percentile(ratio[:m0][onset[:m0]],10)),'hf_burst_coverage_gate_ge_0_9':float(np.mean(gate_frames[:m0][top_hf]>=.9)) if top_hf.any() else None,'active_hf_p95_db_rel_speech':db(p95/speech_level),'active_hf_p99_db_rel_speech':db(p99/speech_level),'spectral_flatness_8_12k_active':flatness,'peak':float(np.max(np.abs(y))),'clipped':int(np.sum(np.abs(y)>=1))})
# refs
for name,x in [('Adobe V2',adobe),('Cap60',cap)]:
 y,g=match(x,active,speech_level); hf_env=env(filt(y,7800)); m=min(len(active),len(hf_env))
 zf,zt,zz=stft(y,fs=SR,nperseg=2048,noverlap=1536,boundary=None); use=(zf>=8000)&(zf<=12000); am=np.interp(zt,np.arange(len(active))*H/SR,active.astype(float),left=0,right=0)>.5
 pp=np.maximum(abs(zz[use][:,am])**2,1e-20); flat=float(np.mean(np.exp(np.mean(np.log(pp),axis=0))/np.mean(pp,axis=0))) if pp.size else None
 hp95,hp99=np.percentile(hf_env[:m][active[:m]],[95,99])
 rows.append({'id':'REF','candidate':name,'level_match_gain_db':db(g),'tail_total_db_rel_speech':db(rms(y[tail])/speech_level),'tail_hf_db_rel_speech':db(rms(filt(y,7800)[tail])/speech_level),'active_hf_db_rel_speech':db(rms(hf_env[:m][active[:m]])/speech_level),'hf_speech_to_tail_db':db(rms(hf_env[:m][active[:m]])/max(rms(filt(y,7800)[tail]),1e-15)),'active_p10_vs_cap60_db':None,'weak_p10_vs_cap60_db':None,'onset_p10_vs_cap60_db':None,'hf_burst_coverage_gate_ge_0_9':None,'active_hf_p95_db_rel_speech':db(hp95/speech_level),'active_hf_p99_db_rel_speech':db(hp99/speech_level),'spectral_flatness_8_12k_active':flat,'peak':float(np.max(np.abs(y))),'clipped':int(np.sum(np.abs(y)>=1))})
report={'scope':'single user test.wav; exploratory consumed sample; no training; Adobe used descriptively only','mask':'Cap60 RMS 20ms >3.5% cap60 peak; hop 10ms','tail_window_s':TAIL,'audio_band_note':'StuPASE is 16 kHz and cannot reconstruct content above 8 kHz; 7.8kHz+ residual comes from Cap60. HF gains tested: 0.125 (-18.06 dB), 0.25 (-12.04 dB), 0.375 (-8.52 dB), 0.5 (-6.02 dB). Frozen gate opens 20ms early, 5ms attack, 40ms release.','rows':rows}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');
with (OUT/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# active spectral PSD & tail envelope
fig,axs=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
colors=['#de8b19','#168a69','#555','#bd3c13','#9a4f96','#277da1','#40a6a0']
sel=['Adobe V2','Cap60','00 StuPASE baseline','01 Presence EQ +1.5 dB 3.5k +2 dB 5.5k','10 HF -6 dB gated','12 HF -6 dB gated + soft EQ']
for name,col in zip(sel,colors+['#285f99']):
 x=audio[name] if name in audio else match(adobe if name=='Adobe V2' else cap,active,speech_level)[0]
 f,t,z=stft(x,fs=SR,nperseg=2048,noverlap=1536,boundary=None); am=np.interp(t,np.arange(len(active))*H/SR,active.astype(float),left=0,right=0)>.5
 psd=np.mean(abs(z[:,am])**2,axis=1); axs[0].plot(f,10*np.log10(np.maximum(psd,1e-16)),label=name,color=col,lw=1.3)
axs[0].set(xlim=(100,14000),ylabel='PSD actif (dB)',title='Spectre de parole actif — restitution HF contrôlée');axs[0].grid(alpha=.25);axs[0].legend(fontsize=8,ncol=2)
for name,col in zip(sel,colors+['#285f99']):
 x=audio[name] if name in audio else match(adobe if name=='Adobe V2' else cap,active,speech_level)[0]
 e=env(x); axs[1].plot(np.arange(len(e))*H/SR,20*np.log10(np.maximum(e/speech_level,1e-9)),label=name,color=col,lw=1.2)
axs[1].axvspan(*TAIL,color='gold',alpha=.2);axs[1].set(xlim=(5.25,6.18),ylim=(-80,5),xlabel='Temps (s)',ylabel='RMS / RMS parole (dB)',title='Queue terminale (fenêtre courte, n=1)');axs[1].grid(alpha=.25);axs[1].legend(fontsize=8,ncol=2)
fig.savefig(OUT/'presence-matrix02.png',dpi=170);plt.close(fig)
print(json.dumps(report,indent=2))
