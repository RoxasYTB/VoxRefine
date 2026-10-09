#!/usr/bin/env python3
"""Analyze dereverb candidates on one recording without claiming dry-reference truth.

Candidates are pre-rendered, duration-aligned mono/stereo WAVs. Speech offsets are
selected from one supplied denoised reference (never per candidate); report tail
energy relative to the 100 ms before each offset and onset energy relative to the
same reference. All listening files get one fixed gain per file from active RMS.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft


def read(path: Path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr

def db(x): return 20*np.log10(np.maximum(np.asarray(x), 1e-12))
def rms_frames(x, frame):
    n=len(x)//frame
    return np.sqrt(np.mean(x[:n*frame].reshape(n,frame)**2,axis=1)+1e-24)
def segments(mask):
    d=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def detect_events(reference, sr, frame_ms=10, bridge_ms=40, min_speech_ms=100, min_pause_ms=80):
    frame=round(sr*frame_ms/1000); env=rms_frames(reference,frame); level=db(env)
    # VAD is driven by the denoised reference only; decisions are frozen for every candidate.
    threshold=float(np.percentile(level,20)+8.0)
    active=level>threshold
    # bridge brief intra-word energy dips
    maxgap=max(1,round(bridge_ms/frame_ms))
    seg=segments(~active)
    for a,b in seg:
        if b-a<=maxgap and a>0 and b<len(active): active[a:b]=True
    speech=[(a,b) for a,b in segments(active) if b-a>=round(min_speech_ms/frame_ms)]
    events=[]; minpause=round(min_pause_ms/frame_ms)
    for (a,b),(c,d) in zip(speech,speech[1:]):
        if c-b>=minpause:
            events.append({'offset_frame':int(b),'next_onset_frame':int(c),'pause_frames':int(c-b),'terminal':False})
    if speech and len(active)-speech[-1][1]>=minpause:
        b=speech[-1][1]
        events.append({'offset_frame':int(b),'next_onset_frame':len(active),'pause_frames':int(len(active)-b),'terminal':True})
    return frame,threshold,speech,events

def band_energy_db(x, sr, lo, hi):
    # Waveform RMS after FFT band selection, windowed; diagnostic only.
    if len(x)<2: return float('nan')
    win=np.hanning(len(x)); X=np.fft.rfft(x*win); f=np.fft.rfftfreq(len(x),1/sr)
    m=(f>=lo)&(f<hi)
    return float(10*np.log10(np.sum(np.abs(X[m])**2)+1e-24))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True,type=Path,help='MossFormer2 baseline used to select speech activity')
    ap.add_argument('--candidate',action='append',required=True,help='LABEL=PATH; repeat; include Adobe and every candidate')
    ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    ref,sr=read(args.reference); parsed={'MossFormer2':(ref,sr)}
    for item in args.candidate:
        label,path=item.split('=',1); x,s=read(Path(path))
        if s!=sr or len(x)!=len(ref): raise ValueError(f'{label}: sample rate/duration differs')
        parsed[label]=(x,s)
    frame,threshold,speech,events=detect_events(ref,sr)
    if not events: raise RuntimeError('No stable speech-offset / pause events found; inspect VAD threshold.')
    active_mask=np.zeros(len(rms_frames(ref,frame)),dtype=bool)
    for a,b in speech:active_mask[a:b]=True
    ref_frames=rms_frames(ref,frame); target=float(np.median(ref_frames[active_mask]))
    metrics=[]; audition={}; matched={}
    for label,(x,s) in parsed.items():
        xx=rms_frames(x,frame); n=min(len(xx),len(active_mask)); gain=target/(np.median(xx[:n][active_mask[:n]])+1e-12)
        y=x*gain; audition[label]=y; matched[label]=gain
        row={'method':label,'fixed_audition_gain_db':float(db(gain)),'rms_dbfs':float(db(np.sqrt(np.mean(x*x)))),'peak_dbfs':float(db(np.max(np.abs(x)))),'duration_seconds':len(x)/s,'clipped_samples':int(np.sum(np.abs(y)>=1))}
        # Long pauses estimate the residual floor descriptively; no claim of isolated noise.
        pauses=[(a,b) for a,b in segments(~active_mask) if (b-a)*frame/s>=.25]
        noise=[]
        for a,b in pauses:
            q=y[a*frame:min(b*frame,len(y))]
            if len(q):noise.append(float(db(np.sqrt(np.mean(q*q)+1e-24))))
        row['long_pause_rms_dbfs_median']=float(np.median(noise)) if noise else None
        for i,e in enumerate(events):
            t=e['offset_frame']*frame
            # Each window energy is relative to the preceding active 100 ms.
            pre=y[max(0,t-round(.1*s)):t]
            if len(pre)<round(.06*s):continue
            base=np.sqrt(np.mean(pre*pre)+1e-24)
            # Censor every window touched by the next onset; otherwise following
            # speech would be falsely counted as reverberation tail.
            valid_until=e['next_onset_frame']*frame
            for name,a,b in [('early_0_50ms',0,.05),('tail_50_150ms',.05,.15),('tail_150_300ms',.15,.30),('tail_300_600ms',.30,.60)]:
                q0=t+round(a*s);q1=t+round(b*s)
                measured_end=min(q1,len(y),valid_until if not e['terminal'] else len(y))
                q=y[q0:measured_end]
                valid=(q1<=valid_until and q1<=len(y)) or (e['terminal'] and len(q)>=round(.04*s))
                row[f'event{i+1}_{name}_db_vs_pre']=float(db(np.sqrt(np.mean(q*q)+1e-24)/base)) if len(q) and valid else None
                row[f'event{i+1}_{name}_measured_ms']=len(q)*1000/s if valid else 0.0
            start=e['next_onset_frame']*frame
            preon=y[max(0,start-round(.08*s)):start]; onset=y[start:min(len(y),start+round(.04*s))]; early=y[start+round(.04*s):min(len(y),start+round(.12*s))]
            for name,q in [('pre_onset_80ms',preon),('onset_0_40ms',onset),('early_speech_40_120ms',early)]:
                row[f'event{i+1}_{name}_dbfs']=float(db(np.sqrt(np.mean(q*q)+1e-24))) if len(q) else None
        metrics.append(row)
        fname=''.join(ch.lower() if ch.isalnum() else '-' for ch in label).strip('-')
        sf.write(args.out/f'{fname}-levelmatched.wav',y,s,subtype='PCM_24')
    with (args.out/'event-metrics.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in metrics for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(metrics)
    # Plot 1: matched spectrograms. Plot 2: same offset-relative tail curves per event.
    fig,axs=plt.subplots(len(parsed),1,figsize=(12,3.3*len(parsed)),sharex=True,sharey=True,squeeze=False,constrained_layout=True)
    for ax,(label,y) in zip(axs[:,0],audition.items()):
        f,t,z=stft(y,fs=sr,nperseg=1024,noverlap=768,boundary=None,padded=False);keep=(f>=80)&(f<=16000)
        im=ax.pcolormesh(t,f[keep]/1000,20*np.log10(np.abs(z[keep])+1e-8),shading='auto',vmin=-110,vmax=-35,cmap='magma')
        ax.set_title(label);ax.set_ylabel('kHz')
        for e in events:ax.axvline(e['offset_frame']*frame/s,color='cyan',lw=.7,alpha=.6)
    axs[-1,0].set_xlabel('Time (s)');fig.colorbar(im,ax=axs[:,0].tolist(),label='Magnitude dBFS; fixed active-level gain',shrink=.8);fig.savefig(args.out/'spectrograms.png',dpi=160);plt.close(fig)
    windows=['early_0_50ms','tail_50_150ms','tail_150_300ms','tail_300_600ms']
    fig,axs=plt.subplots(1,len(events),figsize=(4*len(events),4),squeeze=False,constrained_layout=True)
    for i,e in enumerate(events):
        ax=axs[0,i]
        for label,row in zip([r for r in parsed],metrics):
            vals=[row.get(f'event{i+1}_{w}_db_vs_pre') for w in windows]
            ax.plot(windows,vals,marker='o',label=label)
        ax.set_title(f"Offset {e['offset_frame']*frame/s:.2f}s");ax.set_ylabel('RMS relatif au pré-offset (dB)');ax.tick_params(axis='x',rotation=35);ax.grid(alpha=.25)
    axs[0,0].legend(fontsize=7);fig.savefig(args.out/'offset-decay.png',dpi=160);plt.close(fig)
    report={'sample_rate_hz':sr,'duration_seconds':len(ref)/sr,'reference_for_fixed_event_selection':str(args.reference),'vad_threshold_dbfs':threshold,'speech_regions_seconds':[[a*frame/sr,b*frame/sr] for a,b in speech],'offsets_seconds':[e['offset_frame']*frame/sr for e in events],'offset_event_details':events,'methods':metrics,'limits':['No dry/anechoic reference: offset-decay is descriptive, not RT60 or proof of dereverberation.','Speech offsets selected once from the denoised reference; short conversational pauses can contain speech.','Long-pause energy can include residual noise and room response.','Fixed gain changes listening level only; no EQ/compression applied.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'out':str(args.out),'offsets':report['offsets_seconds'],'vad_threshold_dbfs':threshold,'rows':len(metrics)},indent=2))
if __name__=='__main__':main()
