#!/usr/bin/env python3
"""Frozen MossFormer2 -> mono WPE dereverb screen on known synthetic RIR truth.

Reuses an existing four-speaker VCTK held-out RIR-only subset. It never tunes
parameters; the exact early target and digital silence after the weak event make
late-tail suppression and early-speech preservation independently measurable.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/'results/noise-rir-truth-01/vctk-rir-dereverb-truth-01'
RUN=BENCH/'mossformer2-wpe-screen-01'
RATE=48000
sys.path.insert(0,str(ROOT/'scripts/benchmarks/universal_enhancer'))
from bench_vctk_wpe_truth import wpe

def read(p):
    x,s=sf.read(p,dtype='float64',always_2d=True)
    if s!=RATE: raise ValueError(f'{p}: {s} Hz')
    return x.mean(axis=1)
def db_energy(x): return float(10*np.log10(np.mean(np.asarray(x,dtype=np.float64)**2)+1e-24))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    selection=json.loads((RUN/'selection.json').read_text())['conditions']
    moss=RUN/'mossformer2'
    if not moss.is_dir() or len(list(moss.glob('*.wav')))!=len(selection):
        raise RuntimeError('Expected the completed 24-clip MossFormer2 render set')
    renders=RUN/'wpe-after-mossformer2';renders.mkdir(exist_ok=True)
    prior={}
    prior_csv=RUN/'event-metrics.csv'
    if prior_csv.exists():
        with prior_csv.open(newline='') as f:
            prior={(r['speaker'],r['utterance'],r['rir']):float(r.get('wpe_seconds',0) or 0) for r in csv.DictReader(f)}
    rows=[];elapsed=0.
    for c in selection:
        source=ROOT/c['input']; drypath=ROOT/c['dry_reference']; inp=read(source); dry=read(drypath)
        moss_path=moss/source.name; m=read(moss_path)
        if not (len(inp)==len(dry)==len(m)): raise ValueError(f'length mismatch: {source.name}')
        outpath=renders/source.name.replace('.wav','-wpe.wav')
        key=(c['speaker'],c['utterance'],c['rir'])
        if outpath.exists():
            y=read(outpath);sec=prior.get(key,0.0)
        else:
            t=time.perf_counter(); y=wpe(m); sec=time.perf_counter()-t
            sf.write(outpath,y,RATE,subtype='PCM_16'); y=read(outpath)
        elapsed+=sec
        start=round(float(c['event_start_seconds'])*RATE); end=start+round(float(c['event_duration_seconds'])*RATE)
        rr={'speaker':c['speaker'],'utterance':c['utterance'],'rir':c['rir'],'backend':'wpe-after-mossformer2',
            'input_sha256':sha(source),'mossformer2_sha256':sha(moss_path),'output_sha256':sha(outpath),
            'duration_s':len(y)/RATE,'clipped_samples':int(np.sum(np.abs(y)>=1.)),
            'wpe_seconds':sec,'wpe_rtf':sec/(len(y)/RATE),'event_start_s':start/RATE}
        for method,sig in [('input',inp),('mossformer2',m),('wpe_after_moss',y)]:
            rr[f'{method}_event_energy_delta_vs_dry_db']=db_energy(sig[start:end])-db_energy(dry[start:end])
            for label,a,b in [('onset_0_40',0,.04),('early_40_120',.04,.12)]:
                lo=start+round(a*RATE);hi=start+round(b*RATE)
                rr[f'{method}_{label}_gain_vs_dry_db']=db_energy(sig[lo:hi])-db_energy(dry[lo:hi])
            for label,a,b in [('tail_50_150',.05,.15),('tail_150_300',.15,.30),('tail_300_600',.30,.60)]:
                lo=end+round(a*RATE);hi=min(len(sig),end+round(b*RATE))
                rr[f'{method}_{label}_dbfs']=db_energy(sig[lo:hi])
        for label in ('tail_50_150','tail_150_300','tail_300_600'):
            rr[f'wpe_{label}_suppression_vs_input_db']=rr[f'input_{label}_dbfs']-rr[f'wpe_after_moss_{label}_dbfs']
            rr[f'moss_{label}_suppression_vs_input_db']=rr[f'input_{label}_dbfs']-rr[f'mossformer2_{label}_dbfs']
        rows.append(rr)
        print(f"{c['speaker']}/{c['utterance']}/{c['rir']}: WPE {sec:.2f}s",flush=True)
    with (RUN/'event-metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for rir in ('t60-0.45s','t60-0.80s'):
        for key in ('mossformer2','wpe_after_moss'):
            group=[r for r in rows if r['rir']==rir]
            for metric in (f'{key}_event_energy_delta_vs_dry_db',f'{key}_onset_0_40_gain_vs_dry_db',f'{key}_early_40_120_gain_vs_dry_db',f'{key}_tail_150_300_dbfs',f'{key}_tail_300_600_dbfs'):
                vals=np.array([r[metric] for r in group],dtype=float)
                summary.append({'rir':rir,'backend':key,'metric':metric,'n':len(vals),'median':float(np.median(vals)),'p10':float(np.percentile(vals,10)),'p90':float(np.percentile(vals,90)),'min':float(vals.min()),'max':float(vals.max())})
            for metric in ('wpe_tail_150_300_suppression_vs_input_db','wpe_tail_300_600_suppression_vs_input_db','moss_tail_150_300_suppression_vs_input_db','moss_tail_300_600_suppression_vs_input_db'):
                if key=='wpe_after_moss' or metric.startswith('moss_'):
                    vals=np.array([r[metric] for r in group],dtype=float)
                    summary.append({'rir':rir,'backend':key,'metric':metric,'n':len(vals),'median':float(np.median(vals)),'p10':float(np.percentile(vals,10)),'p90':float(np.percentile(vals,90)),'min':float(vals.min()),'max':float(vals.max())})
    with (RUN/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    # Paired Pareto: output tail attenuation vs preserved weak-event energy.
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    colors={'t60-0.45s':'#2673b8','t60-0.80s':'#d65f17'}
    for ax,tail in zip(axs,('tail_150_300','tail_300_600')):
        for rir in colors:
            g=[r for r in rows if r['rir']==rir]
            for method,label,marker in [('mossformer2','MossFormer2','o'),('wpe_after_moss','MossFormer2 → WPE','^')]:
                ax.scatter([r[f'wpe_{tail}_suppression_vs_input_db'] if method=='wpe_after_moss' else r[f'moss_{tail}_suppression_vs_input_db'] for r in g],
                           [r[f'{method}_event_energy_delta_vs_dry_db'] for r in g],label=f'{label}, {rir}',marker=marker,color=colors[rir],alpha=.75)
        ax.axhline(0,color='black',lw=.8);ax.axvline(8,color='#666',ls='--',lw=.8)
        ax.set_xlabel(f'Réduction de queue {tail.replace("_","–")} vs entrée (dB)');ax.set_ylabel('Énergie événement faible vs cible sèche (dB)');ax.grid(alpha=.25)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(RUN/'weak-event-tail-pareto.png',dpi=160);plt.close(fig)
    runtime=json.loads((RUN/'runtime.json').read_text());runtime.update({'wpe_seconds_total':elapsed,'wpe_rtf_total':elapsed/(3.*len(rows)),'cases':len(rows),'wpe_settings':{'rate_hz':RATE,'n_fft':1024,'hop':256,'taps':32,'delay_frames':3,'iterations':3,'regularization':1e-6}})
    (RUN/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'mossformer2':runtime,'median':{x:float(np.median([r[x] for r in rows])) for x in ('wpe_tail_150_300_suppression_vs_input_db','wpe_tail_300_600_suppression_vs_input_db','wpe_after_moss_event_energy_delta_vs_dry_db','wpe_after_moss_onset_0_40_gain_vs_dry_db')}},indent=2))
if __name__=='__main__':main()
