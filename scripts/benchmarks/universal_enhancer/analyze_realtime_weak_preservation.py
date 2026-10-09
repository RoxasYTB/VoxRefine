#!/usr/bin/env python3
"""Compare existing weak-event renders via exact speech/noise-stem projections."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/noise-rir-truth-01/weak-speech-backend-heldout-02'
NEW=ROOT/'results/noise-rir-truth-01/realtime-weak-preservation-01'
OUT=NEW
RATE=48000;EVENT_N=12000;START=round(1.25*RATE);FRAME=960;METHODS=('input','dpdfnet2_48khz','dfn18','dfn100','resemble-denoiser-only')

def read(p):
 x,sr=sf.read(p,dtype='float64')
 if sr!=RATE or x.ndim!=1 or not np.isfinite(x).all():raise ValueError(f'invalid audio {p}')
 return x

def db(x):return float(20*np.log10(max(abs(float(x)),1e-12)))
def frame_rms(x):
 n=len(x)//FRAME
 return np.sqrt(np.mean(x[:n*FRAME].reshape(n,FRAME)**2,axis=1)+1e-24)

def coeffs(s,n,y):
 c,*_=np.linalg.lstsq(np.column_stack((s,n)),y,rcond=None)
 voice=abs(float(c[0]));noise=abs(float(c[1]))
 return {'voice_gain_db':db(voice),'voice_loss_db':-db(voice),'noise_gain_db':db(noise),'projected_snr_db':float(10*np.log10((voice*voice*np.mean(s*s)+1e-20)/(noise*noise*np.mean(n*n)+1e-20)))}

def main():
 selected=json.loads((BASE/'selected-events.json').read_text())['events']
 ev={(r['voice'],int(r['event_id'])):r for r in selected}
 inputs=sorted((BASE/'inputs').glob('*-input.wav'));rows=[]
 for inp in inputs:
  stem=inp.name[:-len('-input.wav')];parts=stem.split('-');voice=parts[0];eid=int(parts[1][1:]);kind='-'.join(parts[2:-1]);snr=int(parts[-1][3:])
  meta=ev[(voice,eid)];speech=read(inp.with_name(stem+'-speech.wav'));noise=read(inp.with_name(stem+'-noise.wav'))
  lo=START;hi=lo+EVENT_N
  s=speech[lo:hi];n=noise[lo:hi]
  outs={
   'input':inp,
   'dpdfnet2_48khz':NEW/'renders'/(stem+'-dpdfnet2-48khz.wav'),
   'dfn18':BASE/'renders'/(stem+'-dfn18.wav'),
   'dfn100':BASE/'renders'/(stem+'-dfn100.wav'),
   'resemble-denoiser-only':BASE/'renders'/(stem+'-resemble.wav'),
  }
  for method,path in outs.items():
   y=read(path)
   if len(y)!=len(speech):raise ValueError(f'duration mismatch {path}: {len(y)} != {len(speech)}')
   p=coeffs(s,n,y[lo:hi])
   # Define weak frames from the clean speech stem, never from candidate output.
   clean_frames=frame_rms(speech)
   active=clean_frames >= np.max(clean_frames)*10**(-30/20)
   weak=active & (clean_frames <= np.percentile(clean_frames[active],25)) if np.any(active) else np.zeros_like(active,dtype=bool)
   out_frames=frame_rms(y)
   relative_db=20*np.log10((out_frames+1e-12)/(clean_frames+1e-12))
   weak_delta=relative_db[weak]
   exact_floor=(out_frames[weak] <= 1e-12) if np.any(weak) else np.array([],dtype=bool)
   weak_stats={'weak_p10_delta_db':float(np.percentile(weak_delta,10)) if weak_delta.size else None,'weak_p1_delta_db':float(np.percentile(weak_delta,1)) if weak_delta.size else None,'weak_fraction_below_minus20db':float(np.mean(weak_delta < -20)) if weak_delta.size else None,'weak_fraction_below_minus40db':float(np.mean(weak_delta < -40)) if weak_delta.size else None,'weak_fraction_below_minus80db':float(np.mean(weak_delta < -80)) if weak_delta.size else None,'weak_exact_floor_frames':int(np.sum(exact_floor)),'weak_frame_count':int(weak_delta.size)}
   if np.any(weak):
    weak_corr=float(np.corrcoef(np.log(clean_frames[weak]+1e-12),np.log(out_frames[weak]+1e-12))[0,1]) if np.std(out_frames[weak])>0 and np.std(clean_frames[weak])>0 else None
   else: weak_corr=None
   rows.append({'voice':voice,'event_id':eid,'noise_type':kind,'target_snr_db':snr,'source_event_rms_dbfs':meta['source_event_rms_dbfs'],'speech_active_fraction':meta['speech_active_fraction'],'backend':method,'rtf':None,'weak_envelope_corr':weak_corr,**weak_stats,**p,'output_peak_dbfs':db(np.max(np.abs(y)))})
 render=json.loads((NEW/'render-manifest.json').read_text())
 for r in rows:
  if r['backend']=='dpdfnet2_48khz':
   key=Path(r['voice']+f"-e{r['event_id']}-"+r['noise_type']+f"-snr{r['target_snr_db']}-dpdfnet2-48khz.wav")
   hit=next(x for x in render['outputs'] if Path(x['output']).name==key.name);r['rtf']=hit['rtf']
 # Existing heldout metadata has per-render timing for DFN/Resemble.
 # Use per-output timing from the frozen held-out event metrics.
 with (BASE/'event-metrics.csv').open(newline='') as f:
  oldrows=list(csv.DictReader(f))
 oldrtf={Path(x['output_path']).name:float(x['rtf']) for x in oldrows if x.get('output_path') and x.get('rtf')}
 for r in rows:
  if r['backend'] in ('dfn18','dfn100','resemble-denoiser-only'):
   suffix={'dfn18':'dfn18','dfn100':'dfn100','resemble-denoiser-only':'resemble'}[r['backend']]
   name=f"{r['voice']}-e{r['event_id']}-{r['noise_type']}-snr{r['target_snr_db']}-{suffix}.wav";r['rtf']=oldrtf.get(name)
 with (OUT/'event-metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Macro speaker: event median first, speakers equally weighted within condition.
 conditions=sorted({(r['noise_type'],r['target_snr_db']) for r in rows})
 summary=[]
 for kind,snr in conditions:
  for method in METHODS:
   events=[r for r in rows if r['noise_type']==kind and r['target_snr_db']==snr and r['backend']==method]
   speakers=[]
   for v in sorted({r['voice'] for r in events}):
    group=[r for r in events if r['voice']==v]
    speakers.append({'voice':v,**{k:float(np.median([r[k] for r in group])) for k in ('voice_loss_db','noise_gain_db','projected_snr_db')},'rtf':float(np.median([r['rtf'] for r in group if r['rtf'] is not None])) if any(r['rtf'] is not None for r in group) else None})
   if not speakers:continue
   losses=np.array([x['voice_loss_db'] for x in speakers]);noise_g=np.array([x['noise_gain_db'] for x in speakers]);snrs=np.array([x['projected_snr_db'] for x in speakers])
   summary.append({'noise_type':kind,'target_snr_db':snr,'backend':method,'speakers':len(speakers),'events':len(events),'macro_speaker_median_voice_loss_db':float(np.median(losses)),'speaker_p90_voice_loss_db':float(np.percentile(losses,90)),'worst_speaker_voice_loss_db':float(np.max(losses)),'event_fraction_voice_loss_gt6db':float(np.mean([r['voice_loss_db']>6 for r in events])),'event_fraction_voice_loss_gt10db':float(np.mean([r['voice_loss_db']>10 for r in events])),'weak_p10_delta_db':float(np.median([r['weak_p10_delta_db'] for r in events if r['weak_p10_delta_db'] is not None])),'weak_p1_delta_db':float(np.median([r['weak_p1_delta_db'] for r in events if r['weak_p1_delta_db'] is not None])),'weak_fraction_below_minus20db':float(np.mean([r['weak_fraction_below_minus20db'] for r in events if r['weak_fraction_below_minus20db'] is not None])),'weak_fraction_below_minus40db':float(np.mean([r['weak_fraction_below_minus40db'] for r in events if r['weak_fraction_below_minus40db'] is not None])),'weak_fraction_below_minus80db':float(np.mean([r['weak_fraction_below_minus80db'] for r in events if r['weak_fraction_below_minus80db'] is not None])),'weak_exact_floor_frames':int(sum(r['weak_exact_floor_frames'] for r in events)),'weak_envelope_corr':float(np.nanmedian([r['weak_envelope_corr'] for r in events if r['weak_envelope_corr'] is not None])) if any(r['weak_envelope_corr'] is not None for r in events) else None,'macro_speaker_median_noise_gain_db':float(np.median(noise_g)),'macro_speaker_median_projected_snr_db':float(np.median(snrs)),'macro_speaker_median_rtf':float(np.median([x['rtf'] for x in speakers if x['rtf'] is not None])) if any(x['rtf'] is not None for x in speakers) else None,'speaker_values':speakers})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Primary 10 dB cases: voice loss vs projected noise suppression by method, macro speaker values.
 focus=[('classroom-babble',10),('crowd-babble-cc0',10),('fan-motor-cc0',10),('synthetic-pink',10)]
 fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
 colors={'input':'#9ca3af','dpdfnet2_48khz':'#2563eb','dfn18':'#2a9d8f','dfn100':'#e76f51','resemble-denoiser-only':'#111827'}
 labels={'input':'Entrée','dpdfnet2_48khz':'DPDFNet2 48 kHz','dfn18':'DFN cap18','dfn100':'DFN cap100*','resemble-denoiser-only':'Resemble denoiser-only'}
 for ax,(kind,snr) in zip(axes.flat,focus):
  for m in METHODS:
   r=next(x for x in summary if x['noise_type']==kind and x['target_snr_db']==snr and x['backend']==m)
   ax.scatter(-r['macro_speaker_median_noise_gain_db'],r['macro_speaker_median_voice_loss_db'],label=labels[m],color=colors[m],s=55)
  ax.axhline(6,color='#555',ls='--',lw=1);ax.set_title(kind);ax.set_xlabel('Baisse du bruit projeté (dB)');ax.set_ylabel('Perte de voix projetée (dB)');ax.grid(alpha=.25)
 axes[0,0].legend(fontsize=8);fig.suptitle('Holdout 7 locuteurs · événements faibles −18 dB · 10 dB SNR · macro par locuteur')
 fig.savefig(OUT/'weak-preservation-pareto.png',dpi=170)
 report={'title':'Realtime weak-speech backend screen — same frozen noisy inputs','protocol':'Reuses exactly the seven-speaker/three-event frozen inputs from weak-speech-backend-heldout-02. DPDFNet2 causal native 48 kHz was rendered locally; DFN18/100 and Resemble denoiser-only are reused from the same frozen input set. All projections use exact 250 ms attenuated speech/noise stems. Macro summaries median the three events within speaker first, then weight speakers equally.','primary_findings':'see summary.json and markdown report','methods':['input','DPDFNet2 48 kHz causal CPU','DeepFilterNet cap 18 (existing; not cap 6)','DeepFilterNet cap 100 aggressive historical control','Resemble denoiser-only existing offline reference'],'limitations':['Seven French LibriVox speakers from one book/recording corpus; event holdout is not an independent recording-device domain.','No Adobe v2 output on these exact 126 renders; Adobe cross-check is a separate three-voice cohort.','The benchmark tests 250 ms events with controlled mixtures, not listener preference or intelligibility.','DeepFilterNet cap6 is unavailable in this local runtime and was not substituted with a different method.','cap100 is an aggressive control, not a live selection candidate.'],'dfn_runtime_source':'heldout backend report 2026-10-08; RTF may include the old benchmark invocation settings.','sample_rate_hz':RATE,'event_duration_s':0.25,'weak_event_attenuation_db':-18,'results_csv':'event-metrics.csv','summary_json':'summary.json','figure':'weak-preservation-pareto.png'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 # Concise markdown with all 10 dB conditions and GPU/CPU RTF.
 lines=['# Weak-first real-time candidate screen — 2026-10-09','','## Frozen comparison','',report['protocol'],'','The same 126 exact inputs were used by every listed method. This screen adds 126 DPDFNet2 causal 48 kHz CPU renders; it does not re-render DFN or Resemble. Adobe v2 is not present on these inputs.','', '| Noise / SNR | Backend | Voice loss median / speaker p90 / worst (dB) | Events >6 dB | Noise reduction median (dB) | Projected SNR median (dB) | RTF median |','|---|---|---:|---:|---:|---:|---:|']
 for kind,snr in focus:
  for method in METHODS:
   r=next(x for x in summary if x['noise_type']==kind and x['target_snr_db']==snr and x['backend']==method)
   red=-r['macro_speaker_median_noise_gain_db'];rtf='n.d.' if r['macro_speaker_median_rtf'] is None else f"{r['macro_speaker_median_rtf']:.3f}"
   lines.append(f"| {kind} / {snr} dB | {labels[method]} | {r['macro_speaker_median_voice_loss_db']:.2f} / {r['speaker_p90_voice_loss_db']:.2f} / {r['worst_speaker_voice_loss_db']:.2f} | {100*r['event_fraction_voice_loss_gt6db']:.1f}% | {red:.2f} | {r['macro_speaker_median_projected_snr_db']:.2f} | {rtf} |")
 lines += ['', '*DFN cap100 is retained only as an aggressive historical comparator. Positive noise reduction means lower projected noise component than the input stem.*','','## Figure','','![Voice preservation against projected noise suppression](weak-preservation-pareto.png)','','## Candidate gates from GPT Web','','Use weak-frame survival first: clean-defined event mask; report p10/p1, fractions below −20/−40/−80 dB, exact-floor count, and weak-zone envelope correlation. The three Adobe pairs are a secondary behavior check only (candidate envelope correlation ≥ Adobe −0.03, weak p10 no more than 6 dB below Adobe, and no >40 dB collapse if Adobe has none).','','The full proposed primary gates are recorded in the Adobe v2 plan: catastrophic-collapse zero tolerance; weak-event p10 ≥−24 dB at speaker level, no speaker below −30 dB, speaker p90 event loss ≤6 dB and no event above 10 dB; normal-voice median event loss <1 dB, speaker p90 <3 dB, envelope median ≥0.90; noise suppression medians ≥8 dB on chatter10 and ≥12 dB on stationary10 with ≥75% cases improving; early-target p90 loss <3 dB and measurable late-tail reduction ≥8 dB (medium RIR) / ≥10 dB (long RIR). No composite score.','','## Decisions and limits','','- This is a useful held-out weak-event screen, not proof of Adobe parity or universal quality.','- The exact local runtime has no DeepFilterNet cap6, so this screen cannot choose between cap6 and cap18. No product default changes are justified by this screen alone.','- DPDFNet2 runtime is CPU-only, causal, native 48 kHz. Its median RTF is recorded per case and is a throughput metric, not an end-to-end microphone latency guarantee.','- Adobe v2 comparisons remain limited to the separate 3-voice matched cohort; all Adobe comparisons use original mix outputs, not these heldout renders.','','## Reproduction','','```bash','.venv/bin/python scripts/benchmarks/universal_enhancer/analyze_realtime_weak_preservation.py','.venv/bin/python scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py','```','','Audio renders are ignored local files under `results/noise-rir-truth-01/realtime-weak-preservation-01/`.']
 (ROOT/'docs/benchmarking/realtime-weak-preservation-2026-10-09.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps([{'condition':r['noise_type'],'snr':r['target_snr_db'],'backend':r['backend'],'loss':r['macro_speaker_median_voice_loss_db'],'p90':r['speaker_p90_voice_loss_db'],'worst':r['worst_speaker_voice_loss_db'],'noise_reduction':-r['macro_speaker_median_noise_gain_db'],'rtf':r['macro_speaker_median_rtf']} for r in summary if r['target_snr_db']==10],indent=2))
if __name__=='__main__':main()
