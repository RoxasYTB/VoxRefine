#!/usr/bin/env python3
"""Evaluate actual CLI NFE16 renders against the exact paired Adobe reference."""
from __future__ import annotations
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np,soundfile as sf,pyloudnorm as pyln
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.benchmarks.universal_enhancer.tone_balance_sweep import RATE,read,frame_rms,spectral_curve,macroband
OUT=ROOT/'results/adobe-curve-eq-2026-10-09';PAIR=ROOT/'results/resemble-enhance-01/controlled-challenger-03/audio'
files={'dry':PAIR/'noise-snr10-adobe-dry.wav','adobe':PAIR/'adobe-v2-reference.wav','flat':OUT/'controlled-nfe16-flat-cpu.wav','C':OUT/'controlled-nfe16-C-cpu.wav','adobe_curve':OUT/'controlled-nfe16-adobe-curve-cpu.wav'}
if any(not p.is_file() for p in files.values()):raise SystemExit('Run flat, C, and adobe-curve CLI renders first.')
x={k:read(v) for k,v in files.items()};n=min(map(len,x.values()));x={k:v[:n] for k,v in x.items()};meter=pyln.Meter(RATE);dry_lufs=float(meter.integrated_loudness(x['dry']));base_lufs=float(meter.integrated_loudness(x['C']))
r=frame_rms(x['dry']);mask=r>=r.max()*10**(-35/20);mask=np.convolve(np.pad(mask.astype(np.int8),6),np.ones(13,dtype=np.int32),mode='same')[6:-6]>0
# Curves use one fixed gain per file to the dry reference LUFS, matching the benchmark protocol.
fixed={k:v*10**((dry_lufs-float(meter.integrated_loudness(v)))/20) for k,v in x.items() if k!='dry'}
f,ref=spectral_curve(fixed['adobe'],mask);keep=(f>=80)&(f<=12000);refbands=macroband(fixed['adobe'],mask)
rows=[];curves={}
for key in ('flat','C','adobe_curve'):
 sig=fixed[key];freq,c=spectral_curve(sig,mask);b=macroband(sig,mask);curves[key]=c
 listen=sig*10**((base_lufs-float(meter.integrated_loudness(sig)))/20)
 wav=OUT/f'actual-{key}-levelmatched.wav';sf.write(wav,listen,RATE,subtype='PCM_24');mp3=wav.with_suffix('.mp3')
 subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(wav),'-codec:a','libmp3lame','-b:a','192k',str(mp3)],check=True)
 rows.append({'profile':key,'raw_lufs':float(meter.integrated_loudness(x[key])),'fixed_gain_to_dry_lufs_db':dry_lufs-float(meter.integrated_loudness(x[key])),'listening_lufs':float(meter.integrated_loudness(listen)),'spectrum_mae_20_12k_db':float(np.mean(abs(c[keep]-ref[keep]))),'delta_4_8k_db':b['4000-8000Hz_db']-refbands['4000-8000Hz_db'],'delta_8_12k_db':b['8000-12000Hz_db']-refbands['8000-12000Hz_db'],'sample_peak_dbfs':float(20*np.log10(np.max(abs(listen))+1e-12))})
# Adobe reference at same listening loudness as all three actual candidates.
adobe_listen=fixed['adobe']*10**((base_lufs-float(meter.integrated_loudness(fixed['adobe'])))/20);wav=OUT/'actual-adobe-levelmatched.wav';sf.write(wav,adobe_listen,RATE,subtype='PCM_24');subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(wav),'-codec:a','libmp3lame','-b:a','192k',str(wav.with_suffix('.mp3'))],check=True)
with (OUT/'actual-cli-metrics.csv').open('w',newline='') as fobj:
 w=csv.DictWriter(fobj,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'protocol':'Same 6 s noisy input, three real CLI runs with identical Resemble NFE16 / CPU / 3s chunks / dynamics off / de-esser off / -2.5dB output gain. Only tone profile varies. Spectra fixed-gain matched to dry-stem integrated LUFS; speech mask from dry stem max-35dB with ±60ms guard. Listening MP3s matched to profile C LUFS.','duration_s':n/RATE,'sample_rate_hz':RATE,'dry_lufs':dry_lufs,'listening_target_lufs':base_lufs,'metrics':rows,'notes':['Spectral curve distance is descriptive, not perceived quality.','The fitted curve EQ is stationary; its denoising model output may still contain time-varying artifacts.']}
(OUT/'actual-cli-report.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axs=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
colors={'flat':'#64748b','C':'#e76f51','adobe_curve':'#2a9d8f'}
for key,c in curves.items():axs[0].plot(f,c,label=key,color=colors[key]);axs[1].plot(f,c-ref,label=key,color=colors[key])
axs[0].plot(f,ref,'k--',label='Adobe v2');axs[1].axhline(0,color='black',lw=.8)
for ax in axs:ax.set_xscale('log');ax.set_xlim(100,12000);ax.grid(True,which='both',alpha=.2);ax.legend();ax.set_xlabel('Frequency (Hz)')
axs[0].set_title('Actual CLI NFE16 results vs Adobe v2');axs[0].set_ylabel('Active-speech power (dB)')
axs[1].set_title('Curve residual: candidate − Adobe');axs[1].set_ylabel('dB')
fig.savefig(OUT/'actual-cli-spectra.png',dpi=170)
print(json.dumps(report,indent=2))
