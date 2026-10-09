#!/usr/bin/env python3
"""Same-source static treble balance A/B against Adobe v2."""
import sys, subprocess, json
from pathlib import Path
import numpy as np, soundfile as sf, pyloudnorm as pyln
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from voxrefine.toneshape import apply_shelves
from scripts.benchmarks.universal_enhancer.tone_balance_sweep import RATE,RUN,read,spectral_curve,macroband,frame_rms
OUT=ROOT/'results/treble-balance-ab-2026-10-09'; AD=RUN/'audio'
base=read(AD/'noise-snr10-adobe-resemble-nfe16-C.wav');dry=read(AD/'noise-snr10-adobe-dry.wav');adobe=read(AD/'adobe-v2-reference.wav');n=min(map(len,[base,dry,adobe]));base,dry,adobe=[x[:n] for x in(base,dry,adobe)]
meter=pyln.Meter(RATE);base_lufs=float(meter.integrated_loudness(base));dry_lufs=float(meter.integrated_loudness(dry));adobe*=10**((dry_lufs-float(meter.integrated_loudness(adobe)))/20)
frames=frame_rms(dry);speech=frames>=np.max(frames)*10**(-35/20);speech=np.convolve(np.pad(speech.astype(np.int8),6),np.ones(13,dtype=np.int32),mode='same')[6:-6]>0
variants=[('A-current',base),('B-trim-1p5',apply_shelves(base,RATE,treble_db=-1.5,treble_corner_hz=4000)),('C-trim-2p5',apply_shelves(base,RATE,treble_db=-2.5,treble_corner_hz=4000)),('D-trim-3p5',apply_shelves(base,RATE,treble_db=-3.5,treble_corner_hz=4000))]
variants.append(('E-adobe-v2',adobe));OUT.mkdir(parents=True,exist_ok=True);rows=[]
for label,x in variants:
 fixed=dry_lufs-float(meter.integrated_loudness(x)) if not label.startswith('E-') else 0
 measured=x*10**(fixed/20)
 if label.startswith('E-'):listen=measured
 else:listen=x*10**((base_lufs-float(meter.integrated_loudness(x)))/20)
 path=OUT/f'{label}.wav';sf.write(path,listen,RATE,subtype='PCM_24');subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(path),'-codec:a','libmp3lame','-b:a','192k',str(path.with_suffix('.mp3'))],check=True)
 curve=spectral_curve(measured,speech)[1];ref=spectral_curve(adobe,speech)[1];bands=macroband(measured,speech)
 row={'variant':label,'gain_to_dry_lufs_db':fixed,'listen_lufs':float(meter.integrated_loudness(listen)),'spectrum_mae_vs_adobe_db_20_12k':float(np.mean(np.abs(curve-ref))),'delta_4_8k_db':bands['4000-8000Hz_db']-macroband(adobe,speech)['4000-8000Hz_db'],'delta_8_12k_db':bands['8000-12000Hz_db']-macroband(adobe,speech)['8000-12000Hz_db'],'sample_peak_dbfs':float(20*np.log10(np.max(np.abs(listen))+1e-12))}
 rows.append(row);print(row)
(OUT/'metrics.json').write_text(json.dumps({'sample_rate_hz':RATE,'duration_s':n/RATE,'speech_mask':'known dry stem, max-35dB with ±60ms guard','spectral_level_match':'each candidate fixed to dry integrated LUFS; audition files level-matched to current NFE16-C LUFS','variants':rows},indent=2)+'\n')
# Plot comparable fixed-gain curves and their residuals.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
ref=spectral_curve(adobe,speech)[1];freq=spectral_curve(adobe,speech)[0]
colors=['#64748b','#f4a261','#2a9d8f','#e76f51','#111827']
for (row,(label,x)),color in zip(zip(rows,variants),colors):
 fixed=row['gain_to_dry_lufs_db'];curve=spectral_curve(x*10**(fixed/20),speech)[1]
 axs[0].plot(freq,curve,label=label,color=color,lw=1.5);axs[1].plot(freq,curve-ref,label=label,color=color,lw=1.4)
axs[1].axhline(0,color='black',lw=.8)
for ax in axs:
 ax.set_xscale('log');ax.set_xlim(100,12000);ax.grid(True,which='both',alpha=.22);ax.legend(fontsize=8,ncol=3);ax.set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Speech-active power (dB)');axs[0].set_title('Fixed-gain 1/6-octave curves: Resemble tone trims vs Adobe v2')
axs[1].set_ylabel('Difference vs Adobe (dB)');axs[1].set_title('Spectral residual; descriptor, not a quality score')
fig.savefig(OUT/'treble-balance-spectra.png',dpi=170)
