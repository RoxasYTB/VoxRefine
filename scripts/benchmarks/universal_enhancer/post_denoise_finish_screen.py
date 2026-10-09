#!/usr/bin/env python3
"""Level/tone/post-denoiser screen on existing NFE64-C voice renders."""
from pathlib import Path
import csv, hashlib, importlib.metadata, json, platform, subprocess, sys
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from voxrefine.toneshape import apply_gentle_compression, apply_shelves
RATE=48000; OUT=ROOT/'results/post-denoise-finish-01'
IDS=['stephanie_fan18','remi_crowd18','emy_mixed_ambience18']
def read(p):
 x,sr=sf.read(p,dtype='float64');
 if sr!=RATE: raise ValueError(f'{p}: {sr}')
 return np.asarray(x).reshape(-1)
def db(x):return float(20*np.log10(max(float(x),1e-12)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def frames(x):
 n=960;return np.sqrt(np.mean(x[:len(x)//n*n].reshape(-1,n)**2,axis=1)+1e-20)
def active_mask(clean):
 r=frames(clean);m=r>=np.max(r)*10**(-35/20)
 return np.convolve(np.pad(m.astype(int),3),np.ones(7,dtype=int),mode='same')[3:-3]>0
def spec_mae(x,clean,mask):
 nfft=2048;hop=480;freq=np.fft.rfftfreq(nfft,1/RATE);w=np.hanning(nfft);a=[];b=[]
 for j in np.flatnonzero(mask):
  i=j*hop
  if i+nfft>min(len(x),len(clean)):continue
  a.append(np.abs(np.fft.rfft(x[i:i+nfft]*w))**2);b.append(np.abs(np.fft.rfft(clean[i:i+nfft]*w))**2)
 if not a:return None
 A=np.mean(a,axis=0);B=np.mean(b,axis=0);vals=[]
 for c in np.geomspace(80,12000,100):
  ix=(freq>=c*2**(-1/12))&(freq<c*2**(1/12))
  if ix.any():vals.append(abs(10*np.log10(A[ix].mean()+1e-20)-10*np.log10(B[ix].mean()+1e-20)))
 return float(np.mean(vals))
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for sid in IDS:
  source=ROOT/f'results/clear-noisy-mix-01/audio/{sid}-resemble-nfe64-C.wav';cleanp=ROOT/f'corpus/samples/clear-noisy-mix-01/clean/{sid}-clean-reference.wav'
  base=read(source);clean=read(cleanp);n=min(len(base),len(clean));base=base[:n];clean=clean[:n];mask=active_mask(clean)
  tone=apply_shelves(base,RATE,bass_db=-3,bass_corner_hz=100,treble_db=-2.5,treble_corner_hz=3500)
  default,_=apply_gentle_compression(base,RATE)
  soft,_=apply_gentle_compression(tone,RATE)
  variants={'A_current_C_no_finish':base,
            'B_C_gentle_compressor_minus2dB':default*10**(-2/20),
            'C_softedges_gentle_compressor_minus2dB':soft*10**(-2/20)}
  for limit in (6,18):
   p=OUT/f'deepfilter-finished/{sid}/{limit}/{sid}-C-gentle-pre-postfilter.wav'
   if p.exists():
    variants[f'D_C_gentle_DFN{limit}_minus2dB']=read(p)[:n]*10**(-2/20)
  for label,x in variants.items():
   r=frames(x);speech=r[mask[:len(r)]];peak=float(np.max(np.abs(resample_poly(x,4,1))))
   # Match active RMS before comparing spectral shape, separating timbre from gain.
   ref_active=np.median(frames(clean)[mask[:len(frames(clean))]])
   out_active=np.median(speech)
   matched=x*(ref_active/max(out_active,1e-12))
   row={'sample':sid,'variant':label,'duration_s':len(x)/RATE,'LUFS_I_dB':float(pyln.Meter(RATE).integrated_loudness(x)),'active_speech_RMS_p50_dBFS':db(out_active),'active_speech_RMS_p90_dBFS':db(np.percentile(speech,90)),'active_speech_RMS_p99_dBFS':db(np.percentile(speech,99)),'active_gain_needed_to_match_clean_dB':db(ref_active/out_active),'true_peak_4x_dBTP':db(peak),'spectral_MAE_vs_clean_level_matched_dB':spec_mae(matched,clean,mask)}
   rows.append(row)
   wav=OUT/'audio'/sid/f'{label}.wav';wav.parent.mkdir(parents=True,exist_ok=True);sf.write(wav,x,RATE,subtype='PCM_24')
   if sid=='emy_mixed_ambience18':
    mp3=wav.with_suffix('.mp3');subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(wav),'-codec:a','libmp3lame','-b:a','192k',str(mp3)],check=True)
 with (OUT/'metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 source_hashes={sid:{'enhanced_sha256':sha(ROOT/f'results/clear-noisy-mix-01/audio/{sid}-resemble-nfe64-C.wav'),'clean_sha256':sha(ROOT/f'corpus/samples/clear-noisy-mix-01/clean/{sid}-clean-reference.wav')} for sid in IDS}
 versions={name:importlib.metadata.version(dist) for name,dist in [('numpy','numpy'),('scipy','scipy'),('soundfile','soundfile'),('pyloudnorm','pyloudnorm')]}
 (OUT/'report.json').write_text(json.dumps({'study':'Level, tone and post-denoiser screening of existing renders','date':'2026-10-09','sample_rate_hz':RATE,'python':platform.python_version(),'software_versions':versions,'source_sha256':source_hashes,'pipeline_under_test':{'tone':'existing Resemble C; optional soft-edges −3 dB below 100 Hz and −2.5 dB above 3.5 kHz','compressor':'20 ms centered RMS detector; −16 dBFS threshold; 1.5:1; 6 dB soft knee; 10 ms attack; 120 ms release','final_output_gain_db':-2.0,'post_denoiser_sweep_db':[6,18]},'measurement':'Speech activity from exact clean reference (20 ms RMS, max-35 dB, ±60 ms dilation); spectrum MAE compared after matching active speech median RMS. True peak estimated at 4x oversampling.','candidates':rows,'limitations':['No Resemble re-inference: checkpoint is unavailable locally.','Flat gain changes loudness and peaks but cannot remove denoiser artifacts.','The spectral MAE is descriptive and is not a listening-quality metric.','Only three voice/noise pairs.']},indent=2)+'\n')
 print(f'wrote {OUT}')
if __name__=='__main__':main()
