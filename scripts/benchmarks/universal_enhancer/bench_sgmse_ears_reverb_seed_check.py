import csv, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
ROOT=Path.cwd(); SRC=ROOT/'.tools/sgmse-src'; BENCH=ROOT/'results/noise-rir-truth-01/vctk-rir-dereverb-truth-01'; OUT=ROOT/'results/sgmse-ears-reverb-screen-01/context-seed-check'
orig_cuda_available=torch.cuda.is_available; torch.cuda.is_available=lambda:False
sys.path.insert(0,str(SRC)); from sgmse.model import ScoreModel; from sgmse.util.other import pad_spec
import importlib; opmod=importlib.import_module('sgmse.backbones.ncsnpp_utils.op.upfirdn2d'); import sgmse.backbones.ncsnpp_utils.up_or_down_sampling as upsampler
def native_upfirdn(input,kernel,up=1,down=1,pad=(0,0)): return opmod.upfirdn2d_native(input,kernel,up,up,down,down,pad[0],pad[1],pad[0],pad[1])
opmod.upfirdn2d=native_upfirdn; upsampler.upfirdn2d=native_upfirdn; torch.cuda.is_available=orig_cuda_available
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)+1e-24))
def db(x): return float(20*np.log10(max(float(x),1e-12)))
CASES=(('p292','001','t60-0.45s'),('p276','002','t60-0.80s'))
OUT.mkdir(parents=True,exist_ok=True); print('loading official model',flush=True); t=time.perf_counter(); model=ScoreModel.load_from_checkpoint(str(SRC/'ears-reverb-48k.ckpt'),map_location='cpu',weights_only=False); model.eval().to('cuda'); load_s=time.perf_counter()-t; torch.set_num_threads(4); print('loaded',round(load_s,2),'s',flush=True)
rows=[]
for spk,utt,rir in CASES:
 for cond in ('dry','rir-only'):
  stem=f'{spk}-{utt}-{rir}-{cond}'; path=BENCH/'inputs'/f'{stem}.wav'; full,sr=sf.read(path,dtype='float32'); assert sr==48000 and full.shape==(144000,)
  peak=float(np.max(np.abs(full)))
  # The 3 s full-context pass exceeded the available VRAM on this 4 GiB card.
  for context in (2,):
   start=48000 if context==2 else 0; raw=full[start:start+context*sr]
   for seed in (1,2):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); y=torch.from_numpy(raw.copy()).unsqueeze(0).to('cuda')/peak
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    with torch.inference_mode():
     spec=model._forward_transform(model._stft(y)); Y=pad_spec(spec.unsqueeze(0),mode='reflection'); sampler=model.get_pc_sampler('reverse_diffusion','ald',Y,N=30,corrector_steps=1,snr=.5); sample,nfe=sampler(); out=model.to_audio(sample.squeeze(),y.shape[-1])*peak
    torch.cuda.synchronize(); elapsed=time.perf_counter()-t; arr=out.detach().cpu().numpy().astype(np.float32).squeeze(); assert arr.shape==raw.shape
    # Score same absolute-time event (1.25-1.50 s) and tail windows for 2 s and 3 s contexts.
    def win(a,b): return slice(round((a-start/sr)*sr),round((b-start/sr)*sr))
    event=win(1.25,1.50); tails=[win(1.65,1.80),win(1.80,2.10)]
    inp_event=rms(full[round(1.25*sr):round(1.50*sr)]); out_event=rms(arr[event]); event_delta=db(out_event/inp_event)
    tail_red=[]; tail_direct=[]
    for w in tails:
     ins=rms(full[round(1.65*sr):round(1.80*sr)] if w==tails[0] else full[round(1.80*sr):round(2.10*sr)])
     outs=rms(arr[w]); reduction=db(ins/outs); tail_red.append(reduction); tail_direct.append(reduction+event_delta)
    row={'speaker':spk,'utterance':utt,'rir':rir,'condition':cond,'context_s':context,'seed':seed,'sampler':'official-PC-reverse_diffusion-ald','N':30,'corrector_steps':1,'snr':.5,'score_event_s':'1.25-1.50','tail_windows_s':'1.65-1.80;1.80-2.10','event_delta_db':event_delta,'tail_150_300_reduction_db':tail_red[0],'tail_300_600_reduction_db':tail_red[1],'tail_event_ratio_improvement_150_300_db':tail_direct[0],'tail_event_ratio_improvement_300_600_db':tail_direct[1],'elapsed_s':elapsed,'rtf':elapsed/context,'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_vram_mib':torch.cuda.max_memory_reserved()/1024**2,'peak_dbfs':db(np.max(np.abs(arr))),'clipped_samples':int(np.sum(np.abs(arr)>=1))}
    outpath=OUT/f'{stem}-ctx{context}-seed{seed}.wav'; sf.write(outpath,arr,48000,subtype='FLOAT'); row['output_path']=str(outpath.relative_to(ROOT)); rows.append(row)
    with (OUT/'metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('DONE',json.dumps(row),flush=True)
(OUT/'manifest.json').write_text(json.dumps({'description':'Same central absolute-time event/tail scored for 2s crop versus full 3s utterance, dry and matched RIR, seeds 0/1/2, no parameter tuning. Full utterance may still be a short synthetic screen, not product generalization.','checkpoint_sha256':__import__('hashlib').sha256((SRC/'ears-reverb-48k.ckpt').read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(0),'model_load_s':load_s,'rows':rows},indent=2)+'\n')
