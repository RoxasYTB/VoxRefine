import csv,json,sys,time
from pathlib import Path
import numpy as np,soundfile as sf,torch
ROOT=Path.cwd(); SRC=ROOT/'.tools/sgmse-src'; BENCH=ROOT/'results/noise-rir-truth-01/vctk-rir-dereverb-truth-01'; OUT=ROOT/'results/sgmse-ears-reverb-screen-01'
orig_cuda_available=torch.cuda.is_available
torch.cuda.is_available=lambda:False
sys.path.insert(0,str(SRC))
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec
import importlib
opmod=importlib.import_module('sgmse.backbones.ncsnpp_utils.op.upfirdn2d')
import sgmse.backbones.ncsnpp_utils.up_or_down_sampling as upsampler
def native_upfirdn(input,kernel,up=1,down=1,pad=(0,0)):
 return opmod.upfirdn2d_native(input,kernel,up,up,down,down,pad[0],pad[1],pad[0],pad[1])
opmod.upfirdn2d=native_upfirdn; upsampler.upfirdn2d=native_upfirdn
torch.cuda.is_available=orig_cuda_available

def db(x):return float(20*np.log10(max(float(x),1e-12)))
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)+1e-24))
CASES=(('p276','002','t60-0.80s'),('p292','001','t60-0.45s'),('p341','001','t60-0.80s'))
print('load model',flush=True); t=time.perf_counter(); model=ScoreModel.load_from_checkpoint(str(SRC/'ears-reverb-48k.ckpt'),map_location='cpu',weights_only=False);model.eval().to('cuda');load=time.perf_counter()-t;print('loaded',round(load,2),flush=True)
torch.set_num_threads(4)
OUT.mkdir(parents=True,exist_ok=True); metrics=[]
for spk,utt,rir in CASES:
 for cond in ('dry','rir-only'):
  stem=f'{spk}-{utt}-{rir}-{cond}'; p=BENCH/'inputs'/f'{stem}.wav'; full,sr=sf.read(p,dtype='float32')
  if sr!=48000 or full.ndim!=1 or len(full)!=144000:raise ValueError((p,sr,full.shape))
  full_norm=float(np.max(np.abs(full))); crop=full[48000:144000]
  torch.manual_seed(0);torch.cuda.manual_seed_all(0)
  y=torch.from_numpy(crop.copy()).unsqueeze(0).to('cuda')/full_norm
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode():
   spec=model._forward_transform(model._stft(y)); Y=pad_spec(spec.unsqueeze(0),mode='reflection')
   sampler=model.get_pc_sampler('reverse_diffusion','ald',Y,N=30,corrector_steps=1,snr=.5)
   sample,nfe=sampler(); out=model.to_audio(sample.squeeze(),y.shape[-1])*full_norm
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;arr=out.detach().cpu().numpy().astype(np.float32).squeeze()
  if arr.shape!=(len(crop),):raise RuntimeError((stem,arr.shape,len(crop)))
  op=OUT/'renders'/f'{stem}-window-1to3-sgmse.wav';op.parent.mkdir(exist_ok=True)
  sf.write(op,arr,48000,subtype='FLOAT')
  lo,hi=round(.25*sr),round(.50*sr)
  row={'speaker':spk,'utterance':utt,'rir':rir,'condition':cond,'seed':0,'sampler':'official-PC-reverse_diffusion-ald','N':30,'corrector_steps':1,'snr':.5,'crop_start_s':1.0,'crop_duration_s':2.0,
       'event_energy_delta_vs_input_db':db(rms(arr[lo:hi])/rms(crop[lo:hi])),
       'tail_50_150_output_dbfs':db(rms(arr[round(.55*sr):round(.65*sr)])),
       'tail_150_300_output_dbfs':db(rms(arr[round(.65*sr):round(.80*sr)])),
       'tail_300_600_output_dbfs':db(rms(arr[round(.80*sr):round(1.10*sr)])),
       'tail_150_300_reduction_vs_input_db':db(rms(crop[round(.65*sr):round(.80*sr)])/rms(arr[round(.65*sr):round(.80*sr)])),
       'tail_300_600_reduction_vs_input_db':db(rms(crop[round(.80*sr):round(1.10*sr)])/rms(arr[round(.80*sr):round(1.10*sr)])),
       'elapsed_s':elapsed,'rtf':elapsed/2.0,'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_vram_mib':torch.cuda.max_memory_reserved()/1024**2,
       'peak_dbfs':db(np.max(np.abs(arr))),'clipped_samples':int(np.sum(np.abs(arr)>=1)),'output_path':str(op.relative_to(ROOT))}
  metrics.append(row)
  with (OUT/'frozen-crop-metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
  print('DONE',json.dumps(row),flush=True)
manifest={'description':'Fixed 2 s crop from 1.0-3.0 s of matched 3 s VCTK truth files; inference only, no tuning. Chunking is solely a VRAM workaround and is not a validated whole-file algorithm.','repo_commit':__import__('subprocess').check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip(),'checkpoint_sha256':__import__('hashlib').sha256((SRC/'ears-reverb-48k.ckpt').read_bytes()).hexdigest(),'checkpoint_id':'1PunXuLbuyGkknQCn_y-RCV2dTZBhyE3V','device':torch.cuda.get_device_name(0),'model_load_s':load,'native_operator':'official pure-PyTorch upfirdn2d fallback, used on CUDA to avoid compiling custom extension','rows':metrics}
(OUT/'frozen-crop-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
