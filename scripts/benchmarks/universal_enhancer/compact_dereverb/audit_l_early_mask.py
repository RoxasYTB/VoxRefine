#!/usr/bin/env python3
"""Synthetic-only early/late oracle-mask feasibility preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SR, N, PAUSE = 16_000, 32_000, 23_200
N_FFT, HOP = 512, 128
TRAIN_RIR, EVAL_RIR = 12, 8
TRAIN_DRY, EVAL_DRY = 4, 4
STEPS, SEED = 1_000, 710_2026


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dry(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / SR
    f0 = float(rng.uniform(90, 220))
    vib = rng.uniform(1., 4.) * np.sin(2*np.pi*rng.uniform(3., 6.)*t)
    phase = 2*np.pi*np.cumsum(f0+vib)/SR
    env = np.zeros_like(t)
    starts = np.array([.15, .46, .80, 1.12]) + rng.uniform(-.035, .035, 4)
    for start, width in zip(starts, rng.uniform(.18, .28, 4)):
        u = (t-start)/width
        pulse = np.where((u >= 0)&(u < 1), np.sin(np.pi*np.clip(u,0,1))**.62, 0.)
        env = np.maximum(env, pulse)
    env *= .75 + .25*np.sin(2*np.pi*rng.uniform(3.4,6.8)*t+rng.uniform(0,6))
    formants = rng.uniform([550, 1100, 2300], [900, 1850, 3300])
    widths = rng.uniform([350, 500, 750], [650, 900, 1200])
    voice = np.zeros_like(t)
    for k in range(1, 20):
        hz = f0*k
        amp = (np.exp(-((hz-formants[0])/widths[0])**2) +
            .62*np.exp(-((hz-formants[1])/widths[1])**2) +
            .28*np.exp(-((hz-formants[2])/widths[2])**2))/k
        voice += amp*np.sin(k*phase+rng.uniform(-.2,.2))
    clean = (env*voice*.12/max(np.max(np.abs(voice)),1e-8)).astype(np.float32)
    return clean


def make_rir(seed: int):
    rng = np.random.default_rng(seed)
    length = int(.55*SR)
    early = np.zeros(length, np.float64)
    late = np.zeros(length, np.float64)
    early[0] = 1.0
    for _ in range(5):
        lag = int(rng.uniform(.008, .049)*SR)
        early[lag] += rng.choice([-1.,1.])*rng.uniform(.025,.12)
    decay = rng.uniform(.17,.32)
    for lag in range(round(.05*SR), length, 80):
        late[lag] += rng.normal()*.10*np.exp(-(lag/SR-.05)/decay)
    return early, late


def convolve(dry: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = dry.astype(np.float64).copy()
    for lag in np.flatnonzero(h[1:])+1:
        out[lag:] += h[lag]*dry[:-lag]
    return out.astype(np.float32)


def spec_np(audio: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(audio.astype(np.float32))[None]
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=torch.hann_window(N_FFT), center=True, return_complex=True)[0].numpy()


def istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=torch.hann_window(N_FFT,device=spec.device,dtype=spec.real.dtype),
        center=True, length=length)


def rir_item(index: int, eval_set: bool):
    base = SEED + (20_000 if eval_set else 0)
    dry = make_dry(base + 2000 + index)
    h_early, h_late = make_rir(base + 5000 + index)
    h_full = h_early+h_late
    # One shared scalar for full and early paths; retain identical timeline.
    active = np.abs(dry) > np.max(np.abs(dry))*.02
    q = .08/(np.sqrt(np.mean(dry[active].astype(np.float64)**2))+1e-12)
    dry = (dry*q).astype(np.float32)
    full = convolve(dry,h_full)
    early = convolve(dry,h_early)
    xspec, espec = spec_np(full), spec_np(early)
    target = np.clip(np.abs(espec)/(np.abs(xspec)+1e-7),0.,1.).astype(np.float32)
    return {"kind":"rir","id":f"{'eval' if eval_set else 'train'}-rir-{index}",
        "dry":dry,"input":full,"early":early,"xspec":xspec,"target":target,
        "early_spec":espec,"rir_cut_ms":50}


def dry_item(index: int, eval_set: bool):
    base = SEED + (40_000 if eval_set else 0)
    dry = make_dry(base+3000+index)
    active = np.abs(dry)>np.max(np.abs(dry))*.02
    q=.08/(np.sqrt(np.mean(dry[active].astype(np.float64)**2))+1e-12)
    dry=(dry*q).astype(np.float32)
    xspec=spec_np(dry)
    return {"kind":"dry","id":f"{'eval' if eval_set else 'train'}-dry-{index}",
        "dry":dry,"input":dry,"early":dry,"xspec":xspec,
        "target":np.ones_like(np.abs(xspec),dtype=np.float32),
        "early_spec":xspec,"rir_cut_ms":None}


class LEarlyMask(nn.Module):
    """Small dilated time-frequency mask estimator, no phase/residual head."""
    INIT_SEED=2026101012
    def __init__(self):
        super().__init__()
        widths=(16,16,16,16)
        layers=[]
        in_ch=2
        for dilation,width in zip((1,2,4,8),widths):
            layers.extend((nn.Conv2d(in_ch,width,3,padding=(1,dilation),
                                     dilation=(1,dilation)),nn.PReLU(width)))
            in_ch=width
        self.features=nn.Sequential(*layers)
        self.mask_head=nn.Conv2d(in_ch,1,1)
        with torch.no_grad():
            g=torch.Generator(device="cpu").manual_seed(self.INIT_SEED)
            w=torch.empty_like(self.mask_head.weight,device="cpu")
            nn.init.normal_(w,mean=0.,std=1e-4,generator=g)
            self.mask_head.weight.copy_(w.to(self.mask_head.weight))
            self.mask_head.bias.fill_(-3.0)

    def mask_from_spec(self, spec: torch.Tensor):
        features=torch.stack((spec.real,spec.imag),dim=1)
        logits=self.mask_head(self.features(features))[:,0]
        return 1.0-.75*torch.sigmoid(logits)

    def forward_spec(self,spec:torch.Tensor):
        return spec*self.mask_from_spec(spec)

    def forward(self,audio:torch.Tensor):
        spec=torch.stft(audio,n_fft=N_FFT,hop_length=HOP,win_length=N_FFT,
            window=torch.hann_window(N_FFT,device=audio.device,dtype=audio.dtype),
            center=True,return_complex=True)
        mask=self.mask_from_spec(spec)
        return istft(spec*mask,audio.shape[-1]),mask


def oracle_apply(item):
    return istft(torch.from_numpy(item["xspec"]*item["target"]),N).numpy()


def band_reduction(y:np.ndarray,x:np.ndarray,lo_ms:int,hi_ms:int):
    ys,xs=spec_np(y),spec_np(x)
    freq=np.fft.rfftfreq(N_FFT,d=1/SR)
    bins=(freq>=150)&(freq<=300)
    centers=np.arange(xs.shape[-1])*HOP
    frames=(centers>=PAUSE+lo_ms*16)&(centers<PAUSE+hi_ms*16)
    ein=np.abs(xs[np.ix_(bins,frames)])**2
    eout=np.abs(ys[np.ix_(bins,frames)])**2
    return float(10*np.log10((ein.mean()+1e-20)/(eout.mean()+1e-20)))


def activity(dry):
    x=torch.from_numpy(dry.astype(np.float32))
    rms=x.unfold(0,320,160).square().mean(-1).sqrt()
    peak=rms.max().clamp_min(1e-8)
    active=rms>torch.maximum(peak*.02,rms.new_tensor(1e-5))
    weak=active&(rms<peak*.35)
    onset=active&(rms>F.pad(rms[:-1],(1,0))*1.5)
    return active.numpy(),weak.numpy(),onset.numpy()


def preservation(y,x,dry):
    yy=torch.from_numpy(y.astype(np.float32)).unfold(0,320,160)
    xx=torch.from_numpy(x.astype(np.float32)).unfold(0,320,160)
    db=(20*torch.log10((yy.square().mean(-1).sqrt()+1e-8)/
                       (xx.square().mean(-1).sqrt()+1e-8))).numpy()
    active,weak,onset=activity(dry)
    n=min(db.size,active.size)
    db,active,weak,onset=db[:n],active[:n],weak[:n],onset[:n]
    def q(mask,p):
        if not mask.any(): return None
        return float(np.quantile(db[mask],p))
    return {"active_p50_db":q(active,.5),"active_p10_db":q(active,.1),
        "weak_p10_db":q(weak,.1),"onset_p10_db":q(onset,.1),
        "weak_below_minus3_fraction":float(np.mean(db[weak]<-3)),
        "weak_below_minus6_fraction":float(np.mean(db[weak]<-6))}


def weighted_mask_loss(mask,target,spec):
    energy=spec.abs().square()
    return (F.smooth_l1_loss(mask,target,reduction="none")*energy).sum()/(energy.sum()+1e-8)


def setup_determinism(device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type=="cuda": torch.cuda.manual_seed_all(SEED)


def evaluate_oracle(eval_rir,device):
    metrics=[]
    with torch.inference_mode():
        for item in eval_rir:
            y=oracle_apply(item)
            metrics.append({"id":item["id"],
                "w1_mlow_reduction_db":band_reduction(y,item["input"],150,300),
                "w2_mlow_reduction_db":band_reduction(y,item["input"],300,600)})
    vals=[m["w1_mlow_reduction_db"] for m in metrics]
    return {"overall":"PASS" if float(np.median(vals))>=3.0 and sum(v>0 for v in vals)>=7 else "FAIL",
        "w1_median_db":float(np.median(vals)),"w1_positive_count":sum(v>0 for v in vals),
        "w1_gate_median_ge_3db":float(np.median(vals))>=3.,
        "w1_gate_positive_ge_7_of_8":sum(v>0 for v in vals)>=7,
        "rooms":metrics}


def train(train_rir,train_dry,eval_dry,device):
    setup_determinism(device)
    model=LEarlyMask().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    items=train_rir+train_dry
    rng=torch.Generator(device="cpu").manual_seed(SEED+1)
    order=[]
    while len(order)<STEPS:
        order.extend(torch.randperm(len(items),generator=rng).tolist())
    losses=[]
    for step,idx in enumerate(order[:STEPS]):
        item=items[idx]
        spec=torch.from_numpy(item["xspec"].astype(np.complex64))[None].to(device)
        target=torch.from_numpy(item["target"].astype(np.float32))[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        mask=model.mask_from_spec(spec)
        loss=weighted_mask_loss(mask,target,spec)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite mask loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        if not torch.isfinite(grad): raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        if step in (0,99,499,999): losses.append({"step":step+1,"loss":float(loss.detach())})
    return model,losses


def run(device):
    train_rir=[rir_item(i,False) for i in range(TRAIN_RIR)]
    eval_rir=[rir_item(i,True) for i in range(EVAL_RIR)]
    train_dry=[dry_item(i,False) for i in range(TRAIN_DRY)]
    eval_dry=[dry_item(i,True) for i in range(EVAL_DRY)]
    ids=[x["id"] for x in train_rir+train_dry]
    eval_ids=[x["id"] for x in eval_rir+eval_dry]
    if set(ids)&set(eval_ids): raise RuntimeError("train/eval source leakage")
    oracle=evaluate_oracle(eval_rir,device)
    if oracle["overall"]!="PASS":
        return {"experiment":"L-early-mask-synthetic-preflight-v1","overall":"NO_GO_ORACLE",
            "seed":SEED,"oracle":oracle,"training_performed":False,
            "train_count":{"rir":TRAIN_RIR,"dry":TRAIN_DRY},
            "eval_count":{"rir":EVAL_RIR,"dry":EVAL_DRY},
            "train_eval_ids_disjoint":True,"corpus_read":False,"dev_or_holdout_read":False,
            "test_wav_accessed":False,"source_sha256":{"audit":sha(Path(__file__))}}
    model,losses=train(train_rir,train_dry,eval_dry,device)
    candidate_rooms=[]
    with torch.inference_mode():
        for item in eval_rir:
            audio=torch.from_numpy(item["input"])[None].to(device)
            y,mask=model(audio)
            y_np=y[0].cpu().numpy()
            pred_mask=mask[0].cpu().numpy()
            target=item["target"]
            xs=torch.from_numpy(item["xspec"].astype(np.complex64))[None].to(device)
            target_t=torch.from_numpy(target)[None].to(device)
            mask_t=torch.from_numpy(pred_mask)[None].to(device)
            mask_loss=float(weighted_mask_loss(mask_t,target_t,xs))
            candidate_rooms.append({"id":item["id"],
                "w1_mlow_reduction_db":band_reduction(y_np,item["input"],150,300),
                "w2_mlow_reduction_db":band_reduction(y_np,item["input"],300,600),
                "weighted_oracle_mask_smoothl1":mask_loss,
                "mask_median_w1":float(np.median(pred_mask[:,200:220])),
                "oracle_mask_median_w1":float(np.median(target[:,200:220]))})
        dry_metrics=[]
        for item in eval_dry:
            audio=torch.from_numpy(item["input"])[None].to(device)
            y,_=model(audio)
            dry_metrics.append({"id":item["id"],**preservation(y[0].cpu().numpy(),item["input"],item["dry"])})
    w1=np.array([m["w1_mlow_reduction_db"] for m in candidate_rooms])
    w2=np.array([m["w2_mlow_reduction_db"] for m in candidate_rooms])
    def med_dry(key): return float(np.median([m[key] for m in dry_metrics]))
    def dry_gate(key,fn): return all(m[key] is not None and fn(m[key]) for m in dry_metrics)
    zero=torch.zeros(1,N,device=device)
    zeros=[]
    with torch.inference_mode():
        for _ in range(100):
            z,_=model(zero)
            zeros.append(float(z.abs().max()))
    finite=bool(np.isfinite(w1).all() and np.isfinite(w2).all() and
        all(math.isfinite(v) for m in dry_metrics for k,v in m.items() if k!="id") and
        all(np.isfinite(m["w1_mlow_reduction_db"]) and np.isfinite(m["w2_mlow_reduction_db"]) for m in candidate_rooms))
    gates={
        "oracle_preflight_pass":oracle["overall"]=="PASS",
        "w1_median_ge_2db":float(np.median(w1))>=2.,
        "w1_positive_ge_7_of_8":int(np.sum(w1>0))>=7,
        "w2_median_ge_minus_0_5db":float(np.median(w2))>=-.5,
        "no_w2_below_minus1db":bool(np.all(w2>=-1.)),
        "dry_active_p50_range_each":dry_gate("active_p50_db",lambda x:-.25<=x<=.25),
        "dry_active_p10_gt_minus0_75_each":dry_gate("active_p10_db",lambda x:x>-.75),
        "dry_weak_p10_gt_minus1_each":dry_gate("weak_p10_db",lambda x:x>-1.),
        "dry_onset_p10_gt_minus0_75_each":dry_gate("onset_p10_db",lambda x:x>-.75),
        "dry_weak_below_minus3_le_3pct_each":dry_gate("weak_below_minus3_fraction",lambda x:x<=.03),
        "dry_weak_below_minus6_le_1pct_each":dry_gate("weak_below_minus6_fraction",lambda x:x<=.01),
        "exact_zero_100_of_100":len(zeros)==100 and all(z==0. for z in zeros),
        "finite_metrics":finite}
    pred_atten=np.array([1.-m["mask_median_w1"] for m in candidate_rooms])
    oracle_atten=np.array([1.-m["oracle_mask_median_w1"] for m in candidate_rooms])
    result={"experiment":"L-early-mask-synthetic-preflight-v1",
        "overall":"PASS" if all(gates.values()) else "FAIL","seed":SEED,
        "device":str(device),"gpu_name":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
        "torch_version":torch.__version__,"cuda_version":torch.version.cuda,
        "architecture":{"type":"small dilated time-frequency gain mask",
            "time_dilations":[1,2,4,8],"receptive_field_frames":31,
            "receptive_field_ms":31*HOP*1000/SR,"gain":"1-.75*sigmoid(a)",
            "phase_head":False,"confidence_head":False,"parameters":sum(p.numel() for p in model.parameters())},
        "training":{"steps":STEPS,"learning_rate":2e-4,"weight_decay":1e-4,
            "gradient_clip":3.,"loss":"energy-weighted SmoothL1 on mask target; dry target=1",
            "loss_checkpoints":losses},
        "counts":{"train_rir":TRAIN_RIR,"eval_rir":EVAL_RIR,
            "train_dry":TRAIN_DRY,"eval_dry":EVAL_DRY,"train_eval_ids_disjoint":True},
        "oracle":oracle,"candidate":{"w1_median_db":float(np.median(w1)),
            "w1_positive_count":int(np.sum(w1>0)),"w2_median_db":float(np.median(w2)),
            "w2_below_minus1_count":int(np.sum(w2< -1.)),"rooms":candidate_rooms,
            "dry_metrics":dry_metrics,
            "dry_speech_medians":{k:med_dry(k) for k in ("active_p50_db","active_p10_db","weak_p10_db","onset_p10_db","weak_below_minus3_fraction","weak_below_minus6_fraction")}},
        "diagnostics":{"predicted_vs_oracle_w1_attenuation_median_abs_error":float(np.median(np.abs(pred_atten-oracle_atten))),
            "zero_output_max_abs":max(zeros),"finite":finite},
        "gates":gates,"corpus_read":False,"dev_or_holdout_read":False,
        "test_wav_accessed":False,"source_sha256":{"audit":sha(Path(__file__))}}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
        else "cpu" if args.device=="auto" else args.device)
    result=run(device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))
    if result["overall"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
