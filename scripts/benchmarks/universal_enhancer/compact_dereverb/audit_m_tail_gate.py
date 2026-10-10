#!/usr/bin/env python3
"""Synthetic-only preflight for a conservative learned late-tail gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SR, N, PAUSE = 16_000, 32_000, 23_200
N_FFT, HOP = 512, 128
TRAIN_WET, EVAL_WET = 24, 8
TRAIN_DRY, EVAL_DRY = 12, 8
STEPS, SEED = 2_000, 810_2026
GAIN = 10.0 ** (-4.0 / 20.0)


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


def activity(dry: np.ndarray):
    # 20ms / 10ms RMS activity; protect speech with +/-20ms dilation.
    frames = np.lib.stride_tricks.sliding_window_view(dry, 320)[::160]
    rms = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
    peak = max(float(rms.max()), 1e-8)
    raw = rms > max(peak*.02, 1e-5)
    active = raw.copy()
    radius = 2
    for shift in range(1, radius+1):
        active[shift:] |= raw[:-shift]
        active[:-shift] |= raw[shift:]
    weak = raw & (rms < peak*.35)
    previous = np.r_[0., rms[:-1]]
    onset = raw & (rms > previous*1.5)
    return rms, active, weak, onset


def spec(audio: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(audio.astype(np.float32))[None]
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=torch.hann_window(N_FFT), center=True, return_complex=True)[0].numpy()


def frame_activity(mask: np.ndarray, frames: int) -> np.ndarray:
    # Map 10ms activity grid to STFT frame centers, then protect +/-20ms already dilated.
    centers = np.arange(frames)*HOP/SR
    idx = np.clip(np.rint((centers-.01)/.01).astype(int), 0, len(mask)-1)
    return mask[idx]


def wet_item(index: int, eval_set: bool):
    base = SEED + (20_000 if eval_set else 0)
    dry = make_dry(base + 2000 + index)
    h_early, h_late = make_rir(base + 5000 + index)
    active_raw = np.abs(dry) > max(np.max(np.abs(dry))*.02, 1e-5)
    q = .08/(np.sqrt(np.mean(dry[active_raw].astype(np.float64)**2))+1e-12)
    dry = (dry*q).astype(np.float32)
    _, protected, weak, onset = activity(dry)
    early, late = convolve(dry,h_early), convolve(dry,h_late)
    full = (early.astype(np.float64)+late.astype(np.float64)).astype(np.float32)
    sx, se, sl = spec(full), spec(early), spec(late)
    late_level = np.sqrt(np.mean(late.astype(np.float64)**2))
    # Framewise level and energy ratio labels from known synthetic decomposition.
    win = np.hanning(N_FFT).astype(np.float64)
    def frame_power(x):
        pad = N_FFT//2
        xp = np.pad(x.astype(np.float64),(pad,pad))
        fr = np.lib.stride_tricks.sliding_window_view(xp,N_FFT)[::HOP]
        return np.mean((fr*win)**2,axis=1)
    ep, lp = frame_power(early), frame_power(late)
    ratio = lp/(ep+lp+1e-20)
    p = min(len(ratio), sx.shape[-1])
    ratio, lp = ratio[:p], lp[:p]
    prot = frame_activity(protected,p)
    # Frame-local RMS is expressed relative to full-scale; -50 dBFS power threshold.
    tail_level = 10*np.log10(lp+1e-20) > -50.
    tail = (~prot) & tail_level & (ratio>=.8)
    # Exclude ambiguous quiet-speech / diffuse-tail overlap and uncertain tail fractions.
    ambiguous = (~prot) & tail_level & (ratio<.8)
    speech = prot
    valid = (~ambiguous) | speech | tail
    return {"id":f"{'eval' if eval_set else 'train'}-wet-{index}","kind":"wet",
        "dry":dry,"input":full,"early":early,"late":late,"xspec":sx,
        "tail":tail,"speech":speech,"valid":valid,"protected":prot,
        "weak":frame_activity(weak,p),"onset":frame_activity(onset,p),
        "late_level_dbfs":float(10*np.log10(np.mean(late.astype(np.float64)**2)+1e-20)),
        "late_ratio":ratio,"late_frame_level_dbfs":10*np.log10(lp+1e-20)}


def dry_item(index: int, eval_set: bool):
    base = SEED + (40_000 if eval_set else 0)
    dry = make_dry(base+3000+index)
    active_raw = np.abs(dry)>max(np.max(np.abs(dry))*.02,1e-5)
    q=.08/(np.sqrt(np.mean(dry[active_raw].astype(np.float64)**2))+1e-12)
    dry=(dry*q).astype(np.float32)
    _,protected,weak,onset=activity(dry)
    sx=spec(dry); n=sx.shape[-1]
    speech=frame_activity(protected,n)
    return {"id":f"{'eval' if eval_set else 'train'}-dry-{index}","kind":"dry",
        "dry":dry,"input":dry,"xspec":sx,"tail":np.zeros(n,bool),"speech":speech,
        "valid":np.ones(n,bool),"protected":speech,"weak":frame_activity(weak,n),
        "onset":frame_activity(onset,n),"late_ratio":np.zeros(n),
        "late_frame_level_dbfs":np.full(n,-200.)}


def feature_np(audio: np.ndarray) -> np.ndarray:
    s=spec(audio); mag=np.abs(s).astype(np.float64)
    edges=np.linspace(0,mag.shape[0],33).round().astype(int)
    band=np.stack([np.mean(mag[edges[i]:max(edges[i+1],edges[i]+1)]**2,axis=0)
                   for i in range(32)])
    db=np.clip(10*np.log10(band+1e-12),-100.,20.)
    norm=(db+100.)/120.
    deriv=np.clip(np.diff(db,axis=1,prepend=db[:,:1]),-40.,40.)/40.
    flux=np.maximum(mag[:,1:]-mag[:,:-1],0.).sum(axis=0,keepdims=True)
    flux=np.pad(flux,((0,0),(1,0)))
    flux=np.log1p(flux)/(np.log1p(flux).max()+1e-8)
    return np.concatenate((norm,deriv,flux),axis=0).T.astype(np.float32)


class TailTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(65,64,1)
        self.blocks=nn.ModuleList([nn.Conv1d(64,64,3,dilation=d) for d in (1,2,4,8,10)])
        self.head=nn.Conv1d(64,2,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self,x):
        # x [batch,frames,features]; causal left padding at each dilation.
        h=self.input(x.transpose(1,2))
        for layer,d in zip(self.blocks,(1,2,4,8,10)):
            z=F.pad(h,(2*d,0))
            h=h+torch.tanh(layer(z))*.1
        return self.head(h).transpose(1,2)


def setup_determinism(device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    torch.manual_seed(SEED); np.random.seed(SEED)
    if device.type=="cuda": torch.cuda.manual_seed_all(SEED)


def build_examples(items,device):
    result=[]
    for item in items:
        feats=torch.from_numpy(feature_np(item["input"]))[None].to(device)
        n=min(feats.shape[1],len(item["tail"]))
        labels=torch.from_numpy(np.stack((item["tail"][:n],item["speech"][:n]),axis=-1).astype(np.float32))[None].to(device)
        valid=torch.from_numpy(np.broadcast_to(item["valid"][:n,None],(n,2)).copy())[None].to(device)
        result.append((item,feats[:,:n],labels,valid))
    return result


def train(train_wet,train_dry,device):
    setup_determinism(device)
    model=TailTCN().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    examples=build_examples(train_wet+train_dry,device)
    rng=torch.Generator(device="cpu").manual_seed(SEED+1)
    order=[]
    while len(order)<STEPS: order.extend(torch.randperm(len(examples),generator=rng).tolist())
    for idx in order[:STEPS]:
        _,x,y,valid=examples[idx]
        opt.zero_grad(set_to_none=True)
        logits=model(x)
        loss=(F.binary_cross_entropy_with_logits(logits,y,reduction="none")*valid).sum()/(valid.sum()+1e-8)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite BCE")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        if not torch.isfinite(norm): raise FloatingPointError("nonfinite gradient")
        opt.step()
    return model


def envelope_from_logits(logits, n_samples):
    probs=torch.sigmoid(logits).detach().cpu().numpy()
    p_tail,p_speech=probs[:,0],probs[:,1]
    decision=(p_tail>=.95)&(p_speech<=.05)
    gate=np.zeros_like(decision)
    gate[1:]=decision[1:]&decision[:-1]
    # Interpolate the binary frame command to samples, then ramp each edge over 10ms.
    frame_idx=np.clip(np.rint(np.arange(n_samples)*SR/(HOP*SR)).astype(int),0,len(gate)-1)
    binary=gate[frame_idx].astype(np.float64)
    alpha=1.0/(.010*SR)
    env=np.ones(n_samples,np.float64)
    target=1.0
    for i,b in enumerate(binary):
        desired=GAIN if b else 1.0
        target += np.clip(desired-target,-alpha,alpha)
        env[i]=target
    return env.astype(np.float32),decision,probs


def apply(model,item,device):
    feat=torch.from_numpy(feature_np(item["input"]))[None].to(device)
    with torch.inference_mode(): logits=model(feat)[0]
    env,decision,probs=envelope_from_logits(logits,item["input"].size)
    y=item["input"]*env
    return y,decision,probs,env


def band_reduction(y,x,lo_ms,hi_ms):
    ys,xs=spec(y),spec(x); freq=np.fft.rfftfreq(N_FFT,d=1/SR)
    bins=(freq>=150)&(freq<=300); centers=np.arange(xs.shape[-1])*HOP
    frames=(centers>=PAUSE+lo_ms*16)&(centers<PAUSE+hi_ms*16)
    ein=np.abs(xs[np.ix_(bins,frames)])**2; eout=np.abs(ys[np.ix_(bins,frames)])**2
    return float(10*np.log10((ein.mean()+1e-20)/(eout.mean()+1e-20)))


def preservation(y,x,dry):
    yy=np.lib.stride_tricks.sliding_window_view(y,320)[::160]
    xx=np.lib.stride_tricks.sliding_window_view(x,320)[::160]
    count=min(len(yy),len(xx)); yy,xx=yy[:count],xx[:count]
    db=20*np.log10((np.sqrt(np.mean(yy.astype(np.float64)**2,axis=1))+1e-8)/
                   (np.sqrt(np.mean(xx.astype(np.float64)**2,axis=1))+1e-8))
    _,active,weak,onset=activity(dry); n=min(count,len(active))
    db,active,weak,onset=db[:n],active[:n],weak[:n],onset[:n]
    def q(mask,p): return float(np.quantile(db[mask],p)) if mask.any() else None
    return {"active_p50_db":q(active,.5),"active_p10_db":q(active,.1),
        "weak_p10_db":q(weak,.1),"onset_p10_db":q(onset,.1),
        "weak_below_minus3_fraction":float(np.mean(db[weak]<-3)) if weak.any() else 0.,
        "weak_below_minus6_fraction":float(np.mean(db[weak]<-6)) if weak.any() else 0.}


def metrics(model, wet, dry, device):
    wet_rows=[]; dry_rows=[]; zero_acts=0; zero_total=0
    for item in wet:
        y,dec,prob,env=apply(model,item,device)
        wet_rows.append({"id":item["id"],"w1_mlow_reduction_db":band_reduction(y,item["input"],150,300),
            "w2_mlow_reduction_db":band_reduction(y,item["input"],300,600),
            "tail_precision":float(np.sum(dec&item["tail"])/(np.sum(dec)+1e-8)),
            "tail_recall":float(np.sum(dec&item["tail"])/(np.sum(item["tail"])+1e-8)),
            "gate_frame_fraction":float(np.mean(dec)),
            "w1_tail_energy_coverage":float(np.sum(item["late"][PAUSE+150*16:PAUSE+300*16]**2 *
                (env[PAUSE+150*16:PAUSE+300*16]<.999))/
                (np.sum(item["late"][PAUSE+150*16:PAUSE+300*16]**2)+1e-20)),
            "p_tail_regions":{"speech":float(np.median(prob[item["protected"][:len(prob)],0])) if item["protected"].any() else 0.,
                "tail":float(np.median(prob[item["tail"][:len(prob)],0])) if item["tail"].any() else 0.}})
    for item in dry:
        y,dec,prob,env=apply(model,item,device)
        keep=min(len(dec),len(item["protected"]))
        dry_rows.append({"id":item["id"],**preservation(y,item["input"],item["dry"]),
            "tail_false_positive_fraction":float(np.mean(dec)),
            "active_actuation_fraction":float(np.mean(dec[item["protected"][:keep]])) if item["protected"][:keep].any() else 0.,
            "onset_actuation_fraction":float(np.mean(dec[item["onset"][:keep]])) if item["onset"][:keep].any() else 0.})
    z=np.zeros(N,np.float32)
    for _ in range(100):
        y,dec,_,_=apply(model,{"input":z},device)
        zero_total+=1; zero_acts+=int(np.any(dec))
        if not np.array_equal(y,z): raise RuntimeError("zero input changed")
    return wet_rows,dry_rows,zero_total,zero_acts


def run(device):
    train_wet=[wet_item(i,False) for i in range(TRAIN_WET)]
    eval_wet=[wet_item(i,True) for i in range(EVAL_WET)]
    train_dry=[dry_item(i,False) for i in range(TRAIN_DRY)]
    eval_dry=[dry_item(i,True) for i in range(EVAL_DRY)]
    train_ids={x["id"] for x in train_wet+train_dry}; eval_ids={x["id"] for x in eval_wet+eval_dry}
    if train_ids & eval_ids: raise RuntimeError("train/eval ID leakage")
    model=train(train_wet,train_dry,device)
    wet_rows,dry_rows,zero_total,zero_acts=metrics(model,eval_wet,eval_dry,device)
    w1=np.array([x["w1_mlow_reduction_db"] for x in wet_rows]); w2=np.array([x["w2_mlow_reduction_db"] for x in wet_rows])
    dry_gate=lambda key,fn: all(x[key] is not None and fn(x[key]) for x in dry_rows)
    active_act=max(x["active_actuation_fraction"] for x in dry_rows)
    onset_act=max(x["onset_actuation_fraction"] for x in dry_rows)
    gates={"w1_median_ge_2db":float(np.median(w1))>=2.,"w1_positive_ge_7_of_8":int(np.sum(w1>0))>=7,
        "w2_median_ge_minus_0_5db":float(np.median(w2))>=-.5,"no_w2_below_minus1db":bool(np.all(w2>=-1.)),
        "dry_active_p50_range_each":dry_gate("active_p50_db",lambda x:-.25<=x<=.25),
        "dry_active_p10_gt_minus0_75_each":dry_gate("active_p10_db",lambda x:x>-.75),
        "dry_weak_p10_gt_minus1_each":dry_gate("weak_p10_db",lambda x:x>-1.),
        "dry_onset_p10_gt_minus0_75_each":dry_gate("onset_p10_db",lambda x:x>-.75),
        "dry_weak_below_minus3_le_3pct_each":dry_gate("weak_below_minus3_fraction",lambda x:x<=.03),
        "dry_weak_below_minus6_le_1pct_each":dry_gate("weak_below_minus6_fraction",lambda x:x<=.01),
        "dry_active_actuation_le_0_5pct":active_act<=.005,"dry_onset_actuation_le_0_5pct":onset_act<=.005,
        "exact_zero_100_of_100":zero_total==100 and zero_acts==0,
        "finite_metrics":bool(np.isfinite(w1).all() and np.isfinite(w2).all() and
            all(math.isfinite(v) for row in dry_rows for k,v in row.items() if k!="id"))}
    return {"experiment":"M-tail-gate-synthetic-preflight-v1","overall":"PASS" if all(gates.values()) else "NO_GO",
        "device":str(device),"gpu_name":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
        "torch_version":torch.__version__,"cuda_version":torch.version.cuda,"seed":SEED,
        "architecture":{"type":"causal TCN two-head detector plus fixed conditional attenuation",
            "features":65,"feature_components":{"log_energy_bands":32,"temporal_derivative_bands":32,"spectral_flux":1},
            "hidden_channels":64,"dilations":[1,2,4,8,10],"receptive_field_frames":51,
            "receptive_field_ms":51*HOP*1000/SR,"parameters":sum(p.numel() for p in model.parameters()),
            "gate":"two consecutive p_tail>=.95 and p_speech<=.05","gain_db":-4.,"attack_release_ms":10.},
        "training":{"steps":STEPS,"learning_rate":2e-4,"weight_decay":1e-4,"gradient_clip":3.,
            "loss":"masked two-head BCE, weights 1/1","updates_deterministic":True},
        "counts":{"train_wet":TRAIN_WET,"eval_wet":EVAL_WET,"train_dry":TRAIN_DRY,"eval_dry":EVAL_DRY,
            "train_eval_ids_disjoint":True,"new_seed_namespace":True},
        "thresholds":{"w1_median_db_min":2.,"w1_positive_min":7,"w2_median_db_min":-.5,"w2_each_db_min":-1.,
            "dry_active_p50_db_range":[-.25,.25],"dry_active_p10_db_gt":-.75,"dry_weak_p10_db_gt":-1.,
            "dry_onset_p10_db_gt":-.75,"weak_below_minus3_fraction_max":.03,"weak_below_minus6_fraction_max":.01,
            "dry_active_actuation_max":.005,"dry_onset_actuation_max":.005},
        "wet_metrics":{"w1_median_db":float(np.median(w1)),"w1_positive_count":int(np.sum(w1>0)),
            "w2_median_db":float(np.median(w2)),"rooms":wet_rows},
        "dry_metrics":{"maximum_active_actuation_fraction":active_act,"maximum_onset_actuation_fraction":onset_act,
            "fixtures":dry_rows},"gates":gates,"exact_zero":{"checks":zero_total,"actuations":zero_acts},
        "corpus_read":False,"dev_or_holdout_read":False,"test_wav_accessed":False,
        "source_sha256":{"audit":sha(Path(__file__))}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    result=run(device); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n"); print(json.dumps(result,indent=2,allow_nan=False))
    if result["overall"]!="PASS": raise SystemExit(2)

if __name__=="__main__": main()
