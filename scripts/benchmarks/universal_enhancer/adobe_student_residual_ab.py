#!/usr/bin/env python3
"""Speaker-held-out A/B for direct imitation vs structured residual learning.

Teacher-derived files/checkpoints remain ignored local research artifacts.
The user's test.wav and its Adobe render are never read.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from adobe_student_pilot import HOP, NFFT, RATE, SPEAKERS, Student, score, speech_masks
from adobe_student_overfit import mrstft
from adobe_student_multidomain import (
    ROOT, alignment_audit, dataset, load_pair_arrays, preservation_metrics,
)


class ResidualStudent(Student):
    """Same 69k-parameter trunk, predicting a complex component to remove."""
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        length = audio.shape[-1]
        window = self.window.to(audio)
        spec = torch.stft(audio, NFFT, HOP, NFFT, window, center=True,
                          return_complex=True)
        feats = torch.stack((spec.real, spec.imag), dim=1)
        e1 = self.enc1(feats)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        u2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat((u2, e2), dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat((u1, e1), dim=1))
        # Complex subtraction is bounded at |X| per real/imag component; the
        # model cannot arbitrarily synthesize unbounded content.
        removal = torch.tanh(self.head(u1)) * spec.abs().unsqueeze(1)
        removed_spec = torch.complex(removal[:, 0], removal[:, 1])
        enhanced = spec - removed_spec
        return torch.istft(enhanced, NFFT, HOP, NFFT, window, center=True, length=length)


def complex_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    terms = []
    for nfft in (256, 512, 1024, 2048):
        hop = nfft//4
        w = torch.hann_window(nfft, device=pred.device, dtype=pred.dtype)
        p = torch.stft(pred, nfft, hop, nfft, w, center=True, return_complex=True)
        t = torch.stft(target, nfft, hop, nfft, w, center=True, return_complex=True)
        terms.append((F.l1_loss(p.real, t.real)+F.l1_loss(p.imag, t.imag))/2)
    return torch.stack(terms).mean()


def masked_logmag_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    terms = []
    for nfft in (256, 512, 1024, 2048):
        hop = nfft//4
        w = torch.hann_window(nfft, device=pred.device, dtype=pred.dtype)
        p = torch.stft(pred, nfft, hop, nfft, w, center=True, return_complex=True).abs()
        t = torch.stft(target, nfft, hop, nfft, w, center=True, return_complex=True).abs()
        # Ignore teacher residual bins below -50 dB relative to each crop peak.
        mask = t > t.amax(dim=(-2,-1), keepdim=True).clamp_min(1e-8)*10**(-50/20)
        err = torch.abs(torch.log1p(25*p)-torch.log1p(25*t))
        terms.append((err*mask).sum()/mask.sum().clamp_min(1))
    return torch.stack(terms).mean()


def mean_mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value*mask).sum()/mask.sum().clamp_min(1)


def losses(model: nn.Module, x: torch.Tensor, teacher: torch.Tensor,
           clean: torch.Tensor | None, arm: str) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(x)
    if arm == "direct-control":
        imitation = mrstft(output, teacher)+2*F.l1_loss(output, teacher)
        protect = torch.zeros((), device=x.device)
        silence = torch.zeros((), device=x.device)
        if clean is not None:
            active, weak, onset = speech_masks(clean)
            p = torch.sqrt(output.unfold(-1,480,240).square().mean(-1)+1e-10)
            c = torch.sqrt(clean.unfold(-1,480,240).square().mean(-1)+1e-10)
            delta = 20*torch.log10((p+1e-5)/(c+1e-5))
            drop = F.relu(-1-delta).square()
            protect = (mean_mask(drop,active)+1.5*mean_mask(drop,weak)+mean_mask(drop,onset))
            silence = mean_mask(p,1-active)
        total = imitation+0.025*protect+0.04*silence
        return total, {"teacher":float(imitation.detach()),"protect":float(protect.detach()),
                       "silence":float(silence.detach())}

    r_teacher = x-teacher
    r_hat = x-output
    residual = (F.l1_loss(r_hat,r_teacher)
                +0.5*complex_stft_loss(r_hat,r_teacher)
                +0.25*masked_logmag_loss(r_hat,r_teacher))
    output_tie = 0.25*(F.l1_loss(output,teacher)+mrstft(output,teacher))
    total = residual+output_tie
    stats = {"residual":float(residual.detach()),"output_tie":float(output_tie.detach())}
    if arm == "residual-naive" or clean is None:
        return total, stats

    # Clean/noise decomposition: n = x-c. In STFT bins where clean speech
    # dominates (>90% power), discourage placing that speech in removal.
    n = x-clean
    mask_res = []
    for nfft in (512,1024):
        hop=nfft//4
        w=torch.hann_window(nfft,device=x.device,dtype=x.dtype)
        X=torch.stft(x,nfft,hop,nfft,w,center=True,return_complex=True)
        C=torch.stft(clean,nfft,hop,nfft,w,center=True,return_complex=True)
        N=torch.stft(n,nfft,hop,nfft,w,center=True,return_complex=True)
        R=torch.stft(r_hat,nfft,hop,nfft,w,center=True,return_complex=True)
        q=C.abs().square()/(C.abs().square()+N.abs().square()+1e-10)
        speech_bin=q>0.9
        leak=(R.abs().square()/(X.abs().square()+1e-8))
        mask_res.append((leak*speech_bin).sum()/speech_bin.sum().clamp_min(1))
    leak=torch.stack(mask_res).mean()
    active,weak,onset=speech_masks(clean)
    p=torch.sqrt(output.unfold(-1,480,240).square().mean(-1)+1e-10)
    c=torch.sqrt(clean.unfold(-1,480,240).square().mean(-1)+1e-10)
    gain=20*torch.log10((p+1e-5)/(c+1e-5))
    drop=F.relu(-1-gain).square()
    protect=(mean_mask(drop,weak)+mean_mask(drop,onset))
    silence=mean_mask(p,1-active)
    total=total+0.5*leak+0.5*protect+0.1*silence
    stats.update({"speech_leak":float(leak.detach()),"weak_onset_drop":float(protect.detach()),
                  "clean_silence":float(silence.detach())})
    return total,stats


def make_schedule(held: str, pairs: list, arrays: dict, steps: int, seed: int):
    random.seed(seed)
    train=[p for p in pairs if p.speaker_id!=held]
    by_domain={d:[p for p in train if p.domain==d] for d in sorted({p.domain for p in train})}
    crop=2*RATE; schedule=[]
    for _ in range(steps):
        domain=random.choice(list(by_domain))
        pair=random.choice(by_domain[domain])
        x=arrays[pair.pair_id][0]
        length=min(crop,len(x))
        pos=random.randrange(max(1,len(x)-length+1))
        schedule.append((pair.pair_id,pos,length,domain))
    return schedule


def train_arm(held: str, arm: str, pairs: list, arrays: dict, schedule: list,
              out: Path, seed: int, width: int, device: torch.device) -> tuple[nn.Module,dict]:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    model=(Student(width) if arm=="direct-control" else ResidualStudent(width)).to(device)
    params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=1e-5)
    fold_dir=out/held/arm;fold_dir.mkdir(parents=True,exist_ok=True)
    domains={};history=[];start=time.perf_counter();model.train()
    for i,(pair_id,pos,length,domain) in enumerate(schedule,1):
        domains[domain]=domains.get(domain,0)+1
        x,t,c=arrays[pair_id]
        tx=torch.from_numpy(x[pos:pos+length]).unsqueeze(0).to(device)
        tt=torch.from_numpy(t[pos:pos+length]).unsqueeze(0).to(device)
        tc=torch.from_numpy(c[pos:pos+length]).unsqueeze(0).to(device) if c is not None else None
        opt.zero_grad(set_to_none=True)
        loss,parts=losses(model,tx,tt,tc,arm)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        if i==1 or i%1000==0 or i==len(schedule):
            history.append({"step":i,"loss":float(loss.detach()),"domain":domain,"pair_id":pair_id,**parts})
    duration=time.perf_counter()-start
    torch.save({"model":model.cpu().state_dict(),"arm":arm,"held_out_speaker":held,
        "steps":len(schedule),"seed":seed,"width":width,"parameters":params,
        "sample_rate_hz":RATE,"architecture":"shared 69k Student trunk; bounded complex residual head",
        "train_seconds":duration},fold_dir/"checkpoint.pt")
    model.to(device).eval()
    info={"held_out_speaker":held,"arm":arm,"steps":len(schedule),"seed":seed,
          "parameters":params,"train_seconds":duration,"sampled_by_domain":domains,"history":history}
    (fold_dir/"training.json").write_text(json.dumps(info,indent=2)+"\n")
    return model,info


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261011)
    p.add_argument("--width",type=int,default=12)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--out",type=Path,default=ROOT/"results/adobe-student-residual-ab-2026-10-11")
    args=p.parse_args();torch.set_num_threads(2)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else
                        "cpu" if args.device=="auto" else args.device)
    pairs=dataset();audit=[alignment_audit(p) for p in pairs]
    if not all(a["included"] for a in audit):raise SystemExit("An unaligned pair failed the audit")
    arrays={p.pair_id:load_pair_arrays(p) for p in pairs}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"dataset-manifest.json").write_text(json.dumps({
        "pairs":[p.__dict__ for p in pairs],"alignment_audit":audit,
        "user_test.wav":"sealed; never opened"},indent=2)+"\n")
    arms=("direct-control","residual-naive","residual-structured")
    results=[];training=[]
    for fold_i,held in enumerate(SPEAKERS):
        seed=args.seed+fold_i
        schedule=make_schedule(held,pairs,arrays,args.steps,seed)
        for arm in arms:
            print(json.dumps({"event":"training_start","held":held,"arm":arm,"steps":args.steps}),flush=True)
            checkpoint=args.out/held/arm/"checkpoint.pt"
            training_log=args.out/held/arm/"training.json"
            if checkpoint.exists() and training_log.exists():
                model=(Student(args.width) if arm=="direct-control" else ResidualStudent(args.width))
                saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                model.load_state_dict(saved["model"]);model.to(device).eval()
                info=json.loads(training_log.read_text())
                print(json.dumps({"event":"checkpoint_reused","held":held,"arm":arm}),flush=True)
            else:
                model,info=train_arm(held,arm,pairs,arrays,schedule,args.out,seed,args.width,device)
            training.append(info)
            x,teacher,clean=arrays[held];assert clean is not None
            render_dir=args.out/held/arm/"heldout";render_dir.mkdir(parents=True,exist_ok=True)
            row=score(model,held,(x,teacher,clean),render_dir,device)
            row.update({"held_out_speaker":held,"arm":arm,"train_seconds":info["train_seconds"],
                        "parameters":info["parameters"],"preservation":preservation_metrics(row["audio_paths"])})
            results.append(row)
            print(json.dumps({"event":"fold_complete",**row}),flush=True)
    for held in SPEAKERS:
        baseline=next(r for r in results if r["held_out_speaker"]==held and r["arm"]=="direct-control")
        for row in results:
            if row["held_out_speaker"]==held:
                row["teacher_distance_gain_vs_input_db"]=row["input_adobe_logstft_mae_db"]-row["student_adobe_logstft_mae_db"]
                row["teacher_distance_gain_vs_direct_db"]=baseline["student_adobe_logstft_mae_db"]-row["student_adobe_logstft_mae_db"]
    fields=list(results[0])
    with (args.out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    colors={"direct-control":"#64748b","residual-naive":"#f59e0b","residual-structured":"#7c3aed"}
    xx=np.arange(len(SPEAKERS));width=.23
    for j,arm in enumerate(arms):
        rows=[next(r for r in results if r["held_out_speaker"]==s and r["arm"]==arm) for s in SPEAKERS]
        ax[0].bar(xx+(j-1)*width,[r["student_adobe_logstft_mae_db"] for r in rows],width,label=arm,color=colors[arm])
        ax[1].bar(xx+(j-1)*width,[r["preservation"]["student"]["weak_level_p10_db_vs_clean"] for r in rows],width,label=arm,color=colors[arm])
    ax[0].set_xticks(xx,SPEAKERS,rotation=10);ax[0].set_ylabel("Student → Adobe log-STFT MAE dB (lower closer)")
    ax[0].set_title("Speaker-held-out teacher distance");ax[0].grid(axis="y",alpha=.2);ax[0].legend()
    ax[1].axhline(-3,color="#dc2626",ls="--",lw=1,label="−3 dB guard")
    ax[1].set_xticks(xx,SPEAKERS,rotation=10);ax[1].set_ylabel("Weak-frame p10 dB vs clean (higher preserves)")
    ax[1].set_title("Clean speech preservation");ax[1].grid(axis="y",alpha=.2);ax[1].legend()
    fig.savefig(args.out/"summary.png",dpi=180);plt.close(fig)
    report={"title":"Adobe teacher/student residual-decomposition A/B",
      "purpose":"Test whether clean/noise decomposition constraints improve held-out transfer beyond a residual reparameterization.",
      "device":str(device),"steps_per_fold_arm":args.steps,"seed":args.seed,"sample_rate_hz":RATE,
      "holdout":"speaker-strict Emy/Rémi/Stéphanie LOSO; test.wav and its Adobe render remain sealed.",
      "arms":list(arms),"losses":{"direct-control":"4-res MR-STFT + 2x waveform L1 + pooled clean weak/onset/silence guard",
        "residual-naive":"waveform L1 residual + 0.5 complex-STFT L1 residual + 0.25 masked log-magnitude residual + 0.25 output L1/MR-STFT tie",
        "residual-structured":"residual-naive + 0.5 speech-dominant-bin removal penalty + 0.5 weak/onset one-sided drop guard + 0.1 clean-silence output penalty; clean terms only when reliable clean stem exists"},
      "speech_dominant_bins":"q=|STFT(clean)|^2/(|STFT(clean)|^2+|STFT(input-clean)|^2+eps)>0.9; penalize |Rhat|^2/(|X|^2+eps)",
      "alignment":{"all_pairs_passed":True,"method":"48k mono sample counts and five-spaced 10ms RMS envelope lag checks; no warping."},
      "teacher_derivative_status":"All Adobe-derived data/checkpoints stay local pending applicable terms review.",
      "pre_registered_gate":{"teacher_distance_gain_each_db_min":0.20,"median_gain_db_min":0.30,
        "all_three_folds_must_improve":True,"weak_p10_db_floor":-3.0,"onset_p10_db_floor":-3.0,
        "must_beat_residual_naive_folds_min":2,"test.wav_used":False},
      "training_details":training,"folds":results,
      "peak_allocated_vram_mib":torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None,
      "limitations":["Only three held-out voices; crops are not independent speakers.",
       "r=x-Adobe alone is algebraically a reparameterization; only clean/noise constraints distinguish structured residual.",
       "Adobe output remains a style target, not clean truth or evidence of internal implementation.",
       "Complex residual phase follows teacher waveform and is not a perceptual objective.",
       "24kHz caps output at 12kHz; no streaming latency or universal-quality claim."]}
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"report":str(args.out/"report.json"),"summary":str(args.out/"summary.png"),
        "peak_vram_mib":report["peak_allocated_vram_mib"],
        "median_train_seconds":float(np.median([r["train_seconds"] for r in results])),
        "median_inference_seconds":float(np.median([r["inference_s"] for r in results]))},indent=2))


if __name__=="__main__":main()
