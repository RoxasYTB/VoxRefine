#!/usr/bin/env python3
"""Phase-aware and nuisance-removal diagnostics for existing residual A/B WAVs.

Only reads local held-out renders from adobe_student_residual_ab.py. It never
opens the user's test.wav and does not train or alter any checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "results/adobe-student-residual-ab-2026-10-11"
SPEAKERS = ("emy_mixed_ambience18", "remi_crowd18", "stephanie_fan18")
ARMS = ("direct-control", "residual-naive", "residual-structured")
NFFTS = (256, 512, 1024, 2048)


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    return data.mean(axis=1), int(rate)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-30))


def corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-20 else None


def complex_stfts(x: np.ndarray, rate: int) -> list[np.ndarray]:
    result = []
    for nfft in NFFTS:
        _, _, z = stft(x, fs=rate, window="hann", nperseg=nfft,
                       noverlap=3*nfft//4, nfft=nfft, boundary="zeros",
                       padded=True, return_onesided=True, scaling="spectrum")
        result.append(z)
    return result


def normalized_wave_l1(pred: np.ndarray, target: np.ndarray) -> float:
    # Absolute waveform distance normalized by the fixed teacher RMS. No
    # candidate-specific gain matching is applied.
    return float(np.mean(np.abs(pred-target)) / max(rms(target), 1e-12))


def normalized_complex_stft(pred: np.ndarray, target: np.ndarray, rate: int) -> float:
    terms = []
    for p, t in zip(complex_stfts(pred, rate), complex_stfts(target, rate)):
        terms.append(float(np.mean(np.abs(p-t)) / max(np.mean(np.abs(t)), 1e-12)))
    return float(np.mean(terms))


def waveform_nmse_db(pred: np.ndarray, target: np.ndarray) -> float:
    return float(10*np.log10((np.mean((pred-target)**2)+1e-30)/(np.mean(target**2)+1e-30)))


def masked_complex_cosine(a: list[np.ndarray], b: list[np.ndarray], mask: list[np.ndarray]) -> float | None:
    av = np.concatenate([x[m] for x, m in zip(a, mask)])
    bv = np.concatenate([x[m] for x, m in zip(b, mask)])
    den = np.linalg.norm(av)*np.linalg.norm(bv)
    if den < 1e-20:
        return None
    # Signed real part: +1 means aligned removal/noise components.
    return float(np.real(np.vdot(av, bv))/den)


def analyze_fold(run: Path, speaker: str) -> list[dict]:
    base = run/speaker
    x_path = base/"direct-control/heldout"/f"{speaker}-noisy.wav"
    c_path = base/"direct-control/heldout"/f"{speaker}-clean.wav"
    a_path = base/"direct-control/heldout"/f"{speaker}-adobe-v2.wav"
    x, xr = read_mono(x_path)
    c, cr = read_mono(c_path)
    adobe, ar = read_mono(a_path)
    if len({xr, cr, ar}) != 1 or len({len(x),len(c),len(adobe)}) != 1:
        raise ValueError(f"Rate/sample mismatch for {speaker}: {xr}/{cr}/{ar}, {len(x)}/{len(c)}/{len(adobe)}")
    rate = xr
    nuisance = x-c
    xs, cs, ns = complex_stfts(x,rate), complex_stfts(c,rate), complex_stfts(nuisance,rate)
    speech_masks=[];noise_masks=[]
    for C,N in zip(cs,ns):
        q=(np.abs(C)**2)/(np.abs(C)**2+np.abs(N)**2+1e-20)
        speech_masks.append(q>0.9)
        noise_masks.append(q<0.1)

    references={"input":x,"direct-control":None,"residual-naive":None,
                "residual-structured":None,"adobe":adobe}
    rows=[]
    for arm in ARMS:
        y_path=base/arm/"heldout"/f"{speaker}-student.wav"
        y, yr=read_mono(y_path)
        if yr!=rate or len(y)!=len(x):
            raise ValueError(f"Student sample mismatch for {speaker}/{arm}")
        references[arm]=y
    for name,y in references.items():
        if y is None:
            continue
        removed=x-y
        ys=complex_stfts(y,rate)
        rs=[X-Y for X,Y in zip(xs,ys)]
        speech_energy=sum(float(np.sum(np.abs(R[m])**2)) for R,m in zip(rs,speech_masks))
        clean_speech_energy=sum(float(np.sum(np.abs(C[m])**2)) for C,m in zip(cs,speech_masks))
        noise_energy=sum(float(np.sum(np.abs(N[m])**2)) for N,m in zip(ns,noise_masks))
        removed_noise_energy=sum(float(np.sum(np.abs(R[m])**2)) for R,m in zip(rs,noise_masks))
        is_noop=name=="input"
        row={"speaker":speaker,"candidate":name,
             "waveform_l1_over_adobe_rms":normalized_wave_l1(y,adobe),
             "waveform_nmse_db_vs_adobe":waveform_nmse_db(y,adobe),
             "complex_stft_l1_relative":normalized_complex_stft(y,adobe,rate),
             "adobe_residual_waveform_corr":corr(removed,x-adobe),
             "removed_component_waveform_corr_with_noise":None if is_noop else corr(removed,nuisance),
             "removed_component_complex_cosine_noise_dominant":None if is_noop else masked_complex_cosine(rs,ns,noise_masks),
             "removed_component_complex_cosine_speech_dominant":None if is_noop else masked_complex_cosine(rs,cs,speech_masks),
             "speech_leakage_db_removed_over_clean_speech_q_gt_0_9":None if is_noop else 10*np.log10((speech_energy+1e-30)/(clean_speech_energy+1e-30)),
             "noise_capture_db_removed_over_noise_q_lt_0_1":None if is_noop else 10*np.log10((removed_noise_energy+1e-30)/(noise_energy+1e-30)),
             "removed_rms_dbfs":None if is_noop else 20*np.log10(rms(removed)+1e-15),
             "sample_rate_hz":rate,"samples":len(y)}
        rows.append(row)
    return rows


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=DEFAULT_RUN)
    args=p.parse_args()
    rows=[r for speaker in SPEAKERS for r in analyze_fold(args.run,speaker)]
    by={(r["speaker"],r["candidate"]):r for r in rows}
    for row in rows:
        base=by[(row["speaker"],"input")]
        for metric in ("waveform_l1_over_adobe_rms","complex_stft_l1_relative"):
            row[f"{metric}_relative_improvement_vs_input_pct"] = (
                100*(base[metric]-row[metric])/base[metric] if base[metric] else None)
    structured=[by[(s,"residual-structured")] for s in SPEAKERS]
    old_rows={}
    with (args.run/"metrics.csv").open(newline="") as f:
        for old in csv.DictReader(f):
            old_rows[(old["held_out_speaker"],old["arm"])]=ast.literal_eval(old["preservation"])
    weak_p10=[float(old_rows[(s,"residual-structured")]["student"]["weak_level_p10_db_vs_clean"]) for s in SPEAKERS]
    onset_p10=[float(old_rows[(s,"residual-structured")]["student"]["onset_level_p10_db_vs_clean"]) for s in SPEAKERS]
    median_wave_improvement=float(np.median([r["waveform_l1_over_adobe_rms_relative_improvement_vs_input_pct"] for r in structured]))
    median_complex_improvement=float(np.median([r["complex_stft_l1_relative_relative_improvement_vs_input_pct"] for r in structured]))
    gates={
        "structured_beats_input_waveform_l1_all_3":all(r["waveform_l1_over_adobe_rms_relative_improvement_vs_input_pct"]>0 for r in structured),
        "structured_beats_input_complex_stft_all_3":all(r["complex_stft_l1_relative_relative_improvement_vs_input_pct"]>0 for r in structured),
        "structured_nuisance_alignment_positive_all_3":all((r["removed_component_complex_cosine_noise_dominant"] or 0)>0 for r in structured),
        "structured_weak_p10_above_minus3_db_all_3":all(v>=-3 for v in weak_p10),
        "structured_onset_p10_above_minus3_db_all_3":all(v>=-3 for v in onset_p10),
        "median_waveform_improvement_pct":median_wave_improvement,
        "median_complex_stft_improvement_pct":median_complex_improvement,
    }
    out=args.run/"phase-aware-metrics"
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps({
        "title":"Phase-aware and nuisance-removal diagnostics for existing residual A/B renders",
        "run_path":str(args.run),"test.wav":"sealed and never opened",
        "waveform_l1":"mean(abs(candidate-Adobe))/Adobe RMS; no per-candidate gain matching",
        "complex_stft":"mean complex absolute error / mean Adobe complex magnitude over FFT 256/512/1024/2048",
        "noise_and_speech_masks":"q=|STFT(clean)|^2/(|STFT(clean)|^2+|STFT(input-clean)|^2); noise q<0.1, speech q>0.9",
        "interpretation":"Descriptive local diagnostics only; three held-out speakers, correlated TF bins are not independent samples.",
        "pre_registered_gate":{"structured_must_beat_input_on_waveform_and_complex_stft_all_folds":True,
          "at_least_one_teacher_distance_metric_must_improve_over_5pct":True,
          "nuisance_capture_must_be_coherent_across_folds":True,
          "weak_p10_and_onset_p10_must_remain_above_minus3_db":True},
        "computed_gates":gates,
        "folds":rows},indent=2)+"\n")
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    colors={"direct-control":"#64748b","residual-naive":"#f59e0b","residual-structured":"#7c3aed"}
    xx=np.arange(len(SPEAKERS));width=.23
    for j,arm in enumerate(ARMS):
        selected=[by[(s,arm)] for s in SPEAKERS]
        wave=[r["waveform_l1_over_adobe_rms_relative_improvement_vs_input_pct"] for r in selected]
        cstft=[r["complex_stft_l1_relative_relative_improvement_vs_input_pct"] for r in selected]
        ax[0].bar(xx+(j-1)*width,wave,width,color=colors[arm],label=arm)
        ax[1].bar(xx+(j-1)*width,cstft,width,color=colors[arm],label=arm)
    for panel,title in ((ax[0],"Onde L1"),(ax[1],"STFT complexe")):
        panel.axhline(0,color="#111827",lw=.8)
        panel.set_xticks(xx,[s.replace("_", "\n") for s in SPEAKERS])
        panel.set_ylabel("Réduction d'erreur vs entrée (%)")
        panel.set_title(title);panel.grid(axis="y",alpha=.2);panel.legend(fontsize=8)
    for j,arm in enumerate(ARMS):
        selected=[by[(s,arm)] for s in SPEAKERS]
        vals=[r["removed_component_complex_cosine_noise_dominant"] for r in selected]
        ax[2].bar(xx+(j-1)*width,[0 if v is None else v for v in vals],width,
                  color=colors[arm],label=arm)
    ax[2].axhline(0,color="#111827",lw=.8)
    ax[2].set_xticks(xx,[s.replace("_", "\n") for s in SPEAKERS])
    ax[2].set_ylim(-1,1);ax[2].set_ylabel("Cosinus complexe résidu retiré / nuisance")
    ax[2].set_title("Bins dominés par la nuisance (q < 0,1)")
    ax[2].grid(axis="y",alpha=.2);ax[2].legend(fontsize=8)
    fig.savefig(out/"summary.png",dpi=180);plt.close(fig)
    for row in rows:
        print(json.dumps(row))
    print(f"Wrote {out}")


if __name__=="__main__":
    main()
