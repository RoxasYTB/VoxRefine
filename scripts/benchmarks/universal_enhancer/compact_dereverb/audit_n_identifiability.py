#!/usr/bin/env python3
"""Fresh diagnostic-only separability audit for M tail labels/features; emits no audio."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import audit_m_tail_gate as m

SEED=910_2026
TRAIN_ROOMS=16
TEST_ROOMS=16


def fixture(i, eval_set):
    base=SEED+(30_000 if eval_set else 0)
    dry=m.make_dry(base+10_000+i)
    h_e,h_l=m.make_rir(base+20_000+i)
    raw=np.abs(dry)>max(np.max(np.abs(dry))*.02,1e-5)
    q=.08/(np.sqrt(np.mean(dry[raw].astype(np.float64)**2))+1e-12)
    dry=(dry*q).astype(np.float32)
    _,protected,weak,onset=m.activity(dry)
    early,late=m.convolve(dry,h_e),m.convolve(dry,h_l)
    wet=(early.astype(np.float64)+late.astype(np.float64)).astype(np.float32)
    # Reproduce M's exact framewise label definition.
    win=np.hanning(m.N_FFT).astype(np.float64)
    def power(x):
        xp=np.pad(x.astype(np.float64),(m.N_FFT//2,m.N_FFT//2))
        fr=np.lib.stride_tricks.sliding_window_view(xp,m.N_FFT)[::m.HOP]
        return np.mean((fr*win)**2,axis=1)
    ep,lp=power(early),power(late)
    n=min(len(ep),len(lp),m.spec(wet).shape[-1])
    ratio=lp[:n]/(ep[:n]+lp[:n]+1e-20)
    protected_f=m.frame_activity(protected,n)
    late_db=10*np.log10(lp[:n]+1e-20)
    loud=late_db>-50.
    tail=(~protected_f)&loud&(ratio>=.8)
    ambiguous=(~protected_f)&loud&(ratio<.8)
    valid=(~ambiguous)|protected_f|tail
    return {"id":f"{'test' if eval_set else 'train'}-room-{i}","dry":dry,"wet":wet,
        "tail":tail,"protected":protected_f,"weak":m.frame_activity(weak,n),
        "onset":m.frame_activity(onset,n),"valid":valid,"ambiguous":ambiguous,
        "late_ratio":ratio,"late_db":late_db}


def vectors(item, audio, n=None):
    f=m.feature_np(audio)
    return f[:min(len(f),len(item['tail']) if n is None else n)]


def summarize_labels(items):
    rows=[]
    for x in items:
        n=len(x['tail']); t=int(x['tail'].sum()); valid=int(x['valid'].sum())
        rows.append({"id":x['id'],"frames":n,"tail_frames":t,
            "tail_fraction_of_valid":float(t/max(valid,1)),
            "tail_duration_s":float(t*m.HOP/m.SR),
            "ambiguous_fraction":float(x['ambiguous'].mean()),
            "tail_frames_late_energy_gt_m50":int(np.sum(x['tail']&(x['late_db']>-50))),
            "tail_frames_ratio_ge_0_8":int(np.sum(x['tail']&(x['late_ratio']>=.8)))})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    train=[fixture(i,False) for i in range(TRAIN_ROOMS)]
    test=[fixture(i,True) for i in range(TEST_ROOMS)]
    train_ids={x['id'] for x in train};test_ids={x['id'] for x in test}
    if train_ids&test_ids: raise RuntimeError('room/source leakage')
    X=[];y=[]
    for x in train:
        f=vectors(x,x['wet']);n=min(len(f),len(x['tail']))
        mask=x['valid'][:n]
        X.append(f[:n][mask]);y.append(x['tail'][:n][mask].astype(int))
    Xtr=np.concatenate(X);ytr=np.concatenate(y)
    # Fixed, non-tuned linear probe; balanced weighting handles the rare TAIL class.
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',
        solver='lbfgs',max_iter=500,random_state=SEED))
    clf.fit(Xtr,ytr)
    all_y=[];all_s=[];tail_s=[];active_scores=[];onset_scores=[];rows=[]
    dry_active_scores=[];dry_onset_scores=[]
    for x in test:
        f=vectors(x,x['wet']);n=min(len(f),len(x['tail']))
        valid=x['valid'][:n];scores=clf.predict_proba(f[:n])[:,1]
        yt=x['tail'][:n][valid];st=scores[valid]
        all_y.extend(yt.tolist());all_s.extend(st.tolist())
        tail_s.extend(scores[:n][x['tail'][:n]].tolist())
        active=x['protected'][:n];onset=x['onset'][:n]
        active_scores.extend(scores[:n][active].tolist());onset_scores.extend(scores[:n][onset].tolist())
        # Separate dry challenge: speech-active and onset from same held-out source.
        fd=vectors(x,x['dry'],n);sd=clf.predict_proba(fd)[:,1]
        dry_active_scores.extend(sd[:n][active].tolist());dry_onset_scores.extend(sd[:n][onset].tolist())
        rows.append({"id":x['id'],"tail_frames":int(x['tail'][:n].sum()),
            "tail_score_median":float(np.median(scores[:n][x['tail'][:n]])) if x['tail'][:n].any() else None,
            "wet_active_score_median":float(np.median(scores[:n][active])) if active.any() else None,
            "dry_active_score_median":float(np.median(sd[:n][active])) if active.any() else None})
    all_y=np.asarray(all_y);all_s=np.asarray(all_s)
    if len(np.unique(all_y))<2: raise RuntimeError('probe-test lacks a label class')
    auc=float(roc_auc_score(all_y,all_s));ap=float(average_precision_score(all_y,all_s))
    dact=np.asarray(dry_active_scores);dons=np.asarray(dry_onset_scores)
    best=None
    # Diagnostic existence test only: select the highest-recall ROC point meeting both FPR limits.
    for threshold in np.unique(all_s):
        recall=float(np.mean(np.asarray(tail_s)>=threshold)) if tail_s else 0.
        fpr_a=float(np.mean(dact>=threshold)) if len(dact) else 0.
        fpr_o=float(np.mean(dons>=threshold)) if len(dons) else 0.
        if recall>=.5 and fpr_a<=.005 and fpr_o<=.005:
            cand={"threshold_diagnostic_only":float(threshold),"tail_recall":recall,
                  "dry_active_fpr":fpr_a,"onset_fpr":fpr_o}
            if best is None or cand['tail_recall']>best['tail_recall']:best=cand
    if auc<.85 or best is None: decision='STOP_TAIL_DETECTOR_FAMILY'
    elif auc>=.95 and best is not None: decision='IDENTIFIABLE_TRAINING_OR_CALIBRATION_FAILURE'
    else: decision='INCONCLUSIVE_NO_NEW_AUDIO_ARCHITECTURE'
    result={"experiment":"N-identifiability-diagnostic-v1","decision":decision,
        "seed":SEED,"train_rooms":TRAIN_ROOMS,"test_rooms":TEST_ROOMS,
        "train_test_room_ids_disjoint":True,"train_frames":int(len(ytr)),
        "train_tail_positive_fraction":float(ytr.mean()),"test_roc_auc":auc,"test_pr_auc":ap,
        "operating_point_feasible":best is not None,"operating_point":best,
        "test_score_medians":{"tail":float(np.median(tail_s)) if tail_s else None,
            "wet_active":float(np.median(active_scores)) if active_scores else None,
            "dry_active":float(np.median(dact)) if len(dact) else None,
            "dry_onset":float(np.median(dons)) if len(dons) else None},
        "test_label_counts":{"valid":int(len(all_y)),"tail":int(all_y.sum()),"non_tail":int(len(all_y)-all_y.sum()),
            "ambiguous":int(sum(x['ambiguous'].sum() for x in test))},
        "train_room_labels":summarize_labels(train),"test_room_labels":summarize_labels(test),
        "test_room_score_medians":rows,
        "probe":{"model":"StandardScaler + LogisticRegression(L2,C=1,class_weight=balanced,lbfgs,max_iter=500)",
            "threshold_search":"diagnostic ROC operating-point existence only; never a product threshold"},
        "feature_caveat":"M feature implementation computes spectral-flux normalization using the whole utterance maximum; this is non-causal at inference and is reported as an inherited diagnostic limitation.",
        "audio_written":False,"corpus_read":False,"dev_or_holdout_read":False,"test_wav_accessed":False,
        "source_sha256":{"audit_m":hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
            "audit_n":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
