#!/usr/bin/env python3
"""DCCT-NIDS Block 09A-C-R2: read-only score-shift investigation with tqdm progress.

Score the saved 09A-B-R2 MLP checkpoint on its original R2 internal holdout,
official validation, and temporal test. All split scores and class summaries
are DIAGNOSTIC only. No threshold re-selection, model refit, or test feedback
to the predictor. Preserve the original 09A-B-R2 operating threshold.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/model_ready/compact_r2_6320003"
R2 = ROOT / "outputs/manifests/block_09a_a_r2_versioned_compact_manifest.json"
MLP = ROOT / "outputs/metrics/metric_09a_b_r2_gpu_mlp_summary.json"
CHECKPOINT = ROOT / "outputs/checkpoints/block_09a_b_r2_mlp_seed42.pt"
TABLES = ROOT / "outputs/tables"
METRICS = ROOT / "outputs/metrics"
MANIFESTS = ROOT / "outputs/manifests"
FIGURES = ROOT / "outputs/figures"
CAPTIONS = ROOT / "reports/captions"
REPORTS = ROOT / "reports/experiment_notes"
EXPECTED = {"train": 4_202_458, "validation": 1_145_712, "test": 971_833}
BATCH = 65_536


def insist(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def jsonout(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csvout(path, rows):
    insist(bool(rows), "No rows to write: " + str(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


@torch.inference_mode()
def predict(model, x, device):
    return model(torch.from_numpy(x).to(device)).sigmoid().cpu().numpy().astype(np.float64)


def holdout_membership(path, seed):
    print("Reconstructing the exact original internal 15% stratified holdout...", flush=True)
    # The 09A-B model selected this seed and split on fine labels in file order.
    fine = pd.read_parquet(path, columns=["fine_grained_label"])["fine_grained_label"].astype(str).to_numpy()
    insist(len(fine) == EXPECTED["train"], "Training count changed")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit, hold = next(splitter.split(np.zeros(len(fine), dtype=np.int8), fine))
    selection = np.zeros(len(fine), dtype=np.bool_)
    selection[hold] = True
    insist(int(selection.sum()) == 630369, "Internal holdout size changed")
    del fine, fit, hold
    return selection


def scan_split(split, m, cohort, model, device, hold_mask, threshold, clip):
    path = DATA / (split + "_compact.parquet")
    manifest = next(item for item in cohort["outputs"] if item["split"] == split)
    insist(digest(path) == manifest["sha256"], split + " file checksum changed")
    pf = pq.ParquetFile(path)
    insist(pf.metadata.num_rows == EXPECTED[split], split + " row count mismatch")
    cols = ["row_id", "fine_grained_label", "attack_family", "binary_label"] + m["features"]
    bundles = defaultdict(list)
    seen = 0
    selected = 0
    with tqdm(total=pf.metadata.num_rows, desc="GPU inference / " + split,
              unit="rows", unit_scale=True, dynamic_ncols=True, leave=True) as bar:
        for chunk in pf.iter_batches(batch_size=BATCH, columns=cols):
            frame = chunk.to_pandas()
            count = len(frame)
            ids = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
            insist(np.array_equal(ids, np.arange(seen + 1 + m["offsets"][split],
                                                  seen + count + 1 + m["offsets"][split])),
                   split + ": altered row_id ordering")
            choose = (hold_mask[seen:seen+count] if split == "train"
                      else np.ones(count, dtype=np.bool_))
            if choose.any():
                chunk_df = frame.loc[choose]
                features = chunk_df[m["features"]].to_numpy(dtype=np.float32, copy=True)
                insist(np.isfinite(features).all(), split + ": nonfinite feature")
                np.clip(features, -clip, clip, out=features)
                scores = predict(model, features, device)
                fam = chunk_df["attack_family"].astype(str).to_numpy()
                fine = chunk_df["fine_grained_label"].astype(str).to_numpy()
                binary = chunk_df["binary_label"].astype(str).to_numpy()
                insist(np.array_equal(fam == "Benign", binary == "Benign"),
                       split + ": family/binary mismatch")
                for family in np.unique(fam):
                    family_mask = fam == family
                    for label in np.unique(fine[family_mask]):
                        mask = family_mask & (fine == label)
                        bundles[(str(family), str(label))].append(scores[mask])
                selected += int(choose.sum())
            seen += count
            bar.update(count)
    insist(seen == EXPECTED[split], split + ": scan incomplete")
    insist(selected == (630369 if split == "train" else EXPECTED[split]),
           split + ": selected-row count invalid")
    group_scores = {key: np.concatenate(parts) for key,parts in bundles.items()}
    benign = np.concatenate([v for (family,_),v in group_scores.items()
                             if family == "Benign"])
    attack = np.concatenate([v for (family,_),v in group_scores.items()
                             if family != "Benign"])
    total = len(benign) + len(attack)
    y = np.r_[np.zeros(len(benign), dtype=np.int8), np.ones(len(attack), dtype=np.int8)]
    pred = np.r_[benign,attack]
    tp = int(np.count_nonzero(attack >= threshold))
    fp = int(np.count_nonzero(benign >= threshold))
    report_split = {"train":"internal_holdout",
                    "validation":"official_validation",
                    "test":"strict_temporal_test"}[split]
    original = m["original_metrics"][report_split]
    current = {
        "split":report_split,"rows":total,
        "attack_rows":len(attack),"benign_rows":len(benign),
        "threshold_fixed_from_internal":threshold,
        "recomputed_false_positive":fp,
        "original_false_positive":int(original["false_positive"]),
        "false_positive_delta":fp-int(original["false_positive"]),
        "recomputed_true_positive":tp,
        "original_true_positive":int(original["true_positive"]),
        "true_positive_delta":tp-int(original["true_positive"]),
        "recomputed_benign_fpr":fp/len(benign),
        "recomputed_attack_recall":tp/len(attack),
        "recomputed_auroc":float(roc_auc_score(y,pred)),
        "original_auroc":float(original["auroc"]),
        "recomputed_aupr":float(average_precision_score(y,pred)),
        "original_aupr":float(original["aupr"]),
    }
    dist = []
    for (family,label),values in sorted(group_scores.items()):
        vals = np.asarray(values)
        qs = np.quantile(vals, [0,.01,.10,.25,.50,.75,.90,.99,1])
        dist.append({
            "split":report_split, "attack_family":family,
            "fine_grained_label":label, "rows":len(vals),
            "mean_score":float(vals.mean()),
            "minimum_score":float(qs[0]),"p01":float(qs[1]),
            "p10":float(qs[2]),"p25":float(qs[3]),
            "median_score":float(qs[4]),"p75":float(qs[5]),
            "p90":float(qs[6]),"p99":float(qs[7]),
            "maximum_score":float(qs[8]),
            "original_threshold_detection_fraction":float((vals>=threshold).mean()),
        })
    group_aucs = []
    for family in sorted({k[0] for k in group_scores if k[0] != "Benign"}):
        family_scores = np.concatenate([v for (f,_),v in group_scores.items() if f==family])
        labels = np.r_[np.zeros(len(benign), dtype=np.int8),
                       np.ones(len(family_scores), dtype=np.int8)]
        vals = np.r_[benign,family_scores]
        group_aucs.append({
            "split":report_split,"attack_family":family,
            "family_rows":len(family_scores),
            "benign_rows":len(benign),
            "family_versus_benign_auroc":float(roc_auc_score(labels,vals)),
            "original_threshold_recall":float((family_scores>=threshold).mean()),
            "benign_fpr_at_fixed_threshold":float((benign>=threshold).mean()),
        })
    histogram = {}
    for category,vals in [("benign",benign),("attack",attack)]:
        hist,edges = np.histogram(np.log10(np.clip(vals,1e-12,1.0)),
                                  bins=np.linspace(-12,0,101),density=True)
        histogram[category]=(hist,edges)
    return current,dist,group_aucs,histogram


def plot(histograms,dist,threshold):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.titlesize":13,"axes.titleweight":"bold","pdf.fonttype":42})
    outputs=[]
    def save(fig,stem,description):
        png=FIGURES/(stem+".png")
        pdf=FIGURES/(stem+".pdf")
        note=CAPTIONS/(stem+".md")
        fig.savefig(png,dpi=600,bbox_inches="tight")
        fig.savefig(pdf,bbox_inches="tight")
        plt.close(fig)
        note.write_text("**Figure.** "+description+"\n",encoding="utf-8")
        outputs.extend([png,pdf,note])

    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=False,constrained_layout=True)
    for ax,category in zip(axes,["benign","attack"]):
        for split,title in [("internal_holdout","Internal holdout"),
                            ("official_validation","Validation"),
                            ("strict_temporal_test","Temporal test")]:
            heights,edges=histograms[split][category]
            ax.plot((edges[1:]+edges[:-1])/2,heights,label=title,lw=1.9)
        ax.axvline(np.log10(max(threshold,1e-12)),ls="--",
                   color="black",lw=1,label="Fixed threshold")
        ax.set(xlabel="log10 attack probability",ylabel="Probability density",
               title=category.capitalize()+" score distributions")
        ax.grid(alpha=.18)
        ax.legend(frameon=False,fontsize=8)
    save(fig,"figure_09a_c_r2_01_score_shift",
         "Comparison of fixed-model attack-probability distributions across "
         "internal holdout, official validation and temporal test. Dashed line "
         "is the unchanged internal-holdout selected operating threshold.")

    # Per-group medians with interdecile intervals; family counts and class names
    # identify genuinely unseen attack families without changing model calibration.
    subset=[r for r in dist if r["split"]=="strict_temporal_test"]
    label_rows=[]
    for family in ["Benign","Botnet","Infiltration"]:
        matching=[r for r in subset if r["attack_family"]==family]
        for row in matching:
            label_rows.append(row)
    insist(len(label_rows)>0,"No family groups for temporal plot")
    names=[x["fine_grained_label"] for x in label_rows]
    med=np.array([max(x["median_score"],1e-12) for x in label_rows])
    lo=np.array([max(x["p10"],1e-12) for x in label_rows])
    hi=np.array([max(x["p90"],1e-12) for x in label_rows])
    fig,ax=plt.subplots(figsize=(9,max(3.4,1.1*len(names))),constrained_layout=True)
    idx=np.arange(len(names))
    ax.errorbar(med,idx,xerr=[med-lo,hi-med],fmt="o",capsize=3,color="#0284c7")
    ax.axvline(threshold,ls="--",color="gray",label="Internal threshold")
    ax.set_xscale("symlog",linthresh=1e-8)
    ax.set_yticks(idx,labels=names)
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.2)
    ax.set(xlabel="Attack score (median ± P10/P90)",title="Temporal test attack-score distribution")
    ax.legend(frameon=False)
    save(fig,"figure_09a_c_r2_02_temporal_group_scores",
         "Temporal test group medians with 10th–90th percentile spans; "
         "threshold is fixed from internal training holdout only.")
    return outputs


def main():
    started=time.monotonic()
    for d in [TABLES,METRICS,MANIFESTS,FIGURES,CAPTIONS,REPORTS]:
        d.mkdir(parents=True,exist_ok=True)
    insist(torch.cuda.is_available(),"CUDA accelerator required to reproduce original MLP inference")
    for p in [R2,MLP,CHECKPOINT]:
        insist(p.is_file(),"Missing prior research artifact: "+str(p))
    cohort=json.loads(R2.read_text(encoding="utf-8"))
    previous=json.loads(MLP.read_text(encoding="utf-8"))
    insist(cohort["status"]=="verified_distinct_cohort_not_historical_matched" and
           cohort["current_retained_rows"]==6320003,"Wrong R2 cohort")
    insist(previous["status"]=="completed_distinct_r2_cohort","Wrong MLP summary")
    insist(digest(R2)==previous["r2_manifest_sha256"],"R2 manifest changed after training")
    insist(digest(CHECKPOINT)==previous["local_checkpoint_sha256"],
           "Trained checkpoint hash differs from original summary")
    insist(not previous["used_official_val_or_test_for_selection"],
           "Original experiment selection protocol mismatch")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device=torch.device("cuda:0")
    payload=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    insist(payload["seed"]==42 and payload["features"]==json.loads(
           (ROOT/"configs/datasets/cse_cic_ids2018_features_compact.json").read_text())["features"],
           "Checkpoint metadata mismatch")
    insist(payload["dataset_manifest_sha256"]==digest(R2),"Checkpoint cohort is different")
    model=Model().to(device)
    model.load_state_dict(payload["model_state"],strict=True)
    model.eval()
    threshold=float(previous["threshold"])
    clip=float(previous["fixed_neural_input_clip"][1])
    insist(threshold>0 and clip==20.0,"Unexpected fixed operating policy")
    source=previous["split_metrics"]
    offsets={"train":0,"validation":EXPECTED["train"],
             "test":EXPECTED["train"]+EXPECTED["validation"]}
    context={"features":payload["features"],
             "original_metrics":source,"offsets":offsets}
    print("="*84,flush=True)
    print("BLOCK 09A-C-R2 — READ-ONLY GPU MLP SCORE-SHIFT AUDIT",flush=True)
    print("GPU:",torch.cuda.get_device_name(device),flush=True)
    print("Fixed internal-selected threshold:",threshold,flush=True)
    print("Progress bars track every Parquet row scanned and scored.",flush=True)
    print("NO threshold tuning, NO model update, NO git push.",flush=True)
    hold=holdout_membership(DATA/"train_compact.parquet",42)
    summaries,distributions,aucrows,histograms=[],[],[],{}
    phases=[("train","internal_holdout"),("validation","official_validation"),
            ("test","strict_temporal_test")]
    for split,key in tqdm(phases,desc="Scoring splits",unit="split",dynamic_ncols=True):
        result,dist,aucs,hist=scan_split(split,context,cohort,model,device,hold,threshold,clip)
        summaries.append(result)
        distributions.extend(dist)
        aucrows.extend(aucs)
        histograms[key]=hist
        print("  %s: AUROC=%.6f, original=%.6f; FP delta=%+d, TP delta=%+d"%
              (key,result["recomputed_auroc"],result["original_auroc"],
               result["false_positive_delta"],result["true_positive_delta"]),flush=True)

    detail=TABLES/"table_09a_c_r2_01_score_distributions.csv"
    aucfile=TABLES/"table_09a_c_r2_02_family_pairwise_auroc.csv"
    reproduction=TABLES/"table_09a_c_r2_03_score_reproduction.csv"
    csvout(detail,distributions)
    csvout(aucfile,aucrows)
    csvout(reproduction,summaries)
    figures=plot(histograms,distributions,threshold)
    summary_path=METRICS/"metric_09a_c_r2_score_shift_summary.json"
    status=("reproduced_heldout_checkpoint" if all(
            x["false_positive_delta"]==0 and x["true_positive_delta"]==0
            for x in summaries)
            else "score_reproduction_differences_audited")
    report=REPORTS/"block_09a_c_r2_score_shift.md"
    temporal=next(x for x in summaries if x["split"]=="strict_temporal_test")
    report.write_text(
        "# Block 09A-C-R2 — GPU MLP fixed-score shift audit\n\n"
        "Existing trained checkpoint on the verified, distinct R2 cohort. "
        "This audit **does not update the model or choose a new threshold**.\n\n"
        f"- Reproduction status: **{status}**.\n"
        f"- Training-internal selected threshold: **{threshold:.12g}**.\n"
        f"- R2 temporal AUROC: **{temporal['recomputed_auroc']:.6f}** "
        f"(original summary {temporal['original_auroc']:.6f}).\n"
        "- Internal holdout, official validation, and temporal test all scored "
        "from the same untouched checkpoint. Individual family score distributions "
        "and pairwise family-versus-benign AUROCs are descriptive diagnostics only.\n\n"
        "## Limitations\n\n"
        "Original 08B–08E tree baselines are **not cohort-matched** to R2. "
        "The historical 08A frozen preprocessor was fitted using its entire "
        "training split, creating preprocessing-fit exposure for the internal "
        "holdout. Evaluating family AUROC on the temporal test must **not** be "
        "used to tune future architecture, hyperparameters, or thresholds on "
        "this same benchmark test partition.\n",
        encoding="utf-8")
    output_files=[detail,aucfile,reproduction,summary_path,report]+figures
    obj={
        "block":"09A-C-R2","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":status,"model_checkpoint_sha256":digest(CHECKPOINT),
        "r2_manifest_sha256":digest(R2),
        "original_mlp_summary_sha256":digest(MLP),
        "fixed_training_selected_threshold":threshold,
        "test_used_for_selection":False,
        "test_score_inspection_only":True,
        "group_auroc_interpretation":"diagnostic only, not model selection",
        "rows_audited":{r["split"]:r["rows"] for r in summaries},
        "reproduction":summaries,"pairwise_family_auroc":aucrows,
        "frozen_historical_preprocessor":True,
        "historical_cohort_matched":False,
        "elapsed_seconds":time.monotonic()-started,
    }
    jsonout(summary_path,obj)
    artifacts=[detail,aucfile,reproduction,summary_path,report]+figures
    manifest=MANIFESTS/"block_09a_c_r2_score_shift_manifest.json"
    jsonout(manifest,{
        "block":"09A-C-R2","status":status,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "artifact_sha256":{str(p.relative_to(ROOT)):digest(p) for p in artifacts},
        "checkpoint_git_ignored":True,"no_test_threshold_selection":True,
        "historical_tree_models_cohort_matched":False,
    })
    print("\n"+"="*85,flush=True)
    print("BLOCK 09A-C-R2 — SCORE-SHIFT AUDIT COMPLETE",flush=True)
    print("Reproduction status:",status,flush=True)
    for r in summaries:
        print(" %s: AUROC %.6f, AUPR %.6f, FP %d (delta %+d), TP %d (delta %+d)"%
              (r["split"],r["recomputed_auroc"],r["recomputed_aupr"],
               r["recomputed_false_positive"],r["false_positive_delta"],
               r["recomputed_true_positive"],r["true_positive_delta"]),flush=True)
    for row in aucrows:
        print(" Pairwise %s / %s vs benign: AUROC %.6f"%
              (row["split"],row["attack_family"],
               row["family_versus_benign_auroc"]),flush=True)
    print("Saved: three CSV tables, JSON summary, manifest, report, two 600-DPI PNG/PDF/caption sets.",flush=True)
    print("No training, threshold tuning or Git push.",flush=True)
    print("="*85,flush=True)


if __name__=="__main__":
    main()
