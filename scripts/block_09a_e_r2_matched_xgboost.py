#!/usr/bin/env python3
"""DCCT-NIDS 09A-E-R2: fixed-protocol XGBoost reference on SAME R2 cohort as GPU MLP.

Preregister one GPU-hist unweighted configuration and never tune from official
validation or test. Seed-42 StratifiedShuffleSplit(15%, fine-grained labels)
reproduces MLP internal allocation. Use identical verified R2 32 input columns,
the identical fixed [-20,+20] input clipping, and exact <=1% benign FPR
threshold chosen on INTERNAL train holdout only. Best boosting iteration is
chosen from this internal holdout logloss; temporal test scored once thereafter.
Prior test set has already been inspected in 09A-B/C/D: all test comparisons
are therefore EXPLORATORY, not an independent confirmatory experiment.

Outputs remain local until Kaggle Git push is separately resolved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import xgboost as xgb
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/model_ready/compact_r2_6320003"
R2_MANIFEST = ROOT / "outputs/manifests/block_09a_a_r2_versioned_compact_manifest.json"
MLP_METRICS = ROOT / "outputs/metrics/metric_09a_b_r2_gpu_mlp_summary.json"
MLP_ALLOCATION = ROOT / "outputs/tables/table_09a_b_r2_01_internal_holdout_allocation.csv"
FEATURE_FILE = ROOT / "configs/datasets/cse_cic_ids2018_features_compact.json"
OUT_TABLE = ROOT / "outputs/tables"
OUT_METRICS = ROOT / "outputs/metrics"
OUT_MANIFEST = ROOT / "outputs/manifests"
OUT_REPORT = ROOT / "reports/experiment_notes"
OUT_CAPTION = ROOT / "reports/captions"
OUT_FIGURE = ROOT / "outputs/figures"
OUT_MODEL = ROOT / "outputs/checkpoints"

SPLITS = {"train": 4_202_458, "validation": 1_145_712, "test": 971_833}
TEST_FAMILIES = {"Benign":769928,"Botnet":144535,"Infiltration":57370}
SEED = 42
CLIP = 20.0
INTERNAL_HOLDOUT_FRACTION = 0.15
FPR_BUDGET = 0.01
BATCH = 65_536
MAX_BOOST_ROUNDS = 700
EARLY_STOPPING_ROUNDS = 35
FIXED_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cuda",
    "n_estimators": MAX_BOOST_ROUNDS,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1.0,
    "max_bin": 256,
    "random_state": SEED,
    "n_jobs": 4,
    "eval_metric": "logloss",
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}
BLOCK = "09A-E-R2"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hash_file(path, bar_label=None):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        chunks = iter(lambda: handle.read(4 * 1024 * 1024), b"")
        if bar_label:
            with tqdm(total=path.stat().st_size, desc=bar_label, unit="B",
                      unit_scale=True, dynamic_ncols=True, leave=False) as progress:
                for data in chunks:
                    digest.update(data)
                    progress.update(len(data))
        else:
            for data in chunks:
                digest.update(data)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    require(bool(records), "No records for "+str(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


class BoostingProgress(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.total = total
        self.bar = None
        self.iterations = 0

    def before_training(self, model):
        self.bar = tqdm(total=self.total, desc="GPU boosting rounds",
                        unit="round", dynamic_ncols=True)
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.iterations = int(epoch) + 1
        self.bar.update(1)
        if evals_log:
            key = next(iter(evals_log))
            if "logloss" in evals_log[key]:
                self.bar.set_postfix(logloss=f"{evals_log[key]['logloss'][-1]:.5f}")
        return False

    def after_training(self, model):
        if self.bar:
            self.bar.close()
        return model


def load_split(split, features, manifests, mlp_clip):
    path = DATA / f"{split}_compact.parquet"
    require(path.is_file(), f"Missing versioned cohort: {path}")
    expected_sha = manifests[split]["sha256"]
    require(hash_file(path, "Checksum " + split) == expected_sha,
            f"{split} differs from verified Block 09A-A-R2 Parquet")
    pf = pq.ParquetFile(path)
    expected_columns = ["row_id","file_order","source_file","capture_date",
                        "fine_grained_label","attack_family","binary_label",*features]
    require(pf.schema_arrow.names == expected_columns,
            f"Incorrect canonical compact column order in {split}")
    require(pf.metadata.num_rows == SPLITS[split],
            f"Unexpected R2 split length: {split}")
    count = SPLITS[split]
    x = np.empty((count, len(features)), dtype=np.float32)
    y = np.empty(count, dtype=np.int8)
    fine = np.empty(count, dtype=object) if split == "train" else None
    family = np.empty(count, dtype=object)
    clipped = 0
    offset = sum(SPLITS[key] for key in ("train","validation","test")
                 if list(SPLITS).index(key) < list(SPLITS).index(split))
    seen = 0
    columns = ["row_id", "fine_grained_label", "attack_family", "binary_label", *features]
    with tqdm(total=count, desc="Loading R2 " + split, unit="rows",
              unit_scale=True, dynamic_ncols=True) as bar:
        for record in pf.iter_batches(batch_size=BATCH, columns=columns):
            part = record.to_pandas()
            n = len(part)
            ids = part["row_id"].to_numpy(dtype=np.int64)
            expect = np.arange(seen + offset + 1, seen + offset + n + 1, dtype=np.int64)
            require(np.array_equal(ids, expect), f"Chronological row_id drift in {split}")
            vals = part[features].to_numpy(dtype=np.float32, copy=True)
            require(np.isfinite(vals).all(), f"{split} contains NaN/infinite feature values")
            clipped += int(np.count_nonzero((vals < -CLIP) | (vals > CLIP)))
            np.clip(vals, -CLIP, CLIP, out=vals)
            x[seen:seen+n] = vals
            current_family = part["attack_family"].astype(str).to_numpy()
            current_binary = part["binary_label"].astype(str).to_numpy()
            require(np.all((current_family == "Benign") == (current_binary == "Benign")),
                    f"Inconsistent attack family versus binary labels in {split}")
            require(set(np.unique(current_binary)) <= {"Benign","Attack"},
                    f"Nonbinary training target in {split}")
            y[seen:seen+n] = (current_binary == "Attack").astype(np.int8)
            family[seen:seen+n] = current_family
            if fine is not None:
                fine[seen:seen+n] = part["fine_grained_label"].astype(str).to_numpy()
            seen += n
            bar.update(n)
    require(seen == count, "Incomplete split load: "+split)
    require(int(clipped) == int(mlp_clip[split]),
            f"Input clipping count differs from MLP for {split}: {clipped} != {mlp_clip[split]}")
    print(f"  {split}: rows={seen:,}; clipped feature cells={clipped:,} MATCH MLP", flush=True)
    return x,y,fine,family,clipped


def score_gpu(model, x, split):
    result = np.empty(len(x), dtype=np.float64)
    with tqdm(total=len(x), desc="GPU XGBoost inference / " + split,
              unit="rows",unit_scale=True,dynamic_ncols=True) as bar:
        for start in range(0,len(x),BATCH):
            part=x[start:start+BATCH]
            # sklearn predict_proba accepts NumPy inputs with GPU-hist models;
            # this may print a one-time device-copy fallback warning in XGBoost.
            scores=model.predict_proba(part)[:,1]
            result[start:start+len(scores)]=scores.astype(np.float64)
            bar.update(len(part))
    require(np.isfinite(result).all(), "Non-finite probabilities: "+split)
    require(np.all((result >= 0) & (result <= 1)), "Invalid probability bounds")
    return result


def select_internal_fpr(scores, truth):
    benign=np.sort(scores[truth==0])[::-1]
    cap=int(math.floor(FPR_BUDGET * len(benign)))
    require(0 <= cap < len(benign), "Invalid internal benign FPR budget")
    # Threshold in float64 and comparison in float64: tie-safe.
    threshold=float(np.nextafter(float(benign[cap]), np.inf))
    fp=int(np.count_nonzero(benign >= threshold))
    require(fp <= cap, "Selected threshold violates strict 1% benign FPR")
    return threshold,cap,fp,len(benign)


def metric_row(y, scores, threshold, split):
    target=y.astype(np.int8)
    pred=(scores>=threshold).astype(np.int8)
    tn=int(np.count_nonzero((target==0) & (pred==0)))
    fp=int(np.count_nonzero((target==0) & (pred==1)))
    fn=int(np.count_nonzero((target==1) & (pred==0)))
    tp=int(np.count_nonzero((target==1) & (pred==1)))
    require(tn+fp+fn+tp==len(target),"Incorrect confusion totals for "+split)
    benign,attack=tn+fp,fn+tp
    require(benign>0 and attack>0,"Both labels required in "+split)
    return {
        "split":split, "model":"XGBoost-R2-unweighted",
        "dataset_revision":"compact_r2_6320003",
        "threshold_policy":"internal_benign_fpr_at_most_1_percent",
        "threshold":threshold, "rows":len(target),
        "benign_rows":benign,"attack_rows":attack,
        "accuracy":float(accuracy_score(target,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(target,pred)),
        "precision":float(precision_score(target,pred,zero_division=0)),
        "recall":float(recall_score(target,pred,zero_division=0)),
        "f1":float(f1_score(target,pred,zero_division=0)),
        "mcc":float(matthews_corrcoef(target,pred)),
        "auroc":float(roc_auc_score(target,scores)),
        "aupr":float(average_precision_score(target,scores)),
        "benign_fpr":fp/benign,"attack_fnr":fn/attack,
        "true_negative":tn,"false_positive":fp,"false_negative":fn,
        "true_positive":tp,
    }


def family_rows(y,scores,groups,threshold,split):
    rows=[]
    for name in sorted(set(groups)):
        choose=groups==name
        n=int(np.count_nonzero(choose))
        attacks=int(np.count_nonzero(y[choose]))
        flagged=int(np.count_nonzero(scores[choose] >= threshold))
        require((name=="Benign" and attacks==0) or
                (name!="Benign" and attacks==n),
                "Mixed binary targets within attack family: "+name)
        rows.append({
            "split":split,"attack_family":name,"row_count":n,
            "true_attack_rows":attacks,"predicted_attack_rows":flagged,
            "detection_rate_or_fpr":flagged/n,
            "interpretation":"false_positive_rate" if name=="Benign" else "attack_detection_rate",
            "mean_attack_probability":float(np.mean(scores[choose])),
            "median_attack_probability":float(np.median(scores[choose])),
        })
    return rows


def plot_results(history, metrics, families, test_y, test_scores):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":13,"axes.titleweight":"bold",
                         "pdf.fonttype":42})
    outputs=[]
    def save(fig,name,caption):
        base="figure_09a_e_r2_"+name
        png,pdf,note=(OUT_FIGURE/(base+".png"),OUT_FIGURE/(base+".pdf"),
                      OUT_CAPTION/(base+".md"))
        fig.savefig(png,dpi=600,bbox_inches="tight")
        fig.savefig(pdf,bbox_inches="tight")
        plt.close(fig)
        note.write_text("**Figure.** "+caption+"\n",encoding="utf-8")
        outputs.extend([png,pdf,note])

    fig,ax=plt.subplots(figsize=(8,4.3),constrained_layout=True)
    ax.plot(range(1,len(history)+1),history,color="#0f766e",lw=1.7)
    ax.set(xlabel="Boosting iteration",ylabel="Internal holdout log loss",
           title="R2 XGBoost — early-stopping curve")
    ax.grid(alpha=.2)
    save(fig,"01_internal_logloss",
         "XGBoost internal-holdout logloss; boosting epoch chosen by early stopping on "
         "the same 15% stratified R2 training holdout used in 09A-B-R2.")

    fpr,tpr,_=roc_curve(test_y,test_scores)
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    ax.plot(fpr,tpr,color="#0369a1",lw=2,
            label="XGBoost R2 AUROC=%.4f"%roc_auc_score(test_y,test_scores))
    ax.plot([0,1],[0,1],ls="--",color="gray")
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel="False positive rate",
           ylabel="True positive rate",title="Exploratory R2 strict-temporal ROC")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig,"02_temporal_roc",
         "Exploratory strict-temporal ROC for R2 XGBoost. The test partition was "
         "previously inspected in MLP analysis; this is not independent confirmation.")

    p,r,_=precision_recall_curve(test_y,test_scores)
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    ax.plot(r,p,color="#d97706",lw=2,
            label="XGBoost R2 AUPR=%.4f"%average_precision_score(test_y,test_scores))
    ax.axhline(float(np.mean(test_y)),color="gray",ls="--",label="Attack prevalence")
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel="Recall",
           ylabel="Precision",title="Exploratory R2 strict-temporal PR")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig,"03_temporal_pr",
         "Exploratory precision-recall curve of versioned R2 XGBoost.")

    family={row["attack_family"]:row for row in families}
    require(set(family)==set(TEST_FAMILIES),"Unexpected R2 test family coverage")
    order=["Benign","Botnet","Infiltration"]
    values=[family[name]["detection_rate_or_fpr"] for name in order]
    fig,ax=plt.subplots(figsize=(8.3,4.7),constrained_layout=True)
    bars=ax.bar(order,values,width=.52,color=["#64748b","#0284c7","#f97316"])
    for bar,val in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,val+.028,
                "%.2f%%"%(val*100),ha="center",fontweight="bold")
    ax.set(ylim=(0,1.12),ylabel="Fraction predicted attack",
           title="R2 temporal benign FPR and unseen-family recalls")
    ax.grid(axis="y",alpha=.2)
    save(fig,"04_family_temporal",
         "Benign false alarm rate and detection recall for unseen Botnet/Infiltration "
         "at XGBoost's independently internal-holdout-selected operating threshold.")
    return outputs


def main():
    started=time.monotonic()
    for folder in [OUT_TABLE,OUT_METRICS,OUT_MANIFEST,OUT_REPORT,
                   OUT_CAPTION,OUT_FIGURE,OUT_MODEL]:
        folder.mkdir(parents=True,exist_ok=True)
    require(torch.cuda.is_available(),"Kaggle CUDA GPU required for fixed GPU XGBoost protocol")
    for path in [R2_MANIFEST,MLP_METRICS,MLP_ALLOCATION,FEATURE_FILE]:
        require(path.is_file(),"Missing prerequisite: "+str(path))
    r2=json.loads(R2_MANIFEST.read_text(encoding="utf-8"))
    mlp=json.loads(MLP_METRICS.read_text(encoding="utf-8"))
    require(r2["status"]=="verified_distinct_cohort_not_historical_matched" and
            r2["current_retained_rows"]==6_320_003,
            "Expected verified, DISTINCT 6,320,003-row R2 cohort")
    require(r2["split_row_counts"]==SPLITS,"R2 split sizes have changed")
    require(r2["identical_cohort_to_historical_tree_baselines"] is False,
            "Historical 08E comparison incorrectly marked equivalent")
    require(mlp["status"]=="completed_distinct_r2_cohort",
            "R2 MLP baseline not completed")
    require(mlp["r2_manifest_sha256"]==hash_file(R2_MANIFEST),
            "Original R2 cohort manifest changed since MLP experiment")
    require(mlp["seed"]==SEED and mlp["fixed_neural_input_clip"]==[-CLIP,CLIP],
            "Cannot match MLP seed/input policy")
    require(mlp["trained_on_r2_internal_fit_only"] is True and
            mlp["used_official_val_or_test_for_selection"] is False,
            "MLP baseline selection protocol mismatch")
    features=json.loads(FEATURE_FILE.read_text(encoding="utf-8"))["features"]
    require(len(features)==32 and len(set(features))==32,
            "Exactly 32 frozen compact feature names required")
    checksums={v["split"]:v for v in r2["outputs"]}
    require(set(checksums)==set(SPLITS),"Missing R2 split checksums")
    print("="*88,flush=True)
    print("BLOCK 09A-E-R2 — EXPLORATORY MATCHED-COHORT GPU XGBOOST REFERENCE",flush=True)
    print("="*88,flush=True)
    print("GPU:",torch.cuda.get_device_name(0), "| XGBoost:",xgb.__version__,flush=True)
    print("Same cohort, seed, train/holdout allocation, 32 features, clipping and FPR policy as MLP.",flush=True)
    print("ONE prespecified unweighted XGBoost config; NO test-derived tuning.",flush=True)
    print("tqdm bars: checksums, Parquet loading, GPU rounds, inference.",flush=True)

    train_x,train_y,train_fine,train_families,train_clipped=load_split(
        "train",features,checksums,mlp["clipped_feature_cells"])
    del train_families
    splitter=StratifiedShuffleSplit(n_splits=1,test_size=INTERNAL_HOLDOUT_FRACTION,
                                    random_state=SEED)
    fit_idx,hold_idx=next(splitter.split(
        np.zeros(len(train_fine),dtype=np.int8),train_fine))
    require(len(fit_idx)==3_572_089 and len(hold_idx)==630_369,
            "Internal R2 split allocation differs from MLP")
    old_allocation=pd.read_csv(MLP_ALLOCATION)
    allocation=[]
    for label in sorted(set(train_fine)):
        orig=int(np.count_nonzero(train_fine==label))
        fit=int(np.count_nonzero(train_fine[fit_idx]==label))
        hold=int(np.count_nonzero(train_fine[hold_idx]==label))
        require(orig==fit+hold and fit>0 and hold>0,
                "Internal stratification invalid for "+label)
        old=old_allocation.loc[old_allocation["fine_grained_label"]==label]
        require(len(old)==1 and int(old.iloc[0]["total_rows"])==orig and
                int(old.iloc[0]["internal_fit_rows"])==fit and
                int(old.iloc[0]["internal_holdout_rows"])==hold,
                "Internal holdout fine-grained allocation does not match GPU MLP for "+label)
        allocation.append({"fine_grained_label":label,
                           "total_rows":orig,"internal_fit_rows":fit,
                           "internal_holdout_rows":hold})
    require(len(allocation)==len(old_allocation),"Fine-label allocation list differs")
    alloc_path=OUT_TABLE/"table_09a_e_r2_01_matched_internal_allocation.csv"
    write_csv(alloc_path,allocation)
    print(f"Verified identical internal allocation: fit {len(fit_idx):,}, "
          f"holdout {len(hold_idx):,}",flush=True)
    del train_fine

    # Copy ONLY fitting and internal holdout rows into scikit-learn/xgboost arrays.
    fit_x=np.ascontiguousarray(train_x[fit_idx])
    fit_y=train_y[fit_idx]
    hold_x=np.ascontiguousarray(train_x[hold_idx])
    hold_y=train_y[hold_idx]
    print("Training a SINGLE GPU-hist XGBoost (unweighted, fixed config)...",flush=True)
    progress=BoostingProgress(MAX_BOOST_ROUNDS)
    model=xgb.XGBClassifier(**FIXED_PARAMS,callbacks=[progress])
    fit_started=time.monotonic()
    model.fit(fit_x,fit_y,eval_set=[(hold_x,hold_y)],verbose=False)
    fit_elapsed=time.monotonic()-fit_started
    require(progress.iterations>0,"GPU boosting did not complete any iteration")
    best_iteration=int(model.best_iteration)
    require(0<=best_iteration<progress.iterations,"Invalid internal-selected boosting round")
    print(f"Best boosting iteration: {best_iteration}; rounds run: "
          f"{progress.iterations}; seconds: {fit_elapsed:.2f}",flush=True)

    checkpoint=OUT_MODEL/"block_09a_e_r2_xgb_reference_seed42.ubj"
    model.save_model(str(checkpoint))
    checkpoint_sha=hash_file(checkpoint)
    # A separate json config + manifest captures all model/metric provenance.
    print("Scoring the original fixed INTERNAL holdout...",flush=True)
    hold_score=score_gpu(model,hold_x,"internal_holdout")
    threshold,allow_fp,actual_fp,benign_count=select_internal_fpr(hold_score,hold_y)
    scores=[metric_row(hold_y,hold_score,threshold,"internal_holdout")]
    require(scores[0]["false_positive"]==actual_fp and
            actual_fp<=allow_fp,"Internal threshold integrity failure")
    print(f"Exact internal FPR threshold: {threshold:.12g}; "
          f"{actual_fp:,}/{benign_count:,} benign false positives; "
          f"FPR {actual_fp/benign_count:.6f}",flush=True)
    del fit_x,fit_y,hold_x,hold_y,train_x,train_y,fit_idx,hold_idx,hold_score
    families=[]
    test_true=None
    test_probs=None
    test_groups=None
    for name in ("validation","test"):
        print("\nScoring full, evaluation-only "+name+" split...",flush=True)
        x,y,_,group,_=load_split(name,features,checksums,mlp["clipped_feature_cells"])
        probabilities=score_gpu(model,x,name)
        split="official_validation" if name=="validation" else "strict_temporal_test"
        row=metric_row(y,probabilities,threshold,split)
        scores.append(row)
        group_result=family_rows(y,probabilities,group,threshold,split)
        families.extend(group_result)
        print(f"  {split}: F1={row['f1']:.6f}, MCC={row['mcc']:.6f}, "
              f"AUROC={row['auroc']:.6f}, AUPR={row['aupr']:.6f}, "
              f"benign FPR={row['benign_fpr']:.6f}",flush=True)
        if name=="test":
            actual_families={v["attack_family"]:v["row_count"] for v in group_result}
            require(actual_families==TEST_FAMILIES,
                    "Unexpected temporal test family composition")
            test_true,test_probs,test_groups=y,probabilities,group_result
        del x,group
    require(test_true is not None,"Temporal test not scored")
    require(len(scores)==3 and scores[-1]["rows"]==SPLITS["test"],
            "Unexpected evaluation metric count")

    hist=model.evals_result()["validation_0"]["logloss"]
    history=[{"boosting_round":i+1,"internal_holdout_logloss":float(value)}
             for i,value in enumerate(hist)]
    history_path=OUT_TABLE/"table_09a_e_r2_04_boosting_history.csv"
    metrics_path=OUT_TABLE/"table_09a_e_r2_02_xgboost_operational_metrics.csv"
    family_path=OUT_TABLE/"table_09a_e_r2_03_family_detection.csv"
    threshold_path=OUT_TABLE/"table_09a_e_r2_05_exact_internal_threshold.csv"
    write_csv(history_path,history)
    write_csv(metrics_path,scores)
    write_csv(family_path,families)
    write_csv(threshold_path,[{
        "policy":"at_most_1_percent_internal_benign_fpr",
        "threshold":threshold,"internal_benign_count":benign_count,
        "permitted_false_positives":allow_fp,
        "realized_false_positives":actual_fp,
        "exact_internal_benign_fpr":actual_fp/benign_count,
        "selected_using_official_validation":False,
        "selected_using_official_test":False,
    }])
    figures=plot_results(hist,scores,test_groups,test_true,test_probs)
    summary_path=OUT_METRICS/"metric_09a_e_r2_xgboost_reference_summary.json"
    compare_original=mlp["split_metrics"]["strict_temporal_test"]
    xgb_test=scores[-1]
    comparison_path=OUT_TABLE/"table_09a_e_r2_06_exploratory_r2_mlp_xgb_comparison.csv"
    fields=["f1","mcc","auroc","aupr","balanced_accuracy","recall","benign_fpr"]
    compare=[]
    for name,record in [("MLP-R2",compare_original),("XGBoost-R2-unweighted",xgb_test)]:
        compare.append({
            "model":name,"cohort":"compact_r2_6320003","test_rows":int(record["rows"]),
            "train_internal_fit_rows":3_572_089,
            "train_internal_holdout_rows":630_369,
            "fixed_input_clip":20,
            "threshold_selection":"internal_benign_fpr_at_most_1_percent",
            "test_inspected_previously":True,
            "not_independent_confirmatory_result":True,
            **{field:float(record[field]) for field in fields},
            "false_positive":int(record["false_positive"]),
            "true_positive":int(record["true_positive"]),
        })
    write_csv(comparison_path,compare)
    summary={
        "block":BLOCK,"status":"completed_exploratory_matched_r2_xgboost",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "dataset_cohort":"compact_r2_6320003",
        "historical_08e_tree_cohort_equivalent":False,
        "matched_current_r2_mlp_cohort":True,
        "internal_split_seed":SEED,
        "same_stratified_internal_split_as_mlp":True,
        "same_32_input_features":True,
        "same_fixed_input_clipping_as_mlp":True,
        "same_exact_fpr_policy_as_mlp":True,
        "not_architecture_only_ablation":True,
        "train_validation_test_rows":SPLITS,
        "model":"XGBoost-GPU-hist-single-config-unweighted",
        "params":FIXED_PARAMS,
        "versions":{"xgboost":xgb.__version__,"numpy":np.__version__,
                    "torch":torch.__version__,"python":platform.python_version()},
        "gpu":torch.cuda.get_device_name(0),
        "r2_manifest_sha256":hash_file(R2_MANIFEST),
        "mlp_summary_sha256":hash_file(MLP_METRICS),
        "fine_label_holdout_allocation_path":str(alloc_path.relative_to(ROOT)),
        "checkpoint":str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256":checkpoint_sha,
        "trained_iterations":progress.iterations,
        "best_iteration_zero_indexed":best_iteration,
        "fitting_seconds":fit_elapsed,
        "threshold":threshold,
        "max_internal_benign_fp":allow_fp,
        "actual_internal_benign_fp":actual_fp,
        "selection_uses_test":False,
        "selection_uses_official_validation":False,
        "preprocessor_historical_train_fitted_and_frozen":True,
        "internal_holdout_preprocessing_fit_exposure":True,
        "historical_mlp_holdout_preprocessing_exposure_is_same":True,
        "test_was_previous_posthoc_inspected":True,
        "test_results_are_exploratory":True,
        "per_split_metrics":{v["split"]:v for v in scores},
        "test_attack_family_results":[v for v in families
                                       if v["split"]=="strict_temporal_test"],
        "comparability_caveat":"Same dataset, features, clipping, stratified internal split, "
        "and internal FPR threshold policy; different algorithms/fit objectives. "
        "R2 test has already been extensively examined; no independent confirmatory "
        "test inference, no statistical superiority claim, and not historical 08E cohort.",
    }
    write_json(summary_path,summary)
    report=OUT_REPORT/"block_09a_e_r2_xgboost_matched_cohort.md"
    report.write_text(
        "# Block 09A-E-R2 — Fixed-protocol GPU XGBoost on R2 cohort\n\n"
        f"- R2 flows **6,320,003**. Seed **{SEED}**. Train fit **3,572,089**, "
        "internal holdout **630,369**.\n"
        "- Input features and fixed ±20 clipping match R2 MLP.\n"
        "- Unweighted single prespecified XGBoost histogram GPU model, "
        "early stopping by internal holdout logloss, no official validation/test selection.\n"
        f"- Best iteration: **{best_iteration}** (0-indexed), "
        f"fit time **{fit_elapsed:.2f} seconds**.\n"
        f"- Training-internal selected exact FPR threshold **{threshold:.12g}**; "
        f"{actual_fp}/{benign_count} benign FPs, FPR {actual_fp/benign_count:.6f}.\n\n"
        "## Exploratory temporal-test performance\n\n"
        "| Metric | Value |\n|---|---:|\n"
        + "".join(f"| {key} | {xgb_test[key]:.6f} |\n" for key in fields)
        + "\n**Important:** This test set has already been used in post-hoc "
        "score and feature-distribution audits in the MLP R2 investigation. "
        "Current comparisons are exploratory, not independent confirmation or "
        "a causal architecture-only ablation. Historical Block 08E models "
        "used a different 48-flow-smaller cohort; do not conflate them. "
        "Historical train-fitted preprocessing is reused unchanged and has "
        "internal holdout preprocessing-fit exposure. This model was NOT "
        "tuned using temporal-test metrics.\n",
        encoding="utf-8"
    )
    artifacts=[alloc_path,history_path,metrics_path,family_path,threshold_path,
               comparison_path,summary_path,report,*figures]
    manifest_path=OUT_MANIFEST/"block_09a_e_r2_xgboost_reference_manifest.json"
    write_json(manifest_path,{
        "block":BLOCK,"status":"completed_exploratory_matched_r2_xgboost",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256":checkpoint_sha,"checkpoint_git_ignored":True,
        "historical_08e_cohort_equivalent":False,
        "matched_r2_mlp_cohort":True,
        "model_hyperparameter_selection_from_test":False,
        "test_inspected_before_reference":True,
        "artifacts_sha256":{str(p.relative_to(ROOT)):hash_file(p) for p in artifacts},
        "elapsed_seconds":time.monotonic()-started,
    })
    print("\n"+"="*90,flush=True)
    print("BLOCK 09A-E-R2 — GPU XGBOOST MATCHED-COHORT REFERENCE COMPLETE",flush=True)
    print("="*90,flush=True)
    print("Dataset cohort: compact_r2_6320003 (same as 09A-B-R2 MLP)",flush=True)
    print(f"Best boosting iteration: {best_iteration}; threshold: {threshold:.12g}",flush=True)
    print(f"Internal benign FPR: {actual_fp/benign_count:.6f}",flush=True)
    print(f"Temporal F1: {xgb_test['f1']:.6f}; MCC: {xgb_test['mcc']:.6f}; "
          f"AUROC: {xgb_test['auroc']:.6f}; AUPR: {xgb_test['aupr']:.6f}; "
          f"benign FPR: {xgb_test['benign_fpr']:.6f}",flush=True)
    for item in test_groups:
        print(" %s: %.3f%% %s"%(item["attack_family"],
              100*item["detection_rate_or_fpr"],
              "FPR" if item["attack_family"]=="Benign" else "recall"),flush=True)
    print("Rows compared on same cohort: 971,833; results exploratory.",flush=True)
    print("Artifacts: 6 CSV, 4 PNG/PDF/caption triplets, JSON summary/manifest and report.",flush=True)
    print("Checkpoint Git-ignored. Git push not attempted.",flush=True)
    print("="*90,flush=True)


if __name__ == "__main__":
    main()
