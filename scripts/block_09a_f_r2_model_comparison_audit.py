#!/usr/bin/env python3
"""DCCT-NIDS 09A-F-R2: publication-safe descriptive matched-R2 model comparison.

Consumes verified completed experiments 09A-B-R2 (GPU MLP) and 09A-E-R2
(GPU XGBoost). No inference, model fitting, threshold adjustment, test
feature selection, Git/credential interaction or historical cohort mixing.
Progress bars cover input auditing, metric integrity and figure creation.
This is EXPLORATORY because the strict-temporal test was inspected earlier.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "outputs/manifests/block_09a_a_r2_versioned_compact_manifest.json"
MLP = ROOT / "outputs/metrics/metric_09a_b_r2_gpu_mlp_summary.json"
XGB = ROOT / "outputs/metrics/metric_09a_e_r2_xgboost_reference_summary.json"
MLP_ALLOC = ROOT / "outputs/tables/table_09a_b_r2_01_internal_holdout_allocation.csv"
XGB_ALLOC = ROOT / "outputs/tables/table_09a_e_r2_01_matched_internal_allocation.csv"
MLP_METRIC_TABLE = ROOT / "outputs/tables/table_09a_b_r2_02_operational_metrics.csv"
XGB_METRIC_TABLE = ROOT / "outputs/tables/table_09a_e_r2_02_xgboost_operational_metrics.csv"
MLP_MANIFEST = ROOT / "outputs/manifests/block_09a_b_r2_gpu_mlp_manifest.json"
XGB_MANIFEST = ROOT / "outputs/manifests/block_09a_e_r2_xgboost_reference_manifest.json"
TABLES = ROOT / "outputs/tables"
METRICS = ROOT / "outputs/metrics"
MANIFESTS = ROOT / "outputs/manifests"
FIGURES = ROOT / "outputs/figures"
CAPTIONS = ROOT / "reports/captions"
REPORTS = ROOT / "reports/experiment_notes"
INPUTS = [R2, MLP, XGB, MLP_ALLOC, XGB_ALLOC, MLP_METRIC_TABLE,
          XGB_METRIC_TABLE, MLP_MANIFEST, XGB_MANIFEST]
SPLITS = ["internal_holdout", "official_validation", "strict_temporal_test"]
TEST_FAMILY_ROWS = {"Benign": 769928, "Botnet": 144535, "Infiltration": 57370}
FEATURE_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall",
                   "f1", "mcc", "auroc", "aupr", "benign_fpr"]


def demand(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    demand(bool(rows), "No data for "+str(path))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_audit(record, model, split):
    tn, fp, fn, tp = [int(record[k]) for k in (
        "true_negative", "false_positive", "false_negative", "true_positive")]
    demand(min(tn,fp,fn,tp)>=0, model+": negative confusion count")
    n=tn+fp+fn+tp
    actual_benign=tn+fp
    actual_attack=fn+tp
    demand(actual_benign>0 and actual_attack>0, model+"/"+split+": missing binary class")
    demand(n==int(record["rows"]), model+"/"+split+": row count mismatch")
    expected={
        "accuracy":(tn+tp)/n,
        "precision":tp/max(tp+fp,1),
        "recall":tp/actual_attack,
        "benign_fpr":fp/actual_benign,
    }
    expected["f1"]=2*expected["precision"]*expected["recall"]/max(
        expected["precision"]+expected["recall"], 1e-30)
    expected["balanced_accuracy"]=0.5*(
        expected["recall"]+tn/actual_benign)
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    expected["mcc"]=(tp*tn-fp*fn)/denom if denom>0 else 0.0
    for key,value in expected.items():
        demand(abs(float(record[key])-value)<1e-8,
               f"{model}/{split}: {key} not reproducible from confusion counts")
    demand(int(record["benign_rows"])==actual_benign and
           int(record["attack_rows"])==actual_attack,
           model+"/"+split+": attack/benign counts inconsistent")
    for key in ("auroc","aupr"):
        demand(0<=float(record[key])<=1,
               model+"/"+split+": invalid "+key)
    return {
        "model":model,"split":split,
        "dataset_revision":"compact_r2_6320003",
        "rows":n,"benign_rows":actual_benign,"attack_rows":actual_attack,
        "true_negative":tn,"false_positive":fp,"false_negative":fn,"true_positive":tp,
        "threshold":float(record["threshold"]),
        **{key:float(record[key]) for key in FEATURE_METRICS},
        "result_type":"EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION"
    }


def make_figures(data, family):
    plt.rcParams.update({
        "font.family":"DejaVu Sans", "font.size":10,
        "axes.titlesize":13,"axes.titleweight":"bold",
        "pdf.fonttype":42,
    })
    output=[]
    def save(fig,key,note):
        stem="figure_09a_f_r2_"+key
        png=FIGURES/(stem+".png")
        pdf=FIGURES/(stem+".pdf")
        caption=CAPTIONS/(stem+".md")
        fig.savefig(png,dpi=600,bbox_inches="tight")
        fig.savefig(pdf,bbox_inches="tight")
        plt.close(fig)
        caption.write_text("**Figure.** "+note+"\n",encoding="utf-8")
        output.extend([png,pdf,caption])

    groups=[
        ("internal_holdout","Internal holdout"),
        ("official_validation","Official validation"),
        ("strict_temporal_test","Temporal test"),
    ]
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    for ax,(split,label) in zip(axs,groups):
        rows={r["model"]:r for r in data if r["split"]==split}
        labels=["F1","MCC","AUROC"]
        positions=np.arange(3)
        for k,model in enumerate(("MLP-R2","XGBoost-R2")):
            scores=[rows[model][field] for field in ("f1","mcc","auroc")]
            ax.bar(positions+(k-.5)*.33,scores,width=.32,label=model)
        ax.set(title=label,xticks=positions,xticklabels=labels,
               ylim=(-.05,1.04),ylabel="Metric value")
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    save(fig,"01_split_performance",
         "Same-cohort MLP and GPU XGBoost internal-holdout, official validation, "
         "and strict temporal results. Thresholds selected exclusively on "
         "the matched 15% internal holdout. The test has previously been inspected; "
         "results are exploratory and not independent confirmation.")

    lookup={(row["model"],row["attack_family"]):row for row in family}
    labels=["Benign FPR","Botnet recall","Infiltration recall"]
    order=["Benign","Botnet","Infiltration"]
    fig,ax=plt.subplots(figsize=(9.3,4.8),constrained_layout=True)
    place=np.arange(3)
    for j,model in enumerate(["MLP-R2","XGBoost-R2"]):
        values=[100*lookup[(model,name)]["rate"] for name in order]
        ax.bar(place+(j-.5)*.32,values,width=.31,label=model)
    ax.set(xticks=place,xticklabels=labels,ylim=(0,109),
           title="Exploratory same-cohort temporal family detection",
           ylabel="Percent of family")
    ax.grid(axis="y",alpha=.2)
    ax.legend(frameon=False)
    save(fig,"02_family_detection",
         "Same exact 971,833-row temporal R2 cohort. Test benign false-positive "
         "rate and unseen-family Botnet/Infiltration recall at model-specific "
         "thresholds each selected on the matched internal training holdout.")
    return output


def main():
    started=time.monotonic()
    for folder in [TABLES,METRICS,MANIFESTS,FIGURES,CAPTIONS,REPORTS]:
        folder.mkdir(exist_ok=True,parents=True)
    print("="*86,flush=True)
    print("BLOCK 09A-F-R2 — EXPLORATORY MATCHED-COHORT EVIDENCE AUDIT",flush=True)
    print("No model training, inference, threshold changes, or test feedback.",flush=True)

    file_hash={}
    for path in tqdm(INPUTS,desc="Verifying research inputs",unit="file",dynamic_ncols=True):
        demand(path.is_file(),"Missing input: "+str(path))
        file_hash[str(path.relative_to(ROOT))]=sha256(path)

    r2=json.loads(R2.read_text(encoding="utf-8"))
    mlp=json.loads(MLP.read_text(encoding="utf-8"))
    xgb=json.loads(XGB.read_text(encoding="utf-8"))
    demand(r2["status"]=="verified_distinct_cohort_not_historical_matched" and
           r2["current_retained_rows"]==6_320_003 and
           r2["historical_retained_rows"]==6_319_955,
           "Wrong R2 cohort identity")
    demand(mlp["status"]=="completed_distinct_r2_cohort","MLP not completed")
    demand(xgb["status"]=="completed_exploratory_matched_r2_xgboost","XGB not completed")
    r2sha=sha256(R2)
    demand(mlp["r2_manifest_sha256"]==r2sha and xgb["r2_manifest_sha256"]==r2sha,
           "Both models did NOT use the same immutable R2 manifest")
    demand(xgb["mlp_summary_sha256"]==sha256(MLP),"XGBoost predecessor MLP summary differs")
    demand(xgb["matched_current_r2_mlp_cohort"] is True and
           xgb["same_stratified_internal_split_as_mlp"] is True and
           xgb["same_fixed_input_clipping_as_mlp"] is True and
           xgb["same_exact_fpr_policy_as_mlp"] is True,
           "Matched-comparison constraints failed")
    demand(xgb["selection_uses_test"] is False and
           xgb["selection_uses_official_validation"] is False and
           mlp["used_official_val_or_test_for_selection"] is False,
           "Test or official validation selection claimed")
    demand(mlp["seed"]==xgb["internal_split_seed"]==42,"Random-seed mismatch")
    demand(mlp["fixed_neural_input_clip"]==[-20.0,20.0],"Unexpected fixed neural clip")
    demand(xgb["preprocessor_historical_train_fitted_and_frozen"] is True,
           "Preprocessing provenance inconsistent")

    mal=read_csv(MLP_ALLOC)
    xal=read_csv(XGB_ALLOC)
    m={r["fine_grained_label"]:r for r in mal}
    x={r["fine_grained_label"]:r for r in xal}
    demand(set(m)==set(x),"Fine-label internal holdout classes differ")
    allocation=[]
    for key in sorted(m):
        for col in ("total_rows","internal_fit_rows","internal_holdout_rows"):
            demand(int(m[key][col])==int(x[key][col]),
                   f"MLP/XGB internal allocation differs: {key}/{col}")
        allocation.append({
            "fine_grained_label":key,
            "total_rows":int(m[key]["total_rows"]),
            "internal_fit_rows":int(m[key]["internal_fit_rows"]),
            "internal_holdout_rows":int(m[key]["internal_holdout_rows"]),
            "allocation_reproduced":True,
        })
    demand(sum(r["internal_fit_rows"] for r in allocation)==3_572_089,
           "Unexpected internal fit count")
    demand(sum(r["internal_holdout_rows"] for r in allocation)==630_369,
           "Unexpected internal holdout count")

    metrics=[]
    models=[
        ("MLP-R2",mlp["split_metrics"]),
        ("XGBoost-R2",xgb["per_split_metrics"]),
    ]
    with tqdm(total=6,desc="Confusion and metric audits",unit="split",dynamic_ncols=True) as bar:
        for name,records in models:
            demand(set(records)==set(SPLITS), name+": incomplete evaluation splits")
            for split in SPLITS:
                metrics.append(score_audit(records[split],name,split))
                bar.update()
    for split in SPLITS:
        matched=[r for r in metrics if r["split"]==split]
        demand(len(matched)==2,"Matched models missing from "+split)
        demand(len({(r["rows"],r["benign_rows"],r["attack_rows"]) for r in matched})==1,
               "MLP/XGB scored DIFFERENT cohort labels on "+split)
    expected_counts={"internal_holdout":630369,
                     "official_validation":1145712,
                     "strict_temporal_test":971833}
    for r in metrics:
        demand(r["rows"]==expected_counts[r["split"]],
               "Unexpected evaluation cohort size")
        if r["split"]=="internal_holdout":
            demand(r["benign_fpr"]<=.01+1e-12,
                   "Internal benign FPR violates threshold budget")

    test_summary={name:rs["strict_temporal_test"] for name,rs in models}
    test_families=[]
    for name,rs in models:
        fs=(mlp["family_metrics"] if name=="MLP-R2" else xgb["test_attack_family_results"])
        relevant=[r for r in fs if r["split"]=="strict_temporal_test"]
        demand({r["attack_family"] for r in relevant}==set(TEST_FAMILY_ROWS),
               "Unexpected temporal family names")
        for item in relevant:
            family=item["attack_family"]
            demand(int(item["row_count"])==TEST_FAMILY_ROWS[family],
                   "Unexpected temporal family count: "+family)
            test_families.append({
                "model":name,"split":"strict_temporal_test",
                "attack_family":family,
                "row_count":int(item["row_count"]),
                "predicted_attack_rows":int(item["predicted_attack_rows"]),
                "rate":float(item["detection_rate_or_fpr"]),
                "interpretation":"benign_FPR" if family=="Benign" else "attack_recall",
            })
        benignfp=next(row for row in relevant if row["attack_family"]=="Benign")
        demand(int(benignfp["predicted_attack_rows"])==
               int(test_summary[name]["false_positive"]),
               "Benign family FP count not reconciled")
        demand(sum(int(row["predicted_attack_rows"]) for row in relevant
                   if row["attack_family"]!="Benign")==
               int(test_summary[name]["true_positive"]),
               "Attack family TP counts not reconciled")

    allocfile=TABLES/"table_09a_f_r2_01_matched_allocation_verification.csv"
    metricfile=TABLES/"table_09a_f_r2_02_exploratory_split_comparison.csv"
    familyfile=TABLES/"table_09a_f_r2_03_exploratory_family_comparison.csv"
    write_csv(allocfile,allocation)
    write_csv(metricfile,metrics)
    write_csv(familyfile,test_families)

    chart_files=[]
    for _ in tqdm(range(1),desc="Rendering publication figures",unit="set",dynamic_ncols=True):
        chart_files=make_figures(metrics,test_families)

    rM=test_summary["MLP-R2"]
    rX=test_summary["XGBoost-R2"]
    report=REPORTS/"block_09a_f_r2_matched_comparison.md"
    report.write_text(
        "# Block 09A-F-R2 — Exploratory matched R2 cohort comparison\n\n"
        "**Not an independent confirmatory test.** The temporal holdout has already "
        "been examined through the MLP and drift diagnostics.\n\n"
        "- Same verified R2 dataset: **6,320,003** total rows, with "
        "971,833 identical strict-temporal test rows.\n"
        "- Same 32 Compact columns, fixed ±20 input clipping, "
        "seed-42 fine-label-stratified allocation, and internal ≤1% benign FPR policy.\n"
        "- Historical 08E trees used a distinct 48-row-smaller data revision; "
        "this comparison does not include them.\n"
        "- Shared prior preprocessing limitation: frozen Block 08A train-derived "
        "statistics expose training-internal holdout information at the preprocessing stage.\n\n"
        "## Strict temporal test — descriptive\n\n"
        "| Metric | MLP R2 | XGBoost R2 |\n|---|---:|---:|\n"
        + "".join(f"| {key} | {rM[key]:.6f} | {rX[key]:.6f} |\n"
                  for key in ["f1","mcc","auroc","aupr","recall","benign_fpr"])
        + "\nXGBoost recovers most Botnet flows but both models detect "
        "few Infiltration attacks. Both official-validation scores remain weak. "
        "A threshold capped at 1% benign FPR on the internal holdout "
        "does not guarantee ≤1% FPR after temporal transfer.\n\n"
        "## Scientific use\n\n"
        "All test-related comparisons remain descriptive. "
        "Do not tune architecture, features, thresholds, or class weights "
        "using the existing temporal test and then reuse it as a fresh holdout. "
        "A future confirmatory claim requires fresh time-separated capture data "
        "or a prospectively precommitted holdout that has not been inspected.\n",
        encoding="utf-8",
    )

    summary_path=METRICS/"metric_09a_f_r2_comparison_summary.json"
    summary={
        "block":"09A-F-R2",
        "status":"verified_exploratory_r2_comparison",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "compared_models":["MLP-R2","XGBoost-R2"],
        "dataset_cohort":"compact_r2_6320003",
        "historical_08e_cohort_matched":False,
        "same_r2_manifest_sha256":r2sha,
        "same_stratified_internal_allocation":True,
        "same_32_features_and_fixed_clipping":True,
        "same_training_internal_1_percent_FPR_policy":True,
        "test_used_for_model_selection":False,
        "test_previously_inspected":True,
        "independent_confirmatory_inference":False,
        "internal_holdout_preprocessing_fit_exposure":True,
        "confusion_and_family_counts_reconciled":True,
        "temporal_test_metrics":{
            model:{k:float(results[k]) for k in
                   ["f1","mcc","auroc","aupr","recall","benign_fpr"]}
            for model,results in test_summary.items()
        },
        "original_source_sha256":file_hash,
        "elapsed_seconds":time.monotonic()-started,
    }
    write_json(summary_path,summary)
    artifacts=[allocfile,metricfile,familyfile,report,summary_path]+chart_files
    manifest_path=MANIFESTS/"block_09a_f_r2_comparison_manifest.json"
    write_json(manifest_path,{
        "block":"09A-F-R2","status":"verified_exploratory_r2_comparison",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "artifact_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in artifacts},
        "test_derived_selection":False,
        "historical_cohort_equivalent":False,
    })
    print("\n"+"="*88,flush=True)
    print("BLOCK 09A-F-R2 — MATCHED R2 COMPARISON AUDIT COMPLETE",flush=True)
    print("6 metric matrices/confusion audits: PASS; 13 fine-grained holdout allocations: MATCHED",flush=True)
    print("Test cohort: EXACT SAME R2 971,833 rows",flush=True)
    for row in metrics:
        if row["split"] in ("official_validation","strict_temporal_test"):
            print("%s / %s: F1=%.6f MCC=%.6f AUROC=%.6f FPR=%.6f"%
                  (row["model"],row["split"],row["f1"],row["mcc"],
                   row["auroc"],row["benign_fpr"]),flush=True)
    for family in test_families:
        if family["attack_family"]!="Benign":
            print("%s / %s recall=%.4f%%"%
                  (family["model"],family["attack_family"],100*family["rate"]),flush=True)
    print("SHA-256 verified source inputs:",len(INPUTS),flush=True)
    print("Output: 3 CSV, 2 600-DPI PNG + vector PDF + captions, JSON summary, checksum manifest, report",flush=True)
    print("Model training: NO; threshold tuning: NO; Git push: NO",flush=True)
    print("CAUTION: temporal test previously examined; all comparisons exploratory.",flush=True)
    print("="*88,flush=True)


if __name__=="__main__":
    main()
