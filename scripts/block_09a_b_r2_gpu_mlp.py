#!/usr/bin/env python3
"""DCCT-NIDS Block 09A-B-R2: GPU MLP strict-temporal binary baseline.

R2-only cohort. Historical Block 08A frozen train-derived preprocessing;
NOT a paired comparison with historical Blocks 08B-08E. Select best
checkpoint and <=1% benign-FPR threshold on a 15% fine-label-stratified
internal holdout of the R2 train partition only. No network/Git/token access.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/model_ready/compact_r2_6320003"
R2_MANIFEST = ROOT / "outputs/manifests/block_09a_a_r2_versioned_compact_manifest.json"
FEATURE_FILE = ROOT / "configs/datasets/cse_cic_ids2018_features_compact.json"
SPLIT_COUNTS = {"train":4202458, "validation":1145712, "test":971833}
EXPECTED_TEST_FAMILY = {"Benign":769928,"Botnet":144535,"Infiltration":57370}
SEED, EPOCHS, PATIENCE, BATCH, INFER = 42, 12, 3, 8192, 65536
CLIP, FPR_LIMIT = 20.0, 0.01
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures"
METRICS = ROOT / "outputs/metrics"
MANIFESTS = ROOT / "outputs/manifests"
REPORTS = ROOT / "reports/experiment_notes"
CAPTIONS = ROOT / "reports/captions"
CHECKPOINTS = ROOT / "outputs/checkpoints"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, doc):
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    check(bool(rows), "No rows for "+str(path))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BinaryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.10),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.layers(x).squeeze(-1)


def load_split(split, columns, expected_sha):
    path = DATA / (split + "_compact.parquet")
    check(path.is_file(), "Missing R2 Parquet: "+str(path))
    check(digest(path) == expected_sha, "R2 Parquet differs from verified manifest: "+split)
    pqf = pq.ParquetFile(path)
    required_columns = ["row_id","file_order","source_file","capture_date",
                        "fine_grained_label","attack_family","binary_label"] + columns
    check(pqf.schema_arrow.names == required_columns, split+": wrong field ordering")
    check(pqf.metadata.num_rows == SPLIT_COUNTS[split], split+": row count mismatch")
    frame = pd.read_parquet(path, columns=[
        "row_id","fine_grained_label","attack_family","binary_label"] + columns)
    ids = frame["row_id"].to_numpy(dtype=np.int64, copy=True)
    check(np.all(np.diff(ids)==1), split+": row IDs are not ordered and contiguous")
    x = frame[columns].to_numpy(dtype=np.float32, copy=True)
    check(x.shape == (SPLIT_COUNTS[split], 32) and np.isfinite(x).all(),
          split+": feature matrix invalid")
    changed_cells = int(np.count_nonzero((x < -CLIP) | (x > CLIP)))
    np.clip(x, -CLIP, CLIP, out=x)
    binary = frame["binary_label"].astype(str).to_numpy(copy=True)
    check(set(np.unique(binary)) == {"Benign","Attack"}, split+": bad binary labels")
    y = (binary=="Attack").astype(np.float32)
    fine = frame["fine_grained_label"].astype(str).to_numpy(copy=True)
    family = frame["attack_family"].astype(str).to_numpy(copy=True)
    check(np.all((family=="Benign")==(y==0)), split+": family/binary mismatch")
    del frame
    return x,y,fine,family,changed_cells


@torch.inference_mode()
def score_model(model, x, device):
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), INFER):
        batch = torch.from_numpy(x[start:start+INFER]).to(device)
        values = model(batch).sigmoid().cpu().numpy()
        out[start:start+len(values)] = values.astype(np.float64)
    check(np.isfinite(out).all(), "Non-finite probability prediction")
    return out


@torch.inference_mode()
def val_loss(model, x_gpu, y_gpu):
    model.eval()
    acc = 0.0
    for start in range(0,len(x_gpu),INFER):
        logits = model(x_gpu[start:start+INFER])
        loss = F.binary_cross_entropy_with_logits(logits, y_gpu[start:start+INFER], reduction="sum")
        acc += float(loss.item())
    return acc/len(x_gpu)


def select_threshold(scores, y):
    # Use float64 nextafter to exclude benign boundary ties. Never exceed budget.
    benign = np.sort(scores[y==0])[::-1]
    cap = int(math.floor(FPR_LIMIT*len(benign)))
    check(0 <= cap < len(benign), "Invalid benign FPR budget")
    threshold = float(np.nextafter(float(benign[cap]), np.inf))
    fp = int(np.count_nonzero(benign>=threshold))
    check(fp<=cap, "Internal benign FPR budget exceeded")
    return threshold,cap,fp,len(benign)


def operational(y,scores,t,split):
    labels = y.astype(np.int8)
    pred = (scores>=t).astype(np.int8)
    tn = int(np.count_nonzero((labels==0)&(pred==0)))
    fp = int(np.count_nonzero((labels==0)&(pred==1)))
    fn = int(np.count_nonzero((labels==1)&(pred==0)))
    tp = int(np.count_nonzero((labels==1)&(pred==1)))
    check(tn+fp+fn+tp==len(y), "Confusion matrix mismatch: "+split)
    nbenign,nattack=tn+fp,fn+tp
    check(nbenign>0 and nattack>0, "Both labels required")
    return {
        "cohort":"compact_r2_6320003","model":"MLP-128-64",
        "threshold_policy":"internal_fpr_at_most_1_percent",
        "split":split,"threshold":t,"rows":len(y),
        "benign_rows":nbenign,"attack_rows":nattack,
        "accuracy":float(accuracy_score(labels,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(labels,pred)),
        "precision":float(precision_score(labels,pred,zero_division=0)),
        "recall":float(recall_score(labels,pred,zero_division=0)),
        "f1":float(f1_score(labels,pred,zero_division=0)),
        "mcc":float(matthews_corrcoef(labels,pred)),
        "auroc":float(roc_auc_score(labels,scores)),
        "aupr":float(average_precision_score(labels,scores)),
        "benign_fpr":fp/nbenign,"attack_fnr":fn/nattack,
        "true_negative":tn,"false_positive":fp,
        "false_negative":fn,"true_positive":tp,
    }


def family_table(y,scores,families,t,split):
    rows=[]
    for name in sorted(np.unique(families)):
        mask = families==name
        total=int(mask.sum())
        pos=int(np.count_nonzero(scores[mask]>=t))
        real=int(np.count_nonzero(y[mask]==1))
        check((name=="Benign" and real==0) or (name!="Benign" and real==total),
              "Family-label inconsistency: "+name)
        rows.append({
            "split":split,"attack_family":name,"row_count":total,
            "true_attack_rows":real,"predicted_attack_rows":pos,
            "detection_rate_or_fpr":pos/total,
            "interpretation":"false_positive_rate" if name=="Benign" else "attack_detection_rate",
            "mean_attack_probability":float(np.mean(scores[mask])),
            "median_attack_probability":float(np.median(scores[mask])),
        })
    return rows


def create_figures(history,y,prob,tfamily):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":13,"axes.titleweight":"bold","pdf.fonttype":42})
    figs=[]
    def save(fig,key,caption):
        stem = "figure_09a_b_r2_"+key
        png=FIGURES/(stem+".png")
        pdf=FIGURES/(stem+".pdf")
        note=CAPTIONS/(stem+".md")
        fig.savefig(png,dpi=600,bbox_inches="tight")
        fig.savefig(pdf,bbox_inches="tight")
        plt.close(fig)
        note.write_text("**Figure.** "+caption+"\n",encoding="utf-8")
        figs.extend([png,pdf,note])

    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ax.plot([h["epoch"] for h in history],[h["fit_bce"] for h in history],
            marker="o",lw=1.7,label="Internal fit")
    ax.plot([h["epoch"] for h in history],[h["holdout_bce"] for h in history],
            marker="s",lw=1.7,label="Internal holdout")
    ax.set(xlabel="Epoch",ylabel="Binary cross-entropy",title="MLP training on R2 cohort")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig,"01_training_curves",
         "Training and 15% fine-label-stratified internal-holdout BCE; epoch selected exclusively by internal BCE.")

    fpr,tpr,_=roc_curve(y.astype(np.int8),prob)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    ax.plot(fpr,tpr,lw=2,label="MLP AUROC = %.4f"%roc_auc_score(y,prob))
    ax.plot([0,1],[0,1],color="gray",ls="--",label="Chance")
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel="False positive rate",
           ylabel="True positive rate",title="Strict temporal ROC — R2 cohort")
    ax.grid(alpha=.2)
    ax.legend(frameon=False,loc="lower right")
    save(fig,"02_temporal_roc",
         "Strict temporal R2 test ROC; all model selection occurred on the internal training holdout.")

    precision,recall,_=precision_recall_curve(y.astype(np.int8),prob)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    ax.plot(recall,precision,lw=2,
            label="MLP AUPR = %.4f"%average_precision_score(y,prob))
    ax.axhline(float(np.mean(y)),ls="--",color="gray",label="Attack prevalence")
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel="Recall",
           ylabel="Precision",title="Strict temporal precision–recall — R2")
    ax.grid(alpha=.2)
    ax.legend(frameon=False,loc="lower left")
    save(fig,"03_temporal_pr",
         "Strict temporal R2 test precision–recall curve; baseline denotes attack prevalence.")

    by={r["attack_family"]:r for r in tfamily}
    check(set(by)==set(EXPECTED_TEST_FAMILY),"Unexpected test family coverage")
    names=["Benign","Botnet","Infiltration"]
    vals=[by[n]["detection_rate_or_fpr"] for n in names]
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    bars=ax.bar(names,vals,width=.58,color=["#64748b","#0284c7","#f97316"])
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v+.025,"%.2f%%"%(100*v),
                ha="center",fontsize=10,fontweight="bold")
    ax.set(ylim=(0,1.13),ylabel="Fraction predicted attack",
           title="Temporal shift: benign FPR and unseen-family detection")
    ax.grid(axis="y",alpha=.2)
    save(fig,"04_family_detection",
         "Benign false positive rate and unseen-family recalls in the versioned R2 test.")
    return figs


def main():
    started=time.monotonic()
    for folder in [TABLES,FIGURES,METRICS,MANIFESTS,REPORTS,CAPTIONS,CHECKPOINTS]:
        folder.mkdir(parents=True,exist_ok=True)
    check(torch.cuda.is_available(),"Kaggle CUDA GPU is required; enable GPU")
    check(R2_MANIFEST.is_file() and FEATURE_FILE.is_file(),
          "Versioned R2 manifest or Compact feature specification is missing")
    r2=json.loads(R2_MANIFEST.read_text(encoding="utf-8"))
    check(r2["status"]=="verified_distinct_cohort_not_historical_matched" and
          r2["current_retained_rows"]==6320003 and r2["historical_retained_rows"]==6319955,
          "Dataset revision identity invalid")
    check(r2["identical_cohort_to_historical_tree_baselines"] is False,
          "Incorrect matched-cohort claim")
    check(r2["split_row_counts"]==SPLIT_COUNTS,"R2 manifest split sizes changed")
    features=json.loads(FEATURE_FILE.read_text(encoding="utf-8"))["features"]
    check(len(features)==32 and len(set(features))==32, "32 unique features required")
    manifest_split={x["split"]:x for x in r2["outputs"]}
    check(set(manifest_split)==set(SPLIT_COUNTS),"R2 checksums incomplete")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.set_num_threads(4)
    device=torch.device("cuda:0")
    gpu=torch.cuda.get_device_name(device)
    print("="*85,flush=True)
    print("BLOCK 09A-B-R2 — GPU MLP BINARY STRICT-TEMPORAL BASELINE",flush=True)
    print("GPU:",gpu,"| Seed:",SEED,"| Cohort: distinct R2, not historical tree cohort",flush=True)
    print("No official validation/test labels used for training or threshold selection.",flush=True)

    print("\nLoading verified R2 training Parquet...",flush=True)
    x,y,fine,family,train_clips=load_split("train",features,manifest_split["train"]["sha256"])
    splitter=StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=SEED)
    fit_idx,hold_idx=next(splitter.split(np.zeros(len(fine),dtype=np.int8),fine))
    check(len(fit_idx)+len(hold_idx)==len(y),"Bad internal split")
    by_label=[]
    for label in sorted(np.unique(fine)):
        mask=fine==label
        fit=int(np.count_nonzero(fine[fit_idx]==label))
        hold=int(np.count_nonzero(fine[hold_idx]==label))
        check(fit+hold==int(mask.sum()) and fit>0 and hold>0,
              "Bad stratum allocation: "+label)
        by_label.append({"fine_grained_label":label,"total_rows":int(mask.sum()),
                         "internal_fit_rows":fit,"internal_holdout_rows":hold,
                         "holdout_fraction":hold/int(mask.sum())})
    write_csv(TABLES/"table_09a_b_r2_01_internal_holdout_allocation.csv",by_label)
    del fine,family
    print("Internal fit rows:",len(fit_idx),"holdout rows:",len(hold_idx),flush=True)

    xt=torch.from_numpy(x).to(device)
    yt=torch.from_numpy(y).to(device)
    fit_indices=torch.from_numpy(fit_idx.astype(np.int64)).to(device)
    hidx=torch.from_numpy(hold_idx.astype(np.int64)).to(device)
    xh=xt[hidx]
    yh=yt[hidx]
    model=BinaryMLP().to(device)
    params=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    criterion=nn.BCEWithLogitsLoss()
    best=float("inf")
    best_epoch=0
    best_state=None
    stalls=0
    history=[]
    train_start=time.monotonic()
    print("Training 32-128-64-1 MLP:",params,"parameters",flush=True)
    for epoch in range(1,EPOCHS+1):
        model.train()
        order=torch.randperm(len(fit_idx),device=device)
        loss_sum=0.
        for offset in range(0,len(order),BATCH):
            positions=fit_indices[order[offset:offset+BATCH]]
            xb,yb=xt[positions],yt[positions]
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb)
            loss=criterion(logits,yb)
            check(bool(torch.isfinite(loss).item()),"Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            loss_sum+=float(loss.item())*len(positions)
        hold_bce=val_loss(model,xh,yh)
        entry={"epoch":epoch,"fit_bce":loss_sum/len(fit_idx),
               "holdout_bce":hold_bce,"elapsed_seconds":time.monotonic()-train_start}
        history.append(entry)
        print("Epoch %02d fit_BCE=%.6f holdout_BCE=%.6f"%
              (epoch,entry["fit_bce"],hold_bce),flush=True)
        if hold_bce<best-1e-5:
            best=hold_bce
            best_epoch=epoch
            best_state={name:value.detach().cpu().clone()
                        for name,value in model.state_dict().items()}
            stalls=0
        else:
            stalls+=1
            if stalls>=PATIENCE:
                print("Internal early stopping (best epoch %d)"%best_epoch,flush=True)
                break
    training_seconds=time.monotonic()-train_start
    check(best_state is not None,"No trained checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    ckpt=CHECKPOINTS/"block_09a_b_r2_mlp_seed42.pt"
    torch.save({"model_state":best_state,"features":features,"seed":SEED,
                "best_epoch":best_epoch,"clip":CLIP,
                "dataset_manifest_sha256":digest(R2_MANIFEST)},ckpt)
    ckpt_sha=digest(ckpt)

    print("\nComputing INTERNAL holdout threshold...",flush=True)
    hold_score=score_model(model,x[hold_idx],device)
    hold_y=y[hold_idx]
    threshold,cap,actual_fp,num_benign=select_threshold(hold_score,hold_y)
    metrics=[operational(hold_y,hold_score,threshold,"internal_holdout")]
    check(metrics[0]["false_positive"]==actual_fp,"Threshold selection audit mismatch")
    print("Threshold %.12g | internal benign FPR %.6f (%d/%d)"%
          (threshold,actual_fp/num_benign,actual_fp,num_benign),flush=True)
    del xt,yt,fit_indices,hidx,xh,yh,x,y,hold_score,hold_y
    torch.cuda.empty_cache()

    all_families=[]
    clip_counts={"train":train_clips}
    test_y,test_p,test_family=None,None,None
    for split in ["validation","test"]:
        print("\nScoring evaluation-only "+split+"...",flush=True)
        x,y,fine,family,clipped=load_split(split,features,manifest_split[split]["sha256"])
        clip_counts[split]=clipped
        pred=score_model(model,x,device)
        split_name="official_validation" if split=="validation" else "strict_temporal_test"
        met=operational(y,pred,threshold,split_name)
        metrics.append(met)
        group=family_table(y,pred,family,threshold,split_name)
        all_families.extend(group)
        print("  F1=%.5f MCC=%.5f AUROC=%.5f AUPR=%.5f FPR=%.5f"%
              (met["f1"],met["mcc"],met["auroc"],met["aupr"],met["benign_fpr"]),flush=True)
        if split=="test":
            check({v["attack_family"]:v["row_count"] for v in group}==EXPECTED_TEST_FAMILY,
                  "Test family counts incorrect")
            test_y,test_p,test_family=y,pred,group
        del x,fine,family
    check(test_y is not None, "Test evaluation not performed")

    table_paths=[
        TABLES/"table_09a_b_r2_01_internal_holdout_allocation.csv",
        TABLES/"table_09a_b_r2_02_operational_metrics.csv",
        TABLES/"table_09a_b_r2_03_family_detection.csv",
        TABLES/"table_09a_b_r2_04_learning_curves.csv",
        TABLES/"table_09a_b_r2_05_threshold_audit.csv",
    ]
    write_csv(table_paths[1],metrics)
    write_csv(table_paths[2],all_families)
    write_csv(table_paths[3],history)
    write_csv(table_paths[4],[{
        "threshold_policy":"internal_fpr_at_most_1_percent",
        "threshold":threshold,"benign_internal_holdout_rows":num_benign,
        "allowed_false_positives":cap,"actual_false_positives":actual_fp,
        "realized_internal_fpr":actual_fp/num_benign,
        "official_validation_used_for_selection":False,
        "official_test_used_for_selection":False
    }])
    figure_paths=create_figures(history,test_y,test_p,test_family)
    summary_path=METRICS/"metric_09a_b_r2_gpu_mlp_summary.json"
    summary={
        "block":"09A-B-R2","status":"completed_distinct_r2_cohort",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "dataset":"compact_r2_6320003",
        "historical_tree_baseline_matched_cohort":False,
        "r2_manifest_sha256":digest(R2_MANIFEST),
        "model":"MLP-128-64","architecture":[32,128,64,1],
        "trainable_parameters":params,"seed":SEED,"gpu":gpu,
        "software":{"python":sys.version.split()[0],"torch":torch.__version__},
        "preprocessor_fitted_on_historical_train":True,
        "preprocessor_refitted_for_r2":False,
        "internal_holdout_preprocessing_fit_exposure":True,
        "fixed_neural_input_clip":[-CLIP,CLIP],
        "clipped_feature_cells":clip_counts,
        "trained_on_r2_internal_fit_only":True,
        "used_official_val_or_test_for_selection":False,
        "checkpoint_selection":"lowest_internal_holdout_BCE",
        "best_epoch":best_epoch,"best_internal_bce":best,
        "epochs_completed":len(history),"training_seconds":training_seconds,
        "threshold_selection":"benign_internal_holdout_FPR_at_most_1_percent",
        "threshold":threshold,"max_internal_false_positives":cap,
        "actual_internal_false_positives":actual_fp,
        "split_metrics":{r["split"]:r for r in metrics},
        "family_metrics":all_families,
        "local_only_checkpoint":str(ckpt.relative_to(ROOT)),
        "local_checkpoint_sha256":ckpt_sha,
        "scientific_caution":"R2 full-row cohort differs from historical tree cohort by 48 records. "
          "Preprocessing was frozen from old training set, not refitted on R2. "
          "Internal holdout shares historic training-fitted preprocessing parameters. "
          "No matched-cohort superiority or statistical significance is claimed.",
    }
    write_json(summary_path,summary)
    report_path=REPORTS/"block_09a_b_r2_gpu_mlp.md"
    tm=metrics[-1]
    report_path.write_text(
        "# Block 09A-B-R2 — GPU MLP temporal baseline\n\n"
        f"- GPU: {gpu}; seed {SEED}\n"
        "- R2 cohort: 6,320,003 deduplicated rows; not historically matched to trees.\n"
        f"- MLP: 32-128-64-1 with LayerNorm/GELU/dropout; {params:,} parameters.\n"
        f"- Internal fine-label-stratified train holdout: 15%; best epoch {best_epoch}.\n"
        f"- Threshold selected on internal benign FPR <=1%: {threshold:.12g}, "
        f"realized {actual_fp/num_benign:.6f}.\n\n"
        "## Strict temporal test\n\n| Metric | Value |\n|---|---:|\n"+
        "".join(f"| {k} | {tm[k]:.6f} |\n" for k in
                ["accuracy","balanced_accuracy","precision","recall",
                 "f1","mcc","auroc","aupr","benign_fpr"])+
        "\n## Interpretation\n\n"
        "Official validation/test were not used for checkpoint or threshold selection. "
        "The strict-temporal test includes families unseen in training. "
        "The original Block 08A transform is reused without refitting on R2; "
        "its historical training-fitted values overlap the internal holdout, "
        "which is a preprocessing-fit exposure and limitation. "
        "Input clipping at +/-20 is an additional fixed MLP-only transform. "
        "R2 differs by 48 flows, and original trees were not retrained on this cohort; "
        "do not make paired, controlled model-comparison claims. "
        "One seed/split does not establish statistical significance.\n",
        encoding="utf-8"
    )
    all_artifacts=table_paths+figure_paths+[summary_path,report_path]
    manifest_path=MANIFESTS/"block_09a_b_r2_gpu_mlp_manifest.json"
    write_json(manifest_path,{
        "block":"09A-B-R2","status":"completed_distinct_r2_cohort",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "historical_baselines_matched":False,
        "dataset_manifest_sha256":digest(R2_MANIFEST),
        "best_epoch":best_epoch,"operating_threshold":threshold,
        "model_checkpoint_sha256":ckpt_sha,"checkpoint_git_ignored":True,
        "artifact_sha256":{str(p.relative_to(ROOT)):digest(p) for p in all_artifacts},
        "elapsed_seconds":time.monotonic()-started,
    })
    print("\n"+"="*87,flush=True)
    print("BLOCK 09A-B-R2 — GPU MLP COMPLETE (RESULTS LOCAL; NO GIT PUSH)",flush=True)
    print("="*87,flush=True)
    print("GPU:",gpu,"| best epoch:",best_epoch,"| threshold:",threshold,flush=True)
    print("Internal holdout FPR:",metrics[0]["benign_fpr"],flush=True)
    print("Strict temporal: F1=%.6f MCC=%.6f AUROC=%.6f AUPR=%.6f FPR=%.6f"%
          (tm["f1"],tm["mcc"],tm["auroc"],tm["aupr"],tm["benign_fpr"]),flush=True)
    for row in test_family:
        print("  %s: %.4f%% %s"%(row["attack_family"],
              100*row["detection_rate_or_fpr"],
              "FPR" if row["attack_family"]=="Benign" else "recall"),flush=True)
    print("Research outputs: 5 CSV + summary/report/manifest + 4 PNG/PDF/caption trios",flush=True)
    print("Local checkpoint:",ckpt.relative_to(ROOT),flush=True)
    print("NOTE: historical tree models NOT matched to this R2 cohort.",flush=True)
    print("GitHub push: NOT ATTEMPTED.",flush=True)


if __name__=="__main__":
    main()
