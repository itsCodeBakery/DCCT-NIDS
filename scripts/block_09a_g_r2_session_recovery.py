#!/usr/bin/env python3
"""DCCT-NIDS 09A-G-R2: restart-safe rebuild after ephemeral Kaggle runtime reset.

A missing /kaggle/working artifact is not evidence of a flawed research model.
Reproduce missing 09A-A-R2 data, 09A-B-R2 MLP, 09A-E-R2 GPU XGBoost, and
09A-F-R2 exploratory comparison IN ORDER using pinned committed source.
Never overwrites incomplete or unverified local model checkpoints. Never
pushes, pulls, merges, resets Git or handles credentials. Read-only git show
accesses pre-fetched commit objects; caller must fetch before invocation.
All training/temporal results remain exploratory; a new Kaggle execution is
a rerun, not a cryptographic reconstruction of prior local lost checkpoints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
RAW=Path("/kaggle/input/datasets/dhoogla/csecicids2018")
R2_DATA=ROOT/"data/model_ready/compact_r2_6320003"
MAN=ROOT/"outputs/manifests"
MET=ROOT/"outputs/metrics"
TAB=ROOT/"outputs/tables"
CKPT=ROOT/"outputs/checkpoints"

PINNED_SOURCES=[
    ("09A-A-R2","c99d7135d5be9cb18e8182325d306c9fa0cb4819",
     "scripts/block_09a_a_r2_versioned_compact.py"),
    ("09A-B-R2","b247c414ee892ea7dad4b32e6de3f5c6e5e03788",
     "scripts/block_09a_b_r2_gpu_mlp.py"),
    ("09A-E-R2","86c28fb375df1acbcf42dd732df6f949abe54a42",
     "scripts/block_09a_e_r2_matched_xgboost.py"),
    ("09A-F-R2","2028666cdfbd73077a198fdc78272e5294748598",
     "scripts/block_09a_f_r2_model_comparison_audit.py"),
]
EXPECTED_ROWS={"train":4202458,"validation":1145712,"test":971833}
STAGE_FILES={
    "09A-A-R2":[
        MAN/"block_09a_a_r2_versioned_compact_manifest.json",
        TAB/"table_09a_a_r2_01_capture_retention_deltas.csv",
        TAB/"table_09a_a_r2_02_capture_family_deltas.csv",
        *[R2_DATA/f"{key}_compact.parquet" for key in EXPECTED_ROWS],
    ],
    "09A-B-R2":[
        MET/"metric_09a_b_r2_gpu_mlp_summary.json",
        MAN/"block_09a_b_r2_gpu_mlp_manifest.json",
        CKPT/"block_09a_b_r2_mlp_seed42.pt",
        TAB/"table_09a_b_r2_01_internal_holdout_allocation.csv",
        TAB/"table_09a_b_r2_02_operational_metrics.csv",
        TAB/"table_09a_b_r2_03_family_detection.csv",
    ],
    "09A-E-R2":[
        MET/"metric_09a_e_r2_xgboost_reference_summary.json",
        MAN/"block_09a_e_r2_xgboost_reference_manifest.json",
        CKPT/"block_09a_e_r2_xgb_reference_seed42.ubj",
        TAB/"table_09a_e_r2_01_matched_internal_allocation.csv",
        TAB/"table_09a_e_r2_02_xgboost_operational_metrics.csv",
        TAB/"table_09a_e_r2_03_family_detection.csv",
        TAB/"table_09a_e_r2_06_exploratory_r2_mlp_xgb_comparison.csv",
    ],
    "09A-F-R2":[
        MET/"metric_09a_f_r2_comparison_summary.json",
        MAN/"block_09a_f_r2_comparison_manifest.json",
        TAB/"table_09a_f_r2_02_exploratory_split_comparison.csv",
        TAB/"table_09a_f_r2_03_exploratory_family_comparison.csv",
    ],
}
TEST_PREVIOUS={
    "MLP-R2":{"f1":.027680,"mcc":.004204,"auroc":.397708,
              "aupr":.170376,"benign_fpr":.013551},
    "XGBoost-R2":{"f1":.802110,"mcc":.770763,"auroc":.845102,
                  "aupr":.800056,"benign_fpr":.012851},
}

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def show_commit(sha,relative):
    # 'git show' is read-only and NEVER updates local working tree or branches.
    result=subprocess.run(
        ["git","show",f"{sha}:{relative}"],
        cwd=ROOT,capture_output=True,text=True,check=False,
    )
    check(result.returncode==0,
          "Pinned source commit unavailable locally: "+sha+" / "+relative+
          ". Fetch origin/main again (without pull/reset), then retry.\n"+
          result.stderr[-800:])
    return result.stdout

def verify_pinned_scripts():
    for block,sha,relative in tqdm(PINNED_SOURCES,desc="Pinning immutable scripts",
                                    unit="script",dynamic_ncols=True):
        source=show_commit(sha,relative)
        check(len(source)>1000,"Unexpectedly short pinned source: "+relative)
        compile(source,relative,"exec")
        script=ROOT/relative
        script.parent.mkdir(parents=True,exist_ok=True)
        if script.is_file():
            check(script.read_text(encoding="utf-8").strip()==source.strip(),
                  "Different local script already exists: "+str(script)+
                  ". Preserved without overwriting.")
        else:
            script.write_text(source.rstrip("\n")+"\n",encoding="utf-8")
    print("✓ Original committed sources pinned; local Git branch unchanged",flush=True)

def inventory():
    print("\n"+("="*86))
    print("RECOVERY INVENTORY — existing artifact safety checks")
    print("="*86,flush=True)
    statuses={}
    for block,paths in STAGE_FILES.items():
        present=[p for p in paths if p.is_file()]
        missing=[p for p in paths if not p.is_file()]
        status="COMPLETE" if not missing else ("MISSING" if not present else "PARTIAL")
        statuses[block]=status
        print(f"  {block}: {status} ({len(present)}/{len(paths)} key artifacts)",flush=True)
        if status=="PARTIAL":
            for p in missing:
                print("    missing:",p.relative_to(ROOT),flush=True)
    return statuses

def guard_recovery(statuses):
    # A partially existing stage can represent an interrupted write or valuable
    # model checkpoint. Do not erase/replace it behind the researcher's back.
    for block in ["09A-B-R2","09A-E-R2","09A-F-R2"]:
        check(statuses[block]!="PARTIAL",
              "Partial "+block+" recovery detected. Existing artifacts are preserved. "
              "Investigate the missing files before rerunning this stage.")
    if statuses["09A-A-R2"]=="PARTIAL":
        files=STAGE_FILES["09A-A-R2"]
        data=[R2_DATA/f"{k}_compact.parquet" for k in EXPECTED_ROWS]
        all_parquets=all(p.is_file() for p in data)
        any_parquets=any(p.is_file() for p in data)
        checks=(TAB/"table_09a_a_r2_01_capture_retention_deltas.csv",
                TAB/"table_09a_a_r2_02_capture_family_deltas.csv")
        can_reaudit=(all_parquets and all(p.is_file() for p in checks) and
                     not (MAN/"block_09a_a_r2_versioned_compact_manifest.json").exists())
        check(can_reaudit,
              "Partial 09A-A-R2 reconstruction exists, not automatically repairable. "
              "Do not delete it; preserve and inspect before retrying.")

def check_raw_input():
    check(RAW.is_dir(),
          "Kaggle input dataset not mounted: "+str(RAW)+
          "\nAttach /kaggle/input/datasets/dhoogla/csecicids2018 to the notebook.")
    files=sorted(RAW.glob("*.parquet"))
    check(len(files)==10,f"Expected exactly 10 raw Parquet captures, found {len(files)}")
    total=0
    with tqdm(total=10,desc="Auditing mounted raw captures",
              unit="file",dynamic_ncols=True) as bar:
        for file in files:
            total+=pq.ParquetFile(file).metadata.num_rows
            bar.update()
    check(total==6_659_532,
          f"Raw inputs changed: observed {total:,} rows, expected 6,659,532")
    space=shutil.disk_usage(ROOT).free/(1024**3)
    print(f"✓ Raw capture cohort: {total:,} rows, 10 Parquet files",flush=True)
    print(f"✓ Available workspace: {space:.1f} GiB",flush=True)
    check(space>=7,"Insufficient free space for full-row R2 reconstruction (<7 GiB)")

def verify_r2():
    m=load_json(STAGE_FILES["09A-A-R2"][0])
    check(m["status"]=="verified_distinct_cohort_not_historical_matched",
          "R2 audit status is not verified")
    check(m["current_retained_rows"]==6_320_003 and
          m["historical_retained_rows"]==6_319_955,
          "R2 and historical cohort counts have changed")
    check(m["split_row_counts"]==EXPECTED_ROWS,"R2 split counts altered")
    manifest_splits={v["split"]:v for v in m["outputs"]}
    check(set(manifest_splits)==set(EXPECTED_ROWS),
          "R2 manifest split checksum entries invalid")
    for name in tqdm(EXPECTED_ROWS,desc="Verifying R2 datasets",
                     unit="split",dynamic_ncols=True):
        path=R2_DATA/f"{name}_compact.parquet"
        check(path.is_file(),"R2 Parquet missing: "+str(path))
        check(pq.ParquetFile(path).metadata.num_rows==EXPECTED_ROWS[name],
              "R2 split row count changed: "+name)
        check(digest(path)==manifest_splits[name]["sha256"],
              "R2 Parquet SHA256 changed: "+name)

def verify_mlp():
    obj=load_json(MET/"metric_09a_b_r2_gpu_mlp_summary.json")
    check(obj["status"]=="completed_distinct_r2_cohort","Incorrect MLP status")
    check(obj["r2_manifest_sha256"]==digest(STAGE_FILES["09A-A-R2"][0]),
          "MLP R2 source manifest SHA does not match current reconstruction")
    check(digest(CKPT/"block_09a_b_r2_mlp_seed42.pt")==
          obj["local_checkpoint_sha256"],"MLP checkpoint hash differs from summary")
    check(obj["seed"]==42 and obj["fixed_neural_input_clip"]==[-20.0,20.0],
          "MLP input protocol/seed mismatch")
    check(obj["used_official_val_or_test_for_selection"] is False,
          "MLP selection protocol mismatch")

def verify_xgb():
    obj=load_json(MET/"metric_09a_e_r2_xgboost_reference_summary.json")
    check(obj["status"]=="completed_exploratory_matched_r2_xgboost",
          "Incorrect XGBoost R2 status")
    check(obj["r2_manifest_sha256"]==digest(STAGE_FILES["09A-A-R2"][0]),
          "XGBoost R2 source manifest hash mismatch")
    check(obj["mlp_summary_sha256"]==
          digest(MET/"metric_09a_b_r2_gpu_mlp_summary.json"),
          "XGBoost MLP predecessor summary hash mismatch")
    check(digest(CKPT/"block_09a_e_r2_xgb_reference_seed42.ubj")==
          obj["checkpoint_sha256"],"XGBoost checkpoint checksum mismatch")
    check(obj["matched_current_r2_mlp_cohort"] is True and
          obj["same_stratified_internal_split_as_mlp"] is True and
          obj["selection_uses_official_validation"] is False and
          obj["selection_uses_test"] is False,
          "XGBoost matching/test-selection protocol mismatch")

def verify_comparison():
    obj=load_json(MET/"metric_09a_f_r2_comparison_summary.json")
    check(obj["status"]=="verified_exploratory_r2_comparison",
          "Comparison audit not verified")
    check(obj["same_r2_manifest_sha256"]==
          digest(STAGE_FILES["09A-A-R2"][0]),
          "Comparison source R2 manifest checksum mismatch")
    check(obj["same_stratified_internal_allocation"] is True and
          obj["independent_confirmatory_inference"] is False,
          "Exploratory matched-cohort declaration missing")
    manifest=load_json(MAN/"block_09a_f_r2_comparison_manifest.json")
    for name,expected in tqdm(manifest["artifact_sha256"].items(),
                              desc="Verifying comparison artifacts",
                              unit="file",dynamic_ncols=True):
        path=ROOT/name
        check(path.is_file() and digest(path)==expected,
              "Comparison output missing/changed: "+name)
    return obj

def run_stage(block,relative):
    print("\n"+("="*86))
    print("REPRODUCING MISSING "+block)
    print("="*86,flush=True)
    environment=os.environ.copy()
    environment["PYTHONUNBUFFERED"]="1"
    result=subprocess.run(
        [sys.executable,"-u",str(ROOT/relative)],
        cwd=ROOT,env=environment,check=False,
    )
    check(result.returncode==0,
          block+" stopped at an integrity/training check; inspect the traceback. "
          "No Git push was attempted.")
    print("✓ "+block+" completed",flush=True)

def main():
    started=time.monotonic()
    check((ROOT/".git").is_dir(),"Expected a checked-out DCCT-NIDS Git repository")
    print("="*86,flush=True)
    print("DCCT-NIDS / RECOVERY AFTER KAGGLE RUNTIME RESET",flush=True)
    print("No forced file deletion, no Git branch changes, no pushes.",flush=True)
    print("Experiments from a fresh run are new REPRODUCTIONS of prior scripts.",flush=True)
    verify_pinned_scripts()
    statuses=inventory()
    guard_recovery(statuses)
    check_raw_input()
    stage_actions=[]
    for block,sha,relative in tqdm(PINNED_SOURCES,
                                   desc="Recovery stage progress",
                                   unit="stage",dynamic_ncols=True):
        if statuses[block]=="COMPLETE":
            print("\nREUSE VERIFIED "+block,flush=True)
            stage_actions.append({"block":block,"action":"verified_existing"})
        else:
            run_stage(block,relative)
            stage_actions.append({"block":block,"action":"rerun_from_pinned_source"})
        if block=="09A-A-R2":
            verify_r2()
        elif block=="09A-B-R2":
            verify_mlp()
        elif block=="09A-E-R2":
            verify_xgb()
        else:
            comparison=verify_comparison()
    # Explicitly identify these as *new* snapshots, not vanished bytewise prior artifacts.
    previous_compare={}
    for model,obj in comparison["temporal_test_metrics"].items():
        old=TEST_PREVIOUS[model]
        previous_compare[model]={
            metric:{"new":obj[metric],"old_printed":old[metric],
                    "absolute_difference":abs(obj[metric]-old[metric])}
            for metric in old
        }
    recovery={
        "block":"09A-G-R2",
        "status":"reconstructed_and_audited_after_kaggle_reset",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_pins":{block:{"commit":sha,"path":relative}
                       for block,sha,relative in PINNED_SOURCES},
        "actions":stage_actions,
        "r2_manifest_sha256":digest(STAGE_FILES["09A-A-R2"][0]),
        "mlp_summary_sha256":digest(STAGE_FILES["09A-B-R2"][0]),
        "xgboost_summary_sha256":digest(STAGE_FILES["09A-E-R2"][0]),
        "comparison_summary_sha256":digest(STAGE_FILES["09A-F-R2"][0]),
        "comparisons_with_previously_printed_results":previous_compare,
        "scientific_warning":"This run may regenerate numerical outputs with different "
            "byte SHA values from the lost previous Kaggle run. Testing previously "
            "inspected temporal data is exploratory, not independent confirmation.",
        "elapsed_seconds":time.monotonic()-started,
        "git_push_attempted":False,
    }
    out=MAN/"block_09a_g_r2_session_recovery_manifest.json"
    write_json= lambda p,d: p.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",
                                         encoding="utf-8")
    write_json(out,recovery)
    print("\n"+("="*88))
    print("09A-G-R2 — SESSION RECOVERY COMPLETE; 09A-F AUDIT VERIFIED")
    print("="*88,flush=True)
    for record in stage_actions:
        print(f"  {record['block']}: {record['action']}",flush=True)
    for model,v in previous_compare.items():
        print(f"  {model} new temporal metrics:",flush=True)
        for key,data in v.items():
            print(f"    {key}: {data['new']:.6f} (previous printed {data['old_printed']:.6f}; "
                  f"|Δ|={data['absolute_difference']:.6g})",flush=True)
    print("Recovery manifest:",out.relative_to(ROOT),flush=True)
    print("All historical 08E files untouched. No arbitrary row deletion.",flush=True)
    print("No Git push. Final comparisons EXPLORATORY.",flush=True)
    print("="*88,flush=True)

if __name__=="__main__":
    main()
