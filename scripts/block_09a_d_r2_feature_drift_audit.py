#!/usr/bin/env python3
"""DCCT-NIDS Block 09A-D-R2 — read-only 32-feature temporal drift audit.

Uses the verified R2 Compact Parquet dataset; tracks all 6,320,003 rows with
tqdm progress bars. Deterministic hash sampling yields stable class-conditional
feature distributions. Reports KS, quantile shifts, and quantile-based PSI
between training and later time-split populations. All results are descriptive.
Do NOT tune feature selection, threshold, model, or hyperparameters on test.
No model fitting, no checkpoint writes, no Git/network/secrets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/model_ready/compact_r2_6320003"
R2_MANIFEST = ROOT / "outputs/manifests/block_09a_a_r2_versioned_compact_manifest.json"
MLP_SUMMARY = ROOT / "outputs/metrics/metric_09a_b_r2_gpu_mlp_summary.json"
SCORE_SUMMARY = ROOT / "outputs/metrics/metric_09a_c_r2_score_shift_summary.json"
FEATURES_FILE = ROOT / "configs/datasets/cse_cic_ids2018_features_compact.json"
OUTPUT_TABLES = ROOT / "outputs/tables"
OUTPUT_METRICS = ROOT / "outputs/metrics"
OUTPUT_MANIFESTS = ROOT / "outputs/manifests"
OUTPUT_FIGURES = ROOT / "outputs/figures"
OUTPUT_CAPTIONS = ROOT / "reports/captions"
OUTPUT_REPORTS = ROOT / "reports/experiment_notes"
ROWS = {"train": 4_202_458, "validation": 1_145_712, "test": 971_833}
CLIP = 20.0
SAMPLE_MASK = 31  # sample 1 / 32 of the rows through fixed 64-bit row-ID hash
BATCH = 65_536
BLOCK = "09A-D-R2"


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def write_csv(path, rows):
    require(bool(rows), "No records to write: " + str(path))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_sample(row_ids):
    # SplitMix64 hash; independent of ordering and reproducible across sessions.
    x = row_ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(SAMPLE_MASK)) == np.uint64(0)


def select_r2_split(split, features, manifest):
    path = DATA / (split + "_compact.parquet")
    require(path.is_file(), "Missing R2 split: " + str(path))
    prior_sha = next(r["sha256"] for r in manifest["outputs"] if r["split"] == split)
    require(sha256(path) == prior_sha, split + " data hash differs from verified R2 cohort")
    pf = pq.ParquetFile(path)
    require(pf.metadata.num_rows == ROWS[split], "R2 row count altered: " + split)
    expected = ["row_id", "file_order", "source_file", "capture_date",
                "fine_grained_label", "attack_family", "binary_label"] + features
    require(pf.schema_arrow.names == expected, split + " field sequence changed")
    return pf


def scan_all(cohort, features):
    print("Phase 1/3 — One exact pass over every model-ready R2 flow.", flush=True)
    samples = defaultdict(list)
    population_rows = defaultdict(int)
    population_clipped_cells = defaultdict(int)
    feature_clip_rows = defaultdict(lambda: np.zeros(len(features), dtype=np.int64))
    sample_counts = defaultdict(int)

    split_offsets = {"train": 0,
                     "validation": ROWS["train"],
                     "test": ROWS["train"] + ROWS["validation"]}

    with tqdm(total=3, desc="Parquet split progress", unit="split", dynamic_ncols=True) as split_bar:
        for split in ("train", "validation", "test"):
            parquet = select_r2_split(split, features, cohort)
            seen = 0
            with tqdm(total=ROWS[split], desc="Scanning " + split,
                      unit="rows", unit_scale=True, dynamic_ncols=True, leave=True) as pbar:
                for batch in parquet.iter_batches(
                    batch_size=BATCH,
                    columns=["row_id", "attack_family", "binary_label"] + features,
                ):
                    frame = batch.to_pandas()
                    n = len(frame)
                    ids = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
                    first = split_offsets[split] + seen + 1
                    expected_ids = np.arange(first, first + n, dtype=np.int64)
                    require(np.array_equal(ids, expected_ids),
                            split + ": altered chronological row IDs")
                    families = frame["attack_family"].astype(str).to_numpy(copy=False)
                    labels = frame["binary_label"].astype(str).to_numpy(copy=False)
                    require(np.all((families == "Benign") == (labels == "Benign")),
                            split + ": attack-family label mismatch")
                    x = frame[features].to_numpy(dtype=np.float32, copy=True)
                    require(np.isfinite(x).all(), split + ": non-finite transformed feature")
                    clipped = (x < -CLIP) | (x > CLIP)
                    np.clip(x, -CLIP, CLIP, out=x)
                    sample_mask = stable_sample(ids)

                    for family in np.unique(families):
                        group = (split, str(family))
                        inside = families == family
                        population_rows[group] += int(inside.sum())
                        population_clipped_cells[group] += int(clipped[inside].sum())
                        feature_clip_rows[group] += clipped[inside].sum(axis=0)
                        # Preserve rare validation strata rather than
                        # risking a zero-sample group from 1/32 hashing.
                        # These are descriptive samples, not training weights.
                        keep_all_rare = (
                            family == "Web Attack" or
                            (split == "validation" and family == "Brute Force")
                        )
                        wanted = inside if keep_all_rare else (sample_mask & inside)
                        if np.any(wanted):
                            subset = x[wanted].copy()
                            samples[group].append(subset)
                            sample_counts[group] += len(subset)

                    seen += n
                    pbar.update(n)
            require(seen == ROWS[split], "Incomplete R2 split scan: " + split)
            split_bar.update(1)
    full = {key: np.concatenate(chunks, axis=0)
            for key, chunks in samples.items()}
    for key, count in population_rows.items():
        require(key in full and len(full[key]) >= 20,
                "Insufficient deterministic sample from cohort group " + str(key))
        require(sample_counts[key] == len(full[key]), "Sampling count inconsistency")
    print("  Exact scanned rows:", sum(population_rows.values()), flush=True)
    print("  Deterministic sampled rows:", sum(map(len, full.values())), flush=True)
    return full, population_rows, population_clipped_cells, feature_clip_rows


def reference_pair(samples, split, families):
    arrays = [array for (s, fam), array in samples.items()
              if s == split and ((families == "Benign" and fam == "Benign")
                                 or (families == "Attack" and fam != "Benign")
                                 or (families == fam))]
    require(bool(arrays), "No sample population: " + split + " / " + families)
    return np.concatenate(arrays, axis=0)


def psi(reference, comparison):
    # Ref-derived equal-frequency bins, no validation or test fitting.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) <= 2:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    first = np.histogram(reference, bins=edges)[0].astype(np.float64)
    second = np.histogram(comparison, bins=edges)[0].astype(np.float64)
    # Smoothing prevents division-by-zero when a new class occupies unseen bins.
    a = np.maximum(first / len(reference), 1e-6)
    b = np.maximum(second / len(comparison), 1e-6)
    value = float(np.sum((b-a) * np.log(b/a)))
    require(np.isfinite(value), "Non-finite PSI")
    return value


def make_comparisons(groups, feature_names):
    comparisons = [
        ("benign_train_to_validation", "train", "Benign", "validation", "Benign"),
        ("benign_train_to_test", "train", "Benign", "test", "Benign"),
        ("attack_train_to_validation", "train", "Attack", "validation", "Attack"),
        ("attack_train_to_botnet_test", "train", "Attack", "test", "Botnet"),
        ("attack_train_to_infiltration_test", "train", "Attack", "test", "Infiltration"),
        ("benign_train_to_botnet_test", "train", "Benign", "test", "Botnet"),
    ]
    results = []
    with tqdm(total=len(comparisons)*len(feature_names),
              desc="Feature KS/PSI comparisons", unit="feature", dynamic_ncols=True) as bar:
        for label, a_split, a_family, b_split, b_family in comparisons:
            reference = reference_pair(groups, a_split, a_family)
            target = reference_pair(groups, b_split, b_family)
            for j, name in enumerate(feature_names):
                a, b = reference[:, j], target[:, j]
                d = float(ks_2samp(a, b, method="asymp").statistic)
                a_med, b_med = float(np.median(a)), float(np.median(b))
                a_p90, b_p90 = float(np.quantile(a, .9)), float(np.quantile(b, .9))
                calculated_psi = psi(a,b)
                results.append({
                    "comparison": label,
                    "feature": name,
                    "reference_sample_rows": len(a),
                    "target_sample_rows": len(b),
                    "ks_distance": d,
                    "reference_median": a_med,
                    "target_median": b_med,
                    "median_change_target_minus_reference": b_med-a_med,
                    "reference_p90": a_p90,
                    "target_p90": b_p90,
                    "psi_train_reference_bins": calculated_psi if calculated_psi is not None else "",
                    "psi_identifiable": calculated_psi is not None,
                })
                bar.update()
    return results


def plot_outputs(comparisons, feature_names):
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Compare train-to-test benign drift and train attack-to-test Botnet drift,
    # not to select features for retraining; simply to document the shift.
    views = [
        ("benign_train_to_test", "Benign: train → test"),
        ("attack_train_to_botnet_test", "Attack: train → test Botnet"),
        ("attack_train_to_infiltration_test", "Attack: train → test Infiltration"),
    ]
    index = {(row["comparison"],row["feature"]):row["ks_distance"] for row in comparisons}
    fig, axs = plt.subplots(1,3,figsize=(16,10),sharey=True,constrained_layout=True)
    for ax,(group,heading) in zip(axs,views):
        scores=np.asarray([index[(group,f)] for f in feature_names])
        places=np.argsort(scores)[::-1]
        ordered=[feature_names[i] for i in places]
        ax.barh(np.arange(len(ordered)),scores[places],color=color[0],height=.75)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(ordered)), labels=ordered, fontsize=7)
        ax.set(xlim=(0,1),xlabel="Sampled KS distance",title=heading)
        ax.grid(axis="x",alpha=.15)
    outputs=[]
    basename="figure_09a_d_r2_01_feature_shift"
    png=OUTPUT_FIGURES/(basename+".png")
    pdf=OUTPUT_FIGURES/(basename+".pdf")
    caption=OUTPUT_CAPTIONS/(basename+".md")
    fig.savefig(png,dpi=600,bbox_inches="tight")
    fig.savefig(pdf,bbox_inches="tight")
    plt.close(fig)
    caption.write_text(
        "**Figure.** Descriptive frozen-feature distribution shift using a deterministic "
        "1/32 hash sample of each R2 split. Each panel ranks the 32 frozen Compact "
        "features by KS distance; official test features were inspected only after "
        "the MLP experiment completed and must not be used for hyperparameter "
        "or feature-selection tuning on the same held-out test.\n",
        encoding="utf-8")
    outputs.extend([png,pdf,caption])
    return outputs


def main():
    start=time.monotonic()
    for folder in [OUTPUT_TABLES, OUTPUT_METRICS, OUTPUT_MANIFESTS,
                   OUTPUT_FIGURES, OUTPUT_CAPTIONS, OUTPUT_REPORTS]:
        folder.mkdir(parents=True,exist_ok=True)
    for p in [R2_MANIFEST, MLP_SUMMARY, SCORE_SUMMARY, FEATURES_FILE]:
        require(p.is_file(), "Missing verified prior artifact " + str(p))
    r2=json.loads(R2_MANIFEST.read_text(encoding="utf-8"))
    mlp=json.loads(MLP_SUMMARY.read_text(encoding="utf-8"))
    shift=json.loads(SCORE_SUMMARY.read_text(encoding="utf-8"))
    require(r2["status"]=="verified_distinct_cohort_not_historical_matched",
            "Wrong R2 cohort status")
    require(r2["split_row_counts"]==ROWS,"R2 row counts changed")
    require(mlp["status"]=="completed_distinct_r2_cohort",
            "MLP experiment status incompatible")
    require(shift["status"]=="reproduced_heldout_checkpoint",
            "MLP score reproduction has not passed")
    require(shift["fixed_training_selected_threshold"]==mlp["threshold"],
            "Original operating threshold changed")
    require(mlp["r2_manifest_sha256"]==sha256(R2_MANIFEST),
            "MLP trained on a different R2 cohort manifest")
    require(shift["r2_manifest_sha256"]==sha256(R2_MANIFEST),
            "Score audit used another R2 cohort")
    check_names=json.loads(FEATURES_FILE.read_text(encoding="utf-8"))["features"]
    require(len(check_names)==32 and len(set(check_names))==32,
            "Expected 32 frozen Compact features")
    print("="*87, flush=True)
    print("BLOCK 09A-D-R2 — FROZEN 32-FEATURE TEMPORAL DRIFT",flush=True)
    print("No model retraining, threshold tuning, or Github push.",flush=True)
    print("All 6,320,003 rows scanned, 1/32 sampled for distribution comparisons.",flush=True)
    groups,counts,clipped,feature_clip=scan_all(r2,check_names)
    require(sum(counts.values())==6_320_003, "Total row audit failed")
    print("Phase 2/3 — Train-referenced feature drift diagnostics.",flush=True)
    diffs=make_comparisons(groups,check_names)
    drift_csv=OUTPUT_TABLES/"table_09a_d_r2_01_feature_distribution_shift.csv"
    write_csv(drift_csv,diffs)
    clip_rows=[]
    for (split,family),rows in sorted(counts.items()):
        clipped_cells=clipped[(split,family)]
        per_feature=feature_clip[(split,family)]
        clip_rows.append({
            "split":split,"attack_family":family,"population_rows":rows,
            "sampled_rows":len(groups[(split,family)]),
            "clipped_feature_cells":clipped_cells,
            "clipped_cells_divided_by_all_feature_cells":clipped_cells/(rows*32),
            "features_with_clipping":int(np.count_nonzero(per_feature)),
            "most_clipped_feature":check_names[int(np.argmax(per_feature))],
            "most_clipped_feature_cell_count":int(per_feature.max()),
        })
    clip_csv=OUTPUT_TABLES/"table_09a_d_r2_02_input_clipping_audit.csv"
    write_csv(clip_csv,clip_rows)
    top=[]
    for comparison in sorted({v["comparison"] for v in diffs}):
        matches=sorted((d for d in diffs if d["comparison"]==comparison),
                       key=lambda item:item["ks_distance"],reverse=True)[:10]
        for rank,row in enumerate(matches,1):
            top.append({"comparison":comparison,"rank":rank,"feature":row["feature"],
                        "ks_distance":row["ks_distance"],
                        "reference_median":row["reference_median"],
                        "target_median":row["target_median"]})
    top_csv=OUTPUT_TABLES/"table_09a_d_r2_03_top_feature_shifts.csv"
    write_csv(top_csv,top)
    print("Phase 3/3 — Building 600-DPI diagnostic charts and integrity records.",flush=True)
    figures=plot_outputs(diffs,check_names)
    report=OUTPUT_REPORTS/"block_09a_d_r2_feature_shift.md"
    report.write_text(
        "# Block 09A-D-R2 — Descriptive R2 feature-distribution drift\n\n"
        "- Original MLP 09A-B-R2 checkpoint and threshold unchanged.\n"
        "- All 6,320,003 flows processed; stable 1/32 hash sampling for 32-dimensional "
        "frozen-feature distribution statistics.\n"
        "- Quantified class-conditional Kolmogorov–Smirnov distances and "
        "reference-quantile population stability indices (PSI).\n"
        "- Compared prespecified class groups on train, official validation, "
        "and strict-temporal test; all 32 frozen model-ready features analyzed.\n"
        "- Independently recorded exact population-level neural input clipping "
        "counts by chronological split and attack family.\n"
        "- **This is descriptive post-hoc test examination, not feature-selection "
        "or model/threshold tuning.** The official test should not be used as "
        "the development set for a new MLP experiment.\n"
        "- R2 cohort is not identical to historical tree-model cohort. "
        "The frozen preprocessor derives from historical full training and "
        "shares fitted statistics with internal holdout.\n",
        encoding="utf-8")
    summary_path=OUTPUT_METRICS/"metric_09a_d_r2_feature_shift_summary.json"
    top_feature={group:next(row["feature"] for row in top if row["comparison"]==group)
                 for group in sorted({r["comparison"] for r in top})}
    write_json(summary_path,{
        "block":BLOCK,"status":"completed_descriptive_r2_drift_audit",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "dataset_revision":"compact_r2_6320003",
        "r2_manifest_sha256":sha256(R2_MANIFEST),
        "prior_mlp_summary_sha256":sha256(MLP_SUMMARY),
        "prior_score_shift_summary_sha256":sha256(SCORE_SUMMARY),
        "population_rows_scanned":sum(counts.values()),
        "sampling_method":"SplitMix64 deterministic row-id hash, 1/32; full census for Web Attack and validation Brute Force rare strata",
        "sampled_total_rows":sum(map(len,groups.values())),
        "feature_count":len(check_names),
        "clip_policy":"fixed MLP-only [-20,20] after historical frozen preprocessing",
        "top_ks_feature_by_comparison":top_feature,
        "test_derived_feature_selection":False,
        "test_derived_threshold_selection":False,
        "model_modified":False,
        "diagnostics_only":True,
        "historical_tree_models_cohort_matched":False,
        "elapsed_seconds":time.monotonic()-start,
    })
    artifacts=[drift_csv,clip_csv,top_csv,summary_path,report]+figures
    manifest_path=OUTPUT_MANIFESTS/"block_09a_d_r2_feature_shift_manifest.json"
    write_json(manifest_path,{
        "block":BLOCK,"status":"completed_descriptive_r2_drift_audit",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "artifact_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in artifacts},
        "trained_model_modified":False,"selection_from_test":False,
        "prior_mlp_checkpoint_sha256":mlp["local_checkpoint_sha256"],
    })
    print("\n"+"="*88,flush=True)
    print("BLOCK 09A-D-R2 — FEATURE DRIFT AUDIT COMPLETE",flush=True)
    print("Original MLP weights and threshold unchanged; no Git push.",flush=True)
    print("Population scanned:",sum(counts.values()),"Sampled:",sum(map(len,groups.values())),flush=True)
    print("Main class-conditional KS shifts (top 5 each):",flush=True)
    for comp in ["benign_train_to_test","attack_train_to_botnet_test",
                 "attack_train_to_infiltration_test"]:
        print(" ",comp,flush=True)
        for row in [r for r in top if r["comparison"]==comp][:5]:
            print("   %s KS=%.4f"%(row["feature"],row["ks_distance"]),flush=True)
    print("Artifacts: 3 tables, 600-DPI PNG/PDF/caption, metric JSON, manifest, report.",flush=True)
    print("Do not use test-derived drift rankings for model tuning.",flush=True)


if __name__ == "__main__":
    main()
