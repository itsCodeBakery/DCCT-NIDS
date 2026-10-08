#!/usr/bin/env python3
"""DCCT-NIDS Block 8E: reproducible comparison of strict-temporal tree baselines.

Reads previously published JSON/CSV experiment artifacts. No dataset, checkpoint,
inference or model retraining is required. Does not access GitHub credentials.
Usage: python scripts/block_08e_tree_benchmarks.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"
MANIFESTS = ROOT / "outputs" / "manifests"
CAPTIONS = ROOT / "reports" / "captions"
REPORTS = ROOT / "reports" / "experiment_notes"
REGISTRY = ROOT / "configs" / "experiments" / "experiment_registry.json"
FEATURES = ROOT / "configs" / "datasets" / "cse_cic_ids2018_features_compact.json"

SOURCES = {
    "XGBoost": {
        "summary": METRICS / "metric_08b_fix_v2_operational_summary.json",
        "family": TABLES / "table_08b_fix_06_family_results.csv",
        "training": TABLES / "table_08b_fix_05_training_summary.csv",
    },
    "LightGBM": {
        "summary": METRICS / "metric_08c_lightgbm_operational_summary.json",
        "family": TABLES / "table_08c_04_lightgbm_family_results.csv",
        "training": TABLES / "table_08c_03_lightgbm_training_summary.csv",
    },
    "CatBoost": {
        "summary": METRICS / "metric_08d_catboost_operational_summary.json",
        "family": TABLES / "table_08d_06_catboost_family_results.csv",
        "training": TABLES / "table_08d_05_catboost_training_summary.csv",
    },
}
ORDER = list(SOURCES)
MAIN_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "auroc", "aupr", "benign_fpr"]
RANK_METRICS = ["mcc", "f1", "balanced_accuracy", "auroc", "aupr", "recall", "benign_fpr", "infiltration_recall"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def check_close(actual, expected, label, tolerance=1e-7):
    if not (math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance):
        raise RuntimeError(f"Integrity failure: {label}: saved={actual}, calculated={expected}")


def audit_confusion(model: str, split: str, metrics: dict) -> dict:
    tn, fp, fn, tp = (int(metrics[k]) for k in ["true_negative", "false_positive", "false_negative", "true_positive"])
    if min(tn, fp, fn, tp) < 0:
        raise RuntimeError(f"Negative confusion count: {model}/{split}")
    n = tn + fp + fn + tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    calculated = {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "benign_fpr": 1.0 - specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-30),
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
    }
    for key, value in calculated.items():
        check_close(metrics[key], value, f"{model}/{split}/{key}")
    return {"model": model, "split": split, "rows": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "status": "PASS"}


def save_table(frame: pd.DataFrame, name: str, created: list[Path]) -> Path:
    path = TABLES / name
    frame.to_csv(path, index=False, float_format="%.12g")
    created.append(path)
    return path


def save_chart(fig, stem: str, caption: str, created: list[Path]):
    png, pdf, txt = FIGURES / (stem + ".png"), FIGURES / (stem + ".pdf"), CAPTIONS / (stem + ".txt")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.16)
    txt.write_text(caption + "\n", encoding="utf-8")
    created.extend([png, pdf, txt])
    plt.close(fig)


def fmt_bars(axis, bars, decimals=3, suffix=""):
    for bar in bars:
        v = bar.get_height()
        axis.annotate(f"{v:.{decimals}f}{suffix}", (bar.get_x() + bar.get_width() / 2, v),
                      xytext=(0, 4), textcoords="offset points", ha="center",
                      va="bottom", fontsize=8.2, fontweight="bold")


def main():
    for folder in [TABLES, FIGURES, METRICS, MANIFESTS, CAPTIONS, REPORTS]:
        folder.mkdir(parents=True, exist_ok=True)
    required = [FEATURES, REGISTRY] + [path for items in SOURCES.values() for path in items.values()]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required GitHub-backed research artifacts missing:\n" + "\n".join(missing))

    feature_cfg = load_json(FEATURES)
    if len(feature_cfg.get("features", [])) != 32:
        raise RuntimeError("Expected exactly 32 Compact features")
    registry = load_json(REGISTRY)
    if registry.get("next_block", {}).get("block") not in ("08E", "09A"):
        print("NOTE: registry next block differs; preserving historical completed entries.")

    created: list[Path] = []
    summary_rows, family_rows, training_rows, checks, warnings = [], [], [], [], []
    for model in ORDER:
        paths = SOURCES[model]
        source = load_json(paths["summary"])
        policy = source.get("selected_threshold_policy", "fpr_at_most_1_percent")
        variant = str(source["selected_variant"])
        internal = source["internal_selection_metrics"]
        validation = source["official_validation_metrics"]
        test = source["strict_temporal_test_metrics"]

        for split_name, record in [("internal_holdout", internal), ("official_validation", validation), ("strict_temporal_test", test)]:
            checks.append(audit_confusion(model, split_name, record))

        if test.get("split") != "strict_temporal_test":
            raise RuntimeError(f"{model}: unexpected test protocol")

        if model == "XGBoost":
            audit_path = TABLES / "table_08b_fix_09_exact_threshold_audit.csv"
            if not audit_path.exists():
                raise RuntimeError("XGBoost audit CSV missing")
            audit = pd.read_csv(audit_path)
            matching = audit[(audit["variant"].astype(str) == variant) & (audit["policy"] == policy)]
            if len(matching) != 1:
                raise RuntimeError("XGBoost selected threshold audit not unique")
            actual = int(internal["false_positive"])
            claimed = int(matching.iloc[0]["realized_false_positives"])
            if actual != claimed:
                warnings.append({
                    "model": model,
                    "severity": "requires_recalculation",
                    "finding": "Internal confusion counts differ from reconstructed exact-FPR audit",
                    "metric_false_positives": actual,
                    "audit_false_positives": claimed,
                    "action": "Preserve original results; recompute exact threshold and scores from original float64 predictions before publication.",
                })

        family = pd.read_csv(paths["family"])
        family = family[(family["variant"].astype(str) == variant) & (family["threshold_policy"].astype(str) == policy)].copy()
        if len(family) != 3 or set(family["attack_family"].astype(str)) != {"Benign", "Botnet", "Infiltration"}:
            raise RuntimeError(f"{model}: selected family results incomplete")
        benign = family.loc[family["attack_family"] == "Benign"].iloc[0]
        botnet = family.loc[family["attack_family"] == "Botnet"].iloc[0]
        infiltration = family.loc[family["attack_family"] == "Infiltration"].iloc[0]
        check_close(float(benign["detection_rate_or_fpr"]), test["benign_fpr"], model + "/family benign FPR")
        if int(benign["predicted_attack_rows"]) != int(test["false_positive"]):
            raise RuntimeError(f"{model}: family benign FP doesn't match test confusion matrix")
        if int(botnet["predicted_attack_rows"]) + int(infiltration["predicted_attack_rows"]) != int(test["true_positive"]):
            raise RuntimeError(f"{model}: family attack TP doesn't match test confusion matrix")
        if sum(family["row_count"].astype(int)) != sum(int(test[k]) for k in ["true_negative", "false_positive", "false_negative", "true_positive"]):
            raise RuntimeError(f"{model}: family row count doesn't match test")

        summary_rows.append({
            "model": model, "backend": source.get("selected_backend", test.get("backend", "cuda_gpu" if model == "XGBoost" else "native_gpu")),
            "selected_variant": variant, "threshold_policy": policy, "threshold": float(source["selected_threshold"]),
            "internal_mcc": float(internal["mcc"]), "internal_fpr_from_confusion": float(internal["benign_fpr"]),
            **{metric: float(test[metric]) for metric in MAIN_METRICS},
            "true_negative": int(test["true_negative"]), "false_positive": int(test["false_positive"]),
            "false_negative": int(test["false_negative"]), "true_positive": int(test["true_positive"]),
            "botnet_recall": float(botnet["detection_rate_or_fpr"]),
            "infiltration_recall": float(infiltration["detection_rate_or_fpr"]),
            "test_rows": int(sum(int(test[k]) for k in ["true_negative", "false_positive", "false_negative", "true_positive"])),
        })
        family.insert(0, "model", model)
        family_rows.append(family)
        train = pd.read_csv(paths["training"])
        selected = train[train["variant"].astype(str) == variant]
        if len(selected) != 1:
            raise RuntimeError(f"{model}: training row missing or duplicated")
        row = selected.iloc[0]
        training_rows.append({
            "model": model, "backend": summary_rows[-1]["backend"], "variant": variant,
            "best_iteration": int(row["best_iteration"]),
            "training_seconds": float(row["training_seconds"]),
            "combined_inference_seconds": float(row["inference_seconds"]),
            "inference_scope": "internal_holdout + official_validation + strict_temporal_test (combined)",
        })

    comparison = pd.DataFrame(summary_rows).set_index("model").loc[ORDER].reset_index()
    families = pd.concat(family_rows, ignore_index=True)
    runtimes = pd.DataFrame(training_rows)
    integrity = pd.DataFrame(checks)
    warnings_df = pd.DataFrame(warnings, columns=["model", "severity", "finding", "metric_false_positives", "audit_false_positives", "action"])
    if len(set(comparison["test_rows"])) != 1:
        raise RuntimeError("Test row counts differ across models")

    ranking = comparison[["model"] + RANK_METRICS].copy()
    for metric in RANK_METRICS:
        ranking["rank_" + metric] = ranking[metric].rank(method="min", ascending=(metric == "benign_fpr"))
    ranking["mean_rank"] = ranking[["rank_" + x for x in RANK_METRICS]].mean(axis=1)
    ranking = ranking.sort_values(["mean_rank", "model"]).reset_index(drop=True)
    ranking["descriptive_position"] = np.arange(1, len(ranking) + 1)
    ranking["note"] = "Descriptive only: one strict temporal test set, no statistical significance inference"

    table_specs = [
        (comparison, "table_08e_01_tree_baseline_comparison.csv"),
        (families, "table_08e_02_tree_family_comparison.csv"),
        (ranking, "table_08e_03_tree_descriptive_ranking.csv"),
        (integrity, "table_08e_04_confusion_integrity_checks.csv"),
        (warnings_df, "table_08e_05_audit_findings.csv"),
        (runtimes, "table_08e_06_training_and_combined_inference.csv"),
    ]
    for frame, name in table_specs:
        save_table(frame, name, created)
    print("\nSTRICT-TEMPORAL BENCHMARK:")
    print(comparison[["model", "accuracy", "f1", "mcc", "auroc", "aupr", "recall", "benign_fpr", "botnet_recall", "infiltration_recall"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAUDIT WARNINGS:")
    print(warnings_df.to_string(index=False) if len(warnings_df) else "None")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 15,
                         "axes.titleweight": "bold", "axes.labelweight": "bold", "figure.dpi": 120})
    xs = np.arange(4)
    figure, ax = plt.subplots(figsize=(12, 6.8))
    for i, model in enumerate(ORDER):
        row = comparison.iloc[i]
        bars = ax.bar(xs + (i - 1) * 0.24, [row[k] for k in ["f1", "mcc", "auroc", "aupr"]],
                      width=0.23, label=model)
        fmt_bars(ax, bars)
    ax.set(xticks=xs, xticklabels=["F1-score", "MCC", "AUROC", "AUPR"],
           ylim=(0, 1.04), ylabel="Strict temporal score", title="Tree-Based Intrusion Detection: Strict Temporal Test")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_chart(figure, "figure_08e_01_temporal_metrics",
               "Figure 8E.1. Strict temporal-test F1-score, MCC, AUROC and AUPR of the selected XGBoost, LightGBM and CatBoost models. Bold numeric labels identify exact rounded values. No random-split scores are mixed into this comparison.", created)

    figure, ax = plt.subplots(figsize=(11.5, 6.3))
    xs = np.arange(3)
    for i, model in enumerate(ORDER):
        row = comparison.iloc[i]
        bars = ax.bar(xs + (i - 1) * 0.24,
                      [100 * row[k] for k in ["botnet_recall", "infiltration_recall", "benign_fpr"]],
                      width=0.23, label=model)
        fmt_bars(ax, bars, 2, "%")
    ax.set(xticks=xs, xticklabels=["Botnet detection", "Infiltration detection", "Benign false alarms"],
           ylim=(0, 107), ylabel="Percent of family", title="Unseen-Family Detection and False-Positive Cost")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_chart(figure, "figure_08e_02_family_generalization",
               "Figure 8E.2. Per-family detection on the chronological test set for previously unseen Botnet and Infiltration attacks, shown alongside benign false-positive rate. The marked Infiltration recall deficit motivates open-set temporal generalization research.", created)

    figure, ax = plt.subplots(figsize=(9.4, 6.2))
    for _, row in comparison.iterrows():
        ax.scatter(100 * row["benign_fpr"], row["mcc"], s=140)
        ax.annotate(row["model"] + f"  ({row['mcc']:.4f}, {100 * row['benign_fpr']:.3f}%)",
                    (100 * row["benign_fpr"], row["mcc"]), xytext=(8, 7),
                    textcoords="offset points", fontweight="bold", fontsize=9)
    ax.set(xlabel="Test benign false-positive rate (%)", ylabel="Test MCC",
           title="Operational Trade-Off: Detection Quality vs False Alarms")
    ax.margins(x=0.3, y=0.25); ax.grid(alpha=0.3)
    figure.tight_layout()
    save_chart(figure, "figure_08e_03_operational_tradeoff",
               "Figure 8E.3. Strict temporal-test MCC versus benign false-positive rate for tree-based baselines. Upper-left represents the preferred operational direction. Thresholds were selected on the training-internal holdout, not the temporal test set.", created)

    figure, ax = plt.subplots(figsize=(9.4, 6))
    bars = ax.bar(runtimes["model"], runtimes["training_seconds"])
    fmt_bars(ax, bars, 1, " s")
    ax.set(ylabel="Selected-variant training time (seconds)",
           title="Training Cost of Selected Tree-Based Baselines")
    ax.set_ylim(0, max(runtimes["training_seconds"]) * 1.19)
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_chart(figure, "figure_08e_04_training_cost",
               "Figure 8E.4. Reported selected-variant fitting time for XGBoost CUDA, LightGBM OpenCL and CatBoost GPU. These backend-specific timings are descriptive and are not a controlled hardware-throughput benchmark.", created)

    report = REPORTS / "block_08e_tree_baseline_comparison.md"
    report.write_text(
        "# Block 8E — Reproducible strict-temporal tree-model comparison\n\n"
        "No model was retrained. Inputs were verified against each model's saved JSON summary and selected family table.\n\n"
        "## Test comparison\n\n" +
        comparison[["model", "backend", "accuracy", "balanced_accuracy", "f1", "mcc", "auroc", "aupr", "recall", "benign_fpr", "botnet_recall", "infiltration_recall"]].to_markdown(index=False) +
        "\n\n## Descriptive rank (not a statistical test)\n\n" +
        ranking[["model", "mean_rank", "descriptive_position"]].to_markdown(index=False) +
        "\n\n## Integrity findings\n\n" +
        (warnings_df.to_markdown(index=False) if not warnings_df.empty else "No discrepancies detected.") +
        "\n\n## Limitations and publication conditions\n\n"
        "- One fixed strict-temporal test split is insufficient for statistical-significance or SOTA claims.\n"
        "- The training-internal stratified holdout is not a temporal-generalization benchmark.\n"
        "- A 1% FPR threshold selected internally does not guarantee <=1% FPR on shifted test traffic.\n"
        "- Reconstructed XGBoost threshold audit includes an off-by-one internal FP mismatch: recompute from original float64 probabilities before publication.\n"
        "- Model-ready Parquet and checkpoints are Git-ignored. Scripts, configs, summary results and figures are reproducible from repository metadata plus the Kaggle dataset.\n"
        "- The reported inference seconds are combined for three partitions, not exclusive per-test flow latency.\n",
        encoding="utf-8",
    )
    created.append(report)
    summary_path = METRICS / "metric_08e_tree_baseline_summary.json"
    metrics_summary = {
        "block": "08E", "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": ORDER, "test_rows": int(comparison["test_rows"].iloc[0]),
        "best_mcc_model": str(comparison.loc[comparison["mcc"].idxmax(), "model"]),
        "best_aupr_model": str(comparison.loc[comparison["aupr"].idxmax(), "model"]),
        "descriptive_rank_order": ranking["model"].tolist(),
        "warnings_count": len(warnings_df),
        "statistical_significance_claimed": False,
    }
    summary_path.write_text(json.dumps(metrics_summary, indent=2) + "\n", encoding="utf-8")
    created.append(summary_path)

    manifest_path = MANIFESTS / "block_08e_tree_baseline_comparison_manifest.json"
    manifest = {
        "block": "08E", "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in required},
        "outputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in created},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "test_rows": metrics_summary["test_rows"],
        "warnings": warnings,
        "retrained": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    created.append(manifest_path)

    completed = [b for b in registry.get("completed_blocks", []) if b.get("block") != "08E"]
    completed.append({"block": "08E", "name": "Audited tree-model strict temporal comparison",
                      "status": "completed", "manifest": str(manifest_path.relative_to(ROOT))})
    registry["completed_blocks"] = completed
    registry["next_block"] = {"block": "09A", "name": "GPU MLP temporal baseline", "status": "pending"}
    registry["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    for p in created + [REGISTRY]:
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output missing: {p}")
    print("\nAUDIT: 9/9 confusion matrices passed; family totals consistent.")
    print("FIGURES: 4 PNG at 600 DPI + 4 vector PDFs, with 4 caption files.")
    print("TABLES: 6 CSV files, report, summary JSON, checksummed manifest.")
    print("REGISTRY: 08E completed; next 09A.")
    print("Ready for GitHub synchronization.")


if __name__ == "__main__":
    main()
