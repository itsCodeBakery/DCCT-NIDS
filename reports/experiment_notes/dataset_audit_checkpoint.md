# CSE-CIC-IDS2018 Dataset Audit Checkpoint

Generated: 2026-07-10T06:02:02.979045+00:00

## Dataset

- Source: Kaggle-hosted CSE-CIC-IDS2018
- Files: 10 Parquet files
- Rows: 6,659,532
- Columns: 78
- Numeric columns: 77
- Nonnumeric columns: 1
- Detected classes: 15

## Schema harmonization

- Physical-type mismatch columns: 31
- Canonical schema: `configs/datasets/cse_cic_ids2018_canonical_schema.json`
- Dictionary-encoded labels are normalized to UTF-8 strings.

## Data-quality summary

- Total null cells: 0
- Total infinite cells: 0
- Constant columns: 8
- Near-constant columns: 10
- Duplicate sample size: 66396
- Estimated duplicate prevalence: 0.07831797096210615%

## Evaluation implications

The dataset is strongly class-imbalanced and chronologically heterogeneous. Primary model selection will therefore use Macro-F1 rather than accuracy. The study will also report MCC, balanced accuracy, per-class recall, benign false-positive rate, unknown-attack AUPR, AUROC, OSCR, and computational efficiency.

Random row-level splits will be treated as a secondary conventional benchmark. The principal evaluation will use leakage-resistant chronological and leave-one-attack-family-out protocols.

## Reproducibility artifacts

- Canonical schema configuration
- Global and per-file class distributions
- File-level quality summary
- Column-level quality summary
- Duplicate-prevalence estimate
- High-resolution PNG figures
- Vector PDF figures
- Manuscript-ready captions
- Machine-readable Block 2B manifest
