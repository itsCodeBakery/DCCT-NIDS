# DCCT-NIDS

**Drift-Calibrated Cross-View Contrastive Network Intrusion Detection System**

A reproducible research repository for leakage-resistant, chronological, open-world network intrusion detection using CSE-CIC-IDS2018 and related public benchmark datasets.

## Research objectives

- Audit and clean CSE-CIC-IDS2018 without train/test leakage.
- Compare random and chronological evaluation protocols.
- Evaluate known-attack classification and unseen-attack rejection.
- Develop cross-view flow/context contrastive representations.
- Calibrate open-set thresholds under chronological distribution shift.
- Report statistical significance, calibration, efficiency, and reproducibility.

## Primary dataset

Kaggle input path:

```text
/kaggle/input/datasets/dhoogla/csecicids2018
```

Raw datasets are never committed to Git. Lightweight tables, figures, configurations, manifests, and source code are versioned. Large checkpoints and intermediate arrays remain excluded and may later be published as release artifacts.

## Repository layout

```text
configs/          Dataset, model, and experiment configurations
data/             Small manifests and metadata only
notebooks/        Numbered Kaggle/Jupyter notebooks
src/dcct_nids/    Reusable Python package
scripts/          Command-line and repository utilities
tests/            Automated tests
outputs/          Metrics, tables, figures, logs, and manifests
reports/          Manuscript assets and experiment documentation
```

## Reproducibility rules

Every experiment must record its configuration, seed, package versions, dataset manifest/checksums, split definition, metrics, execution time, and Git commit SHA. Predictions and embeddings must either be versioned when lightweight or referenced through an immutable artifact manifest.

## Figure standard

Publication figures must be exported as high-resolution PNG (minimum 300 DPI) and vector PDF/SVG where supported. Titles and data labels must be bold, axes must include meaningful definitions or units, and each figure must have a manuscript-ready caption.

## Git identity used in Kaggle

```bash
git config user.name "itscodebakery"
git config user.email "ashayan29@gmail.com"
```

## Current phase

Phase 1 — repository initialization and dataset audit.
