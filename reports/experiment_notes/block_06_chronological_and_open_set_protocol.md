# Block 6 — Chronological and Open-Set Protocol

Generated: 2026-07-10T06:36:52.820621+00:00

## Provenance reconstruction

Block 4 produced a deduplicated feature matrix without source-file
columns. Block 6 reproduced the exact full-row deduplication rule and
created a row-aligned provenance sidecar.

- Deduplicated rows: 6,319,955
- Provenance rows: 6,319,955
- Alignment mismatches: 0
- Provenance SHA-256: `46c3d718117850d221b9966aae8f0e5a3c4917f7cd022c478dae2cfbc71a95d9`

## Strict chronological protocol

- Training capture files: 1–6
- Validation capture files: 7–8
- Test capture files: 9–10

Known families represented in both training and testing:

- Benign

Temporally unseen families appearing in testing but absent from training:

- Botnet
- Infiltration

The strict protocol is therefore interpreted as a temporal
distribution-shift and unknown-class experiment—not as a conventional
closed-set split.

## Open-set protocol

The primary open-set design uses leave-one-attack-family-out folds.
For each fold:

1. all rows from the held-out attack family are removed from training;
2. all rows from the held-out attack family are removed from validation;
3. the family appears only in testing and is mapped to `Unknown`;
4. rejection thresholds are selected without using unknown test labels;
5. results are reported per family and macro-averaged across folds.

Eligible unknown families:

- Botnet
- Brute Force
- DDoS
- DoS
- Infiltration
- Web Attack

## Primary open-set metrics

- Known-class Macro-F1
- Unknown AUROC
- Unknown AUPR
- Unknown precision, recall and F1
- FPR at 95% unknown TPR
- OSCR
- Benign false-positive rate
