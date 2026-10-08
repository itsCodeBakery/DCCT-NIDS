# Block 8E — Reproducible strict-temporal tree-model comparison

No model was retrained. Inputs were verified against each model's saved JSON summary and selected family table.

## Test comparison

| model    | backend    |   accuracy |   balanced_accuracy |       f1 |      mcc |    auroc |     aupr |   recall |   benign_fpr |   botnet_recall |   infiltration_recall |
|:---------|:-----------|-----------:|--------------------:|---------:|---------:|---------:|---------:|---------:|-------------:|----------------:|----------------------:|
| XGBoost  | cuda_gpu   |   0.928366 |            0.845519 | 0.803238 | 0.772022 | 0.864978 | 0.80104  | 0.703777 |    0.0127376 |        0.960051 |             0.0581314 |
| LightGBM | gpu_opencl |   0.927767 |            0.844968 | 0.801813 | 0.769981 | 0.86602  | 0.816348 | 0.703306 |    0.0133701 |        0.959    |             0.059125  |
| CatBoost | native_gpu |   0.92668  |            0.844336 | 0.799461 | 0.76629  | 0.847927 | 0.803257 | 0.703455 |    0.0147819 |        0.959027 |             0.0595782 |

## Descriptive rank (not a statistical test)

| model    |   mean_rank |   descriptive_position |
|:---------|------------:|-----------------------:|
| XGBoost  |       1.625 |                      1 |
| LightGBM |       1.875 |                      2 |
| CatBoost |       2.5   |                      3 |

## Integrity findings

| model   | severity               | finding                                                             |   metric_false_positives |   audit_false_positives | action                                                                                                                |
|:--------|:-----------------------|:--------------------------------------------------------------------|-------------------------:|------------------------:|:----------------------------------------------------------------------------------------------------------------------|
| XGBoost | requires_recalculation | Internal confusion counts differ from reconstructed exact-FPR audit |                     4704 |                    4703 | Preserve original results; recompute exact threshold and scores from original float64 predictions before publication. |

## Limitations and publication conditions

- One fixed strict-temporal test split is insufficient for statistical-significance or SOTA claims.
- The training-internal stratified holdout is not a temporal-generalization benchmark.
- A 1% FPR threshold selected internally does not guarantee <=1% FPR on shifted test traffic.
- Reconstructed XGBoost threshold audit includes an off-by-one internal FP mismatch: recompute from original float64 probabilities before publication.
- Model-ready Parquet and checkpoints are Git-ignored. Scripts, configs, summary results and figures are reproducible from repository metadata plus the Kaggle dataset.
- The reported inference seconds are combined for three partitions, not exclusive per-test flow latency.
