# Block 8D-B — CatBoost GPU Operational Baseline

Generated: 2026-07-10T10:16:44.052348+00:00

## Protocol

- Model: CatBoost
- Backend: Native GPU
- Feature set: Compact (32 features)
- Internal fitting rows: 3,572,049
- Internal holdout rows: 630,362
- Internal split: deterministic fine-grained-label stratification
- Official validation used for selection: No
- Strict temporal test used for selection: No
- GPU: Tesla T4, 15360 MiB
Tesla T4, 15360 MiB

## Class-weight variants

| variant     |   positive_class_weight |   best_iteration |   tree_count |   training_seconds |   inference_seconds |   threshold_maximum_internal_f1 |   threshold_fpr_at_most_1_percent |   threshold_fpr_at_most_0_1_percent |
|:------------|------------------------:|-----------------:|-------------:|-------------------:|--------------------:|--------------------------------:|----------------------------------:|------------------------------------:|
| unweighted  |                 1       |             1013 |         1013 |            55.1253 |             2.05386 |                        0.388538 |                       0.00032321  |                          0.00564006 |
| sqrt_weight |                 1.7143  |              993 |          993 |            53.9981 |             2.0028  |                        0.491499 |                       0.000522555 |                          0.00634362 |
| full_weight |                 2.93882 |             1419 |         1419 |            73.9947 |             2.66065 |                        0.757165 |                       0.000457898 |                          0.0085566  |

## Selected model

- Variant: full_weight
- Positive-class weight: 2.93882446
- Best iteration: 1419
- Threshold: 0.0004578983
- Selection rule: maximum internal MCC under exact benign FPR ≤ 1%

## Internal operational result

| variant     |   positive_class_weight | threshold_policy      | split            |   best_iteration |   tree_count |   training_seconds |   inference_seconds |   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |   auroc |   aupr |   specificity |   benign_fpr |   attack_fnr |   true_negative |   false_positive |   false_negative |   true_positive |   benign_fpr_exact_audit |
|:------------|------------------------:|:----------------------|:-----------------|-----------------:|-------------:|-------------------:|--------------------:|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|--------:|-------:|--------------:|-------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|-------------------------:|
| full_weight |                 2.93882 | fpr_at_most_1_percent | internal_holdout |             1419 |         1419 |            73.9947 |             2.66065 | 0.000457898 |   0.992546 |            0.995005 |    0.971476 |        1 | 0.985532 | 0.980699 |       1 |      1 |      0.990009 |   0.00999098 |            0 |          465625 |             4699 |                0 |          160038 |               0.00999098 |

## Official validation result

| variant     |   positive_class_weight | threshold_policy      | split               |   best_iteration |   tree_count |   training_seconds |   inference_seconds |   threshold |   accuracy |   balanced_accuracy |   precision |    recall |        f1 |       mcc |    auroc |      aupr |   specificity |   benign_fpr |   attack_fnr |   true_negative |   false_positive |   false_negative |   true_positive |
|:------------|------------------------:|:----------------------|:--------------------|-----------------:|-------------:|-------------------:|--------------------:|------------:|-----------:|--------------------:|------------:|----------:|----------:|----------:|---------:|----------:|--------------:|-------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|
| full_weight |                 2.93882 | fpr_at_most_1_percent | official_validation |             1419 |         1419 |            73.9947 |             2.66065 | 0.000457898 |   0.935858 |            0.516476 |    0.129725 | 0.0507784 | 0.0729874 | 0.0518502 | 0.469445 | 0.0624582 |      0.982174 |    0.0178261 |     0.949222 |         1069330 |            19408 |            54080 |            2893 |

## Strict temporal test result

| variant     |   positive_class_weight | threshold_policy      | split                |   best_iteration |   tree_count |   training_seconds |   inference_seconds |   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |     mcc |    auroc |     aupr |   specificity |   benign_fpr |   attack_fnr |   true_negative |   false_positive |   false_negative |   true_positive |
|:------------|------------------------:|:----------------------|:---------------------|-----------------:|-------------:|-------------------:|--------------------:|------------:|-----------:|--------------------:|------------:|---------:|---------:|--------:|---------:|---------:|--------------:|-------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|
| full_weight |                 2.93882 | fpr_at_most_1_percent | strict_temporal_test |             1419 |         1419 |            73.9947 |             2.66065 | 0.000457898 |    0.92668 |            0.844336 |    0.925814 | 0.703455 | 0.799461 | 0.76629 | 0.847927 | 0.803257 |      0.985218 |    0.0147819 |     0.296545 |          758547 |            11381 |            59874 |          142031 |

## Family-wise strict temporal result

| variant     | threshold_policy      | attack_family   |   row_count |   true_attack_rows |   predicted_attack_rows |   detection_rate_or_fpr |   mean_attack_probability |   median_attack_probability |   minimum_attack_probability |   maximum_attack_probability | interpretation        |
|:------------|:----------------------|:----------------|------------:|-------------------:|------------------------:|------------------------:|--------------------------:|----------------------------:|-----------------------------:|-----------------------------:|:----------------------|
| full_weight | fpr_at_most_1_percent | Benign          |      769928 |                  0 |                   11381 |               0.0147819 |               0.000166949 |                 4.89711e-07 |                  5.21152e-11 |                     0.9999   | false_positive_rate   |
| full_weight | fpr_at_most_1_percent | Botnet          |      144535 |             144535 |                  138613 |               0.959027  |               0.0185184   |                 0.0177297   |                  8.38375e-08 |                     0.41235  | attack_detection_rate |
| full_weight | fpr_at_most_1_percent | Infiltration    |       57370 |              57370 |                    3418 |               0.0595782 |               0.000988518 |                 3.66682e-07 |                  5.75138e-11 |                     0.996106 | attack_detection_rate |

## Interpretation

Botnet and Infiltration were not represented in the official
training period. Their family-specific recall therefore measures
temporal transfer to unseen attack families rather than ordinary
closed-set recognition.
