# Block 8B — GPU XGBoost Binary Temporal Baseline

Generated: 2026-07-10T07:35:17.812496+00:00

## Experimental scope

- Feature space: Compact (32 features)
- Training rows: 4,202,411
- Validation rows: 1,145,711
- Test rows: 971,833
- Test-period unseen attack families: Botnet and Infiltration
- GPU: Tesla T4, 15360 MiB
Tesla T4, 15360 MiB
- XGBoost: 3.2.0

## Leakage control

- Feature preprocessing was fitted on training data only.
- The classification threshold was selected on validation only.
- Test probabilities and labels were not used for training,
  early stopping, hyperparameter selection, or threshold selection.

## Validation metrics

|   threshold |   accuracy |   balanced_accuracy |   precision |   recall_detection_rate |      f1 |       mcc |   roc_auc |   average_precision_aupr |   specificity |   benign_false_positive_rate |   attack_false_negative_rate |   true_negative |   false_positive |   false_negative |   true_positive |
|------------:|-----------:|--------------------:|------------:|------------------------:|--------:|----------:|----------:|-------------------------:|--------------:|-----------------------------:|-----------------------------:|----------------:|-----------------:|-----------------:|----------------:|
|    0.475027 |   0.337422 |            0.548437 |   0.0563549 |                0.782757 | 0.10514 | 0.0455613 |   0.56355 |                0.0598491 |      0.314118 |                     0.685882 |                     0.217243 |          341992 |           746746 |            12377 |           44596 |

## Strict temporal test metrics

|   threshold |   accuracy |   balanced_accuracy |   precision |   recall_detection_rate |       f1 |      mcc |   roc_auc |   average_precision_aupr |   specificity |   benign_false_positive_rate |   attack_false_negative_rate |   true_negative |   false_positive |   false_negative |   true_positive |
|------------:|-----------:|--------------------:|------------:|------------------------:|---------:|---------:|----------:|-------------------------:|--------------:|-----------------------------:|-----------------------------:|----------------:|-----------------:|-----------------:|----------------:|
|    0.475027 |   0.459185 |            0.634108 |    0.268996 |                0.933385 | 0.417633 | 0.242589 |  0.590287 |                 0.256965 |      0.334831 |                     0.665169 |                    0.0666155 |          257796 |           512132 |            13450 |          188455 |

## Family-wise temporal test performance

| attack_family   |   row_count |   true_attack_rows |   predicted_attack_rows |   detection_rate_or_fpr |   mean_attack_probability |   median_attack_probability |   minimum_attack_probability |   maximum_attack_probability | interpretation        |
|:----------------|------------:|-------------------:|------------------------:|------------------------:|--------------------------:|----------------------------:|-----------------------------:|-----------------------------:|:----------------------|
| Benign          |      769928 |                  0 |                  512132 |                0.665169 |                  0.475968 |                    0.475027 |                     0.475021 |                     0.524976 | false_positive_rate   |
| Botnet          |      144535 |             144535 |                  144533 |                0.999986 |                  0.475167 |                    0.475027 |                     0.475027 |                     0.524563 | attack_detection_rate |
| Infiltration    |       57370 |              57370 |                   43922 |                0.765592 |                  0.476753 |                    0.475027 |                     0.475021 |                     0.524975 | attack_detection_rate |

## Interpretation

The strict temporal test measures attack-versus-benign detection under
distribution shift. Botnet and Infiltration were absent from training,
so their family-specific detection rates quantify zero-day transfer
rather than ordinary closed-set classification.
