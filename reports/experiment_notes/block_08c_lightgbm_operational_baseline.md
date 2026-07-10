# Block 8C-B — Stable LightGBM Operational Baseline

Generated: 2026-07-10T09:31:16.431810+00:00

## Stability recovery

The original OpenCL configuration reached almost perfect internal
separation and subsequently triggered LightGBM's empty-child split
assertion. The recovered configuration used:

- `max_bin=63`;
- `num_leaves=63`;
- `max_depth=12`;
- `min_data_in_leaf=500`;
- no row or feature subsampling;
- GPU double-precision accumulation;
- maximum 600 boosting rounds.

A controlled CPU retry was permitted only after an OpenCL fatal error.

## Selected model

- Variant: unweighted
- Backend: gpu_opencl
- Positive-class weight: 1.00000000
- Best iteration: 445
- Threshold: 0.0000827547

## Internal operational result

| variant    | backend    |   positive_class_weight | threshold_policy      | split            |   best_iteration |   training_seconds |   inference_seconds |   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |    auroc |     aupr |   specificity |   benign_fpr |   attack_fnr |   true_negative |   false_positive |   false_negative |   true_positive |   benign_fpr_exact_audit |
|:-----------|:-----------|------------------------:|:----------------------|:-----------------|-----------------:|-------------------:|--------------------:|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|---------:|---------:|--------------:|-------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|-------------------------:|
| unweighted | gpu_opencl |                       1 | fpr_at_most_1_percent | internal_holdout |              445 |            284.994 |             31.5931 | 8.27547e-05 |   0.992544 |            0.995001 |    0.971476 | 0.999994 | 0.985528 | 0.980694 | 0.999998 | 0.999994 |      0.990009 |   0.00999098 |  6.24852e-06 |          465625 |             4699 |                1 |          160037 |               0.00999098 |

## Official validation result

| variant    | backend    |   positive_class_weight | threshold_policy      | split               |   best_iteration |   training_seconds |   inference_seconds |   threshold |   accuracy |   balanced_accuracy |   precision |    recall |        f1 |       mcc |   auroc |     aupr |   specificity |   benign_fpr |   attack_fnr |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------|:-----------|------------------------:|:----------------------|:--------------------|-----------------:|-------------------:|--------------------:|------------:|-----------:|--------------------:|------------:|----------:|----------:|----------:|--------:|---------:|--------------:|-------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|
| unweighted | gpu_opencl |                       1 | fpr_at_most_1_percent | official_validation |              445 |            284.994 |             31.5931 | 8.27547e-05 |    0.93686 |            0.519116 |    0.145284 | 0.0552367 | 0.0800417 | 0.0610222 |   0.515 | 0.070822 |      0.982995 |     0.017005 |     0.944763 |         1070224 |            18514 |            53826 |            3147 |

## Strict temporal test result

| variant    | backend    |   positive_class_weight | threshold_policy      | split                |   best_iteration |   training_seconds |   inference_seconds |   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |   auroc |     aupr |   specificity |   benign_fpr |   attack_fnr |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------|:-----------|------------------------:|:----------------------|:---------------------|-----------------:|-------------------:|--------------------:|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|--------:|---------:|--------------:|-------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|
| unweighted | gpu_opencl |                       1 | fpr_at_most_1_percent | strict_temporal_test |              445 |            284.994 |             31.5931 | 8.27547e-05 |   0.927767 |            0.844968 |    0.932407 | 0.703306 | 0.801813 | 0.769981 | 0.86602 | 0.816348 |       0.98663 |    0.0133701 |     0.296694 |          759634 |            10294 |            59904 |          142001 |

## Family-wise strict temporal result

| variant    | backend    | threshold_policy      | attack_family   |   row_count |   true_attack_rows |   predicted_attack_rows |   detection_rate_or_fpr |   mean_attack_probability |   median_attack_probability |   minimum_attack_probability |   maximum_attack_probability | interpretation        |
|:-----------|:-----------|:----------------------|:----------------|------------:|-------------------:|------------------------:|------------------------:|--------------------------:|----------------------------:|-----------------------------:|-----------------------------:|:----------------------|
| unweighted | gpu_opencl | fpr_at_most_1_percent | Benign          |      769928 |                  0 |                   10294 |               0.0133701 |               0.000286281 |                 5.41342e-07 |                  1.09733e-07 |                     0.997269 | false_positive_rate   |
| unweighted | gpu_opencl | fpr_at_most_1_percent | Botnet          |      144535 |             144535 |                  138609 |               0.959     |               0.0185788   |                 0.0316715   |                  6.05687e-07 |                     0.177117 | attack_detection_rate |
| unweighted | gpu_opencl | fpr_at_most_1_percent | Infiltration    |       57370 |              57370 |                    3392 |               0.059125  |               0.00177038  |                 5.69647e-07 |                  2.05146e-07 |                     0.998581 | attack_detection_rate |

## Training backend audit

| variant     | backend    |   positive_class_weight |   best_iteration |   training_seconds |   inference_seconds | gpu_error_before_fallback   |   threshold_maximum_internal_f1 |   threshold_fpr_at_most_1_percent |   threshold_fpr_at_most_0_1_percent |
|:------------|:-----------|------------------------:|-----------------:|-------------------:|--------------------:|:----------------------------|--------------------------------:|----------------------------------:|------------------------------------:|
| unweighted  | gpu_opencl |                 1       |              445 |            284.994 |             31.5931 |                             |                        0.46582  |                       8.27547e-05 |                          0.00249716 |
| sqrt_weight | gpu_opencl |                 1.7143  |              471 |            310.547 |             35.4327 |                             |                        0.604105 |                       9.33439e-05 |                          0.00274304 |
| full_weight | gpu_opencl |                 2.93882 |              471 |            300.841 |             31.7609 |                             |                        0.723178 |                       8.10215e-05 |                          0.00296088 |
