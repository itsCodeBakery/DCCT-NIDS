# Block 8A — Compact Model-Ready Dataset

Generated: 2026-07-10T07:25:52.350631+00:00

## Runtime

- NVIDIA GPU: Tesla T4, 15360 MiB, 580.159.04, 7.5
Tesla T4, 15360 MiB, 580.159.04, 7.5
- PyTorch CUDA available: True
- DuckDB version: 1.3.2

GPU acceleration is reserved for model training. Parquet processing,
quantile fitting, and dataset materialization are performed with
DuckDB because these are primarily storage and CPU-memory operations.

## Feature space

- Feature set: Compact
- Features: 32
- log1p-transformed features: 20

## Leakage-safe preprocessing

Every preprocessing parameter was fitted using strict chronological
training rows only.

1. Non-finite values were treated as missing.
2. Missing values were replaced with the training median.
3. Eligible nonnegative skewed features received `log1p`.
4. Each feature was centered by its transformed training median.
5. Each feature was divided by its transformed training IQR.
6. Zero-IQR features used a scale denominator of 1.
7. Outputs were stored as float32.

Validation and test feature statistics were not used.

## Materialized splits

| split      |    rows |   columns |   row_groups |   file_size_mb |   invalid_feature_cells |   missing_label_rows |
|:-----------|--------:|----------:|-------------:|---------------:|------------------------:|---------------------:|
| train      | 4202411 |        39 |           42 |       221.874  |                       0 |                    0 |
| validation | 1145711 |        39 |           12 |        68.518  |                       0 |                    0 |
| test       |  971833 |        39 |           10 |        54.0928 |                       0 |                    0 |

## Scientific caution

The strict chronological test partition contains attack families
that do not occur in training. It must therefore be evaluated as a
temporal open-set/distribution-shift benchmark rather than as a
standard closed-set multiclass benchmark.
