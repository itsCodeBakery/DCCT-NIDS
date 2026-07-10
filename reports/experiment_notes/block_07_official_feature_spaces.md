# Block 7 — Official Feature Spaces

Generated: 2026-07-10T07:17:19.892693+00:00

## Leakage-control scope

All feature selection statistics were fitted using only the strict
chronological training partition corresponding to source-file orders
1–6.

- Training rows: 4,202,411
- Training statistical sample: 350,000
- Validation rows used for selection: 0
- Test rows used for selection: 0
- Random seed: 42

## Frozen feature spaces

| Feature space | Count | Definition |
|---|---:|---|
| Full | 67 | All leakage-safe primary features |
| Reduced | 51 | Full set after near-perfect training-only correlation pruning |
| Compact | 32 | Top training-only mutual-information features from Reduced |

## Redundancy policy

A pair is considered near-perfectly redundant when either:

- absolute Pearson correlation ≥ 0.9999; or
- absolute Spearman correlation ≥ 0.9999.

When one feature must be removed, the retained feature is selected
using the following deterministic priority:

1. higher attack-family mutual information;
2. higher training variance;
3. lexicographic order as a final tie-break.

Removed features: 16

- `Active Mean`
- `Avg Bwd Segment Size`
- `Avg Fwd Segment Size`
- `Bwd Packets Length Total`
- `Flow Duration`
- `Flow IAT Max`
- `Flow IAT Min`
- `Fwd IAT Std`
- `Fwd Packets Length Total`
- `Idle Max`
- `Packet Length Variance`
- `RST Flag Count`
- `SYN Flag Count`
- `Subflow Bwd Packets`
- `Subflow Fwd Packets`
- `Total Fwd Packets`

## Compact feature policy

The Compact set contains the top 32 features
ranked by attack-family mutual information after redundancy pruning.

1. `Fwd Header Length`
2. `Subflow Fwd Bytes`
3. `Fwd Packet Length Mean`
4. `Fwd Packet Length Max`
5. `Bwd Packet Length Mean`
6. `Bwd Packet Length Max`
7. `Subflow Bwd Bytes`
8. `Init Fwd Win Bytes`
9. `Fwd Packet Length Std`
10. `Bwd Header Length`
11. `Bwd Packet Length Std`
12. `Packet Length Std`
13. `Packet Length Mean`
14. `Avg Packet Size`
15. `Fwd IAT Mean`
16. `Fwd IAT Total`
17. `Fwd IAT Max`
18. `Total Backward Packets`
19. `Packet Length Max`
20. `Fwd Packets/s`
21. `Flow IAT Mean`
22. `Init Bwd Win Bytes`
23. `Bwd Packets/s`
24. `Bwd IAT Min`
25. `Flow Packets/s`
26. `Fwd IAT Min`
27. `Flow IAT Std`
28. `Bwd IAT Total`
29. `Bwd IAT Max`
30. `Flow Bytes/s`
31. `Bwd IAT Mean`
32. `Bwd IAT Std`

## Preprocessing policy

- Replace infinities with missing values.
- Fit median imputation on training data only.
- Apply `log1p` only to nonnegative training features with absolute
  skewness ≥ 2.0.
- Fit RobustScaler independently inside each training fold.
- Never reuse global validation or test statistics.
- Use identical feature sets for all competing models in a given
  ablation experiment.

## Required ablation study

Every principal model should be evaluated with:

1. Full feature set;
2. Reduced feature set;
3. Compact feature set.

The final paper should report predictive performance, inference time,
training time, peak memory, model size, and feature count.
