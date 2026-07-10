# Block 5 — Feature and Label Policy

Generated: 2026-07-10T06:17:32.256044+00:00

## Dataset integrity

- Deduplicated rows: 6,319,955
- Canonical columns: 78
- SHA-256: `1a4e07447a721dc6a5d148b98fdf2565076407cb9ebbb51d311162092bc5335b`

## Label spaces

- Binary classes: 2
- Attack-family classes: 6
- Fine-grained classes: 15

## Primary feature policy

- All canonical columns: 78
- Hard exclusions: 9
- Near-constant sensitivity exclusions: 2
- Candidate features: 69
- Primary features: 67

### Hard exclusions

- `Bwd Avg Bulk Rate`
- `Bwd Avg Bytes/Bulk`
- `Bwd Avg Packets/Bulk`
- `Bwd PSH Flags`
- `Bwd URG Flags`
- `Fwd Avg Bulk Rate`
- `Fwd Avg Bytes/Bulk`
- `Fwd Avg Packets/Bulk`
- `Label`

### Near-constant sensitivity exclusions

- `CWE Flag Count`
- `Fwd URG Flags`

## Leakage control

The target label, direct identifiers, timestamps, constant fields, and primary near-constant fields are not provided to the primary classifiers. All imputation, transformations, scaling, feature selection, and class-weight estimation must be fitted using training data only.

## Evaluation roles

1. Binary detection establishes attack-versus-benign performance.
2. Family-level classification is the principal closed-set and open-set protocol.
3. Fine-grained classification evaluates rare-class sensitivity.
4. Held-out attack families are mapped to `Unknown` only during test evaluation.
