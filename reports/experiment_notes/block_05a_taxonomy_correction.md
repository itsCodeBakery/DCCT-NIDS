# Block 5A — Attack Taxonomy Correction

Generated: 2026-07-10T06:27:19.578144+00:00

## Reason for correction

The initial heuristic mapping failed to recognize four dataset labels
containing the token `DoS attacks-` and the dataset's misspelled
`Infilteration` label. Those labels were temporarily assigned to
`Other Attack`.

The corrected workflow replaces heuristic substring matching with an
authoritative explicit taxonomy.

## Corrected mappings

| Fine-grained label | Corrected family |
|---|---|
| DoS attacks-Hulk | DoS |
| DoS attacks-GoldenEye | DoS |
| DoS attacks-Slowloris | DoS |
| DoS attacks-SlowHTTPTest | DoS |
| Infilteration | Infiltration |

## Final label spaces

- Binary classes: 2
- Attack-family classes: 7
- Fine-grained classes: 15
- Labels assigned to `Other Attack`: 0

## Corrected family counts

| attack_family   |   count |   fine_grained_class_count |   percentage |
|:----------------|--------:|---------------------------:|-------------:|
| Benign          | 4994157 |                          1 |  79.022      |
| DDoS            |  775955 |                          3 |  12.2779     |
| DoS             |  196523 |                          4 |   3.10956    |
| Botnet          |  144535 |                          1 |   2.28696    |
| Infiltration    |  113819 |                          1 |   1.80095    |
| Brute Force     |   94882 |                          4 |   1.50131    |
| Web Attack      |      84 |                          1 |   0.00132912 |

## Reproducibility policy

All future split-generation, preprocessing, training, evaluation,
confusion-matrix, open-set, and statistical-analysis blocks must load:

`configs/datasets/cse_cic_ids2018_attack_taxonomy.json`

No future block should independently reimplement the taxonomy.
