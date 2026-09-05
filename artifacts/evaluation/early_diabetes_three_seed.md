# Early-diabetes module: retained three-seed evidence

Generated on 2026-09-04 with the pinned Python 3.14 stack. The machine-readable
record is `early_diabetes_three_seed.json`.

## Fixed protocol

- Dataset: UCI Early Stage Diabetes Risk Prediction, 520 rows, CC BY 4.0
- Seeds: 42, 123, 2026; independent stratified 80/20 splits
- Preprocessing fit on training rows only: median imputation, standard scaling,
  `SelectKBest(f_classif, k=4)`, inward quantum scaling
- Quantum training pool: 20 stratified rows per seed
- Four data-reuploading VQCs, 100 epochs each; angle/4-qubit and
  amplitude/2-qubit fidelity-kernel SVMs
- OOB inverse-MSE weights and weighted soft voting; untouched 104-row test set
- Condition-positive class: UCI `Positive`, displayed as “positive screening
  signal” (class 0)

## Actual results

| Measure | Mean | Range |
|---|---:|---:|
| Six-model ensemble accuracy | 83.01% | 76.92–92.31% |
| Condition sensitivity | 81.25% | 67.19–96.88% |
| Specificity | 85.83% | 80.00–92.50% |
| Full-feature Logistic Regression accuracy | 90.06% | 83.65–97.12% |
| Full-feature Random Forest accuracy | 95.19% | 93.27–98.08% |
| Full-feature XGBoost accuracy | 92.95% | 88.46–96.15% |
| Full-feature SVM accuracy | 95.51% | 94.23–98.08% |
| Same-four Logistic Regression accuracy | 86.54% | 81.73–92.31% |
| Same-four Random Forest accuracy | 85.58% | 81.73–89.42% |
| Same-four XGBoost accuracy | 86.54% | 81.73–92.31% |
| Same-four SVM accuracy | 86.54% | 81.73–92.31% |

The weakest VQC (`vqc_4q_3l_cnot`) measured 58.01% mean accuracy with a
39.42–69.23% range. Its cost still decreased on every seed, and OOB weighting
reduced its weight to almost zero on the worst seed. The full ensemble itself
remained variable. Nothing was retuned or removed after seeing test outcomes.

## Interpretation boundary

This is useful evidence that the architecture can train a second, differently
shaped biomedical dataset end to end. It is not evidence that quantum models
beat classical baselines: they did not here. It is also not clinical or
external validation. The data are a small, single-center symptom questionnaire
study from Bangladesh; strong symptom variables can inflate apparent
performance, the binary sex/gender field is limited, and no Indian-cohort or
prospective result exists. Present it as an honest scalability demonstration.
