# Phase 7 evaluation summary — SIH26139 evidence completion

Generated from the retained machine-readable reports on 2026-09-03. Malignant (`0`) is the positive clinical condition throughout. Values below are measured, not estimates.

## Experimental records

- `classical_three_seed_metrics.json`: stratified 80/20 splits at seeds 42, 123, and 2026; all four classical algorithms in both feature configurations; 3.639 seconds total on this run.
- `five_fold_cross_validation.json`: five stratified folds at seed 42; imputer, standard scaler, ANOVA selector, and quantum-range scaler refitted inside every training fold; four VQCs trained for 100 epochs; six-model quantum pool limited to 20 rows; 549.694 seconds total on this run.
- All five folds independently selected `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`.
- Timing is machine/workload dependent. Quantum member time includes its fit, OOB work, and internal held-out evaluation. Ensemble time covers the complete six-model pipeline, OOB weighting, and internal held-out evaluation.

## Three-seed classical evidence

| Configuration / model | Accuracy | Malignant precision | Malignant sensitivity | Specificity | Malignant F1 | ROC-AUC | Mean fit | Mean predict | Total FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full 30 / Logistic Regression | 97.66% | 97.58% | 96.03% | 98.61% | 96.80% | 0.9940 | 3.88 ms | 0.47 ms | 5 |
| Full 30 / Random Forest | 95.91% | 93.82% | 95.24% | 96.30% | 94.49% | 0.9941 | 391.44 ms | 42.85 ms | 6 |
| Full 30 / XGBoost | 97.08% | 96.08% | 96.03% | 97.69% | 95.99% | 0.9964 | 180.30 ms | 2.70 ms | 5 |
| Full 30 / SVM | 97.95% | 96.93% | 97.62% | 98.15% | 97.25% | 0.9966 | 3.53 ms | 2.53 ms | 3 |
| Same 4 / Logistic Regression | 93.57% | 88.30% | 95.24% | 92.59% | 91.60% | 0.9915 | 4.13 ms | 0.72 ms | 6 |
| Same 4 / Random Forest | 93.27% | 89.32% | 92.86% | 93.52% | 91.04% | 0.9852 | 343.68 ms | 33.28 ms | 9 |
| Same 4 / XGBoost | 93.27% | 88.76% | 93.65% | 93.06% | 91.12% | 0.9852 | 39.68 ms | 2.07 ms | 8 |
| Same 4 / SVM | 93.86% | 89.57% | 94.44% | 93.52% | 91.91% | 0.9902 | 2.26 ms | 1.68 ms | 7 |

The JSON artifact retains each seed's clinically oriented confusion matrix (`TP=malignant correctly detected`, `FN=malignant predicted benign`, `FP=benign predicted malignant`, `TN=benign correctly detected`) and full-precision metrics.

## Leakage-safe five-fold evidence

| Paradigm / model | Mean accuracy (range) | Malignant precision | Malignant sensitivity | Specificity | Malignant F1 | ROC-AUC | Mean measured time | Total FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical full 30 / Logistic Regression | 97.37% (94.74–99.12%) | 98.63% | 94.36% | 99.16% | 96.33% | 0.9953 | 0.005 s | 12 |
| Classical full 30 / Random Forest | 95.61% (92.98–96.49%) | 94.51% | 93.89% | 96.64% | 94.06% | 0.9894 | 0.394 s | 13 |
| Classical full 30 / XGBoost | 96.66% (94.74–99.12%) | 96.29% | 94.82% | 97.77% | 95.44% | 0.9936 | 0.149 s | 11 |
| Classical full 30 / SVM | 97.72% (94.74–99.12%) | 98.10% | 95.76% | 98.88% | 96.88% | 0.9945 | 0.004 s | 9 |
| Classical same 4 / Logistic Regression | 94.03% (90.35–95.61%) | 93.41% | 90.59% | 96.08% | 91.78% | 0.9887 | 0.004 s | 20 |
| Classical same 4 / Random Forest | 93.33% (88.60–95.58%) | 91.79% | 90.60% | 94.96% | 90.89% | 0.9783 | 0.329 s | 20 |
| Classical same 4 / XGBoost | 94.03% (91.23–94.74%) | 93.09% | 91.05% | 95.80% | 91.84% | 0.9745 | 0.044 s | 19 |
| Classical same 4 / SVM | 93.85% (89.47–96.46%) | 93.48% | 90.12% | 96.08% | 91.50% | 0.9844 | 0.003 s | 21 |
| VQC 4q/3l/CNOT | 91.92% (89.47–92.98%) | 92.28% | 85.85% | 95.53% | 88.72% | 0.9838 | 23.769 s | 30 |
| VQC 3q/2l/CNOT | 86.12% (72.81–93.86%) | 92.26% | 70.85% | 95.26% | 76.63% | 0.9627 | 18.009 s | 62 |
| VQC 4q/2l/CZ | 83.31% (75.44–87.72%) | 95.65% | 59.99% | 97.22% | 71.45% | 0.9436 | 23.843 s | 85 |
| VQC 4q/3l/CZ reversed | 89.45% (87.72–92.11%) | 93.51% | 77.82% | 96.37% | 84.62% | 0.9740 | 29.691 s | 47 |
| QSVM angle/4q | 91.22% (85.09–95.61%) | 97.24% | 79.26% | 98.31% | 86.13% | 0.9864 | 5.351 s | 44 |
| QSVM amplitude/2q | 54.61% (29.20–62.28%) | 11.23% | 19.52% | 75.67% | 14.15% | 0.4965 | 7.985 s | 171 |
| Six-model OOB ensemble | 91.57% (87.72–93.86%) | 93.39% | 83.50% | 96.36% | 87.96% | 0.9810 | 108.651 s | 35 |

## Interpretation

The results do not support a raw-accuracy quantum-superiority claim. Full-feature classical SVM was strongest in the fold study. The same-four classical comparison remained closer to the best quantum models, which independently reproduces D-14's feature-access finding.

The amplitude/2-qubit QSVM was much less stable here than in the separate 200-row Phase 2 benchmark (54.61% versus the earlier 73.98% mean). That is a real warning about a tiny training pool and this feature map, not evidence of a software crash: its ROC-AUC averaged approximately random, and it must be described as an unstable diversity member under this configuration.

The six-model ensemble remained above 87% in every fold and did not beat full-feature classical ML. No model or weight was retuned or removed after viewing these folds. External-cohort, prospective, subgroup, calibration, noise, and real-QPU validation remain future work.
