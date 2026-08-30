# Phase 2 six-model quantum ensemble benchmark

Run date: 2026-08-30  
Seeds: 42, 123, 2026  
Split: independent stratified 80/20 split per seed  
Training data: stratified 200-row pool from each training split; full held-out test split retained  
VQC budget: 100 full-batch Adam epochs, learning rate 0.05  
Weighting: one independent bootstrap per member, normalized inverse OOB probability MSE  
Voting: weighted soft vote of class-1/benign probabilities  
QSVM implementation: PennyLane fidelity matrix into `SVC(kernel="precomputed", probability=True)`

Test rows were not used for model fitting, OOB error, or weight assignment.

## Per-seed member results

| Seed | Type | Model | Accuracy | OOB MSE | Assigned weight | OOB rows |
|---:|---|---|---:|---:|---:|---:|
| 42 | VQC | `vqc_4q_3l_cnot` | 90.35% | 0.052009 | 21.29% | 71 |
| 42 | VQC | `vqc_3q_2l_cnot` | 92.98% | 0.075615 | 14.65% | 72 |
| 42 | VQC | `vqc_4q_2l_cz` | 92.98% | 0.060519 | 18.30% | 69 |
| 42 | VQC | `vqc_4q_3l_cz_reversed` | 91.23% | 0.073525 | 15.06% | 81 |
| 42 | QSVM | `qsvm_angle_4q` | 93.86% | 0.045506 | 24.34% | 67 |
| 42 | QSVM | `qsvm_amplitude_2q` | 73.68% | 0.174222 | 6.36% | 69 |
| 123 | VQC | `vqc_4q_3l_cnot` | 92.98% | 0.084128 | 16.34% | 70 |
| 123 | VQC | `vqc_3q_2l_cnot` | 92.98% | 0.059860 | 22.96% | 71 |
| 123 | VQC | `vqc_4q_2l_cz` | 90.35% | 0.110821 | 12.40% | 71 |
| 123 | VQC | `vqc_4q_3l_cz_reversed` | 94.74% | 0.077182 | 17.81% | 76 |
| 123 | QSVM | `qsvm_angle_4q` | 94.74% | 0.060336 | 22.78% | 75 |
| 123 | QSVM | `qsvm_amplitude_2q` | 75.44% | 0.177896 | 7.73% | 80 |
| 2026 | VQC | `vqc_4q_3l_cnot` | 90.35% | 0.071594 | 13.70% | 72 |
| 2026 | VQC | `vqc_3q_2l_cnot` | 89.47% | 0.048604 | 20.18% | 75 |
| 2026 | VQC | `vqc_4q_2l_cz` | 91.23% | 0.055362 | 17.72% | 69 |
| 2026 | VQC | `vqc_4q_3l_cz_reversed` | 90.35% | 0.060772 | 16.14% | 71 |
| 2026 | QSVM | `qsvm_angle_4q` | 94.74% | 0.036741 | 26.70% | 66 |
| 2026 | QSVM | `qsvm_amplitude_2q` | 72.81% | 0.176130 | 5.57% | 81 |

## Across-seed summary

| Type | Model | Accuracy mean | Accuracy range | Weight mean | Weight range |
|---|---|---:|---:|---:|---:|
| VQC | `vqc_4q_3l_cnot` | 91.23% | 90.35-92.98% | 17.11% | 13.70-21.29% |
| VQC | `vqc_3q_2l_cnot` | 91.81% | 89.47-92.98% | 19.26% | 14.65-22.96% |
| VQC | `vqc_4q_2l_cz` | 91.52% | 90.35-92.98% | 16.14% | 12.40-18.30% |
| VQC | `vqc_4q_3l_cz_reversed` | 92.11% | 90.35-94.74% | 16.34% | 15.06-17.81% |
| QSVM | `qsvm_angle_4q` | 94.44% | 93.86-94.74% | 24.60% | 22.78-26.70% |
| QSVM | `qsvm_amplitude_2q` | 73.98% | 72.81-75.44% | 6.55% | 5.57-7.73% |

The amplitude/2-qubit QSVM was the expected weak diversity member and received the smallest weight on every seed.

## Ensemble versus best single member

| Seed | Ensemble accuracy | Best single member | Best-single accuracy | Ensemble beat/tied best? | Paired t | Two-sided p |
|---:|---:|---|---:|---|---:|---:|
| 42 | 94.74% | `qsvm_angle_4q` | 93.86% | Yes | 0.4456 | 0.656711 |
| 123 | 93.86% | `vqc_4q_3l_cz_reversed` (tied by angle QSVM) | 94.74% | No | -0.4456 | 0.656711 |
| 2026 | 93.86% | `qsvm_angle_4q` | 94.74% | No | -0.5757 | 0.565993 |

Across-seed ensemble accuracy: **94.15% mean, 93.86-94.74% range**.  
Across-seed best-single accuracy: **94.44% mean, 93.86-94.74% range**.

The ensemble did not beat the best single model on mean accuracy; it trailed by 0.29 percentage points. None of the per-seed paired differences was statistically significant. Weights were not tuned against the test set to force the roadmap target.

## Carried Phase 1 same-four-feature classical means (not rerun)

| Model | Mean accuracy |
|---|---:|
| Logistic Regression | 93.57% |
| Random Forest | 93.27% |
| XGBoost | 93.27% |
| SVM | 93.86% |

## Verification output

```text
platform win32 -- Python 3.14.0, pytest-9.1.1
collected 39 items
39 passed, 2 warnings in 8.46s
```

The warnings were the expected scikit-learn 1.9 `SVC(probability=True)` deprecation from D-18 and a OneDrive pytest-cache warning. Neither was a test or model failure.
