# Phase 3 real-model explainability verification

Run date: 2026-08-30  
Seed: 42  
Model: actual six-member Phase 2 ensemble  
VQC budget: 100 full-batch Adam epochs for every VQC  
Training pool: stratified 20-row smoke-test pool  
Explanation background: 10 training rows  
SHAP budget: 16 perturbation samples  
LIME budget: 250 neighbourhood samples

This is an integration check of real PennyLane VQC and fidelity-kernel QSVM
artifacts, not a replacement benchmark. The intentionally small training and
background samples make the check repeatable during development. D-17's
production-like measurement remains the latency expectation for the UI:
QSVM-inclusive explanations are explicitly presented as taking at least 70
seconds.

## Feature-name and boundary checks

- Selected names passed verbatim to SHAP and LIME: `mean concave points`,
  `worst radius`, `worst perimeter`, `worst concave points`.
- Generic `PC1` or `feature_0` names are rejected by the service and tests.
- Observed training quantum range: `[0.000001, 3.141591653589793]`, matching
  the inward D-21 range.

## VQC-heavy patient

Test position 46 (dataset row 307), true label 1/benign. The fast VQC scope
used all four VQC members and predicted benign probability 0.961058.

- Combined SHAP + LIME time: 3.4068s
- SHAP additivity residual: 0.0
- LIME surrogate additivity residual: 0.0
- Top-four overlap: 4/4
- Comparable-direction agreement: 4/4

| Rank | SHAP feature | SHAP contribution toward benign | LIME feature | LIME patient contribution toward benign |
|---:|---|---:|---|---:|
| 1 | mean concave points | +0.164992 | mean concave points | +0.182345 |
| 2 | worst perimeter | +0.108203 | worst concave points | +0.124179 |
| 3 | worst concave points | +0.066422 | worst perimeter | +0.099336 |
| 4 | worst radius | +0.043317 | worst radius | +0.051795 |

## QSVM-heavy patient

Test position 19 (dataset row 242), true label 1/benign. The diagnostic scope
used both QSVM members and predicted benign probability 0.962004.

- SHAP time: 7.8214s
- SHAP additivity residual: 0.0
- Ranked contributions: `mean concave points` +0.124399, `worst radius`
  +0.093097, `worst perimeter` +0.055349, `worst concave points` +0.038435.

## Full six-model explanation

The explicit slow opt-in explained the same QSVM-heavy patient through all six
models using their frozen OOB weights. It predicted benign probability
0.833157.

- SHAP time under the reduced smoke-test settings: 9.1341s
- SHAP additivity residual: 0.0
- Ranked contributions: `mean concave points` +0.099775, `worst perimeter`
  +0.071255, `worst radius` +0.056675, `worst concave points` +0.013996.

All top features are real selected clinical measurements and are expected
breast-cancer predictors. Nothing resembling a preprocessing component or
generic feature identifier topped any result. The observed rankings therefore
do not indicate a preprocessing-name leak.

## Test output

```text
platform win32 -- Python 3.14.0, pytest-9.1.1
collected 50 items
50 passed, 6 warnings in 6.29s
```

The six warnings were three SHAP/Matplotlib pending deprecations, the two
expected D-18 `SVC(probability=True)` warnings, and the existing OneDrive
pytest-cache warning. The mandatory D-02 focused run also passed:

```text
tests/test_circuits.py::test_cost_decreases_during_training PASSED
1 passed, 9 deselected, 1 warning in 4.18s
```
