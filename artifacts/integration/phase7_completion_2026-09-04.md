# SIH26139 evidence closure — September 4, 2026

This record closes the four implementation gaps carried from D-26 and maps the resulting project to the main SIH26139 objectives. It is an evidence record for a simulator-based research prototype, not a medical-device or quantum-advantage claim.

## Implemented gap closures

- Clinical reporting now treats **malignant disease as the positive condition** and retains accuracy, malignant precision, malignant sensitivity/recall, specificity, malignant F1, clinical confusion counts, ROC-AUC, fit time, and prediction time for Logistic Regression, Random Forest, XGBoost, and SVM in both full-30 and same-four configurations.
- The five-fold generalization study uses stratified folds and refits the imputer, standard scaler, ANOVA feature selector, and quantum-range scaler only on each fold's training rows. Every fold evaluates all eight classical fits, four 100-epoch VQCs, both quantum-kernel SVMs, and their frozen OOB-weighted ensemble.
- `/explain` and both frontends now expose classical full-feature and same-four SHAP/LIME alongside the quantum fast/full scopes. Explanations retain real clinical feature names and return attribution only, never a second confidence number.
- The live model is tied to `artifacts/models/runtime_manifest.json` through configuration ID `qtrace-wbcd-s42-e100-q20-v1`. `/health` and both interfaces disclose its seed, epoch budget, quantum device, quantum pool, classical pool, and deterministic-refit loading mode.
- Exact named 30-feature CSV intake is implemented through `/ingest`, including schema validation and observed-range warnings. The judge-facing interface exposes the retained malignant metrics, generalization comparison, and efficiency evidence.
- PennyLane device selection is configurable. The regression suite exercises the VQC on both `lightning.qubit` and `default.qubit`; real-QPU execution remains a future validation step.

## Actual automated verification

Windows, pinned Python 3.14 virtual environment:

```text
87 passed, 5 warnings in 21.27s
```

Mandatory circuit regression, run separately after the quantum device/timing changes:

```text
1 passed, 10 deselected in 8.76s
```

The five warnings are three upstream SHAP/Matplotlib pending deprecations and two expected scikit-learn `SVC(probability=True)` FutureWarnings documented by D-18. Additional checks:

```text
pip check: No broken requirements found.
python -m compileall -q backend dev-dashboard tests scripts: passed
node --check frontend/app.js: passed
DOCX accessibility audit: 0 high, 0 medium, 0 low findings
DOCX table geometry: all 10 tables internally consistent
```

## Retained measured evidence

The machine-readable reports are:

- `artifacts/evaluation/classical_three_seed_metrics.json`
- `artifacts/evaluation/five_fold_cross_validation.json`
- `artifacts/evaluation/phase7_evaluation_summary.md`

Across seeds 42, 123, and 2026, full-feature classical mean accuracy ranged from 95.91% to 97.95%; same-four classical mean accuracy ranged from 93.27% to 93.86%. The strongest full-feature result was SVM at 97.95% mean accuracy, 97.62% malignant sensitivity, and 98.15% specificity.

In the leakage-safe five-fold study, the six-model quantum ensemble averaged 91.57% accuracy, 83.50% malignant sensitivity, 96.36% specificity, and 0.981 ROC-AUC. Full-feature SVM averaged 97.72% accuracy, 95.76% malignant sensitivity, and 98.88% specificity. The amplitude-embedding two-qubit QSVM was unstable under the deliberately small 20-row quantum pool (54.61% mean accuracy, 29.20%–62.28% range). These numbers are reported as limitations, not tuned away after validation, and they do not support a quantum-superiority claim.

## Actual browser verification

The production FastAPI application was started from the pinned local environment at `http://127.0.0.1:8000/`. The runtime reached ready with six quantum members, eight classical fits, seed 42, 100 epochs, 20 quantum rows, 455 classical rows, and the real names `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`.

Verified interactions:

- The retained five-fold table rendered with accuracy, malignant sensitivity, specificity, and ROC-AUC.
- A one-row CSV with all 30 exact names validated successfully and was identified as an unlabelled uploaded record.
- The uploaded benchmark row produced quantum benign at 64.7% model confidence, full-feature classical malignant at 63.4%, and same-four classical benign at 60.7%. These are one-record model confidences, not accuracy or clinical-risk estimates.
- The real amber disagreement banner appeared.
- Quantum and classical result cards measured the same **430.4 × 407.15 CSS pixels**.
- Fast quantum explanation displayed a timed computing state and then completed with the four real selected names.
- Full-feature classical explanation completed with all 30 real clinical names.
- Same-four classical explanation completed with the same four selected clinical names.
- The explanation panel rendered directions and attribution only; it did not render a competing confidence.
- Browser console warning/error log was empty.

## Docker host note

Docker Compose and the project images were verified in the preceding September 2 integration record. During this final September 4 run, Docker Desktop 4.89.0 failed before it could start any project container because Windows denied access while Docker recreated its own local `sailor-ingest.sock` and `engine.sock` reparse points under the user's AppData runtime folders. The stale folders were moved to named recoverable backups; Docker then recreated the same failing socket, proving this was a host-runtime issue rather than stale Q-TRACE state. No image, volume, credential, project file, or Docker setting was deleted, and **Reset to factory defaults was not used**. The complete app was reverified through the pinned local runtime instead.

## Remaining honest boundaries

- One public diagnostic breast-mass dataset; no external or prospective clinical cohort.
- Simulator execution only; no measured real-QPU, noise, or hardware-speed evidence.
- The live/five-fold quantum pool is 20 rows, while the separate Phase 2 headline benchmark uses 200.
- No authentication, HTTPS termination, hospital integration, PHI workflow, regulatory validation, or clinical threshold study.
- SHAP/LIME explain model behavior, not biological causality or clinical action.
