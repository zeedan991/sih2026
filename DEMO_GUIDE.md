# Q-TRACE: September 8 demo guide

Status reviewed September 3, 2026. The hackathon date comes from the user; confirm venue, submission format, and presentation template with the organizer.

## What this project actually does

Q-TRACE compares two quantum ML families with classical baselines on the Wisconsin Breast Cancer dataset (569 records, 30 numeric measurements). It is a simulator-based research demonstration, not a screening test or medical device. Dataset records come from existing breast-mass measurements; this benchmark does not establish prospective early-detection performance.

The pipeline splits the data first, fits imputation/scaling and `SelectKBest(f_classif, k=4)` on training rows, and preserves real clinical column names. Quantum inputs are clipped into an inward angle range; amplitude embedding also handles direct zero vectors safely. Label 0 means malignant; label 1 means benign.

The quantum side combines four data-reuploading VQCs with angle/4-qubit and amplitude/2-qubit kernel SVMs, using inverse OOB probability-MSE weights and soft voting. Each VQC trains for at least 100 epochs. The classical side trains Logistic Regression, Random Forest, XGBoost, and SVM on both all 30 features and the same selected four. The live comparison dial uses Logistic Regression; all eight classical results are available in the developer dashboard.

FastAPI serves the judge-facing HTML/CSS/JS interface and its prediction/explanation endpoints. Streamlit is a separate API client, not a second training implementation. Explanations use SHAP plus LIME with real feature names. The default explains only VQCs; full-six-model attribution is a clearly slower opt-in. Neither explanation response nor panel supplies a competing confidence verdict.

## Where to look

| Folder/file | Purpose |
|---|---|
| `backend/data/pipeline.py` | Train-fitted preprocessing, selected names, label conversion |
| `backend/quantum/` | VQC circuits, fidelity kernels, ensemble training, OOB voting |
| `backend/classical/`, `backend/benchmark.py` | Two-configuration classical models and comparison runner |
| `backend/explain/` | Scoped SHAP/LIME, slow consent, attribution audits |
| `backend/main.py`, `runtime.py`, `request_limits.py` | API contracts, model lifecycle, bounded inference |
| `frontend/`, `dev-dashboard/` | Judge workspace and internal diagnostics |
| `tests/` | Pipeline, circuit, kernel, ensemble, explanation, API, dashboard and deployment regressions |
| `artifacts/` | Preserved measurements and dated verification evidence |
| `prd.md`, `architecture.md`, `decisions.md`, `roadmap.md` | Requirements, design, decisions, deadline and remaining work |

`Project_Blueprint_SIH26139.docx` is retained as historical planning material. Its old dates, cross-validation/hardware claims, and single-run figures must not be copied into the final presentation as current evidence. It was content-reviewed, not rewritten or visually revalidated in this cleanup.

## Results you can defend

Use [the saved Phase 2 report](artifacts/models/phase2_benchmark_100epochs_200pool.md), not a new claim inferred from the live dial. Its independent stratified splits use seeds 42, 123, and 2026, 100 VQC epochs, and a 200-row quantum training pool.

- Ensemble: **94.15% mean, 93.86–94.74% range**.
- Strongest single family, angle QSVM: **94.44% mean, 93.86–94.74% range**.
- Individual VQC variants span **89.47–94.74%** across their runs.
- Amplitude QSVM: **73.98% mean, 72.81–75.44% range**, correctly down-weighted, not removed.

State both findings: the ensemble supports the VQC-family instability concern through a narrower observed range, and it **did not beat** the strongest single model. The paired tests detected no significant difference; this is not proof of statistical equivalence, clinical validity, or computational quantum advantage.

The repeated three-seed classical study is now retained in `artifacts/evaluation/classical_three_seed_metrics.json`, with accuracy, malignant precision/recall/F1, sensitivity, specificity, clinically oriented confusion counts, ROC-AUC, and timing for all eight model/configuration combinations. Same-four mean accuracies reproduce the prior record: Logistic Regression 93.57%, Random Forest 93.27%, XGBoost 93.27%, and SVM 93.86%.

The September 3 leakage-safe five-fold study refits every preprocessing stage inside each fold and trains all six quantum members with 100 VQC epochs and a 20-row quantum pool. Its most useful comparison is:

| Model view | Mean accuracy | Malignant sensitivity | Specificity | Mean ROC-AUC |
|---|---:|---:|---:|---:|
| Six-model quantum ensemble | 91.57% | 83.50% | 96.36% | 0.981 |
| Classical Logistic Regression, all 30 | 97.37% | 94.36% | 99.16% | 0.995 |
| Classical Logistic Regression, same 4 | 94.03% | 90.59% | 96.08% | 0.989 |

This fold study is stronger generalization evidence than repeated holdout, but the deliberately small quantum training pool also makes it a stress test, not a replacement for the separate 200-row Phase 2 benchmark. The amplitude/2-qubit QSVM was especially unstable here (54.61% mean accuracy, 29.20–62.28% range); retain and report that result as a limitation. It was not used to retune weights after seeing validation results.

## Demo configuration versus benchmark

The default live app follows `artifacts/models/runtime_manifest.json`, configuration `qtrace-wbcd-s42-e100-q20-v1`: seed 42, four VQCs at 100 epochs, six quantum members, a 20-row quantum training pool, and all 455 training rows for classical fits. It deterministically refits this identified configuration at startup; it does not pretend to load the separate 200-row benchmark weights. Both UIs and `/health` show the live identifier and settings. Matching selected features controls input dimension, not sample count. Do not quote saved benchmark accuracy as the live model's measured accuracy.

The model-level OOB rows were excluded from each member's fit, but shared preprocessing was fitted on the overall training split. These are not fully nested preprocessing-OOB estimates. The held-out test set was excluded from preprocessing, fitting, and weight selection. Repeatedly viewing that test set during development still limits how independently confirmatory it can be.

## Remaining boundaries — don't call these complete

The four September 2 implementation gaps are closed: disease-oriented metrics, fold-isolated cross-validation, classical attribution, and an identified live configuration are now in the code, APIs, UI, tests, and retained artifacts. The current remaining boundaries are research scope, not hidden implementation claims:

1. **Independent clinical validation:** no external cohort, prospective study, calibrated clinical threshold, subgroup/fairness analysis, or regulatory validation.
2. **Hardware evidence:** the device is configurable and both `lightning.qubit` and `default.qubit` simulator paths are tested, but this repository does not claim a real-QPU or noise study.
3. **Deployment/privacy:** this is a local research demo with no accounts, hospital integration, encryption termination, or workflow for identifiable health data.
4. **Human rehearsal:** a teammate should still perform a clean-clone run and rehearse the pitch on the presentation machine.

No multimodal bonus work or post-validation weight retuning is part of this completion pass.

## Rehearse before September 8

1. Follow the clone/Docker steps in `README.md` on the presentation computer. Keep it plugged in; wait for ready status before judging. Download/build beforehand and check the demo without internet. Fonts can fall back locally; predictions do not need a cloud quantum service.
2. Select several real held-out records, including an agreement and a disagreement, then load the example named CSV once. The demo catalog deliberately prioritizes disagreements for teaching; it is not a representative sample for estimating accuracy. Confirm both equal-size result cards and both classical feature configurations work.
3. Run fast quantum attribution, then one classical attribution, and narrate that attribution is anchored to the existing verdict rather than being a second verdict. Allow roughly 7–30 seconds depending on machine/configuration. Demonstrate the full quantum opt-in only if time permits; "70 seconds or longer" is a warning, not a promised upper bound.
4. Change the patient after completion and verify previous results disappear. If another request is running, wait and retry after HTTP 429 instead of repeatedly clicking.
5. Open the saved benchmark and answer: "Why quantum?" — to investigate two quantum approaches with a transparent classical comparison and explainability, not to claim a classical simulator has demonstrated quantum advantage.
6. Have a friend rehearse from a clean clone; keep this known-good Git commit available. Prepare a short screen recording as a labeled backup, not as a replacement for claimed live inference.

Use public WBCD demo records only. The app has no account system, HTTPS termination, or clinical privacy workflow. Keep Docker local by default; trusted-LAN sharing is an explicit README option. Never put passwords, access tokens, or real identifiable patient records in GitHub.
