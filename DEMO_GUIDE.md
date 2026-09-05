# Q-TRACE: September 8 demo guide

Status reviewed September 4, 2026. The hackathon date comes from the user; confirm venue, submission format, and presentation template with the organizer.

## What this project actually does

Q-TRACE is a reusable hybrid quantum-classical disease-signal analysis platform demonstrated through **two separately trained modules**: Wisconsin breast-mass classification (569 records, 30 numeric measurements) and UCI early-diabetes questionnaire screening (520 records, 16 fields). It is a simulator-based research demonstration, not a diagnostic test or medical device. Never describe it as one universal model that can diagnose any disease; each new disease requires an appropriate dataset, independent training, validation, labels, and limitations.

Each module splits the data first, fits imputation/scaling and `SelectKBest(f_classif, k=4)` on training rows, and preserves real clinical or symptom names. Quantum inputs are clipped into an inward angle range; amplitude embedding also handles direct zero vectors safely. WBCD label 0 means malignant and label 1 means benign. The diabetes mapping separately defines UCI `Positive` as class 0 and `Negative` as class 1.

The quantum side combines four data-reuploading VQCs with angle/4-qubit and amplitude/2-qubit kernel SVMs, using inverse OOB probability-MSE weights and soft voting. Each VQC trains for at least 100 epochs. The classical side trains Logistic Regression, Random Forest, XGBoost, and SVM on both all available module features and the same selected four. The live comparison dial uses Logistic Regression; all eight classical results are available in the developer dashboard.

FastAPI serves the judge-facing HTML/CSS/JS interface and its prediction/explanation/report endpoints. Streamlit is a separate API client, not a second training implementation. Explanations use SHAP plus LIME with real feature names. The default explains only VQCs; full-six-model attribution is a clearly slower opt-in. Neither explanation response nor panel supplies a competing confidence verdict. The “AI-assisted” report is deterministic local evidence synthesis from real model outputs—not a chatbot—and exports as self-contained HTML or through Print/Save-PDF without a paid API or cloud patient-data transfer.

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

### Second-disease scalability evidence

The early-diabetes three-seed run uses seeds 42, 123, and 2026, 100 VQC epochs, a 20-row quantum training pool, and the full held-out split. The ensemble measured **83.01% mean accuracy (76.92–92.31%), 81.25% mean condition sensitivity, and 85.83% mean specificity**. The strongest quantum mean was the amplitude QSVM at **84.29%**; full-feature classical SVM reached **95.51%** mean. One VQC fell to 39.42% on a seed and was nearly zero-weighted by OOB voting. Say this plainly: the second module demonstrates reuse and exposes model instability; it does not demonstrate quantum superiority or external clinical validity.

## Demo configuration versus benchmark

The default live app follows `artifacts/models/runtime_manifest.json`, configuration `qtrace-multidisease-s42-e100-q20-v1`: seed 42, four VQCs at 100 epochs and six quantum members per disease module, a 20-row quantum training pool per module, and the full training split for classical fits. It deterministically refits this identified configuration at startup; it does not pretend to load the separate 200-row breast benchmark weights. Both UIs and `/health` show the live identifier and settings. Matching selected features controls input dimension, not sample count. Do not quote a saved benchmark accuracy as a live patient confidence or vice versa.

The model-level OOB rows were excluded from each member's fit, but shared preprocessing was fitted on the overall training split. These are not fully nested preprocessing-OOB estimates. The held-out test set was excluded from preprocessing, fitting, and weight selection. Repeatedly viewing that test set during development still limits how independently confirmatory it can be.

## Remaining boundaries — don't call these complete

The four September 2 implementation gaps are closed: disease-oriented metrics, fold-isolated cross-validation, classical attribution, and an identified live configuration are now in the code, APIs, UI, tests, and retained artifacts. The current remaining boundaries are research scope, not hidden implementation claims:

1. **Independent clinical validation:** no external cohort, prospective study, calibrated clinical threshold, comprehensive fairness analysis, or regulatory validation. The diabetes report includes only a small held-out sex/gender audit, explicitly not proof of fairness.
2. **Hardware evidence:** the device is configurable and both `lightning.qubit` and `default.qubit` simulator paths are tested, but this repository does not claim a real-QPU or noise study.
3. **Deployment/privacy:** this is a local research demo with no accounts, hospital integration, encryption termination, or workflow for identifiable health data.
4. **Human rehearsal:** a teammate should still perform a clean-clone run and rehearse the pitch on the presentation machine.

No multimodal bonus work or post-validation weight retuning is part of this completion pass. The diabetes source is a single-hospital Bangladesh questionnaire dataset whose symptom overlap can inflate apparent accuracy; it has no Indian-cohort validation and its binary sex/gender field is a source limitation.

## Rehearse before September 8

1. Follow the clone/Docker steps in `README.md` on the presentation computer. Keep it plugged in; wait for ready status before judging. Download/build beforehand and check the demo without internet. Fonts can fall back locally; predictions do not need a cloud quantum service.
2. Begin with breast oncology because it has the strongest retained evidence. Select several real held-out records, including an agreement and a disagreement, then load the example named CSV once. The catalog prioritizes disagreements for teaching; it is not a representative accuracy sample. Confirm both equal-size result cards and both classical feature configurations work.
3. Generate the local evidence report and show that it contains the real quantum ensemble, both classical views, six-member audit, selected inputs, dataset citation, limitations, and any subgroup audit. Download HTML or use Print/Save-PDF. Call it decision-support evidence, never a diagnosis or personal risk score.
4. Switch to early diabetes and explain the architecture-reuse story: a new schema and separately trained model bundle, not relabeling a breast model. Use a real held-out record or the manual symptom form. State the second module's weaker/variable quantum evidence without hiding it.
5. Run fast quantum attribution, then one classical attribution, and narrate that attribution is anchored to the existing verdict rather than being a second verdict. Allow roughly 7–30 seconds depending on machine/configuration. Demonstrate the full quantum opt-in only if time permits; "70 seconds or longer" is a warning, not a promised upper bound.
6. Change the patient after completion and verify previous results disappear. If another request is running, wait and retry after HTTP 429 instead of repeatedly clicking.
7. Answer “Where is AI?” — in the trainable VQCs, quantum-kernel SVMs, classical ML, SHAP/LIME, OOB ensemble weighting, and local evidence synthesis. Explain that omitting a hosted LLM makes the demo reproducible, private, offline-capable, and free.
8. Answer “Why quantum?” — to investigate two quantum approaches under an equal, transparent classical comparison and explainability, not to claim a classical simulator has demonstrated quantum advantage.
9. Have a friend rehearse from a clean clone; keep this known-good Git commit available. Prepare a short screen recording as a labeled backup, not as a replacement for claimed live inference.

Use only public benchmark or synthetic/manual demo records. The app has no account system, HTTPS termination, or clinical privacy workflow. Keep Docker local by default; trusted-LAN sharing is an explicit README option. Never put passwords, access tokens, or real identifiable patient records in GitHub.
