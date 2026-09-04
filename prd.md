# PRD — Hybrid Quantum ML Platform for Early Disease Detection

**Problem statement:** SIH26139, posted by Egreen Quanta, MedTech/BioTech/HealthTech theme. See `architecture.md` for how this gets built and `decisions.md` for why key choices were made this way.

## 1. Official problem statement (verbatim scope anchor)

Design and develop a hybrid quantum-classical machine learning platform for early disease detection. The platform must integrate classical pre-processing and feature engineering with quantum-enhanced learning models (QSVM, QNN, VQC), apply this to biomedical datasets for early identification of disease, and support data ingestion, hybrid model training, prediction, explainability, and performance evaluation against classical baselines.

**Official objectives (what judges will score against):**
1. Hybrid quantum-classical architecture suitable for early disease detection
2. Quantum-enhanced classification for high-dimensional biomedical data
3. Detection accuracy/sensitivity/specificity benchmarked vs. classical ML
4. Scalable, interpretable, compatible with simulators and near-term hardware
5. Data pre-processing, feature selection, and explainability modules
6. Rigorous benchmarking: accuracy, computational efficiency, generalization

Every requirement below traces back to one of these six. Objectives 4 (scalable/hardware-compatible) and 6 (computational efficiency/generalization) are addressed throughout, and spelled out explicitly in one place at `architecture.md` §8.

## 2. Product vision

A working, honestly-benchmarked, explainable hybrid quantum-classical classifier for early breast cancer detection — deployed as a clean web demo — that a technically literate judge could interrogate on methodology and come away convinced the team understands both the quantum ML and the clinical-trust angle, not just that they wired some libraries together.

**This is not:** a diagnostic device, a claim of quantum computational advantage, or a finished clinical product. See `decisions.md` D-07 for why we don't pitch it as beating classical ML.

## 3. Users

| User | Context | What they need from this |
|---|---|---|
| SIH judges/evaluators | 5–10 minutes per team, technical background likely | Working live demo, honest methodology, clear answers to "why quantum," visible explainability |
| Team members (you) | Building under a hard deadline | Unambiguous specs, a build order that de-risks the hardest part first, a working core even if bonus features slip |
| (Aspirational) Clinicians/researchers | Not a near-term user of this prototype | Represented via the explainability requirement — the design should make sense to someone who'd eventually be this user |

## 4. Functional requirements (MoSCoW)

### Must have (core deliverable — required to satisfy the PS)
- **M1.** Load and preprocess WBCD: impute, scale, **select** 4 features (`SelectKBest`, ANOVA F-value — not PCA, see `decisions.md` D-13), quantum-range scale — see `architecture.md` §3.2
- **M2.** Working quantum classifiers from **both** paradigms the PS names — a VQC (with data re-uploading + weight re-mapping, `architecture.md` §3.3 / `decisions.md` D-11) **and** a quantum-kernel SVM (`architecture.md` §3.4–3.5 / `decisions.md` D-10) — using the corrected patterns in `decisions.md` D-02, D-03, D-15
- **M3.** Classical baselines trained and evaluated on the same split, in **two configurations**: full 30 features (the realistic benchmark) and the same 4 selected features the quantum path sees (the fair comparison, `decisions.md` D-14) — at least 4 model types (LogReg, Random Forest, XGBoost, SVM) in each configuration
- **M4.** **Every prediction shows both the quantum result and the classical result together, with equal visual weight, every time — not one primary and one buried.** This makes PS objectives 3 and 6 (benchmarking against classical, rigorously) visible in the product rather than buried in a report table — see `architecture.md` §1.1/§6.2 and `decisions.md` D-12 for the precise framing. Accuracy, malignant precision/recall/F1, sensitivity, specificity, confusion counts, ROC-AUC, and timing are retained across ≥3 random seeds for **both classical configurations**; the fold-isolated report supplies five-fold hybrid generalization evidence.
- **M5.** SHAP-based explanation for at least one real prediction, showing feature attribution **using real clinical feature names** (achieved automatically by M1's feature-selection approach — see `decisions.md` D-13), for both a quantum and a classical prediction
- **M6.** A functioning end-to-end web demo: upload or select patient data → see both predictions side by side → see explanation
- **M7.** Reproducible setup: `requirements.txt` + Dockerfile, one-command run
- **M8.** Documentation of methodology and honest limitations (feeds the docx submission document)
- **M9.** Two specific regression tests exist and pass before any UI work begins: cost-decreases (`decisions.md` D-02) and label-encoding-direction (`decisions.md` D-15) — both are silent-failure bugs that pass every accuracy check while being visibly wrong in a live demo

### Should have (strong differentiators, attempt after Must-haves are solid)
- **S1.** Full 6-model quantum ensemble spanning both paradigms (4 VQCs + 2 quantum-kernel SVMs, not VQC-only) — `architecture.md` §3.6
- **S2.** LIME as a second explainability method alongside SHAP, for cross-validation of feature attributions
- **S3.** The custom judge-facing frontend per the design system in `architecture.md` §6, not just the Streamlit debug view
- **S4.** Statistical significance test (paired t-test) between ensemble and best single baseline
- **S5.** Disagreement banner when quantum and classical predict different labels (`decisions.md` D-12) — surfaced honestly, not hidden

### Could have (bonus, time-boxed — see `roadmap.md` cut-line)
- **C1.** Multimodal extension: CBIS-DDSM (imaging) + TCGA-BRCA (genomics) fusion, Q RadFusion-style
- **C2.** Noise-aware simulation via `default.mixed` to show robustness under simulated hardware noise
- **C3.** Real quantum-hardware backend run (via a PennyLane hardware plugin) as a "we also validated on real hardware" appendix

### Won't have (explicitly out of scope — say so if asked, don't apologize for it)
- Clinical validation, regulatory submission, or any claim of diagnostic readiness
- MIMIC-IV integration (credentialing timeline doesn't fit)
- Real-time hospital system integration
- Any claim that the quantum model outperforms classical ML on raw accuracy (`decisions.md` D-07)

## 5. Non-functional requirements

| Requirement | Target |
|---|---|
| Runs on | A standard laptop CPU, no GPU/cloud quantum credits required |
| Full ensemble training time | VQCs: ~30 min for 4 models sequentially. QSVM kernel matrices: budget a few minutes each at full (~455-sample) training-set scale — measured ~17s at 120 samples, scales roughly O(n²) (`architecture.md` §3.7) |
| Single-prediction latency | Under 2 seconds for the prediction itself. SHAP explanation: ~7s if explaining VQC models only, ~70s once a QSVM model is included (measured, `architecture.md` §3.7, `decisions.md` D-17) — default the UI to the fast VQC explanation, offer the full-ensemble explanation as an explicit, slower opt-in |
| Frontend | Responsive to mobile width, visible keyboard focus, respects `prefers-reduced-motion` |
| Cost | $0 — every tool in the stack is free/open-source (verified license table in `architecture.md`) |
| Reproducibility | A fresh clone + `docker compose up` produces a running demo with no manual steps |

## 6. Success criteria

**Implementation audit, September 3, 2026:** The user approved the D-27 extension. M4 now retains malignant-focused three-seed classical metrics for both feature configurations; M5 exposes attribution-only quantum and classical SHAP/LIME scopes; structured CSV ingestion validates the exact named 30-feature schema; and five-fold preprocessing is isolated inside every fold. D-22 still accepts the ensemble's lack of a demonstrated accuracy win without retuning it. External-cohort validation, real hardware, and clinical readiness remain explicitly out of scope.

| Criterion | Target | Evidence |
|---|---|---|
| Quantum model trains correctly | Final cost is lower than initial cost; individual epochs need not be monotonic | Cost-decreases regression test in `roadmap.md` day 1 |
| Quantum-kernel SVM accuracy | Verified achievable: 95.6% on WBCD with selected (not PCA) features | `decisions.md` D-10, D-13 |
| VQC (re-upload + re-map) accuracy | Verified achievable: ~93% on WBCD with selected features | `decisions.md` D-11 |
| Ensemble versus best single quantum model | Original win target was not achieved; the measured non-significant difference is accepted under D-22 | Both findings reported: reduced VQC-family variability and no demonstrated accuracy win; no equivalence proof |
| **Fair classical comparison** | Classical on the *same 4 features* as quantum is reported alongside classical-on-all-30; the retained three-seed same-4 Logistic mean is 93.57% | `artifacts/evaluation/classical_three_seed_metrics.json` and D-14 — don't report only one feature view |
| **Hybrid display** | Every prediction shows quantum AND classical results together, equal visual weight | Live demo — this should be undeniable within 5 seconds of any demo run |
| Explainability | SHAP output shown for a real prediction from both paradigms, using real clinical feature names (not PC1-style labels) — verified: `mean concave points`, `worst radius`, `worst perimeter`, `worst concave points` | Screenshot + live demo, `decisions.md` D-13 |
| Label direction correct | Explicit test confirms sklearn's 0=malignant/1=benign convention is respected end-to-end | `decisions.md` D-15 — this is a silent-failure risk, verify, don't assume |
| Web app | Fully working live demo matching the design system in `architecture.md` §6, not a mockup or hardcoded screen | Live at judging, rehearsed with real (non-hardcoded) inputs |
| Reproducibility | One-command Docker setup works on a machine that hasn't seen the repo before | Tested by a teammate who didn't write the code |
| Honesty | Team can answer "why not just classical ML" without overclaiming | Rehearsed answer, see `decisions.md` D-07 |

## 7. Open questions for the team to resolve early

1. Final team-role assignment against the pipeline stages in `architecture.md` §5 (recommend 2 people cross-trained on the quantum circuits specifically — see `roadmap.md`)
2. Whether to attempt the multimodal Could-have (C1) at all, given CBIS-DDSM/TCGA-BRCA's heavier data-engineering cost relative to benefit for the internal round
3. Team name / branding for the custom frontend's copy and visuals
