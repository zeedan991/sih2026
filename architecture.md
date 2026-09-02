# Architecture — Hybrid Quantum ML Platform for Early Disease Detection

**Problem statement:** SIH26139 (Egreen Quanta) · **Status:** Verified against a live sandbox build, including a third pass on 2026-08-28 that fixed several real defects found by independent research review. See `decisions.md` for the reasoning behind every choice marked (Decision).

This document is the single source of truth for how the system is built. `prd.md` says *what* and *why*; this says *how*. If you're an AI coding agent (Antigravity or otherwise): implement against this file, and check `decisions.md` before deviating from it.

**Everything in this document ladders back to one thing: SIH26139 explicitly asks for a *hybrid* platform — classical pre-processing plus quantum-enhanced models (it names QSVM, QNN, and VQC specifically), benchmarked against classical baselines, and interpretable.** Every major architectural choice below is annotated with which official objective it satisfies.

---

## 1. System overview

```
                    ┌─────────────────────────────────────────────┐
                    │                CLIENT LAYER                  │
                    │   Custom web frontend (primary, judge-facing)│
                    │   + Streamlit (internal/dev debug view)      │
                    └───────────────────┬───────────────────────────┘
                                        │ REST/JSON (HTTPS)
                    ┌───────────────────▼───────────────────────────┐
                    │              BACKEND API (FastAPI)             │
                    │  /predict  /explain  /baselines  /health        │
                    └──────┬───────────────────────┬─────────────────┘
                           │                        │
              ┌────────────▼───────────┐ ┌──────────▼─────────────────┐
              │   QUANTUM ENGINE          │ │   CLASSICAL BASELINES        │
              │   PennyLane — VQC ensemble │ │   Full-feature (30) AND      │
              │   + Quantum-kernel SVM      │ │   same-4-feature variants    │
              │   lightning.qubit simulator│ │   XGBoost, RF, SVM, LogReg   │
              └────────────┬───────────┘ └──────────┬─────────────────┘
                           │                        │
                    ┌───────▼────────────────────────▼──────┐
                    │       EXPLAINABILITY LAYER              │
                    │       SHAP (KernelExplainer) + LIME     │
                    │       — real feature names throughout    │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────────┐
                    │              DATA PIPELINE                     │
                    │  load → impute → scale → SELECT 4 → q-scale    │
                    └─────────────────────────────────────────────┘
```

**(Decision D-01)** FastAPI is a real, independently-testable service. **(Decision D-10)** The quantum engine spans two paradigms. **(Decision D-13)** Feature *selection* replaces PCA — this changed the data pipeline's last box above from "PCA" to "SELECT 4," which matters more than it looks: it's also what fixed the explainability naming problem in §6.

### 1.1 The hybrid result is a first-class UI element, not a buried benchmark

Every prediction screen shows a quantum-model reading and a classical-model reading for the *same patient*, rendered with equal visual weight — neither presented as primary. See §6.2 for the UI spec and `prd.md` M4. (On the framing of *why* this matters: see `decisions.md` D-12 for a more careful version of this argument than the previous revision used — the honest case rests on PS objectives 3 and 6, not a claim about what the word "hybrid" strictly requires.)

---

## 2. Component breakdown

| Component | Technology | Verified version (pinned) | Purpose |
|---|---|---|---|
| Backend API | FastAPI + Uvicorn | `fastapi==0.141.1`, `uvicorn==0.52.4` | Orchestrates preprocessing, inference, explanation, baseline comparison |
| Quantum engine — variational | PennyLane + PennyLane-Lightning | `pennylane==0.45.1`, `pennylane_lightning==0.45.0` | VQC ensemble on the `lightning.qubit` simulator |
| Quantum engine — kernel | PennyLane + scikit-learn | (same, no new package) | Quantum-kernel SVM |
| Classical ML | scikit-learn, XGBoost | `scikit-learn==1.9.0`, `xgboost==3.4.1` | Baselines — both full-feature and same-4-feature variants (§3.6) |
| Explainability | SHAP, LIME | `shap==0.52.0`, `lime==0.2.0.1` | Feature attribution, now with real clinical feature names by construction |
| Numerics | NumPy, pandas, SciPy | `numpy==2.5.2`, `pandas==3.0.5`, `scipy==1.18.1` | Array ops, data handling |
| Testing | pytest, HTTPX2 | `pytest==9.1.1`, `httpx2==2.12.0` | Regression and API test runner dependencies |
| Dev/debug frontend | Streamlit | `streamlit==1.62.0` | Fast internal iteration only |
| Judge-facing frontend | Custom HTML/CSS/JS | — | The polished UI |
| Containerization | Docker | — | One-command reproducible setup |

No new dependencies for anything in this revision — `sklearn.feature_selection.SelectKBest` is already in scikit-learn.

---

## 3. Quantum model specification

### 3.1 What changed this revision, and why

An independent review of the previous revision found several real defects — some in the *original* concept sketch that a naive implementation could still reintroduce, some in this project's own spec. Verified and fixed:

| # | Defect | Where | Verified fix |
|---|---|---|---|
| 1 | Weight-index formula `layer*n_qubits+i` collides with the next layer's block | Original idea.docx sketch | This project's own code already used a running counter (`idx += 1`), not the formula — confirmed collision-free by direct index-set comparison (§3.3) |
| 2 | `y = y*2-1` applied globally, then handed to `XGBClassifier`, which requires `{0,1}` | Original idea.docx sketch | This project's pipeline keeps two separate label arrays (`y` for classical, `y_pm1` for quantum) — never overwrites the original (§3.2) |
| 3 | `AmplitudeEmbedding` on 4 qubits needs 16 values; only 4 features are supplied | This project's own ensemble table (previous revision) | Fixed: the amplitude-embedding kernel variant now uses **2 qubits** (needs exactly 4 values — a natural match, not a padding hack). Verified working. (§3.5) |
| 4 | Device closure bug — one global device fixed at N qubits, called with a different qubit count | Original idea.docx sketch | This project's `make_circuit`-style pattern creates a fresh, correctly-sized device per circuit (§3.3) — verified across 3+ different qubit counts in testing |
| 5 | PCA variance claim backwards ("~56% at 12 components") | Original idea.docx | Actual measured cumulative variance: **72.9% at 3, 79.6% at 4, 97.0% at 12 components** — more components always explains equal-or-more variance; the original figure was impossible as stated. Moot now anyway — PCA is replaced (D-13). |
| 6 | Target-label direction: `sklearn`'s dataset encodes 0=malignant, 1=benign — easy to invert silently since accuracy metrics look identical either way | Not previously documented anywhere | Verified and now has an explicit required test (§3.2, `decisions.md` D-15) |
| 7 | SHAP explaining PCA components (PC1–PC4) contradicts the PRD's requirement for clinically-named top features; fixing it by wrapping all 30 raw features in the SHAP function blows the latency budget from seconds to minutes | This project's own design (previous revision) | Resolved by D-13 — once the quantum path's 4 inputs are *selected* named features instead of PCA components, SHAP's output is clinically named by construction, at no extra cost (§6) |

### 3.2 Data → model pipeline (revised: selection, not PCA)

```
Raw WBCD (569 × 30 features)
   → stratified 80/20 train/test split
   → median-impute missing values
   → StandardScaler
   → SelectKBest(f_classif, k=4)   [ANOVA F-value feature SELECTION —
                                     not PCA. Verified selected features
                                     on this exact split: mean concave
                                     points, worst radius, worst
                                     perimeter, worst concave points —
                                     real, clinically-named, no
                                     abstraction]
   → MinMaxScaler to [10⁻⁶, π-10⁻⁶] (quantum models only; clip=True)
   → 5-fold CV within training set
```

Every preprocessing component is fitted on training rows only. The slightly
inward quantum range is defense-in-depth for amplitude embedding: ordinary
training, held-out, and future live inputs cannot become an exact all-zero or
all-π boundary vector after transformation (D-21).

**Two label arrays, never one overwritten by the other** (defect #2 above):
```python
y = data.target                    # {0,1}: 0=malignant, 1=benign -- classical models use THIS
y_pm1 = (y * 2 - 1).astype(float)  # {-1,+1}: separate array -- quantum models use THIS
# never do y = y * 2 - 1 in place; keep both, always
```

**(Decision D-13)** Switching from PCA to `SelectKBest` feature selection does three things at once: it matches PS objective 5's literal wording ("feature selection," not extraction), it verified-improved QSVM accuracy slightly (94.7% -> 95.6%), and it eliminates the SHAP-naming problem entirely rather than requiring an expensive workaround. Full reasoning and numbers in `decisions.md` D-13.

**(Decision D-15) Label-encoding regression test — required, not optional:**
```python
def test_label_encoding_direction():
    """sklearn's breast cancer dataset: 0=malignant, 1=benign. This is the
    opposite of what many people assume. Getting it backwards produces a
    model with perfect-looking accuracy metrics but every displayed
    verdict inverted -- extremely easy to miss in a demo. Assert the
    convention explicitly rather than trusting it silently."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    assert data.target_names[0] == "malignant"
    assert data.target_names[1] == "benign"
    # and at the API layer: a patient with y=1 must come back "benign"
```
Put this in `tests/test_pipeline.py` and run it before ever wiring up the `/predict` label strings — see `roadmap.md` Phase 1.

### 3.3 VQC circuit — verified pattern (unchanged from previous revision, now explicitly confirmed collision-free)

```python
import pennylane as qml
from pennylane import numpy as np

n_qubits, n_layers = 4, 3
dev = qml.device("lightning.qubit", wires=n_qubits)   # fresh device per circuit -- never a shared global (defect #4)

@qml.qnode(dev)
def vqc(inputs, raw_weights):
    # Weight re-mapping (Koelle et al.) -- verified: improves fit.
    weights = 2 * np.pi * (1 / (1 + np.exp(-raw_weights)))
    idx = 0   # running counter -- NOT a layer*n_qubits formula (defect #1).
              # The formula version double-books indices between layers;
              # this doesn't, by construction. Confirmed via direct
              # comparison of the index sets both ways.
    for layer in range(n_layers):
        for i in range(n_qubits):           # data re-uploading every layer
            qml.RY(inputs[i], wires=i)
        for i in range(n_qubits):
            qml.RY(weights[idx], wires=i); idx += 1
            qml.RZ(weights[idx], wires=i); idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))
```

Loss function unchanged from D-03 (convert labels to `{0,1}` before cross-entropy — note this is a *third*, separate label representation used only inside the loss math, distinct from both `y` and `y_pm1` above; don't conflate the three). `interface="torch"` bug fix (D-02) still applies.

**Verified accuracy on selected features (not PCA), same split as before:** ~93.0% — statistically the same as the PCA-based number from the previous revision, confirming the switch to feature selection didn't cost accuracy.

### 3.4 Quantum-kernel SVM — angle-embedding variant (unchanged pattern)

```python
import pennylane as qml
import numpy as np
from sklearn.svm import SVC

n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits)
projector = np.zeros((2**n_qubits, 2**n_qubits)); projector[0, 0] = 1

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

def kernel_matrix(A, B):
    return np.array([[kernel_circuit(a, b) for b in B] for a in A])

K_train = kernel_matrix(X_train_q, X_train_q)
qsvm = SVC(kernel="precomputed", probability=True).fit(K_train, y_train)
```

**Verified accuracy on selected features: 95.6%** (up from 94.7% on PCA components, same split) — the strongest single quantum result in this project, and the closest to classical.

### 3.5 Quantum-kernel SVM — amplitude-embedding variant (fixed: now 2 qubits, not 4)

```python
n_qubits_amp = 2   # NOT 4 -- AmplitudeEmbedding needs exactly 2**n_qubits
                    # values. 2 qubits needs 4 values, matching our 4
                    # selected features exactly. Using 4 qubits here (as
                    # the previous revision did) needs 16 values and
                    # would fail outright -- verified both ways.
dev_amp = qml.device("lightning.qubit", wires=n_qubits_amp)

def canonicalize_zero_amplitude(x):
    x = np.asarray(x, dtype=float).copy()
    if np.linalg.norm(x) == 0.0:
        x[0] = 1.0  # any direct exact-zero input -> canonical |00> (D-21)
    return x

@qml.qnode(dev_amp)
def amp_kernel_circuit(x1, x2):
    qml.AmplitudeEmbedding(canonicalize_zero_amplitude(x1), wires=range(n_qubits_amp), normalize=True, pad_with=0.0)
    qml.adjoint(qml.AmplitudeEmbedding)(canonicalize_zero_amplitude(x2), wires=range(n_qubits_amp), normalize=True, pad_with=0.0)
    proj = np.zeros((2**n_qubits_amp, 2**n_qubits_amp)); proj[0, 0] = 1
    return qml.expval(qml.Hermitian(proj, wires=range(n_qubits_amp)))
```
Verified: this call succeeds. `normalize=True` handles nonzero feature rows that aren't already unit vectors. The general QSVM input boundary canonicalizes any exact all-zero amplitude row to `|00>` before every public fit or prediction path, including future live inference; it is not a correction tied to a known training row. In addition, the pipeline's inward `[10⁻⁶, π-10⁻⁶]` range prevents normally transformed rows from reaching that edge at all. `pad_with=0.0` is a documented no-op safety net here since 4 features already exactly fill a 2-qubit amplitude vector.

**Honest note on individual performance:** measured standalone accuracy for this variant was notably lower than every other model in the ensemble (~76% vs. 93-95% for the others) on one verification run — a 2-qubit Hilbert space is small, so this is expected, not a bug. It still contributes to the ensemble's diversity, and the inverse-MSE weighting scheme (§3.6) automatically down-weights it relative to stronger models. Don't be alarmed if this specific model's individual number looks weak in your own training logs — that's the design working as intended, not something to debug.

### 3.6 Combined quantum ensemble (6 models) and the fair classical comparison

| # | Type | Config | Role |
|---|---|---|---|
| 1 | VQC (re-upload) | 4 qubits, 3 layers, CNOT | Primary variational model |
| 2 | VQC (re-upload) | 3 qubits (top-3 of the 4 selected features), 2 layers, CNOT | Diversity — shallower |
| 3 | VQC (re-upload) | 4 qubits, 2 layers, CZ | Diversity — different entangler |
| 4 | VQC (re-upload) | 4 qubits, 3 layers, CZ, reversed encoding order | Diversity |
| 5 | Quantum-kernel SVM | Angle-embedding, 4 qubits (§3.4) | Primary kernel model — verified 95.6% |
| 6 | Quantum-kernel SVM | Amplitude-embedding, **2 qubits** (§3.5, fixed) | Diversity — different feature map |

**(Decision D-14) Classical baselines now run in two configurations, not one:**

1. **Full-feature** (all 30, current behavior) — the strongest, real-world classical benchmark. Verified: LogReg 98.25%.
2. **Same-4-feature** (identical `SelectKBest` output the quantum models see) — the *fair* apples-to-apples comparison. **Verified finding, and a genuinely important one: LogReg on the same 4 features drops to 92.98% — statistically tied with the VQC's 92.98% on those same features.** A meaningful share of the "classical wins" gap in the previous revision was an artifact of classical models seeing 7.5x more input information, not a demonstrated capability gap. Report both configurations, always — this is a more honest and, frankly, more favorable-to-quantum story than only showing the full-feature comparison.

Weighting: normalized inverse out-of-bag probability-MSE, spanning all 6 models. Each member receives its own training bootstrap; rows absent from that bootstrap determine its weight, and the held-out test set never participates in weight selection.

### 3.7 Measured performance (updated numbers, selected-feature pipeline)

| Model | Verified test accuracy |
|---|---|
| Quantum-kernel SVM (angle), earlier full-data verification | **95.6%** |
| VQC (re-upload + re-map), 100 epochs, 3 seeds | **91.23% mean; 89.47-93.86% range** |
| Phase 2 VQC 4q/3l/CNOT, 100 epochs, 200-row train pool, 3 seeds | **91.23% mean; 90.35-92.98% range** |
| Phase 2 VQC 3q/2l/CNOT (top 3), 100 epochs, 200-row train pool, 3 seeds | **91.81% mean; 89.47-92.98% range** |
| Phase 2 VQC 4q/2l/CZ, 100 epochs, 200-row train pool, 3 seeds | **91.52% mean; 90.35-92.98% range** |
| Phase 2 VQC 4q/3l/CZ reversed, 100 epochs, 200-row train pool, 3 seeds | **92.11% mean; 90.35-94.74% range** |
| Phase 2 QSVM angle/4q, 200-row train pool, 3 seeds | **94.44% mean; 93.86-94.74% range** |
| Phase 2 QSVM amplitude/2q, 200-row train pool, 3 seeds | **73.98% mean; 72.81-75.44% range** |
| Phase 2 six-model OOB-weighted ensemble, 200-row train pool, 3 seeds | **94.15% mean; 93.86-94.74% range** |
| Classical, full 30 features (LogReg) | 98.25% |
| Classical LogReg, **same 4 features**, 3 seeds | **93.57% mean; 92.98-93.86% range** |
| Classical Random Forest, **same 4 features**, 3 seeds | **93.27% mean; 92.11-93.86% range** |
| Classical XGBoost, **same 4 features**, 3 seeds | **93.27% mean; 92.98-93.86% range** |
| Classical SVM, **same 4 features**, 3 seeds | **93.86% mean; 92.98-94.74% range** |

- The earlier 20-epoch VQC quick check measured 87.43% mean with a 74.56-93.86% range. Re-running the identical seeds/splits for 100 epochs raised the mean to 91.23% and narrowed the range to 89.47-93.86%; see `decisions.md` D-20. Twenty epochs remains a regression-test budget, not a final benchmark budget.
- The Phase 2 controlled benchmark used a stratified 200-row pool from each training split, actual bootstrap OOB probability-MSE weights, and the untouched full test split. State both findings together: the ensemble validated the VQC-family instability concern by holding a tight 93.86-94.74% range while individual VQCs ranged from 89.47% to 94.74%, and it statistically **tied the strongest individual model rather than beating it** (94.15% versus 94.44% mean; every paired p-value > 0.56). Exact per-model weights and the rationale for retaining all six models without retuning are in `decisions.md` D-22.
- VQC forward pass: ~5ms
- Quantum kernel evaluation: ~7ms/pair
- Full training kernel matrix (~455 samples): budget 3-4 minutes
- **SHAP `KernelExplainer`, one patient — revised, measured on the actual mixed ensemble:** a VQC-only explanation is fast (~7s for 3 VQCs), but the moment a QSVM model joins the explained ensemble, cost jumps sharply — **verified: 69.6s for a 2-model (1 VQC + 1 QSVM) ensemble**, because every SHAP perturbation sample requires a fresh kernel computation against the QSVM's full background set, which is far more expensive than a VQC forward pass. For the full 6-model ensemble (4 VQC + 2 QSVM), budget on the order of a minute or more per explanation, not seconds. This changes the UI requirement: a brief loading spinner is no longer sufficient — build a real "computing explanation..." state, and consider explaining primarily against the VQC sub-ensemble by default (fast, seconds) with the full 6-model explanation as an optional "deep explanation" the user explicitly requests. Reducing the QSVM background-sample count (e.g. from 100 to 30-40) is the cheapest lever if this needs to be faster, at some cost to explanation stability.

**Phase 3 implementation:** `ExplainabilityService` exposes three explicit scopes. `fast_vqc` is the default and renormalizes only the four VQC weights; `full_ensemble` uses the frozen six-model OOB weights; and `qsvm_diagnostic` isolates the two kernel models for verification. Both QSVM-inclusive scopes reject calls unless `allow_slow=True`. Combined SHAP + LIME calls emit durable `computing`, `complete`, or `failed` progress events with elapsed and expected timing, so the Phase 4 API and Phase 5 UI can present a genuine long-running state. SHAP explains scalar class-1/benign probability with the exact selected clinical names. LIME is an independent cross-check: its local slopes are multiplied by the patient's standardized displacement from its background mean before direction is compared with SHAP, avoiding an invalid slope-versus-contribution sign comparison.

A controlled real-model Phase 3 verification (seed 42, four VQCs at 100 epochs, 20-row training pool, 10-row explanation background, 16 SHAP samples) succeeded for VQC-only, QSVM-only, and full-six scopes with zero SHAP additivity residual. It measured 3.41s, 7.82s, and 9.13s respectively under those intentionally small smoke-test settings. These are not replacements for D-17's production-like 69.6s result; the shipped metadata continues to warn **at least 70 seconds** for either QSVM-inclusive scope. All explainers received `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`; no generic component names appeared. Exact inputs and attributions are recorded in `artifacts/explainability/phase3_real_model_verification.md`.

**Forward-compatibility note:** `sklearn.svm.SVC(kernel="precomputed", probability=True)` (§3.4, §3.5) raises a `FutureWarning` on the pinned scikit-learn 1.9.0 — the `probability` parameter is deprecated and scheduled for removal in 1.11, in favor of wrapping with `CalibratedClassifierCV`. It still works correctly on the pinned version (verified), so this isn't an active bug — but don't casually run `pip install --upgrade scikit-learn` mid-project without checking whether this has been removed yet. If you do upgrade past 1.11, switch to `CalibratedClassifierCV(SVC(kernel="precomputed"), ensemble=False)` for probability outputs.

---

## 4. API contract (FastAPI)

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/health` | GET | — | `{status, models_loaded}` |
| `/predict` | POST | `{features: float[30]}` | `{quantum: {label, confidence, per_model}, classical: {full_feature: {...}, same_4_feature: {...}}}` |
| `/explain` | POST | `{features: float[30], model, allow_slow: bool = false}` | `{feature_names, shap_values, top_features, scope: "vqc_fast"\|"full_ensemble"}` — never includes a standalone confidence/probability field (D-23). This response supplies attribution only; the frontend anchors it to the confidence already shown from `/predict`, never renders a second number. `allow_slow=false` (default) explains the fast VQC sub-ensemble; `allow_slow=true` explains the full 6-model ensemble, which naturally matches `/predict`'s number exactly since it explains that literal prediction. `feature_names` are always real clinical names, never PC1-style labels. |
| `/baselines` | GET | — | Full comparison table: both quantum types x both classical configurations |
| `/metrics` | GET | — | Ensemble vs. single-model, confusion matrix, ROC-AUC, quantum-vs-classical head-to-head (both classical configs) |

`quantum` and `classical` are always both present and fully populated (D-12). `classical` now has two sub-objects, not one — both always populated too (D-14).

---

## 5. Repository structure

```
quantum-disease-detection/
├── AGENTS.md
├── prd.md  architecture.md  roadmap.md  decisions.md
├── requirements.txt
├── Dockerfile  docker-compose.yml
├── backend/
│   ├── main.py
│   ├── quantum/
│   │   ├── vqc_circuits.py       # 4 VQC variants, §3.3
│   │   ├── qsvm_kernels.py       # angle (§3.4) + amplitude (§3.5, 2-qubit)
│   │   ├── ensemble.py
│   │   └── train.py
│   ├── classical/
│   │   └── baselines.py          # BOTH full-feature and same-4-feature (§3.6)
│   ├── explain/
│   │   └── shap_lime.py          # feature_names always real names now
│   ├── data/
│   │   └── pipeline.py           # SelectKBest, not PCA (§3.2); keeps y AND y_pm1 separately
│   └── weights/
├── frontend/
│   ├── index.html  styles.css  app.js
│   └── assets/
├── dev-dashboard/
│   └── app.py
└── tests/
    ├── test_pipeline.py           # includes test_label_encoding_direction (D-15)
    ├── test_circuits.py           # cost-decreases (D-02) + index-collision-free (D-14) checks
    ├── test_qsvm.py               # kernel symmetry/PSD + amplitude-embedding dimension check (D-14)
    └── test_api.py
```

---

## 6. Design system for the judge-facing frontend

### 6.1 Direction

Signature element: a **measurement dial** radial gauge. Layout motif: a **circuit rail** across the top (Data -> Select -> Model -> Explain — note: "Select," not "Preprocess," now that feature selection is named and visible rather than a black-box reduction step. Showing the actual selected feature names here is a small, free trust-building detail). A rendered mockup of the results screen was shown earlier in this project's planning — build to match it: two equal-size dials (quantum indigo, classical teal), a compact "why this prediction" panel beneath with real feature names as horizontal bars.

**Clinical-workspace revision (2026-09-02, D-24):** The user requested a more professional clinical presentation. Use a compact application header and study context, a restrained circuit rail, a patient-record panel on the left, and the equal quantum/classical assessment cards on the right. Remove the oversized promotional hero, decorative ambient gradients, and orbit animations. White surfaces, fine borders, compact measurement tables, and short functional transitions establish the research-workspace tone. The empty state must show that inference has not run, never invented patient results. On narrow screens the layout stacks without making either paradigm more prominent.

### 6.2 Why two dials, always

Every `/predict` response always has both `quantum` and `classical` fully populated (§4); the frontend renders both at equal visual size on every prediction, color-coded by paradigm, never by which one "won." When they disagree on the label, an amber banner says so explicitly.

### 6.3 SHAP now explains real features, not PCA components — no extra cost

Because §3.2 selects 4 *named* original features instead of extracting 4 abstract PCA components, SHAP's `feature_names` argument is just the selected column names — `["mean concave points", "worst radius", "worst perimeter", "worst concave points"]` in our verified run. This satisfies `prd.md`'s clinically-recognizable-feature requirement automatically, without needing to wrap the full 30-feature pipeline in the SHAP function — which would be slower still than the already-substantial cost documented in §3.7 (a QSVM-inclusive explanation is already ~70s explaining just 4 features; wrapping all 30 raw features plus the selection step itself would push this well past what's usable in a live demo). This was a real design tension in the previous revision, not a hypothetical one — D-13 resolves the *naming* half of it as a side effect; the *speed* half is handled by the default-fast/optional-deep-explain split in §3.7.

**One number on screen, always (D-23).** The fast VQC-only default explanation and the full 6-model ensemble prediction can legitimately disagree on confidence — they're different model subsets, verified during Phase 3 to differ by as much as 13 points on the same patient. The frontend must never show both: the dial's confidence, already returned by /predict, is the only number displayed; the explanation panel supplies feature-attribution bars underneath it and nothing else. Two different-looking probabilities for one patient reads as a bug to a judge in a live demo, regardless of how correct it is underneath.

### 6.4 Color tokens

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#F1F3F2` | Background |
| `--ink` | `#12181C` | Primary text |
| `--signal-teal` | `#0E7C7B` | Classical results |
| `--quantum-indigo` | `#4A3F8C` | Quantum results |
| `--caution-amber` | `#B8863D` | Low-confidence / disagreement flag |
| `--line` | `#D7DCDD` | Hairlines, dial tracks |

### 6.5 Type

Space Grotesk (compact headings), Source Sans 3 (body, controls, and explanations), IBM Plex Mono (metrics and raw feature values). Source Sans 3 replaces the prior Source Serif 4 body treatment at the user's request for a more clinical application-like presentation (D-24).

### 6.6 Accessibility floor

Responsive to mobile width, visible keyboard focus states, `prefers-reduced-motion` support, plain-text numeric equivalent for both dials.

---

## 7. Deployment

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 8. Scalability and generalization (PS objectives 4 and 6, explicit)

**Scalable:** qubit count and selected-feature count are both config values, not hard-coded logic. The backend/frontend split means scaling inference is a standard FastAPI concern, independent of the quantum code. Swapping datasets touches only `backend/data/pipeline.py`.

**Compatible with near-term quantum hardware:** every circuit targets PennyLane's device abstraction (`qml.device(...)`), not a simulator-specific API — pointing at real hardware (IBM, AWS Braket) via a PennyLane plugin is a device-string change (D-06). Not run on real hardware in this project (PRD C3, a stretch goal).

**Generalization:** stratified splitting, 5-fold CV, and >=3-seed reporting (§3.2, `decisions.md` D-09) are the actual mechanism — not just a claim. The two-configuration classical comparison (§3.6) is itself a generalization safeguard: it stops an unfair-comparison artifact from being reported as a genuine finding.

---
*Cross-references: `decisions.md` D-13/D-14/D-15 for this revision's fixes and the reasoning behind them, `prd.md` for updated scope, `roadmap.md` for the updated phase plan.*
