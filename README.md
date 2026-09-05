# Q-TRACE

Q-TRACE is a hybrid quantum-classical early disease signal analysis platform for Smart India Hackathon problem statement SIH26139. It now demonstrates the same complete workflow on two independently trained modules: Wisconsin breast-mass classification and UCI early-diabetes questionnaire screening. Each compares a six-member quantum ensemble (four VQCs and two quantum-kernel SVMs) with classical baselines.

The project is a research prototype, not a medical device. It deliberately presents quantum and classical results side by side and does not claim that quantum ML beats classical ML on raw accuracy.

**Hackathon: September 8, 2026 (user-confirmed).** Start with [the demo guide](DEMO_GUIDE.md) for the project map, rehearsal steps, and remaining research gaps. A working demo is not a claim of clinical validation or completion of every PRD target.

## Get the project

```powershell
git clone https://github.com/zeedan991/sih2026.git
Set-Location sih2026
```

Share the repository link with collaborators. For a source ZIP, run the following **inside this repository** after committing the files you want to share:

```powershell
git archive --format=zip --output=Q-TRACE-project.zip HEAD
```

The ZIP contains committed source, docs, and benchmark evidence, not `.venv`, Git history, credentials, or uncommitted edits. Your friend installs dependencies or runs Docker; don't send the local virtual environment.

## Run locally on Windows

The repository is pinned for Python 3.14. From PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the judge-facing interface. Model initialization runs in the background; `/health` reports `loading` until both disease modules are ready (12 quantum members and 16 classical fits in total).

Run the internal Streamlit console in a second PowerShell window:

```powershell
$env:QML_API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run dev-dashboard\app.py
```

The Streamlit application is an API client only. It does not train a second copy of the models.

## API contract

- `GET /health` — runtime readiness, loaded modules, selected feature names, and actual training configuration.
- `GET /diseases` — disease registry, dataset attribution/license, labels, schemas, and observed ranges.
- `GET /patients?disease_id=...` — held-out records for a reproducible module-specific demo.
- `POST /ingest` — validate and order one exact named record for the selected disease, with observed-range warnings.
- `POST /predict` — always returns the six-model `quantum` result plus both `classical.full_feature` and `classical.same_4_feature` results.
- `POST /explain` — returns attribution only for quantum, classical full-feature, or classical same-four views. Quantum defaults to `vqc_fast`; `allow_slow=true` explicitly enables the slower full ensemble. This response never contains a second confidence or probability.
- `POST /report` — returns a local, deterministic evidence report; the judge UI downloads HTML or opens Print/Save-PDF. No hosted LLM or paid API is used.
- `GET /baselines` and `GET /metrics` — module-specific benchmark/debug information used by the Streamlit console.

The live default uses seed 42, a **20-row quantum training pool**, and 100 epochs per VQC; classical models use all 455 training rows. The saved three-seed quantum benchmark uses a **200-row pool**. Both UIs disclose the live configuration. These are different experiments: a live prediction is not a reproduction of the saved benchmark, and matching four input features does not also match training sample counts. Keep those qualifications in any presentation.

The live settings are identified by `artifacts/models/runtime_manifest.json`. Startup deterministically refits that manifest; `/health` exposes its configuration ID and any resulting runtime settings. Complete malignant-focused three-seed classical metrics and fold-isolated five-fold evidence are generated with `python -m backend.evaluation` and retained under `artifacts/evaluation/`.

Inference accepts one prediction or explanation at a time. A competing request gets HTTP 429 with `Retry-After`; wait and retry manually. Request bodies over 16 KiB receive HTTP 413. The API is a single-process research demo, not a public clinical service.

The fixed-split WBCD features are `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`. The early-diabetes module selects `gender`, `polyuria`, `polydipsia`, and `partial paresis`. These are source columns, never PCA labels.

The UCI diabetes benchmark is bundled unchanged under CC BY 4.0 and cited in `backend/data/datasets/README.md`. It has 520 questionnaire records from a single hospital study and can show unusually high accuracy because several symptoms strongly overlap the target. This is a scalability demonstration, not proof of pre-symptomatic detection, Indian-population validity, fairness, or clinical readiness.

Its retained three-seed six-model ensemble measured 83.01% mean accuracy (76.92–92.31%), 81.25% mean condition sensitivity, and 85.83% mean specificity under the intentionally small 20-row quantum training budget. Full-feature classical models measured 90.06–95.51% mean accuracy. These non-superiority and variability findings are intentional parts of the evidence record, not numbers to hide; see `artifacts/evaluation/early_diabetes_three_seed.md`.

## SIH26139 requirement coverage

| Problem-statement objective | Implemented evidence |
|---|---|
| Data ingestion and preprocessing | Held-out WBCD catalog plus exact named 30-feature CSV intake; train-only imputation, scaling, ANOVA selection, and inward quantum-range scaling |
| Quantum-enhanced classification | Four data-reuploading VQCs plus angle/4-qubit and amplitude/2-qubit quantum-kernel SVMs; OOB weighted soft voting |
| Classical benchmarking | Logistic Regression, Random Forest, XGBoost, and SVM on all 30 and the identical selected 4 features |
| Accuracy, sensitivity, specificity | Malignant-focused precision/recall/F1, sensitivity, specificity, confusion counts, ROC-AUC, and timing in retained three-seed and five-fold artifacts |
| Explainability | Attribution-only SHAP and LIME for quantum fast/full and both classical feature views, always using real clinical names |
| Generalization and efficiency | Leakage-safe stratified five-fold evaluation, three-seed evidence, measured timing, and explicit experiment boundaries |
| Simulator / near-term path | Configurable PennyLane device abstraction; `lightning.qubit` and `default.qubit` tested. Real-QPU execution is a documented future validation step, not a current claim |
| Reproducibility and demonstration | Identified runtime manifest, pinned Python 3.14 environment, Docker Compose, FastAPI, Streamlit console, professional judge-facing UI, and automated regressions |

The authoritative evidence summary is `artifacts/evaluation/phase7_evaluation_summary.md`. The system remains a simulator-based research prototype on one public benchmark dataset; it is not prospectively or clinically validated.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q backend dev-dashboard tests scripts
node --check frontend\app.js
```

The Phase 4–6 implementation was verified on 2026-09-01 with 63 passing tests and a real browser walkthrough of prediction, disagreement, fast explanation, and the explicit slow opt-in. See [the integration record](artifacts/integration/phase4_6_verification.md).

The clinical-workspace redesign was verified on 2026-09-02 with 66 passing tests on Windows and inside the Linux Docker image. Live browser checks covered equal result cards, disagreement, real SHAP/LIME completion, patient/scope locking, default-scope reset, raw measurement display, and mobile layout. See [the redesign and Docker record](artifacts/integration/clinical_redesign_docker_verification.md).

The September 2 pre-push review increased the suite to **77 passing tests on both Windows and Linux**, verified the real fast/full explanation flows, fixed stale dashboard state and bounded expensive requests, and cleaned generated caches. See [the pre-push review](artifacts/integration/prepush_review_2026-09-02.md), including its security-review limitations at that time.

The SIH26139 evidence closure on 2026-09-04 now produces **93 passing tests**, including a separately passing 20-epoch VQC cost-decrease regression, complete disease-oriented classical reports, fold-isolated five-fold breast-oncology evidence, three-seed evidence for both modules, classical and quantum SHAP/LIME, a named runtime manifest, and browser walkthroughs. See [the earlier Phase 7 completion record](artifacts/integration/phase7_completion_2026-09-04.md) and the current diabetes evidence under `artifacts/evaluation/`. Docker Desktop itself failed to start during the earlier final run because its Windows host could not recreate a local Unix-socket reparse point; the app was therefore reverified through the pinned local Python runtime. This host failure did not change project files, and no factory reset was performed.

## Containers

With Docker installed:

```powershell
docker compose up --build
```

The API is available at `http://127.0.0.1:8000` and the Streamlit console at `http://127.0.0.1:8501`. Both bind to this computer only by default. Compose waits for trained-model readiness before starting the dashboard. Initial image creation needs internet access; model fitting repeats on startup.

The real Docker build and startup were verified on Docker Desktop 4.89.0 / Engine 29.7.2. The image keeps every requirement pin unchanged and uses one OpenMP thread for the small quantum state vectors. The API and Streamlit service each have their own health check.

If a newly installed per-user Docker Desktop is not found in the current PowerShell session, open a new terminal. For the per-user installation used by this project, the following adds its executable and standard credential helper to the current process only:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
docker compose up --build -d
```

No Docker account password belongs in the repository or Compose file.

### Optional trusted-network demo

Only on a trusted private network, you can deliberately expose the judge-facing app to friends:

```powershell
$env:QML_BIND_HOST = "0.0.0.0"
docker compose up -d
```

They use your computer's LAN IP and port 8000; Windows Firewall must allow the connection. The internal dashboard remains local. There is **no authentication or HTTPS**, so do not forward router ports, expose this directly to the public internet, or enter real identifiable patient information. To restore local-only access:

```powershell
Remove-Item Env:QML_BIND_HOST
docker compose up -d
```
