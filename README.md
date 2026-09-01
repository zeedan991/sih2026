# Q-TRACE

Q-TRACE is a hybrid quantum-classical early disease signal analysis prototype for Smart India Hackathon problem statement SIH26139. It compares a six-member quantum ensemble (four VQCs and two quantum-kernel SVMs) with classical baselines on the Wisconsin Breast Cancer dataset.

The project is a research prototype, not a medical device. It deliberately presents quantum and classical results side by side and does not claim that quantum ML beats classical ML on raw accuracy.

## Run locally on Windows

The repository is pinned for Python 3.14. From PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the judge-facing interface. Model initialization runs in the background; `/health` reports `loading` until all four 100-epoch VQCs, both QSVMs, and the classical baselines are ready.

Run the internal Streamlit console in a second PowerShell window:

```powershell
$env:QML_API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run dev-dashboard\app.py
```

The Streamlit application is an API client only. It does not train a second copy of the models.

## API contract

- `GET /health` — runtime readiness and selected feature names.
- `GET /patients` — held-out WBCD records for a reproducible demo.
- `POST /predict` — always returns the six-model `quantum` result plus both `classical.full_feature` and `classical.same_4_feature` results.
- `POST /explain` — returns attribution only. The default `allow_slow=false` scope is `vqc_fast`; `allow_slow=true` explicitly enables the slower full ensemble. This response never contains a second confidence or probability.
- `GET /baselines` and `GET /metrics` — benchmark/debug information used by the Streamlit console.

The four selected clinical features on the fixed split are `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q backend dev-dashboard tests
```

The Phase 4–6 implementation was verified on 2026-09-01 with 63 passing tests and a real browser walkthrough of prediction, disagreement, fast explanation, and the explicit slow opt-in. See [the integration record](artifacts/integration/phase4_6_verification.md).

## Containers

With Docker installed:

```powershell
docker compose up --build
```

The API is exposed on port 8000 and the Streamlit console on port 8501. Compose waits for trained-model readiness before starting the dashboard.

