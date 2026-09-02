# Q-TRACE

Q-TRACE is a hybrid quantum-classical early disease signal analysis prototype for Smart India Hackathon problem statement SIH26139. It compares a six-member quantum ensemble (four VQCs and two quantum-kernel SVMs) with classical baselines on the Wisconsin Breast Cancer dataset.

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

Open `http://127.0.0.1:8000/` for the judge-facing interface. Model initialization runs in the background; `/health` reports `loading` until all four 100-epoch VQCs, both QSVMs, and the classical baselines are ready.

Run the internal Streamlit console in a second PowerShell window:

```powershell
$env:QML_API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run dev-dashboard\app.py
```

The Streamlit application is an API client only. It does not train a second copy of the models.

## API contract

- `GET /health` — runtime readiness, selected feature names, and actual training configuration.
- `GET /patients` — held-out WBCD records for a reproducible demo.
- `POST /predict` — always returns the six-model `quantum` result plus both `classical.full_feature` and `classical.same_4_feature` results.
- `POST /explain` — returns attribution only. The default `allow_slow=false` scope is `vqc_fast`; `allow_slow=true` explicitly enables the slower full ensemble. This response never contains a second confidence or probability.
- `GET /baselines` and `GET /metrics` — benchmark/debug information used by the Streamlit console.

The live default uses seed 42, a **20-row quantum training pool**, and 100 epochs per VQC; classical models use all 455 training rows. The saved three-seed quantum benchmark uses a **200-row pool**. Both UIs disclose the live configuration. These are different experiments: a live prediction is not a reproduction of the saved benchmark, and matching four input features does not also match training sample counts. Keep those qualifications in any presentation.

Inference accepts one prediction or explanation at a time. A competing request gets HTTP 429 with `Retry-After`; wait and retry manually. Request bodies over 16 KiB receive HTTP 413. The API is a single-process research demo, not a public clinical service.

The four selected clinical features on the fixed split are `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q backend dev-dashboard tests
```

The Phase 4–6 implementation was verified on 2026-09-01 with 63 passing tests and a real browser walkthrough of prediction, disagreement, fast explanation, and the explicit slow opt-in. See [the integration record](artifacts/integration/phase4_6_verification.md).

The clinical-workspace redesign was verified on 2026-09-02 with 66 passing tests on Windows and inside the Linux Docker image. Live browser checks covered equal result cards, disagreement, real SHAP/LIME completion, patient/scope locking, default-scope reset, raw measurement display, and mobile layout. See [the redesign and Docker record](artifacts/integration/clinical_redesign_docker_verification.md).

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
