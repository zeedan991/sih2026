# Clinical-workspace redesign and Docker verification — 2026-09-02

## Scope

Replaced the promotional frontend with a compact clinical research workspace while preserving the model and API contracts. The page now prioritizes the patient record, paired equal-size model assessments, concise clinical measurements, and attribution. The original indigo/teal tokens, circuit rail, amber disagreement, both classical configurations, research disclaimer, and full-ensemble opt-in remain intact.

No quantum model definitions, weights, random seeds, training data, training budgets, or package versions were changed. Source Sans 3 replaces the serif body type as recorded in D-24. The request-state guards prevent patient or scope changes while computation is active, and a generation check prevents stale responses from being displayed.

## Browser evidence against the real Docker API

- Patient WBCD-541: quantum benign at 64.7%, full-feature classical malignant at 63.4%, same-four classical benign at 60.7%.
- The genuine disagreement triggered the amber notice. Both model perspectives remained present.
- At the default browser viewport, both cards measured 278 × 376.9 px. Their strokes were exactly `rgb(74, 63, 140)` and `rgb(14, 124, 123)`.
- Fast SHAP/LIME completed with the elapsed display at 16 seconds. The loading state persisted, and patient selection plus both scope buttons were disabled during the request.
- The completed scope was `VQC FAST`, with all four clinical feature names. Raw measurements displayed as 0.0389, 16.22, 113.5, and 0.1205 for mean concave points, worst radius, worst perimeter, and worst concave points respectively.
- No percentage/confidence value was displayed in the explanation panel.
- Selecting full ensemble exposed the explicit 70-seconds-or-longer disclosure. Changing to WBCD-261 cleared the prior prediction/explanation and restored the fast default before a new live inference.
- At the 390 px mobile test size, cards stacked and remained equal at 343.2 × 391.25 px. There was no horizontal overflow. The temporary viewport override was reset after verification.
- Browser warning/error logs were empty.

## Docker findings and verification

- Docker Desktop 4.89.0 and Engine 29.7.2 were available through the per-user installation. The calling shell needed that installation's `resources/bin` directory on PATH for the standard credential helper; no user password was used or stored.
- Both images built successfully from Python 3.14 with the exact requirements file. `pip check` during the image build reported no broken requirements.
- A ten-forward-pass probe under the live workload took 0.156 seconds with default OpenMP parallelism and 0.015 seconds with `OMP_NUM_THREADS=1`. The image now uses one thread for these tiny state vectors. This is an overhead probe, not a claim that the complete application is ten times faster.
- The shared image's API health check was overridden for Streamlit to target its own port-8501 health endpoint.
- The API reached ready with six quantum members, eight classical fits, and the exact four selected clinical feature names. The live demo continues to use the documented 20-row training pool with four 100-epoch VQCs; the separate three-seed reference benchmark uses a 200-row pool.
- Final `docker compose up --build -d` completed successfully. Both `api` and `dev-dashboard` reported healthy; the Streamlit health endpoint returned HTTP 200 and `ok`. A final container `pip check` reported no broken requirements.

## Automated checks

Windows Python 3.14, entire test suite:

```text
66 passed, 5 warnings in 9.82s
```

Linux Python 3.14 inside the built Docker image, read-only source mount:

```text
66 passed, 5 warnings in 6.93s
```

Warnings are the existing upstream SHAP/Matplotlib pending deprecations and the expected scikit-learn SVC probability deprecation covered by D-18. Tests were run without the optional pytest cache writer to avoid the unrelated Windows ownership warning.
