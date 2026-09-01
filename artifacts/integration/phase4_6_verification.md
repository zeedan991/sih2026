# Phases 4–6 verification — 2026-09-01

Environment: Windows, Python 3.14.0, pytest 9.1.1.

## Live model service

- `/health`: `ready`, 6 quantum members, 8 classical model/configuration combinations.
- Selected clinical features: `mean concave points`, `worst radius`, `worst perimeter`, `worst concave points`.
- Runtime configuration: four VQCs at 100 epochs, training pool limit 20, both QSVM variants enabled.
- The tested held-out patient 541 produced a real disagreement: quantum `benign` at 64.7% confidence, full-feature classical `malignant` at 63.4%, and same-four classical `benign` at 60.7%.
- The quantum response contained all six per-model records. The classical response contained both required configurations.
- Fast `/explain` completed in 27.3 seconds on this host. It returned `scope=vqc_fast`, four SHAP values, four LIME values, and the four real clinical feature names. Recursive inspection found no `confidence` or `probability` field.

The explanation duration is hardware-sensitive. The UI keeps a real elapsed-time state visible rather than assuming the earlier ~7-second reference applies to every machine.

## Browser walkthrough

The judge-facing application was exercised through the in-app browser against the live service:

1. Loaded the real held-out patient catalog.
2. Ran hybrid inference and rendered the result screen.
3. Confirmed the quantum and classical cards were exactly equal-sized at the tested viewport: 582 × 591.6 px each.
4. Confirmed quantum stroke `rgb(74, 63, 140)` and classical stroke `rgb(14, 124, 123)`.
5. Confirmed the real disagreement displayed the amber banner.
6. Started fast attribution and observed the computing state with a live elapsed timer.
7. Waited for the real SHAP/LIME response and confirmed four attribution bars with the real clinical names.
8. Confirmed the explanation panel had no percentage, confidence, or probability value.
9. Selected the full-ensemble option and confirmed the `70 seconds or longer` disclosure and explicit request label.
10. Checked browser console warnings/errors: none.

## Automated verification

```text
platform win32 -- Python 3.14.0, pytest-9.1.1, pluggy-1.6.0
collected 63 items
======================= 63 passed, 6 warnings in 13.34s =======================
```

The warnings were three upstream SHAP/Matplotlib pending deprecations, the expected scikit-learn `SVC(probability=True)` FutureWarning covered by D-18, and a non-functional pytest cache warning caused by the Windows account ownership boundary. `pip check` reported no broken requirements, and `compileall` completed successfully.

## Container limitation

The Dockerfile and Compose contracts are covered by tests, but the image was not built on this machine because the Docker CLI is not installed (`docker` is not recognized). A clean `docker compose up --build` remains the one host-dependent verification step.

