# Pre-push repository review — September 2, 2026

Target: `https://github.com/zeedan991/sih2026`, branch `main`. Working hackathon deadline: **September 8**, supplied by the user. Final application-source commit reviewed: `bc3458b`; this record is a subsequent documentation-only commit.

## Scope and limits

Reviewed the first-party Python backend, quantum/classical algorithms, preprocessing, explainability, both frontends, tests, requirements, Docker configuration, project specifications, and retained benchmark/integration reports. Inventoried the folder including ignored files. Read the original Word blueprint's document XML and checked its archive XML parts for common credential patterns; preserved it as historical source, without rewriting or claiming visual layout QA.

The local `.venv` was preserved and its installed direct pins and dependency consistency checked. Its third-party source was not exhaustively audited. The formal Codex Security service failed to create a usable scan context; **no formal security scan completed**, and no current vulnerability-advisory database scan was performed. The manual review and pattern checks below are not a security certification or proof that every possible secret/vulnerability is absent.

## Changes and regression evidence

- Added a 16 KiB request-body bound before JSON parsing, including requests without a reliable Content-Length. Oversized bodies receive HTTP 413.
- Allowed one expensive prediction/explanation at a time; competing calls receive HTTP 429 and `Retry-After: 2`, without blocking health/catalog access. The inference slot is released on errors.
- Cleared Streamlit prediction/explanation state when the patient or scope changes; handled API failures visibly rather than leaving a stale result or stack trace.
- Kept frontend readiness polling until the patient catalog actually loads.
- Added actual live seed/epoch/training counts to `/health` and both UIs; distinguished the 20-row live quantum pool from the separate 200-row saved benchmark and 455-row classical training set.
- Made Docker local-only by default. Trusted-LAN API binding is explicit; the developer dashboard remains on loopback. This is not a public authenticated deployment.
- Corrected a loading-state sentence: QSVM perturbation predictions evaluate kernels against fitted training samples, not against the SHAP background as if it were the fitted reference set.
- Recorded the September 8 deadline, real remaining research gaps, D-23's patient-ID evidence correction, and limitations of the statistical/generalization claims. No model/weight retuning or dependency substitution occurred.

Tests for request bounds, concurrent inference and dashboard stale-state failures were run first and failed before their fixes. Deployment binding assertions also failed before the Compose change and passed afterward.

## Actual test output

Final Windows run, Python 3.14 virtual environment:

```text
77 passed, 5 warnings in 12.36s
```

Final Linux run, rebuilt project Docker image with the source mounted read-only:

```text
77 passed, 5 warnings in 10.81s
```

Commands:

```text
python -m pytest tests -q -p no:cacheprovider
python -m pip check
node --check frontend/app.js
git diff --check
git fsck --full
```

The full suite includes the cost-decrease, label-direction, amplitude zero-vector, six-member/OOB, explanation scope, dual-classical API, Streamlit AppTest, frontend contract, and deployment regressions. Baseline before this review was 66 passing tests. The five warnings remain upstream SHAP/Matplotlib pending deprecations and the expected D-18 SVC probability warning. `pip check` reported **No broken requirements found**; all **15** direct requirement pins matched the active environment. JavaScript syntax, whitespace checks, and Git object integrity passed.

Both Docker images rebuilt and both services reached healthy on loopback ports 8000 and 8501. Existing dependency build layers were reused; this is not a fresh uncached build on a different person's computer. The final frontend was reloaded after the last image refresh.

## Real browser/API verification

The real six-member runtime loaded alongside eight classical models: seed 42, 100 VQC epochs, 20 quantum training rows, 455 classical training rows. These settings were visible in the page and `/health`.

On public held-out WBCD record 541, real inference produced quantum **benign (64.7% confidence)**, full-feature Logistic Regression **malignant (63.4%)**, and same-four-feature Logistic Regression **benign (60.7%)**. These are one patient's model confidences, **not accuracy measurements or clinical risk estimates**. The amber disagreement banner appeared without forcing or faking model output. Both result cards measured approximately **430.4 × 407.15 CSS pixels**, confirming equal visual size.

Both fast VQC SHAP/LIME and opted-in full-six SHAP/LIME completed with the real names `mean concave points`, `worst radius`, `worst perimeter`, and `worst concave points`. Their panels showed direction/bars and scope, not another confidence. The full explanation remained visibly computing at **01:58 elapsed**, then completed on a later check; exact completion duration was not captured. This reinforces the fast default and the need to allow several minutes for the full path.

During the full explanation, a real concurrent prediction received **HTTP 429**, `Retry-After: 2`, while `/health` still reported ready. Changing to record 256 cleared prior results and restored the fast scope after a new analysis. Browser error/warning logs were empty at the final flow check. The Streamlit health endpoint returned HTTP 200; its state/error regressions passed using Streamlit's AppTest.

## Cleanup and publication hygiene

- Removed **42 generated files, 578,526 bytes** across the root pytest cache and seven first-party Python bytecode cache directories. They can be regenerated by running Python/tests.
- Preserved `.git`, the working `.venv`, all source, the original blueprint, empty architecture placeholder directories, and all benchmark/verification evidence.
- Expanded Git/Docker exclusions for environments, generated caches, common credential files, disposable archives, logs, and OS metadata.
- Checked common token/private-key/password-assignment patterns across **36 reachable commits** at `bc3458b`: **zero matched file references**. Word archive XML checks also found no common token/private-key patterns. No matching credential/archive filenames were present outside Git/the local environment.
- At that commit, 51 tracked project files contained no virtual environment, cache, credential file, or ZIP archive. The new review report adds one documentation file.
- Verified origin points to the user's requested repository; `git ls-remote --heads origin` returned no branches before publication. Use an ordinary non-forced push and verify the remote `main` SHA afterward. Never place a Docker/GitHub password in the repository.

## What remains incomplete

The running demo is not a claim that every PRD objective has been met. Five-fold cross-validation, classical SHAP through an agreed API contract, and complete retained three-seed classical precision/recall/F1/confusion-matrix/ROC evidence remain gaps. Current binary recall uses benign as the positive class, not malignant sensitivity. No external cohort, prospective clinical validation, quantum-hardware execution, or independent second-person rehearsal was performed here.

Existing quantum three-seed benchmarks and the carried same-four-feature classical means were preserved, **not rerun or altered**. The ensemble showed a narrower observed VQC-family range and no demonstrated win over the strongest single model. A non-significant paired test does not prove equivalence. See `DEMO_GUIDE.md` and D-26 for the full, honest presentation boundary.
