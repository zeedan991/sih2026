# Roadmap — mapped to the real SIH 2026 calendar

Today: **August 29, 2026**. SIH 2026 launched August 21; college-level internal hackathons run through September, and SIH26139's own listed idea-submission deadline is **September 20, 2026** — that's about 3 weeks out (0/500 ideas submitted for this PS as of this check, so the window is genuinely open, not filling up). If your team clears the internal round, national screening (PPT + video) follows in October, with the Grand Finale a 36-hour build in December. This roadmap targets a strong, honest, fully-working core by the September deadline — not a rushed attempt at everything in the concept doc at once. (The "Day 1, Day 2..." labels below are relative to whenever your team actually starts building, not fixed calendar dates.)

Build order deliberately front-loads the highest-risk, least-familiar piece (the quantum circuit) and treats the multimodal extension as something you *earn the right to attempt*, not a default.

---

## Phase 0 — Setup (before Day 1)

- Clone repo, add `AGENTS.md`, `prd.md`, `architecture.md`, `roadmap.md`, `decisions.md` to the root
- Create a Python 3.14 virtualenv and run `pip install -r requirements.txt` — this exact file was tested end-to-end on Python 3.14 (see `architecture.md` §2 and `decisions.md` D-19). The pinned XGBoost requires Python 3.12 or newer.
- Assign roles against `architecture.md` §5's folder structure. Recommend **at least 2 people** ramp up on `backend/quantum/` specifically — it's the piece with no ready-made template and the one place a silent bug (see below) can eat days
- Everyone reads `decisions.md` D-02, D-03, and D-15 once. These are verified silent-failure bugs — code that runs cleanly and even looks numerically fine, but is wrong. Know them going in so nobody rediscovers them the hard way, and so nobody "fixes" this file back to the buggy pattern while refactoring.

## Phase 1 — Core VQC + classical baselines (Days 1–2)

**Goal:** one working VQC, trained correctly, benchmarked against classical models, on real data, end to end.

- Day 1: data pipeline (`backend/data/pipeline.py`) using **feature selection, not PCA** (`SelectKBest(f_classif, k=4)` — `architecture.md` §3.2, `decisions.md` D-13) + one VQC circuit using the re-upload + weight-remap pattern (`architecture.md` §3.3, **without** `interface="torch"`, with the corrected loss function per `decisions.md` D-02/D-03). Keep `y` (for classical) and `y_pm1` (for quantum) as two separate arrays — never overwrite one with the other (`architecture.md` §3.2).
- Day 1, before anything else builds on top of it — **write two regression tests, not one:**
  1. Cost-decreases: train 20 epochs, assert cost at epoch 20 < cost at epoch 0 (`decisions.md` D-02).
  2. Label-encoding direction: assert `sklearn`'s `target_names[0] == "malignant"` and `[1] == "benign"`, and assert a known-benign feature vector round-trips to the string `"benign"` through your actual prediction code, not just a number above 0.5 (`decisions.md` D-15). This one matters just as much as the first — it's the kind of bug that passes every accuracy metric and only shows up as an obviously-wrong diagnosis live in front of judges.
- Day 2: classical baselines in **both configurations** — full 30 features and the same 4 selected features the quantum path sees (PRD M3, `decisions.md` D-14) — 3-seed variance check on both.

**Exit criteria for Phase 1:** cost curve actually decreases (not flat), test accuracy in a plausible range (verified achievable: ~93% for this circuit design on selected features — `decisions.md` D-11, D-13 — so treat anything near 50% as broken, not "just unlucky"), both regression tests pass, both classical configurations run cleanly. Do not proceed to Phase 2 with a broken core.

## Phase 2 — Full quantum ensemble: VQCs + quantum-kernel SVMs (Days 3–4)

- Implement the 4 VQC variants **and** the 2 quantum-kernel SVM variants (`architecture.md` §3.4, §3.5) — both paradigms the PS names, not VQC alone. **The amplitude-embedding QSVM variant uses 2 qubits, not 4** — `AmplitudeEmbedding` needs exactly 2ⁿ input values, and 4 qubits would need 16 values against only 4 features available. This isn't a style choice, it would fail outright at 4 qubits (`decisions.md` D-10, `architecture.md` §3.5).
- Train every Phase 2 VQC for **at least 100 epochs**. The 20-epoch duration exists only for the fast D-02 cost-decrease regression; the controlled Phase 1 rerun showed that 100 epochs materially improves cross-seed convergence (`decisions.md` D-20).
- The QSVM path is a different shape of work than the VQC path: no training loop, instead a kernel-matrix computation (`architecture.md` §3.4's kernel_matrix function) feeding `sklearn.svm.SVC(kernel="precomputed")`. Budget real time for this — it's O(n²) in sample count, measured at ~17s for 120×120, so plan for minutes rather than seconds at full training-set scale
- Out-of-bag error → inverse-MSE weighting → weighted soft voting across all 6 models
- Log **every individual model's** accuracy and assigned weight, split out by type (VQC vs. QSVM) — this table is genuinely interesting on its own (verified in this project: the angle-embedding QSVM reached 95.6% vs. the VQC's ~93%, the closest either quantum approach has come to classical, and statistically tied with classical when classical is limited to the same 4 features — `decisions.md` D-14)
- Statistical test (paired t-test, PRD S4) between ensemble and best single model

**Exit criteria:** ensemble accuracy ≥ best single model's accuracy (VQC or QSVM, whichever wins), with the comparison numbers written down, whichever way they land.

## Phase 3 — Explainability (Day 5)

- SHAP `KernelExplainer` wrapping the ensemble's predict function — **revised timing, verified on the actual mixed ensemble: a VQC-only explanation is fast (~7s for 3 VQCs), but jumps to ~70s once a QSVM model joins the explained ensemble** (`architecture.md` §3.7, `decisions.md` D-17) — each SHAP perturbation needs a fresh QSVM kernel computation against the background set. Build a real "computing explanation..." state, not a brief spinner, and default to explaining against the fast VQC sub-ensemble with the full 6-model explanation as an opt-in "deep explanation."
- LIME as a cross-check (PRD S2)
- Because the pipeline now selects real named features instead of PCA components (`decisions.md` D-13), SHAP's `feature_names` are literally `mean concave points`, `worst radius`, `worst perimeter`, `worst concave points` (verified on this project's split) — confirm your explainability code passes these through rather than defaulting to generic `feature_0`-style labels, which is the easy way to accidentally lose this benefit
- Confirm SHAP runs against **both** a VQC prediction and a QSVM prediction — the explainability layer should treat both quantum model types the same way

**Verified 2026-08-30:** Phase 3 now has a VQC-only default, explicit slow consent for both QSVM-inclusive scopes, structured long-running progress events, SHAP additivity audits, and a LIME patient-contribution cross-check. A real fitted-ensemble smoke test succeeded for VQC-only, QSVM-only, and full-six SHAP using the exact four selected clinical names. The small verification settings completed faster than the production-like D-17 benchmark; the QSVM-inclusive contract intentionally retains its "at least 70 seconds" warning.

## Phase 4 — Backend + dev dashboard (Days 6–7)

- FastAPI routes per `architecture.md` §4 — confirm every `/predict` response has **both** `quantum` and `classical` populated, never one omitted (`decisions.md` D-12, PRD M4). Write the test for this explicitly, not just eyeball it once.
- Streamlit dev dashboard hitting the same API (internal use only — `decisions.md` D-04)
- `pytest` coverage for the API routes

**Verified 2026-09-01:** FastAPI now serves the model runtime and the judge-facing frontend from one origin. Contract tests explicitly require the quantum result and both classical configurations on every prediction. `/explain` is attribution-only under D-23. The Streamlit console is an HTTP-only client of this API, so it cannot drift into a second model runtime.

## Phase 5 — Judge-facing frontend (Days 8–9)

- Build against the design system in `architecture.md` §6 — the two-dial layout (quantum indigo / classical teal, equal size, always both populated) is the actual spec here, not a style preference. See the rendered mockup referenced in `architecture.md` §6.1.
- Implement the disagreement banner (PRD S5) — cheap to build, and it's a genuinely honest feature: when quantum and classical predict different labels, say so rather than silently picking one
- This is a small number of screens (upload/select → results with both dials → explanation), not a large app — resist scope growth here

**Verified 2026-09-01:** A live browser run rendered equal 582 px result cards with the required indigo and teal tokens, triggered the amber banner on a genuine model disagreement, held a real elapsed-time explanation state through computation, and rendered four real-name SHAP/LIME attribution rows without a competing confidence value. The full six-model explanation is a visibly slower, explicitly confirmed opt-in. Browser console errors and warnings were empty.

## Phase 6 — Integration, Docker, rehearsal (Day 10)

- `docker compose up` from a clean checkout, tested by someone who didn't write the code
- Full rehearsal with **real, non-hardcoded inputs** — pick 3–4 real WBCD test rows in advance you're comfortable narrating, but run the actual pipeline live, not a canned screen
- Rehearse the "why not just classical ML" answer (`decisions.md` D-07) out loud, as a team, at least once

**Engineering verification 2026-09-01:** The full suite passes (63 tests), `pip check` is clean, Python compilation succeeds, both local applications return HTTP 200, and a real model/browser flow passes. Dockerfile and Compose behavior are contract-tested, but the clean image build remains pending because Docker is not installed on the verification host. Human demo rehearsal remains a team activity rather than a code task.

**By here you have a complete, honest, working core.** Everything past this point is optional and time-boxed.

## Phase 7 — Buffer + optional bonus (remaining days before Sept 20)

- First priority for any remaining time: polish, bug-fix, and rehearse the core again. A polished Must-have beats a half-built Should-have.
- **Only if the core is fully solid:** attempt the multimodal extension (PRD C1). Time-box it explicitly — e.g., "we attempt this only if Phase 6 is done by day 12; otherwise it becomes a documented future-roadmap slide, not a half-built demo feature." A clearly-explained future direction reads better to judges than a broken bonus feature.
- Prepare the docx-based submission material and the pitch deck from the same verified numbers used throughout — no new claims introduced at this stage that weren't tested earlier.
- **Download and use the official SIH 2026 PPT template from sih.gov.in — don't submit `Project_Blueprint_SIH26139.docx` as-is.** The internal round requires the idea presentation in the official template format (confirmed: colleges require submissions "strictly as per the SIH 2026 PPT format"). The blueprint docx is your source content — pull the verified numbers, architecture summary, and honest framing from it into the mandated slide template, not the other way around.
- Confirm your team's registration status early: internal-round rules for this cycle require a 6-member team (at least 1 female member) with a faculty mentor, and most colleges run their own pre-screening before nominating to the national portal — don't assume selecting this PS in the "up to 2 problem statements" step guarantees a slot without clearing your college's own internal round first.

## Phase 8 — If selected: October–November window

- National screening needs a PPT + video, not new code — reuse the docx content and the live demo recording
- Use this window to harden whatever was cut in Phase 7 (multimodal, noise-aware simulation via `default.mixed`, or a real-hardware run via a PennyLane hardware plugin — PRD C1–C3)

## Phase 9 — December Grand Finale (36-hour build)

- By this point you should already have a working system from September; the 36 hours are for final integration, live-demo hardening, and responding to any judge feedback from screening — not building from scratch
- Keep a "known good" tagged commit you can roll back to at any point during the 36 hours if an experiment goes wrong

---

## Explicit cut-lines (decide these *before* you're under time pressure, not during)

| If behind schedule at... | Cut this first | Keep this no matter what |
|---|---|---|
| End of Phase 2 | Reduce ensemble from 6 → 2 models (1 VQC + 1 QSVM — keep both paradigms even if you cut count) | Single working VQC + baselines (Phase 1) |
| End of Phase 5 | Custom frontend → fall back to polished Streamlit only | Working backend + SHAP explanation |
| Phase 7 | Multimodal extension entirely | Everything through Phase 6 |
