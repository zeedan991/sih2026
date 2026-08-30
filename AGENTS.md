# AGENTS.md

Standing instructions for any AI coding agent (Antigravity, Claude Code, or otherwise) working in this repository. Read this before starting any task. Antigravity reads this file automatically at session start.

## Non-negotiables (in priority order)

1. **Never reintroduce the D-02 bug.** Quantum circuits (`backend/quantum/`) never use `@qml.qnode(dev, interface="torch")` together with `qml.AdamOptimizer`. This combination runs without error but silently fails to train — verified. If you touch any circuit file, run `pytest tests/test_circuits.py -k cost_decreases` before committing. If that test doesn't exist yet, write it first: train 20 epochs, assert cost at epoch 20 < cost at epoch 0.
2. **Never get the label direction backwards (D-15).** `sklearn`'s breast cancer dataset encodes `0=malignant, 1=benign` — the reverse of what's intuitive. This bug passes every accuracy metric (which compares integers, not strings) while showing an inverted diagnosis live. `tests/test_pipeline.py` must assert `target_names[0]=="malignant"`, and any code that turns a prediction into a "benign"/"malignant" string must be covered by a test that checks the string, not just a number crossing 0.5.
3. **Never claim the quantum model beats classical ML on raw accuracy** in code comments, docstrings, UI copy, or generated docs. See `decisions.md` D-07. Every accuracy number in the codebase must be traceable to an actual test run in this repo, reported as a range across seeds — not a single number, and never copied from the original concept doc's reference list without independent verification (`decisions.md` D-08).
4. **`/predict` must always return both `quantum` and `classical`, fully populated, every time** — and `classical` has two sub-objects (full-feature and same-4-feature, D-14), both always populated too. Never make any of these conditional, optional, or nested as secondary. See `decisions.md` D-12 for the precise justification (PS objectives 3 and 6 made visible — not a claim about the word "hybrid" itself, that framing was corrected). If you're building any UI against this endpoint, quantum and classical results render at equal visual size — quantum in indigo (`--quantum-indigo`), classical in teal (`--signal-teal`), per `architecture.md` §6.4.
5. **Quantum means both VQC and quantum-kernel SVM, not VQC alone.** The PS names QSVM, QNN, and VQC explicitly; the ensemble spans both the variational family and the kernel family (`architecture.md` §3.6). Don't quietly drop the QSVM half because it's less familiar — it's currently the stronger of the two quantum approaches (verified: 95.6% vs. ~93%, `decisions.md` D-10). Note the amplitude-embedding QSVM variant specifically needs **2 qubits, not 4** — `architecture.md` §3.5 explains why; using 4 would fail outright, not just underperform.
6. **Feature selection, not PCA (D-13).** The pipeline uses `SelectKBest(f_classif, k=4)`, not `PCA`. This isn't cosmetic — PCA output has no clinical meaning and breaks the explainability requirement (M5); feature selection keeps real names (`mean concave points`, `worst radius`, etc.) all the way through. If you find PCA anywhere in the quantum data path, that's stale code from an earlier revision — replace it.
7. **SHAP explanations are slow once QSVM is involved — design the UI for it, don't fight it (D-17).** Verified: ~7s for a VQC-only explanation, ~70s once a QSVM model joins the explained ensemble. Default the "explain this prediction" feature to the fast VQC sub-ensemble; make the full 6-model explanation an explicit, clearly-slower opt-in. Don't spend effort trying to make the full-ensemble explanation feel instant — it isn't, and pretending otherwise with just a spinner is a worse UX than being upfront about the wait.
8. **`SVC(probability=True)` raises a `FutureWarning` on our pinned scikit-learn — expected, not a bug (D-18).** Don't "fix" it by upgrading scikit-learn without reading D-18 first; the upgrade path (`CalibratedClassifierCV`) is a different API shape, not a drop-in swap.
9. **Read `prd.md`, `architecture.md`, and `decisions.md` before generating an implementation plan** for any non-trivial task. They contain the verified tech stack, the API contract, the design system, and the reasoning behind choices already made — don't re-derive decisions that are already recorded.
10. **Follow the phase order in `roadmap.md`.** Don't start the multimodal extension (Phase 7 / PRD "Could have") before Phases 1–6 pass their exit criteria. If asked to build a bonus feature out of order, flag it rather than silently complying.

## Stack (detect, don't assume)

This is a Python (FastAPI + PennyLane + scikit-learn) backend with two frontends: a Streamlit dev dashboard and a custom HTML/CSS/JS judge-facing UI. Exact pinned versions are in `requirements.txt` — install with `pip install -r requirements.txt`, don't freelance different versions. If a task looks like it needs a library not in that file, check `decisions.md` for whether it was deliberately excluded (e.g. `torch`) before adding it.

## Before making changes

- Check `git status` and current branch before starting
- For anything touching `backend/quantum/`: run the cost-decreases regression test (see Non-negotiable 1) after your change, not just before
- For anything touching `backend/main.py` (API routes): the contract in `architecture.md` §4 is what the frontend(s) depend on — changing a response shape breaks both UIs, so update all three together or don't ship partial

## Test / build commands

```bash
pytest tests/                          # full backend test suite
pytest tests/test_circuits.py -v       # quantum circuit regression tests specifically
uvicorn backend.main:app --reload      # run the API locally
streamlit run dev-dashboard/app.py     # run the internal debug dashboard
docker compose up --build              # full stack, clean build
```

If any of these commands stop matching reality (renamed a file, changed a port), update this file in the same commit — a stale command here is worse than no command. Note: `requirements.txt` includes `httpx2` specifically for `test_api.py`'s use of FastAPI's `TestClient` — this was a real, non-obvious dependency found during testing, not guessed.

## Design constraints for the frontend

`architecture.md` §6 has the full design system (colors, type, the measurement-dial signature element). Don't default to a generic dashboard/admin-template look — that section explains specifically what to avoid and why. If you're generating UI and haven't read that section in this session, read it before writing markup.

## What "done" looks like

Match against `prd.md` §6 (Success criteria) and the relevant phase's exit criteria in `roadmap.md` — not against "the code runs without an error." D-02 is the standing reminder of why those aren't the same thing here.

## Escalate, don't guess

If a requirement in `prd.md` conflicts with something in `architecture.md`, or a decision in `decisions.md` seems wrong given something you've learned, say so explicitly and propose an update to the relevant doc rather than silently picking one. These files are living documents, not immutable specs — but changes to them should be visible, not implicit in code.
