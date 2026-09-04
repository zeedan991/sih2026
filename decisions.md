# Decisions — Architecture Decision Records (ADRs)

Each entry: what we decided, why, and what we gave up. Read this before overturning a choice made here — if you disagree with one, add a new entry rather than silently deviating, so the team (and any agent) can see the history.

---

### D-01. Keep FastAPI as a real backend, not a pass-through

**Context:** The original concept doc's architecture diagram shows Frontend → Backend API → Quantum Engine, but its own Phase 5 sample code has Streamlit importing model functions directly (`from models import quantum_predict, ...`), skipping FastAPI entirely.

**Decision:** FastAPI is a real, independently running, independently testable service. The frontend (whichever one — Streamlit or custom) only ever talks to it over HTTP.

**Why:** Decoupling means (a) the judge-facing custom UI and the internal Streamlit debug view can both hit the same backend without duplicating inference code, (b) the API is testable with `pytest` + `httpx` without spinning up any UI, (c) it matches what the team already told the problem-statement owner they were building.

**Cost:** A little more boilerplate on day one (an actual FastAPI app instead of direct function calls). Worth it — this is exactly the kind of shortcut that becomes a rewrite under deadline pressure later.

---

### D-02. Drop `interface="torch"` from the quantum circuits — verified critical bug

**Context:** The source doc's VQC is decorated `@qml.qnode(dev, interface="torch")` but trained with `qml.AdamOptimizer` (PennyLane's native, autograd-based optimizer) on `pennylane.numpy` weights — not `torch.optim` on `torch.Tensor` weights.

**Verification (not theoretical — actually run):** Executing this exact pattern for 60 optimizer steps produced a cost that was numerically frozen at `0.05293` for every single epoch. No exception was raised — it fails silently. Removing `interface="torch"` (so the qnode uses PennyLane's default autograd interface, matching what `qml.AdamOptimizer` expects) fixed it immediately: cost dropped from `0.313` to `0.306` in 60 quick epochs on a 40-sample subset, reaching 86% test accuracy.

**Decision:** No `interface="torch"` anywhere in the quantum circuits unless a future change genuinely needs PyTorch autograd (e.g., fusing a CNN feature extractor's gradients with the quantum circuit's in Phase 4). If that happens, switch consistently: torch weights, `torch.optim.Adam`, no `qml.AdamOptimizer`. Don't mix the two ecosystems.

**Why this matters enough to be first:** This is the failure mode that looks like success. The code runs, no error appears, training "completes" — and you'd only discover the model learned nothing when accuracy on real data comes back near chance level, possibly late in the build. `roadmap.md` puts a regression test for this on day one specifically because of how easy it is to reintroduce silently.

---

### D-03. Fix the label encoding in the loss function

**Context:** Labels are converted to `{-1, +1}` for the quantum measurement convention, but the cross-entropy formula `-mean(y*log(p) + (1-y)*log(1-p))` assumes `y ∈ {0,1}`.

**Verification:** Computed the loss both ways for a confidently-wrong prediction. As written: `15.20`. With `y` converted to `{0,1}` first: `7.60` — exactly half. The as-written version systematically over-penalizes one class relative to the other, which biases training rather than crashing it (harder to notice than D-02, easier to blame on "the model" rather than the loss function).

**Decision:** Always convert `y` to `{0,1}` immediately inside the cost function, right before the cross-entropy calculation, and keep `{-1,+1}` only for the raw measurement/PauliZ convention.

---

### D-04. Custom frontend for the judge-facing demo; Streamlit stays internal-only

**Context:** The brief calls for the web app to be "professional, high quality, eye-catching." Streamlit is excellent for fast internal iteration but has a recognizable default look that fights heavy customization.

**Decision:** Build two UIs against the same FastAPI backend (D-01 makes this cheap): a Streamlit dashboard for the team's own use while developing (fast to build, good for debugging model outputs), and a custom HTML/CSS/JS frontend — following the design system in `architecture.md` §6 — as the actual thing judges see and interact with.

**Alternative considered:** Heavily re-skin Streamlit with custom CSS injection instead of a separate frontend. Rejected as primary path because fighting a framework's default component styling under deadline pressure is a worse time trade than building a small number of static, hand-styled screens (upload, results, explanation) from scratch — there are really only 2-3 screens here, not a large app.

**Cost:** Two frontends to maintain instead of one. Mitigated by keeping the Streamlit one deliberately minimal (a debug tool, not a product) and by both sharing the same backend contract, so there's no logic duplication, only presentation duplication.

---

### D-05. Wisconsin Breast Cancer Dataset (WBCD) as the primary dataset; imaging/genomics is a bonus phase, not core

**Context:** The concept doc lists five datasets (WBCD, CBIS-DDSM, TCGA-BRCA, METABRIC, MIMIC-IV).

**Decision:** WBCD is the dataset the core deliverable is built, tested, and demoed against. CBIS-DDSM + TCGA-BRCA (the "Q RadFusion"-style multimodal extension) is attempted only after the core is fully working and polished — see `roadmap.md`'s explicit cut-line. METABRIC and MIMIC-IV are out of scope entirely (the latter requires PhysioNet credentialing with a training requirement, which is not a same-week turnaround).

**Why:** WBCD is 569 tabular samples, loads in one line (`sklearn.datasets.load_breast_cancer`), needs no specialized preprocessing, and is the dataset the PS objectives can be fully satisfied against (hybrid architecture, classical benchmarking, explainability). Imaging and genomic data need materially more data-engineering effort per unit of "does this satisfy the PS" — good bonus, bad foundation.

---

### D-06. PennyLane + `lightning.qubit`, not Qiskit or real quantum hardware

**Decision:** Simulation only, via PennyLane's `lightning.qubit` backend. No IBM Quantum / real-hardware job submission anywhere in the core deliverable.

**Why:** At 3–4 qubits, simulation is essentially instant and runs on any laptop — verified: <10ms for one circuit evaluation. Real quantum hardware queues, calibration drift, and shot noise would add engineering overhead with no upside at this problem size, and the PS itself explicitly permits and expects a simulator-based hybrid approach. PennyLane over Qiskit specifically because its autograd-native training loop (once D-02 is respected) is simpler to get right for a classifier-training use case, and its tutorials are oriented around exactly this kind of variational classifier.

**Revisit if:** the team wants a "we also ran this on real quantum hardware" slide for the finale — PennyLane supports IBM/AWS/other hardware backends via plugins with minimal circuit-code changes, so this is a plausible stretch goal for December, not September.

---

### D-07. Honest positioning: this does not claim to beat classical ML

**Context:** Independently verified published results on this exact dataset put VQC accuracy in the same band as classical models (~94–96%), not above them. Several numbers in the concept doc's own reference list (99% recall, 100% accuracy in places) were not independently verifiable in the form cited, and 100%-accuracy claims on small medical datasets are a known overfitting red flag, not typically a credible headline result.

**Decision:** The team's pitch and all documentation state plainly: the goal is a **rigorously benchmarked, interpretable, hybrid system that meets classical performance while satisfying the PS's explicit hybrid/explainability requirements** — not a claim of quantum superiority. Every accuracy number presented to judges must be one the team generated themselves on their own train/test split, reported with a range across seeds, not a single cherry-picked run.

**Why:** This is the more defensible pitch, not a weaker one. A judge asking "why not just use classical ML, which already gets 95%+ here" deserves a ready, honest answer — the PS asked for hybrid infrastructure and explainability, and that's what's being delivered.

---

### D-08. Citation policy for anything shown to judges

**Decision:** No statistic goes on a slide, in the docx, or in spoken pitch unless a team member has personally opened the source and can produce author name, venue, and year on request. Where a claim couldn't be independently verified in this form (several entries in the original reference list), it's rephrased as a range ("published work in this area reports accuracy in the low-to-mid 90s") rather than presented as a precise, specific citation.

**Verified independently and safe to cite as-is:** "Q RadFusion" (2025 preprint; AUC 0.96, 94% accuracy fusing mammography + genomic data via QAOA feature selection + VQC) and the general finding that VQCs on this dataset land in the ~94–96% range across at least two independently published sources.

**Not independently verified in the cited form:** the specific "95.71–96.43%" ensemble figure, "99% recall," "100% accuracy" (QSVM/QCNN) claims, and several journal attributions in the original reference list (vague venue names, no authors/DOIs — a pattern typical of AI-generated citation lists). Treat these as leads to verify, not facts to present.

---

### D-09. Success is defined before building, not after

**Decision:** The targets in `prd.md` §6 (accuracy ranges, not single numbers; working demo; ensemble beats best single model; SHAP output for at least one real prediction) are fixed now and not renegotiated upward mid-build to chase a more impressive-sounding number. If the ensemble beats classical baselines by a smaller margin than hoped, that's a reportable, honest result — not a failure to hide.

---

### D-10. Add a quantum-kernel SVM (QSVM) alongside the VQC ensemble — verified improvement

**Context:** Two things prompted this. First, the PS itself names three approaches — "QSVM, QNN, VQC" — and the previous revision only implemented one. Second, a literature check found real, repeated evidence that kernel-based quantum methods suit exactly this kind of small tabular dataset: SVMs generally are noted in the literature as well suited to small-data problems, and a compositional-optimization study on quantum kernels reports they can outperform standard classical kernels specifically in small-data regimes. A comparative study of QSVM vs. VQC on a different biomedical classification task (B-cell epitope prediction) found the two approaches have genuinely different strengths, supporting using both rather than picking one. **Re-confirmed on a later pass:** a hybrid QSVM→QNN(VQC) methodology on this exact dataset (Breast Cancer Wisconsin), using fidelity quantum kernels feeding into a variational classifier, appears in a peer-reviewed-track publication (PeerJ Computer Science, March 2026) — independent confirmation that combining these two paradigms on this exact dataset is an active, credible published approach, not just this project's own idea.

**Verification (not theoretical):** Implemented PennyLane's own documented quantum-kernel pattern (`qml.AngleEmbedding` + `qml.adjoint` fidelity kernel → `sklearn.svm.SVC(kernel="precomputed")`) and ran it on the same WBCD train/test split used throughout this project. Initial result (on PCA components): 94.7% test accuracy. **Re-verified after D-13 (switch to feature selection): 95.6%** — meaningfully ahead of the VQC's ~93%, and the closest any quantum approach has gotten to the classical baselines in this project.

**Decision:** Two of the six quantum-ensemble slots are quantum-kernel SVMs — one angle-embedding (4 qubits), one amplitude-embedding (**2 qubits**, corrected — see `architecture.md` §3.1 defect #3 and §3.5; the original plan of 4 qubits for this variant would have failed outright, since `AmplitudeEmbedding` needs exactly 2ⁿ input values and 4 qubits needs 16, not the 4 features available). The other four slots are VQCs. See `architecture.md` §3.6. No new dependencies required.

**Cost trade-off:** kernel matrix computation is O(n²) in sample count (measured: ~17s for a 120×120 matrix), versus the VQC's O(n) training. Budget more time for this step at full training-set scale — see `architecture.md` §3.4's cost note.

---

### D-11. Adopt data re-uploading + weight re-mapping in the VQC — verified improvement

**Context:** Literature search surfaced two well-established, complementary techniques for improving VQC accuracy without adding qubits: **data re-uploading** (Pérez-Salinas et al.) — re-encoding the classical input at every variational layer instead of once at the start, which increases circuit expressivity — and **weight re-mapping** (Kölle et al.) — mapping trainable weights through a periodic function into a bounded range before using them as rotation angles, since raw unbounded weights create ambiguous, harder-to-train parameter regions. The weight re-mapping paper reports up to a 10-percentage-point test-accuracy improvement on a benchmark dataset (Wine) from this change alone.

**Verification:** Trained two otherwise-identical 4-qubit/3-layer circuits on the same 200-sample/100-epoch/random-seed setup — one with the original single-encode design, one with re-uploading + sigmoid weight re-mapping added. Training cost dropped from 0.326 to 0.224 (meaningfully better fit), and test accuracy improved from 92.1% to 93.0%. Modest but real, and reproducible — not a cherry-picked run (same seed, same data, only the circuit design differed). Re-confirmed after D-13's switch from PCA to feature selection: accuracy held at ~93.0% on the new feature set, so the gain from this decision and the gain from D-13 are independent and additive, not double-counted.

**Decision:** Every VQC in the ensemble uses re-uploading + weight re-mapping by default now (`architecture.md` §3.3). This does not replace D-02's fix (no `interface="torch"`) or D-03's fix (label conversion before cross-entropy) — both still apply and were verified independently of this change.

**Honest framing:** this is a modest, not dramatic, improvement (+0.9 points in our test). Don't overstate it in the pitch — "we applied two published techniques and measured a real, if modest, gain" is a more credible claim than implying it was transformative.

---

### D-12. Quantum and classical results are always shown together, with equal visual weight

**Context:** The previous version of this entry justified the dual-display design by calling it "the literal, direct expression of hybrid" from the PS. On reflection (prompted by outside review) that's overstated: in the PS, "hybrid" specifically describes classical pre-processing feeding quantum models — which this design also does, upstream of display. The UI choice doesn't follow from the word "hybrid" by itself, and a judge could reasonably push back on that framing if stated that strongly.

**Decision (framing corrected, design unchanged):** Every `/predict` response always contains both a fully-populated `quantum` object and a fully-populated `classical` object (`architecture.md` §4), rendered as equal-sized dials, never one nested as primary. The actual justification: this makes PS objectives 3 ("benchmark accuracy/sensitivity/specificity vs. classical") and 6 ("rigorous benchmarking... generalization") *visible in the product* rather than buried in a report table someone has to go looking for. That's a defensible, specific claim — "we made our required benchmarking result impossible to miss" — rather than a rhetorical stretch about what one word in the PS supposedly demands.

**New behavior this adds:** when the two paradigms disagree on the predicted label, the UI surfaces this explicitly (an amber banner, `architecture.md` §6.2) rather than silently defaulting to one model's answer — a genuine, honest feature, not decoration.

**Why this matters for judging:** a judge checking whether classical benchmarking (objective 3) actually happened shouldn't have to dig for it. That's the claim to make — not a claim about what "hybrid" linguistically requires.

---

### D-13. Feature selection replaces PCA for the quantum path

**Context:** Three separate problems converged on the same fix. First, PS objective 5 says "feature selection" specifically — PCA is feature *extraction* (it builds new synthetic axes from combinations of all 30 features), not selection, so the previous design didn't literally match the PS's own wording. Second, PCA's output (principal components) has no clinical meaning — "PC1" isn't a thing a doctor recognizes — which put the design at odds with `prd.md`'s own success criterion that SHAP's top features be clinically recognizable (e.g. "mean concavity"). The only fix that preserves both PCA *and* clinical-naming would be running SHAP over the full 30 raw features with the entire preprocessing pipeline wrapped inside the explained function — verified to be a real cost, not a hypothetical one: this multiplies SHAP's per-patient cost well past whatever the baseline explanation cost already is (see D-17 for the current, corrected baseline — it turned out higher than first thought, which if anything strengthens this argument), because KernelExplainer's sample requirement scales with the number of explained features. Third, an outside review flagged that the previously-documented PCA variance figures didn't hold up — see D-16.

**Verification:** Replaced `PCA(n_components=4)` with `SelectKBest(f_classif, k=4)` in the pipeline and re-ran both quantum approaches on an identical split. Selected features: `mean concave points`, `worst radius`, `worst perimeter`, `worst concave points` — real, clinically-named, no abstraction. VQC accuracy: 93.0% (statistically unchanged from the PCA version). QSVM accuracy: 95.6% (up from 94.7% on PCA components — selection didn't just preserve accuracy, it improved the strongest model).

**Decision:** `SelectKBest(f_classif, k=4)` is now the default dimensionality-reduction step for both quantum paths (`architecture.md` §3.2). PCA is no longer used anywhere in the core pipeline. This simultaneously: matches the PS's literal objective-5 wording, resolves the SHAP-naming problem as a side effect rather than a workaround (`architecture.md` §6.3), and measurably helped rather than hurt accuracy.

---

### D-14. Classical baselines run on two feature sets, not one — the comparison must be fair in both directions

**Context:** An outside review raised a sharp, fair point: the previous revision's classical baselines trained on all 30 original features, while the quantum models saw only 4 PCA components (now 4 selected features). A technically literate judge could reasonably ask whether the "classical wins" result was actually about the algorithms, or just about classical models having 7.5x more information to work with.

**Verification:** Trained Logistic Regression both ways on the same split. All 30 features: 98.25%. The *same* 4 selected features the quantum models see: **92.98%** — statistically tied with the VQC's 92.98% on those same 4 features. This is a genuinely important result: a meaningful share of the apparent classical-vs-quantum gap in the previous revision was an artifact of an unequal comparison, not a demonstrated classical advantage.

**Decision:** Every classical baseline now runs in two configurations — full-feature (the strongest realistic classical benchmark, kept because it's what a hospital would actually deploy) and same-4-feature (the fair comparison against quantum). Both are always reported together (`architecture.md` §3.6, §4) — never just the full-feature number alone, which would flatter classical unfairly, and never just the same-4-feature number alone, which would hide that classical models can in fact use more information than the current quantum circuits can.

**Pitch implication:** this is a *better* story than the previous revision's, not a worse one — "when compared fairly on identical inputs, our quantum models are statistically competitive with classical ones; classical's advantage in the real-world configuration comes from using more input features, which is itself an interesting finding about where the actual gap lives" is a stronger, more specific claim than a flat "classical wins."

---

### D-15. Explicit, required test for the target-label encoding direction

**Context:** `sklearn.datasets.load_breast_cancer` encodes target `0` as malignant and `1` as benign — confirmed via `data.target_names`. This is the reverse of what many people would assume (0=bad/1=good is a common unstated prior). A naive label-to-string mapping written the "obvious" way, without checking, has a real chance of being backwards. Because accuracy metrics (`accuracy_score(y_test, predictions)`) compare integers to integers, they'd report a perfect-looking number even if every *displayed string* was inverted — this is exactly the kind of bug that survives a full test suite and only shows up live, in front of judges, as a diagnosis that's obviously wrong to anyone who knows the case.

**Verification:** Confirmed directly — `data.target_names == ['malignant', 'benign']`, i.e. index 0 is malignant. Traced this project's own label pipeline (`y_pm1 = y*2-1`, then `(raw_output+1)/2` as a "benign-ness" probability, then `"benign" if proba > 0.5 else "malignant"`) and confirmed the direction is currently correct throughout — but confirmed by hand-tracing, not by an automated check, which is not good enough for something this easy to get backwards silently.

**Decision:** `tests/test_pipeline.py` must include an explicit assertion of the encoding direction (`architecture.md` §3.2), and `tests/test_api.py` must include at least one test that sends a known-benign feature vector and asserts the API's string response says "benign," not just that some numeric field is above 0.5. This is now a Day 1 task in `roadmap.md`, alongside the D-02 regression test — both are in the same category: bugs that don't crash anything and don't show up in an accuracy number, only in a live demo, at the worst possible time.

---

### D-16. Corrected: the original concept doc's PCA-variance figure was mathematically impossible as stated

**Context:** The original idea.docx claimed roughly 56% cumulative variance at 12 PCA components, against ~80% at 3-4 components. Cumulative explained variance is monotonically non-decreasing in the number of components — 12 components cannot explain less variance than 3-4 components, by construction. This should have been caught earlier; it wasn't, because the figure was relayed from the source document rather than independently computed at the time.

**Verification:** Computed directly on this project's actual train split: 72.9% at 3 components, 79.6% at 4, 97.0% at 12. The 4-component figure already in use (~80%) held up; the 12-component figure never should have been repeated without checking.

**Decision:** Mostly moot now that D-13 replaces PCA entirely — but recorded here as a standing reminder: numbers relayed from the original AI-generated source document get independently recomputed before they're reused, the same standard already applied to its citations (D-08), not just its citations.

---

### D-17. SHAP timing revised upward once QSVM joins the explained ensemble — verified, not assumed

**Context:** Earlier timing notes (from before D-10's QSVM addition) measured SHAP against VQC-only ensembles and reported ~7-15s per explanation. That number was never re-verified after QSVM became part of the ensemble being explained — an oversight this final pre-build check caught.

**Verification:** Ran `shap.KernelExplainer` against a real 2-model ensemble (1 VQC + 1 angle-embedding QSVM, 50-point background, 60 perturbation samples). Result: **69.6 seconds** for one patient — roughly 10x the earlier VQC-only figure. The cause: every SHAP perturbation sample requires the QSVM component to compute a fresh kernel row against its full background set, which is far more expensive per call than a VQC forward pass. This scales further for the full 6-model ensemble (4 VQC + 2 QSVM).

**Decision:** The frontend must not assume a few-second wait for explanations once QSVM is in the mix. Two changes: (1) build a real "computing explanation..." UI state, not just a brief spinner (`architecture.md` §3.7); (2) default the "explain this prediction" feature to the fast VQC sub-ensemble, with the full 6-model explanation available as an explicit, clearly-slower "deep explanation" the user opts into, rather than the default path. If speed still matters more than this trade-off, reducing the QSVM background sample count (e.g. 100 → 30-40) is the next cheapest lever, at some cost to explanation stability — measure before committing to a number.

---

### D-18. `SVC(probability=True)` deprecation — forward-compatibility note, not an active bug

**Context:** Running the pinned stack (scikit-learn 1.9.0) surfaces a `FutureWarning`: the `probability` parameter of `SVC` is deprecated and scheduled for removal in scikit-learn 1.11, in favor of `CalibratedClassifierCV`.

**Verification:** Confirmed the warning fires but every QSVM call in this project still executes correctly and produces correct probabilities on the pinned version — this is a deprecation warning, not a current failure.

**Decision:** Keep `SVC(kernel="precomputed", probability=True)` as documented (§3.4, §3.5) since it works on the pinned stack, but don't upgrade scikit-learn mid-project without checking this specifically. If the team upgrades past 1.11 (deliberately or via an unpinned `pip install -U`), switch to `CalibratedClassifierCV(SVC(kernel="precomputed"), ensemble=False)` for probability outputs — the rest of the QSVM code is unaffected either way.

---

### D-19. Python 3.14 is the reproducible runtime; pytest is an explicit pin

**Context:** The first clean Phase 0 install used Python 3.11 because the deployment sketch still named Python 3.10. It failed before installation because the pinned `xgboost==3.4.1` declares `Requires-Python >=3.12`. The machine had Python 3.14 available, and the user explicitly approved using it. A second environment defect appeared when Phase 1's required regression tests could not start: `AGENTS.md` documented `pytest` commands, but `requirements.txt` did not include pytest.

**Verification:** A fresh Python 3.14 virtual environment installed every application pin unchanged, `pip check` reported no broken requirements, all application imports succeeded, and `lightning.qubit` initialized. Adding `pytest==9.1.1` made the documented test commands executable on Python 3.14; the full Phase 1 suite passed.

**Decision:** Python 3.14 is the local and container runtime for this pinned stack. Keep `xgboost==3.4.1` unchanged. Pin `pytest==9.1.1` under testing dependencies so a fresh environment can run the repository's required checks without an undocumented global tool.

---

### D-20. Final VQC benchmarks use at least 100 epochs; 20 epochs is only a quick regression budget

**Context:** The first three-seed Phase 1 benchmark used 20 full-batch Adam epochs at learning rate 0.05. Seeds 42 and 123 reached 93.86% accuracy, but seed 2026 reached only 74.56% and still had a much higher final training cost (0.528 versus 0.270 and 0.233). That run was sufficient to prove training was active, but not sufficient to treat the accuracy range as converged.

**Verification:** Re-ran the identical code, seeds (42, 123, 2026), stratified splits, preprocessing, initializations, and learning rate for 100 epochs. The 20-epoch accuracies were 93.86%, 93.86%, and 74.56% (87.43% mean; 74.56-93.86% range). At 100 epochs they were 90.35%, 93.86%, and 89.47% (91.23% mean; 89.47-93.86% range). Final costs fell from 0.270/0.233/0.528 at epoch 20 to 0.200/0.207/0.232 at epoch 100. Seed 2026 improved by 14.91 percentage points and the total accuracy range narrowed from 19.30 to 4.39 points. Seed 42 declined by 3.51 points despite lower training loss, so additional epochs improve convergence stability but do not guarantee monotonically better held-out accuracy.

**Decision:** Use at least 100 epochs for every Phase 2 VQC and for reported VQC/ensemble benchmarks. Keep 20 epochs only for fast D-02 regression checks. Report both the 20- and 100-epoch results when discussing this transition; do not hide the seed-42 decline or imply test accuracy must improve monotonically with training loss. Carry forward the already-measured same-four-feature classical means without rerunning them for the ensemble comparison: LogReg 93.57%, Random Forest 93.27%, XGBoost 93.27%, and SVM 93.86%.

---

### D-21. Exact zero vectors in amplitude embedding map to the canonical all-zero basis state

**Context:** The original quantum-range `MinMaxScaler` mapped the training-row minima to zero. On each Phase 2 benchmark split, one real row was the minimum of all four selected features and therefore became the exact vector `[0, 0, 0, 0]`. A zero vector has no direction and cannot be normalized into a quantum amplitude state; passing it directly to `AmplitudeEmbedding(normalize=True)` produces invalid values. The earlier architecture note that `normalize=True` alone handled every raw feature vector missed this boundary case.

**Decision:** Two defenses apply. First, only for the 2-qubit amplitude kernel, map any exact all-zero row to the canonical computational basis vector `[1, 0, 0, 0]` (the `|00>` state) before `AmplitudeEmbedding`. This is implemented in the shared feature-vector/matrix validation used by `fit`, kernel evaluation, `predict`, `predict_proba`, and `score`, so training, held-out evaluation, and future live inference are protected identically; it is not a one-off row correction. Nonzero rows are unchanged and continue to use `normalize=True, pad_with=0.0`. Second, fit the quantum `MinMaxScaler` to `[10⁻⁶, π-10⁻⁶]` with `clip=True`, preventing any normally transformed train, test, or live row from reaching an exact boundary. Tests cover arbitrary all-zero live-like inference without caller mutation and extreme raw live inputs clipped to the inward range.

---

### D-22. Phase 2 OOB-weighted ensemble did not beat the best single model on mean accuracy

**Context:** The completed Phase 2 implementation trains four VQC variants and two fidelity-kernel QSVM variants. Each member is fitted on its own bootstrap sample; its weight is normalized inverse probability-MSE measured only on that bootstrap's out-of-bag rows. Test rows are used only once for held-out evaluation and never for weight selection. All VQCs use 100 full-batch Adam epochs. To make the three-seed six-model measurement tractable, the controlled benchmark uses a stratified 200-row pool from each seed's training split while retaining the full held-out 20% test split. Seeds are 42, 123, and 2026.

**Verification:** Across seeds, the angle/4-qubit QSVM reached 94.44% mean accuracy (93.86-94.74%) and received 24.60% mean weight. VQC variants ranged from 91.23% to 92.11% in mean accuracy, with the complete individual ranges recorded in `architecture.md` §3.7. The deliberately smaller amplitude/2-qubit QSVM reached 73.98% mean (72.81-75.44%) and received the smallest mean weight, 6.55%, matching the intended diversity/down-weighting behavior. The weighted ensemble reached 94.15% mean (93.86-94.74%) versus 94.44% mean (93.86-94.74%) for the best single model on each seed. It beat the best single model on seed 42 (94.74% vs. 93.86%) but trailed by one test patient on seeds 123 and 2026 (93.86% vs. 94.74%). Paired two-sided per-patient tests were non-significant on every seed (p=0.657, 0.657, and 0.566).

**Decision:** Keep the honest OOB scheme and all six required models; do not retune weights or drop the amplitude model to force the roadmap target. State the two findings together: the ensemble validated the VQC-family instability concern by holding a tight 93.86-94.74% range while individual VQC variants ranged from 89.47% to 94.74%, and it statistically tied the strongest individual model rather than beating it (94.15% versus 94.44% mean; every paired p-value > 0.56). This is not treated as a broken implementation because every family reproduced its expected performance band, training costs decreased, the weak amplitude model was correctly down-weighted, and the difference is one held-out patient on each non-winning seed. Any later weighting change must be selected on training/OOB data and re-evaluated on untouched test data.

---

### D-23. `/explain`'s fast default must not display its own confidence number — verified as a real risk, not theoretical

**Context:** Phase 3's own verification surfaced this directly: for the same patient, VQC-only scope reported 96.11% benign, QSVM-only 96.20%, and the full six-model ensemble 83.32% — a large, clinically meaningful spread across scopes. This is mathematically expected (different model subsets, different outputs), but it's a real UI hazard: architecture.md section 6 puts the ensemble's confidence on the main results dial; if the fast-default explanation view then shows a different confidence number for feature attribution, a judge sees two disagreeing numbers for one patient with no time in a live demo to understand why. That reads as a bug, not a nuance, regardless of how correct it is underneath.

**Decision:** The `/explain` response must never include a standalone confidence value that gets rendered as if it were a second verdict. It returns feature attribution only (names, SHAP/LIME values, direction) — always visually and logically anchored to the single confidence number already shown from `/predict`, never a competing one. This applies specifically to the fast VQC-only default (D-17); the full opt-in (`allow_slow=True`) explanation naturally matches the dial exactly anyway, since it explains the literal ensemble prediction, so no fix is needed on that path.

**Evidence audit, September 2, 2026:** The original wording above is retained as the reviewed decision record, but its same-patient attribution needs correction. `artifacts/explainability/phase3_real_model_verification.md` records the VQC-only value for patient 307, while QSVM-only and full-ensemble values belong to patient 242. The approximately 12.88-point same-patient spread is therefore QSVM-only versus full ensemble; it does not establish that exact VQC-only/full-ensemble spread. The general UI hazard and attribution-only contract remain valid.

---

### D-24. Present the judge UI as a clinical research workspace

**Context:** After reviewing the Phase 5 frontend, the user requested a more professional clinical/quantum website. The previous oversized hero, diffuse gradients, large gauges, and decorative motion made the interaction feel more like a promotional page than a research application.

**Decision:** Put patient selection and paired model assessments in the primary working area. Retain the exact indigo/teal color tokens, equal-size dials, circuit rail, amber disagreement, both classical configurations, and attribution-only explanation contract. Use white surfaces, fine borders, compact headings, Source Sans 3 body text, and restrained functional motion. Keep the research-only disclaimer explicit rather than suggesting diagnostic readiness. Show raw patient measurements alongside attributions, not unlabeled quantum-scaled angles. Lock patient and explanation-scope controls while a request is active and discard any stale-generation response, preventing a result from appearing under a different selected record.

---

### D-25. Limit container OpenMP parallelism for tiny quantum circuits

**Context:** The first real Docker run on 2026-09-02 successfully installed every pinned requirement, but model initialization used roughly twenty CPU cores for the tiny 2–4 qubit circuits. A targeted ten-forward-pass probe under the live workload measured 0.156 seconds with the default OpenMP setting versus 0.015 seconds with one thread. This probe is an overhead diagnostic, not an end-to-end performance benchmark.

**Decision:** Set `OMP_NUM_THREADS=1` in the image. This controls execution overhead only; it does not change model definitions, seeds, training data, 100-epoch budgets, or dependency versions. Re-measure if the project moves to materially larger state vectors. Docker's per-user executable directory must also be on the calling shell's PATH so its standard credential helper can be found; no password is needed in the source or Compose file.

---

### D-26. Pre-push review: bound the demo workload and distinguish working features from research claims

**Context:** The user requested a whole-folder review before publishing to GitHub, with a confirmed September 8 hackathon. Review found that Streamlit could retain a previous patient's explanation after selection changed, concurrent expensive inference could queue, oversized request bodies were not bounded before parsing, and the live 20-row quantum training budget was not clearly distinguished from the saved 200-row benchmark.

**Decision:** Clear patient/scope-dependent state, handle request failures visibly, admit one prediction/explanation at a time with immediate HTTP 429 for competing inference, and limit incoming JSON bodies to 16 KiB with HTTP 413 before parsing. Health/catalog/metrics stay available while inference runs. Expose actual seed/epochs/training counts in `/health` and both frontends. Compose defaults to loopback for both services; trusted-LAN API access is explicit and the internal dashboard stays local. These safeguards are not authentication, distributed rate limiting, or a claim of production security. No model definitions, weights, training budgets, or dependency pins were retuned.

**Research qualifications:** Five-fold CV, classical SHAP, and a complete retained three-seed classical precision/recall/F1 report remain gaps, not completed features. Current binary precision/recall/F1 use class 1 (benign); recall must not be called malignant sensitivity. D-22's word "tie" means the tests found no significant difference, not that equivalence was proven. The weak amplitude model changes the feature map and discards vector magnitude during normalization; a smaller Hilbert space alone has not been isolated as its cause. Shared preprocessing makes the OOB weights model-level estimates, not fully nested pipeline validation. Preserve all six models and the existing reports.

**Source preservation:** Keep the original Word blueprint as explicitly historical source material, along with benchmark and verification reports. Clean generated caches only; exclude local environments, credentials, and disposable archives from Git and Docker build contexts. The current deadline and limitations live in `DEMO_GUIDE.md` and `roadmap.md`, not the old blueprint's calendar or aspirational claims.

---

### D-27. Close the core evidence gaps without changing the honest model claim

**Context:** The September 2 audit left four core gaps: metrics treated benign as the positive class, five-fold cross-validation was absent, classical explanations were missing, and the live refit was not tied to a saved configuration identifier. SIH26139 also calls for data ingestion, computational-efficiency evidence, and generalization evidence. On September 3 the user explicitly approved the API/UI extensions needed to close them.

**Decision:** Clinical evaluation now defines malignant (`0`) as the positive condition and retains sensitivity, specificity, malignant precision/F1, ROC-AUC, and a clinically oriented confusion matrix (`TP=malignant correctly detected`, `FN=malignant predicted benign`). Five-fold stratified evaluation refits imputation, scaling, ANOVA selection, and quantum scaling inside every fold; the full six-member ensemble uses 100 VQC epochs in every fold. The three-seed classical study retains complete metrics and measured fit/predict timings for all eight model/configuration combinations.

`/explain` now accepts `classical_full_feature` and `classical_same_4_feature` in addition to the existing quantum scopes. All responses remain attribution-only under D-23 and use original clinical names. `/ingest` validates and orders an exact named 30-feature record, returning range warnings rather than treating benchmark ranges as clinical validity limits. The browser and Streamlit clients expose both additions.

The live runtime is identified by `artifacts/models/runtime_manifest.json`. Startup deterministically refits that declared configuration; it does not pretend a pickle is an independently validated model release. Environment overrides remain visible in `/health`. These changes strengthen evaluation and reproducibility but do not change D-07 or D-22: no quantum-superiority, clinical-readiness, hardware-execution, or external-generalization claim is introduced.

**Measured outcome:** The five-fold six-model ensemble measured 91.57% mean accuracy (87.72–93.86%), 83.50% malignant sensitivity, 96.36% specificity, and 0.981 mean ROC-AUC. Full-feature Logistic Regression measured 97.37%, 94.36%, 99.16%, and 0.995 respectively; same-four Logistic Regression measured 94.03%, 90.59%, 96.08%, and 0.989. The amplitude/2-qubit QSVM was unstable under the 20-row fold training constraint (54.61% mean, 29.20–62.28% range). These results are retained rather than optimized away, and reinforce the no-quantum-superiority position. D-27 supersedes D-26's dated list of missing implementation evidence; D-26 remains as the audit history that motivated this work.
