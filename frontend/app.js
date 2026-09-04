"use strict";

const API_BASE = window.location.origin;
const DIAL_CIRCUMFERENCE = 2 * Math.PI * 82;

const state = {
  health: null,
  catalog: null,
  patient: null,
  prediction: null,
  explanation: null,
  deepExplanation: false,
  explanationModel: "quantum",
  healthTimer: null,
  explanationTimer: null,
  isBusy: false,
  generation: 0,
};

const elements = {
  serviceState: document.querySelector("#service-state"),
  serviceStateLabel: document.querySelector("#service-state-label"),
  runtimeContext: document.querySelector("#runtime-context"),
  patientSource: document.querySelector("#patient-source"),
  csvIngestion: document.querySelector("#csv-ingestion"),
  csvFile: document.querySelector("#csv-file"),
  csvExample: document.querySelector("#csv-example"),
  ingestionWarnings: document.querySelector("#ingestion-warnings"),
  patientSelect: document.querySelector("#patient-select"),
  patientSelectLabel: document.querySelector("#patient-select-label"),
  patientHelp: document.querySelector("#patient-help"),
  selectedFeatureChips: document.querySelector("#selected-feature-chips"),
  selectedFeatureList: document.querySelector("#selected-feature-list"),
  predictButton: document.querySelector("#predict-button"),
  analysisLoading: document.querySelector("#analysis-loading"),
  errorBanner: document.querySelector("#error-banner"),
  resultsSection: document.querySelector("#results-section"),
  resultsEmpty: document.querySelector("#results-empty"),
  recordId: document.querySelector("#record-id"),
  recordSource: document.querySelector("#record-source"),
  recordPartition: document.querySelector("#record-partition"),
  resultRecordId: document.querySelector("#result-record-id"),
  explainSection: document.querySelector("#explain-section"),
  referenceLabel: document.querySelector("#reference-label"),
  disagreementBanner: document.querySelector("#disagreement-banner"),
  disagreementCopy: document.querySelector("#disagreement-copy"),
  agreementNote: document.querySelector("#agreement-note"),
  agreementCopy: document.querySelector("#agreement-copy"),
  quantumProgress: document.querySelector("#quantum-progress"),
  quantumConfidence: document.querySelector("#quantum-confidence"),
  quantumLabel: document.querySelector("#quantum-label"),
  classicalProgress: document.querySelector("#classical-progress"),
  classicalConfidence: document.querySelector("#classical-confidence"),
  classicalLabel: document.querySelector("#classical-label"),
  quantumMembers: document.querySelector("#quantum-members"),
  sameFourLabel: document.querySelector("#same-four-label"),
  sameFourFill: document.querySelector("#same-four-fill"),
  sameFourConfidence: document.querySelector("#same-four-confidence"),
  scopeOptions: [...document.querySelectorAll(".scope-option")],
  slowDisclosure: document.querySelector("#slow-disclosure"),
  explainButton: document.querySelector("#explain-button"),
  explanationLoading: document.querySelector("#explanation-loading"),
  explanationLoadingTitle: document.querySelector("#explanation-loading-title"),
  explanationLoadingDetail: document.querySelector("#explanation-loading-detail"),
  explanationElapsed: document.querySelector("#explanation-elapsed"),
  explanationExpected: document.querySelector("#explanation-expected"),
  explanationResult: document.querySelector("#explanation-result"),
  scopeReadout: document.querySelector("#scope-readout"),
  attributionList: document.querySelector("#attribution-list"),
  clinicalMetricStatus: document.querySelector("#clinical-metric-status"),
  cvStatus: document.querySelector("#cv-status"),
  efficiencyStatus: document.querySelector("#efficiency-status"),
  cvComparisonBody: document.querySelector("#cv-comparison-body"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status})`;
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatFeatureValue(value) {
  const numeric = Number(value);
  if (Math.abs(numeric) >= 100) return numeric.toFixed(1);
  if (Math.abs(numeric) >= 10) return numeric.toFixed(2);
  return numeric.toFixed(4);
}

function labelCase(value) {
  const text = String(value || "");
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : "—";
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}

function setBusy(busy) {
  state.isBusy = busy;
  elements.patientSelect.disabled = busy || !state.catalog;
  elements.patientSource.disabled = busy || !state.catalog;
  elements.csvFile.disabled = busy || !state.catalog;
  elements.predictButton.disabled = busy || !state.patient;
  elements.explainButton.disabled = busy || !state.prediction;
  elements.scopeOptions.forEach((option) => { option.disabled = busy; });
  elements.predictButton.querySelector("span").textContent = busy
    ? "Analysis in progress…"
    : "Run hybrid analysis";
}

function studyRecordId(patient) {
  return `WBCD-${String(patient.id).padStart(3, "0")}`;
}

function setServiceState(status, label) {
  elements.serviceState.dataset.state = status;
  elements.serviceStateLabel.textContent = label;
}

function activateRail(stepName) {
  const order = ["data", "select", "model", "explain"];
  const activeIndex = order.indexOf(stepName);
  document.querySelectorAll(".rail-step").forEach((step) => {
    const index = order.indexOf(step.dataset.step);
    step.classList.toggle("is-active", index === activeIndex);
    step.classList.toggle("is-complete", index < activeIndex);
  });
}

async function checkHealth() {
  try {
    const health = await fetchJson("/health");
    state.health = health;
    if (health.models_loaded) {
      setServiceState("ready", "Model service ready");
      const config = health.runtime_configuration;
      if (config) {
        elements.runtimeContext.textContent = `Live runtime: ${config.quantum_training_limit} quantum training rows, ${config.classical_training_rows} classical training rows · ${config.vqc_epochs} VQC epochs · seed ${config.seed}. Benchmark evidence below is a separate three-seed, 200-row quantum run.`;
      }
      if (!state.catalog) await loadPatients();
      await loadEvidence();
      // Keep polling after a transient catalog failure so a page reload is not
      // the only recovery path once the models themselves are ready.
      if (state.catalog && state.healthTimer) {
        window.clearInterval(state.healthTimer);
        state.healthTimer = null;
      }
      return;
    }
    if (health.status === "error") {
      setServiceState("error", "Model loading failed");
      elements.patientHelp.textContent = health.error || "Check the API terminal for details.";
      showError(health.error || "The model runtime could not start.");
      return;
    }
    setServiceState("loading", "Training models · please wait");
    elements.patientHelp.textContent =
      "Preparing four 100-epoch VQCs and two quantum SVMs. First startup can take a few minutes.";
  } catch (error) {
    setServiceState("error", "Service unavailable");
    elements.patientHelp.textContent = "Start FastAPI on port 8000, then keep this page open.";
  }
}

async function loadPatients() {
  clearError();
  try {
    state.catalog = await fetchJson("/patients?limit=12");
    elements.patientSelect.innerHTML = state.catalog.patients
      .map((patient, index) => {
        const suffix = patient.has_disagreement ? " · comparison case" : "";
        return `<option value="${index}">${escapeHtml(studyRecordId(patient))}${suffix}</option>`;
      })
      .join("");
    elements.patientSelect.disabled = false;
    elements.patientSource.disabled = false;
    elements.predictButton.disabled = false;
    elements.patientHelp.textContent =
      "A real held-out study record. The reference label is concealed until analysis completes.";
    renderSelectedPatient(0);
    renderSelectedFeatureList();
    activateRail("select");
  } catch (error) {
    showError(`Could not load patient records: ${error.message}`);
  }
}

function renderSelectedFeatureList() {
  if (!state.catalog) return;
  elements.selectedFeatureList.innerHTML = state.catalog.selected_feature_names
    .map((name) => `<li>${escapeHtml(name)}</li>`)
    .join("");
}

function renderSelectedPatient(index) {
  if (!state.catalog) return;
  state.patient = state.catalog.patients[index];
  elements.recordId.textContent = studyRecordId(state.patient);
  elements.recordSource.textContent = "Wisconsin Breast Cancer";
  elements.recordPartition.textContent = "Held-out test set";
  elements.selectedFeatureChips.innerHTML = state.catalog.selected_feature_names
    .map(
      (name, featureIndex) => `
        <div class="feature-chip">
          <span title="${escapeHtml(name)}">${escapeHtml(name)}</span>
          <strong>${formatFeatureValue(state.patient.selected_values[featureIndex])}</strong>
        </div>`,
    )
    .join("");
  resetResultForPatientChange();
}

function parseCsvRecord(text) {
  const rows = text.trim().split(/\r?\n/).filter(Boolean);
  if (rows.length !== 2) throw new Error("CSV must contain exactly one header row and one patient row.");
  const parseRow = (row) => {
    const cells = [];
    let value = "";
    let quoted = false;
    for (let index = 0; index < row.length; index += 1) {
      const character = row[index];
      if (character === '"' && quoted && row[index + 1] === '"') { value += '"'; index += 1; }
      else if (character === '"') quoted = !quoted;
      else if (character === "," && !quoted) { cells.push(value.trim()); value = ""; }
      else value += character;
    }
    if (quoted) throw new Error("CSV contains an unclosed quoted value.");
    cells.push(value.trim());
    return cells;
  };
  const names = parseRow(rows[0]);
  const values = parseRow(rows[1]).map(Number);
  if (names.length !== 30 || values.length !== 30 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("CSV must contain exactly 30 named, finite numeric values.");
  }
  return Object.fromEntries(names.map((name, index) => [name, values[index]]));
}

function downloadExampleCsv() {
  if (!state.catalog?.patients?.length) return;
  const quote = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const csv = `${state.catalog.feature_names.map(quote).join(",")}\n${state.catalog.patients[0].features.join(",")}\n`;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  link.download = "qtrace-example-record.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

async function loadCsvRecord(file) {
  clearError();
  try {
    const record = parseCsvRecord(await file.text());
    const validated = await fetchJson("/ingest", {
      method: "POST",
      body: JSON.stringify({ record }),
    });
    const selectedIndices = state.catalog.selected_feature_names.map((name) => validated.feature_names.indexOf(name));
    state.patient = {
      id: "uploaded",
      name: "Uploaded clinical record",
      true_label: null,
      features: validated.features,
      selected_values: selectedIndices.map((index) => validated.features[index]),
      has_disagreement: false,
    };
    elements.recordId.textContent = "UPLOADED-CSV";
    elements.recordSource.textContent = "User-supplied named CSV";
    elements.recordPartition.textContent = "Unlabelled inference record";
    elements.selectedFeatureChips.innerHTML = state.catalog.selected_feature_names
      .map((name, index) => `<div class="feature-chip"><span title="${escapeHtml(name)}">${escapeHtml(name)}</span><strong>${formatFeatureValue(state.patient.selected_values[index])}</strong></div>`)
      .join("");
    elements.ingestionWarnings.hidden = false;
    elements.ingestionWarnings.textContent = validated.warnings.length
      ? `${validated.warnings.length} value(s) are outside the benchmark's observed range. Interpret this result cautiously.`
      : "Schema validated. All values are within the benchmark's observed ranges.";
    elements.predictButton.disabled = false;
    resetResultForPatientChange();
  } catch (error) {
    state.patient = null;
    elements.predictButton.disabled = true;
    showError(`CSV could not be loaded: ${error.message}`);
  }
}

async function loadEvidence() {
  try {
    const evidence = await fetchJson("/metrics");
    const repeated = evidence.generalization?.classical_three_seed;
    const crossValidation = evidence.generalization?.five_fold_cross_validation;
    if (repeated) {
      const key = "classical/full_feature/logistic_regression";
      const metrics = repeated.summaries[key].metrics;
      elements.clinicalMetricStatus.textContent = `Full-feature LogReg: ${(metrics.malignant_sensitivity.mean * 100).toFixed(1)}% sensitivity, ${(metrics.specificity.mean * 100).toFixed(1)}% specificity across 3 seeds.`;
      elements.efficiencyStatus.textContent = `Mean fit ${(repeated.summaries[key].mean_fit_seconds * 1000).toFixed(1)} ms; measured values are available for every classical model.`;
    }
    if (crossValidation) {
      const summaries = crossValidation.summaries;
      const quantumKey = "quantum/same_4_quantum_range/six_model_oob_ensemble";
      const fullKey = "classical/full_feature/logistic_regression";
      const sameKey = "classical/same_4_feature/logistic_regression";
      const summary = summaries[quantumKey];
      elements.cvStatus.textContent = summary
        ? `Five isolated folds: ${formatPercent(summary.metrics.accuracy.mean)} ensemble accuracy and ${formatPercent(summary.metrics.malignant_sensitivity.mean)} malignant sensitivity.`
        : "Five isolated classical folds are retained; quantum fold evidence is not available.";
      const quantumSeconds = summary?.mean_fit_seconds;
      if (Number.isFinite(quantumSeconds)) {
        elements.efficiencyStatus.textContent += ` Full six-model fold training/OOB/evaluation averaged ${quantumSeconds.toFixed(1)} s on this CPU run.`;
      }
      const rows = [
        ["Quantum · six-model ensemble", summaries[quantumKey]],
        ["Classical · Logistic, 30 features", summaries[fullKey]],
        ["Classical · Logistic, same 4", summaries[sameKey]],
      ];
      elements.cvComparisonBody.innerHTML = rows
        .filter(([, item]) => item)
        .map(([label, item]) => `<tr><th scope="row">${escapeHtml(label)}</th><td>${formatPercent(item.metrics.accuracy.mean)}</td><td>${formatPercent(item.metrics.malignant_sensitivity.mean)}</td><td>${formatPercent(item.metrics.specificity.mean)}</td><td>${Number(item.metrics.roc_auc.mean).toFixed(3)}</td></tr>`)
        .join("");
    }
  } catch (_error) {
    elements.clinicalMetricStatus.textContent = "Evidence is available from the developer console when the service is ready.";
    elements.cvComparisonBody.innerHTML = '<tr><td colspan="5">Retained fold evidence is temporarily unavailable.</td></tr>';
  }
}

function resetResultForPatientChange() {
  state.generation += 1;
  state.prediction = null;
  state.explanation = null;
  elements.resultsEmpty.hidden = false;
  elements.resultsSection.hidden = true;
  elements.explainSection.hidden = true;
  elements.explanationResult.hidden = true;
  elements.explanationLoading.hidden = true;
  elements.disagreementBanner.hidden = true;
  elements.agreementNote.hidden = true;
  selectExplanationScope("quantum-fast");
  clearError();
  activateRail("select");
}

function setDial(progressElement, valueElement, confidence) {
  const bounded = Math.max(0, Math.min(1, Number(confidence)));
  progressElement.style.strokeDasharray = `${DIAL_CIRCUMFERENCE}`;
  progressElement.style.strokeDashoffset = `${DIAL_CIRCUMFERENCE}`;
  valueElement.textContent = "0.0%";
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      progressElement.style.strokeDashoffset = `${DIAL_CIRCUMFERENCE * (1 - bounded)}`;
      animateNumericPercent(valueElement, bounded);
    });
  });
}

function animateNumericPercent(element, target) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    element.textContent = formatPercent(target);
    return;
  }
  const duration = 650;
  const started = performance.now();
  const tick = (now) => {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = formatPercent(target * eased);
    if (progress < 1) window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
}

function renderAgreement(agreement) {
  elements.disagreementBanner.hidden = agreement.agrees;
  elements.agreementNote.hidden = !agreement.agrees;
  if (agreement.agrees) {
    elements.agreementCopy.textContent = agreement.message;
  } else {
    elements.disagreementCopy.textContent = agreement.message;
  }
}

function renderPrediction(payload) {
  state.prediction = payload;
  const quantum = payload.quantum;
  const classical = payload.classical.full_feature;
  const sameFour = payload.classical.same_4_feature;

  elements.referenceLabel.textContent = state.patient.true_label ? labelCase(state.patient.true_label) : "Not provided";
  elements.resultRecordId.textContent = studyRecordId(state.patient);
  setDial(elements.quantumProgress, elements.quantumConfidence, quantum.confidence);
  setDial(elements.classicalProgress, elements.classicalConfidence, classical.confidence);
  elements.quantumLabel.textContent = labelCase(quantum.label);
  elements.classicalLabel.textContent = labelCase(classical.label);
  elements.sameFourLabel.textContent = labelCase(sameFour.label);
  elements.sameFourConfidence.textContent = formatPercent(sameFour.confidence);
  window.requestAnimationFrame(() => {
    elements.sameFourFill.style.width = `${sameFour.confidence * 100}%`;
  });
  elements.quantumMembers.innerHTML = quantum.per_model
    .map(
      (member) => `
        <div class="member-row">
          <span title="${escapeHtml(member.name)}">${escapeHtml(member.name)}</span>
          <span class="member-kind">${escapeHtml(member.paradigm)} · ${(member.weight * 100).toFixed(1)}% weight</span>
          <strong>${escapeHtml(labelCase(member.label))} · ${formatPercent(member.confidence)}</strong>
        </div>`,
    )
    .join("");
  renderAgreement(payload.agreement);
  elements.resultsEmpty.hidden = true;
  elements.resultsSection.hidden = false;
  elements.explainSection.hidden = false;
  activateRail("model");
  if (window.matchMedia("(max-width: 720px)").matches) {
    elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function runPrediction() {
  if (!state.patient || state.isBusy) return;
  resetResultForPatientChange();
  const generation = state.generation;
  const features = state.patient.features.slice();
  setBusy(true);
  elements.analysisLoading.hidden = false;
  try {
    const payload = await fetchJson("/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    });
    if (generation === state.generation) renderPrediction(payload);
  } catch (error) {
    showError(`Prediction could not complete: ${error.message}`);
  } finally {
    elements.analysisLoading.hidden = true;
    setBusy(false);
  }
}

function selectExplanationScope(scope) {
  const selectedScope = scope === false ? "quantum-fast" : scope;
  state.deepExplanation = selectedScope === "quantum-deep";
  state.explanationModel = selectedScope === "classical-full"
    ? "classical_full_feature"
    : selectedScope === "classical-same"
      ? "classical_same_4_feature"
      : "quantum";
  elements.scopeOptions.forEach((option) => {
    const selected = option.dataset.scope === selectedScope;
    option.classList.toggle(
      "is-selected",
      selected,
    );
    option.setAttribute("aria-pressed", String(selected));
  });
  elements.slowDisclosure.hidden = !state.deepExplanation;
  elements.explainButton.querySelector("span").textContent = state.deepExplanation
    ? "Request full-ensemble attribution"
    : state.explanationModel === "quantum"
      ? "Generate quantum attribution"
      : "Generate classical attribution";
  elements.explanationResult.hidden = true;
  state.explanation = null;
}

function formatElapsed(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")} elapsed`;
}

function beginExplanationLoading(deep) {
  elements.explanationLoading.hidden = false;
  elements.explanationResult.hidden = true;
  elements.explainButton.disabled = true;
  const classical = state.explanationModel !== "quantum";
  elements.explanationLoadingTitle.textContent = deep
    ? "Recomputing full ensemble kernels"
    : classical ? "Evaluating classical perturbations" : "Evaluating VQC perturbations";
  elements.explanationLoadingDetail.textContent = deep
    ? "Two QSVM members recompute kernel rows against their fitted training samples for each perturbed input."
    : classical
      ? "SHAP and LIME are checking the selected classical feature view with real clinical names."
      : "SHAP and LIME are checking four named clinical features across four VQC members.";
  elements.explanationExpected.textContent = deep
    ? "Typical: 70 seconds or longer"
    : "Reference: 7–30 seconds · device dependent";
  const started = Date.now();
  elements.explanationElapsed.textContent = formatElapsed(0);
  state.explanationTimer = window.setInterval(() => {
    elements.explanationElapsed.textContent = formatElapsed(
      Math.floor((Date.now() - started) / 1000),
    );
  }, 1000);
}

function endExplanationLoading() {
  if (state.explanationTimer) window.clearInterval(state.explanationTimer);
  state.explanationTimer = null;
  elements.explanationLoading.hidden = true;
  elements.explainButton.disabled = false;
}

function renderExplanation(payload) {
  state.explanation = payload;
  const scopeLabels = { full_ensemble: "FULL ENSEMBLE", vqc_fast: "VQC FAST", classical_full_feature: "CLASSICAL FULL", classical_same_4_feature: "CLASSICAL MATCHED" };
  elements.scopeReadout.textContent = scopeLabels[payload.scope] || payload.scope.toUpperCase();
  const maximum = Math.max(
    0.000001,
    ...payload.shap_values.map((value) => Math.abs(Number(value))),
    ...payload.lime_values.map((value) => Math.abs(Number(value))),
  );
  elements.attributionList.innerHTML = payload.feature_names
    .map((name, index) => {
      const shapValue = Number(payload.shap_values[index]);
      const limeValue = Number(payload.lime_values[index]);
      const width = Math.min(50, (Math.abs(shapValue) / maximum) * 50);
      const marker = 50 + (limeValue / maximum) * 50;
      const direction = payload.directions[index];
      const directionLabel = direction.replaceAll("_", " ");
      const directionClass = direction.replaceAll("_", "-");
      const signClass = shapValue >= 0 ? "is-positive" : "is-negative";
      // Show the patient's original measurement, not the quantum-scaled angle.
      const selectedIndex = state.catalog.selected_feature_names.indexOf(name);
      const rawIndex = state.catalog.feature_names.indexOf(name);
      const featureValue = selectedIndex >= 0
        ? state.patient.selected_values[selectedIndex]
        : rawIndex >= 0 ? state.patient.features[rawIndex] : undefined;
      return `
        <div class="attribution-row">
          <div class="attribution-name">
            <strong>${escapeHtml(name)}</strong>
            <span>${featureValue === undefined ? "Selected clinical input" : `Raw ${formatFeatureValue(featureValue)}`}</span>
          </div>
          <div class="attribution-scale" aria-label="${escapeHtml(name)}: SHAP ${shapValue.toFixed(4)}, LIME ${limeValue.toFixed(4)}">
            <span class="attribution-axis"></span>
            <span class="attribution-zero"></span>
            <span class="attribution-bar ${signClass}" style="--bar-width:${width}%"></span>
            <span class="lime-marker" style="--marker-position:${Math.max(0, Math.min(100, marker))}%"></span>
          </div>
          <span class="attribution-direction ${directionClass}">${escapeHtml(directionLabel)}</span>
        </div>`;
    })
    .join("");
  elements.explanationResult.hidden = false;
  activateRail("explain");
  window.setTimeout(() => {
    elements.explanationResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 80);
}

async function runExplanation() {
  if (!state.patient || !state.prediction || state.isBusy) return;
  if (
    state.deepExplanation &&
    !window.confirm(
      "Full-ensemble SHAP recomputes quantum kernels and may take 70 seconds or longer. Continue?",
    )
  ) {
    return;
  }
  clearError();
  const generation = state.generation;
  const features = state.patient.features.slice();
  setBusy(true);
  beginExplanationLoading(state.deepExplanation);
  try {
    const payload = await fetchJson("/explain", {
      method: "POST",
      body: JSON.stringify({
        features,
        model: state.explanationModel,
        allow_slow: state.deepExplanation,
      }),
    });
    if (generation === state.generation) renderExplanation(payload);
  } catch (error) {
    showError(`Explanation could not complete: ${error.message}`);
  } finally {
    endExplanationLoading();
    setBusy(false);
  }
}

function forceDisagreementForVerification() {
  if (!state.prediction) throw new Error("Run a prediction before forcing the banner.");
  renderAgreement({
    agrees: false,
    message:
      "Verification override: quantum and classical labels differ, so neither result is hidden.",
  });
  return !elements.disagreementBanner.hidden;
}

elements.patientSelect.addEventListener("change", (event) => {
  if (!state.isBusy) renderSelectedPatient(Number(event.target.value));
});
elements.patientSource.addEventListener("change", (event) => {
  const upload = event.target.value === "csv";
  elements.csvIngestion.hidden = !upload;
  elements.patientSelectLabel.hidden = upload;
  elements.patientSelect.parentElement.hidden = upload;
  elements.patientSelect.disabled = upload || state.isBusy;
  if (!upload) renderSelectedPatient(Number(elements.patientSelect.value || 0));
  else { state.patient = null; resetResultForPatientChange(); elements.predictButton.disabled = true; }
});
elements.csvFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file && !state.isBusy) loadCsvRecord(file);
});
elements.csvExample.addEventListener("click", downloadExampleCsv);
elements.predictButton.addEventListener("click", runPrediction);
elements.explainButton.addEventListener("click", runExplanation);
elements.scopeOptions.forEach((option) => {
  option.addEventListener("click", () => {
    if (!state.isBusy) selectExplanationScope(option.dataset.scope);
  });
});

// Deliberately narrow browser-verification hook; it changes presentation only.
window.QTraceTest = {
  forceDisagreement: forceDisagreementForVerification,
  getState: () => ({
    modelsReady: Boolean(state.health?.models_loaded),
    patientId: state.patient?.id ?? null,
    hasPrediction: Boolean(state.prediction),
    hasExplanation: Boolean(state.explanation),
    scope: state.explanation?.scope ?? null,
  }),
};

checkHealth();
state.healthTimer = window.setInterval(checkHealth, 2500);
