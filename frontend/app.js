"use strict";

const API_BASE = window.location.origin;
const DIAL_CIRCUMFERENCE = 2 * Math.PI * 82;

const state = {
  health: null,
  diseases: null,
  activeDiseaseId: "breast_cancer",
  activeModule: null,
  catalog: null,
  patient: null,
  prediction: null,
  explanation: null,
  report: null,
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
  diseaseSelect: document.querySelector("#disease-select"),
  diseaseBreadcrumb: document.querySelector("#disease-breadcrumb"),
  diseaseTitle: document.querySelector("#disease-title"),
  diseaseDescription: document.querySelector("#disease-description"),
  studyBadge: document.querySelector("#study-badge"),
  moduleDescription: document.querySelector("#module-description"),
  railDataCaption: document.querySelector("#rail-data-caption"),
  patientPanelSubtitle: document.querySelector("#patient-panel-subtitle"),
  csvHelp: document.querySelector("#csv-help"),
  footerDataset: document.querySelector("#footer-dataset"),
  patientSource: document.querySelector("#patient-source"),
  csvIngestion: document.querySelector("#csv-ingestion"),
  csvFile: document.querySelector("#csv-file"),
  csvExample: document.querySelector("#csv-example"),
  manualIntake: document.querySelector("#manual-intake"),
  manualFields: document.querySelector("#manual-fields"),
  useManualRecord: document.querySelector("#use-manual-record"),
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
  classicalModelSubtitle: document.querySelector("#classical-model-subtitle"),
  classicalScopeCaption: document.querySelector("#classical-scope-caption"),
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
  evidenceMethodBadge: document.querySelector("#evidence-method-badge"),
  evidencePrimaryLabel: document.querySelector("#evidence-primary-label"),
  evidencePrimaryValue: document.querySelector("#evidence-primary-value"),
  evidencePrimaryDetail: document.querySelector("#evidence-primary-detail"),
  evidenceSecondaryLabel: document.querySelector("#evidence-secondary-label"),
  evidenceSecondaryValue: document.querySelector("#evidence-secondary-value"),
  evidenceSecondaryDetail: document.querySelector("#evidence-secondary-detail"),
  evidenceTertiaryLabel: document.querySelector("#evidence-tertiary-label"),
  evidenceTertiaryValue: document.querySelector("#evidence-tertiary-value"),
  evidenceTertiaryDetail: document.querySelector("#evidence-tertiary-detail"),
  evidenceNoteTitle: document.querySelector("#evidence-note-title"),
  evidenceNoteCopy: document.querySelector("#evidence-note-copy"),
  clinicalMetricTitle: document.querySelector("#clinical-metric-title"),
  generalizationTitle: document.querySelector("#generalization-title"),
  comparisonTitle: document.querySelector("#comparison-title"),
  positiveConditionLabel: document.querySelector("#positive-condition-label"),
  towardConditionLabel: document.querySelector("#toward-condition-label"),
  towardReferenceLabel: document.querySelector("#toward-reference-label"),
  reportButton: document.querySelector("#report-button"),
  reportSection: document.querySelector("#report-section"),
  reportSummary: document.querySelector("#report-summary"),
  reportFacts: document.querySelector("#report-facts"),
  downloadReport: document.querySelector("#download-report"),
  printReport: document.querySelector("#print-report"),
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

function syncPredictButtonAvailability() {
  elements.predictButton.disabled = state.isBusy || !state.patient;
}

function setBusy(busy) {
  state.isBusy = busy;
  elements.patientSelect.disabled = busy || !state.catalog;
  elements.diseaseSelect.disabled = busy || !state.diseases;
  elements.patientSource.disabled = busy || !state.catalog;
  elements.csvFile.disabled = busy || !state.catalog;
  elements.useManualRecord.disabled = busy || !state.catalog;
  elements.manualFields.querySelectorAll("input, select").forEach((field) => { field.disabled = busy; });
  syncPredictButtonAvailability();
  elements.explainButton.disabled = busy || !state.prediction;
  elements.reportButton.disabled = busy || !state.prediction;
  elements.scopeOptions.forEach((option) => { option.disabled = busy; });
  elements.predictButton.querySelector("span").textContent = busy
    ? "Analysis in progress…"
    : "Run hybrid analysis";
}

function studyRecordId(patient) {
  if (patient.id === "uploaded") return "UPLOADED-CSV";
  if (patient.id === "manual") return "MANUAL-INPUT";
  const prefix = state.activeDiseaseId === "early_diabetes" ? "UCI-DM" : "WBCD";
  return `${prefix}-${String(patient.id).padStart(3, "0")}`;
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
        const moduleCount = health.disease_modules?.length || 2;
        updateRuntimeContext(moduleCount, config);
      }
      if (!state.diseases) await loadDiseases();
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
      "Preparing two disease modules, each with four 100-epoch VQCs and two quantum SVMs. First startup can take a few minutes.";
  } catch (error) {
    setServiceState("error", "Service unavailable");
    elements.patientHelp.textContent = "Start FastAPI on port 8000, then keep this page open.";
  }
}

async function loadDiseases() {
  state.diseases = await fetchJson("/diseases");
  state.activeDiseaseId = state.diseases.default_disease_id || "breast_cancer";
  elements.diseaseSelect.innerHTML = state.diseases.modules
    .map((module) => `<option value="${escapeHtml(module.disease_id)}">${escapeHtml(module.short_title)}</option>`)
    .join("");
  elements.diseaseSelect.value = state.activeDiseaseId;
  elements.diseaseSelect.disabled = false;
  updateDiseasePresentation();
}

function updateDiseasePresentation() {
  state.activeModule = state.diseases?.modules.find(
    (module) => module.disease_id === state.activeDiseaseId,
  ) || null;
  if (!state.activeModule) return;
  const module = state.activeModule;
  const [conditionLabel, referenceLabel] = module.class_labels;
  elements.diseaseBreadcrumb.textContent = module.short_title;
  elements.diseaseTitle.textContent = module.title;
  elements.diseaseDescription.textContent = module.description;
  elements.studyBadge.textContent = `SIH26139 · ${module.dataset_name}`;
  elements.moduleDescription.textContent = `${module.domain}. Independently trained; ${module.dataset_license}.`;
  elements.railDataCaption.textContent = `${module.feature_names.length} real clinical inputs`;
  elements.patientPanelSubtitle.textContent = `${module.dataset_name} · held-out records`;
  elements.csvHelp.textContent = `The header must contain all ${module.feature_names.length} selected-module feature names exactly. Binary questionnaire values use 0=No/Female and 1=Yes/Male.`;
  elements.footerDataset.textContent = `Simulator-based · ${module.dataset_name}`;
  elements.classicalModelSubtitle.textContent = `Logistic regression · all ${module.feature_names.length} module features`;
  elements.classicalScopeCaption.textContent = `LogReg · all ${module.feature_names.length} features`;
  elements.towardConditionLabel.textContent = `Toward ${conditionLabel}`;
  elements.towardReferenceLabel.textContent = `Toward ${referenceLabel}`;
  if (state.health?.runtime_configuration) {
    updateRuntimeContext(state.health.disease_modules?.length || 2, state.health.runtime_configuration);
  }
  renderManualForm();
}

function updateRuntimeContext(moduleCount, config) {
  const healthModule = state.health?.disease_modules?.find(
    (module) => module.disease_id === state.activeDiseaseId,
  );
  const classicalRows = healthModule?.classical_training_rows ?? config.classical_training_rows;
  const evidenceScope = state.activeDiseaseId === "early_diabetes"
    ? "Evidence below includes its separate three-seed 20-row quantum evaluation."
    : "Benchmark evidence below includes the separate three-seed, 200-row quantum run.";
  elements.runtimeContext.textContent = `Live runtime: ${moduleCount} independent disease modules · ${config.quantum_training_limit} quantum training rows per module · ${classicalRows} classical training rows in this module · ${config.vqc_epochs} VQC epochs · seed ${config.seed}. ${evidenceScope}`;
}

function renderManualForm() {
  if (!state.activeModule) return;
  elements.manualFields.innerHTML = state.activeModule.feature_schema
    .map((feature, index) => {
      const label = escapeHtml(feature.name);
      if (feature.kind === "binary") {
        const options = feature.name === "gender"
          ? '<option value="0">Female (0)</option><option value="1">Male (1)</option>'
          : '<option value="0">No (0)</option><option value="1">Yes (1)</option>';
        return `<div class="manual-field"><label for="manual-${index}">${label}</label><select id="manual-${index}" data-feature-index="${index}">${options}</select></div>`;
      }
      const midpoint = (Number(feature.observed_min) + Number(feature.observed_max)) / 2;
      return `<div class="manual-field"><label for="manual-${index}">${label}</label><input id="manual-${index}" data-feature-index="${index}" type="number" step="any" min="${feature.observed_min}" max="${feature.observed_max}" value="${midpoint}" /></div>`;
    })
    .join("");
}

async function loadPatients() {
  clearError();
  try {
    state.catalog = await fetchJson(`/patients?limit=12&disease_id=${encodeURIComponent(state.activeDiseaseId)}`);
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
  elements.recordSource.textContent = state.catalog.dataset_name || state.activeModule?.dataset_name || "Benchmark dataset";
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
  syncPredictButtonAvailability();
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
  const expectedCount = state.catalog?.feature_names?.length || 0;
  if (names.length !== expectedCount || values.length !== expectedCount || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`CSV must contain exactly ${expectedCount} named, finite numeric values.`);
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
      body: JSON.stringify({ disease_id: state.activeDiseaseId, record }),
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
    resetResultForPatientChange();
    syncPredictButtonAvailability();
  } catch (error) {
    state.patient = null;
    syncPredictButtonAvailability();
    showError(`CSV could not be loaded: ${error.message}`);
  }
}

function applyManualRecord({ reset = true, showValidationError = true } = {}) {
  if (!state.catalog || !state.activeModule) return false;
  try {
    const fields = [...elements.manualFields.querySelectorAll("input, select")];
    if (fields.some((field) => String(field.value).trim() === "")) {
      throw new Error("Every module field needs a value.");
    }
    const features = fields.map((field) => Number(field.value));
    if (features.length !== state.catalog.feature_names.length || features.some((value) => !Number.isFinite(value))) {
      throw new Error("Every module field needs a finite numeric value.");
    }
    state.patient = {
      id: "manual",
      name: "Manual module record",
      true_label: null,
      features,
      selected_values: state.catalog.selected_feature_names.map(
        (name) => features[state.catalog.feature_names.indexOf(name)],
      ),
      has_disagreement: false,
    };
    elements.recordId.textContent = "MANUAL-INPUT";
    elements.recordSource.textContent = "User-entered module form";
    elements.recordPartition.textContent = "Unlabelled inference record";
    elements.selectedFeatureChips.innerHTML = state.catalog.selected_feature_names
      .map((name, index) => `<div class="feature-chip"><span title="${escapeHtml(name)}">${escapeHtml(name)}</span><strong>${formatFeatureValue(state.patient.selected_values[index])}</strong></div>`)
      .join("");
    if (reset) resetResultForPatientChange();
    syncPredictButtonAvailability();
    return true;
  } catch (error) {
    state.patient = null;
    if (reset) resetResultForPatientChange();
    syncPredictButtonAvailability();
    if (showValidationError) showError(`Manual record could not be used: ${error.message}`);
    return false;
  }
}

function useManualRecord() {
  applyManualRecord();
}

async function loadEvidence() {
  try {
    const evidence = await fetchJson("/metrics" + `?disease_id=${encodeURIComponent(state.activeDiseaseId)}`);
    const repeated = evidence.generalization?.classical_three_seed;
    const crossValidation = evidence.generalization?.five_fold_cross_validation;
    const diabetesRepeated = evidence.generalization?.early_diabetes_three_seed;
    const positiveName = state.activeModule?.positive_class_name || "condition present";
    elements.clinicalMetricTitle.textContent = `${labelCase(positiveName)} sensitivity & specificity`;
    elements.positiveConditionLabel.textContent = `${labelCase(positiveName)} is the positive condition`;
    if (repeated) {
      elements.evidenceMethodBadge.textContent = "3 seeds · 100 epochs · 200-row training pool";
      elements.evidencePrimaryLabel.textContent = "QUANTUM ENSEMBLE";
      elements.evidencePrimaryValue.textContent = "93.86–94.74%";
      elements.evidencePrimaryDetail.textContent = "Held-out accuracy range";
      elements.evidenceSecondaryLabel.textContent = "STRONGEST QUANTUM MEMBER";
      elements.evidenceSecondaryValue.textContent = "93.86–94.74%";
      elements.evidenceSecondaryDetail.textContent = "Angle-embedding QSVM";
      elements.evidenceTertiaryLabel.textContent = "PAIRED COMPARISON";
      elements.evidenceTertiaryValue.textContent = "p > 0.56";
      elements.evidenceTertiaryDetail.textContent = "No significant difference detected";
      elements.evidenceNoteTitle.textContent = "Stable results, not a superiority claim.";
      elements.evidenceNoteCopy.textContent = "The ensemble reduced the observed VQC-family variability and statistically tied the strongest individual quantum model rather than beating it. Classical baselines are retained in both feature configurations.";
      elements.generalizationTitle.textContent = "Five-fold generalization";
      elements.comparisonTitle.textContent = "Leakage-safe five-fold comparison";
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
    } else {
      const live = evidence.quantum?.ensemble;
      const classical = evidence.classical?.live_runtime?.full_feature?.logistic_regression;
      if (live && classical) {
        if (diabetesRepeated) {
          const accuracy = diabetesRepeated.summaries.ensemble_accuracy;
          const strongest = diabetesRepeated.summaries.members.qsvm_amplitude_2q.accuracy;
          const strongestClassical = diabetesRepeated.summaries.classical_accuracy["full_feature/svm"];
          elements.evidenceMethodBadge.textContent = "3 seeds · 100 epochs · 20-row training pool";
          elements.evidencePrimaryLabel.textContent = "QUANTUM ENSEMBLE";
          elements.evidencePrimaryValue.textContent = `${(accuracy.min * 100).toFixed(2)}–${(accuracy.max * 100).toFixed(2)}%`;
          elements.evidencePrimaryDetail.textContent = `${(accuracy.mean * 100).toFixed(2)}% mean accuracy`;
          elements.evidenceSecondaryLabel.textContent = "STRONGEST QUANTUM MEAN";
          elements.evidenceSecondaryValue.textContent = `${(strongest.mean * 100).toFixed(2)}%`;
          elements.evidenceSecondaryDetail.textContent = "Amplitude-embedding QSVM";
          elements.evidenceTertiaryLabel.textContent = "STRONGEST CLASSICAL MEAN";
          elements.evidenceTertiaryValue.textContent = `${(strongestClassical.mean * 100).toFixed(2)}%`;
          elements.evidenceTertiaryDetail.textContent = "Full-feature SVM";
          elements.evidenceNoteTitle.textContent = "Scalability evidence, not a superiority claim.";
          elements.evidenceNoteCopy.textContent = "The same complete workflow trained on a second biomedical dataset. Classical models were stronger and the quantum ensemble varied across seeds; both findings remain visible.";
        }
        elements.generalizationTitle.textContent = "Three-seed generalization";
        elements.comparisonTitle.textContent = "Held-out module comparison";
        elements.clinicalMetricStatus.textContent = diabetesRepeated
          ? `Three seeds: ${(diabetesRepeated.summaries.ensemble_condition_sensitivity.mean * 100).toFixed(1)}% mean sensitivity and ${(diabetesRepeated.summaries.ensemble_specificity.mean * 100).toFixed(1)}% mean specificity.`
          : `Live held-out module: ${(live.condition_sensitivity * 100).toFixed(1)}% sensitivity and ${(live.specificity * 100).toFixed(1)}% specificity.`;
        elements.cvStatus.textContent = diabetesRepeated
          ? `Three-seed ensemble accuracy ${formatPercent(diabetesRepeated.summaries.ensemble_accuracy.mean)} (${formatPercent(diabetesRepeated.summaries.ensemble_accuracy.min)}–${formatPercent(diabetesRepeated.summaries.ensemble_accuracy.max)}); external validation is not claimed.`
          : "Second-disease module is a scalability demonstration; repeated/fold evidence is not yet claimed.";
        elements.efficiencyStatus.textContent = "All inference is local on a free PennyLane simulator; no paid service or patient-data upload is used.";
        elements.cvComparisonBody.innerHTML = [
          ["Quantum · six-model ensemble", live],
          [`Classical · Logistic, ${state.catalog.feature_names.length} features`, classical],
        ].map(([label, metrics]) => `<tr><th scope="row">${escapeHtml(label)}</th><td>${formatPercent(metrics.accuracy)}</td><td>${formatPercent(metrics.condition_sensitivity)}</td><td>${formatPercent(metrics.specificity)}</td><td>${Number(metrics.roc_auc).toFixed(3)}</td></tr>`).join("");
      }
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
  state.report = null;
  elements.resultsEmpty.hidden = false;
  elements.resultsSection.hidden = true;
  elements.explainSection.hidden = true;
  elements.reportSection.hidden = true;
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
  if (state.isBusy) return;
  if (elements.patientSource.value === "manual" && !applyManualRecord({ reset: false })) return;
  if (!state.patient) return;
  resetResultForPatientChange();
  const generation = state.generation;
  const features = state.patient.features.slice();
  setBusy(true);
  elements.analysisLoading.hidden = false;
  try {
    const payload = await fetchJson("/predict", {
      method: "POST",
      body: JSON.stringify({ disease_id: state.activeDiseaseId, features }),
    });
    if (generation === state.generation) renderPrediction(payload);
  } catch (error) {
    showError(`Prediction could not complete: ${error.message}`);
  } finally {
    elements.analysisLoading.hidden = true;
    setBusy(false);
  }
}

function buildPrintableReport(report) {
  const prediction = report.prediction;
  const quantum = prediction.quantum;
  const classical = prediction.classical.full_feature;
  const matched = prediction.classical.same_4_feature;
  const measurements = report.selected_measurements
    .map((item) => `<tr><td>${escapeHtml(item.feature_name)}</td><td>${escapeHtml(formatFeatureValue(item.value))}</td></tr>`)
    .join("");
  const members = quantum.per_model
    .map((member) => `<tr><td>${escapeHtml(member.name)}</td><td>${escapeHtml(member.paradigm)}</td><td>${escapeHtml(labelCase(member.label))}</td><td>${escapeHtml(formatPercent(member.confidence))}</td><td>${escapeHtml(formatPercent(member.weight))}</td></tr>`)
    .join("");
  const limitations = report.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const fairness = report.fairness_audit
    ? `<h2>Subgroup audit</h2><p>${escapeHtml(report.fairness_audit.interpretation)}</p><table><thead><tr><th>System</th><th>Group</th><th>n</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th></tr></thead><tbody>${Object.entries(report.fairness_audit.quantum_ensemble).map(([group, metrics]) => `<tr><td>Quantum ensemble</td><td>${escapeHtml(group)}</td><td>${metrics.n}</td><td>${formatPercent(metrics.accuracy)}</td><td>${metrics.condition_sensitivity == null ? "—" : formatPercent(metrics.condition_sensitivity)}</td><td>${metrics.specificity == null ? "—" : formatPercent(metrics.specificity)}</td></tr>`).join("")}${Object.entries(report.fairness_audit.classical_full_feature_logistic_regression).map(([group, metrics]) => `<tr><td>Classical full</td><td>${escapeHtml(group)}</td><td>${metrics.n}</td><td>${formatPercent(metrics.accuracy)}</td><td>${metrics.condition_sensitivity == null ? "—" : formatPercent(metrics.condition_sensitivity)}</td><td>${metrics.specificity == null ? "—" : formatPercent(metrics.specificity)}</td></tr>`).join("")}</tbody></table>`
    : "";
  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(report.report_id)} · Q-TRACE</title><style>
    body{font:15px/1.5 Arial,sans-serif;color:#12181c;margin:40px;max-width:960px}h1,h2{font-family:Arial,sans-serif}h1{border-bottom:4px solid #4a3f8c;padding-bottom:12px}.meta{color:#52616a}.notice{border-left:4px solid #b8863d;background:#fbf7ef;padding:12px 16px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid #d7dcdd;padding:14px}.card.q{border-top:4px solid #4a3f8c}.card.c{border-top:4px solid #0e7c7b}table{border-collapse:collapse;width:100%;margin:12px 0 22px}th,td{border:1px solid #d7dcdd;padding:8px;text-align:left}th{background:#f1f3f2}small{color:#52616a}@media print{body{margin:20px}.no-print{display:none}}
  </style></head><body>
    <h1>Q-TRACE screening analysis report</h1>
    <p class="meta">${escapeHtml(report.report_id)} · ${escapeHtml(report.generated_at)}<br>${escapeHtml(report.title)} · ${escapeHtml(report.dataset.name)}</p>
    <p class="notice"><strong>Research use only.</strong> Model confidence is not clinical risk, and this report is not a diagnosis.</p>
    <h2>AI-assisted evidence summary</h2><p>${escapeHtml(report.ai_evidence_summary)}</p>
    <div class="cards"><div class="card q"><small>QUANTUM ENSEMBLE</small><h3>${escapeHtml(labelCase(quantum.label))}</h3><strong>${escapeHtml(formatPercent(quantum.confidence))} confidence</strong></div><div class="card c"><small>CLASSICAL FULL</small><h3>${escapeHtml(labelCase(classical.label))}</h3><strong>${escapeHtml(formatPercent(classical.confidence))} confidence</strong></div><div class="card c"><small>CLASSICAL MATCHED-4</small><h3>${escapeHtml(labelCase(matched.label))}</h3><strong>${escapeHtml(formatPercent(matched.confidence))} confidence</strong></div></div>
    <h2>Selected clinical inputs</h2><table><thead><tr><th>Feature</th><th>Raw value</th></tr></thead><tbody>${measurements}</tbody></table>
    <h2>Quantum ensemble audit</h2><table><thead><tr><th>Model</th><th>Type</th><th>Output</th><th>Confidence</th><th>OOB weight</th></tr></thead><tbody>${members}</tbody></table>
    <h2>Evidence boundary</h2><p>Condition-positive sensitivity: ${escapeHtml(formatPercent(report.held_out_evidence.quantum_ensemble.condition_sensitivity))}; specificity: ${escapeHtml(formatPercent(report.held_out_evidence.quantum_ensemble.specificity))}. ${escapeHtml(report.held_out_evidence.scope)}</p>
    ${fairness}
    <h2>Limitations and responsible use</h2><ul>${limitations}</ul>
    <p><small>Dataset: <a href="${escapeHtml(report.dataset.source)}">${escapeHtml(report.dataset.source)}</a> · ${escapeHtml(report.dataset.license)}. Generated locally; the server does not retain the record.</small></p>
  </body></html>`;
}

function renderReport(report) {
  state.report = report;
  elements.reportSummary.textContent = report.ai_evidence_summary;
  const prediction = report.prediction;
  elements.reportFacts.innerHTML = [
    ["Report ID", report.report_id],
    ["Disease module", report.title],
    ["Model agreement", prediction.agreement.agrees ? "Quantum and classical agree" : "Disagreement preserved"],
  ].map(([label, value]) => `<div class="report-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  elements.reportSection.hidden = false;
  elements.reportSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runReport() {
  if (!state.patient || !state.prediction || state.isBusy) return;
  clearError();
  const generation = state.generation;
  setBusy(true);
  elements.reportButton.textContent = "Generating evidence report…";
  try {
    const report = await fetchJson("/report", {
      method: "POST",
      body: JSON.stringify({
        disease_id: state.activeDiseaseId,
        features: state.patient.features.slice(),
      }),
    });
    if (generation === state.generation) renderReport(report);
  } catch (error) {
    showError(`Report could not be generated: ${error.message}`);
  } finally {
    elements.reportButton.textContent = "Generate AI evidence report";
    setBusy(false);
  }
}

function downloadReportHtml() {
  if (!state.report) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([buildPrintableReport(state.report)], { type: "text/html" }));
  link.download = `${state.report.report_id}.html`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function printReportDocument() {
  if (!state.report) return;
  const printable = window.open("", "_blank");
  if (!printable) {
    showError("The browser blocked the print window. Allow pop-ups, then try again.");
    return;
  }
  printable.opener = null;
  printable.document.open();
  printable.document.write(buildPrintableReport(state.report));
  printable.document.close();
  printable.focus();
  window.setTimeout(() => printable.print(), 250);
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
        disease_id: state.activeDiseaseId,
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
elements.diseaseSelect.addEventListener("change", async (event) => {
  if (state.isBusy) return;
  state.activeDiseaseId = event.target.value;
  state.catalog = null;
  state.patient = null;
  updateDiseasePresentation();
  resetResultForPatientChange();
  elements.patientSource.value = "catalog";
  elements.csvIngestion.hidden = true;
  elements.manualIntake.hidden = true;
  elements.patientSelectLabel.hidden = false;
  elements.patientSelect.parentElement.hidden = false;
  elements.patientSelect.disabled = true;
  elements.patientSelect.innerHTML = "<option>Loading held-out records…</option>";
  await loadPatients();
  await loadEvidence();
});
elements.patientSource.addEventListener("change", (event) => {
  const upload = event.target.value === "csv";
  const manual = event.target.value === "manual";
  elements.csvIngestion.hidden = !upload;
  elements.manualIntake.hidden = !manual;
  elements.patientSelectLabel.hidden = upload || manual;
  elements.patientSelect.parentElement.hidden = upload || manual;
  elements.patientSelect.disabled = upload || manual || state.isBusy;
  if (!upload && !manual) {
    renderSelectedPatient(Number(elements.patientSelect.value || 0));
  } else if (manual) {
    applyManualRecord({ showValidationError: false });
  } else {
    state.patient = null;
    resetResultForPatientChange();
    syncPredictButtonAvailability();
  }
});
elements.csvFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file && !state.isBusy) loadCsvRecord(file);
});
elements.csvExample.addEventListener("click", downloadExampleCsv);
elements.useManualRecord.addEventListener("click", useManualRecord);
elements.manualFields.addEventListener("input", () => {
  if (!state.isBusy && elements.patientSource.value === "manual") {
    applyManualRecord({ showValidationError: false });
  }
});
elements.predictButton.addEventListener("click", runPrediction);
elements.explainButton.addEventListener("click", runExplanation);
elements.reportButton.addEventListener("click", runReport);
elements.downloadReport.addEventListener("click", downloadReportHtml);
elements.printReport.addEventListener("click", printReportDocument);
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
