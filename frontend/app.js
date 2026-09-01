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
  healthTimer: null,
  explanationTimer: null,
};

const elements = {
  serviceState: document.querySelector("#service-state"),
  serviceStateLabel: document.querySelector("#service-state-label"),
  patientSelect: document.querySelector("#patient-select"),
  patientHelp: document.querySelector("#patient-help"),
  selectedFeatureChips: document.querySelector("#selected-feature-chips"),
  selectedFeatureList: document.querySelector("#selected-feature-list"),
  predictButton: document.querySelector("#predict-button"),
  analysisLoading: document.querySelector("#analysis-loading"),
  errorBanner: document.querySelector("#error-banner"),
  resultsSection: document.querySelector("#results-section"),
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
      setServiceState("ready", "Models ready · 6Q + 8C");
      if (state.healthTimer) {
        window.clearInterval(state.healthTimer);
        state.healthTimer = null;
      }
      if (!state.catalog) await loadPatients();
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
      "The service is fitting four 100-epoch VQCs and two QSVMs. This first startup usually takes about a minute.";
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
        const suffix = patient.has_disagreement ? " · known model disagreement" : "";
        return `<option value="${index}">${escapeHtml(patient.name)}${suffix}</option>`;
      })
      .join("");
    elements.patientSelect.disabled = false;
    elements.predictButton.disabled = false;
    elements.patientHelp.textContent =
      "Real held-out row · ground truth appears only after the model response.";
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

function resetResultForPatientChange() {
  state.prediction = null;
  state.explanation = null;
  elements.resultsSection.hidden = true;
  elements.explainSection.hidden = true;
  elements.explanationResult.hidden = true;
  elements.explanationLoading.hidden = true;
  elements.disagreementBanner.hidden = true;
  elements.agreementNote.hidden = true;
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
  const duration = 1050;
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

  elements.referenceLabel.textContent = labelCase(state.patient.true_label);
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
          <span class="member-kind">${escapeHtml(member.paradigm)} · ${(member.weight * 100).toFixed(1)}w</span>
          <strong>${escapeHtml(labelCase(member.label))} · ${formatPercent(member.confidence)}</strong>
        </div>`,
    )
    .join("");
  renderAgreement(payload.agreement);
  elements.resultsSection.hidden = false;
  elements.explainSection.hidden = false;
  activateRail("model");
  window.setTimeout(() => {
    elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 120);
}

async function runPrediction() {
  if (!state.patient) return;
  clearError();
  elements.predictButton.disabled = true;
  elements.analysisLoading.hidden = false;
  try {
    const payload = await fetchJson("/predict", {
      method: "POST",
      body: JSON.stringify({ features: state.patient.features }),
    });
    renderPrediction(payload);
  } catch (error) {
    showError(`Prediction could not complete: ${error.message}`);
  } finally {
    elements.analysisLoading.hidden = true;
    elements.predictButton.disabled = false;
  }
}

function selectExplanationScope(deep) {
  state.deepExplanation = deep;
  elements.scopeOptions.forEach((option) => {
    option.classList.toggle(
      "is-selected",
      (option.dataset.scope === "deep") === deep,
    );
  });
  elements.slowDisclosure.hidden = !deep;
  elements.explainButton.querySelector("span").textContent = deep
    ? "Request full six-model attribution"
    : "Explain with fast VQC scope";
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
  elements.explanationLoadingTitle.textContent = deep
    ? "Recomputing full ensemble kernels"
    : "Evaluating VQC perturbations";
  elements.explanationLoadingDetail.textContent = deep
    ? "Two QSVM members recompute kernels against the explanation background for every SHAP sample."
    : "SHAP and LIME are checking four named clinical features across four VQC members.";
  elements.explanationExpected.textContent = deep
    ? "Typical: 70 seconds or longer"
    : "Typical: about 7 seconds";
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
  elements.scopeReadout.textContent = payload.scope === "full_ensemble" ? "FULL ENSEMBLE" : "VQC FAST";
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
      const featureValue = payload.top_features.find(
        (feature) => feature.feature_name === name,
      )?.feature_value;
      return `
        <div class="attribution-row">
          <div class="attribution-name">
            <strong>${escapeHtml(name)}</strong>
            <span>${featureValue === undefined ? "selected clinical input" : `input ${formatFeatureValue(featureValue)}`}</span>
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
  if (!state.patient || !state.prediction) return;
  if (
    state.deepExplanation &&
    !window.confirm(
      "Full-ensemble SHAP recomputes quantum kernels and may take 70 seconds or longer. Continue?",
    )
  ) {
    return;
  }
  clearError();
  beginExplanationLoading(state.deepExplanation);
  try {
    const payload = await fetchJson("/explain", {
      method: "POST",
      body: JSON.stringify({
        features: state.patient.features,
        model: "quantum",
        allow_slow: state.deepExplanation,
      }),
    });
    renderExplanation(payload);
  } catch (error) {
    showError(`Explanation could not complete: ${error.message}`);
  } finally {
    endExplanationLoading();
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
  renderSelectedPatient(Number(event.target.value));
});
elements.predictButton.addEventListener("click", runPrediction);
elements.explainButton.addEventListener("click", runExplanation);
elements.scopeOptions.forEach((option) => {
  option.addEventListener("click", () => {
    selectExplanationScope(option.dataset.scope === "deep");
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
