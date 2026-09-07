"""Static UI contract checks that complement browser verification."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_dashboard_is_an_http_only_api_client() -> None:
    source = (PROJECT_ROOT / "dev-dashboard" / "app.py").read_text(encoding="utf-8")

    assert "from backend" not in source
    assert "import backend" not in source
    for endpoint in ("/health", "/patients", "/predict", "/explain", "/baselines", "/metrics"):
        assert endpoint in source
    assert '"full_feature"' in source
    assert '"same_4_feature"' in source
    assert "70 seconds or longer" in source


def test_judge_frontend_contains_required_design_system_and_equal_dials() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    css = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8").lower()

    for token in (
        "--paper: #f1f3f2",
        "--ink: #12181c",
        "--signal-teal: #0e7c7b",
        "--quantum-indigo: #4a3f8c",
        "--caution-amber: #b8863d",
        "--line: #d7dcdd",
    ):
        assert token in css
    for stage in ("Data", "Select", "Model", "Explain"):
        assert f"<strong>{stage}</strong>" in html
    assert 'id="quantum-progress"' in html
    assert 'id="classical-progress"' in html
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert 'id="disagreement-banner"' in html
    assert 'id="explanation-loading"' in html
    assert "prefers-reduced-motion" in css


def test_frontend_explanation_is_attribution_only_with_explicit_slow_opt_in() -> None:
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert 'fetchJson("/explain"' in source
    assert "allow_slow: state.deepExplanation" in source
    assert "state.deepExplanation = false" not in source
    assert "prediction_probability" not in source
    assert "benign_probability" not in source[source.index("function renderExplanation") :]
    assert "70 seconds or longer" in source
    assert "forceDisagreement" in source
    assert '"classical_full_feature"' in source
    assert '"classical_same_4_feature"' in source


def test_judge_frontend_supports_named_csv_ingestion() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert 'id="patient-source"' in html
    assert 'id="csv-file"' in html
    assert 'id="csv-example"' in html
    assert 'id="ingestion-warnings"' in html
    assert 'fetchJson("/ingest"' in source
    assert "parseCsvRecord" in source


def test_judge_frontend_supports_disease_modules_manual_intake_and_reports() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "disease-select",
        "manual-intake",
        "manual-fields",
        "report-button",
        "report-section",
        "download-report",
        "print-report",
    ):
        assert f'id="{element_id}"' in html
    for endpoint in ('fetchJson("/diseases"', 'fetchJson("/report"'):
        assert endpoint in source
    assert "disease_id: state.activeDiseaseId" in source
    assert "buildPrintableReport" in source
    assert "Print / Save PDF" in html


def test_frontend_surfaces_malignant_sensitivity_and_cross_validation() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert 'id="clinical-evidence"' in html
    assert "malignant sensitivity" in html.lower()
    assert "five-fold" in html.lower()
    assert 'fetchJson("/metrics"' in source


def test_clinical_workspace_preserves_the_complete_dom_binding_contract() -> None:
    import re

    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    ids = re.findall(r'id="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "UI IDs must be unique"
    for selector_id in re.findall(r'querySelector\("#([^"]+)"\)', source):
        assert selector_id in ids, f"Missing DOM binding: {selector_id}"
    assert 'class="clinical-workspace"' in html
    assert 'class="hero"' not in html
    assert "Research use only" in html
    assert "Source+Sans+3" in html


def test_patient_and_scope_are_locked_during_async_requests() -> None:
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert "state.isBusy" in source
    assert "elements.patientSelect.disabled = busy" in source
    assert "option.disabled = busy" in source
    assert "generation === state.generation" in source
    assert 'selectExplanationScope("quantum-fast")' in source


def test_attribution_measurements_come_from_the_raw_patient_record() -> None:
    source = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    render = source[source.index("function renderExplanation"):source.index("async function runExplanation")]

    assert "state.patient.selected_values[selectedIndex]" in render
    assert "Raw ${formatFeatureValue(featureValue)}" in render
    assert "payload.top_features" not in render


def test_live_training_budget_is_not_presented_as_the_saved_benchmark() -> None:
    html = (PROJECT_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "frontend/app.js").read_text(encoding="utf-8")
    assert 'id="runtime-context"' in html
    assert "config.quantum_training_limit" in source
    assert "config.classical_training_rows" in source
    assert "separate three-seed, 200-row quantum run" in source


def test_catalog_failure_does_not_stop_health_retry_polling() -> None:
    source = (PROJECT_ROOT / "frontend/app.js").read_text(encoding="utf-8")
    assert "if (state.catalog && state.healthTimer)" in source
    assert source.index("await loadPatients()") < source.index("window.clearInterval(state.healthTimer)")


def test_run_button_uses_current_catalog_or_manual_record_without_stale_state() -> None:
    html = (PROJECT_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "frontend/app.js").read_text(encoding="utf-8")

    selected_patient = source[
        source.index("function renderSelectedPatient") : source.index("function parseCsvRecord")
    ]
    manual_record = source[
        source.index("function applyManualRecord") : source.index("async function loadEvidence")
    ]
    run_prediction = source[
        source.index("async function runPrediction") : source.index("function buildPrintableReport")
    ]

    assert "syncPredictButtonAvailability();" in selected_patient
    assert 'String(field.value).trim() === ""' in manual_record
    assert "state.patient = null;" in manual_record
    assert 'elements.patientSource.value === "manual"' in run_prediction
    assert "applyManualRecord({ reset: false })" in run_prediction
    assert 'elements.manualFields.addEventListener("input"' in source
    assert '/static/app.js?v=0.6.4' in html
