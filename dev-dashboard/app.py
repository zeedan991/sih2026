"""Internal Streamlit console for the independently running FastAPI service."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import streamlit as st


API_BASE_URL = os.getenv("QML_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180


def api_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call FastAPI without importing any backend implementation code."""

    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"API unavailable at {API_BASE_URL}: {getattr(error, 'reason', error)}") from error


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


st.set_page_config(
    page_title="Q-Trace · Dev Console",
    page_icon="⚛",
    layout="wide",
)
st.markdown(
    """
    <style>
      :root { --quantum:#4A3F8C; --classical:#0E7C7B; --amber:#B8863D; }
      [data-testid="stAppViewContainer"] { background:#f5f7f6; }
      .dev-kicker { font:600 .74rem/1.2 monospace; letter-spacing:.16em;
                    color:#4A3F8C; text-transform:uppercase; }
      .dev-title { font-size:2.2rem; font-weight:750; letter-spacing:-.04em;
                   margin:.25rem 0 .2rem; }
      .dev-note { color:#53605d; max-width:70ch; }
      .result-card { background:white; border:1px solid #d7dcdd;
                     border-radius:16px; padding:1.1rem 1.25rem; }
      .result-card.quantum { border-top:4px solid var(--quantum); }
      .result-card.classical { border-top:4px solid var(--classical); }
      .result-label { font:600 .75rem/1.2 monospace; letter-spacing:.12em;
                      text-transform:uppercase; color:#69726f; }
      .result-diagnosis { font-size:1.65rem; font-weight:750; margin:.35rem 0; }
      .result-confidence { font:600 1.15rem/1.3 monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="dev-kicker">Internal tooling · API-only client</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-title">Q-Trace model console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dev-note">Inspect the real six-model quantum ensemble beside both '
    'classical feature configurations. The judge-facing experience is served at '
    '<code>http://127.0.0.1:8000/</code>.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Service")
    st.code(API_BASE_URL, language=None)
    try:
        health = api_json("/health")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    if health["models_loaded"]:
        st.success("Models ready")
        st.metric("Quantum members", health["quantum_members"])
        st.metric("Classical fits", health["classical_models"])
        runtime_config = health["runtime_configuration"]
        st.caption(
            f"Live training: {runtime_config['quantum_training_limit']} quantum rows / "
            f"{runtime_config['classical_training_rows']} classical rows; "
            f"{runtime_config['vqc_epochs']} VQC epochs, seed {runtime_config['seed']}. "
            "The saved three-seed benchmark uses a separate 200-row quantum pool."
        )
    elif health["status"] == "error":
        st.error(health.get("error") or "Model loading failed")
        st.stop()
    else:
        st.info("Models are training. Refresh in about a minute.")
        if st.button("Refresh status", width="stretch"):
            st.rerun()
        st.stop()

    try:
        disease_catalog = api_json("/diseases")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
    module_by_id = {
        module["disease_id"]: module for module in disease_catalog["modules"]
    }
    disease_id = st.selectbox(
        "Disease module",
        tuple(module_by_id),
        format_func=lambda identifier: module_by_id[identifier]["short_title"],
    )
    active_module = module_by_id[disease_id]

    st.divider()
    st.caption("Selected clinical features")
    for feature_name in active_module["selected_feature_names"]:
        st.markdown(f"- `{feature_name}`")

prediction_tab, benchmark_tab, raw_tab = st.tabs(
    ["Live inference", "Benchmark evidence", "Raw API"]
)

with prediction_tab:
    try:
        catalog = api_json(f"/patients?limit=12&disease_id={disease_id}")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    patients = catalog["patients"]
    source_mode = st.radio(
        "Input source",
        ("Held-out benchmark record", "Named one-row CSV"),
        horizontal=True,
    )
    if source_mode == "Held-out benchmark record":
        selected_patient = st.selectbox(
            f"Held-out {active_module['short_title']} record",
            patients,
            format_func=lambda patient: (
                f'{patient["name"]}'
                + (" · model disagreement" if patient["has_disagreement"] else "")
            ),
        )
    else:
        feature_count = len(catalog["feature_names"])
        uploaded = st.file_uploader(
            f"Upload a one-row CSV with the exact {feature_count} feature names",
            type="csv",
        )
        if uploaded is None:
            st.info("Upload one named record to enable inference.")
            st.stop()
        frame = pd.read_csv(uploaded)
        if frame.shape != (1, feature_count):
            st.error(f"The CSV must contain exactly one row and {feature_count} columns.")
            st.stop()
        try:
            validated = api_json(
                "/ingest",
                method="POST",
                payload={
                    "disease_id": disease_id,
                    "record": {name: float(frame.iloc[0][name]) for name in frame.columns},
                },
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
            st.stop()
        selected_positions = [validated["feature_names"].index(name) for name in catalog["selected_feature_names"]]
        selected_patient = {
            "id": "uploaded",
            "name": "Uploaded clinical record",
            "true_label": None,
            "features": validated["features"],
            "selected_values": [validated["features"][position] for position in selected_positions],
            "has_disagreement": False,
        }
        if validated["warnings"]:
            st.warning(f'{len(validated["warnings"])} value(s) fall outside the benchmark observed ranges.')
        else:
            st.success("CSV schema and observed ranges validated.")
    previous_patient = st.session_state.get("dev_patient")
    active_record_key = f"{disease_id}:{selected_patient['id']}"
    if st.session_state.get("dev_record_key") != active_record_key:
        for key in ("dev_prediction", "dev_patient", "dev_explanation", "dev_report"):
            st.session_state.pop(key, None)
        st.session_state["dev_deep"] = False
        st.session_state["dev_record_key"] = active_record_key
    selected_frame = pd.DataFrame(
        {
            "feature": catalog["selected_feature_names"],
            "raw value": selected_patient["selected_values"],
        }
    )
    st.dataframe(selected_frame, hide_index=True, width="stretch")

    if st.button("Run hybrid prediction", type="primary", width="stretch"):
        for key in ("dev_prediction", "dev_patient", "dev_explanation", "dev_report"):
            st.session_state.pop(key, None)
        try:
            with st.spinner("Running all six quantum members and both classical views…"):
                st.session_state["dev_prediction"] = api_json(
                    "/predict",
                    method="POST",
                    payload={
                        "disease_id": disease_id,
                        "features": selected_patient["features"],
                    },
                )
                st.session_state["dev_patient"] = selected_patient
        except RuntimeError as error:
            st.error(str(error))

    prediction = st.session_state.get("dev_prediction")
    current_patient = st.session_state.get("dev_patient")
    if prediction and current_patient:
        if prediction["agreement"]["agrees"]:
            st.success(prediction["agreement"]["message"])
        else:
            st.warning(prediction["agreement"]["message"])

        quantum_column, classical_column = st.columns(2)
        quantum = prediction["quantum"]
        classical = prediction["classical"]
        with quantum_column:
            st.markdown(
                f"""
                <div class="result-card quantum">
                  <div class="result-label">Quantum · six-model OOB ensemble</div>
                  <div class="result-diagnosis">{quantum['label'].title()}</div>
                  <div class="result-confidence">{percentage(quantum['confidence'])} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Per-model quantum output"):
                st.dataframe(
                    pd.DataFrame(quantum["per_model"]),
                    hide_index=True,
                    width="stretch",
                )

        with classical_column:
            full = classical["full_feature"]
            same = classical["same_4_feature"]
            st.markdown(
                f"""
                <div class="result-card classical">
                  <div class="result-label">Classical · full {len(catalog['feature_names'])}-feature LogReg</div>
                  <div class="result-diagnosis">{full['label'].title()}</div>
                  <div class="result-confidence">{percentage(full['confidence'])} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Same-four comparison: "
                f"**{same['label'].title()}**, {percentage(same['confidence'])} confidence"
            )

        st.divider()
        st.subheader("Explain model attribution")
        explanation_choice = st.selectbox(
            "Explanation scope",
            (
                "Quantum · VQC fast",
                "Quantum · full six-model ensemble",
                f"Classical · full {len(catalog['feature_names'])}-feature LogReg",
                "Classical · matched 4-feature LogReg",
            ),
        )
        deep = explanation_choice == "Quantum · full six-model ensemble"
        explanation_model = {
            f"Classical · full {len(catalog['feature_names'])}-feature LogReg": "classical_full_feature",
            "Classical · matched 4-feature LogReg": "classical_same_4_feature",
        }.get(explanation_choice, "quantum")
        previous_explanation = st.session_state.get("dev_explanation")
        selected_scope = (
            "full_ensemble" if deep else explanation_model if explanation_model != "quantum" else "vqc_fast"
        )
        if previous_explanation and previous_explanation["scope"] != selected_scope:
            st.session_state.pop("dev_explanation", None)
        if deep:
            st.warning(
                "Deep mode recomputes quantum kernels for each perturbation and may "
                "take 70 seconds or longer."
            )
        if st.button(
            "Compute deep attribution" if deep else "Compute fast attribution",
            width="stretch",
        ):
            expected = "70 seconds or longer" if deep else "approximately 7–30 seconds"
            st.session_state.pop("dev_explanation", None)
            started = time.perf_counter()
            with st.status(
                f"Computing explanation… expected {expected}",
                expanded=True,
            ) as status:
                st.write(
                    "SHAP and LIME are evaluating the same selected clinical features."
                )
                try:
                    explanation = api_json(
                        "/explain",
                        method="POST",
                        payload={
                            "disease_id": disease_id,
                            "features": current_patient["features"],
                            "model": explanation_model,
                            "allow_slow": deep,
                        },
                    )
                    st.session_state["dev_explanation"] = explanation
                    status.update(
                        label=f"Attribution complete in {time.perf_counter() - started:.1f}s",
                        state="complete",
                        expanded=False,
                    )
                except RuntimeError as error:
                    status.update(label="Attribution could not complete", state="error")
                    st.error(str(error))

        explanation = st.session_state.get("dev_explanation")
        if explanation:
            # D-23: the explanation panel intentionally renders no confidence.
            explanation_frame = pd.DataFrame(
                {
                    "feature": explanation["feature_names"],
                    "SHAP": explanation["shap_values"],
                    "LIME": explanation["lime_values"],
                    "direction": explanation["directions"],
                }
            )
            st.caption(f'Attribution scope: `{explanation["scope"]}`')
            st.dataframe(explanation_frame, hide_index=True, width="stretch")
            st.bar_chart(
                explanation_frame.set_index("feature")[["SHAP", "LIME"]],
                horizontal=True,
            )

        st.divider()
        if st.button("Generate local evidence report", width="stretch"):
            try:
                st.session_state["dev_report"] = api_json(
                    "/report",
                    method="POST",
                    payload={
                        "disease_id": disease_id,
                        "features": current_patient["features"],
                    },
                )
            except RuntimeError as error:
                st.error(str(error))
        if report := st.session_state.get("dev_report"):
            st.subheader("AI-assisted evidence summary")
            st.write(report["ai_evidence_summary"])
            st.download_button(
                "Download report JSON",
                json.dumps(report, indent=2),
                file_name=f"{report['report_id']}.json",
                mime="application/json",
                width="stretch",
            )

with benchmark_tab:
    try:
        baseline_payload = api_json(f"/baselines?disease_id={disease_id}")
        metrics_payload = api_json(f"/metrics?disease_id={disease_id}")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
    st.info(metrics_payload["positioning"])
    for configuration, detail in baseline_payload["configurations"].items():
        st.subheader(configuration.replace("_", " ").title())
        frame = pd.DataFrame(detail["models"]).T.reset_index(names="model")
        st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander("Verified three-seed quantum record"):
        st.json(metrics_payload["quantum"])
    with st.expander("Leakage-safe generalization and clinical metrics"):
        st.json(metrics_payload.get("generalization", {}))

with raw_tab:
    st.caption("Useful when diagnosing response-shape or label-direction issues.")
    st.json(
        {
            "health": health,
            "patient": st.session_state.get("dev_patient"),
            "prediction": st.session_state.get("dev_prediction"),
            "explanation": st.session_state.get("dev_explanation"),
        }
    )
