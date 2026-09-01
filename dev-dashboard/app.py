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
    except urllib.error.URLError as error:
        raise RuntimeError(f"API unavailable at {API_BASE_URL}: {error.reason}") from error


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
    elif health["status"] == "error":
        st.error(health.get("error") or "Model loading failed")
        st.stop()
    else:
        st.info("Models are training. Refresh in about a minute.")
        if st.button("Refresh status", width="stretch"):
            st.rerun()
        st.stop()

    st.divider()
    st.caption("Selected clinical features")
    for feature_name in health["selected_features"]:
        st.markdown(f"- `{feature_name}`")

prediction_tab, benchmark_tab, raw_tab = st.tabs(
    ["Live inference", "Benchmark evidence", "Raw API"]
)

with prediction_tab:
    try:
        catalog = api_json("/patients?limit=12")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    patients = catalog["patients"]
    selected_patient = st.selectbox(
        "Held-out WBCD patient",
        patients,
        format_func=lambda patient: (
            f'{patient["name"]}'
            + (" · model disagreement" if patient["has_disagreement"] else "")
        ),
    )
    selected_frame = pd.DataFrame(
        {
            "feature": catalog["selected_feature_names"],
            "raw value": selected_patient["selected_values"],
        }
    )
    st.dataframe(selected_frame, hide_index=True, width="stretch")

    if st.button("Run hybrid prediction", type="primary", width="stretch"):
        with st.spinner("Running all six quantum members and both classical views…"):
            st.session_state["dev_prediction"] = api_json(
                "/predict",
                method="POST",
                payload={"features": selected_patient["features"]},
            )
            st.session_state["dev_patient"] = selected_patient
            st.session_state.pop("dev_explanation", None)

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
                  <div class="result-label">Classical · full 30-feature LogReg</div>
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
        st.subheader("Explain the quantum attribution")
        deep = st.toggle(
            "Use full six-model explanation",
            help="Off: VQC-only, typically ~7s. On: includes QSVMs, typically 70s+.",
        )
        if deep:
            st.warning(
                "Deep mode recomputes quantum kernels for each perturbation and may "
                "take 70 seconds or longer."
            )
        if st.button(
            "Compute deep attribution" if deep else "Compute fast attribution",
            width="stretch",
        ):
            expected = "70 seconds or longer" if deep else "about 7 seconds"
            started = time.perf_counter()
            with st.status(
                f"Computing explanation… expected {expected}",
                expanded=True,
            ) as status:
                st.write(
                    "SHAP and LIME are evaluating the same selected clinical features."
                )
                explanation = api_json(
                    "/explain",
                    method="POST",
                    payload={
                        "features": current_patient["features"],
                        "model": "quantum",
                        "allow_slow": deep,
                    },
                )
                st.session_state["dev_explanation"] = explanation
                status.update(
                    label=f"Attribution complete in {time.perf_counter() - started:.1f}s",
                    state="complete",
                    expanded=False,
                )

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

with benchmark_tab:
    baseline_payload = api_json("/baselines")
    metrics_payload = api_json("/metrics")
    st.info(metrics_payload["positioning"])
    for configuration, detail in baseline_payload["configurations"].items():
        st.subheader(configuration.replace("_", " ").title())
        frame = pd.DataFrame(detail["models"]).T.reset_index(names="model")
        st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander("Verified three-seed quantum record"):
        st.json(metrics_payload["quantum"])

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
