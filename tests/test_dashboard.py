"""Real Streamlit state transitions with an offline HTTP fixture."""

import io
import json
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import urlparse

import pytest
from streamlit.testing.v1 import AppTest

from test_api import StubRuntime


@pytest.fixture
def dashboard(monkeypatch):
    runtime = StubRuntime()

    def urlopen(request, timeout):
        path = urlparse(request.full_url).path
        if path == "/health":
            payload = runtime.health_payload()
        elif path == "/diseases":
            payload = {
                "default_disease_id": "breast_cancer",
                "modules": [
                    {
                        "disease_id": "breast_cancer",
                        "short_title": "Breast oncology",
                        "dataset_name": "Wisconsin Breast Cancer Diagnostic",
                        "selected_feature_names": runtime.health_payload()["selected_features"],
                    }
                ],
            }
        elif path == "/patients":
            payload = runtime.patients_payload(12)
            second = dict(payload["patients"][0], id=8, name="WBCD test patient 008")
            payload["patients"].append(second)
        elif path == "/predict":
            payload = runtime.predict_payload(json.loads(request.data)["features"])
        elif path == "/explain":
            body = json.loads(request.data)
            payload = runtime.explain_payload(
                body["features"],
                model=body["model"],
                allow_slow=body["allow_slow"],
            )
        elif path == "/ingest":
            payload = runtime.ingest_payload(json.loads(request.data)["record"])
        elif path == "/baselines":
            payload = runtime.baselines_payload()
        elif path == "/metrics":
            payload = runtime.metrics_payload()
        else:
            raise AssertionError(f"Unexpected API route: {path}")
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dev-dashboard/app.py"))
    app.run()
    assert not app.exception
    return app


def test_selecting_another_patient_clears_old_results(dashboard):
    dashboard.button[0].click().run()
    assert "dev_prediction" in dashboard.session_state
    dashboard.selectbox[0].select_index(1).run()
    assert not dashboard.exception
    assert "dev_prediction" not in dashboard.session_state
    assert "dev_patient" not in dashboard.session_state
    assert "dev_explanation" not in dashboard.session_state


def test_switching_explanation_scope_clears_old_attribution(dashboard):
    dashboard.button[0].click().run()
    dashboard.button[1].click().run()
    assert dashboard.session_state["dev_explanation"]["scope"] == "vqc_fast"
    dashboard.selectbox[1].select_index(1).run()
    assert not dashboard.exception
    assert "dev_explanation" not in dashboard.session_state


def test_busy_api_error_is_displayed_without_a_stack_trace(dashboard, monkeypatch):
    original = urllib.request.urlopen

    def busy(request, timeout):
        if urlparse(request.full_url).path == "/predict":
            raise urllib.error.HTTPError(
                request.full_url, 429, "Busy", {},
                io.BytesIO(b'{"detail":"Another explanation is running. Please retry."}'),
            )
        return original(request, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", busy)
    dashboard.button[0].click().run()
    assert not dashboard.exception
    assert dashboard.error
