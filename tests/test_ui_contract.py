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
