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
