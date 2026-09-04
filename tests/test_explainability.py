"""Phase 3 regression tests for SHAP/LIME explainability behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from backend.explain import shap_lime
from backend.explain.shap_lime import ClassicalExplainabilityService, ExplainabilityService


CLINICAL_NAMES = np.asarray(
    [
        "mean concave points",
        "worst radius",
        "worst perimeter",
        "worst concave points",
    ]
)


def test_classical_shap_and_lime_keep_real_clinical_feature_names() -> None:
    from backend.classical.baselines import train_evaluate_baselines
    from backend.data.pipeline import prepare_breast_cancer_data

    prepared = prepare_breast_cancer_data(random_state=42)
    estimator = train_evaluate_baselines(
        prepared,
        random_state=42,
        model_names=("logistic_regression",),
    )["same_4_feature"]["logistic_regression"].estimator
    service = ClassicalExplainabilityService(
        estimator,
        prepared.X_train_selected,
        prepared.selected_feature_names,
        scope="classical_same_4_feature",
        background_size=8,
        shap_nsamples=8,
        lime_num_samples=80,
        random_state=42,
    )

    result = service.explain(prepared.X_test_selected[0])

    assert result.scope == "classical_same_4_feature"
    assert result.shap.feature_names == tuple(prepared.selected_feature_names)
    assert result.lime.feature_names == tuple(prepared.selected_feature_names)
    assert not any(name.startswith("feature_") for name in result.shap.feature_names)
BACKGROUND = np.asarray(
    [
        [0.10, 0.20, 0.30, 0.40],
        [0.20, 0.30, 0.40, 0.50],
        [0.30, 0.40, 0.50, 0.60],
        [0.40, 0.50, 0.60, 0.70],
        [0.50, 0.60, 0.70, 0.80],
        [0.60, 0.70, 0.80, 0.90],
    ],
    dtype=float,
)


@dataclass
class FakeMember:
    name: str
    paradigm: str
    weight: float
    coefficients: np.ndarray
    intercept: float
    calls: int = field(default=0, init=False)

    def predict_benign_proba(self, features):
        self.calls += 1
        X = np.asarray(features, dtype=float)
        return np.clip(self.intercept + X @ self.coefficients, 0.0, 1.0)


def _prepared(names=CLINICAL_NAMES, background=BACKGROUND):
    return SimpleNamespace(
        selected_feature_names=np.asarray(names),
        X_train_quantum=np.asarray(background, dtype=float),
    )


def _members():
    return (
        FakeMember(
            "vqc_a",
            "VQC",
            0.20,
            np.asarray([0.10, -0.03, 0.02, 0.01]),
            0.35,
        ),
        FakeMember(
            "vqc_b",
            "VQC",
            0.30,
            np.asarray([0.02, 0.04, 0.01, -0.02]),
            0.45,
        ),
        FakeMember(
            "qsvm_angle",
            "QSVM",
            0.50,
            np.asarray([-0.04, 0.03, 0.05, 0.02]),
            0.50,
        ),
    )


def _install_fake_explainers(monkeypatch, captured):
    class FakeKernelExplainer:
        def __init__(self, model, data, feature_names, link):
            captured["shap_feature_names"] = tuple(feature_names)
            captured["shap_link"] = link
            captured["background"] = np.asarray(data).copy()
            captured["shap_model_probability"] = float(model(np.ones((1, 4)))[0])
            self.expected_value = 0.40

        def shap_values(self, values, nsamples, **kwargs):
            captured["shap_nsamples"] = nsamples
            return np.asarray([[0.03, -0.02, 0.01, 0.005]], dtype=float)

    class FakeLimeExplanation:
        intercept = {1: 0.41}
        local_pred = np.asarray([0.47])

        def as_map(self):
            return {1: [(0, 0.05), (1, -0.03), (2, 0.02), (3, 0.01)]}

    class FakeLimeTabularExplainer:
        def __init__(
            self,
            training_data,
            *,
            mode,
            feature_names,
            class_names,
            discretize_continuous,
            random_state,
        ):
            captured["lime_feature_names"] = tuple(feature_names)
            captured["lime_class_names"] = tuple(class_names)
            captured["lime_random_state"] = random_state
            self.scaler = SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4))
            assert mode == "classification"
            assert discretize_continuous is False

        def explain_instance(
            self,
            row,
            predict_fn,
            *,
            labels,
            num_features,
            num_samples,
        ):
            captured["lime_num_samples"] = num_samples
            captured["lime_probabilities"] = predict_fn(
                np.asarray(row, dtype=float).reshape(1, -1)
            )
            assert labels == (1,)
            assert num_features == 4
            return FakeLimeExplanation()

    monkeypatch.setattr(shap_lime.shap, "KernelExplainer", FakeKernelExplainer)
    monkeypatch.setattr(
        shap_lime,
        "LimeTabularExplainer",
        FakeLimeTabularExplainer,
    )


def test_default_explanation_is_vqc_only_and_preserves_real_feature_names(monkeypatch):
    captured = {}
    _install_fake_explainers(monkeypatch, captured)
    members = _members()
    service = ExplainabilityService(
        SimpleNamespace(members=members),
        _prepared(),
        background_size=4,
        shap_nsamples=21,
        lime_num_samples=123,
        random_state=17,
    )
    events = []

    result = service.explain(
        np.asarray([0.25, 0.35, 0.45, 0.55]),
        progress=events.append,
    )

    assert result.scope == "fast_vqc"
    assert result.status == "complete"
    assert [event.state for event in events] == ["computing", "complete"]
    assert events[0].expected_timing == "about 7 seconds"
    assert captured["shap_feature_names"] == tuple(CLINICAL_NAMES)
    assert captured["lime_feature_names"] == tuple(CLINICAL_NAMES)
    assert captured["lime_class_names"] == ("malignant", "benign")
    assert captured["shap_link"] == "identity"
    assert captured["shap_nsamples"] == 21
    assert captured["lime_num_samples"] == 123
    assert tuple(item.feature_name for item in result.shap.attributions) == tuple(
        CLINICAL_NAMES
    )
    assert tuple(item.feature_name for item in result.lime.attributions) == tuple(
        CLINICAL_NAMES
    )
    assert result.shap.output_class == "benign"
    assert result.shap.positive_label == 1
    assert result.cross_check.top_feature_overlap
    assert 0.0 <= result.cross_check.top_feature_overlap_ratio <= 1.0
    assert members[0].calls > 0
    assert members[1].calls > 0
    assert members[2].calls == 0

    # The two selected VQC weights (0.2, 0.3) must be renormalized to 0.4/0.6.
    point = np.ones((1, 4))
    expected = 0.4 * members[0].predict_benign_proba(point)[0] + 0.6 * members[
        1
    ].predict_benign_proba(point)[0]
    assert captured["shap_model_probability"] == pytest.approx(expected)


@pytest.mark.parametrize("scope", ["full_ensemble", "qsvm_diagnostic"])
def test_slow_scopes_require_explicit_opt_in(scope):
    service = ExplainabilityService(SimpleNamespace(members=_members()), _prepared())

    with pytest.raises(PermissionError, match="allow_slow=True"):
        service.predict_benign_probability(BACKGROUND[:1], scope=scope)


def test_qsvm_diagnostic_scope_is_available_but_never_used_by_default(monkeypatch):
    captured = {}
    _install_fake_explainers(monkeypatch, captured)
    members = _members()
    service = ExplainabilityService(SimpleNamespace(members=members), _prepared())

    result = service.explain(
        BACKGROUND[0],
        scope="qsvm_diagnostic",
        allow_slow=True,
    )

    assert result.scope == "qsvm_diagnostic"
    assert result.expected_timing == "at least 70 seconds"
    assert members[0].calls == 0
    assert members[1].calls == 0
    assert members[2].calls > 0


def test_full_ensemble_scope_uses_every_member_after_explicit_opt_in(monkeypatch):
    captured = {}
    _install_fake_explainers(monkeypatch, captured)
    members = _members()
    service = ExplainabilityService(SimpleNamespace(members=members), _prepared())

    result = service.explain(
        BACKGROUND[0],
        scope="full_ensemble",
        allow_slow=True,
    )

    assert result.scope == "full_ensemble"
    assert result.member_names == tuple(member.name for member in members)
    assert all(member.calls > 0 for member in members)


@pytest.mark.parametrize(
    "bad_names",
    [
        ["PC1", "PC2", "PC3", "PC4"],
        ["feature_0", "feature_1", "feature_2", "feature_3"],
    ],
)
def test_generic_component_or_feature_names_are_rejected(bad_names):
    with pytest.raises(ValueError, match="real clinical feature names"):
        ExplainabilityService(
            SimpleNamespace(members=_members()),
            _prepared(names=bad_names),
        )


def test_failed_explanation_emits_failed_progress_state(monkeypatch):
    class BrokenKernelExplainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("synthetic SHAP failure")

    monkeypatch.setattr(shap_lime.shap, "KernelExplainer", BrokenKernelExplainer)
    service = ExplainabilityService(SimpleNamespace(members=_members()), _prepared())
    events = []

    with pytest.raises(RuntimeError, match="synthetic SHAP failure"):
        service.explain(BACKGROUND[0], progress=events.append)

    assert [event.state for event in events] == ["computing", "failed"]
    assert "synthetic SHAP failure" in events[-1].message


def test_real_linear_predictor_has_additive_shap_and_clinical_lime_names():
    linear_member = FakeMember(
        "linear_vqc",
        "VQC",
        1.0,
        np.asarray([0.08, -0.05, 0.04, 0.02]),
        0.40,
    )
    service = ExplainabilityService(
        SimpleNamespace(members=(linear_member,)),
        _prepared(),
        background_size=6,
        shap_nsamples=16,
        lime_num_samples=750,
        random_state=11,
    )
    patient = np.asarray([0.25, 0.35, 0.45, 0.55])

    shap_result = service.explain_shap(patient)
    lime_result = service.explain_lime(patient)

    assert shap_result.reconstructed_probability == pytest.approx(
        shap_result.prediction_probability,
        abs=1e-7,
    )
    assert abs(shap_result.additivity_residual) < 1e-7
    assert lime_result.surrogate_additivity_residual == pytest.approx(0.0, abs=1e-7)
    assert tuple(item.feature_name for item in lime_result.attributions) == tuple(
        CLINICAL_NAMES
    )
    assert {item.direction for item in lime_result.attributions}.issubset(
        {"toward_benign", "toward_malignant", "neutral"}
    )
    cross_check = shap_lime.cross_check_attributions(shap_result, lime_result, top_k=4)
    assert cross_check.sign_agreement_ratio == pytest.approx(1.0)
