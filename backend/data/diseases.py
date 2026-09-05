"""Disease-module registry for the multi-dataset Q-TRACE platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer

from backend.data.diabetes import load_early_stage_diabetes, prepare_diabetes_data
from backend.data.pipeline import prepare_breast_cancer_data


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DiseaseDefinition:
    """Metadata and loaders that make one independently validated module."""

    disease_id: str
    title: str
    short_title: str
    domain: str
    dataset_name: str
    dataset_source: str
    dataset_license: str
    description: str
    positive_class_name: str
    class_labels: tuple[str, str]
    record_prefix: str
    prepare_data: Callable[..., Any]
    load_raw_features: Callable[[], FloatArray]
    feature_kinds: tuple[str, ...]

    def label_for_class(self, prediction: int) -> str:
        if prediction not in (0, 1):
            raise ValueError("prediction must be the numeric class 0 or 1")
        return self.class_labels[prediction]


def _load_breast_features() -> FloatArray:
    return np.asarray(load_breast_cancer().data, dtype=float)


def _load_diabetes_features() -> FloatArray:
    features, _targets = load_early_stage_diabetes()
    return np.asarray(features, dtype=float)


DISEASES: Final[dict[str, DiseaseDefinition]] = {
    "breast_cancer": DiseaseDefinition(
        disease_id="breast_cancer",
        title="Breast cancer signal analysis",
        short_title="Breast oncology",
        domain="Diagnostic breast-mass measurements",
        dataset_name="Wisconsin Breast Cancer Diagnostic",
        dataset_source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        dataset_license="UCI benchmark redistributed by scikit-learn",
        description=(
            "Classifies malignant versus benign breast-mass patterns from 30 "
            "numeric diagnostic measurements."
        ),
        positive_class_name="malignant",
        class_labels=("malignant", "benign"),
        record_prefix="WBCD",
        prepare_data=prepare_breast_cancer_data,
        load_raw_features=_load_breast_features,
        feature_kinds=("continuous",) * 30,
    ),
    "early_diabetes": DiseaseDefinition(
        disease_id="early_diabetes",
        title="Early diabetes screening analysis",
        short_title="Diabetes screening",
        domain="Questionnaire signs and symptoms",
        dataset_name="UCI Early Stage Diabetes Risk Prediction",
        dataset_source=(
            "https://archive.ics.uci.edu/dataset/529/"
            "early+stage+diabetes+risk+prediction+dataset"
        ),
        dataset_license="CC BY 4.0",
        description=(
            "Screens for a positive or negative diabetes signal from age and "
            "15 questionnaire variables; it does not diagnose diabetes."
        ),
        positive_class_name="positive screening signal",
        class_labels=("positive screening signal", "negative screening signal"),
        record_prefix="UCI-DM",
        prepare_data=prepare_diabetes_data,
        load_raw_features=_load_diabetes_features,
        feature_kinds=("continuous",) + ("binary",) * 15,
    ),
}
DEFAULT_DISEASE_ID: Final[str] = "breast_cancer"


def disease_definition(disease_id: str) -> DiseaseDefinition:
    try:
        return DISEASES[disease_id]
    except KeyError as error:
        raise ValueError(
            f"unknown disease_id {disease_id!r}; choose one of {', '.join(DISEASES)}"
        ) from error


__all__ = [
    "DEFAULT_DISEASE_ID",
    "DISEASES",
    "DiseaseDefinition",
    "disease_definition",
]
