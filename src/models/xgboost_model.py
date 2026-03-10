from __future__ import annotations

from typing import Any

from joblib import load
import pandas as pd
from xgboost import XGBClassifier

from src.artifacts import artifact_path
from src.model_registry import Model, ModelResponse
from src.preprocessing import align_feature_order, build_xgboost_feature_frame


class XGBoostModel(Model):
    __model: XGBClassifier
    __feature_order: list[str]
    __threshold: float

    @property
    def label(self) -> str:
        return "XGBoost"

    def __init__(self):
        loaded = load(artifact_path("xgboost.joblib"))
        if isinstance(loaded, dict):
            self.__model = loaded["model"]
            self.__feature_order = list(loaded["feature_order"])
            self.__threshold = float(loaded.get("threshold", 0.5))
            return

        if not isinstance(loaded, XGBClassifier):
            raise TypeError("Unsupported XGBoost artifact format.")

        self.__model = loaded
        self.__feature_order = list(getattr(loaded, "feature_names_in_", []))
        if not self.__feature_order:
            raise ValueError("XGBoost artifact is missing feature names.")
        self.__threshold = 0.5

    def ask(self, params: pd.DataFrame) -> ModelResponse:
        X = build_xgboost_feature_frame(params)
        X = align_feature_order(X, self.__feature_order)

        probability = float(self.__model.predict_proba(X)[0, 1])
        defaults = probability >= self.__threshold
        confidence = probability if defaults else 1 - probability
        return ModelResponse(label=self.label, defaults=defaults, confidence=confidence)
