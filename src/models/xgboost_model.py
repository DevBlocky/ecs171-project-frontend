from __future__ import annotations

from typing import Any

from joblib import load
import pandas as pd

from src.artifacts import artifact_path
from src.model_registry import Model, ModelResponse
from src.preprocessing import align_feature_order, build_xgboost_feature_frame


class XGBoostModel(Model):
    __minfo: dict[str, Any]

    @property
    def label(self) -> str:
        return "XGBoost"

    def __init__(self):
        self.__minfo = load(artifact_path("xgboost.joblib"))

    def ask(self, params: pd.DataFrame) -> ModelResponse:
        X = build_xgboost_feature_frame(params)
        X = align_feature_order(X, self.__minfo["feature_order"])

        probability = float(self.__minfo["model"].predict_proba(X)[0, 1])
        threshold = float(self.__minfo.get("threshold", 0.5))
        defaults = probability >= threshold
        confidence = probability if defaults else 1 - probability
        return ModelResponse(label=self.label, defaults=defaults, confidence=confidence)
