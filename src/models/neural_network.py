from __future__ import annotations

from typing import Any

from joblib import load
import pandas as pd
from tensorflow.keras.models import load_model

from src.artifacts import artifact_path
from src.model_registry import Model, ModelResponse
from src.preprocessing import align_feature_order, build_clean_feature_frame


class NeuralNetworkModel(Model):
    __minfo: dict[str, Any]

    @property
    def label(self) -> str:
        return "Artificial Neural Network"

    def __init__(self):
        self.__minfo = load(artifact_path("artificial_neural_network.joblib"))
        self.__minfo["model"] = load_model(artifact_path("artificial_neural_network.keras"))

    def ask(self, params: pd.DataFrame) -> ModelResponse:
        X = build_clean_feature_frame(params)
        X = align_feature_order(X, self.__minfo["feature_order"])
        scaled_cols = self.__minfo["scaled_cols"]
        X[scaled_cols] = self.__minfo["scaler"].transform(X[scaled_cols])

        probability = float(self.__minfo["model"].predict(X, verbose=0).reshape(-1)[0])
        threshold = float(self.__minfo["threshold"])
        defaults = probability >= threshold
        confidence = probability if defaults else 1 - probability
        return ModelResponse(label=self.label, defaults=defaults, confidence=confidence)
