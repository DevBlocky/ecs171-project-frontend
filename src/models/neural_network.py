from __future__ import annotations

from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

from src.artifacts import artifact_path
from src.model_registry import Model, ModelResponse
from src.preprocessing import align_feature_order, fit_clean_preprocessor, transform_clean_feature_frame
from src.sample_data import load_credit_default_data


class NeuralNetworkModel(Model):
    __model: Sequential
    __scaler: StandardScaler
    __scaled_cols: list[str]
    __feature_order: list[str]
    __threshold: float

    @property
    def label(self) -> str:
        return "Artificial Neural Network"

    def __init__(self):
        loaded = load(artifact_path("neural_network.joblib"))
        if not isinstance(loaded, Sequential):
            raise TypeError("Unsupported neural network artifact format.")

        preprocessed = fit_clean_preprocessor(load_credit_default_data())
        X = preprocessed["X"]
        y = preprocessed["y"]
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=y,
        )

        probabilities = loaded.predict(X_test, verbose=0).reshape(-1)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.arange(0.1, 0.9, 0.01):
            predictions = (probabilities >= threshold).astype(int)
            score = f1_score(y_test, predictions)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self.__model = loaded
        self.__scaler = preprocessed["scaler"]
        self.__scaled_cols = list(preprocessed["scaled_cols"])
        self.__feature_order = list(preprocessed["feature_order"])
        self.__threshold = best_threshold

    def ask(self, params: pd.DataFrame) -> ModelResponse:
        X = transform_clean_feature_frame(params, self.__scaler, self.__scaled_cols)
        X = align_feature_order(X, self.__feature_order)

        probability = float(self.__model.predict(X, verbose=0).reshape(-1)[0])
        defaults = probability >= self.__threshold
        confidence = probability if defaults else 1 - probability
        return ModelResponse(label=self.label, defaults=defaults, confidence=confidence)
