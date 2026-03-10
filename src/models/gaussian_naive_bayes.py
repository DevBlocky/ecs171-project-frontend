from __future__ import annotations

from joblib import load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from src.artifacts import artifact_path
from src.model_registry import Model, ModelResponse
from src.preprocessing import CATEGORICAL_COLUMNS, TARGET_COLUMN
from src.preprocessing import align_feature_order, build_summary_feature_frame


class GaussianNaiveBayesModel(Model):
    __model: GaussianNB
    __scaler: StandardScaler
    __scaled_cols: list[str]
    __feature_order: list[str]

    @property
    def label(self) -> str:
        return "Gaussian Naive Bayes"

    def __init__(self):
        loaded = load(artifact_path("bayes_network.joblib"))
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError("Unsupported Bayes artifact format.")

        bayes_df = pd.get_dummies(
            loaded.copy(),
            columns=CATEGORICAL_COLUMNS,
            drop_first=True,
            dtype=int,
        )
        dummy_cols = bayes_df.filter(regex=r"^(SEX|EDUCATION|MARRIAGE)_").columns
        scaled_cols = bayes_df.columns.difference(dummy_cols.union([TARGET_COLUMN]))

        X = bayes_df.drop(columns=[TARGET_COLUMN]).copy()
        y = bayes_df[TARGET_COLUMN].copy()
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.__scaler = StandardScaler()
        X_train[scaled_cols] = self.__scaler.fit_transform(X_train[scaled_cols].astype(float))
        self.__model = GaussianNB()
        self.__model.fit(X_train, y_train)
        self.__scaled_cols = list(scaled_cols)
        self.__feature_order = list(X_train.columns)

    def ask(self, params: pd.DataFrame) -> ModelResponse:
        X = build_summary_feature_frame(params)
        X = align_feature_order(X, self.__feature_order)
        X[self.__scaled_cols] = self.__scaler.transform(X[self.__scaled_cols])

        probability = float(self.__model.predict_proba(X)[0, 1])
        defaults = probability >= 0.5
        confidence = probability if defaults else 1 - probability
        return ModelResponse(label=self.label, defaults=defaults, confidence=confidence)
