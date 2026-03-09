from src.model_registry import Model, ModelResponse
from typing import Any
from pathlib import Path
from joblib import load
import pandas as pd
import numpy as np

class LogisticRegressionModel(Model):
    __minfo: dict[str, Any]

    @property
    def label(self) -> str:
        return "Logistic Regression"

    def __init__(self):
        model_path = Path(__file__).resolve().parents[2] / "artifacts" / "logistic_regression.joblib"
        self.__minfo = load(model_path)

    def ask(self, params: pd.DataFrame) -> ModelResponse:
        # transform this dataframe into expected input for __model
        pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        amt_cols = [1,2,3,4,5,6]
        X = pd.DataFrame({
            "AGE": params['AGE'],

            "LIMIT_BAL": params['LIMIT_BAL'],
            "BILL_TOTAL": params[[f'BILL_AMT{i}' for i in range(1,7)]].sum(axis=1),
            "PAY_TOTAL": params[[f'PAY_AMT{i}' for i in range(1,7)]].sum(axis=1),
            "PAY_MAX": params[pay_cols].max(axis=1),
            "PAY_LATE_COUNT": (params[pay_cols] > 0).sum(axis=1),

            # we need to encode the one-hot columns manually because
            # pd.get_dummies won't create all the required columns
            "SEX_2": (params['SEX'] == 2).astype(int),
            "EDUCATION_2": (params['EDUCATION'] == 2).astype(int),
            "EDUCATION_3": (params['EDUCATION'] == 3).astype(int),
            "EDUCATION_4": (params['EDUCATION'].replace([0,5,6],4) == 4).astype(int),
            "MARRIAGE_1": (params['MARRIAGE'] == 1).astype(int),
            "MARRIAGE_2": (params['MARRIAGE'] == 2).astype(int),
            "MARRIAGE_3": (params['MARRIAGE'] == 3).astype(int),
        })
        for i in amt_cols:
            X[f"AMT{i}"] = np.where(
                params[f"BILL_AMT{i}"] == 0,
                0,  # or np.nan if you want to mark undefined ratios
                params[f"PAY_AMT{i}"] / params[f"BILL_AMT{i}"]
            )
        # make sure features are in the correct order
        feature_order = self.__minfo["feature_order"]
        for col in feature_order:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_order]
        
        # scale the columns
        scaler = self.__minfo["scaler"]
        scaled_cols = self.__minfo["scaled_cols"]
        X[scaled_cols] = scaler.transform(X[scaled_cols])

        # use the model to predict
        model = self.__minfo["model"]
        probability = model.predict_proba(X)[0,1]
        defaults = probability >= self.__minfo["threshold"]
        confidence = probability if defaults else 1-probability

        return ModelResponse(label=self.label, defaults=defaults, confidence=confidence)
