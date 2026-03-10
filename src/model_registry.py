from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from pydantic import BaseModel
import pandas as pd

class ModelResponse(BaseModel):
    label: str
    defaults: bool
    confidence: float

class Model(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        pass
    @abstractmethod
    def ask(self, params: pd.DataFrame) -> ModelResponse:
        pass

models = None
def get_models() -> Mapping[str, Model]:
    global models
    if not models:
        from src.models.logistic_regression import LogisticRegressionModel
        model_builders = {
            "logistic_regression": (
                Path(__file__).resolve().parents[1] / "artifacts" / "logistic_regression.joblib",
                LogisticRegressionModel,
            ),
            "gaussian_naive_bayes": (
                Path(__file__).resolve().parents[1] / "artifacts" / "bayes_network.joblib",
                lambda: __import__(
                    "src.models.gaussian_naive_bayes",
                    fromlist=["GaussianNaiveBayesModel"],
                ).GaussianNaiveBayesModel(),
            ),
            "artificial_neural_network": (
                Path(__file__).resolve().parents[1] / "artifacts" / "neural_network.joblib",
                lambda: __import__(
                    "src.models.neural_network",
                    fromlist=["NeuralNetworkModel"],
                ).NeuralNetworkModel(),
            ),
            "xgboost": (
                Path(__file__).resolve().parents[1] / "artifacts" / "xgboost.joblib",
                lambda: __import__(
                    "src.models.xgboost_model",
                    fromlist=["XGBoostModel"],
                ).XGBoostModel(),
            ),
        }
        loaded_models: dict[str, Model] = {}
        for model_id, (artifact, builder) in model_builders.items():
            if not artifact.exists():
                continue
            try:
                loaded_models[model_id] = builder()
            except Exception:
                continue
        models = loaded_models
    return models
