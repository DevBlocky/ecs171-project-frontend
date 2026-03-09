from abc import ABC, abstractmethod
from collections.abc import Mapping
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
        models = {
            "logistic_regression": LogisticRegressionModel()
        }
    return models
