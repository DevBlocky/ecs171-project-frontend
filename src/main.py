from typing import Any
import pandas as pd
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
from src.model_registry import ModelResponse, get_models

app = FastAPI(title="ECS171 Model Frontend")

@app.get("/models")
def get_model_registry() -> list[dict[str, str]]:
    models = get_models()
    return [{"id": id, "label": models[id].label} for id in models]

@app.post("/models")
def ask_models(payload: dict[str, Any] = Body(...)) -> list[ModelResponse]:
    params = pd.DataFrame([payload])
    models = get_models()
    return [model.ask(params.copy()) for model in models.values()]

@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok", "message": "pong"}

app.mount("/", StaticFiles(directory="public/", html=True), name="public")
