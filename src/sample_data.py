from __future__ import annotations

from functools import lru_cache
import random
from typing import Any

from joblib import load
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.artifacts import artifact_path
from src.preprocessing import (
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    align_feature_order,
    build_summary_feature_frame,
    build_xgboost_feature_frame,
)


@lru_cache(maxsize=1)
def load_credit_default_data() -> pd.DataFrame:
    dataset = fetch_ucirepo(name="Default of Credit Card Clients")
    df = dataset.data.original.copy()
    column_mapping = {
        row["name"]: row["description"] if pd.notna(row["description"]) else row["name"]
        for _, row in dataset.variables.iterrows()
    }
    return df.rename(columns=column_mapping)


def sample_input_row() -> dict[str, object]:
    df = load_credit_default_data()
    row_index = random.randrange(len(df))
    row = df.iloc[row_index]
    record = {column: int(row[column]) for column in RAW_FEATURE_COLUMNS}
    return {
        "source": "UCI Default of Credit Card Clients",
        "row_index": int(row_index),
        "target": int(row[TARGET_COLUMN]),
        "record": record,
    }


def _score_rows(df: pd.DataFrame) -> tuple[str, np.ndarray, float]:
    raw_features = df[RAW_FEATURE_COLUMNS].copy()

    xgb_path = artifact_path("xgboost.joblib")
    if xgb_path.exists():
        loaded = load(xgb_path)
        if isinstance(loaded, dict):
            model = loaded["model"]
            feature_order = list(loaded["feature_order"])
            threshold = float(loaded.get("threshold", 0.5))
        else:
            model = loaded
            feature_order = list(getattr(loaded, "feature_names_in_", []))
            threshold = 0.5

        if feature_order:
            X_xgb = build_xgboost_feature_frame(raw_features)
            X_xgb = align_feature_order(X_xgb, feature_order)
            probabilities = model.predict_proba(X_xgb)[:, 1]
            return "XGBoost", probabilities, threshold

    logistic = load(artifact_path("logistic_regression.joblib"))
    X_log = build_summary_feature_frame(raw_features)
    X_log = align_feature_order(X_log, logistic["feature_order"])
    X_log[logistic["scaled_cols"]] = logistic["scaler"].transform(X_log[logistic["scaled_cols"]])
    probabilities = logistic["model"].predict_proba(X_log)[:, 1]
    return "Logistic Regression", probabilities, float(logistic["threshold"])


def _build_explanation(
    row: pd.Series,
    *,
    reference_model: str,
    probability: float,
    threshold: float,
    bucket: str,
) -> str:
    late_count = int((row[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] > 0).sum())
    recent_status = int(row["PAY_0"])
    utilization = 100 * float(row["BILL_AMT1"]) / max(float(row["LIMIT_BAL"]), 1.0)
    payment_coverage = 100 * float(row["PAY_AMT1"]) / max(abs(float(row["BILL_AMT1"])), 1.0)
    actual_outcome = "Default" if int(row[TARGET_COLUMN]) == 1 else "No Default"
    if bucket == "high_risk":
        return (
            f"Actual outcome: {actual_outcome}. {reference_model} scores this row at "
            f"{probability:.1%} default risk. It shows {late_count} late-payment months, "
            f"PAY_0 = {recent_status}, and current utilization around {utilization:.0f}%."
        )
    if bucket == "borderline":
        return (
            f"Actual outcome: {actual_outcome}. {reference_model} scores this row at "
            f"{probability:.1%}, very close to the {threshold:.0%} decision boundary. "
            f"It mixes PAY_0 = {recent_status}, {late_count} late-payment months, and "
            f"about {payment_coverage:.0f}% payment coverage on the latest bill."
        )
    return (
        f"Actual outcome: {actual_outcome}. {reference_model} scores this row at "
        f"{probability:.1%} default risk. It has {late_count} late-payment months, "
        f"PAY_0 = {recent_status}, and roughly {utilization:.0f}% current utilization."
    )


@lru_cache(maxsize=1)
def load_grounded_presets() -> list[dict[str, Any]]:
    df = load_credit_default_data().copy()
    reference_model, probabilities, threshold = _score_rows(df)
    scored = df.copy()
    scored["reference_probability"] = probabilities
    scored["distance_to_boundary"] = np.abs(scored["reference_probability"] - threshold)
    scored["late_count"] = (scored[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] > 0).sum(axis=1)

    selected_rows = {
        "high_risk": scored[scored[TARGET_COLUMN] == 1]
        .sort_values(["reference_probability", "late_count"], ascending=[False, False])
        .iloc[0],
        "borderline": scored.sort_values("distance_to_boundary", ascending=True).iloc[0],
        "lower_risk": scored[scored[TARGET_COLUMN] == 0]
        .sort_values(["reference_probability", "late_count"], ascending=[True, True])
        .iloc[0],
    }

    presets = [
        ("high_risk", "High Risk"),
        ("borderline", "Borderline"),
        ("lower_risk", "Lower Risk"),
    ]

    results: list[dict[str, Any]] = []
    for key, label in presets:
        row = selected_rows[key]
        record = {column: int(row[column]) for column in RAW_FEATURE_COLUMNS}
        results.append(
            {
                "name": key,
                "label": label,
                "source": "UCI Default of Credit Card Clients",
                "row_index": int(row.name),
                "target": int(row[TARGET_COLUMN]),
                "reference_model": reference_model,
                "reference_probability": float(row["reference_probability"]),
                "explanation": _build_explanation(
                    row,
                    reference_model=reference_model,
                    probability=float(row["reference_probability"]),
                    threshold=threshold,
                    bucket=key,
                ),
                "record": record,
            }
        )
    return results
