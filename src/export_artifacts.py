from __future__ import annotations

import random

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from ucimlrepo import fetch_ucirepo
import tensorflow as tf
import xgboost as xgb

from src.artifacts import artifact_path
from src.preprocessing import (
    PAY_STATUS_COLUMNS,
    TARGET_COLUMN,
    align_feature_order,
    apply_outlier_pruning,
    build_clean_feature_frame,
    build_summary_feature_frame,
    build_xgboost_feature_frame,
    dummy_columns_for_prefixes,
)


def load_dataset() -> pd.DataFrame:
    dataset = fetch_ucirepo(name="Default of Credit Card Clients")
    df = dataset.data.original.copy()
    column_mapping = {
        row["name"]: row["description"] if pd.notna(row["description"]) else row["name"]
        for _, row in dataset.variables.iterrows()
    }
    return df.rename(columns=column_mapping)


def best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    threshold = 0.5
    best_f1 = -1.0
    for candidate in np.arange(0.1, 0.9, 0.01):
        predictions = (probabilities >= candidate).astype(int)
        score = f1_score(y_true, predictions)
        if score > best_f1:
            best_f1 = score
            threshold = float(candidate)
    return threshold


def create_ann_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(30,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", AUC(name="auc")],
    )
    return model


def export_logistic(df: pd.DataFrame) -> None:
    fsdf = build_summary_feature_frame(df)
    X = fsdf.copy()
    y = df[TARGET_COLUMN]

    scores: list[float] = []
    model_infos: list[dict[str, object]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

    dummy_cols = dummy_columns_for_prefixes(X, ["SEX", "EDUCATION", "MARRIAGE"])
    scaled_cols = X.columns.difference(dummy_cols)

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx]

        X_train[scaled_cols] = X_train[scaled_cols].astype(float)
        X_test[scaled_cols] = X_test[scaled_cols].astype(float)

        scaler = MinMaxScaler()
        X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
        X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

        model = LogisticRegression(
            max_iter=2500,
            C=0.2,
            solver="newton-cholesky",
            class_weight="balanced",
            penalty="l2",
        )
        model.fit(X_train, y_train)

        test_score = model.predict_proba(X_test)[:, 1]
        score = float(np.mean(test_score))
        scores.append(score)
        model_infos.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "model": model,
                "scaler": scaler,
                "scaled_cols": list(scaled_cols),
                "feature_order": list(X_train.columns),
            }
        )

    best_idx = int(np.argmax(scores))
    model_info = model_infos[best_idx]
    train_probs = model_info["model"].predict_proba(model_info["X_train"])[:, 1]
    model_info["threshold"] = best_threshold(model_info["y_train"], train_probs)
    del model_info["X_train"]
    del model_info["y_train"]
    dump(model_info, artifact_path("logistic_regression.joblib"))


def export_gaussian_naive_bayes(df: pd.DataFrame) -> None:
    fsdf = build_summary_feature_frame(df)
    X = fsdf.copy()
    y = df[TARGET_COLUMN]

    dummy_cols = dummy_columns_for_prefixes(X, ["SEX", "EDUCATION", "MARRIAGE"])
    scaled_cols = X.columns.difference(dummy_cols)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols].astype(float))

    model = GaussianNB()
    model.fit(X_train, y_train)

    dump(
        {
            "model": model,
            "scaler": scaler,
            "scaled_cols": list(scaled_cols),
            "feature_order": list(X_train.columns),
            "threshold": 0.5,
        },
        artifact_path("gaussian_naive_bayes.joblib"),
    )


def export_neural_network(df: pd.DataFrame) -> None:
    pruned_df = apply_outlier_pruning(df).drop(columns=["ID"])
    clean_df = build_clean_feature_frame(pruned_df)

    dummy_cols = dummy_columns_for_prefixes(clean_df, ["SEX", "EDUCATION", "MARRIAGE"])
    scaled_cols = clean_df.columns.difference(dummy_cols.union([TARGET_COLUMN, *PAY_STATUS_COLUMNS]))

    X = clean_df.drop(columns=[TARGET_COLUMN]).copy()
    y = clean_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols].astype(float))
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols].astype(float))

    model = create_ann_model()
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=[early_stopping],
    )

    probabilities = model.predict(X_test, verbose=0).reshape(-1)
    threshold = best_threshold(y_test, probabilities)
    model.save(artifact_path("artificial_neural_network.keras"))
    dump(
        {
            "feature_order": list(X_train.columns),
            "scaled_cols": list(scaled_cols),
            "scaler": scaler,
            "threshold": threshold,
        },
        artifact_path("artificial_neural_network.joblib"),
    )


def export_xgboost(df: pd.DataFrame) -> None:
    pruned_df = apply_outlier_pruning(df).drop(columns=["ID"])
    xgb_df = build_xgboost_feature_frame(pruned_df)

    X = xgb_df.drop(columns=[TARGET_COLUMN])
    y = xgb_df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=700,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=7,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=0)

    dump(
        {
            "model": model,
            "feature_order": list(X_train.columns),
            "threshold": 0.5,
        },
        artifact_path("xgboost.joblib"),
    )


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    df = load_dataset()
    export_logistic(df)
    export_gaussian_naive_bayes(df)
    export_neural_network(df)
    export_xgboost(df)


if __name__ == "__main__":
    main()
