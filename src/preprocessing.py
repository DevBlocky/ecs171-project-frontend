from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "default payment next month"
CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
PRUNED_COLUMNS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "LIMIT_BAL",
]
RAW_FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
]


def build_summary_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = pd.DataFrame(
        {
            "SEX": df["SEX"],
            "EDUCATION": df["EDUCATION"].replace([0, 5, 6], 4),
            "MARRIAGE": df["MARRIAGE"],
            "AGE": df["AGE"],
            "LIMIT_BAL": df["LIMIT_BAL"],
            "BILL_TOTAL": df[BILL_AMOUNT_COLUMNS].sum(axis=1),
            "PAY_TOTAL": df[PAY_AMOUNT_COLUMNS].sum(axis=1),
            "PAY_MAX": df[PAY_STATUS_COLUMNS].max(axis=1),
            "PAY_LATE_COUNT": (df[PAY_STATUS_COLUMNS] > 0).sum(axis=1),
        }
    )
    for index in range(1, 7):
        bill_column = f"BILL_AMT{index}"
        pay_column = f"PAY_AMT{index}"
        summary_df[f"AMT{index}"] = np.where(
            df[bill_column] == 0,
            0,
            df[pay_column] / df[bill_column],
        )
    return pd.get_dummies(
        summary_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )


def apply_outlier_pruning(df: pd.DataFrame) -> pd.DataFrame:
    prune_mask = pd.Series(True, index=df.index)
    for column in PRUNED_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        prune_mask &= df[column].between(lower, upper)
    return df.loc[prune_mask].copy()


def build_clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        df.copy(),
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )


def fit_clean_preprocessor(df: pd.DataFrame) -> dict[str, object]:
    clean_df = apply_outlier_pruning(df).copy()
    if "ID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["ID"])

    excluded_columns = set(CATEGORICAL_COLUMNS + [TARGET_COLUMN] + PAY_STATUS_COLUMNS)
    scaled_cols = [column for column in clean_df.columns if column not in excluded_columns]

    scaler = StandardScaler()
    clean_df[scaled_cols] = scaler.fit_transform(clean_df[scaled_cols].astype(float))
    encoded_df = pd.get_dummies(
        clean_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )

    X = encoded_df.drop(columns=[TARGET_COLUMN]).copy()
    y = encoded_df[TARGET_COLUMN].copy()
    return {
        "scaler": scaler,
        "scaled_cols": scaled_cols,
        "feature_order": list(X.columns),
        "X": X,
        "y": y,
    }


def transform_clean_feature_frame(
    df: pd.DataFrame,
    scaler: StandardScaler,
    scaled_cols: list[str],
) -> pd.DataFrame:
    clean_df = df.copy()
    if "ID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["ID"])

    present_scaled_cols = [column for column in scaled_cols if column in clean_df.columns]
    clean_df[present_scaled_cols] = scaler.transform(clean_df[present_scaled_cols].astype(float))
    return pd.get_dummies(
        clean_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )


def build_xgboost_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    xgb_df = df.copy()
    eps = 1e-6

    for index in range(1, 7):
        xgb_df[f"UTIL_{index}"] = xgb_df[f"BILL_AMT{index}"] / (xgb_df["LIMIT_BAL"] + eps)
    util_columns = [f"UTIL_{index}" for index in range(1, 7)]
    xgb_df["UTIL_MEAN"] = xgb_df[util_columns].mean(axis=1)
    xgb_df["UTIL_MAX"] = xgb_df[util_columns].max(axis=1)
    xgb_df["UTIL_STD"] = xgb_df[util_columns].std(axis=1)
    xgb_df["UTIL_RECENT"] = xgb_df["UTIL_1"]
    xgb_df["UTIL_CHANGE_1_6"] = xgb_df["UTIL_1"] - xgb_df["UTIL_6"]

    for index in range(1, 7):
        xgb_df[f"PAY_RATIO_{index}"] = xgb_df[f"PAY_AMT{index}"] / (
            xgb_df[f"BILL_AMT{index}"].abs() + 1 + eps
        )
    pay_ratio_columns = [f"PAY_RATIO_{index}" for index in range(1, 7)]
    xgb_df["PAY_RATIO_MEAN"] = xgb_df[pay_ratio_columns].mean(axis=1)
    xgb_df["PAY_RATIO_MIN"] = xgb_df[pay_ratio_columns].min(axis=1)
    xgb_df["PAY_RATIO_STD"] = xgb_df[pay_ratio_columns].std(axis=1)
    xgb_df["PAY_RATIO_RECENT"] = xgb_df["PAY_RATIO_1"]

    for index in range(1, 7):
        xgb_df[f"GAP_{index}"] = xgb_df[f"BILL_AMT{index}"] - xgb_df[f"PAY_AMT{index}"]
    gap_columns = [f"GAP_{index}" for index in range(1, 7)]
    xgb_df["GAP_MEAN"] = xgb_df[gap_columns].mean(axis=1)
    xgb_df["GAP_MAX"] = xgb_df[gap_columns].max(axis=1)
    xgb_df["GAP_RECENT"] = xgb_df["GAP_1"]

    xgb_df["MAX_DELQ"] = xgb_df[PAY_STATUS_COLUMNS].max(axis=1)
    xgb_df["MEAN_DELQ"] = xgb_df[PAY_STATUS_COLUMNS].mean(axis=1)
    xgb_df["NUM_DELQ"] = (xgb_df[PAY_STATUS_COLUMNS] > 0).sum(axis=1)
    xgb_df["NUM_SEVERE_DELQ"] = (xgb_df[PAY_STATUS_COLUMNS] >= 2).sum(axis=1)
    xgb_df["RECENT_DELQ"] = xgb_df["PAY_0"]
    xgb_df["DELQ_CHANGE_0_6"] = xgb_df["PAY_0"] - xgb_df["PAY_6"]

    xgb_df["BILL_MEAN"] = xgb_df[BILL_AMOUNT_COLUMNS].mean(axis=1)
    xgb_df["BILL_STD"] = xgb_df[BILL_AMOUNT_COLUMNS].std(axis=1)
    xgb_df["BILL_MAX"] = xgb_df[BILL_AMOUNT_COLUMNS].max(axis=1)
    xgb_df["BILL_CHANGE_1_6"] = xgb_df["BILL_AMT1"] - xgb_df["BILL_AMT6"]

    xgb_df["PAY_AMT_MEAN"] = xgb_df[PAY_AMOUNT_COLUMNS].mean(axis=1)
    xgb_df["PAY_AMT_STD"] = xgb_df[PAY_AMOUNT_COLUMNS].std(axis=1)
    xgb_df["PAY_AMT_MAX"] = xgb_df[PAY_AMOUNT_COLUMNS].max(axis=1)
    xgb_df["PAY_AMT_CHANGE_1_6"] = xgb_df["PAY_AMT1"] - xgb_df["PAY_AMT6"]

    xgb_df["TOTAL_BILL"] = xgb_df[BILL_AMOUNT_COLUMNS].sum(axis=1)
    xgb_df["TOTAL_PAY_AMT"] = xgb_df[PAY_AMOUNT_COLUMNS].sum(axis=1)
    xgb_df["TOTAL_PAY_TO_BILL"] = xgb_df["TOTAL_PAY_AMT"] / (
        xgb_df["TOTAL_BILL"].abs() + 1 + eps
    )
    xgb_df["TOTAL_BILL_TO_LIMIT"] = xgb_df["TOTAL_BILL"] / (xgb_df["LIMIT_BAL"] + eps)
    xgb_df["TOTAL_PAY_TO_LIMIT"] = xgb_df["TOTAL_PAY_AMT"] / (xgb_df["LIMIT_BAL"] + eps)

    xgb_df["UTIL_X_DELQ"] = xgb_df["UTIL_MEAN"] * xgb_df["MAX_DELQ"]
    xgb_df["PAY_RATIO_X_DELQ"] = xgb_df["PAY_RATIO_MEAN"] * xgb_df["MAX_DELQ"]

    return pd.get_dummies(
        xgb_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )


def dummy_columns_for_prefixes(df: pd.DataFrame, prefixes: list[str]) -> pd.Index:
    escaped = "|".join(prefixes)
    return df.filter(regex=rf"^({escaped})_").columns


def align_feature_order(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for column in feature_order:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[feature_order]
