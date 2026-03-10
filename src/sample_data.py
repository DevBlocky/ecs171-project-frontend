from __future__ import annotations

from functools import lru_cache
import random

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.preprocessing import RAW_FEATURE_COLUMNS, TARGET_COLUMN


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
