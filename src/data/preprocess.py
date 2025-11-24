from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.features.performance_score import (
    PERFORMANCE_SCORE_COL,
    PERFORMANCE_CLASS_COL,
)
from src.utils.config import get_config


cfg = get_config()

ID_COL = "Athlete_ID"

CATEGORICAL_FEATURES = cfg["features"]["categorical"]
NUMERIC_FEATURES = cfg["features"]["numeric"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розбиває повний датафрейм на train/val.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)


def build_preprocessor() -> ColumnTransformer:
    """
    Створює sklearn ColumnTransformer для числових та категоріальних ознак,
    використовуючи списки з config.yaml.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def get_feature_target_matrices(
    df: pd.DataFrame,
):
    """
    Повертає X, y_reg (performance_score), y_cls (performance_class).
    Передбачається, що df вже містить performance_score та performance_class.
    """
    for col in [PERFORMANCE_SCORE_COL, PERFORMANCE_CLASS_COL]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")

    X = df[FEATURE_COLS].copy()
    y_reg = df[PERFORMANCE_SCORE_COL].copy()
    y_cls = df[PERFORMANCE_CLASS_COL].copy()

    return X, y_reg, y_cls
