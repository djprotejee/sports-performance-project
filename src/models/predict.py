from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import FEATURE_COLS
from src.utils.config import get_config


cfg = get_config()
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / cfg["artifacts"]["models_dir"]


@lru_cache(maxsize=1)
def _load_regressor():
    reg_path = MODELS_DIR / "regressor.pkl"
    return joblib.load(reg_path)


@lru_cache(maxsize=1)
def _load_classifier():
    cls_path = MODELS_DIR / "classifier.pkl"
    artifact = joblib.load(cls_path)
    return artifact


def _build_input_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    data = {}
    for col in FEATURE_COLS:
        if col not in payload:
            raise ValueError(f"Missing required field '{col}' in request payload.")
        data[col] = [payload[col]]
    return pd.DataFrame(data)


def predict_performance(payload: Dict[str, Any]) -> Dict[str, Any]:
    X = _build_input_dataframe(payload)

    regressor = _load_regressor()
    reg_score = float(regressor.predict(X)[0])

    cls_artifact = _load_classifier()
    cls_pipeline = cls_artifact["pipeline"]
    le = cls_artifact["label_encoder"]
    classes = cls_artifact["classes"]

    cls_proba = cls_pipeline.predict_proba(X)[0]
    cls_idx = int(np.argmax(cls_proba))
    class_label = le.inverse_transform([cls_idx])[0]

    probs_by_class = {cls_name: float(p) for cls_name, p in zip(classes, cls_proba)}

    return {
        "performance_score": reg_score,
        "performance_class": class_label,
        "class_probabilities": probs_by_class,
    }
