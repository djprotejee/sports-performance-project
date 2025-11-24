import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data.load import load_raw_data
from src.features.performance_score import add_performance_score, add_performance_class
from src.data.preprocess import (
    train_val_split,
    build_preprocessor,
    get_feature_target_matrices,
)
from src.models.metrics import regression_metrics, classification_metrics
from src.utils.config import get_config


cfg = get_config()
ROOT_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT_DIR / cfg["artifacts"]["models_dir"]
REPORTS_DIR = ROOT_DIR / cfg["artifacts"]["reports_dir"]
PLOTS_DIR = ROOT_DIR / cfg["artifacts"]["plots_dir"]

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_regressor():
    params = cfg["models"]["regressor"]["params"]
    return RandomForestRegressor(**params)


def _build_classifier():
    params = cfg["models"]["classifier"]["params"]
    return RandomForestClassifier(**params)


def _plot_feature_importance(
    importances: np.ndarray,
    feature_names: np.ndarray,
    title: str,
    out_path: Path,
    top_n: int = 20,
):
    """Зберігає barplot важливості ознак."""
    idx = np.argsort(importances)[::-1][:top_n]
    imp = importances[idx]
    names = feature_names[idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(imp)), imp[::-1])
    plt.yticks(range(len(imp)), names[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _save_shap_summary(model, X_proc: np.ndarray, feature_names, title: str, out_path: Path, is_classifier=False):
    """Рахує SHAP і зберігає summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)

    plt.figure()
    if is_classifier:
        # для мультикласу беремо середнє по класах
        if isinstance(shap_values, list):
            sv = np.mean(np.stack(shap_values, axis=0), axis=0)
        else:
            sv = shap_values
        shap.summary_plot(sv, X_proc, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_proc, feature_names=feature_names, show=False)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def train():
    # 1. load data
    df = load_raw_data()

    # 2. add performance_score and performance_class
    df = add_performance_score(df)
    df = add_performance_class(df)

    # 3. train/val split
    train_df, val_df = train_val_split(df)

    # 4. features/targets
    X_train, y_reg_train, y_cls_train = get_feature_target_matrices(train_df)
    X_val, y_reg_val, y_cls_val = get_feature_target_matrices(val_df)

    # 5. preprocessor
    preprocessor_reg = build_preprocessor()

    # 6. regression model
    reg_model = _build_regressor()
    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_reg),
            ("model", reg_model),
        ]
    )
    reg_pipeline.fit(X_train, y_reg_train)
    y_reg_pred = reg_pipeline.predict(X_val)
    reg_metrics = regression_metrics(y_reg_val, y_reg_pred)

    # save regressor
    reg_path = MODELS_DIR / "regressor.pkl"
    joblib.dump(reg_pipeline, reg_path)

    # feature importance (regressor)
    # отримаємо фічі після трансформації
    feature_names_reg = reg_pipeline.named_steps["preprocessor"].get_feature_names_out()
    reg_model_inner = reg_pipeline.named_steps["model"]
    reg_importances = reg_model_inner.feature_importances_

    _plot_feature_importance(
        reg_importances,
        np.array(feature_names_reg),
        title="Regressor Feature Importance",
        out_path=PLOTS_DIR / "regressor_feature_importance.png",
    )

    # SHAP для регресора
    X_train_proc_reg = reg_pipeline.named_steps["preprocessor"].transform(X_train)
    _save_shap_summary(
        reg_model_inner,
        X_train_proc_reg,
        feature_names_reg,
        title="Regressor SHAP Summary",
        out_path=PLOTS_DIR / "regressor_shap_summary.png",
        is_classifier=False,
    )

    # 7. classifier model
    preprocessor_cls = build_preprocessor()

    le = LabelEncoder()
    y_cls_train_encoded = le.fit_transform(y_cls_train)
    y_cls_val_encoded = le.transform(y_cls_val)

    cls_model = _build_classifier()
    cls_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_cls),
            ("model", cls_model),
        ]
    )
    cls_pipeline.fit(X_train, y_cls_train_encoded)
    y_cls_pred_encoded = cls_pipeline.predict(X_val)
    y_cls_pred = le.inverse_transform(y_cls_pred_encoded)

    cls_metrics = classification_metrics(y_cls_val, y_cls_pred)

    cls_artifact = {
        "pipeline": cls_pipeline,
        "label_encoder": le,
        "classes": le.classes_.tolist(),
    }
    cls_path = MODELS_DIR / "classifier.pkl"
    joblib.dump(cls_artifact, cls_path)

    # feature importance (classifier)
    feature_names_cls = cls_pipeline.named_steps["preprocessor"].get_feature_names_out()
    cls_model_inner = cls_pipeline.named_steps["model"]
    cls_importances = cls_model_inner.feature_importances_

    _plot_feature_importance(
        cls_importances,
        np.array(feature_names_cls),
        title="Classifier Feature Importance",
        out_path=PLOTS_DIR / "classifier_feature_importance.png",
    )

    # # SHAP для класифікатора
    # X_train_proc_cls = cls_pipeline.named_steps["preprocessor"].transform(X_train)
    # _save_shap_summary(
    #     cls_model_inner,
    #     X_train_proc_cls,
    #     feature_names_cls,
    #     title="Classifier SHAP Summary",
    #     out_path=PLOTS_DIR / "classifier_shap_summary.png",
    #     is_classifier=True,
    # )

    # 8. save metrics
    with open(REPORTS_DIR / "metrics_regression.json", "w", encoding="utf-8") as f:
        json.dump(reg_metrics, f, indent=2, ensure_ascii=False)

    with open(REPORTS_DIR / "metrics_classification.json", "w", encoding="utf-8") as f:
        json.dump(cls_metrics, f, indent=2, ensure_ascii=False)

    print("Training completed.")
    print("Regression metrics:", reg_metrics)
    print("Classification metrics:", cls_metrics)


if __name__ == "__main__":
    train()
