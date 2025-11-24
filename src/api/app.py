import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from src.db.base import get_db
from src.db.crud import create_prediction, get_recent_predictions
from src.models.predict import predict_performance


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"

app = Flask(__name__)


@contextmanager
def db_session():
    """
    Обгортка над get_db() для використання у Flask.
    """
    gen = get_db()
    db = next(gen)
    try:
        yield db
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        payload: Dict[str, Any] = request.get_json()
        if payload is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        prediction_result = predict_performance(payload)

        with db_session() as db:
            create_prediction(
                db,
                request_payload=payload,
                predicted_score=prediction_result["performance_score"],
                predicted_class=prediction_result["performance_class"],
                extra_info={"class_probabilities": prediction_result["class_probabilities"]},
            )

        return jsonify(prediction_result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # у реалі можна логувати через logging
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/history", methods=["GET"])
def history_endpoint():
    limit_param = request.args.get("limit", default="50")
    try:
        limit = int(limit_param)
    except ValueError:
        limit = 50

    with db_session() as db:
        records = get_recent_predictions(db, limit=limit)

    result = []
    for r in records:
        result.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "athlete_id": r.athlete_id,
                "sport": r.sport,
                "gender": r.gender,
                "performance_score": r.performance_score,
                "performance_class": r.performance_class,
            }
        )

    return jsonify(result)


@app.route("/models", methods=["GET"])
def models_info():
    """
    Повертає метрики моделей з JSON-файлів.
    """
    reg_path = REPORTS_DIR / "metrics_regression.json"
    cls_path = REPORTS_DIR / "metrics_classification.json"

    response = {}

    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            response["regression"] = json.load(f)
    else:
        response["regression"] = None

    if cls_path.exists():
        with open(cls_path, "r", encoding="utf-8") as f:
            response["classification"] = json.load(f)
    else:
        response["classification"] = None

    return jsonify(response)


if __name__ == "__main__":
    # для локального запуску без gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
