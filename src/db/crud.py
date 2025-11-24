from typing import List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.db.models import Prediction


def _generate_athlete_id(db: Session) -> str:
    """
    Генерує новий athlete_id, починаючи з "0":
    беремо останній athlete_id, інкрементуємо на 1.
    Якщо немає жодного запису — повертаємо "0".
    """
    last = (
        db.query(Prediction)
        .order_by(desc(Prediction.id))
        .first()
    )

    if last is None or last.athlete_id is None:
        return "0"

    try:
        return str(int(last.athlete_id) + 1)
    except ValueError:
        # якщо там раптом не число — fallback на id
        return str(last.id + 1)


def create_prediction(
    db: Session,
    *,
    request_payload: Dict[str, Any],
    predicted_score: float,
    predicted_class: str,
    model_regressor: str = "random_forest_regressor",
    model_classifier: str = "random_forest_classifier",
    extra_info: Dict[str, Any] | None = None,
) -> Prediction:
    """
    Створює запис про прогноз у БД.
    Якщо athlete_id не переданий з фронта — генеруємо самі (0,1,2,...).
    """
    athlete_id = request_payload.get("Athlete_ID")
    if not athlete_id:
        athlete_id = _generate_athlete_id(db)

    sport = request_payload.get("Sport")
    gender = request_payload.get("Gender")

    obj = Prediction(
        athlete_id=athlete_id,
        sport=sport,
        gender=gender,
        performance_score=predicted_score,
        performance_class=predicted_class,
        model_regressor=model_regressor,
        model_classifier=model_classifier,
        request_payload=request_payload,
        extra_info=extra_info,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_recent_predictions(db: Session, limit: int = 100) -> List[Prediction]:
    """
    Повертає останні N прогнозів.
    """
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )
