from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func

from src.db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    athlete_id = Column(String, nullable=True)
    sport = Column(String, nullable=True)
    gender = Column(String, nullable=True)

    performance_score = Column(Float, nullable=False)
    performance_class = Column(String, nullable=False)

    model_regressor = Column(String, nullable=False)
    model_classifier = Column(String, nullable=False)

    request_payload = Column(JSON, nullable=False)
    extra_info = Column(JSON, nullable=True)
