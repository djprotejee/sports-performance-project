import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from src.utils.config import get_config


cfg = get_config()

_default_url = cfg["database"]["url"]
DATABASE_URL = os.getenv("DATABASE_URL", _default_url)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
