from pathlib import Path
import pandas as pd

from src.utils.config import get_config


def load_raw_data() -> pd.DataFrame:
    """
    Завантажує сирі дані із CSV.

    Шлях береться з configs/config.yaml -> data.raw_csv_path
    """
    cfg = get_config()
    rel_path = cfg["data"]["raw_csv_path"]
    data_path = Path(__file__).resolve().parents[2] / rel_path
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {data_path}")
    df = pd.read_csv(data_path)
    return df
