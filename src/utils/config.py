from functools import lru_cache
from pathlib import Path
import yaml


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Читає configs/config.yaml один раз і кешує результат.
    """
    config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
