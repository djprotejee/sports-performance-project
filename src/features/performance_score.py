import pandas as pd

PERFORMANCE_SCORE_COL = "performance_score"
PERFORMANCE_CLASS_COL = "performance_class"

# Ми явно кажемо, які колонки беремо в розрахунок
PERF_FEATURES = [
    "VO2_Max",
    "Speed_Index",
    "Endurance_Hours",
    "Training_Load",
    "Sleep_Hours",
    "HR_Variability",
    "Step_Count",
    "Injury_History",
    "Risky_Moves",
]

# ---------------- SPORT-SPECIFIC ВАГИ ----------------
# Кожен спорт має свою "формулу" важливості ознак
DEFAULT_WEIGHTS = {
    "VO2_Max": 0.25,
    "Speed_Index": 0.25,
    "Endurance_Hours": 0.15,
    "Training_Load": 0.10,
    "Sleep_Hours": 0.10,
    "HR_Variability": 0.10,
    "Step_Count": 0.05,
    "Injury_History": -0.03,
    "Risky_Moves": -0.02,
}

SPORT_WEIGHTS = {
    # Багато бігу, швидкість + аеробка + навантаження
    "Football": {
        "VO2_Max": 0.25,
        "Speed_Index": 0.20,
        "Endurance_Hours": 0.20,
        "Training_Load": 0.15,
        "Sleep_Hours": 0.05,
        "HR_Variability": 0.10,
        "Step_Count": 0.05,
        "Injury_History": -0.05,
        "Risky_Moves": -0.05,
    },
    # Вибухова швидкість, стрибки, навантаження
    "Basketball": {
        "VO2_Max": 0.20,
        "Speed_Index": 0.30,
        "Endurance_Hours": 0.10,
        "Training_Load": 0.15,
        "Sleep_Hours": 0.10,
        "HR_Variability": 0.10,
        "Step_Count": 0.05,
        "Injury_History": -0.05,
        "Risky_Moves": -0.05,
    },
    # Ракетки: координація, швидка зміна напрямку, нервова система
    "Tennis": {
        "VO2_Max": 0.20,
        "Speed_Index": 0.25,
        "Endurance_Hours": 0.10,
        "Training_Load": 0.10,
        "Sleep_Hours": 0.10,
        "HR_Variability": 0.15,
        "Step_Count": 0.05,
        "Injury_History": -0.03,
        "Risky_Moves": -0.02,
    },
    # Плавання: аеробка + витривалість
    "Swimming": {
        "VO2_Max": 0.30,
        "Speed_Index": 0.20,
        "Endurance_Hours": 0.20,
        "Training_Load": 0.10,
        "Sleep_Hours": 0.10,
        "HR_Variability": 0.10,
        "Step_Count": 0.0,
        "Injury_History": -0.05,
        "Risky_Moves": -0.05,
    },
    # Легка атлетика: максимальний акцент на швидкість
    "Track": {
        "VO2_Max": 0.20,
        "Speed_Index": 0.40,
        "Endurance_Hours": 0.15,
        "Training_Load": 0.10,
        "Sleep_Hours": 0.05,
        "HR_Variability": 0.10,
        "Step_Count": 0.0,
        "Injury_History": -0.05,
        "Risky_Moves": -0.05,
    },
    # Волейбол: вибуховість, координація, навантаження
    "Volleyball": {
        "VO2_Max": 0.20,
        "Speed_Index": 0.25,
        "Endurance_Hours": 0.10,
        "Training_Load": 0.15,
        "Sleep_Hours": 0.10,
        "HR_Variability": 0.10,
        "Step_Count": 0.05,
        "Injury_History": -0.05,
        "Risky_Moves": -0.05,
    },
    # Все інше – загальна формула
    "Other": DEFAULT_WEIGHTS,
}

# Квантілі для класів (Low / Medium / High)
LOW_Q = 0.25
HIGH_Q = 0.75


def _get_weights_for_sport(sport: str) -> dict:
    """
    Повертає словник ваг для конкретного виду спорту.
    Якщо спорт не відомий — використовуємо DEFAULT_WEIGHTS.
    """
    return SPORT_WEIGHTS.get(sport, DEFAULT_WEIGHTS)


def add_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає колонку performance_score на основі sport-specific формули.

    Кроки:
    1) перевіряємо, що всі потрібні фічі є в датасеті
    2) рахуємо z-score по кожній фічі (на рівні всього df)
    3) для кожного рядка:
       - беремо sport
       - беремо ваги для цього sport
       - рахуємо зважену суму z-score по фічах
    """
    missing = [c for c in PERF_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for performance_score: {missing}")

    if "Sport" not in df.columns:
        raise ValueError("Column 'Sport' is required for sport-specific performance_score.")

    df = df.copy()

    # 1. Z-score по всіх фічах
    z_cols = []
    for col in PERF_FEATURES:
        mean = df[col].mean()
        std = df[col].std()
        z_col = f"z_{col}"
        if std == 0 or pd.isna(std):
            df[z_col] = 0.0
        else:
            df[z_col] = (df[col] - mean) / std
        z_cols.append(z_col)

    # 2. Зважена сума за конкретним видом спорту
    def _row_score(row: pd.Series) -> float:
        sport = row.get("Sport", "Other")
        weights = _get_weights_for_sport(str(sport))

        score = 0.0
        for feat, w in weights.items():
            z_col = f"z_{feat}"
            if z_col in row:
                score += w * row[z_col]
        return float(score)

    df[PERFORMANCE_SCORE_COL] = df.apply(_row_score, axis=1)

    return df


def add_performance_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає колонку performance_class з трьома класами:
      - Low    (нижче 25 перцентиля)
      - Medium (25–75 перцентиль)
      - High   (вище 75 перцентиля)
    """
    if PERFORMANCE_SCORE_COL not in df.columns:
        raise ValueError(
            f"Column '{PERFORMANCE_SCORE_COL}' not found. "
            f"Call add_performance_score() first."
        )

    df = df.copy()
    q_low, q_high = df[PERFORMANCE_SCORE_COL].quantile([LOW_Q, HIGH_Q])

    def _label(score: float) -> str:
        if score < q_low:
            return "Low"
        elif score < q_high:
            return "Medium"
        else:
            return "High"

    df[PERFORMANCE_CLASS_COL] = df[PERFORMANCE_SCORE_COL].apply(_label)
    return df
