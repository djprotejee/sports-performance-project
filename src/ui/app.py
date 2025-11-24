import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load import load_raw_data
from src.features.performance_score import (
    add_performance_score,
    add_performance_class,
    PERFORMANCE_SCORE_COL,
    PERFORMANCE_CLASS_COL,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "artifacts" / "reports" / "plots"


# ---------- API HELPERS ----------
def call_api(method: str, endpoint: str, **kwargs):
    url = f"{API_URL}{endpoint}"
    try:
        resp = requests.request(method, url, timeout=10, **kwargs)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text}")
        return None
    except Exception as e:
        st.error(f"Failed to call API: {e}")
        return None


def api_predict(payload: Dict[str, Any]):
    return call_api("POST", "/predict", json=payload)


def api_models():
    return call_api("GET", "/models")


def api_history(limit=50):
    return call_api("GET", "/history", params={"limit": limit})


# ---------- PAGE: PREDICT ----------
def page_predict():
    st.header("🏋️ Athlete Performance Prediction")

    st.markdown("Введи параметри атлета і отримай прогноз результативності.")

    with st.expander("📌 Ввести дані атлета"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 16, 60, 22)
            gender = st.selectbox("Gender", ["M", "F"])
            sport = st.selectbox(
                "Sport",
                [
                    "Basketball",
                    "Football",
                    "Tennis",
                    "Swimming",
                    "Volleyball",
                    "Track",
                    "Other",
                ],
            )
            sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, 7.0, 0.1)
            recovery_time = st.slider(
                "Recovery Time (hours)", 12.0, 72.0, 24.0, 0.5
            )

        with col2:
            hrv = st.slider("HR Variability", 30.0, 100.0, 60.0, 0.5)
            vo2 = st.slider("VO2 Max", 30.0, 80.0, 50.0, 0.5)
            speed_index = st.slider("Speed Index", 3.0, 10.0, 6.5, 0.1)
            endurance_hours = st.slider(
                "Endurance Hours per week", 1.0, 10.0, 4.0, 0.5
            )
            training_load = st.slider(
                "Training Load", 50, 600, 300, 10
            )
            risky_moves = st.slider("Risky Moves", 0, 10, 2)
            injury_history = st.slider("Injury History", 0, 10, 1)
            step_count = st.slider(
                "Daily Step Count", 2000, 25000, 12000, 500
            )
            feedback_level = st.slider(
                "Feedback Level (1–5)", 1, 5, 4
            )

    payload = {
        "Age": age,
        "Gender": gender,
        "Sport": sport,
        "HR_Variability": hrv,
        "VO2_Max": vo2,
        "Speed_Index": speed_index,
        "Endurance_Hours": endurance_hours,
        "Risky_Moves": risky_moves,
        "Sleep_Hours": sleep_hours,
        "Step_Count": step_count,
        "Injury_History": injury_history,
        "Training_Load": training_load,
        "Recovery_Time": recovery_time,
        "Feedback_Level": feedback_level,
    }

    if st.button("🔮 Predict"):
        result = api_predict(payload)
        if result:
            st.subheader("🎯 Prediction")
            st.success(f"**Performance Class:** {result['performance_class']}")
            st.info(f"Performance Score: `{result['performance_score']:.4f}`")

            probs = result["class_probabilities"]
            col_plot, _ = st.columns([1, 3])  # перша колонка вузька

            with col_plot:
                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                ax.bar(list(probs.keys()), list(probs.values()))
                ax.set_title("Class Probabilities", fontsize=9)
                ax.tick_params(labelsize=7)
                st.pyplot(fig, use_container_width=False)


# ---------- PAGE: DATASET ----------
def page_dataset():
    st.header("📊 Dataset Overview")

    df = load_raw_data()
    df = add_performance_score(df)
    df = add_performance_class(df)

    st.subheader("🔍 Preview")
    st.dataframe(df.head())

    st.subheader("📈 Basic Statistics")
    st.write(df.describe(include="all"))

    col1, col2 = st.columns(2)

    with col1:
        st.write("Performance Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df[PERFORMANCE_SCORE_COL], bins=30)
        st.pyplot(fig)

    with col2:
        st.write("Class Distribution")
        class_counts = df[PERFORMANCE_CLASS_COL].value_counts()
        fig, ax = plt.subplots()
        ax.bar(class_counts.index, class_counts.values)
        st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Note")
    st.markdown(
        """
        **performance_score** уже доданий до датасету, він залежить від виду спорту та
        основних фізичних, тренувальних і ризикових показників.
        """
    )


# ---------- PAGE: MODEL ----------
def page_model():
    st.header("🤖 Model Performance")

    data = api_models()
    if not data:
        st.error("Cannot load model info.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regression Metrics")
        st.json(data.get("regression"))

        reg_imp_path = PLOTS_DIR / "regressor_feature_importance.png"
        reg_shap_path = PLOTS_DIR / "regressor_shap_summary.png"

        if reg_imp_path.exists():
            st.markdown("**Regressor Feature Importance**")
            st.image(str(reg_imp_path))

        if reg_shap_path.exists():
            st.markdown("**Regressor SHAP Summary**")
            st.image(str(reg_shap_path))

    with col2:
        st.subheader("Classification Metrics")
        st.json(data.get("classification"))

        cls_imp_path = PLOTS_DIR / "classifier_feature_importance.png"
        cls_shap_path = PLOTS_DIR / "classifier_shap_summary.png"

        if cls_imp_path.exists():
            st.markdown("**Classifier Feature Importance**")
            st.image(str(cls_imp_path))

        if cls_shap_path.exists():
            st.markdown("**Classifier SHAP Summary**")
            st.image(str(cls_shap_path))

    st.subheader("📐 How performance_score is calculated")

    with st.expander("Show formula and sport-specific weights"):
        st.markdown(
            """
        Ми будуємо **performance_score** як зважену суму z-score по основних спортивних показниках.

        Загальна ідея:
        - нормалізуємо кожну ознаку через **z-score** (віднімаємо середнє, ділимо на стандартне відхилення)
        - для кожного виду спорту беремо **свої коефіцієнти (ваги)** для:
          - VO2_Max (аеробна витривалість)
          - Speed_Index (швидкісні якості)
          - Endurance_Hours (обсяг витривалих тренувань)
          - Training_Load (загальне навантаження)
          - Sleep_Hours (сон та відновлення)
          - HR_Variability (стан нервової системи / recovery)
          - Step_Count (загальна активність)
          - Injury_History (мінусова вага — більше травм, гірше результат)
          - Risky_Moves (мінусова вага — ризикований стиль)

        Для кожного спорту використовується свій набір ваг, наприклад:

        **Football**:
        - 0.25 · Z(VO2_Max)
        - 0.20 · Z(Speed_Index)
        - 0.20 · Z(Endurance_Hours)
        - 0.15 · Z(Training_Load)
        - 0.10 · Z(HR_Variability)
        - 0.05 · Z(Step_Count)
        - 0.05 · Z(Sleep_Hours)
        - (-0.05) · Z(Injury_History)
        - (-0.05) · Z(Risky_Moves)

        **Basketball**:
        - 0.20 · Z(VO2_Max)
        - 0.30 · Z(Speed_Index)
        - 0.10 · Z(Endurance_Hours)
        - 0.15 · Z(Training_Load)
        - 0.10 · Z(Sleep_Hours)
        - 0.10 · Z(HR_Variability)
        - 0.05 · Z(Step_Count)
        - (-0.05) · Z(Injury_History)
        - (-0.05) · Z(Risky_Moves)

        **Track**:
        - 0.20 · Z(VO2_Max)
        - 0.40 · Z(Speed_Index)
        - 0.15 · Z(Endurance_Hours)
        - 0.10 · Z(Training_Load)
        - 0.10 · Z(HR_Variability)
        - 0.05 · Z(Sleep_Hours)
        - (-0.05) · Z(Injury_History)
        - (-0.05) · Z(Risky_Moves)

        Для інших видів спорту використовується базова формула з більш збалансованими вагами.

        Після цього ми беремо **25-й і 75-й перцентилі** performance_score:
        - нижче 25% → **Low**
        - 25–75% → **Medium**
        - вище 75% → **High**
        """
        )


# ---------- PAGE: HISTORY ----------
def page_history():
    st.header("🗂 Prediction History")

    limit = st.slider("Number of records", 10, 200, 50, 10)
    records = api_history(limit)
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df)
    else:
        st.info("No records yet.")


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="Sports Performance", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "",
        ("Predict", "Dataset", "Model", "History")
    )

    if page == "Predict":
        page_predict()
    elif page == "Dataset":
        page_dataset()
    elif page == "Model":
        page_model()
    elif page == "History":
        page_history()


if __name__ == "__main__":
    main()
