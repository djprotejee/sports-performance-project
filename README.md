# Sports Performance Project

End-to-end ML-застосунок для прогнозування спортивної продуктивності атлетів
на основі тренувальних, фізіологічних та поведінкових показників.

## Структура

- `data/raw/College_Sports_Dataset.csv` — вихідний датасет
- `src/` — код (дані, фічі, моделі, API, UI, БД)
- `artifacts/` — збережені моделі та метрики
- `alembic/` — міграції бази даних (PostgreSQL)
- `docker/` — Dockerfile'и для API / UI / pgAdmin

## Основні компоненти

- **ML-моделі**:
  - регресія: `performance_score` (штучна метрика продуктивності)
  - класифікація: рівень `Low / Medium / High`
- **REST API (Flask)**:
  - `GET /health` — перевірка стану
  - `POST /predict` — прогноз за параметрами атлета
  - `GET /history` — історія останніх прогнозів
  - `GET /models` — метрики моделей
- **UI (Streamlit)**:
  - вкладка **Predict** — введення параметрів і прогноз
  - вкладка **Dataset** — огляд датасету та базова візуалізація
  - вкладка **Model** — метрики регресії/класифікації
  - вкладка **History** — історія прогнозів з БД

## Запуск локально (без Docker)

1. Створити віртуальне середовище та встановити залежності:

   ```bash
   pip install -r requirements.txt
Налаштувати Postgres (локально або через Docker), виставити DATABASE_URL.

Застосувати міграції:

bash
Копіювати код
alembic upgrade head
Натренувати моделі:

bash
Копіювати код
python -m src.models.train
Запустити API:

bash
Копіювати код
python -m src.api.app
Запустити UI:

bash
Копіювати код
streamlit run src/ui/app.py
Запуск через Docker Compose
Створити файл .env (можна взяти з .env шаблону в репозиторії).

Натренувати моделі локально:

bash
Копіювати код
python -m src.models.train
Підняти сервіси:

bash
Копіювати код
make up
Відкрити:

API: http://localhost:8000/health

UI: http://localhost:8501

pgAdmin: http://localhost:5050

yaml
Копіювати код

---