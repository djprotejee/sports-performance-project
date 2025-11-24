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

## Запуск локально

1. Створити віртуальне середовище та встановити залежності:

   ```
   pip install -r requirements.txt
   ```
2. Налаштувати Postgres (локально або через Docker), виставити DATABASE_URL.

3. Застосувати міграції:
```
alembic upgrade head
``` 

4. Натренувати моделі:

```
 python -m src.models.train
```

5. Запустити API:
```
python -m src.api.app
``` 

6. Запустити UI:
```
streamlit run src/ui/app.py
``` 

## Запуск через Docker Compose
1. Створити файл .env (можна взяти з .env шаблону в репозиторії).

2. Натренувати моделі локально:
```
make train
```
3. Підняти сервіси:
```
make up
```
4. Відкрити:

- API: http://localhost:8000/health

- UI: http://localhost:8501

- pgAdmin: http://localhost:5050