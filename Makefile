PYTHON := python

.PHONY: help train up down logs migrate revision psql

help:
	@echo "Available targets:"
	@echo "  train      - тренувати моделі (regressor + classifier)"
	@echo "  up         - підняти всі сервіси через docker-compose"
	@echo "  down       - зупинити всі сервіси"
	@echo "  logs       - показати логи api та ui"
	@echo "  migrate    - застосувати міграції Alembic (upgrade head)"
	@echo "  revision   - створити нову міграцію Alembic (autogenerate)"
	@echo "  psql       - зайти в psql всередині контейнера db"

train:
	$(PYTHON) -m src.models.train

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api ui

migrate:
	docker compose run api alembic upgrade head

revision:
	alembic revision --autogenerate -m "auto revision"

psql:
	docker exec -it sports_db psql -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-sports}
