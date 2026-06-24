build:
	docker compose up --build

test:
	PYTHONPATH=. pytest -v --tb=short --cov=src

migrate:
	alembic upgrade head

lint:
	ruff check src/ tests/

train:
	PYTHONPATH=. venv/bin/python scripts/train.py

quantize:
	PYTHONPATH=. python scripts/quantize.py

calibrate:
	PYTHONPATH=. python scripts/calibrate.py

mlflow:
	docker compose up mlflow