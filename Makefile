build:
    docker compose up --build

test:
    PYTHONPATH=. pytest -v --tb=short --cov=src

migrate:
	alembic upgrade head

lint:
	ruff check src/ tests/