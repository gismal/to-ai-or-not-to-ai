PY ?= python3
.DEFAULT_GOAL := help
.PHONY: help build up down rebuild logs ps shell psql migrate test lint format typecheck check train quantize mlflow clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build and start the whole stack (foreground)
	docker compose up --build

up: ## Start the stack in the background
	docker compose up -d

down: ## Stop the stack (keeps data)
	docker compose down

rebuild: ## Rebuild images from scratch
	docker compose build --no-cache api worker

logs: ## Follow API and worker logs
	docker compose logs -f api worker

ps: ## Show service status
	docker compose ps

shell: ## Shell inside the API container
	docker compose exec api sh

psql: ## Open psql on the local database
	docker compose exec db sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

migrate: ## Apply database migrations inside the API container
	docker compose exec api alembic upgrade head

test: ## Run the test suite with coverage
	PYTHONPATH=. $(PY) -m pytest -v --tb=short --cov=src

lint: ## Ruff lint
	ruff check src/ tests/

format: ## Ruff format
	ruff format src/ tests/

typecheck: ## mypy
	mypy src/

check: lint typecheck test ## Everything CI runs

train: ## Train a model (see scripts/train_config.py)
	PYTHONPATH=. $(PY) scripts/train.py

quantize: ## INT8-quantize the ONNX model
	PYTHONPATH=. $(PY) scripts/quantize.py

mlflow: ## Start the MLflow tracking server
	docker compose up mlflow

clean: ## Remove containers AND volumes (deletes local database data)
	@read -p "Delete all local containers and data? [y/N] " a; [ "$$a" = "y" ]
	docker compose down -v --remove-orphans