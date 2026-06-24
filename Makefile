.PHONY: help install train generate api dashboard stream test lint format \
        docker-build docker-train docker-up monitoring clean

help:
	@echo "Targets:"
	@echo "  install       Install runtime + dev dependencies"
	@echo "  train         Train ETA + delay models (hybrid source)"
	@echo "  generate      Write a synthetic dataset to data/processed"
	@echo "  api           Run the FastAPI service on :8000"
	@echo "  dashboard     Run the Streamlit dashboard on :8501"
	@echo "  stream        Stream synthetic orders at the running API"
	@echo "  test          Run the pytest suite"
	@echo "  lint          ruff check"
	@echo "  format        black + ruff --fix"
	@echo "  docker-train  Train inside the container into ./models"
	@echo "  docker-up     Build + start API and dashboard"
	@echo "  monitoring    Start API, dashboard, Prometheus and Grafana"
	@echo "  clean         Remove caches and generated artifacts"

install:
	pip install -r requirements-dev.txt

train:
	python scripts/train.py --source hybrid

generate:
	python scripts/generate_data.py --format csv

api:
	uvicorn delivery_delay.api.main:app --reload --app-dir src --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

stream:
	python scripts/stream_simulator.py --rate 2 --n 40

test:
	pytest

lint:
	ruff check src tests scripts dashboard

format:
	black src tests scripts dashboard
	ruff check --fix src tests scripts dashboard

docker-build:
	docker compose build

docker-train:
	docker compose run --rm trainer

docker-up:
	docker compose up --build api dashboard

monitoring:
	docker compose --profile monitoring up --build

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage mlruns
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
