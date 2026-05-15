.PHONY: install test lint run docker-build docker-up clean

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio httpx fastapi uvicorn scipy joblib ruff

test:
	PYTHONPATH=. pytest tests/ -v --tb=short --cov=insurance_predictor --cov=app --cov-report=term-missing

lint:
	ruff check . --select E,F,W,I --ignore E501
	ruff check . --select E,F,W,I --ignore E501 --fix

run:
	PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t insurance-predictor:latest .

docker-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml dist build *.egg-info
