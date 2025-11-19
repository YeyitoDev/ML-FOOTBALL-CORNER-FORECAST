.PHONY: help setup run train docker-build docker-up docker-down clean

help:
	@echo "⚽ Football Prediction Model - Available Commands"
	@echo "=================================================="
	@echo "make setup        - Create venv and install dependencies"
	@echo "make run          - Run data processing pipeline"
	@echo "make train        - Train ML models"
	@echo "make all          - Run complete pipeline (setup + run + train)"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-up    - Run with Docker Compose"
	@echo "make docker-down  - Stop Docker containers"
	@echo "make clean        - Remove venv and generated files"
	@echo "=================================================="

setup:
	@echo "📦 Creating virtual environment..."
	python3 -m venv venv
	@echo "📚 Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "📁 Creating necessary directories..."
	mkdir -p data/models data/output data/predictions
	@echo "✅ Setup complete!"

run:
	@echo "🔄 Running data processing pipeline..."
	./venv/bin/python src/main.py
	@echo "✅ Data processing complete!"

train: run
	@echo "🤖 Training ML models..."
	./venv/bin/python src/train_model.py
	@echo "✅ Model training complete!"

evaluate: run
	@echo "🤖 Evaluating ML models..."
	./venv/bin/python src/evaluate_model.py
	@echo "✅ Model evaluation complete!"

all: setup run train evaluate
	@echo "✅ Complete pipeline finished!"

docker-build:
	@echo "🐳 Building Docker image..."
	docker-compose build
	@echo "✅ Docker image built!"

docker-up:
	@echo "🐳 Starting Docker containers..."
	docker-compose up --build
	@echo "✅ Docker containers running!"

docker-down:
	@echo "🛑 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Docker containers stopped!"

clean:
	@echo "🧹 Cleaning up..."
	rm -rf venv
	rm -rf src/__pycache__
	rm -rf pipelines/__pycache__
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cleanup complete!"
