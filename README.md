# ⚽ Football Prediction Model - Complete Setup & Deployment Guide

This project provides an end-to-end machine learning pipeline for predicting football match statistics, specifically corners.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Setup Methods](#setup-methods)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Development](#development)

## 🚀 Quick Start

### Fastest Way - Using Makefile (Recommended)

```bash
# Run complete pipeline
make all

# Or step by step
make setup      # Create venv and install dependencies
make run        # Process data
make train      # Train models
```

### Using Docker

```bash
docker-compose up --build
```

### Using Setup Script

**Linux/Mac:**
```bash
./setup_and_run.sh
```

**Windows:**
```cmd
setup_and_run.bat
```

## 🛠️ Setup Methods

### Method 1: Makefile (Easiest)

```bash
make help    # See all available commands
make all     # Run everything
```

### Method 2: Automated Script

The scripts handle everything automatically:
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Create necessary directories
- ✅ Run data processing
- ✅ Train ML models

### Method 3: Docker (Production Ready)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

### Method 4: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
mkdir -p data/{models,output,predictions}

# 4. Run pipeline
cd src
python main.py        # Process data
python train_model.py # Train models
```

## 📁 Project Structure

```
FOOTBALL-PREDICTION-MODEL/
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker orchestration
├── Makefile                    # Build automation
├── setup_and_run.sh           # Linux/Mac setup script
├── setup_and_run.bat          # Windows setup script
├── requirements.txt           # Python dependencies
├── DOCKER_SETUP.md           # Docker documentation
│
├── src/
│   ├── main.py               # Data processing pipeline
│   ├── utils.py              # Utility functions
│   ├── models.py             # ML models implementation
│   ├── train_model.py        # Model training script
│   ├── predict.py            # Prediction script
│   ├── MODELS_README.md      # Model documentation
│   └── QUICK_START.md        # Quick reference
│
├── data/
│   ├── fixtures/             # Raw fixture data
│   ├── team_stats/           # Team statistics
│   ├── odds/                 # Betting odds
│   ├── output/               # Processed datasets
│   │   ├── HISTORIC_*.csv    # Training features
│   │   └── TARGET_*.csv      # Training targets
│   ├── models/               # Trained ML models
│   │   ├── *.pkl            # Model files
│   │   ├── *.json           # Metadata
│   │   └── model_comparison.csv
│   └── predictions/          # Future predictions
│
├── pipelines/                # Data pipeline notebooks
└── notebooks/                # Analysis notebooks
```

## 🎯 Usage

### Complete Pipeline

```bash
# Option 1: Makefile
make all

# Option 2: Script
./setup_and_run.sh

# Option 3: Docker
docker-compose up --build

# Option 4: Manual
python src/main.py && python src/train_model.py
```

### Individual Steps

```bash
# 1. Process data only
make run
# or
python src/main.py

# 2. Train models only
make train
# or
python src/train_model.py

# 3. Make predictions
python src/predict.py
```

## 🐳 Docker Deployment

### Basic Usage

```bash
# Build and run
docker-compose up --build

# Detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Advanced Docker Commands

```bash
# Execute command in container
docker-compose exec football-prediction python src/predict.py

# Shell access
docker-compose exec football-prediction /bin/bash

# Rebuild from scratch
docker-compose build --no-cache

# Remove everything
docker-compose down -v
```

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - START_DATE=2025-01-01
  - END_DATE=2025-11-18
```

## 💻 Development

### Setting Up Development Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install jupyter black flake8 pytest
```

### Making Changes

1. Edit code in `src/`
2. Test locally:
   ```bash
   python src/main.py
   python src/train_model.py
   ```
3. Test with Docker:
   ```bash
   docker-compose up --build
   ```

### Running Tests

```bash
# Add tests in tests/ directory
pytest tests/
```

## 📊 Output Files

### After running the pipeline:

**Data Processing (`main.py`):**
- `data/output/HISTORIC_*.csv` - Training features
- `data/output/TARGET_*.csv` - Training targets

**Model Training (`train_model.py`):**
- `data/models/corners_prediction_*.pkl` - Trained model
- `data/models/preprocessors_*.pkl` - Feature scaler
- `data/models/model_metadata_*.json` - Performance metrics
- `data/models/model_comparison.csv` - All models comparison

## 🔧 Makefile Commands

```bash
make help         # Show all commands
make setup        # Create venv and install deps
make run          # Run data processing
make train        # Train models
make all          # Complete pipeline
make docker-build # Build Docker image
make docker-up    # Run with Docker
make docker-down  # Stop Docker
make clean        # Clean up files
```

## 📈 Model Performance

The pipeline trains 8 different models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Best model is automatically selected based on validation MAE.

## 🚨 Troubleshooting

### Virtual Environment Issues

```bash
# Remove and recreate
rm -rf venv
make setup
```

### Docker Issues

```bash
# Clean up Docker
docker system prune -a
docker-compose down -v
docker-compose up --build
```

### Permission Issues (Linux/Mac)

```bash
chmod +x setup_and_run.sh
chmod +x Makefile
```

### Import Errors

```bash
# Ensure you're in venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📝 Requirements

- Python 3.9+
- Docker (optional, for containerized deployment)
- Make (optional, for Makefile usage)

## 🔗 Related Documentation

- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Detailed Docker guide
- [src/MODELS_README.md](src/MODELS_README.md) - ML models documentation
- [src/QUICK_START.md](src/QUICK_START.md) - Quick reference

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Make changes in a feature branch
2. Test locally and with Docker
3. Submit pull request

---

**Quick Commands Summary:**

```bash
# Fastest: Complete pipeline
make all

# Docker: Production deployment
docker-compose up --build

# Script: Automated setup
./setup_and_run.sh

# Manual: Step by step
python src/main.py && python src/train_model.py
```
