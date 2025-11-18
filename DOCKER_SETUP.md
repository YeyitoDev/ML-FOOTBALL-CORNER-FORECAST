# Football Prediction Model - Docker & Setup Guide

## Quick Start Options

### Option 1: Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 2: Local Setup Script (Automated)

**Linux/Mac:**
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

**Windows:**
```cmd
setup_and_run.bat
```

### Option 3: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt

# Run data processing
cd src
python main.py

# Train models
python train_model.py
```

## What Each Method Does

All methods execute the complete pipeline:

1. **Setup Environment** - Creates virtual environment and installs dependencies
2. **Data Processing** - Runs `main.py` to generate training data
3. **Model Training** - Runs `train_model.py` to train and select best model
4. **Save Results** - Stores trained models in `data/models/`

## Docker Details

### Dockerfile
- Base image: Python 3.9 slim
- Installs all dependencies from `requirements.txt`
- Sets up working directory and environment
- Mounts data volumes for persistence

### docker-compose.yml
- Orchestrates the complete pipeline
- Mounts local data directories for persistence
- Runs main.py → train_model.py sequentially
- Environment variables can be configured

### Volume Mounts
```yaml
volumes:
  - ./data:/app/data      # Persists models and outputs
  - ./src:/app/src        # Allows live code updates
```

## Environment Variables

You can customize the pipeline by setting environment variables:

```bash
# In docker-compose.yml or .env file
START_DATE=2025-01-01
END_DATE=2025-11-18
```

## Output Files

After running, you'll find:

```
data/
├── output/
│   ├── HISTORIC_2025-01-01_to_2025-11-18.csv
│   └── TARGET_CORNERS_2025-01-01_to_2025-11-18.csv
├── models/
│   ├── corners_prediction_<ModelName>_<Timestamp>.pkl
│   ├── preprocessors_<Timestamp>.pkl
│   ├── model_metadata_<Timestamp>.json
│   └── model_comparison.csv
└── predictions/
    └── (future predictions will be saved here)
```

## Docker Commands Reference

```bash
# Build only
docker-compose build

# Run and rebuild
docker-compose up --build

# Run in background
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f football-prediction

# Execute command in running container
docker-compose exec football-prediction python src/predict.py

# Remove all containers and volumes
docker-compose down -v
```

## Troubleshooting

### Docker Issues

**Problem:** Port already in use
```bash
# Check what's using the port
docker ps
docker-compose down
```

**Problem:** Out of disk space
```bash
# Clean up Docker
docker system prune -a
```

### Virtual Environment Issues

**Problem:** Permission denied on setup script
```bash
chmod +x setup_and_run.sh
```

**Problem:** Module not found
```bash
# Ensure you're in virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### Data Issues

**Problem:** No data files
- Ensure `data/fixtures/` and `data/team_stats/` directories have data
- Check date ranges in `main.py`

## Development Workflow

1. **Make code changes** in `src/`
2. **Test locally** with virtual environment
3. **Build Docker image** when ready
4. **Run with Docker Compose** for production

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Train Model
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and run
        run: docker-compose up --build
      - name: Upload models
        uses: actions/upload-artifact@v2
        with:
          name: trained-models
          path: data/models/
```

## Notes

- Docker ensures consistent environment across all platforms
- Virtual environment isolates Python dependencies
- All data is persisted through volume mounts
- Models are automatically versioned with timestamps
