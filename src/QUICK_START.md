# Quick Start Guide - Corners Prediction Model

## What I Created

I've implemented a complete machine learning pipeline for predicting the number of corners in football matches.

### Files Created:

1. **`models.py`** - Main module with all ML functions
2. **`train_model.py`** - Script to train and compare models  
3. **`predict.py`** - Script to make predictions
4. **`MODELS_README.md`** - Detailed documentation

## How to Use

### Step 1: Train the Model

```bash
cd src
python train_model.py
```

This will:
- Load your data from `../data/output/`
- Split into train (70%), validation (10%), test (20%)
- Train 8 different models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
- Compare them and select the best one
- Save the best model to `../data/models/`

### Step 2: Make Predictions

```bash
cd src
python predict.py
```

Or in your own code:

```python
from models import load_model, predict_corners
import pandas as pd

# Load model
model_data = load_model("../data/models")

# Prepare features
features = pd.DataFrame([{
    'avg_goals': 1.5,
    'shots_on_goal': 5.2,
    'pass_accuracy': 82.0,
    'possession': 55.0,
    'yellow_cards': 2.0,
    'red_cards': 0.1,
    'corners': 5.5,
    'fouls': 12.0,
    'num_matches': 5,
    'streak_score': 2.1,
    'home_bin_encoded': 1
}])

# Predict
prediction = predict_corners(
    model_data['model'], 
    model_data['preprocessors'], 
    features
)
```

## What the Model Does

### Input Features:
- Team statistics (goals, shots, possession, etc.)
- Recent form (streak of W/D/L)
- Home advantage
- Historical corners average

### Output:
- Predicted number of corners for that team

### Evaluation Metrics:
- **MAE** (Mean Absolute Error) - Average error in corners
- **RMSE** (Root Mean Squared Error)
- **R²** - Variance explained
- **Within 1 Corner %** - How often prediction is ±1 corner

## Model Selection

The pipeline automatically selects the best model based on validation MAE. Typically XGBoost or LightGBM perform best for this type of problem.

## Output Files

After training, you'll find in `../data/models/`:
- `corners_prediction_<ModelName>_<Timestamp>.pkl` - Trained model
- `preprocessors_<Timestamp>.pkl` - Feature scaler
- `model_metadata_<Timestamp>.json` - Performance metrics
- `model_comparison.csv` - Comparison of all 8 models

## Notes

- Make sure all required packages are installed: `pip install -r requirements.txt`
- The model handles missing values automatically
- Recent matches are weighted more heavily in the streak score
- You can retrain anytime with new data
