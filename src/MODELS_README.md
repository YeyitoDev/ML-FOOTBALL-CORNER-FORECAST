# Football Corners Prediction Model

This module provides a complete pipeline for training and using machine learning models to predict the number of corners in football matches.

## Overview

The pipeline includes:
- Data loading and preprocessing
- Feature engineering (streak encoding, home advantage)
- Train/validation/test split
- Multiple model training (8 different algorithms)
- Model comparison and selection
- Model persistence (saving/loading)
- Prediction capabilities

## Models Implemented

The pipeline trains and compares the following models:
1. **Linear Regression** - Simple baseline
2. **Ridge Regression** - L2 regularized linear model
3. **Lasso Regression** - L1 regularized linear model
4. **Decision Tree** - Non-linear tree-based model
5. **Random Forest** - Ensemble of decision trees
6. **Gradient Boosting** - Sequential boosting ensemble
7. **XGBoost** - Optimized gradient boosting
8. **LightGBM** - Fast gradient boosting framework

The best model is automatically selected based on Mean Absolute Error (MAE) on the validation set.

## Files

- `models.py` - Core modeling functions and pipeline
- `train_model.py` - Script to train models
- `predict.py` - Script to make predictions with trained models

## Installation

All required packages are in `requirements.txt`. Key dependencies:
```bash
pip install scikit-learn xgboost lightgbm pandas numpy
```

## Usage

### 1. Train Models

```bash
cd src
python train_model.py
```

This will:
- Load features from `../data/output/HISTORIC_2025-01-01_to_2025-11-18.csv`
- Load targets from `../data/output/TARGET_CORNERS_2025-10-01_to_2025-11-18.csv`
- Train 8 different models
- Compare them and select the best
- Save the best model to `../data/models/`
- Generate a comparison report

### 2. Make Predictions

```python
from models import load_model, predict_corners
import pandas as pd

# Load the trained model
model_data = load_model("../data/models")

# Prepare features for a team
team_features = pd.DataFrame([{
    'avg_goals': 1.5,
    'shots_on_goal': 5.2,
    'pass_accuracy': 82.0,
    'possession': 55.0,
    'yellow_cards': 2.0,
    'red_cards': 0.1,
    'corners_features': 5.5,
    'fouls': 12.0,
    'num_matches': 5,
    'streak_score': 2.1,  # Pre-calculated
    'home_bin_encoded': 1
}])

# Predict
prediction = predict_corners(
    model_data['model'], 
    model_data['preprocessors'], 
    team_features
)

print(f"Predicted corners: {prediction[0]:.1f}")
```

Or run the example script:
```bash
python predict.py
```

## Features Used

### Input Features
- `avg_goals` - Average goals scored in recent matches
- `shots_on_goal` - Average shots on target
- `pass_accuracy` - Pass accuracy percentage
- `possession` - Ball possession percentage
- `yellow_cards` - Average yellow cards per match
- `red_cards` - Average red cards per match
- `corners_features` - Average corners in recent matches
- `fouls` - Average fouls committed
- `num_matches` - Number of historical matches available
- `streak` - Recent match results (W/D/L string)
- `home_bin` - Whether playing at home (True/False)

### Engineered Features
- `streak_score` - Weighted score from recent results (recent matches weighted more)
- `home_bin_encoded` - Binary encoding of home advantage

## Model Output

The trained model is saved as:
- `corners_prediction_<ModelName>_<Timestamp>.pkl` - The model
- `preprocessors_<Timestamp>.pkl` - Scaler and feature info
- `model_metadata_<Timestamp>.json` - Metrics and metadata
- `model_comparison.csv` - Comparison of all models

## Evaluation Metrics

Models are evaluated using:
- **MAE (Mean Absolute Error)** - Average prediction error in corners
- **RMSE (Root Mean Squared Error)** - Penalizes larger errors more
- **R² Score** - Proportion of variance explained
- **Within 1 Corner %** - Percentage of predictions within ±1 corner

## Data Split

- **Training**: 70% of data
- **Validation**: 10% of data (for model selection)
- **Test**: 20% of data (for final evaluation)

## Example Output

```
📊 MODEL COMPARISON (Validation Set)
================================================================================
              Model     MAE    RMSE      R²  Within 1 Corner %
         XGBoost  1.234   1.678   0.456              67.8
        LightGBM  1.256   1.701   0.443              66.2
  Random Forest  1.289   1.734   0.421              64.5
...

🏆 Best Model: XGBoost
   MAE: 1.234 corners

Test Set Performance:
  MAE:  1.241 corners
  RMSE: 1.682 corners
  R²:   0.448
  Predictions within 1 corner: 67.3%
```

## Notes

- The model automatically handles missing values by filling with 0
- Streak encoding gives more weight to recent matches
- Models are saved with timestamps for versioning
- The latest model is automatically loaded if no timestamp specified
