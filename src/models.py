"""
Football Corners Prediction Models
This module contains functions to train, evaluate, and compare different regression models
for predicting the number of corners in football matches.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb


def load_and_prepare_data(features_path: str, target_path: str):
    """
    Load features and target data, merge them, and prepare for modeling
    
    Args:
        features_path: Path to the features CSV file
        target_path: Path to the target CSV file
    
    Returns:
        Merged DataFrame ready for modeling
    """
    # Load data
    features = pd.read_csv(features_path)
    target = pd.read_csv(target_path)
    
    # Check if fixture_id exists in features
    if 'fixture_id' in features.columns and 'fixture_id' in target.columns:
        # Merge on both team_id and fixture_id
        df = features.merge(
            target[['team_id', 'fixture_id', 'corners']], 
            on=['team_id', 'fixture_id'], 
            how='inner',
            suffixes=('_features', '_target')
        )
    else:
        # Merge only on team_id (less precise but works)
        print("⚠️  Warning: fixture_id not found, merging only on team_id")
        df = features.merge(
            target[['team_id', 'corners']], 
            on='team_id', 
            how='inner',
            suffixes=('_features', '_target')
        )
    
    # Remove rows with missing values in critical columns
    target_col = 'corners_target' if 'corners_target' in df.columns else 'corners'
    df = df.dropna(subset=['avg_goals', 'shots_on_goal', 'pass_accuracy', 
                           'possession', target_col])
    
    # Rename target column for consistency
    if 'corners_target' in df.columns:
        df.rename(columns={'corners_target': 'target_corners'}, inplace=True)
    elif 'corners' in df.columns and 'corners_features' not in df.columns:
        df.rename(columns={'corners': 'target_corners'}, inplace=True)
    
    # Rename features corners column
    if 'corners_features' in df.columns:
        df.rename(columns={'corners_features': 'corners'}, inplace=True)
    
    # Remove rows where num_matches is 0 (no historical data)
    if 'num_matches' in df.columns:
        df = df[df['num_matches'] > 0]
    
    print(f"✅ Loaded {len(df)} samples after cleaning")
    print(f"Features shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def preprocess_features(df: pd.DataFrame):
    """
    Preprocess features for modeling
    
    Args:
        df: Input DataFrame
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        preprocessors: Dictionary with scaler and encoder
    """
    # Define feature columns
    numerical_features = [
        'avg_goals', 'shots_on_goal', 'pass_accuracy', 'possession',
        'yellow_cards', 'red_cards', 'fouls', 'num_matches'
    ]
    
    # Add corners if it exists (historical average corners)
    if 'corners' in df.columns:
        numerical_features.append('corners')
    
    # Encode streak (W=3, D=1, L=0) - get recency weighted score
    def encode_streak(streak):
        if pd.isna(streak) or streak == '':
            return 0
        weights = [0.4, 0.3, 0.2, 0.1, 0.05]  # Recent matches have more weight
        score = 0
        for i, result in enumerate(str(streak)[:5]):  # Take last 5 matches
            if i >= len(weights):
                break
            if result == 'W':
                score += 3 * weights[i]
            elif result == 'D':
                score += 1 * weights[i]
        return score
    
    df['streak_score'] = df['streak'].apply(encode_streak) if 'streak' in df.columns else 0
    
    # Add binary home advantage
    df['home_bin_encoded'] = df['home_bin'].astype(int) if 'home_bin' in df.columns else 0
    
    # All features including encoded ones
    all_features = numerical_features + ['streak_score', 'home_bin_encoded']
    
    # Prepare X and y
    X = df[all_features].copy()
    
    # Find target column
    if 'target_corners' in df.columns:
        y = df['target_corners'].values
    elif 'corners_target' in df.columns:
        y = df['corners_target'].values
    else:
        raise ValueError("No target column found. Expected 'target_corners' or 'corners_target'")
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=all_features, index=X.index)
    
    preprocessors = {
        'scaler': scaler,
        'feature_names': all_features
    }
    
    print(f"✅ Preprocessed features: {all_features}")
    print(f"X shape: {X_scaled.shape}, y shape: {y.shape}")
    print(f"Target (y) stats: min={y.min()}, max={y.max()}, mean={y.mean():.2f}")
    
    return X_scaled, y, all_features, preprocessors


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        val_size: Proportion for validation set (from train)
        random_state: Random seed
    
    Returns:
        Dictionary with train, val, test splits
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def evaluate_model(model, X, y, dataset_name=""):
    """
    Evaluate a model and return metrics
    
    Args:
        model: Trained model
        X: Features
        y: True targets
        dataset_name: Name of dataset (for display)
    
    Returns:
        Dictionary with metrics
    """
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Additional metric: percentage within 1 corner
    within_1 = np.mean(np.abs(y - y_pred) <= 1) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Within_1_Corner_%': within_1
    }
    
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.3f}")
        print(f"  Within 1 corner: {within_1:.1f}%")
    
    return metrics


def train_models(data_splits):
    """
    Train multiple regression models
    
    Args:
        data_splits: Dictionary with train/val/test splits
    
    Returns:
        Dictionary with trained models and their validation metrics
    """
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10, min_samples_split=20, random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    }
    
    results = {}
    
    print("\n🚀 Training models...")
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate on train and validation
        train_metrics = evaluate_model(model, X_train, y_train, "Train")
        val_metrics = evaluate_model(model, X_val, y_val, "Validation")
        
        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    
    return results


def compare_models(results):
    """
    Compare models and select the best one
    
    Args:
        results: Dictionary with model results
    
    Returns:
        Name of best model and comparison DataFrame
    """
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON (Validation Set)")
    print("="*80)
    
    comparison = []
    for name, result in results.items():
        val_metrics = result['val_metrics']
        comparison.append({
            'Model': name,
            'MAE': val_metrics['MAE'],
            'RMSE': val_metrics['RMSE'],
            'R²': val_metrics['R2'],
            'Within 1 Corner %': val_metrics['Within_1_Corner_%']
        })
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('MAE')
    
    print(df_comparison.to_string(index=False))
    
    # Select best model based on MAE (most interpretable for this problem)
    best_model_name = df_comparison.iloc[0]['Model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   MAE: {df_comparison.iloc[0]['MAE']:.3f} corners")
    
    return best_model_name, df_comparison


def save_model(model, preprocessors, metrics, model_name, output_dir="../data/models"):
    """
    Save the trained model and associated metadata
    
    Args:
        model: Trained model
        preprocessors: Dictionary with scaler and feature names
        metrics: Model metrics
        model_name: Name of the model
        output_dir: Directory to save the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model as pickle
    model_filename = f"corners_prediction_{model_name.replace(' ', '_')}_{timestamp}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save preprocessors
    preprocessor_filename = f"preprocessors_{timestamp}.pkl"
    preprocessor_path = os.path.join(output_dir, preprocessor_filename)
    
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessors, f)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_names': preprocessors['feature_names'],
        'model_file': model_filename,
        'preprocessor_file': preprocessor_filename
    }
    
    metadata_filename = f"model_metadata_{timestamp}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Model saved:")
    print(f"   Model: {model_path}")
    print(f"   Preprocessors: {preprocessor_path}")
    print(f"   Metadata: {metadata_path}")
    
    return {
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'metadata_path': metadata_path
    }


def load_model(model_dir="../data/models", timestamp=None):
    """
    Load a saved model
    
    Args:
        model_dir: Directory containing the model
        timestamp: Specific timestamp to load (if None, loads latest)
    
    Returns:
        Dictionary with model, preprocessors, and metadata
    """
    if timestamp is None:
        # Find latest model
        metadata_files = [f for f in os.listdir(model_dir) if f.startswith('model_metadata_')]
        if not metadata_files:
            raise FileNotFoundError("No models found in directory")
        latest_metadata = sorted(metadata_files)[-1]
        metadata_path = os.path.join(model_dir, latest_metadata)
    else:
        metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model_path = os.path.join(model_dir, metadata['model_file'])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load preprocessors
    preprocessor_path = os.path.join(model_dir, metadata['preprocessor_file'])
    with open(preprocessor_path, 'rb') as f:
        preprocessors = pickle.load(f)
    
    print(f"✅ Loaded model: {metadata['model_name']}")
    print(f"   Timestamp: {metadata['timestamp']}")
    print(f"   Val MAE: {metadata['metrics']['val_metrics']['MAE']:.3f}")
    
    return {
        'model': model,
        'preprocessors': preprocessors,
        'metadata': metadata
    }


def predict_corners(model, preprocessors, team_features):
    """
    Predict corners for new team features
    
    Args:
        model: Trained model
        preprocessors: Preprocessors dictionary
        team_features: DataFrame or dict with team features
    
    Returns:
        Predicted number of corners
    """
    if isinstance(team_features, dict):
        team_features = pd.DataFrame([team_features])
    
    # Ensure all required features are present
    required_features = preprocessors['feature_names']
    
    # Scale features
    X = preprocessors['scaler'].transform(team_features[required_features])
    
    # Predict
    prediction = model.predict(X)
    
    return prediction


def run_full_pipeline(features_path, target_path, output_dir="../data/models"):
    """
    Run the complete modeling pipeline
    
    Args:
        features_path: Path to features CSV
        target_path: Path to target CSV
        output_dir: Directory to save models
    
    Returns:
        Dictionary with best model and results
    """
    print("FOOTBALL CORNERS PREDICTION - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # 1. Load and prepare data
    print("\n📁 Step 1: Loading data...")
    df = load_and_prepare_data(features_path, target_path)
    
    # 2. Preprocess features
    print("\n🔧 Step 2: Preprocessing features...")
    X, y, feature_names, preprocessors = preprocess_features(df)
    
    # 3. Split data
    print("\n✂️  Step 3: Splitting data...")
    data_splits = split_data(X, y)
    
    # 4. Train models
    print("\n🎯 Step 4: Training models...")
    results = train_models(data_splits)
    
    # 5. Compare models
    print("\n📈 Step 5: Comparing models...")
    best_model_name, comparison_df = compare_models(results)
    
    # 6. Evaluate best model on test set
    print("\n🧪 Step 6: Evaluating best model on test set...")
    best_model = results[best_model_name]['model']
    test_metrics = evaluate_model(
        best_model, 
        data_splits['X_test'], 
        data_splits['y_test'],
        "Test"
    )
    
    # 7. Save best model
    print("\n💾 Step 7: Saving best model...")
    model_info = save_model(
        best_model,
        preprocessors,
        {
            'train_metrics': results[best_model_name]['train_metrics'],
            'val_metrics': results[best_model_name]['val_metrics'],
            'test_metrics': test_metrics
        },
        best_model_name,
        output_dir
    )
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"   Comparison: {comparison_path}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'preprocessors': preprocessors,
        'test_metrics': test_metrics,
        'comparison': comparison_df,
        'model_info': model_info
    }


if __name__ == "__main__":
    # Example usage
    features_path = "../data/output/HISTORIC_2025-01-01_to_2025-11-18.csv"
    target_path = "../data/output/TARGET_CORNERS_2025-10-01_to_2025-11-18.csv"
    
    results = run_full_pipeline(features_path, target_path)
