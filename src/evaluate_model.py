"""
Evaluate the Lasso Regression model for corners prediction
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(__file__))

import models


def load_lasso_model(model_dir="../data/models", model_name="corners_prediction_Lasso_Regression.pkl"):
    """
    Load the Lasso Regression model directly by name
    
    Args:
        model_dir: Directory containing the model
        model_name: Name of the model pickle file
    
    Returns:
        Dictionary with model and preprocessors
    """
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Loaded Lasso Regression model from: {model_name}")
    
    # Find corresponding preprocessor file
    # Try to find the latest metadata file that corresponds to this model
    metadata_files = sorted([f for f in os.listdir(model_dir) if f.startswith('model_metadata_')])
    
    preprocessor_path = None
    metadata = None
    
    # Try to find the matching preprocessor
    for metadata_file in reversed(metadata_files):  # Start from newest
        metadata_path = os.path.join(model_dir, metadata_file)
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            if meta['model_file'] == model_name or meta['model_name'] == 'Lasso Regression':
                preprocessor_path = os.path.join(model_dir, meta['preprocessor_file'])
                metadata = meta
                break
    
    # If no matching metadata found, look for any Lasso Regression metadata
    if preprocessor_path is None:
        for metadata_file in reversed(metadata_files):
            metadata_path = os.path.join(model_dir, metadata_file)
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                if meta['model_name'] == 'Lasso Regression':
                    preprocessor_path = os.path.join(model_dir, meta['preprocessor_file'])
                    metadata = meta
                    break
    
    if preprocessor_path and os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)
        print(f"✅ Loaded preprocessors from: {os.path.basename(preprocessor_path)}")
    else:
        print("⚠️  Warning: Could not find matching preprocessor file")
        preprocessors = None
    
    return {
        'model': model,
        'preprocessors': preprocessors,
        'metadata': metadata
    }


def evaluate_lasso_on_test_data(features_path, target_path, model_data):
    """
    Evaluate Lasso Regression model on test data
    
    Args:
        features_path: Path to features CSV
        target_path: Path to target CSV
        model_data: Dictionary with model and preprocessors
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print("📊 EVALUATING LASSO REGRESSION MODEL ON TEST DATA")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading data...")
    df = models.load_and_prepare_data(features_path, target_path)
    
    # Preprocess features
    X, y, feature_names, preprocessors = models.preprocess_features(df)
    
    # Split data
    data_splits = models.split_data(X, y)
    
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    
    # Get the model
    model = model_data['model']
    
    # Evaluate on all datasets
    print("\n" + "-"*80)
    train_metrics = models.evaluate_model(model, X_train, y_train, "🏋️  TRAIN SET")
    print("-"*80)
    val_metrics = models.evaluate_model(model, X_val, y_val, "✔️  VALIDATION SET")
    print("-"*80)
    test_metrics = models.evaluate_model(model, X_test, y_test, "🧪 TEST SET")
    print("-"*80)
    
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'dataset_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test),
            'total': len(X)
        }
    }
    
    return results


def generate_predictions_with_lasso(features_path, target_path, model_data, output_file=None):
    """
    Generate predictions using Lasso Regression model
    
    Args:
        features_path: Path to features CSV
        target_path: Path to target CSV
        model_data: Dictionary with model and preprocessors
        output_file: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*80)
    print("🔮 GENERATING PREDICTIONS WITH LASSO REGRESSION")
    print("="*80)
    
    # Load data
    df = models.load_and_prepare_data(features_path, target_path)
    
    # Preprocess
    X, y, feature_names, preprocessors = models.preprocess_features(df)
    
    # Get predictions
    model = model_data['model']
    y_pred = model.predict(X)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_corners'] = y_pred
    
    # Find target column
    if 'target_corners' in df.columns:
        results_df['actual_corners'] = results_df['target_corners']
    elif 'corners_target' in df.columns:
        results_df['actual_corners'] = results_df['corners_target']
    else:
        results_df['actual_corners'] = y
    
    results_df['prediction_error'] = np.abs(results_df['predicted_corners'] - results_df['actual_corners'])
    results_df['within_1_corner'] = results_df['prediction_error'] <= 1
    
    # Save if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Predictions saved to: {output_file}")
    
    # Print statistics
    print(f"\n📈 Prediction Statistics:")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Mean prediction: {y_pred.mean():.2f} corners")
    print(f"  Mean actual: {results_df['actual_corners'].mean():.2f} corners")
    print(f"  Mean error: {results_df['prediction_error'].mean():.3f} corners")
    print(f"  Within 1 corner: {results_df['within_1_corner'].sum()} ({results_df['within_1_corner'].sum()/len(results_df)*100:.1f}%)")
    
    return results_df


def print_model_info(model_data):
    """
    Print information about the loaded model
    
    Args:
        model_data: Dictionary with model and metadata
    """
    print("\n" + "="*80)
    print("ℹ️  MODEL INFORMATION")
    print("="*80)
    
    if model_data['metadata']:
        metadata = model_data['metadata']
        print(f"\nModel Name: {metadata['model_name']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"\nFeatures ({len(metadata['feature_names'])}):")
        for i, feature in enumerate(metadata['feature_names'], 1):
            print(f"  {i}. {feature}")
        
        print(f"\nTraining Metrics:")
        for metric, value in metadata['metrics']['train_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print(f"\nValidation Metrics:")
        for metric, value in metadata['metrics']['val_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print(f"\nTest Metrics:")
        for metric, value in metadata['metrics']['test_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print("No metadata available for this model")


def evaluate_lasso_on_custom_csv(csv_path, model_data):
    """
    Evaluate Lasso Regression model on custom CSV data
    
    Args:
        csv_path: Path to the CSV file with all data
        model_data: Dictionary with model and preprocessors
    
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("📊 EVALUATING LASSO REGRESSION MODEL ON CUSTOM CSV DATA")
    print("="*80)
    
    # Load the CSV
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from CSV")
    print(f"   Columns: {list(df.columns)}")
    
    # Preprocess features
    X, y, feature_names, preprocessors = models.preprocess_features(df)
    
    # Split data into train, validation, and test
    data_splits = models.split_data(X, y)
    
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    # Get the model
    model = model_data['model']
    
    # Evaluate on all datasets
    print("\n" + "-"*80)
    train_metrics = models.evaluate_model(model, X_train, y_train, "🏋️  TRAIN SET")
    print("-"*80)
    val_metrics = models.evaluate_model(model, X_val, y_val, "✔️  VALIDATION SET")
    print("-"*80)
    test_metrics = models.evaluate_model(model, X_test, y_test, "🧪 TEST SET")
    print("-"*80)
    
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'dataset_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test),
            'total': len(X)
        },
        'data_splits': data_splits,
        'df': df,
        'X': X,
        'y': y
    }
    
    return results


def generate_predictions_on_custom_csv(csv_path, model_data, output_file=None):
    """
    Generate predictions using Lasso Regression model on custom CSV
    
    Args:
        csv_path: Path to the CSV file
        model_data: Dictionary with model and preprocessors
        output_file: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*80)
    print("🔮 GENERATING PREDICTIONS WITH LASSO REGRESSION ON CUSTOM CSV")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Preprocess
    X, y, feature_names, preprocessors = models.preprocess_features(df)
    
    # Get predictions
    model = model_data['model']
    y_pred = model.predict(X)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_corners'] = y_pred
    
    # Find target column
    if 'target_corners' in df.columns:
        results_df['actual_corners'] = results_df['target_corners']
    elif 'corners_target' in df.columns:
        results_df['actual_corners'] = results_df['corners_target']
    else:
        results_df['actual_corners'] = y
    
    results_df['prediction_error'] = np.abs(results_df['predicted_corners'] - results_df['actual_corners'])
    results_df['within_1_corner'] = results_df['prediction_error'] <= 1
    
    # Save if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Predictions saved to: {output_file}")
    
    # Print statistics
    print(f"\n📈 Prediction Statistics:")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Mean prediction: {y_pred.mean():.2f} corners")
    print(f"  Mean actual: {results_df['actual_corners'].mean():.2f} corners")
    print(f"  Mean error: {results_df['prediction_error'].mean():.3f} corners")
    print(f"  Within 1 corner: {results_df['within_1_corner'].sum()} ({results_df['within_1_corner'].sum()/len(results_df)*100:.1f}%)")
    
    return results_df


def main():
    """
    Main evaluation routine
    """
    # Get paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    from datetime import datetime
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    features_path = os.path.join(base_dir, f"data/output/HISTORIC_{start_date}_to_{end_date}.csv")
    target_path = os.path.join(base_dir, f"data/output/TARGET_CORNERS_{start_date}_to_{end_date}.csv")
    model_dir = os.path.join(base_dir, "data/models")
    output_predictions = os.path.join(base_dir, "data/output/lasso_predictions.csv")
    
    # Path to custom test CSV
    custom_csv_path = os.path.join(base_dir, "data/test/sample_1000_rows.csv")
    custom_output_predictions = os.path.join(base_dir, "data/output/lasso_predictions_custom_csv.csv")
    
    # Load Lasso Regression model
    print("Loading Lasso Regression model...")
    model_data = load_lasso_model(model_dir)
    
    # Print model information
    print_model_info(model_data)
    
    # Check if standard data files exist and evaluate if they do
    if os.path.exists(features_path) and os.path.exists(target_path):
        print("\n" + "="*80)
        print("📊 EVALUATING ON STANDARD DATA FILES")
        print("="*80)
        
        # Evaluate on test data
        evaluation_results = evaluate_lasso_on_test_data(features_path, target_path, model_data)
        
        # Generate predictions
        predictions_df = generate_predictions_with_lasso(features_path, target_path, model_data, output_predictions)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ STANDARD DATA EVALUATION COMPLETE")
        print("="*80)
        print(f"\nTest Set Performance Summary:")
        print(f"  MAE:  {evaluation_results['test_metrics']['MAE']:.3f} corners")
        print(f"  RMSE: {evaluation_results['test_metrics']['RMSE']:.3f} corners")
        print(f"  R²:   {evaluation_results['test_metrics']['R2']:.3f}")
        print(f"  Within 1 corner: {evaluation_results['test_metrics']['Within_1_Corner_%']:.1f}%")
        print(f"\nDataset Breakdown:")
        print(f"  Training samples:   {evaluation_results['dataset_sizes']['train']}")
        print(f"  Validation samples: {evaluation_results['dataset_sizes']['val']}")
        print(f"  Test samples:       {evaluation_results['dataset_sizes']['test']}")
        print(f"  Total samples:      {evaluation_results['dataset_sizes']['total']}")
        print("="*80)
    else:
        print(f"\n⚠️  Standard data files not found:")
        print(f"   Features: {features_path}")
        print(f"   Target:   {target_path}")
        print("   Skipping standard data evaluation...")
    
    # Check if custom CSV exists and evaluate
    if os.path.exists(custom_csv_path):
        print("\n\n" + "="*80)
        print("📊 EVALUATING ON CUSTOM CSV DATA")
        print("="*80)
        
        # Evaluate on custom CSV
        custom_evaluation_results = evaluate_lasso_on_custom_csv(custom_csv_path, model_data)
        
        # Generate predictions on custom CSV
        custom_predictions_df = generate_predictions_on_custom_csv(custom_csv_path, model_data, custom_output_predictions)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ CUSTOM CSV EVALUATION COMPLETE")
        print("="*80)
        print(f"\nTest Set Performance Summary (Custom CSV):")
        print(f"  MAE:  {custom_evaluation_results['test_metrics']['MAE']:.3f} corners")
        print(f"  RMSE: {custom_evaluation_results['test_metrics']['RMSE']:.3f} corners")
        print(f"  R²:   {custom_evaluation_results['test_metrics']['R2']:.3f}")
        print(f"  Within 1 corner: {custom_evaluation_results['test_metrics']['Within_1_Corner_%']:.1f}%")
        print(f"\nDataset Breakdown (Custom CSV):")
        print(f"  Training samples:   {custom_evaluation_results['dataset_sizes']['train']}")
        print(f"  Validation samples: {custom_evaluation_results['dataset_sizes']['val']}")
        print(f"  Test samples:       {custom_evaluation_results['dataset_sizes']['test']}")
        print(f"  Total samples:      {custom_evaluation_results['dataset_sizes']['total']}")
        print("="*80)
    else:
        print(f"\n⚠️  Custom CSV file not found: {custom_csv_path}")
        print("   Skipping custom CSV evaluation...")


if __name__ == "__main__":
    main()
