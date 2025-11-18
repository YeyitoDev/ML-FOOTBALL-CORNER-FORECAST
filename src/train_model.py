"""
Train the corners prediction model
"""
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))

import models

if __name__ == "__main__":
    # Paths - use absolute paths to avoid issues
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use the same dates as main.py generates
    from datetime import datetime
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    features_path = os.path.join(base_dir, f"data/output/HISTORIC_{start_date}_to_{end_date}.csv")
    target_path = os.path.join(base_dir, f"data/output/TARGET_CORNERS_{start_date}_to_{end_date}.csv")
    output_dir = os.path.join(base_dir, "data/models")
    
    # Run pipeline
    results = models.run_full_pipeline(features_path, target_path, output_dir)
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {results['test_metrics']['MAE']:.3f} corners")
    print(f"  RMSE: {results['test_metrics']['RMSE']:.3f} corners")
    print(f"  R²:   {results['test_metrics']['R2']:.3f}")
    print(f"  Predictions within 1 corner: {results['test_metrics']['Within_1_Corner_%']:.1f}%")
    print("\n" + "="*80)
