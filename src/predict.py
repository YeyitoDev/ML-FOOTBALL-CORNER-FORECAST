"""
Make predictions with the trained corners prediction model
"""
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(__file__))

import models

def predict_example():
    """Example of how to use the trained model for predictions"""
    
    # Load the model
    print("Loading trained model...")
    model_data = models.load_model("../data/models")
    
    model = model_data['model']
    preprocessors = model_data['preprocessors']
    
    # Example: Predict for new team features
    # These should match the format of your training data
    example_features = pd.DataFrame([
        {
            'avg_goals': 1.5,
            'shots_on_goal': 5.2,
            'pass_accuracy': 82.0,
            'possession': 55.0,
            'yellow_cards': 2.0,
            'red_cards': 0.1,
            'corners_features': 5.5,
            'fouls': 12.0,
            'num_matches': 5,
            'streak': 'WWDLL',
            'home_bin': True
        }
    ])
    
    # Preprocess the same way as training
    example_features['streak_score'] = example_features['streak'].apply(
        lambda s: sum([
            (3 if c=='W' else 1 if c=='D' else 0) * w 
            for c, w in zip(s[:5], [0.4, 0.3, 0.2, 0.1, 0.05])
        ])
    )
    example_features['home_bin_encoded'] = example_features['home_bin'].astype(int)
    
    # Make prediction
    prediction = models.predict_corners(model, preprocessors, example_features)
    
    print(f"\n🎯 Predicted corners: {prediction[0]:.1f}")
    
    return prediction


def predict_from_csv(input_csv, output_csv="../data/predictions/corners_predictions.csv"):
    """
    Make predictions for a CSV file of team features
    
    Args:
        input_csv: Path to CSV with team features
        output_csv: Path to save predictions
    """
    # Load model
    print("Loading trained model...")
    model_data = models.load_model("../data/models")
    
    model = model_data['model']
    preprocessors = model_data['preprocessors']
    
    # Load input data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Preprocess
    df['streak_score'] = df['streak'].apply(
        lambda s: sum([
            (3 if c=='W' else 1 if c=='D' else 0) * w 
            for c, w in zip(str(s)[:5], [0.4, 0.3, 0.2, 0.1, 0.05])
        ]) if pd.notna(s) else 0
    )
    df['home_bin_encoded'] = df['home_bin'].astype(int)
    
    # Predict
    print("Making predictions...")
    predictions = models.predict_corners(model, preprocessors, df)
    
    # Add predictions to dataframe
    df['predicted_corners'] = predictions
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Predictions saved to {output_csv}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Mean predicted corners: {predictions.mean():.2f}")
    
    return df


if __name__ == "__main__":
    # Run example prediction
    print("="*80)
    print("Example: Single Prediction")
    print("="*80)
    predict_example()
    
    # Optionally predict from CSV
    # Uncomment to use:
    # print("\n" + "="*80)
    # print("Batch Prediction from CSV")
    # print("="*80)
    # predict_from_csv("../data/output/HISTORIC_2025-01-01_to_2025-11-18.csv")
