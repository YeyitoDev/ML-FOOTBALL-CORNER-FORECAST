import utils
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd

#entrenar el modelo de predicción de resultados de fútbol

# PARAMETERS
start_date = "2025-01-01"  # Adjusted to where data exists

#GET ACTUAL DATE    
end_date = datetime.now().strftime("%Y-%m-%d")



# DATA DIRECTORIES - Use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIXTURES_DIR = os.path.join(DATA_DIR, "fixtures")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
TEAM_STATS_DIR = os.path.join(DATA_DIR, "team_stats")
ODDS_DIR = os.path.join(DATA_DIR, "odds")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X, Y = utils.load_transform_fixture_data_per_date_range(start_date, end_date)
    X.to_csv(os.path.join(OUTPUT_DIR, f"HISTORIC_{start_date}_to_{end_date}.csv"))
    Y.to_csv(os.path.join(OUTPUT_DIR, f"TARGET_CORNERS_{start_date}_to_{end_date}.csv"))

    
if __name__ == "__main__":
    main()