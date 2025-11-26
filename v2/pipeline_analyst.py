import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.analyst_ensemble import AnalystEnsemble

def run_analyst_pipeline():
    print("=== Starting Analyst Pipeline (The Truth) ===")
    
    # 1. Load Data
    data_path = 'v2/data/training_data_v2.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}. Run features/aggregator.py first.")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Initialize Model
    model = AnalystEnsemble()
    
    # 3. Train & Calibrate
    model.fit(df)
    
    # 4. Save
    save_path = 'v2/models/analyst_model.pkl'
    model.save(save_path)
    print(f"Analyst Model saved to {save_path}")
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    try:
        run_analyst_pipeline()
    except Exception as e:
        with open('error_log.txt', 'w') as f:
            import traceback
            traceback.print_exc(file=f)
        print("Error logged to error_log.txt")
        sys.exit(1)
