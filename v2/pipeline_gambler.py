import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gambler_model import GamblerModel

def run_gambler_pipeline():
    print("=== Starting Gambler Pipeline (The Money) ===")
    
    # 1. Load Data
    data_path = 'v2/data/training_data_v2.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}. Run features/aggregator.py first.")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Initialize Model
    model = GamblerModel()
    
    # 3. Train
    model.fit(df)
    
    # 4. Save
    save_path = 'v2/models/gambler_model.pkl'
    model.save(save_path)
    print(f"Gambler Model saved to {save_path}")
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    run_gambler_pipeline()
