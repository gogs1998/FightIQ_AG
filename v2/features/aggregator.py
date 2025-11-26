import pandas as pd
import sys
import os

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_elo import calculate_dynamic_elo
from common_opponents import calculate_common_opponent_features
from stoppage import calculate_stoppage_features
from pear import calculate_pear_features

def build_feature_set(data_path='v2/data/UFC_data_with_elo.csv'):
    print("Loading raw data...")
    df = pd.read_csv(data_path)
    
    # 1. Dynamic Elo (Replaces standard Elo)
    print("Calculating Dynamic Elo...")
    df = calculate_dynamic_elo(df)
    
    # 2. Common Opponents
    print("Calculating Common Opponent Features...")
    df = calculate_common_opponent_features(df)
    
    # 3. Stoppage Propensity
    print("Calculating Stoppage Features...")
    df = calculate_stoppage_features(df)
    
    # 4. PEAR (Advanced Stats)
    print("Calculating PEAR Features...")
    df = calculate_pear_features(df)
    
    # 5. Clean and Format
    # Drop rows with missing target or critical features
    df = df.dropna(subset=['winner'])
    
    # Create Target
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    return df

if __name__ == "__main__":
    df = build_feature_set()
    print(f"Feature set built: {df.shape}")
    df.to_csv('v2/data/training_data_v2.csv', index=False)
