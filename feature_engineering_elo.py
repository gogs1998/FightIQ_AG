import pandas as pd
import numpy as np

def calculate_elo(df, k_factor=32, initial_elo=1500):
    print("Calculating Elo Ratings...")
    
    # Ensure date sorted
    df = df.sort_values('event_date').copy()
    
    elo_ratings = {} # fighter_name -> current_elo
    
    # Columns to store historical Elo (entering the fight)
    f1_elo_col = []
    f2_elo_col = []
    
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        winner = row['winner']
        
        # Get current Elo or initial
        r1 = elo_ratings.get(f1, initial_elo)
        r2 = elo_ratings.get(f2, initial_elo)
        
        # Store *pre-fight* Elo
        f1_elo_col.append(r1)
        f2_elo_col.append(r2)
        
        # Calculate Expected Score
        # E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual Score
        if winner == f1:
            s1, s2 = 1, 0
        elif winner == f2:
            s1, s2 = 0, 1
        else:
            # Draw / NC
            s1, s2 = 0.5, 0.5
            
        # Update Elo
        new_r1 = r1 + k_factor * (s1 - e1)
        new_r2 = r2 + k_factor * (s2 - e2)
        
        elo_ratings[f1] = new_r1
        elo_ratings[f2] = new_r2
        
    df['f_1_elo'] = f1_elo_col
    df['f_2_elo'] = f2_elo_col
    df['diff_elo'] = df['f_1_elo'] - df['f_2_elo']
    
    return df

def add_elo_features():
    print("Loading data...")
    df = pd.read_csv('UFC_full_data_golden.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Calculate General Elo
    df = calculate_elo(df)
    
    print("Saving data with Elo...")
    df.to_csv('UFC_data_with_elo.csv', index=False)
    
    # Check correlation with target
    # Create target for check
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    subset = df[f1_wins | f2_wins].copy()
    subset['target'] = (subset['winner'] == subset['f_1_name']).astype(int)
    
    corr = subset[['target', 'diff_elo', 'f_1_elo', 'f_2_elo']].corr()['target']
    print("\nCorrelation with Target:")
    print(corr)

if __name__ == "__main__":
    add_elo_features()
