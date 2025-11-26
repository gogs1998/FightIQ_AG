import pandas as pd
import json

def extract_latest_elo():
    print("Loading Elo data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    elo_db = {}
    
    # Iterate to get the latest Elo for each fighter
    # Since we sorted by date, the last occurrence is the latest
    
    # We need to check both f_1 and f_2 columns
    # Actually, the calculated Elo in the row is the "pre-fight" Elo.
    # But we want the Elo *after* the last fight (or just the pre-fight of their next hypothetical fight).
    # The script calculated "pre-fight" Elo for the row.
    # So for the *latest* fight of a fighter, the `f_1_elo` is their Elo entering that fight.
    # We should ideally update it with the result of that last fight.
    # But `calculate_elo` function in `feature_engineering_elo.py` didn't return the final ratings dict.
    
    # Let's just re-run the Elo calculation logic to get the final state dict
    
    print("Recalculating final Elo state...")
    elo_ratings = {}
    k_factor = 32
    initial_elo = 1500
    
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        winner = row['winner']
        
        r1 = elo_ratings.get(f1, initial_elo)
        r2 = elo_ratings.get(f2, initial_elo)
        
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        if winner == f1:
            s1, s2 = 1, 0
        elif winner == f2:
            s1, s2 = 0, 1
        else:
            s1, s2 = 0.5, 0.5
            
        new_r1 = r1 + k_factor * (s1 - e1)
        new_r2 = r2 + k_factor * (s2 - e2)
        
        elo_ratings[f1] = new_r1
        elo_ratings[f2] = new_r2
        
    print(f"Captured Elo for {len(elo_ratings)} fighters.")
    
    with open('fighter_elo.json', 'w') as f:
        json.dump(elo_ratings, f)
    print("Saved to fighter_elo.json")

if __name__ == "__main__":
    extract_latest_elo()
