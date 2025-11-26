import pandas as pd
import json
import numpy as np

def build_fighter_db():
    print("=== Building Fighter Database (Latest Stats & Elo) ===")
    
    # Assuming running from master/
    # Data is in ../data/training_data.csv relative to script? 
    # No, CWD is master, so data is in ../data/training_data.csv?
    # Wait, previous run read 'data/training_data.csv' successfully.
    # That means CWD was d:/AntiGravity/FightIQ/master but data is at d:/AntiGravity/FightIQ/data/training_data.csv?
    # If CWD is master, 'data/training_data.csv' would look for d:/AntiGravity/FightIQ/master/data/training_data.csv
    # But the previous error was about the OUTPUT file.
    # Let's check where the data file is.
    # d:/AntiGravity/FightIQ/data/training_data.csv exists.
    # If I run from master, I should use '../data/training_data.csv'.
    # BUT the previous run's output showed "Processing 2615 fighters...", so it DID read the CSV.
    # This implies 'data/training_data.csv' worked? 
    # Maybe the CWD was actually d:/AntiGravity/FightIQ?
    # The tool call said Cwd: d:/AntiGravity/FightIQ/master.
    # Let's be safe and use absolute paths or try-catch.
    
    try:
        df = pd.read_csv('../data/training_data.csv')
    except:
        df = pd.read_csv('data/training_data.csv')
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    fighter_db = {}
    
    all_fighters = set(df['f_1_name'].unique()) | set(df['f_2_name'].unique())
    print(f"Processing {len(all_fighters)} fighters...")
    
    stat_cols = [c for c in df.columns if c.startswith('f_1_') and 'odds' not in c and 'name' not in c and 'elo' not in c]
    stat_names = [c.replace('f_1_', '') for c in stat_cols]
    
    for fighter in all_fighters:
        f1_mask = df['f_1_name'] == fighter
        f2_mask = df['f_2_name'] == fighter
        
        last_f1 = df[f1_mask].iloc[-1] if f1_mask.any() else None
        last_f2 = df[f2_mask].iloc[-1] if f2_mask.any() else None
        
        last_row = None
        is_f1 = True
        
        if last_f1 is not None and last_f2 is not None:
            if last_f1['event_date'] > last_f2['event_date']:
                last_row = last_f1
                is_f1 = True
            else:
                last_row = last_f2
                is_f1 = False
        elif last_f1 is not None:
            last_row = last_f1
            is_f1 = True
        elif last_f2 is not None:
            last_row = last_f2
            is_f1 = False
            
        if last_row is None:
            continue
            
        stats = {}
        prefix = 'f_1_' if is_f1 else 'f_2_'
        
        stats['elo'] = last_row[f'{prefix}elo']
        
        for stat in stat_names:
            col = f"{prefix}{stat}"
            if col in last_row:
                stats[stat] = last_row[col]
                
        stats['last_fight_date'] = str(last_row['event_date'])
        fighter_db[fighter] = stats
        
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float64): return float(o)
        return o
        
    # Save to current directory
    with open('fighter_db_production.json', 'w') as f:
        json.dump(fighter_db, f, default=convert)
        
    print(f"Saved database for {len(fighter_db)} fighters to fighter_db_production.json")

if __name__ == "__main__":
    build_fighter_db()
