import pandas as pd
import json
import os

def get_base_feature(diff_feat):
    # diff_X -> X
    # diff_X_r1_... -> X_r1_...
    # The naming convention seems to be diff_FEATURENAME
    # But sometimes it might be diff_FEATURE_NAME_suffix
    # Let's assume it's just removing 'diff_' prefix
    return diff_feat[5:]

def build_db():
    print("Loading top features...")
    with open('top_features.json', 'r') as f:
        features = json.load(f)
        
    # Identify all required base features
    required_generics = set()
    
    for feat in features:
        if feat.startswith('diff_'):
            base = get_base_feature(feat)
            required_generics.add(base)
        elif feat.startswith('f_1_'):
            required_generics.add(feat[4:])
        elif feat.startswith('f_2_'):
            required_generics.add(feat[4:])
        elif feat.endswith('_f_1'):
            required_generics.add(feat[:-4])
        elif feat.endswith('_f_2'):
            required_generics.add(feat[:-4])
            
    print(f"Identified {len(required_generics)} unique base features needed.")
    
    # Load Data
    print("Loading dataset...")
    df = pd.read_csv('UFC_full_data_golden.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    fighter_db = {}
    
    # We iterate and update. 
    # To speed up, we can just group by fighter and take the last row?
    # But a fighter appears in f_1 or f_2 columns.
    # So we need to find the last appearance for each fighter.
    
    # Get all unique fighters
    fighters = set(df['f_1_name'].unique()) | set(df['f_2_name'].unique())
    print(f"Total fighters found: {len(fighters)}")
    
    # It's faster to iterate rows once than to query for each fighter
    count = 0
    for _, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        # Extract stats for F1
        stats_1 = {}
        for gen in required_generics:
            # Try different patterns to find the column in the row
            # Pattern 1: f_1_GEN
            # Pattern 2: GEN_f_1
            
            col_name = None
            if f"f_1_{gen}" in row:
                col_name = f"f_1_{gen}"
            elif f"{gen}_f_1" in row:
                col_name = f"{gen}_f_1"
            
            if col_name:
                stats_1[gen] = row[col_name]
        
        # Extract stats for F2
        stats_2 = {}
        for gen in required_generics:
            col_name = None
            if f"f_2_{gen}" in row:
                col_name = f"f_2_{gen}"
            elif f"{gen}_f_2" in row:
                col_name = f"{gen}_f_2"
            
            if col_name:
                stats_2[gen] = row[col_name]
                
        # Update DB
        if stats_1: fighter_db[f1] = stats_1
        if stats_2: fighter_db[f2] = stats_2
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} fights...")
            
    print(f"DB built for {len(fighter_db)} fighters.")
    
    # Save
    with open('fighter_db.json', 'w') as f:
        # Convert numpy types to python types for json
        # (pandas rows return numpy types)
        def convert(o):
            if isinstance(o, pd.Timestamp): return str(o)
            if hasattr(o, 'item'): return o.item()
            return o
            
        json.dump(fighter_db, f, default=convert)
        
    print("Saved to fighter_db.json")

if __name__ == "__main__":
    build_db()
