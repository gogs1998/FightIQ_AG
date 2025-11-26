import pandas as pd
import json
import re

def find_pairs():
    # Load safe features
    try:
        with open('features_elo.json', 'r') as f:
            safe_features = set(json.load(f))
    except:
        print("features_elo.json not found, using dummy list")
        safe_features = []

    # Load columns
    try:
        df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
        all_cols = set(df.columns)
    except:
        print("UFC_data_with_elo.csv not found")
        return

    pairs = set()
    
    # Regex for extracting base from diffs
    # diff_X -> X
    
    # Regex for f1/f2
    # f_1_X -> X
    # X_f_1 -> X
    
    for feat in safe_features:
        base = None
        
        if feat.startswith('diff_'):
            base = feat[5:] # remove diff_
            
            # Check for f1/f2 versions
            # Try prefix f_1_base / f_2_base
            c1_pre = f"f_1_{base}"
            c2_pre = f"f_2_{base}"
            
            # Try suffix base_f_1 / base_f_2
            c1_suf = f"{base}_f_1"
            c2_suf = f"{base}_f_2"
            
            if c1_pre in all_cols and c2_pre in all_cols:
                pairs.add(tuple(sorted((c1_pre, c2_pre))))
            elif c1_suf in all_cols and c2_suf in all_cols:
                pairs.add(tuple(sorted((c1_suf, c2_suf))))
                
        elif '_f_1' in feat or 'f_1_' in feat:
            # It's an f1 feature
            if feat.startswith('f_1_'):
                c1 = feat
                c2 = feat.replace('f_1_', 'f_2_')
            else:
                c1 = feat
                c2 = feat.replace('_f_1', '_f_2')
                
            if c1 in all_cols and c2 in all_cols:
                pairs.add(tuple(sorted((c1, c2))))
                
        elif '_f_2' in feat or 'f_2_' in feat:
            # It's an f2 feature
            if feat.startswith('f_2_'):
                c2 = feat
                c1 = feat.replace('f_2_', 'f_1_')
            else:
                c2 = feat
                c1 = feat.replace('_f_2', '_f_1')
                
            if c1 in all_cols and c2 in all_cols:
                pairs.add(tuple(sorted((c1, c2))))

    print(f"Found {len(pairs)} unique pairs.")
    for p in list(pairs)[:10]:
        print(p)
        
if __name__ == "__main__":
    find_pairs()
