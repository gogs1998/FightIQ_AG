import json
import pandas as pd

def audit_leakage():
    print("Loading features...")
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    print(f"Total Features: {len(features)}")
    
    # 1. Keyword Audit
    # These words should NEVER appear in a feature name unless it's an average or historical stat
    dangerous_keywords = [
        'landed', 'attempted', 'knockdowns', 'submission', 'control', 'rev', 
        'succ', 'att', 'acc', 'share', 'pct', 'strikes', 'ground', 'head', 'leg', 'distance', 'clinch'
    ]
    
    # These words signify aggregation/history, which makes the dangerous words OK
    safety_keywords = ['avg', 'cum', 'per_min', 'diff', 'slpm', 'sapm', 'def', 'career', 'last', 'streak']
    
    # Special case: 'diff_' features. 
    # If 'diff_head_acc_r1_15' means "Difference in Head Accuracy in Round 1 of the LAST 15 fights", it's safe.
    # If it means "Difference in Head Accuracy in Round 1 of THIS fight", it's LEAKAGE.
    # Given the naming convention usually seen (e.g. '_15', '_10'), it implies historical window.
    # But we must verify if there are any features that lack a window suffix or 'avg'.
    
    suspicious = []
    
    for feat in features:
        feat_lower = feat.lower()
        
        # Skip Odds and Elo (safe)
        if 'odds' in feat_lower or 'elo' in feat_lower:
            continue
            
        # Check for dangerous keywords
        if any(k in feat_lower for k in dangerous_keywords):
            # It has a dangerous word. Is it safe?
            
            # Check 1: Does it have a safety keyword?
            is_safe_keyword = any(k in feat_lower for k in safety_keywords)
            
            # Check 2: Does it have a numeric suffix indicating a window? (e.g. _15, _10, _3)
            # Most features end with _f_1 or _f_2. The window is usually before that.
            # e.g. slpm_15_f_2 -> 15 is the window.
            # e.g. diff_head_acc_r2_11 -> 11 is likely the window.
            
            parts = feat.split('_')
            has_numeric_window = False
            for p in parts:
                if p.isdigit():
                    has_numeric_window = True
                    break
            
            if not is_safe_keyword and not has_numeric_window:
                suspicious.append(feat)
                
    print(f"\nSuspicious Features found: {len(suspicious)}")
    if suspicious:
        print(suspicious)
    else:
        print("No obvious keyword-based leakage found.")

    # 2. Correlation Audit
    # Leakage often has 100% or near 100% correlation with the target or other leakage features.
    # We will check correlation of top features with the target in the training set.
    
    print("\nChecking Correlations...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Check correlation of all features with target
    # We only check numeric ones for speed
    numeric_feats = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    corrs = df[numeric_feats].corrwith(df['target']).abs().sort_values(ascending=False)
    
    print("\nTop 10 Correlated Features with Target:")
    print(corrs.head(10))
    
    # If any correlation is > 0.9, it's highly suspicious of leakage (or just a proxy for the winner like 'score')
    high_corr = corrs[corrs > 0.8]
    if len(high_corr) > 0:
        print("\nWARNING: High correlation features detected (> 0.8):")
        print(high_corr)
    else:
        print("\nNo suspiciously high correlations (> 0.8) detected.")

if __name__ == "__main__":
    audit_leakage()
