import pandas as pd
import json

def deep_audit():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    
    # Check specific suspicious columns manually
    # 'r1_duration', 'r2_duration', 'r3_duration' appeared in the feature list earlier.
    # If these are the duration of the *current* fight's rounds, that is LEAKAGE.
    # If they are average duration of *past* fights, it's fine.
    
    # Let's check if 'r1_duration' exists and what it looks like.
    cols_to_check = ['r1_duration', 'r2_duration', 'r3_duration']
    existing = [c for c in cols_to_check if c in df.columns]
    
    if existing:
        print(f"\nChecking {existing}...")
        print(df[existing].describe())
        
        # Check correlation with 'num_rounds' or 'finish_round' if they exist
        if 'finish_round' in df.columns:
            print("\nCorrelation with finish_round:")
            print(df[existing + ['finish_round']].corr()['finish_round'])
            
    # Also check 'f_2_fight_number' and 'f_1_fight_number'. 
    # These are just counters, safe.
    
    # Check 'wins_15_f_1'. 
    # If this includes the result of the current fight, it's leakage.
    # We can check this by seeing if 'wins' increases for the winner in the row? 
    # No, that's hard to check without previous rows.
    
    # Best check: Look at the feature list again for anything "current".
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    print("\n--- Manual Review of Potential Leakage ---")
    potential_leaks = []
    for f in features:
        if 'duration' in f:
            potential_leaks.append(f)
            
    print(f"Duration features: {potential_leaks}")

if __name__ == "__main__":
    deep_audit()
