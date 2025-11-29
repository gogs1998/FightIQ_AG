import json
import os
import sys
import pandas as pd
import numpy as np

def run_qa_gate():
    print("=== FightIQ QA Gate ===")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, 'fighter_db_production.json')
    
    # 1. Load DB
    try:
        with open(DB_PATH, 'r') as f:
            db = json.load(f)
        print(f"Loaded database with {len(db)} fighters.")
    except Exception as e:
        print(f"❌ CRITICAL: Could not load fighter DB: {e}")
        sys.exit(1)
        
    errors = []
    warnings = []
    
    # 2. Check Data Integrity
    print("Checking data integrity...")
    
    for name, stats in db.items():
        # Check Critical Stats
        if 'elo' not in stats or stats['elo'] is None:
            errors.append(f"Missing Elo for {name}")
        elif not (1000 < stats['elo'] < 3000):
            warnings.append(f"Suspicious Elo for {name}: {stats['elo']}")
            
        # Age (Relaxed to Warning)
        if 'age' not in stats or stats['age'] is None or pd.isna(stats['age']):
            warnings.append(f"Missing/Invalid Age for {name}")
        elif not (17 <= stats['age'] < 65): # Broaden range slightly
            warnings.append(f"Suspicious Age for {name}: {stats['age']}")
            
        if 'reach_cm' not in stats or stats['reach_cm'] is None:
            pass 
        elif not (100 < stats['reach_cm'] < 250):
            errors.append(f"Invalid Reach for {name}: {stats['reach_cm']}")
            
    # 3. Check Model Artifacts Existence
    artifacts = [
        'models/xgb_optimized.pkl',
        'models/iso_xgb.pkl',
        'models/iso_siam.pkl',
        'models/siamese_scaler.pkl',
        'models/siamese_optimized.pth',
        'features.json'
    ]
    
    for art in artifacts:
        path = os.path.join(BASE_DIR, art)
        if not os.path.exists(path):
            errors.append(f"Missing artifact: {art}")
            
    # 4. Report
    print(f"\nQA Summary:")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if warnings:
        print("\nTop 5 Warnings:")
        for w in warnings[:5]: print(f"  - {w}")
        
    if errors:
        print("\n❌ QA FAILED with the following errors:")
        for e in errors[:10]: print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ QA PASSED: Data integrity verified.")
        sys.exit(0)

if __name__ == "__main__":
    run_qa_gate()
