import pandas as pd
import numpy as np
import joblib
import sys
import os
import json

# Add v2 to path
sys.path.append(os.path.join(os.getcwd(), 'v2'))
from models.analyst_ensemble import AnalystEnsemble
from models.siamese_model import SiameseModel

def generate_error_episodes(model, df):
    print("Generating Error Episodes...")
    
    # Predict
    # We use the full dataset to find ALL historical errors for "film study"
    # In a real production loop, this might be just the latest batch
    X = model.preprocess(df, is_train=False)
    p_xgb = model.xgb_model.predict_proba(X)[:, 1]
    p_sia = model.siamese_model.predict_proba(df)
    p_ens = (p_xgb + p_sia) / 2.0
    
    df['prob_ens'] = p_ens
    df['pred_winner'] = (p_ens > 0.5).astype(int)
    df['actual_winner'] = df['target']
    df['is_correct'] = df['pred_winner'] == df['actual_winner']
    
    # Calculate Confidence (0.5 - 1.0)
    df['confidence'] = np.where(df['prob_ens'] > 0.5, df['prob_ens'], 1 - df['prob_ens'])
    
    # Filter for Errors
    errors = df[~df['is_correct']].copy()
    
    # Categorize
    def categorize_error(row):
        if row['confidence'] > 0.60:
            return 'High (>60%)'
        elif row['confidence'] > 0.50:
            return 'Medium (50-60%)'
        else:
            return 'Low (<50%)' # Should not happen if pred_winner is based on >0.5, but for completeness
            
    errors['error_type'] = errors.apply(categorize_error, axis=1)
    
    print(f"Found {len(errors)} total errors.")
    print(errors['error_type'].value_counts())
    
    return errors

def diagnose_errors(error_df, full_df):
    print("\nDiagnosing Errors (Feature Fingerprints)...")
    
    # We want to find features that are "misleading".
    # A feature is misleading if it strongly favors the predicted winner (who lost).
    # e.g. Predicted Winner has +20cm Reach (Feature value high), but lost.
    
    # 1. Identify Numeric Features
    numeric_cols = error_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude non-feature cols
    exclude = ['target', 'prob_ens', 'pred_winner', 'actual_winner', 'is_correct', 'confidence', 'fight_id', 'event_id']
    features = [c for c in numeric_cols if c not in exclude and not c.startswith('prob_')]
    
    # 2. Calculate "Misleadingness"
    # For each error, we look at the feature values.
    # If pred_winner == 1 (F1), and F1 lost, then high values of F1-favoring features are misleading.
    # If pred_winner == 0 (F2), and F2 lost, then high values of F2-favoring features are misleading.
    
    # Let's simplify: We look at the correlation of features with 'confidence' WITHIN the error set.
    # If higher confidence correlates with a feature in the error set, that feature is a "trap".
    
    # Better yet: Compare Error Distribution vs Global Distribution
    # If "Reach Diff" mean in Errors is much higher than in Correct predictions, it's a trap.
    
    diagnosis = {}
    
    correct_df = full_df[full_df['is_correct']]
    
    print(f"Comparing {len(error_df)} Errors vs {len(correct_df)} Correct...")
    
    for feat in features:
        # Skip if not enough variance
        if full_df[feat].nunique() < 5: continue
        
        mean_err = error_df[feat].mean()
        mean_corr = correct_df[feat].mean()
        
        # Normalize difference by std dev
        std = full_df[feat].std()
        if std == 0: continue
        
        z_score = (mean_err - mean_corr) / std
        
        if abs(z_score) > 0.2: # Threshold for "interesting" difference
            diagnosis[feat] = {
                'mean_error': mean_err,
                'mean_correct': mean_corr,
                'z_score': z_score,
                'interpretation': 'Higher in Errors' if z_score > 0 else 'Lower in Errors'
            }
            
    # Sort by magnitude of Z-score
    sorted_diag = sorted(diagnosis.items(), key=lambda x: abs(x[1]['z_score']), reverse=True)
    
    return sorted_diag

def run_losing_learning():
    print("=== Phase 3: Losing-Learning Pipeline ===")
    
    # 1. Load
    df = pd.read_csv('v2/data/training_data_v2.csv')
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    try:
        model = AnalystEnsemble.load('v2/models/analyst_model.pkl')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Generate Episodes
    # Use Test Set only for "honest" error analysis? 
    # Or Full History to find ALL patterns?
    # User said: "Pull all fights where FightIQ predicted the wrong winner"
    # Let's use the last 20% to simulate "recent" history for film study
    split_idx = int(len(df) * 0.80)
    study_df = df.iloc[split_idx:].copy()
    
    full_scored_df = study_df.copy() # Will hold is_correct for comparison
    
    errors = generate_error_episodes(model, study_df) # Modifies study_df in place to add probs
    full_scored_df = study_df # Now has is_correct
    
    # Save Episodes
    os.makedirs('experimental/data', exist_ok=True)
    errors.to_csv('experimental/data/error_episodes.csv', index=False)
    print("Saved error_episodes.csv")
    
    # 3. Diagnose
    fingerprints = diagnose_errors(errors, full_scored_df)
    
    print("\n--- Top 10 Misleading Features (Error Fingerprints) ---")
    print("Feature | Z-Score (Diff between Errors and Correct) | Interpretation")
    for k, v in fingerprints[:10]:
        print(f"{k:<30} | {v['z_score']:.4f} | {v['interpretation']}")
        
    # Save Fingerprints
    with open('experimental/data/error_fingerprints.json', 'w') as f:
        json.dump(fingerprints, f, indent=4)
    print("\nSaved error_fingerprints.json")

if __name__ == "__main__":
    run_losing_learning()
