import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add v2 to path
sys.path.append(os.path.join(os.getcwd(), 'v2'))
from models.analyst_ensemble import AnalystEnsemble
from models.siamese_model import SiameseModel

def run_rea():
    print("=== Retrospective Error Analysis (REA) ===")
    
    # 1. Load Data & Model
    print("Loading data and model...")
    df = pd.read_csv('v2/data/training_data_v2.csv')
    
    # Ensure date sorting for correct train/test split simulation
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    try:
        model = AnalystEnsemble.load('v2/models/analyst_model.pkl')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Re-create Test Split (Last 15%)
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    print(f"Analyzing {len(test_df)} test cases...")
    
    # 3. Get Predictions
    # We need to use the model's predict method which returns (probs, sets)
    # But we want the raw probabilities for analysis
    
    # XGB Probs
    X_test = model.preprocess(test_df, is_train=False)
    p_xgb = model.xgb_model.predict_proba(X_test)[:, 1]
    
    # Siamese Probs
    p_sia = model.siamese_model.predict_proba(test_df)
    
    # Ensemble
    p_ens = (p_xgb + p_sia) / 2.0
    
    test_df['prob_xgb'] = p_xgb
    test_df['prob_sia'] = p_sia
    test_df['prob_ens'] = p_ens
    test_df['pred_winner'] = (p_ens > 0.5).astype(int)
    test_df['actual_winner'] = test_df['target']
    test_df['is_correct'] = test_df['pred_winner'] == test_df['actual_winner']
    test_df['error_magnitude'] = np.abs(test_df['actual_winner'] - test_df['prob_ens'])
    test_df['confidence'] = np.where(test_df['prob_ens'] > 0.5, test_df['prob_ens'], 1 - test_df['prob_ens'])
    
    # 4. Error Analysis
    print("\n--- Overall Metrics ---")
    acc = test_df['is_correct'].mean()
    print(f"Accuracy: {acc:.4%}")
    
    errors = test_df[~test_df['is_correct']].copy()
    print(f"Total Errors: {len(errors)}")
    
    # 4.1 Confidence vs Accuracy (Calibration Check)
    # Bin predictions by confidence (0.5-0.6, 0.6-0.7, etc.)
    test_df['conf_bin'] = pd.cut(test_df['confidence'], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    print("\n--- Accuracy by Confidence Bin ---")
    print(test_df.groupby('conf_bin')['is_correct'].mean())
    
    # 4.2 Feature Correlations with Error
    # Do we fail when specific features are high/low?
    # We look at the correlation between 'is_correct' (1=correct, 0=wrong) and features
    # Negative correlation means high feature value -> more errors
    
    print("\n--- Top Correlations with Correctness (What makes us win?) ---")
    # Select numeric features
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
    corrs = test_df[numeric_cols].corrwith(test_df['is_correct']).sort_values(ascending=False)
    
    print("Top 5 Positive (Easier to predict):")
    print(corrs.head(5))
    print("\nTop 5 Negative (Harder to predict):")
    print(corrs.tail(5))
    
    # 4.3 Specific Feature Analysis
    # Dynamic Elo Diff
    test_df['elo_diff_abs'] = test_df['diff_dynamic_elo'].abs()
    test_df['elo_bin'] = pd.cut(test_df['elo_diff_abs'], bins=5)
    print("\n--- Accuracy by Elo Difference ---")
    print(test_df.groupby('elo_bin')['is_correct'].mean())
    
    # Weight Class (if available, usually inferred from weight)
    if 'f_1_weight' in test_df.columns:
        test_df['weight_bin'] = pd.cut(test_df['f_1_weight'], bins=[110, 130, 150, 175, 200, 270], labels=['Fly/Ban', 'Feather/Light', 'Welter', 'Middle', 'LightHeavy/Heavy'])
        print("\n--- Accuracy by Weight Class ---")
        print(test_df.groupby('weight_bin')['is_correct'].mean())
        
    # 4.4 "Upset" Analysis
    # Where did we predict > 70% confidence and LOSE?
    bad_beats = errors[errors['confidence'] > 0.7]
    print(f"\n--- Bad Beats (High Confidence Errors > 70%) ---")
    print(f"Count: {len(bad_beats)}")
    if not bad_beats.empty:
        cols_to_show = ['event_date', 'f_1_name', 'f_2_name', 'prob_ens', 'actual_winner', 'diff_dynamic_elo']
        print(bad_beats[cols_to_show].head(10))
        
    # 4.5 Model Disagreement
    # Where did XGB and Siamese disagree?
    test_df['disagreement'] = np.abs(test_df['prob_xgb'] - test_df['prob_sia'])
    high_disagreement = test_df[test_df['disagreement'] > 0.3]
    print(f"\n--- High Model Disagreement (> 0.3 diff) ---")
    print(f"Count: {len(high_disagreement)}")
    print(f"Accuracy on these: {high_disagreement['is_correct'].mean():.4%}")
    
    # Save report
    with open('experimental/rea_report.txt', 'w') as f:
        f.write("REA Report\n")
        f.write(f"Accuracy: {acc:.4%}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write("\nTop Correlations (Easier):\n")
        f.write(corrs.head(5).to_string())
        f.write("\n\nTop Correlations (Harder):\n")
        f.write(corrs.tail(5).to_string())

if __name__ == "__main__":
    run_rea()
