import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

def stress_test_integrity():
    print("=== FIGHTIQ INTEGRITY STRESS TEST ===")
    print("Objective: Prove the model's profitability is due to REAL SIGNAL, not luck or bugs.")
    
    # 1. Load Real Data (2024-2025 Test Set)
    print("\n[1] Loading Test Data (2024-2025)...")
    
    try:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'].dt.year >= 2024].copy()
    
    # Load Model
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    iso_xgb = joblib.load('models/iso_xgb.pkl')
    # We will focus on XGB for this test as it's the core driver and easier to batch predict
    # The Siamese adds value but the XGB is the "base" truth.
    
    # Prepare Features
    import json
    with open('features.json', 'r') as f: feature_names = json.load(f)
    
    X_test = test_df[feature_names]
    y_test = test_df['target']
    odds_1 = test_df['f_1_odds']
    odds_2 = test_df['f_2_odds']
    
    # Generate Real Predictions
    print("Generating baseline predictions...")
    probs_raw = xgb_model.predict_proba(X_test)[:, 1]
    probs_calib = iso_xgb.predict(probs_raw)
    
    # Strategy Function
    def calculate_roi(probabilities, winners, o1, o2):
        profit = 0
        wagered = 0
        
        for i in range(len(probabilities)):
            prob = probabilities[i]
            winner = winners.iloc[i]
            odd1 = o1.iloc[i]
            odd2 = o2.iloc[i]
            
            # Golden Rule Logic
            if prob > 0.5:
                my_prob = prob
                my_odd = odd1
                target = 1
            else:
                my_prob = 1 - prob
                my_odd = odd2
                target = 0
                
            if my_prob < 0.60: continue # Confidence Cutoff
            if my_odd > 5.0: continue   # Odds Cap
            
            edge = my_prob - (1/my_odd)
            if edge <= 0: continue      # Positive Edge Only
            
            # Kelly Stake (Simplified for speed: 1 unit)
            # We care about ROI, so flat staking is a fine proxy for "signal quality"
            stake = 100 
            wagered += stake
            
            if target == winner:
                profit += stake * (my_odd - 1)
            else:
                profit -= stake
                
        return (profit / wagered) * 100 if wagered > 0 else 0
        
    # 2. Baseline ROI
    real_roi = calculate_roi(probs_calib, y_test, odds_1, odds_2)
    print(f"\n[2] BASELINE (REAL) ROI: {real_roi:.2f}%")
    print("    (This is what we claim the model achieves)")
    
    # 3. The "Monkey" Test (Random Winners)
    print("\n[3] THE 'MONKEY' TEST (Permutation Test)")
    print("    Hypothesis: If we shuffle the fight results, the ROI should drop to -5% (Vig).")
    print("    Running 100 simulations...")
    
    monkey_rois = []
    for _ in tqdm(range(100)):
        # We must align the random winners with the index to ensure iloc works correctly in loop
        random_winners = pd.Series(np.random.randint(0, 2, size=len(y_test)), index=y_test.index)
        r_roi = calculate_roi(probs_calib, random_winners, odds_1, odds_2)
        monkey_rois.append(r_roi)
        
    avg_monkey = np.mean(monkey_rois)
    print(f"    -> Average 'Monkey' ROI: {avg_monkey:.2f}%")
    if real_roi > max(monkey_rois):
        print("    ✅ PASSED: Real ROI is statistically impossible by chance.")
    else:
        print("    ❌ FAILED: Random luck could explain these results.")
        
    # 4. The "Blind" Test (Feature Shuffle)
    print("\n[4] THE 'BLIND' TEST (Feature Destruction)")
    print("    Hypothesis: If we destroy the stats (Elo, Age), the model should fail.")
    print("    Shuffling feature columns and re-predicting...")
    
    # Shuffle X values column-wise to destroy correlations
    X_shuffled = X_test.copy()
    for col in X_shuffled.columns:
        X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
        
    probs_blind = xgb_model.predict_proba(X_shuffled)[:, 1]
    probs_blind_calib = iso_xgb.predict(probs_blind)
    
    blind_roi = calculate_roi(probs_blind_calib, y_test, odds_1, odds_2)
    print(f"    -> 'Blind' Model ROI: {blind_roi:.2f}%")
    
    if blind_roi < 0:
        print("    ✅ PASSED: Without real stats, the model loses money.")
    else:
        print("    ⚠️ WARNING: Model might be getting lucky or relying on artifacts.")

    print("\n" + "="*40)
    print("FINAL VERDICT")
    print("="*40)
    if real_roi > 10 and avg_monkey < 0 and blind_roi < 0:
        print("🏆 INTEGRITY CONFIRMED")
        print("1. The profit disappears if you guess randomly.")
        print("2. The profit disappears if you remove the fighter stats.")
        print("3. Therefore, the profit comes from REAL SIGNAL in the fighter stats.")
    else:
        print("⚠️ INTEGRITY CHECK FAILED")

if __name__ == "__main__":
    stress_test_integrity()
