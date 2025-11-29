import pandas as pd
import numpy as np
import joblib
import json
import random
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

# Config
POPULATION_SIZE = 100
GENERATIONS = 30
MUTATION_RATE = 0.1
ELITISM = 5

def genetic_optimizer():
    print("=== FightIQ Genetic Strategy Optimizer (Leakage-Free) ===")
    print("Objective: Evolve on 2010-2023, Test on 2024-2025.")
    
    # 1. Load Data & Generate Unbiased Predictions (CV)
    print("Generating unbiased CV predictions...")
    
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    o1 = df['f_1_odds'].values
    o2 = df['f_2_odds'].values
    
    # Train/Predict via CV (XGBoost only for speed)
    model = xgb.XGBClassifier(
        max_depth=params['xgb_max_depth'],
        learning_rate=params['xgb_learning_rate'],
        n_estimators=100, 
        n_jobs=-1,
        random_state=42
    )
    
    probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
    
    # Split into Evolution Set (< 2024) and Test Set (>= 2024)
    dates = pd.to_datetime(df['event_date'])
    
    mask_evo = dates < '2024-01-01'
    mask_test = dates >= '2024-01-01'
    
    # Evolution Data
    probs_evo = probs[mask_evo]
    y_evo = y[mask_evo]
    o1_evo = o1[mask_evo]
    o2_evo = o2[mask_evo]
    
    # Test Data
    probs_test = probs[mask_test]
    y_test = y[mask_test]
    o1_test = o1[mask_test]
    o2_test = o2[mask_test]
    
    print(f"Evolution Set: {len(probs_evo)} fights (2010-2023)")
    print(f"Test Set:      {len(probs_test)} fights (2024-2025)")
    
    # 2. Define Genome
    def create_genome():
        return [
            random.uniform(0.50, 0.80), # Conf
            random.uniform(0.05, 0.50), # Kelly
            random.uniform(2.0, 10.0),  # Max Odds
            random.uniform(0.0, 0.10)   # Min Edge
        ]
        
    def run_simulation(genome, p_arr, y_arr, o1_arr, o2_arr):
        conf_thresh, kelly_frac, max_odds, min_edge = genome
        bankroll = 1000.0
        history = [1000.0]
        MAX_BET_PCT = 0.20
        
        for i in range(len(p_arr)):
            p = p_arr[i]
            
            if p > 0.5:
                my_prob = p
                odds = o1_arr[i]
                winner = (y_arr[i] == 1)
            else:
                my_prob = 1 - p
                odds = o2_arr[i]
                winner = (y_arr[i] == 0)
                
            # Filters
            if my_prob < conf_thresh: continue
            if odds > max_odds: continue
            
            edge = my_prob - (1/odds)
            if edge < min_edge: continue
            
            # Kelly
            b = odds - 1
            q = 1 - my_prob
            f = (b * my_prob - q) / b
            stake_pct = f * kelly_frac
            
            if stake_pct <= 0: continue
            
            # Sizing
            stake = bankroll * stake_pct
            if stake > bankroll * MAX_BET_PCT: stake = bankroll * MAX_BET_PCT
            if stake < 5: stake = 5
            
            if winner:
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake
                
            history.append(bankroll)
            if bankroll < 10: break
            
        return bankroll, history

    def fitness(genome):
        # Run on Evolution Set ONLY
        final_bank, hist = run_simulation(genome, probs_evo, y_evo, o1_evo, o2_evo)
        
        # Fitness Logic
        peak = pd.Series(hist).cummax()
        dd = (peak - pd.Series(hist)) / peak
        max_dd = dd.max()
        
        score = final_bank
        if max_dd > 0.30: score *= 0.5
        if max_dd > 0.50: score *= 0.1
        if max_dd > 0.80: score *= 0.01
        
        return score
        
    # 3. Evolution Loop
    population = [create_genome() for _ in range(POPULATION_SIZE)]
    
    print(f"Evolving over {GENERATIONS} generations...")
    
    for gen in range(GENERATIONS):
        scores = [(genome, fitness(genome)) for genome in population]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_genome, best_score = scores[0]
        
        if gen % 5 == 0:
            print(f"Gen {gen}: Evo Score ${best_score:,.2f} | Params: Conf={best_genome[0]:.2f}, Kelly={best_genome[1]:.2f}, MaxOdds={best_genome[2]:.2f}, Edge={best_genome[3]:.3f}")
            
        survivors = [s[0] for s in scores[:POPULATION_SIZE//2]]
        new_pop = survivors[:ELITISM]
        
        while len(new_pop) < POPULATION_SIZE:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            child = []
            for i in range(4):
                if random.random() < 0.5: child.append(p1[i])
                else: child.append(p2[i])
                
            if random.random() < MUTATION_RATE:
                gene_idx = random.randint(0, 3)
                if gene_idx == 0: child[0] = random.uniform(0.50, 0.80)
                elif gene_idx == 1: child[1] = random.uniform(0.05, 0.50)
                elif gene_idx == 2: child[2] = random.uniform(2.0, 10.0)
                elif gene_idx == 3: child[3] = random.uniform(0.0, 0.10)
                
            new_pop.append(child)
            
        population = new_pop
        
    print("\n=== EVOLUTION COMPLETE ===")
    final_scores = [(g, fitness(g)) for g in population]
    final_scores.sort(key=lambda x: x[1], reverse=True)
    best_genome, best_score = final_scores[0]
    
    print(f"🏆 BEST STRATEGY FOUND (Evo Score: ${best_score:,.2f})")
    print(f"   • Confidence Cutoff: {best_genome[0]:.1%}")
    print(f"   • Kelly Fraction:    {best_genome[1]:.2f}")
    print(f"   • Max Odds Cap:      {best_genome[2]:.2f}")
    print(f"   • Min Edge:          {best_genome[3]:.1%}")
    
    # 4. FINAL TEST (The Moment of Truth)
    print("\n=== VALIDATION TEST (2024-2025) ===")
    test_bank, test_hist = run_simulation(best_genome, probs_test, y_test, o1_test, o2_test)
    
    roi = (test_bank - 1000) / 1000
    print(f"Final Test Bankroll: ${test_bank:,.2f}")
    print(f"Test ROI:            {roi:.2%}")
    
    if test_bank > 1000:
        print("✅ STRATEGY GENERALIZES!")
    else:
        print("❌ STRATEGY OVERFITTED.")
    
    strategy = {
        "conf_threshold": best_genome[0],
        "kelly_fraction": best_genome[1],
        "max_odds": best_genome[2],
        "min_edge": best_genome[3],
        "test_roi": roi
    }
    with open(f'{BASE_DIR}/experiment_2/best_genetic_strategy.json', 'w') as f:
        json.dump(strategy, f, indent=4)
    print("Saved to experiment_2/best_genetic_strategy.json")

if __name__ == "__main__":
    genetic_optimizer()
