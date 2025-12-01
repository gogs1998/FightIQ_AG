import pandas as pd
import numpy as np
import joblib
import json
import torch
import random
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

# --- Configuration ---
BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'
POPULATION_SIZE = 50
GENERATIONS = 20
INITIAL_BANKROLL = 1000.0

def generate_cv_predictions(df, features, params):
    print("Generating CV Predictions for 2010-2023 (Evolution Set)...")
    
    # Filter to Evo Set
    mask_evo = df['event_date'] < '2024-01-01'
    evo_df = df[mask_evo].copy()
    X_evo = evo_df[[c for c in features if c in evo_df.columns]].fillna(0)
    y_evo = evo_df['target'].values
    
    # Placeholders for OOF predictions
    oof_xgb = np.zeros(len(evo_df))
    oof_siam = np.zeros(len(evo_df))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # XGBoost CV
    print("  Running XGBoost CV...")
    for train_idx, val_idx in kf.split(X_evo):
        X_tr, X_val = X_evo.iloc[train_idx], X_evo.iloc[val_idx]
        y_tr, y_val = y_evo[train_idx], y_evo[val_idx]
        
        clf = xgb.XGBClassifier(
            max_depth=params['xgb_max_depth'],
            learning_rate=params['xgb_learning_rate'],
            n_estimators=100, # Reduced for speed in CV
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_tr, y_tr)
        oof_xgb[val_idx] = clf.predict_proba(X_val)[:, 1]
        
    # Siamese CV
    print("  Running Siamese CV (This may take a while)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_evo)):
        print(f"    Fold {fold+1}/5...")
        X_tr, X_val = X_evo.iloc[train_idx], X_evo.iloc[val_idx]
        y_tr = y_evo[train_idx]
        
        # Prepare Data
        f1_tr, f2_tr, input_dim, _ = prepare_siamese_data(X_tr, features)
        f1_val, f2_val, _, _ = prepare_siamese_data(X_val, features)
        
        # Scale
        scaler = StandardScaler()
        combined_tr = np.concatenate([f1_tr, f2_tr], axis=0)
        scaler.fit(combined_tr)
        
        f1_tr = scaler.transform(f1_tr)
        f2_tr = scaler.transform(f2_tr)
        f1_val = scaler.transform(f1_val)
        f2_val = scaler.transform(f2_val)
        
        # Train
        model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['siamese_lr'])
        
        train_ds = TensorDataset(torch.FloatTensor(f1_tr), torch.FloatTensor(f2_tr), torch.FloatTensor(y_tr))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
        
        for epoch in range(5): # Reduced epochs for CV speed
            model.train()
            for b1, b2, by in train_loader:
                b1, b2, by = b1.to(device), b2.to(device), by.to(device)
                optimizer.zero_grad()
                loss = symmetric_loss(model, b1, b2, by)
                loss.backward()
                optimizer.step()
                
        # Predict
        model.eval()
        with torch.no_grad():
            t1 = torch.FloatTensor(f1_val).to(device)
            t2 = torch.FloatTensor(f2_val).to(device)
            probs = model(t1, t2).cpu().numpy()
            oof_siam[val_idx] = probs
            
    # Ensemble
    w = params['ensemble_xgb_weight']
    oof_ens = w * oof_xgb + (1 - w) * oof_siam
    
    evo_df['prob'] = oof_ens
    return evo_df

def load_test_predictions(df, features, params):
    print("Loading Test Predictions (2024-2025)...")
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy()
    X_test = df.loc[mask_test, [c for c in features if c in df.columns]].fillna(0)
    
    # Load Pre-trained Models
    xgb_model = joblib.load(f'{BASE_DIR}/models/xgb_optimized.pkl')
    scaler = joblib.load(f'{BASE_DIR}/models/siamese_scaler.pkl')
    
    f1_test, f2_test, input_dim, _ = prepare_siamese_data(X_test, features)
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
    siamese_model.load_state_dict(torch.load(f'{BASE_DIR}/models/siamese_optimized.pth'))
    siamese_model.eval()
    
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    with torch.no_grad():
        t1 = torch.FloatTensor(f1_test).to(device)
        t2 = torch.FloatTensor(f2_test).to(device)
        p_siam = siamese_model(t1, t2).cpu().numpy().flatten()
        
    w = params['ensemble_xgb_weight']
    p_ens = w * p_xgb + (1 - w) * p_siam
    
    test_df['prob'] = p_ens
    return test_df

# --- Genetic Algorithm ---

def evaluate(individual, df):
    min_edge, kelly_frac, min_conf, max_odds = individual
    bankroll = INITIAL_BANKROLL
    history = [bankroll]
    
    for _, row in df.iterrows():
        prob = row['prob']
        target = row['target']
        odds_1 = row['f_1_odds']
        odds_2 = row['f_2_odds']
        
        if prob > 0.5:
            my_prob = prob
            odds = odds_1
            win = (target == 1)
        else:
            my_prob = 1 - prob
            odds = odds_2
            win = (target == 0)
            
        if odds > max_odds: continue
        if my_prob < min_conf: continue
        
        implied = 1 / odds
        edge = my_prob - implied
        if edge < min_edge: continue
        
        b = odds - 1
        q = 1 - my_prob
        f = (b * my_prob - q) / b
        if f < 0: f = 0
        
        stake = bankroll * f * kelly_frac
        if stake < 5: stake = 0
        
        if stake > 0:
            if win: bankroll += stake * (odds - 1)
            else: bankroll -= stake
            
        history.append(bankroll)
        if bankroll < 10: break
        
    # Fitness: Profit * Stability
    profit = bankroll - INITIAL_BANKROLL
    
    # Drawdown Penalty
    hist_s = pd.Series(history)
    peak = hist_s.cummax()
    dd = (peak - hist_s) / peak
    max_dd = dd.max()
    
    score = profit
    if max_dd > 0.30: score *= 0.5
    if max_dd > 0.50: score *= 0.1
    
    return score,

def run_robust_optimization():
    # Load Config
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 1. Generate Evo Data (CV)
    evo_df = generate_cv_predictions(df, features, params)
    
    # 2. Optimize
    import deap.base, deap.creator, deap.tools, deap.algorithms
    
    deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
    deap.creator.create("Individual", list, fitness=deap.creator.FitnessMax)
    
    toolbox = deap.base.Toolbox()
    toolbox.register("attr_edge", random.uniform, 0.0, 0.15)
    toolbox.register("attr_kelly", random.uniform, 0.05, 0.5)
    toolbox.register("attr_conf", random.uniform, 0.5, 0.80)
    toolbox.register("attr_odds", random.uniform, 1.5, 6.0)
    
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_edge, toolbox.attr_kelly, toolbox.attr_conf, toolbox.attr_odds), n=1)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate, df=evo_df)
    toolbox.register("mate", deap.tools.cxTwoPoint)
    toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", deap.tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=POPULATION_SIZE)
    print(f"\nStarting Genetic Optimization on Evolution Set ({GENERATIONS} gens)...")
    
    deap.algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, verbose=True)
    
    best_ind = deap.tools.selBest(pop, 1)[0]
    print("\n=== Best Strategy (Evolved on 2010-2023) ===")
    print(f"Min Edge:       {best_ind[0]:.4f}")
    print(f"Kelly Fraction: {best_ind[1]:.4f}")
    print(f"Min Confidence: {best_ind[2]:.4f}")
    print(f"Max Odds:       {best_ind[3]:.2f}")
    
    # 3. Validate on Test Set (2024-2025)
    test_df = load_test_predictions(df, features, params)
    
    print("\n=== VALIDATION ON HOLDOUT (2024-2025) ===")
    score, = evaluate(best_ind, test_df) # This returns score, but we want pure profit/ROI for display
    
    # Re-run to get exact numbers
    min_edge, kelly_frac, min_conf, max_odds = best_ind
    bankroll = INITIAL_BANKROLL
    invested = 0
    
    for _, row in test_df.iterrows():
        prob = row['prob']
        target = row['target']
        odds_1, odds_2 = row['f_1_odds'], row['f_2_odds']
        
        if prob > 0.5:
            my_prob = prob
            odds = odds_1
            win = (target == 1)
        else:
            my_prob = 1 - prob
            odds = odds_2
            win = (target == 0)
            
        if odds > max_odds: continue
        if my_prob < min_conf: continue
        
        implied = 1 / odds
        edge = my_prob - implied
        if edge < min_edge: continue
        
        b = odds - 1
        q = 1 - my_prob
        f = (b * my_prob - q) / b
        if f < 0: f = 0
        
        stake = bankroll * f * kelly_frac
        if stake < 5: stake = 0
        
        if stake > 0:
            invested += stake
            if win: bankroll += stake * (odds - 1)
            else: bankroll -= stake
            
    roi = (bankroll - INITIAL_BANKROLL) / invested if invested > 0 else 0
    
    print(f"Final Bankroll: ${bankroll:,.2f}")
    print(f"Total Invested: ${invested:,.2f}")
    print(f"ROI:            {roi:.2%} 🚀")
    
    # Save
    strategy = {
        "min_edge": min_edge,
        "kelly_fraction": kelly_frac,
        "min_confidence": min_conf,
        "max_odds": max_odds,
        "holdout_roi": roi
    }
    with open(f'{BASE_DIR}/strategy_robust.json', 'w') as f:
        json.dump(strategy, f, indent=4)
    print("Saved robust strategy to strategy_robust.json")

if __name__ == "__main__":
    run_robust_optimization()
