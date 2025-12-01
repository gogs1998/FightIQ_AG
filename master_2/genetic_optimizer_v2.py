import pandas as pd
import numpy as np
import joblib
import json
import torch
import random
from deap import base, creator, tools, algorithms
from models import SiameseMatchupNet, prepare_siamese_data

# --- Configuration ---
BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'
POPULATION_SIZE = 50
GENERATIONS = 20
INITIAL_BANKROLL = 1000.0

def load_data_and_preds():
    print("Loading Data & Models...")
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter Odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Split Test Set
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy()
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    X_test = X_df[mask_test]
    y_test = df.loc[mask_test, 'target'].values
    
    # Load Models
    xgb_model = joblib.load(f'{BASE_DIR}/models/xgb_optimized.pkl')
    scaler = joblib.load(f'{BASE_DIR}/models/siamese_scaler.pkl')
    
    f1_test, f2_test, input_dim, _ = prepare_siamese_data(X_test, features)
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
    siamese_model.load_state_dict(torch.load(f'{BASE_DIR}/models/siamese_optimized.pth'))
    siamese_model.eval()
    
    # Generate Predictions
    print("Generating Ensemble Predictions...")
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    with torch.no_grad():
        t1 = torch.FloatTensor(f1_test).to(device)
        t2 = torch.FloatTensor(f2_test).to(device)
        p_siam = siamese_model(t1, t2).cpu().numpy().flatten()
        
    w = params['ensemble_xgb_weight']
    p_ens = w * p_xgb + (1 - w) * p_siam
    
    test_df['prob'] = p_ens
    test_df['target'] = y_test
    
    return test_df

# --- Genetic Algorithm ---

# Gene: [min_edge, kelly_fraction, min_conf, max_odds]
# Ranges: [0.0-0.2, 0.05-0.5, 0.5-0.8, 1.5-10.0]

def evaluate(individual, df):
    min_edge, kelly_frac, min_conf, max_odds = individual
    bankroll = INITIAL_BANKROLL
    
    for _, row in df.iterrows():
        prob = row['prob']
        target = row['target']
        odds_1 = row['f_1_odds']
        odds_2 = row['f_2_odds']
        
        # Determine Bet
        if prob > 0.5:
            my_prob = prob
            odds = odds_1
            win = (target == 1)
        else:
            my_prob = 1 - prob
            odds = odds_2
            win = (target == 0)
            
        # Filters
        if odds > max_odds: continue
        if my_prob < min_conf: continue
        
        implied = 1 / odds
        edge = my_prob - implied
        
        if edge < min_edge: continue
        
        # Kelly Criterion
        b = odds - 1
        q = 1 - my_prob
        f = (b * my_prob - q) / b
        if f < 0: f = 0
        
        stake = bankroll * f * kelly_frac
        if stake < 5: stake = 0 # Minimum bet size
        
        if stake > 0:
            if win:
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake
                
        if bankroll < 10: break # Ruin
        
    profit = bankroll - INITIAL_BANKROLL
    return profit,

def run_optimization():
    df = load_data_and_preds()
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attributes
    toolbox.register("attr_edge", random.uniform, 0.0, 0.2)
    toolbox.register("attr_kelly", random.uniform, 0.05, 0.5)
    toolbox.register("attr_conf", random.uniform, 0.5, 0.85)
    toolbox.register("attr_odds", random.uniform, 1.5, 6.0)
    
    # Structure
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_edge, toolbox.attr_kelly, toolbox.attr_conf, toolbox.attr_odds), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate, df=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=POPULATION_SIZE)
    
    print(f"\nStarting Genetic Optimization ({GENERATIONS} gens)...")
    
    # Run
    result, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, verbose=True)
    
    best_ind = tools.selBest(pop, 1)[0]
    print("\n=== Best Strategy Found ===")
    print(f"Min Edge:       {best_ind[0]:.4f}")
    print(f"Kelly Fraction: {best_ind[1]:.4f}")
    print(f"Min Confidence: {best_ind[2]:.4f}")
    print(f"Max Odds:       {best_ind[3]:.2f}")
    print(f"Profit:         ${best_ind.fitness.values[0]:.2f}")
    
    # Save Params
    strategy = {
        "min_edge": best_ind[0],
        "kelly_fraction": best_ind[1],
        "min_confidence": best_ind[2],
        "max_odds": best_ind[3]
    }
    with open(f'{BASE_DIR}/strategy_params.json', 'w') as f:
        json.dump(strategy, f, indent=4)
        
    print("Saved strategy to strategy_params.json")

if __name__ == "__main__":
    run_optimization()
