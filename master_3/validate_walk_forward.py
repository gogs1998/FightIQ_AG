import pandas as pd
import json
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from train import train_model

def run_walk_forward():
    print("=== Master 3: Walk-Forward Validation (2020-2024) ===")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/training_data_enhanced.csv'):
        print("Error: data/training_data_enhanced.csv not found.")
        return

    df = pd.read_csv('data/training_data_enhanced.csv')
    
    # 2. Load Features
    if os.path.exists('features_selected.json'):
        with open('features_selected.json', 'r') as f:
            features = json.load(f)
    else:
        with open('features_enhanced.json', 'r') as f:
            features = json.load(f)
            
    # 3. Load Params
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
        
    if os.path.exists('params_optimized.json'):
        with open('params_optimized.json', 'r') as f:
            opt_params = json.load(f)
            params.update(opt_params)
            
    # Optimize for speed in validation loop?
    # Maybe reduce seeds to 10?
    params['n_seeds'] = 10
    print("Using 10 seeds per year for validation speed.")
    
    years = [2020, 2021, 2022, 2023, 2024]
    results = []
    
    for year in years:
        split_date = f'{year}-01-01'
        test_end_date = f'{year+1}-01-01'
        
        print(f"\n--- Validating Year: {year} ---")
        print(f"Train < {split_date} | Test: {year}")
        
        # Filter DF to only include data up to end of test year
        # We need history for training, and test year data.
        # Future data (next year) should NOT be in the dataframe at all to prevent any accidental leaks.
        df_year = df[df['event_date'] < test_end_date].copy()
        
        metrics = train_model(df_year, split_date, features, params, verbose=False)
        
        print(f"Year {year} Results: Acc={metrics['accuracy']:.4f}, ROI={metrics['roi']:.2%}, N={metrics['n_test']}")
        
        results.append({
            'Year': year,
            'Accuracy': metrics['accuracy'],
            'ROI': metrics['roi'],
            'LogLoss': metrics['log_loss'],
            'N_Fights': metrics['n_test']
        })
        
    # Summary
    print("\n=== Walk-Forward Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df)
    
    avg_acc = res_df['Accuracy'].mean()
    avg_roi = res_df['ROI'].mean()
    print(f"\nAverage Accuracy: {avg_acc:.4f}")
    print(f"Average ROI: {avg_roi:.2%}")
    
    # Save results
    res_df.to_csv('walk_forward_results.csv', index=False)

if __name__ == "__main__":
    run_walk_forward()
