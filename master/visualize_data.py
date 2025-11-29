import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations():
    print("=== FightIQ: Generating Data Visualizations ===")
    
    # 1. Setup
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    VIZ_DIR = f'{BASE_DIR}/visualizations'
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Set Style (Premium Dark Mode)
    plt.style.use('dark_background')
    sns.set_palette("viridis")
    
    # 2. Load Data
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    print(f"Loaded {len(df)} fights.")
    
    # 3. Viz 1: The "Age Curve" (Win Rate by Age Diff)
    # Bin age difference
    df['age_diff_bin'] = pd.cut(df['diff_age'], bins=np.arange(-15, 16, 3))
    age_win_rate = df.groupby('age_diff_bin')['target'].mean()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=age_win_rate.index, y=age_win_rate.values, palette="coolwarm")
    plt.title("The Youth Advantage: Win Rate by Age Difference", fontsize=16, color='white')
    plt.xlabel("Age Difference (Fighter A - Fighter B)", fontsize=12)
    plt.ylabel("Win Probability", fontsize=12)
    plt.axhline(0.5, color='white', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/viz_age_curve.png')
    print("Saved viz_age_curve.png")
    
    # 4. Viz 2: Elo Predictive Power (Sigmoid Check)
    # Bin Elo difference
    df['elo_diff_bin'] = pd.cut(df['diff_elo'], bins=np.arange(-400, 401, 50))
    elo_win_rate = df.groupby('elo_diff_bin')['target'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.arange(len(elo_win_rate)), y=elo_win_rate.values, s=100, color='#00ffcc')
    sns.lineplot(x=np.arange(len(elo_win_rate)), y=elo_win_rate.values, color='#00ffcc', alpha=0.5)
    plt.title("Elo Rating Validity: Win Rate vs Elo Difference", fontsize=16, color='white')
    plt.xlabel("Elo Difference Bins", fontsize=12)
    plt.ylabel("Actual Win Rate", fontsize=12)
    plt.axhline(0.5, color='white', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/viz_elo_validity.png')
    print("Saved viz_elo_validity.png")
    
    # 5. Viz 3: Finish Rate by Weight Class
    # Need to extract weight class if not explicit, but usually it is.
    # Let's assume 'weight_class' column exists or infer from weight.
    # If not, we skip.
    if 'weight_class' in df.columns:
        df['is_finish'] = df['result'].apply(lambda x: 1 if 'Decision' not in str(x) else 0)
        finish_rate = df.groupby('weight_class')['is_finish'].mean().sort_values()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=finish_rate.index, y=finish_rate.values, palette="magma")
        plt.title("Violence Factor: Finish Rate by Weight Class", fontsize=16, color='white')
        plt.xticks(rotation=45)
        plt.ylabel("Finish Probability", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/viz_finish_rates.png')
        print("Saved viz_finish_rates.png")
        
    # 6. Viz 4: Stance Matchup Heatmap
    # Orthodox vs Southpaw etc.
    if 'f_1_stance' in df.columns and 'f_2_stance' in df.columns:
        # Filter for main stances
        stances = ['Orthodox', 'Southpaw', 'Switch']
        mask = df['f_1_stance'].isin(stances) & df['f_2_stance'].isin(stances)
        sub = df[mask]
        
        pivot = sub.pivot_table(index='f_1_stance', columns='f_2_stance', values='target', aggfunc='mean')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0.5, cbar=False)
        plt.title("Stance Advantage Matrix (Win Rate)", fontsize=16, color='white')
        plt.xlabel("Opponent Stance")
        plt.ylabel("Fighter Stance")
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/viz_stance_matrix.png')
        print("Saved viz_stance_matrix.png")

if __name__ == "__main__":
    generate_visualizations()
