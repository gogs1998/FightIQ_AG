import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

def fight_analyst(query_type, **kwargs):
    print(f"=== FightIQ Analyst: {query_type} ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # 2. Handle Query Types
    
    if query_type == "stance_win_rate":
        # "Show me win rate of Southpaw vs Orthodox"
        print("Analyzing Stance Matchups...")
        
        # Filter for valid stances
        valid_stances = ['Orthodox', 'Southpaw', 'Switch']
        df_clean = df[df['f_1_fighter_stance'].isin(valid_stances) & df['f_2_fighter_stance'].isin(valid_stances)].copy()
        
        # Create 'Matchup' column (e.g., "Orthodox vs Southpaw")
        # Ensure consistent ordering for grouping (e.g. always alphabetical)
        # Actually, we want to know win rate of F1 given F1 stance vs F2 stance
        
        # Let's pivot: Win Rate of Stance A against Stance B
        results = []
        
        for s1 in valid_stances:
            for s2 in valid_stances:
                mask = (df_clean['f_1_fighter_stance'] == s1) & (df_clean['f_2_fighter_stance'] == s2)
                fights = df_clean[mask]
                if len(fights) > 50:
                    wins = fights['target'].sum() # Target 1 means F1 wins
                    rate = wins / len(fights)
                    results.append({
                        "Fighter Stance": s1,
                        "Opponent Stance": s2,
                        "Win Rate": rate,
                        "Fights": len(fights)
                    })
                    
        res_df = pd.DataFrame(results)
        
        # Plot Heatmap
        pivot = res_df.pivot(index="Fighter Stance", columns="Opponent Stance", values="Win Rate")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0.5)
        plt.title("Win Rate by Stance Matchup (Historical)")
        plt.savefig(f'{BASE_DIR}/experiment_2/viz_stance_heatmap.png')
        print("Saved viz_stance_heatmap.png")
        
    elif query_type == "elo_trajectory":
        # "Plot Elo of Fighter X"
        fighter_name = kwargs.get('fighter_name', 'Conor McGregor')
        print(f"Plotting Elo for {fighter_name}...")
        
        # Find all fights for this fighter
        mask1 = df['f_1_name'] == fighter_name
        mask2 = df['f_2_name'] == fighter_name
        
        fights = df[mask1 | mask2].sort_values('event_date').copy()
        
        dates = []
        elos = []
        
        for idx, row in fights.iterrows():
            dates.append(row['event_date'])
            if row['f_1_name'] == fighter_name:
                elos.append(row['f_1_elo'])
            else:
                elos.append(row['f_2_elo'])
                
        plt.figure(figsize=(10, 5))
        plt.plot(dates, elos, marker='o', linewidth=2, color='gold')
        plt.title(f"Elo Rating Trajectory: {fighter_name}")
        plt.xlabel("Year")
        plt.ylabel("Elo Rating")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{BASE_DIR}/experiment_2/viz_elo_{fighter_name.replace(" ", "_")}.png')
        print(f"Saved viz_elo_{fighter_name.replace(' ', '_')}.png")

    elif query_type == "finish_rate_by_weight":
        # "Finish rate by weight class"
        print("Analyzing Finish Rates by Weight Class...")
        
        # Clean weight classes
        def clean_weight(w):
            if pd.isna(w): return "Unknown"
            w = str(w).lower()
            if 'heavyweight' in w and 'light' not in w: return 'Heavyweight'
            if 'light heavyweight' in w: return 'Light Heavyweight'
            if 'middleweight' in w: return 'Middleweight'
            if 'welterweight' in w: return 'Welterweight'
            if 'lightweight' in w: return 'Lightweight'
            if 'featherweight' in w: return 'Featherweight'
            if 'bantamweight' in w: return 'Bantamweight'
            if 'flyweight' in w: return 'Flyweight'
            if 'strawweight' in w: return 'Strawweight'
            return "Catch/Other"
            
        df['weight_class_clean'] = df['weight_class'].apply(clean_weight)
        
        # Define Finish
        def is_finish(res):
            r = str(res).lower()
            return 'decision' not in r
            
        df['is_finish'] = df['result'].apply(is_finish)
        
        # Group
        order = ['Heavyweight', 'Light Heavyweight', 'Middleweight', 'Welterweight', 
                 'Lightweight', 'Featherweight', 'Bantamweight', 'Flyweight', 'Strawweight']
                 
        stats = df.groupby('weight_class_clean')['is_finish'].mean().reset_index()
        stats = stats[stats['weight_class_clean'].isin(order)]
        
        # Sort by order
        stats['weight_class_clean'] = pd.Categorical(stats['weight_class_clean'], categories=order, ordered=True)
        stats = stats.sort_values('weight_class_clean')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=stats, x='weight_class_clean', y='is_finish', palette='viridis')
        plt.title("Finish Rate by Weight Class (All Time)")
        plt.ylabel("Finish Probability")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        
        # Add labels
        for i, v in enumerate(stats['is_finish']):
            plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
            
        plt.tight_layout()
        plt.savefig(f'{BASE_DIR}/experiment_2/viz_finish_by_weight.png')
        print("Saved viz_finish_by_weight.png")

if __name__ == "__main__":
    # Simple CLI for testing
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        arg = sys.argv[2] if len(sys.argv) > 2 else None
        fight_analyst(cmd, fighter_name=arg)
    else:
        # Default Test Run
        fight_analyst("stance_win_rate")
        fight_analyst("finish_rate_by_weight")
        fight_analyst("elo_trajectory", fighter_name="Jon Jones")
