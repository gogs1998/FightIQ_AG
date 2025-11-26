import pandas as pd

def check_afm_cols():
    df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
    cols = df.columns.tolist()
    
    # Check for avg stats
    avg_cols = [c for c in cols if 'avg' in c and 'f_1' in c]
    print(f"Avg columns found: {len(avg_cols)}")
    if avg_cols:
        print("Sample:", avg_cols[:20])
        
    # Check for win pct
    win_cols = [c for c in cols if 'win' in c and 'pct' in c]
    print(f"Win Pct columns: {win_cols}")

if __name__ == "__main__":
    check_afm_cols()
