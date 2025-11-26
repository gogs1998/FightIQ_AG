import pandas as pd

def find_proxy_cols():
    df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
    cols = df.columns.tolist()
    
    # Striking
    str_cols = [c for c in cols if ('slpm' in c or 'sig_str' in c) and 'f_1' in c]
    print(f"Striking columns: {str_cols[:10]}")
    
    # Win Pct - maybe it's 'win_rate' or just 'wins'/'losses'
    win_cols = [c for c in cols if ('win' in c or 'loss' in c) and 'f_1' in c]
    print(f"Win/Loss columns: {win_cols[:10]}")

if __name__ == "__main__":
    find_proxy_cols()
