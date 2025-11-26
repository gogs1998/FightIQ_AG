import pandas as pd

def check_rest_cols():
    df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
    cols = df.columns.tolist()
    
    rest_cols = ['referee', 'location', 'venue', 'commission'] # 'location' might be used if venue/commission missing
    found = [c for c in rest_cols if c in cols]
    
    print(f"REST columns found: {found}")
    
    # Check for 'finished' or 'outcome_method' to calculate priors
    outcome_cols = [c for c in cols if 'finish' in c or 'outcome' in c or 'method' in c]
    print(f"Outcome columns: {outcome_cols}")

if __name__ == "__main__":
    check_rest_cols()
