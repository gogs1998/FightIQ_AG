import pandas as pd

def check_golden_outcomes():
    df = pd.read_csv('UFC_full_data_golden.csv', nrows=1)
    cols = df.columns.tolist()
    
    outcome_cols = ['result', 'outcome_method', 'method', 'finish']
    found = [c for c in cols if c in outcome_cols]
    
    print(f"Golden Outcome columns found: {found}")

if __name__ == "__main__":
    check_golden_outcomes()
