import pandas as pd

def check_golden_rest():
    df = pd.read_csv('UFC_full_data_golden.csv', nrows=1)
    cols = df.columns.tolist()
    
    rest_cols = ['referee', 'location', 'venue', 'commission', 'country', 'city']
    found = [c for c in rest_cols if c in cols]
    
    print(f"Golden REST columns found: {found}")

if __name__ == "__main__":
    check_golden_rest()
