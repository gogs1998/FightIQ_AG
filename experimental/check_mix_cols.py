import pandas as pd

def check_mix_cols():
    df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
    cols = df.columns.tolist()
    
    # Check for share columns
    share_cols = [c for c in cols if 'share' in c and 'f_1' in c]
    print(f"Share columns found: {len(share_cols)}")
    if share_cols:
        print("Sample:", share_cols[:20])
        
    # Specifically look for head/body/leg
    targets = ['distance', 'clinch', 'ground']
    for t in targets:
        matches = [c for c in cols if t in c and 'share' in c and 'f_1' in c]
        print(f"{t} share cols: {len(matches)}")
        if matches:
            print(f"  Example: {matches[0]}")

if __name__ == "__main__":
    check_mix_cols()
