import pandas as pd

def check_round_cols():
    df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
    cols = df.columns.tolist()
    
    r1_cols = [c for c in cols if 'sig' in c and 'r1' in c]
    print(f"Round 1 Sig Columns found: {len(r1_cols)}")
    if r1_cols:
        print("Sample:", r1_cols[:20])
        
    # Check for specific metrics needed for PEAR
    # metrics=('sig_str_diff',), pace_col='opp_sig_str_per_min'
    # We likely have 'f_1_r1_sig_strikes_landed', 'f_2_r1_sig_strikes_landed', 'r1_time_seconds' (maybe?)
    
    time_cols = [c for c in cols if 'time' in c and 'r1' in c]
    print(f"Time columns: {time_cols}")

if __name__ == "__main__":
    check_round_cols()
