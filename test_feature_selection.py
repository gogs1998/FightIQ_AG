import pandas as pd

# Read header only
df = pd.read_csv('UFC_full_data_golden.csv', nrows=0)
cols = list(df.columns)

safe_keywords = ['avg', 'odds', 'age', 'height', 'reach', 'weight', 'title', 'gender', 'streak', 'wins', 'losses', 'draws']
unsafe_keywords = ['winner', 'result', 'finish', 'score', 'decision', 'bonus']

# Columns that might be tricky: 'landed', 'attempted' - only keep if they also have 'avg' or 'cum'?
# Actually, looking at the previous output, there are columns like 'f_1_sig_strikes_landed' which are definitely leakage.
# But 'f_1_avg_sig_strikes_landed' would be fine.

kept_cols = []
dropped_cols = []

for c in cols:
    c_lower = c.lower()
    
    # Must not contain obvious leakage keywords
    if any(k in c_lower for k in unsafe_keywords):
        dropped_cols.append(c)
        continue
        
    # If it looks like a raw stat, check if it's an average or cumulative
    is_stat = any(k in c_lower for k in ['landed', 'attempted', 'knockdowns', 'sub_att', 'rev', 'ctrl'])
    is_aggregated = any(k in c_lower for k in ['avg', 'cum', 'pct', 'per_min', 'diff'])
    
    if is_stat and not is_aggregated:
        dropped_cols.append(c)
        continue
        
    kept_cols.append(c)

print(f"Total Columns: {len(cols)}")
print(f"Kept Columns: {len(kept_cols)}")
print(f"Dropped Columns: {len(dropped_cols)}")

print("\n--- Sample Dropped Columns ---")
print(dropped_cols[:20])

print("\n--- Sample Kept Columns ---")
print(kept_cols[:20])
