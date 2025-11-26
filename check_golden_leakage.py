import pandas as pd

# Load a small chunk of Golden
df = pd.read_csv('UFC_full_data_golden.csv', nrows=5)

print("Columns in Golden:")
cols = list(df.columns)

# Check for target
if 'winner' in cols:
    print("Target 'winner' FOUND in Golden.")
else:
    print("Target 'winner' NOT FOUND in Golden.")

# Check for potential leakage columns
# These look like stats from the actual fight happening
leak_candidates = [
    'f_1_sig_strikes_landed', 
    'f_2_sig_strikes_landed', 
    'total_strikes', 
    'knockdowns',
    'finish_round'
]

print("\nChecking for potential leakage columns (exact matches or substrings):")
for cand in leak_candidates:
    matches = [c for c in cols if cand in c]
    if matches:
        print(f"Found potential leakage related to '{cand}': {matches[:5]}...")
    else:
        print(f"No direct matches for '{cand}'")

# Check for 'avg' or 'pre_fight' features
avg_cols = [c for c in cols if 'avg' in c.lower()]
print(f"\nNumber of 'avg' columns: {len(avg_cols)}")
print(f"Examples: {avg_cols[:5]}")
