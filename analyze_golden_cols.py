import pandas as pd

# Read header only
df = pd.read_csv('UFC_full_data_golden.csv', nrows=0)
cols = list(df.columns)

print(f"Total columns: {len(cols)}")

# Keywords to look for
keywords = ['avg', 'cum', 'last', 'diff', 'odds', 'age', 'height', 'reach']
for k in keywords:
    matches = [c for c in cols if k in c.lower()]
    print(f"Columns containing '{k}': {len(matches)}")
    if len(matches) > 0:
        print(f"Examples: {matches[:5]}")

# Check for potential target leaks (stats that happen IN the fight)
leak_keywords = ['landed', 'attempted', 'knockdowns', 'submission']
print("\n--- Potential Leakage Checks (Stats that might be from the current fight) ---")
for k in leak_keywords:
    matches = [c for c in cols if k in c.lower()]
    # We want to see if there are columns that look like current fight stats vs historical
    # e.g. "total_strikes_landed" (leak) vs "avg_total_strikes_landed" (feature)
    print(f"Columns containing '{k}': {len(matches)}")
    print(f"Examples: {matches[:5]}")
