import json
import pandas as pd

with open('top_features.json', 'r') as f:
    features = json.load(f)

print(f"Total features: {len(features)}")

match_specific = []
fighter_specific = []
derived = []

for feat in features:
    if 'diff_' in feat:
        derived.append(feat)
    elif 'odds' in feat:
        match_specific.append(feat)
    elif feat.startswith('f_1_') or feat.startswith('f_2_'):
        fighter_specific.append(feat)
    elif feat.endswith('_f_1') or feat.endswith('_f_2'):
        fighter_specific.append(feat)
    else:
        match_specific.append(feat)

print(f"Match Specific (Odds, etc): {len(match_specific)}")
print(f"Derived (Diffs): {len(derived)}")
print(f"Fighter Specific (f_1_..., f_2_...): {len(fighter_specific)}")

print("\n--- Sample Fighter Specific ---")
print(fighter_specific[:10])

print("\n--- Sample Derived ---")
print(derived[:10])

# We need to map f_1_X to a generic feature X
# e.g. f_1_fighter_reach_cm -> fighter_reach_cm
generic_features = set()
for f in fighter_specific:
    if f.startswith('f_1_'):
        generic_features.add(f[4:])
    elif f.startswith('f_2_'):
        generic_features.add(f[4:])
    elif f.endswith('_f_1'):
        generic_features.add(f[:-4])
    elif f.endswith('_f_2'):
        generic_features.add(f[:-4])

print(f"\nUnique Generic Fighter Features needed: {len(generic_features)}")
print(list(generic_features)[:10])
