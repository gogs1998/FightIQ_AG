import json
with open('top_features.json', 'r') as f:
    feats = json.load(f)
dobs = [f for f in feats if 'dob' in f]
print(f"DOB features: {dobs}")
