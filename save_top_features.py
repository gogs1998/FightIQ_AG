import joblib
import pandas as pd
import json

model = joblib.load('ufc_prediction_model.pkl')
feature_names = model.get_booster().feature_names

importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top 300 features
top_features = importance.head(300)['feature'].tolist()

print(f"Saving {len(top_features)} features to top_features.json")
with open('top_features.json', 'w') as f:
    json.dump(top_features, f)

# Also check for missing values in these features in the original data
df = pd.read_csv('UFC_full_data_golden.csv')
print("\nMissing values in top 10 features:")
print(df[top_features[:10]].isnull().sum())
