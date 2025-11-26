import joblib
import pandas as pd
import xgboost as xgb

model = joblib.load('ufc_prediction_model.pkl')
feature_names = model.get_booster().feature_names

importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 50 Features:")
print(importance.head(50))
