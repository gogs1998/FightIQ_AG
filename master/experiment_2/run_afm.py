import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_afm_experiment():
    print("=== Experiment: Adversarial Fragility Margin (AFM) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # Load Boruta Features (Base)
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    # Filter valid odds & time
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # 2. Train Surrogate Model (Logistic Regression)
    # We need a differentiable/simple model to compute gradients or just perturb
    # We'll use perturbation method as it's model-agnostic and robust
    
    print("Training Surrogate Model (Logistic Regression)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train on pre-2024 data to avoid leakage for the test
    mask_train = df['event_date'] < '2024-01-01'
    X_train_surr = X_scaled[mask_train]
    y_train_surr = y[mask_train]
    
    surrogate = LogisticRegression(C=1.0, max_iter=1000)
    surrogate.fit(X_train_surr, y_train_surr)
    
    # 3. Compute AFM Features
    print("Computing AFM Features (Perturbation)...")
    
    # Perturbation magnitude: 10% of standard deviation for each feature
    # Since X is scaled, std dev is 1. So perturbation is 0.1
    epsilon = 0.1 
    
    # We will perturb each feature +/- epsilon and see the max swing
    # To save time, we can just add random noise N times, or use gradient approximation.
    # Gradient approx: w * x. Change is w * epsilon.
    # For Logistic Regression: P = sigmoid(z), z = w.x + b
    # Sensitivity ~ P * (1-P) * w
    
    # Let's use the "Monte Carlo Perturbation" method for robustness
    # Generate N perturbed versions of X and measure spread
    
    n_perturbations = 50
    n_samples = len(X_scaled)
    n_features = X_scaled.shape[1]
    
    # Original Probs
    base_probs = surrogate.predict_proba(X_scaled)[:, 1]
    
    # Generate noise: shape (n_samples, n_features, n_perturbations)
    # Actually, let's just loop to save memory
    
    upsides = np.zeros(n_samples)
    downsides = np.zeros(n_samples)
    fragile_flags = np.zeros(n_samples)
    
    # We can vectorize this:
    # Create a big batch of perturbed data? No, too big.
    # Let's do it in chunks or just simple gradient heuristic for speed if linear.
    # Since it's Logistic Regression, it IS linear in log-odds space.
    # Max change in z = sum(|w_i| * epsilon_i)
    # Let's calculate the "Theoretical Max Swing" based on weights.
    
    coefs = surrogate.coef_[0] # shape (n_features,)
    # We assume feature noise is independent. 
    # Let's define "Realistic Noise" as 0.5 std dev (since data is scaled, this is 0.5)
    noise_scale = 0.5
    
    # Max possible swing in Z (logits) if all features move in worst direction by 0.5 std
    # z_swing = sum(|coef| * 0.5)
    # This is constant for all samples? No, that's the potential swing.
    # But we want local sensitivity.
    
    # Let's stick to the prompt's definition: "Simulate perturbations"
    # We'll add random noise and see what happens.
    
    print(f"Simulating {n_perturbations} perturbations per fight...")
    
    min_probs = base_probs.copy()
    max_probs = base_probs.copy()
    
    for i in range(n_perturbations):
        noise = np.random.normal(0, 0.2, X_scaled.shape) # 0.2 std dev noise
        X_noisy = X_scaled + noise
        probs_noisy = surrogate.predict_proba(X_noisy)[:, 1]
        
        min_probs = np.minimum(min_probs, probs_noisy)
        max_probs = np.maximum(max_probs, probs_noisy)
        
    # Calculate Metrics
    afm_upside = max_probs - base_probs
    afm_downside = base_probs - min_probs
    afm_skew = afm_upside - afm_downside
    
    # Fragile: Did it cross 0.5?
    # If (base < 0.5 and max > 0.5) OR (base > 0.5 and min < 0.5)
    afm_fragile = ((base_probs < 0.5) & (max_probs > 0.5)) | \
                  ((base_probs > 0.5) & (min_probs < 0.5))
    afm_fragile = afm_fragile.astype(int)
    
    print("AFM Features Computed.")
    
    # 4. Add to DataFrame
    df['afm_upside'] = afm_upside
    df['afm_downside'] = afm_downside
    df['afm_skew'] = afm_skew
    df['afm_fragile'] = afm_fragile
    
    # 5. Train XGBoost with AFM
    print("\nTraining XGBoost with AFM Features...")
    
    afm_features = ['afm_upside', 'afm_downside', 'afm_skew', 'afm_fragile']
    all_features = features + afm_features
    
    X_final = df[all_features].fillna(0)
    
    # Split
    mask_train = df['event_date'] < '2024-01-01'
    mask_test = df['event_date'] >= '2024-01-01'
    
    X_train = X_final[mask_train]
    X_test = X_final[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    # Train
    model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n=== AFM Experiment Results (2024-2025) ===")
    print(f"Accuracy with AFM: {acc:.4%}")
    print(f"Baseline (Boruta): 70.36%")
    print(f"Impact:            {acc - 0.7036:+.4%}")
    
    # Feature Importance
    imp = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(imp.head(10))
    
    # Check if AFM is in top 10
    afm_rank = imp[imp['Feature'].isin(afm_features)]
    print("\nAFM Feature Ranks:")
    print(afm_rank)

if __name__ == "__main__":
    run_afm_experiment()
