import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

def profit_objective(y_true, y_pred):
    """
    Custom objective function for profit maximization.
    (Simplified proxy: We want to penalize errors on high-odds fights more)
    But XGBoost custom obj requires gradient/hessian.
    For simplicity in v2 MVP, we will use sample_weight instead.
    """
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess

def kelly_criterion(prob, odds, fractional=0.25):
    if odds <= 1: return 0.0
    b = odds - 1
    q = 1 - prob
    p = prob
    f_star = (b * p - q) / b
    if f_star <= 0: return 0.0
    return f_star * fractional

class GamblerModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=3, # Shallower trees to avoid overfitting noise
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.features = []
        
    def preprocess(self, df, is_train=True):
        # Strict Feature Selection
        safe_cols = []
        for c in df.columns:
            # 1. Experimental Features
            if c.startswith('dynamic_elo') or c.startswith('common') or c.startswith('diff') or \
               c.endswith('_finish_rate') or c.endswith('_been_finished_rate') or c.endswith('_avg_time'):
                safe_cols.append(c)
                continue
            # 2. Rolling Stats
            if any(x in c for x in ['_3_f_', '_5_f_', 'streak']):
                safe_cols.append(c)
                continue
            # 3. Physical / Static
            if 'height' in c or 'reach' in c or 'age' in c or 'dob' in c or 'weight' in c:
                safe_cols.append(c)
                continue
            # 4. Odds & Rankings
            if 'odds' in c or 'ranking' in c:
                safe_cols.append(c)
                continue
                
        features = [c for c in safe_cols if df[c].dtype in [np.float64, np.int64, np.int32]]
        
        if is_train:
            self.features = features
            print(f"Selected {len(self.features)} safe features.")
        return df[self.features]

    def fit(self, df):
        print("Training Gambler Model...")
        X = self.preprocess(df, is_train=True)
        y = df['target']
        
        # Calculate Sample Weights based on Potential Profit
        # If F1 wins, profit is (Odds1 - 1). If F2 wins, profit is (Odds2 - 1).
        # We weight the training samples by the potential return.
        # This forces the model to care more about high-value fights.
        
        # Assuming odds columns exist: 'f_1_odds', 'f_2_odds' (Decimal)
        # If not, we default to weight=1
        
        weights = np.ones(len(df))
        if 'f_1_odds' in df.columns and 'f_2_odds' in df.columns:
            # Potential profit if we bet on the winner
            # If target=1 (F1 wins), weight = f_1_odds - 1
            # If target=0 (F2 wins), weight = f_2_odds - 1
            
            # Handle American odds conversion if needed, but assuming decimal from golden
            # Golden data usually has 'f_1_odds' as American? Let's check.
            # If American: +150 -> 2.5. -150 -> 1.67.
            
            # For safety, let's just use a simple heuristic:
            # Weight = 1.0 + log(Variance)
            pass
            
        self.model.fit(X, y) # , sample_weight=weights) -> Add back if odds are clean
        
    def predict(self, df):
        X = self.preprocess(df, is_train=False)
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def recommend_bets(self, df, probs, bankroll=1000):
        """
        Generate betting recommendations using Kelly Criterion.
        """
        recommendations = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            p_f1 = probs[i]
            p_f2 = 1 - p_f1
            
            # Get Odds (Decimal)
            # Placeholder: Need to ensure odds are in df
            odds_f1 = row.get('f_1_odds', 2.0) 
            odds_f2 = row.get('f_2_odds', 2.0)
            
            # Convert American to Decimal if needed
            def to_decimal(o):
                if o > 0: return 1 + o/100
                else: return 1 + 100/abs(o)
                
            if odds_f1 > 100 or odds_f1 < -100: odds_f1 = to_decimal(odds_f1)
            if odds_f2 > 100 or odds_f2 < -100: odds_f2 = to_decimal(odds_f2)
            
            # Kelly
            k1 = kelly_criterion(p_f1, odds_f1)
            k2 = kelly_criterion(p_f2, odds_f2)
            
            if k1 > 0:
                wager = bankroll * k1
                recommendations.append({
                    'fighter': row['f_1_name'],
                    'wager': wager,
                    'pct': k1 * 100,
                    'odds': odds_f1,
                    'ev': (p_f1 * odds_f1) - 1
                })
            elif k2 > 0:
                wager = bankroll * k2
                recommendations.append({
                    'fighter': row['f_2_name'],
                    'wager': wager,
                    'pct': k2 * 100,
                    'odds': odds_f2,
                    'ev': (p_f2 * odds_f2) - 1
                })
            else:
                recommendations.append({'fighter': 'No Bet', 'wager': 0})
                
        return recommendations

    def save(self, path='v2/models/gambler_model.pkl'):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path='v2/models/gambler_model.pkl'):
        return joblib.load(path)
