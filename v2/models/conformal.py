import numpy as np

class ConformalClassifier:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None
        
    def fit(self, scores, y):
        n = len(y)
        prob_true = scores[np.arange(n), y]
        non_conformity_scores = 1 - prob_true
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(k, n) 
        sorted_scores = np.sort(non_conformity_scores)
        self.q = sorted_scores[k-1]
        
    def predict(self, scores):
        if self.q is None:
            raise ValueError("Model not calibrated. Call fit() first.")
        threshold = 1 - self.q
        prediction_sets = scores >= threshold
        return prediction_sets
