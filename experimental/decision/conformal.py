import numpy as np

class ConformalClassifier:
    def __init__(self, alpha=0.1):
        """
        Split Conformal Prediction for Classification.
        alpha: Error rate (e.g., 0.1 for 90% coverage).
        """
        self.alpha = alpha
        self.q = None
        
    def fit(self, scores, y):
        """
        Calibrate the quantile q using a calibration set.
        scores: (n, n_classes) array of softmax probabilities.
        y: (n,) array of true class indices (0 or 1).
        """
        n = len(y)
        
        # 1. Calculate conformity scores for the true class
        # Score s_i = 1 - P(y_i | x_i)
        # We want small scores to be "good" (conform), large scores "bad" (non-conform).
        # Wait, standard is non-conformity score.
        # s_i = 1 - prob_of_true_class
        
        # Extract prob of true class
        # y must be int indices
        y = y.astype(int)
        prob_true = scores[np.arange(n), y]
        
        non_conformity_scores = 1 - prob_true
        
        # 2. Calculate quantile
        # We want to cover 1-alpha of the distribution.
        # q = quantile of non_conformity_scores at level (1 - alpha) * (1 + 1/n)
        # Actually, standard definition:
        # k = ceil((n+1)(1-alpha))
        # q = k-th smallest value
        
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(k, n) # Clamp
        
        sorted_scores = np.sort(non_conformity_scores)
        self.q = sorted_scores[k-1]
        
        print(f"Conformal Calibration: n={n}, alpha={self.alpha}, q={self.q:.4f}")
        
    def predict(self, scores):
        """
        Generate prediction sets for new data.
        scores: (m, n_classes) array of probabilities.
        Returns: (m, n_classes) boolean mask. True if class is in set.
        """
        if self.q is None:
            raise ValueError("Model not calibrated. Call fit() first.")
            
        # Include class k if 1 - P(k|x) <= q
        # i.e., P(k|x) >= 1 - q
        
        # Note: This is a simple marginal coverage guarantee.
        # For adaptive/conditional, we'd need APS (Adaptive Prediction Sets).
        # But this is a good start.
        
        threshold = 1 - self.q
        prediction_sets = scores >= threshold
        
        return prediction_sets
