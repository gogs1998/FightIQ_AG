import numpy as np
from typing import Callable

def compute_afm_numeric(f: Callable[[np.ndarray,np.ndarray], float],
                        a: np.ndarray, b: np.ndarray,
                        margin: float = 0.2, eps: float = 1e-4) -> float:
    """
    Compute Adversarial Fragility Margin (AFM) numerically.
    
    f: A function that takes (fighter_a_features, fighter_b_features) and returns a win probability/score.
    a: Feature vector for fighter A.
    b: Feature vector for fighter B.
    margin: The target margin we want to check fragility against (not strictly used in simple gradient norm version but kept for API compat).
    eps: Epsilon for finite difference.
    
    Returns:
        The inverse of the gradient norm (fragility). Higher means MORE STABLE (less fragile).
        Wait, the name is 'Fragility Margin'. 
        Usually Margin = Distance to boundary.
        If gradient is high, distance is small -> Fragile.
        If gradient is low, distance is large -> Stable.
        
        Formula used here: abs(margin) / (norm(gradient) + epsilon)
        So High Gradient -> Low AFM (Fragile).
        Low Gradient -> High AFM (Stable).
    """
    # Concatenate to get full input vector x
    x = np.concatenate([a,b]).astype(float)
    
    def fwrap(xv):
        d = len(a)
        return f(xv[:d], xv[d:])
        
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy(); x1[i]+=eps
        x2 = x.copy(); x2[i]-=eps
        g[i] = (fwrap(x1)-fwrap(x2))/(2*eps)
        
    gnorm = np.linalg.norm(g)+1e-12
    return float(abs(margin)/gnorm)
