import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMatchupNet(nn.Module):
    def __init__(self, in_dim: int, ctx_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Pair features: |a-b|, a*b, a, b -> 4 * hidden
        # Plus context if any
        pair_in = 4 * hidden + (ctx_dim if ctx_dim else 0)
        
        self.head = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def encode(self, x):
        return self.enc(x)

    def pair_features(self, ea, eb, ctx=None):
        # Standard Siamese interaction features
        feats = [torch.abs(ea-eb), ea*eb, ea, eb]
        if ctx is not None:
            feats.append(ctx)
        return torch.cat(feats, dim=-1)

    def forward(self, a, b, ctx=None):
        ea, eb = self.encode(a), self.encode(b)
        h = self.pair_features(ea, eb, ctx)
        logit = self.head(h).squeeze(-1)
        return torch.sigmoid(logit)

def symmetric_loss(model, a, b, y, ctx=None, lam_sym: float = 1.0):
    """
    Loss function that enforces symmetry: P(A>B) = 1 - P(B>A)
    """
    # Forward A vs B
    p_ab = model(a, b, ctx)
    
    # Forward B vs A
    p_ba = model(b, a, ctx)
    
    # BCE Loss on A vs B (ground truth y is 1 if A wins, 0 if B wins)
    bce = F.binary_cross_entropy(p_ab, y)
    
    # Symmetry penalty: (P_ab + P_ba - 1)^2
    # Ideally P_ab + P_ba should be 1.0
    sym = ((p_ab + p_ba - 1.0)**2).mean()
    
    return bce + lam_sym * sym

class OddsDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    def forward(self, ctx, train: bool = True):
        if (not train) or ctx is None or self.p<=0:
            return ctx
        mask = (torch.rand_like(ctx) > self.p).float()
        return ctx*mask
