import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- PyTorch Architecture (Pure Siamese) ---
class SiameseMatchupNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Pair features: |a-b|, a*b, a, b -> 4 * hidden
        pair_in = 4 * hidden
        
        self.head = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def encode(self, x):
        return self.enc(x)

    def pair_features(self, ea, eb):
        feats = [torch.abs(ea-eb), ea*eb, ea, eb]
        return torch.cat(feats, dim=-1)

    def forward(self, a, b):
        ea, eb = self.encode(a), self.encode(b)
        h = self.pair_features(ea, eb)
        logit = self.head(h).squeeze(-1)
        return torch.sigmoid(logit)

def symmetric_loss(model, a, b, y, lam_sym: float = 1.0):
    p_ab = model(a, b)
    p_ba = model(b, a)
    bce = F.binary_cross_entropy(p_ab, y)
    sym = ((p_ab + p_ba - 1.0)**2).mean()
    return bce + lam_sym * sym

# --- Wrapper Class ---
class SiameseModel:
    def __init__(self, hidden_dim=128, lr=0.001, epochs=20, batch_size=64, dropout=0.5):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.net = None
        self.scaler = StandardScaler()
        self.f1_feats = [] # Store the column names for F1

    def _prepare_data(self, df, top_features=None):
        """
        Splits global dataframe into X1 (Fighter A) and X2 (Fighter B) matrices
        based on the provided top_features list.
        """
        if top_features is None:
            # If predicting and no top_features passed, rely on self.f1_feats logic
            pass

        # Identify pairs from top_features
        pairs = set()
        all_cols = set(df.columns)
        
        # If we are fitting, we determine pairs from top_features
        # If we are predicting, we use self.f1_feats to reconstruct
        
        if self.f1_feats:
            # Prediction mode or refit
            f1_cols = self.f1_feats
            f2_cols = []
            for c in f1_cols:
                if c.startswith('f_1_'):
                    f2_cols.append(c.replace('f_1_', 'f_2_'))
                elif c.endswith('_f_1'):
                    f2_cols.append(c.replace('_f_1', '_f_2'))
                else:
                    pass
                    
        # Let's implement the experimental logic exactly
        if top_features:
            candidates = top_features
        else:
            # Fallback if not provided (shouldn't happen during fit)
            candidates = df.columns
            
        numeric_pairs = []
        
        # We need to rebuild the list of (c1, c2) tuples
        # If we already fit, we stored f1_feats.
        
        if not self.f1_feats and top_features:
            # Fitting phase
            for feat in top_features:
                f1_col = None
                f2_col = None
                
                if feat.startswith('diff_'):
                    base = feat[5:] 
                    if f"f_1_{base}" in all_cols and f"f_2_{base}" in all_cols:
                        f1_col = f"f_1_{base}"
                        f2_col = f"f_2_{base}"
                    elif f"{base}_f_1" in all_cols and f"{base}_f_2" in all_cols:
                        f1_col = f"{base}_f_1"
                        f2_col = f"{base}_f_2"
                        
                elif '_f_1' in feat or 'f_1_' in feat:
                    if feat.startswith('f_1_'):
                        f1_col = feat
                        f2_col = feat.replace('f_1_', 'f_2_')
                    else:
                        f1_col = feat
                        f2_col = feat.replace('_f_1', '_f_2')
                    if f2_col not in all_cols: f1_col = None
                    
                elif '_f_2' in feat or 'f_2_' in feat:
                    if feat.startswith('f_2_'):
                        f2_col = feat
                        f1_col = feat.replace('f_2_', 'f_1_')
                    else:
                        f2_col = feat
                        f1_col = feat.replace('_f_2', '_f_1')
                    if f1_col not in all_cols: f1_col = None
                    
                if f1_col and f2_col:
                    if (f1_col, f2_col) not in pairs:
                        pairs.add((f1_col, f2_col))
                        
            # Filter numeric
            for c1, c2 in pairs:
                if pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
                    numeric_pairs.append((c1, c2))
            
            self.f1_feats = [p[0] for p in numeric_pairs]
            
        # Now extract using self.f1_feats
        f1_cols = self.f1_feats
        f2_cols = []
        for c in f1_cols:
             if c.startswith('f_1_'):
                 f2_cols.append(c.replace('f_1_', 'f_2_'))
             elif c.endswith('_f_1'):
                 f2_cols.append(c.replace('_f_1', '_f_2'))
        
        X1 = df[f1_cols].copy().fillna(0)
        X2 = df[f2_cols].copy().fillna(0)
        
        # Rename to generic names to ensure shared scaling and weight sharing
        generic_names = []
        for c in f1_cols:
            if c.startswith('f_1_'):
                generic_names.append(c[4:])
            elif c.endswith('_f_1'):
                generic_names.append(c[:-4])
            else:
                generic_names.append(c)
                
        X1.columns = generic_names
        X2.columns = generic_names
        
        return X1, X2

    def fit(self, df, top_features):
        print("Training Siamese Model (Pure)...")
        
        X1, X2 = self._prepare_data(df, top_features)
        print(f"Siamese Features: {X1.shape[1]} pairs")
        
        # Scale
        # We stack X1 and X2 to fit scaler on all fighter data
        combined = pd.concat([X1, X2], axis=0)
        self.scaler.fit(combined)
        
        X1_scaled = self.scaler.transform(X1)
        X2_scaled = self.scaler.transform(X2)
        y = df['target'].values.astype(np.float32)
        
        # Init Net
        self.net = SiameseMatchupNet(in_dim=X1.shape[1], hidden=self.hidden_dim)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        # Train Loop
        self.net.train()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X1_scaled, dtype=torch.float32),
            torch.tensor(X2_scaled, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for b_x1, b_x2, b_y in loader:
                optimizer.zero_grad()
                loss = symmetric_loss(self.net, b_x1, b_x2, b_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
                
    def predict_proba(self, df):
        self.net.eval()
        
        X1, X2 = self._prepare_data(df) # Uses self.f1_feats
        
        X1_scaled = self.scaler.transform(X1)
        X2_scaled = self.scaler.transform(X2)
        
        t_x1 = torch.tensor(X1_scaled, dtype=torch.float32)
        t_x2 = torch.tensor(X2_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            probs = self.net(t_x1, t_x2).numpy()
            
        return probs

    def save(self, path='v2/models/siamese_model.pkl'):
        state = {
            'net_state': self.net.state_dict(),
            'scaler': self.scaler,
            'f1_feats': self.f1_feats,
            'dims': (len(self.f1_feats), self.hidden_dim)
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path='v2/models/siamese_model.pkl'):
        state = torch.load(path)
        obj = cls(hidden_dim=state['dims'][1])
        obj.f1_feats = state['f1_feats']
        obj.scaler = state['scaler']
        
        obj.net = SiameseMatchupNet(in_dim=state['dims'][0], hidden=state['dims'][1])
        obj.net.load_state_dict(state['net_state'])
        obj.net.eval()
        return obj
