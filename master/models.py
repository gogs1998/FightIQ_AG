import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class SiameseMatchupNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, f1, f2):
        e1 = self.encoder(f1)
        e2 = self.encoder(f2)
        combined = torch.cat([e1, e2], dim=1)
        return self.classifier(combined).squeeze()

def symmetric_loss(model, f1, f2, y):
    pred1 = model(f1, f2)
    loss1 = nn.BCELoss()(pred1, y)
    pred2 = model(f2, f1)
    loss2 = nn.BCELoss()(pred2, 1.0 - y)
    return 0.5 * (loss1 + loss2)

def prepare_siamese_data(X_df, features):
    """
    Prepare data for Siamese network using robust pair finding.
    """
    pairs = set()
    all_cols = set(X_df.columns)
    
    for feat in features:
        base = None
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

    pairs = list(pairs)
    numeric_pairs = []
    for c1, c2 in pairs:
        if c1 in X_df.columns and c2 in X_df.columns:
            numeric_pairs.append((c1, c2))
            
    f1_feats = [p[0] for p in numeric_pairs]
    f2_feats = [p[1] for p in numeric_pairs]
    
    if not f1_feats:
        return np.zeros((len(X_df), 1)), np.zeros((len(X_df), 1)), 1, []

    f1_data = X_df[f1_feats].values
    f2_data = X_df[f2_feats].values
    
    return f1_data, f2_data, len(f1_feats), f1_feats
