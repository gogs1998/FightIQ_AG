import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

class FightSequenceEncoder(nn.Module):
    """
    Encodes a sequence of past fights into a 'Current Form' embedding.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # We only care about the final hidden state
        out, (h_n, c_n) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]
        return self.head(last_hidden)

def get_fighter_history(df):
    """
    Reconstructs a fighter-centric history from the matchups.
    Returns a dictionary: {fighter_id: DataFrame of their fights sorted by date}
    """
    # We need to stack f1 and f2 perspectives
    f1_cols = [c for c in df.columns if c.startswith('f_1_')]
    f2_cols = [c for c in df.columns if c.startswith('f_2_')]
    
    # Map to generic names
    base_cols = [c.replace('f_1_', '') for c in f1_cols]
    
    # Perspective 1: F1 is the "Hero"
    df1 = df[['event_date', 'f_1_name'] + f1_cols].copy()
    df1.columns = ['date', 'fighter_name'] + base_cols
    
    # Perspective 2: F2 is the "Hero"
    df2 = df[['event_date', 'f_2_name'] + f2_cols].copy()
    df2.columns = ['date', 'fighter_name'] + base_cols
    
    # Concatenate and sort
    full_history = pd.concat([df1, df2], axis=0)
    full_history['date'] = pd.to_datetime(full_history['date'])
    full_history = full_history.sort_values(['fighter_name', 'date'])
    
    # Group by fighter
    history_dict = {k: v for k, v in full_history.groupby('fighter_name')}
    return history_dict

def prepare_sequences(df, feature_cols, seq_len=5):
    """
    Prepares sequence data for the LSTM.
    Returns:
        X_seq_f1: (N, seq_len, input_dim)
        X_seq_f2: (N, seq_len, input_dim)
    """
    print("Preparing sequences (Optimized)...")
    
    # 1. Identify pairs and build input matrix
    pairs = []
    used_cols = set()
    for col in feature_cols:
        if col in used_cols: continue
        
        partner = None
        if col.startswith('f_1_'): partner = col.replace('f_1_', 'f_2_')
        elif col.startswith('f_2_'): partner = col.replace('f_2_', 'f_1_')
        elif '_f_1' in col: partner = col.replace('_f_1', '_f_2')
        elif '_f_2' in col: partner = col.replace('_f_2', '_f_1')
            
        if partner and partner in feature_cols:
            c1 = col if 'f_1' in col else partner
            c2 = partner if 'f_1' in col else col
            pairs.append((c1, c2))
            used_cols.add(c1)
            used_cols.add(c2)
            
    input_dim = len(pairs)
    print(f"LSTM Input Dim: {input_dim}")
    
    # Pre-compute data matrix: (N, 2, D)
    # 0=F1 stats, 1=F2 stats
    data_matrix = np.zeros((len(df), 2, input_dim))
    for i, (c1, c2) in enumerate(pairs):
        data_matrix[:, 0, i] = df[c1].fillna(0).values
        data_matrix[:, 1, i] = df[c2].fillna(0).values
        
    # 2. Build Sequences using History Buffer
    X_seq_f1 = np.zeros((len(df), seq_len, input_dim))
    X_seq_f2 = np.zeros((len(df), seq_len, input_dim))
    
    # History: fighter_name -> list of (fight_idx, side_idx)
    history = {}
    
    # Iterate through sorted dataframe
    # Assuming df is already sorted by date!
    
    f1_names = df['f_1_name'].values
    f2_names = df['f_2_name'].values
    
    for i in tqdm(range(len(df)), desc="Building Sequences"):
        n1 = f1_names[i]
        n2 = f2_names[i]
        
        # Retrieve history for F1
        if n1 in history:
            past = history[n1] # List of (idx, side)
            # Take last seq_len
            seq = past[-seq_len:]
            # Fill X_seq_f1
            for t, (prev_idx, side) in enumerate(seq):
                pos = seq_len - len(seq) + t
                X_seq_f1[i, pos, :] = data_matrix[prev_idx, side, :]
        
        # Retrieve history for F2
        if n2 in history:
            past = history[n2]
            seq = past[-seq_len:]
            for t, (prev_idx, side) in enumerate(seq):
                pos = seq_len - len(seq) + t
                X_seq_f2[i, pos, :] = data_matrix[prev_idx, side, :]
                
        # Update History (for NEXT fights)
        if n1 not in history: history[n1] = []
        if n2 not in history: history[n2] = []
        
        history[n1].append((i, 0)) # F1 was side 0
        history[n2].append((i, 1)) # F2 was side 1
        
    return X_seq_f1, X_seq_f2, input_dim
