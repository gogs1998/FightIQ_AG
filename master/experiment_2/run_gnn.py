import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import json
import networkx as nx

def run_gnn_experiment():
    print("=== Experiment: 'MMA Math' Graph Neural Network (GNN) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 2. Build the Graph (Training Set: < 2024)
    print("Building Fight Graph (2010-2023)...")
    mask_train = df['event_date'] < '2024-01-01'
    train_df = df[mask_train]
    
    # Map fighters to IDs
    all_fighters = pd.concat([df['f_1_name'], df['f_2_name']]).unique()
    fighter_to_id = {name: i for i, name in enumerate(all_fighters)}
    n_fighters = len(all_fighters)
    embedding_dim = 16
    
    print(f"Total Fighters: {n_fighters}")
    
    # Prepare Training Pairs
    # We want to learn embeddings such that P(A beats B) is high if A won.
    # Model: Score = MLP(Emb[A], Emb[B]) or just Sigmoid(Emb[A] - Emb[B])
    # Let's use a "Neural Elo" approach: P(A>B) = Sigmoid( (Emb[A] - Emb[B]) . W )
    # This captures multidimensional strengths (Grappling, Striking, etc if dims align)
    
    f1_indices = train_df['f_1_name'].map(fighter_to_id).values
    f2_indices = train_df['f_2_name'].map(fighter_to_id).values
    targets = train_df['target'].values # 1 if F1 wins, 0 if F2 wins
    
    # Convert to Tensors
    f1_tensor = torch.LongTensor(f1_indices)
    f2_tensor = torch.LongTensor(f2_indices)
    y_tensor = torch.FloatTensor(targets)
    
    # 3. Define GNN / Embedding Model
    class NeuralElo(nn.Module):
        def __init__(self, num_fighters, dim):
            super().__init__()
            self.embeddings = nn.Embedding(num_fighters, dim)
            # Initialize with small random weights
            nn.init.normal_(self.embeddings.weight, std=0.01)
            
            # A projection vector to map difference to score
            # If we just do sum(diff), it's like standard Elo.
            # If we do diff @ w, it learns weighted dimensions.
            self.projector = nn.Linear(dim, 1, bias=False)
            
        def forward(self, idx1, idx2):
            emb1 = self.embeddings(idx1)
            emb2 = self.embeddings(idx2)
            
            # Difference vector captures relative advantage in each dimension
            diff = emb1 - emb2 
            
            # Project to scalar score
            score = self.projector(diff).squeeze()
            return torch.sigmoid(score)
            
    model = NeuralElo(n_fighters, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # 4. Train Embeddings
    print("Training Fighter Embeddings...")
    epochs = 50
    batch_size = 1024
    n_samples = len(f1_tensor)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        permutation = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_f1 = f1_tensor[indices]
            batch_f2 = f2_tensor[indices]
            batch_y = y_tensor[indices]
            
            optimizer.zero_grad()
            preds = model(batch_f1, batch_f2)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / (n_samples/batch_size):.4f}")
            
    # 5. Evaluate on Test Set (2024-2025)
    print("\nEvaluating on Holdout Set (2024-2025)...")
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy()
    
    # We can only predict for fighters seen during training
    # Newcomers will get a random/mean embedding (or just fail)
    # For this experiment, we'll filter for "Known vs Known" to test the pure GNN signal
    
    test_f1 = test_df['f_1_name'].map(fighter_to_id)
    test_f2 = test_df['f_2_name'].map(fighter_to_id)
    
    # Filter valid
    valid_mask = test_f1.notna() & test_f2.notna()
    print(f"Test Fights: {len(test_df)}")
    print(f"Valid for GNN (Both fighters seen before): {valid_mask.sum()}")
    
    test_f1 = test_f1[valid_mask].astype(int).values
    test_f2 = test_f2[valid_mask].astype(int).values
    test_y = test_df.loc[valid_mask, 'target'].values
    
    model.eval()
    with torch.no_grad():
        t_f1 = torch.LongTensor(test_f1)
        t_f2 = torch.LongTensor(test_f2)
        gnn_probs = model(t_f1, t_f2).numpy()
        
    gnn_preds = (gnn_probs > 0.5).astype(int)
    acc = accuracy_score(test_y, gnn_preds)
    
    print(f"\n=== GNN Experiment Results ===")
    print(f"GNN Accuracy (Known Fighters Only): {acc:.4%}")
    
    # Compare to Baseline (Odds) on same subset
    # Odds accuracy
    odds_preds = []
    for i, idx in enumerate(test_df[valid_mask].index):
        row = test_df.loc[idx]
        if row['f_1_odds'] < row['f_2_odds']:
            odds_preds.append(1)
        else:
            odds_preds.append(0)
            
    odds_acc = accuracy_score(test_y, odds_preds)
    print(f"Implied Odds Accuracy (Baseline):   {odds_acc:.4%}")
    print(f"GNN Edge: {acc - odds_acc:+.4%}")
    
    # Save Embeddings?
    # Maybe visualize them later (PCA)

if __name__ == "__main__":
    run_gnn_experiment()
