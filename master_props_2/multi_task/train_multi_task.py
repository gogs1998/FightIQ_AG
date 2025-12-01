import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # for models.py

from master_props_2.models import MultiTaskSiameseNet, multi_task_loss

def train_multi_task():
    print("=== Experiment B: Multi-Task Siamese ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('../../master_3/data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Load Features
    with open('../../master_3/features_enhanced.json', 'r') as f:
        features = json.load(f)
        
    # 2. Preprocessing
    # We need to scale features for Neural Net
    print("Scaling features...")
    scaler = StandardScaler()
    X_all = df[features].fillna(0).values
    X_scaled = scaler.fit_transform(X_all)
    
    # Targets
    y_win = df['winner_encoded'].values
    y_finish = df['is_finish'].values
    
    # Method Target (0=KO, 1=Sub, 2=Dec)
    # If Decision, Method is 2.
    # If Finish, check result.
    def get_method_target(row):
        if row['is_finish'] == 0: return 2
        res = str(row['result'])
        if 'Submission' in res: return 1
        return 0 # Default to KO if finish and not sub
        
    y_method = df.apply(get_method_target, axis=1).values
    
    # Round Target (0-4)
    # If Decision, Round is usually 2 (3 rnds) or 4 (5 rnds).
    # Let's use actual round - 1.
    def get_round_target(row):
        try:
            r = int(row['finish_round'])
            return min(max(r - 1, 0), 4)
        except:
            return 2 # Default to R3
            
    y_round = df.apply(get_round_target, axis=1).values
    
    # Split
    train_mask = df['event_date'] < '2024-01-01'
    test_mask = df['event_date'] >= '2024-01-01'
    
    X_train = X_scaled[train_mask]
    y_train_win = y_win[train_mask]
    y_train_finish = y_finish[train_mask]
    y_train_method = y_method[train_mask]
    y_train_round = y_round[train_mask]
    
    X_test = X_scaled[test_mask]
    y_test_win = y_win[test_mask]
    y_test_finish = y_finish[test_mask]
    y_test_method = y_method[test_mask]
    y_test_round = y_round[test_mask]
    
    test_df = df[test_mask].copy().reset_index(drop=True)
    
    print(f"Train Set: {len(X_train)}")
    print(f"Test Set:  {len(X_test)}")
    
    # Tensors
    t_X_train = torch.FloatTensor(X_train).to(device)
    t_y_train_win = torch.FloatTensor(y_train_win).unsqueeze(1).to(device)
    t_y_train_finish = torch.FloatTensor(y_train_finish).unsqueeze(1).to(device)
    t_y_train_method = torch.LongTensor(y_train_method).to(device)
    t_y_train_round = torch.LongTensor(y_train_round).to(device)
    
    t_X_test = torch.FloatTensor(X_test).to(device)
    
    # Dataset
    dataset = TensorDataset(t_X_train, t_y_train_win, t_y_train_finish, t_y_train_method, t_y_train_round)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. Model Setup
    # Robust Pair Finding
    pairs = []
    all_cols = set(features)
    
    for feat in features:
        f1_col = None
        f2_col = None
        
        if feat.startswith('f_1_'):
            f1_col = feat
            f2_col = feat.replace('f_1_', 'f_2_')
        elif '_f_1' in feat:
            f1_col = feat
            f2_col = feat.replace('_f_1', '_f_2')
            
        if f1_col and f2_col and f2_col in all_cols:
            if (f1_col, f2_col) not in pairs:
                pairs.append((f1_col, f2_col))
                
    pairs.sort()
    
    f1_cols = [p[0] for p in pairs]
    f2_cols = [p[1] for p in pairs]
    
    f1_indices = [features.index(c) for c in f1_cols]
    f2_indices = [features.index(c) for c in f2_cols]
    
    input_dim = len(pairs)
    print(f"DEBUG: Found {len(pairs)} pairs.")
    print(f"Siamese Input Dim: {input_dim}")
    
    model = MultiTaskSiameseNet(input_dim=input_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train Loop
    epochs = 50 # Start small
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in loader:
            bx, by_win, by_finish, by_method, by_round = batch
            
            # Split bx into f1, f2
            f1 = bx[:, f1_indices]
            f2 = bx[:, f2_indices]
            
            if epoch == 0 and total_loss == 0:
                print(f"DEBUG: f1.shape={f1.shape}")
                print(f"DEBUG: f2.shape={f2.shape}")
                print(f"DEBUG: len(f2_cols)={len(f2_cols)}")
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(f1, f2) # (win, finish, method, round)
            
            # Loss
            loss, details = multi_task_loss(preds, (by_win, by_finish, by_method, by_round))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            
    # 5. Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/multi_task_siamese.pth')
    joblib.dump(scaler, 'models/multi_task_scaler.pkl')
    # Save feature indices for splitting
    with open('models/feature_indices.json', 'w') as f:
        json.dump({'f1': f1_indices, 'f2': f2_indices}, f)
        
    print("Model saved to master_props_2/multi_task/models/")
    
    # 6. Validate Trifecta
    print("\n=== Validation (Trifecta) ===")
    model.eval()
    
    # Prepare Test Data
    t_f1_test = t_X_test[:, f1_indices]
    t_f2_test = t_X_test[:, f2_indices]
    
    with torch.no_grad():
        p_win, p_finish, p_method, p_round = model(t_f1_test, t_f2_test)
        
    # Convert to numpy
    p_win = p_win.cpu().numpy().flatten()
    p_finish = p_finish.cpu().numpy().flatten()
    p_method = torch.softmax(p_method, dim=1).cpu().numpy()
    p_round = torch.softmax(p_round, dim=1).cpu().numpy()
    
    correct_trifecta = 0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Winner
        if p_win[i] > 0.5:
            pred_winner_idx = 1
        else:
            pred_winner_idx = 0
            
        # Method
        # Logic:
        # If p_finish < 0.5 -> Decision
        # Else -> Max(KO, Sub)
        # OR use the raw probabilities combined?
        # Let's use the explicit heads.
        
        is_finish_pred = p_finish[i] > 0.5
        
        if not is_finish_pred:
            pred_method_str = 'Decision'
        else:
            # Check Method Head (0=KO, 1=Sub, 2=Dec)
            # We ignore 2 here because we predicted finish.
            # Compare 0 vs 1
            if p_method[i][0] > p_method[i][1]:
                pred_method_str = 'KO/TKO'
            else:
                pred_method_str = 'Submission'
                
        # Round
        best_rnd_idx = np.argmax(p_round[i])
        pred_round_val = best_rnd_idx + 1
        
        # Actuals
        actual_winner = row['winner_encoded']
        actual_res = str(row['result'])
        try:
            actual_round = int(row['finish_round'])
        except:
            actual_round = -1
            
        # Check
        is_win_correct = (pred_winner_idx == actual_winner)
        
        is_method_correct = False
        if 'KO' in actual_res and pred_method_str == 'KO/TKO': is_method_correct = True
        if 'Submission' in actual_res and pred_method_str == 'Submission': is_method_correct = True
        if 'Decision' in actual_res and pred_method_str == 'Decision': is_method_correct = True
        
        is_round_correct = (pred_round_val == actual_round)
        
        # Trifecta
        if pred_method_str == 'Decision':
            is_trifecta = is_win_correct and is_method_correct
        else:
            is_trifecta = is_win_correct and is_method_correct and is_round_correct
            
        if is_trifecta:
            correct_trifecta += 1
            
    acc_trifecta = correct_trifecta / len(test_df)
    print(f"Trifecta Accuracy: {acc_trifecta:.2%}")
    print(f"Baseline (Exp A):   33.81%")

if __name__ == "__main__":
    train_multi_task()
