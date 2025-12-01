import torch
import torch.nn as nn
import torch.nn.functional as F

class FightSequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)

class MultiTaskSiameseNet(nn.Module):
    def __init__(self, input_dim, seq_input_dim=0, hidden_dim=128):
        super().__init__()
        
        # --- Shared Backbone ---
        # Tabular Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # Sequence Encoder
        self.seq_encoder = None
        seq_emb_dim = 0
        if seq_input_dim > 0:
            self.seq_encoder = FightSequenceEncoder(seq_input_dim)
            seq_emb_dim = 16
            
        # Combined Dimension
        # (Tabular_Emb * 2) + (Seq_Emb * 2)
        self.combined_dim = (64 * 2) + (seq_emb_dim * 2)
        
        # Shared Layer after concatenation
        self.shared_fc = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # --- Multi-Task Heads ---
        
        # 1. Winner Head (Binary: F1 vs F2)
        self.win_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 2. Finish Head (Binary: Finish vs Decision)
        self.finish_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 3. Method Head (Multiclass: KO, Sub, Dec)
        # Note: Dec is redundant with Finish=0, but useful for supervision
        self.method_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 0=KO, 1=Sub, 2=Dec
        )
        
        # 4. Round Head (Multiclass: 1, 2, 3, 4, 5)
        self.round_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5) # 0=R1, 1=R2, ...
        )
        
    def forward(self, f1, f2, seq_f1=None, seq_f2=None):
        # Encode F1
        e1 = self.encoder(f1)
        if self.seq_encoder and seq_f1 is not None:
            s1 = self.seq_encoder(seq_f1)
            e1 = torch.cat([e1, s1], dim=1)
            
        # Encode F2
        e2 = self.encoder(f2)
        if self.seq_encoder and seq_f2 is not None:
            s2 = self.seq_encoder(seq_f2)
            e2 = torch.cat([e2, s2], dim=1)
            
        # Combine
        combined = torch.cat([e1, e2], dim=1)
        shared_features = self.shared_fc(combined)
        
        # Heads
        win_prob = self.win_head(shared_features)
        finish_prob = self.finish_head(shared_features)
        method_logits = self.method_head(shared_features)
        round_logits = self.round_head(shared_features)
        
        return win_prob, finish_prob, method_logits, round_logits

def multi_task_loss(preds, targets, weights={'win':1.0, 'finish':0.5, 'method':0.5, 'round':0.5}):
    """
    preds: (win_prob, finish_prob, method_logits, round_logits)
    targets: (win_target, finish_target, method_target, round_target)
    """
    p_win, p_finish, p_method, p_round = preds
    t_win, t_finish, t_method, t_round = targets
    
    # 1. Winner Loss (BCE)
    loss_win = nn.BCELoss()(p_win, t_win)
    
    # 2. Finish Loss (BCE)
    loss_finish = nn.BCELoss()(p_finish, t_finish)
    
    # 3. Method Loss (CrossEntropy)
    # t_method should be class indices (0,1,2)
    loss_method = nn.CrossEntropyLoss()(p_method, t_method.long())
    
    # 4. Round Loss (CrossEntropy)
    # Only compute round loss for finishes? 
    # Or for all? If decision, round is usually 3 or 5.
    # Let's compute for all, assuming t_round is correct (e.g. 3 for 3-round dec).
    # But wait, t_round for decision is just the last round.
    # Predicting "Round 3" for a decision is correct.
    loss_round = nn.CrossEntropyLoss()(p_round, t_round.long())
    
    total_loss = (weights['win'] * loss_win + 
                  weights['finish'] * loss_finish + 
                  weights['method'] * loss_method + 
                  weights['round'] * loss_round)
                  
    return total_loss, {'win': loss_win.item(), 'finish': loss_finish.item(), 'method': loss_method.item(), 'round': loss_round.item()}
