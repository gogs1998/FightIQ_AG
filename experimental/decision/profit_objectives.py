import numpy as np
import pandas as pd

def kelly_criterion(prob, odds, fractional=0.25):
    """
    Calculate Kelly Criterion bet size.
    prob: Estimated probability of winning (0-1).
    odds: Decimal odds (e.g., 2.5).
    fractional: Fraction of Kelly to bet (e.g., 0.25 for quarter-Kelly).
    
    Returns: Fraction of bankroll to bet (0-1).
    """
    if odds <= 1:
        return 0.0
        
    b = odds - 1
    q = 1 - prob
    p = prob
    
    # f* = (bp - q) / b
    f_star = (b * p - q) / b
    
    # Only bet if positive edge
    if f_star <= 0:
        return 0.0
        
    # Apply fractional Kelly
    return f_star * fractional

def simulate_betting(probs, y_true, odds_f1, odds_f2, strategy='kelly', initial_bankroll=1000.0):
    """
    Simulate betting on a sequence of fights.
    probs: (n,) array of probability that F1 wins.
    y_true: (n,) array of true outcomes (1 if F1 wins, 0 if F2 wins).
    odds_f1: (n,) array of decimal odds for F1.
    odds_f2: (n,) array of decimal odds for F2.
    strategy: 'kelly' or 'flat'.
    
    Returns: Final Bankroll, ROI, History
    """
    bankroll = initial_bankroll
    history = []
    total_wagered = 0.0
    
    for i in range(len(probs)):
        p_f1 = probs[i]
        p_f2 = 1 - p_f1
        
        o_f1 = odds_f1[i]
        o_f2 = odds_f2[i]
        
        outcome = y_true[i] # 1 if F1 won, 0 if F2 won
        
        # Decide who to bet on
        # Calculate edge for both
        # Edge = Prob * Odds - 1
        edge_f1 = p_f1 * o_f1 - 1
        edge_f2 = p_f2 * o_f2 - 1
        
        bet_on = None
        bet_fraction = 0.0
        
        if edge_f1 > 0 and edge_f1 > edge_f2:
            bet_on = 'f1'
            if strategy == 'kelly':
                bet_fraction = kelly_criterion(p_f1, o_f1)
            elif strategy == 'flat':
                bet_fraction = 0.05 # 5% flat bet
                
        elif edge_f2 > 0:
            bet_on = 'f2'
            if strategy == 'kelly':
                bet_fraction = kelly_criterion(p_f2, o_f2)
            elif strategy == 'flat':
                bet_fraction = 0.05
        
        # Cap bet at 20% of bankroll for safety
        bet_fraction = min(bet_fraction, 0.20)
        
        wager = bankroll * bet_fraction
        if wager < 1.0: # Minimum bet
            wager = 0.0
            
        if wager > 0:
            total_wagered += wager
            bankroll -= wager
            
            won = False
            if bet_on == 'f1' and outcome == 1:
                won = True
                winnings = wager * o_f1
                bankroll += winnings
            elif bet_on == 'f2' and outcome == 0:
                won = True
                winnings = wager * o_f2
                bankroll += winnings
                
        history.append(bankroll)
        
    roi = (bankroll - initial_bankroll) / total_wagered if total_wagered > 0 else 0.0
    
    return bankroll, roi, history
