import pandas as pd
import numpy as np

def calculate_pear_features(df):
    """
    Calculate PEAR (Pace-Elasticity & Attrition Response) features.
    - pace_volatility: Variance in strike output across rounds.
    - cardio_score: Ratio of R3 output to R1 output.
    - damage_absorption: Strikes absorbed per minute.
    """
    df = df.copy()
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    # Placeholder for full PEAR logic. 
    # In v1, this was complex. For v2 MVP, we'll implement a simplified version
    # based on available columns in golden data.
    
    # We need round-by-round data to do this properly.
    # If we don't have it easily, we'll use proxies from full_data_golden.
    
    # Assuming we have 'f_1_fighter_SlpM', 'f_1_fighter_Str_Acc', etc.
    # We can create "Efficiency" metrics.
    
    # 1. Efficiency (Strikes Landed / Strikes Attempted)
    # This is already in Str_Acc, but let's make sure.
    
    # 2. Pace (SlpM)
    
    # 3. Defense (SApM)
    
    # Real PEAR requires round-level parsing. 
    # Let's assume we have the 'round_dynamics' data or can compute it.
    # For now, we will use the "Advanced Stats" available in the golden dataset as PEAR proxies.
    
    # Create interaction features
    if 'f_1_fighter_SlpM' in df.columns and 'f_1_fighter_SApM' in df.columns:
        df['f_1_net_damage'] = df['f_1_fighter_SlpM'] - df['f_1_fighter_SApM']
        df['f_2_net_damage'] = df['f_2_fighter_SlpM'] - df['f_2_fighter_SApM']
        df['diff_net_damage'] = df['f_1_net_damage'] - df['f_2_net_damage']
        
    # Pace vs Defense Interaction
    # High Pace vs Low Defense = High Volatility
    
    return df
