import pandas as pd
import numpy as np

def analyze_recent():
    print("=== Recent Performance Analysis ===")
    df = pd.read_csv('betting_log.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    total_bets = len(df)
    print(f"Total Bets: {total_bets}")
    
    # Last 50 Bets
    last_50 = df.tail(50).copy()
    wins = last_50[last_50['Result'] == 'WIN']
    losses = last_50[last_50['Result'] == 'LOSS']
    
    win_rate = len(wins) / 50
    net_profit = last_50['Profit'].sum()
    
    print(f"\nLast 50 Bets:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Net Profit (Kelly): ${net_profit:,.2f}")
    
    # Rolling Win Rate (Window 20)
    df['Win'] = (df['Result'] == 'WIN').astype(int)
    df['Rolling_WR'] = df['Win'].rolling(20).mean()
    
    print("\nLast 20 Bets Detail:")
    cols = ['Date', 'Fighter1', 'Fighter2', 'Bet_On', 'Odds', 'Prob', 'Result', 'Profit']
    print(df[cols].tail(20).to_string(index=False))
    
    # Check for specific bad run
    print("\nStreak Analysis (Last 20):")
    results = df['Result'].tail(20).tolist()
    print(" -> ".join(results))

if __name__ == "__main__":
    analyze_recent()
