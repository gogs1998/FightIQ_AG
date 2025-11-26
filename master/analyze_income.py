import pandas as pd
import numpy as np

def analyze_income_potential():
    print("=== Income Potential Analysis ===")
    
    # Load the verified bet log
    df = pd.read_csv('final_sequential_bets.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    
    # Clean currency strings
    df['Profit'] = df['Profit'].astype(str).str.replace('$','').str.replace(',','').astype(float)
    df['Stake'] = df['Stake'].astype(str).str.replace('$','').str.replace(',','').astype(float)
    df['Bankroll'] = df['Bankroll'].astype(str).str.replace('$','').str.replace(',','').astype(float)
    
    # Calculate key metrics
    monthly_stats = df.groupby('Month').agg({
        'Profit': 'sum',
        'Stake': 'sum',
        'Date': 'count' # Number of bets
    }).rename(columns={'Date': 'Bets'})
    
    avg_monthly_profit_raw = monthly_stats['Profit'].mean()
    avg_bets_per_month = monthly_stats['Bets'].mean()
    avg_roi = df['Profit'].sum() / df['Stake'].sum()
    
    # Calculate Average Stake % (Stake / Bankroll at time of bet)
    # We need to reconstruct the bankroll at the moment of the bet to get accurate %
    # But we can approximate using the Bankroll column which is post-bet
    # Better: Use the simulation logic. 
    # Let's just look at the average stake size relative to the *starting* bankroll of that month?
    # Actually, simpler: We know the ROI is ~32%.
    # To make £1000, you need to turnover £1000 / 0.32 = £3125 per month.
    # With ~15 bets a month, avg bet is £200.
    # If avg bet is ~2-3% of bankroll (conservative 1/4 Kelly estimate), we can estimate bankroll.
    
    # Let's find the actual average stake percentage used in the log
    # We can't easily get the pre-bet bankroll from the CSV perfectly without re-running, 
    # but we can assume the 'Bankroll' column - 'Profit' is roughly the bankroll during the bet.
    df['PreBetBankroll'] = df['Bankroll'] - df['Profit']
    df['StakePct'] = df['Stake'] / df['PreBetBankroll']
    avg_stake_pct = df['StakePct'].mean()
    
    print(f"Average ROI: {avg_roi*100:.2f}%")
    print(f"Average Bets per Month: {avg_bets_per_month:.1f}")
    print(f"Average Stake Size: {avg_stake_pct*100:.2f}% of Bankroll")
    
    # Target: £1000 / month
    target_profit = 1000.0
    required_turnover = target_profit / avg_roi
    required_avg_bet = required_turnover / avg_bets_per_month
    required_bankroll = required_avg_bet / avg_stake_pct
    
    print(f"\n--- To make £1000 / Month ---")
    print(f"Required Turnover: £{required_turnover:.2f}")
    print(f"Required Avg Bet: £{required_avg_bet:.2f}")
    print(f"ESTIMATED REQUIRED BANKROLL: £{required_bankroll:.2f}")
    
    # Variance Check: Worst Month
    worst_month = monthly_stats['Profit'].min()
    # Scale worst month to the required bankroll scenario
    # The simulation started with $1000. 
    # If we scale up to the required bankroll, what would the worst month look like?
    scale_factor = required_bankroll / 1000.0 
    # Note: This scaling is rough because Kelly scales dynamically, but gives an idea.
    
    print(f"\n--- Risk Analysis ---")
    print(f"Worst Month in Sim (Raw): ${worst_month:.2f}")
    print(f"Best Month in Sim (Raw): ${monthly_stats['Profit'].max():.2f}")
    
    # Let's simulate a fixed bankroll scenario to be precise
    # If we kept bankroll constant at X, would we make 1000?
    
    print("\n--- Monthly Breakdown (Simulated with Flat £5,000 Bankroll) ---")
    # Re-simulating strictly for income estimation
    sim_bankroll = 5000.0
    sim_profits = []
    for _, row in df.iterrows():
        # Recalculate stake based on fixed bankroll to see pure income potential
        # Stake = Bankroll * StakePct
        stake = sim_bankroll * row['StakePct']
        if row['Result'] == 'WIN':
            profit = stake * (float(row['Odds']) - 1)
        else:
            profit = -stake
        sim_profits.append({'Month': row['Month'], 'Profit': profit})
        
    sim_df = pd.DataFrame(sim_profits)
    monthly_sim = sim_df.groupby('Month')['Profit'].sum()
    
    print(f"Avg Monthly Profit with £5k Bankroll: £{monthly_sim.mean():.2f}")
    print(f"Worst Month with £5k Bankroll: £{monthly_sim.min():.2f}")
    
    # Scale to £1000 target
    factor = 1000.0 / monthly_sim.mean()
    final_bankroll_rec = 5000.0 * factor
    print(f"\n>>> FINAL VERDICT: You need ~£{final_bankroll_rec:,.0f} bankroll to average £1000/month. <<<")

if __name__ == "__main__":
    analyze_income_potential()
