import pandas as pd

def show_odds_eras():
    df = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['Year'] = df['event_date'].dt.year
    
    # Define Eras
    # Early Era: 1994-2009
    # Modern Era: 2010-2019
    # Current Era: 2020-2025
    
    eras = [
        (1994, 2009, "Early Era (1994-2009)"),
        (2010, 2019, "Modern Era (2010-2019)"),
        (2020, 2025, "Current Era (2020-2025)")
    ]
    
    print(f"{'Era':<25} | {'Total Fights':<12} | {'With Odds':<10} | {'Coverage %':<10}")
    print("-" * 65)
    
    for start, end, label in eras:
        mask = (df['Year'] >= start) & (df['Year'] <= end)
        subset = df[mask]
        total = len(subset)
        with_odds = subset[subset['f_1_odds'].notna() & subset['f_2_odds'].notna()]
        count = len(with_odds)
        pct = (count / total * 100) if total > 0 else 0
        
        print(f"{label:<25} | {total:<12} | {count:<10} | {pct:>9.1f}%")

if __name__ == "__main__":
    show_odds_eras()
