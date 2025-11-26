import pandas as pd
import os

def check_coverage():
    data_path = 'd:/AntiGravity/FightIQ/master/data/training_data.csv'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['Year'] = df['event_date'].dt.year
    
    print(f"Total rows: {len(df)}")
    
    for year in range(2020, 2026):
        year_df = df[df['Year'] == year]
        total = len(year_df)
        if total == 0:
            print(f"{year}: No data")
            continue
            
        # Check if odds are present
        with_odds = year_df[year_df['f_1_odds'].notna() & year_df['f_2_odds'].notna()]
        count_with = len(with_odds)
        missing = total - count_with
        
        print(f"{year}: Total {total}, With Odds {count_with} ({count_with/total:.1%}), Missing {missing}")

if __name__ == "__main__":
    check_coverage()
