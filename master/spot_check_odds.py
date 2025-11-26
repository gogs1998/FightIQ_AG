import pandas as pd

def spot_check():
    df = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Check a few key fights from 2025
    # UFC 311: Islam Makhachev vs Arman Tsarukyan
    # UFC 312: Dricus Du Plessis vs Sean Strickland
    
    targets = [
        ('Islam Makhachev', 'Arman Tsarukyan'),
        ('Dricus Du Plessis', 'Sean Strickland'),
        ('Ilia Topuria', 'Alexander Volkanovski'), # Assuming rematch or similar
        ('Jon Jones', 'Tom Aspinall') # Hypothetical
    ]
    
    print("Spot Checking Odds for 2025:")
    for f1, f2 in targets:
        # Try both orders
        match = df[((df['f_1_name'].str.contains(f1, case=False)) & (df['f_2_name'].str.contains(f2, case=False))) |
                   ((df['f_1_name'].str.contains(f2, case=False)) & (df['f_2_name'].str.contains(f1, case=False)))]
        
        if not match.empty:
            row = match.iloc[0]
            print(f"\nMatch: {row['f_1_name']} vs {row['f_2_name']}")
            print(f"Date: {row['event_date'].date()}")
            print(f"Odds: {row['f_1_odds']} vs {row['f_2_odds']}")
        else:
            print(f"\nMatch not found: {f1} vs {f2}")

if __name__ == "__main__":
    spot_check()
