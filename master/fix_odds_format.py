import pandas as pd

def fix_outliers():
    df = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    
    # Identify columns
    odds_cols = ['f_1_odds', 'f_2_odds']
    
    count = 0
    for col in odds_cols:
        # Find values > 50 (Likely Positive American)
        mask_pos = df[col] > 50
        # Find values < -50 (Likely Negative American, though usually not present in Decimal columns)
        mask_neg = df[col] < -1
        
        # Convert Positive American to Decimal: (Odds / 100) + 1
        # e.g. 290 -> 3.9
        if mask_pos.any():
            print(f"Fixing {mask_pos.sum()} positive American odds in {col}...")
            # Show samples before
            print(df.loc[mask_pos, col].head().values)
            
            df.loc[mask_pos, col] = (df.loc[mask_pos, col] / 100) + 1
            
            # Show samples after
            print(df.loc[mask_pos, col].head().values)
            count += mask_pos.sum()

        # Convert Negative American to Decimal: (100 / Abs(Odds)) + 1
        # e.g. -150 -> 1.67
        # Note: Decimal odds are never negative. If we see negative values, they are definitely American.
        mask_neg_real = df[col] < 0
        if mask_neg_real.any():
            print(f"Fixing {mask_neg_real.sum()} negative American odds in {col}...")
            print(df.loc[mask_neg_real, col].head().values)
            
            df.loc[mask_neg_real, col] = (100 / df.loc[mask_neg_real, col].abs()) + 1
            
            print(df.loc[mask_neg_real, col].head().values)
            count += mask_neg_real.sum()

    print(f"Fixed {count} total outliers.")
    df.to_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv', index=False)

if __name__ == "__main__":
    fix_outliers()
