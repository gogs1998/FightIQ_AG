import pandas as pd

def check_format():
    df = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    
    print("Sample Odds from 2023:")
    print(df[df['event_date'].str.contains('2023')][['f_1_odds', 'f_2_odds']].head())
    
    print("\nSample Odds from 2024:")
    print(df[df['event_date'].str.contains('2024')][['f_1_odds', 'f_2_odds']].head())
    
    print("\nSample Odds from 2025:")
    print(df[df['event_date'].str.contains('2025')][['f_1_odds', 'f_2_odds']].head())

if __name__ == "__main__":
    check_format()
