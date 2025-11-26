import pandas as pd

def check_fight_id():
    df = pd.read_csv('UFC_data_with_elo.csv', nrows=1)
    if 'fight_id' in df.columns:
        print("fight_id EXISTS")
    else:
        print("fight_id MISSING")
        print(f"Columns: {df.columns.tolist()[:10]}...")

if __name__ == "__main__":
    check_fight_id()
