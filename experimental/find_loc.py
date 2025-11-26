import pandas as pd

def find_location_cols():
    df = pd.read_csv('UFC_full_data_golden.csv', nrows=1)
    cols = df.columns.tolist()
    
    keywords = ['location', 'city', 'country', 'place', 'venue', 'arena', 'state']
    found = []
    for c in cols:
        if any(k in c.lower() for k in keywords):
            found.append(c)
            
    print(f"Location-related columns: {found}")

if __name__ == "__main__":
    find_location_cols()
