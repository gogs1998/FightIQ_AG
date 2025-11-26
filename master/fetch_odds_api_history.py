import pandas as pd
import requests
import time
import json
from datetime import timedelta

API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'

def clean_name(name):
    return name.strip().lower()

def fetch_odds_for_date(date_str):
    # date_str should be YYYY-MM-DD
    # API expects ISO format, e.g. YYYY-MM-DDTHH:MM:SSZ
    # We'll ask for odds at noon UTC on the event day
    iso_date = f"{date_str}T12:00:00Z"
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds-history/?apiKey={API_KEY}&regions=us&markets=h2h&date={iso_date}'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def update_from_api():
    print("Loading data...")
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter for ALL 2025 events (Overwrite mode)
    # User wants to ensure 2025 odds are correct, so we fetch everything.
    # Filter for 2020-2025
    mask_target = df['event_date'] >= '2020-01-01'
    target_dates = df[mask_target]['event_date'].unique()
    
    print(f"Found {len(target_dates)} event dates from 2020-2025 to check.")
    
    updated_count = 0
    
    for date_ts in target_dates:
        date_str = date_ts.strftime('%Y-%m-%d')
        # print(f"Processing {date_str}...") # Reduce spam
        
        # Check if we need to fetch for this date
        # We need to fetch if:
        # 1. It's 2025 (to overwrite/fix format)
        # 2. Any fight on this date is missing odds
        
        day_mask = (df['event_date'] == date_ts)
        day_fights = df[day_mask]
        
        needs_fetch = False
        if date_ts.year == 2025:
            needs_fetch = True
        else:
            if day_fights['f_1_odds'].isna().any():
                needs_fetch = True
        
        if not needs_fetch:
            continue
            
        print(f"Fetching odds for {date_str}...")
        data = fetch_odds_for_date(date_str)
        if not data or 'data' not in data:
            print("  No data found.")
            continue
            
        events = data['data']
        # print(f"  API returned {len(events)} events.")
        
        # Create a map of matches for this date
        odds_map = {}
        
        for event in events:
            home = clean_name(event['home_team'])
            away = clean_name(event['away_team'])
            
            if not event['bookmakers']:
                continue
                
            # Try to find a major bookmaker
            book = event['bookmakers'][0] # Default
            for b in event['bookmakers']:
                if b['key'] in ['draftkings', 'fanduel', 'bovada', 'williamhill_us']:
                    book = b
                    break
            
            h2h = None
            for m in book['markets']:
                if m['key'] == 'h2h':
                    h2h = m['outcomes']
                    break
            
            if h2h:
                o1 = h2h[0]['price']
                n1 = clean_name(h2h[0]['name'])
                o2 = h2h[1]['price']
                n2 = clean_name(h2h[1]['name'])
                
                odds_map[f"{n1}_vs_{n2}"] = (o1, o2, n1, n2)
                odds_map[f"{n2}_vs_{n1}"] = (o2, o1, n2, n1)
        
        # Update dataframe
        for idx, row in day_fights.iterrows():
            # Logic: Overwrite if 2025, otherwise only if missing
            if row['event_date'].year != 2025 and pd.notna(row['f_1_odds']):
                continue

            f1 = clean_name(row['f_1_name'])
            f2 = clean_name(row['f_2_name'])
            
            key = f"{f1}_vs_{f2}"
            
            if key in odds_map:
                o1_dec, o2_dec, n1_api, n2_api = odds_map[key]
                
                # Match f1 to the correct API name
                if f1 == n1_api:
                    new_o1 = o1_dec
                    new_o2 = o2_dec
                else:
                    new_o1 = o2_dec
                    new_o2 = o1_dec
                
                # Update
                df.at[idx, 'f_1_odds'] = new_o1
                df.at[idx, 'f_2_odds'] = new_o2
                    
                print(f"    Updated: {f1} vs {f2} -> {new_o1}/{new_o2}")
                updated_count += 1
            else:
                pass
                
        time.sleep(0.5)
        
    print(f"\nUpdated {updated_count} fights.")
    df.to_csv('data/training_data.csv', index=False)
    print("Saved updated training_data.csv")

if __name__ == "__main__":
    update_from_api()
