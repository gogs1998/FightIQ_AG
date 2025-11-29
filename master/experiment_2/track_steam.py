import requests
import pandas as pd
import time
import os
import json
from datetime import datetime

# API Config
API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'
REGIONS = 'uk'
MARKETS = 'h2h'
ODDS_FORMAT = 'decimal'
DATA_DIR = 'data/odds_history'

def track_steam():
    print("=== FightIQ Steam Chaser (Odds Movement Tracker) ===")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # 1. Fetch Current Odds
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}'
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"API Error: {response.text}")
            return
        data = response.json()
    except Exception as e:
        print(f"Connection Error: {e}")
        return
        
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Fetched {len(data)} events at {timestamp}")
    
    # 2. Save Snapshot
    snapshot_file = os.path.join(DATA_DIR, f'odds_{timestamp}.json')
    with open(snapshot_file, 'w') as f:
        json.dump(data, f)
        
    # 3. Compare with Previous Snapshot (if exists)
    # Find most recent previous file
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('odds_') and f != f'odds_{timestamp}.json'])
    
    if not files:
        print("No previous history to compare. This is the baseline snapshot.")
        return
        
    prev_file = os.path.join(DATA_DIR, files[-1])
    with open(prev_file, 'r') as f:
        prev_data = json.load(f)
        
    print(f"Comparing against {files[-1]}...")
    
    # Build Map of Previous Odds
    # Key: (EventID, FighterName, Bookmaker) -> Odds
    prev_map = {}
    for event in prev_data:
        eid = event['id']
        for book in event['bookmakers']:
            bname = book['key']
            for m in book['markets']:
                if m['key'] == 'h2h':
                    for outcome in m['outcomes']:
                        key = (eid, outcome['name'], bname)
                        prev_map[key] = outcome['price']
                        
    # Detect Movement
    movements = []
    
    for event in data:
        eid = event['id']
        for book in event['bookmakers']:
            bname = book['key']
            for m in book['markets']:
                if m['key'] == 'h2h':
                    for outcome in m['outcomes']:
                        key = (eid, outcome['name'], bname)
                        current_price = outcome['price']
                        
                        if key in prev_map:
                            prev_price = prev_map[key]
                            diff = current_price - prev_price
                            pct_change = (diff / prev_price) * 100
                            
                            if abs(pct_change) > 2.0: # 2% movement threshold
                                movements.append({
                                    "Fighter": outcome['name'],
                                    "Book": bname,
                                    "Old": prev_price,
                                    "New": current_price,
                                    "Change": f"{pct_change:+.1f}%"
                                })
                                
    # Report Steam
    if movements:
        print(f"\n🚨 DETECTED {len(movements)} SIGNIFICANT ODDS MOVEMENTS 🚨")
        print(f"{'Fighter':<20} | {'Book':<15} | {'Old':<6} | {'New':<6} | {'Change':<8}")
        print("-" * 65)
        for m in movements:
            print(f"{m['Fighter']:<20} | {m['Book']:<15} | {m['Old']:<6.2f} | {m['New']:<6.2f} | {m['Change']:<8}")
            
        # Strategy Logic:
        # If odds drop (e.g. 2.00 -> 1.80), that is "Steam" (Money coming in).
        # We want to FOLLOW steam if it's early, or FADE it if it's an overreaction.
        # For now, we just log it.
    else:
        print("No significant movement detected.")

if __name__ == "__main__":
    track_steam()
