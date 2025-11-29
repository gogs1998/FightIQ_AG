import pandas as pd
import requests
import time
import json
import os
from datetime import timedelta

# API Config
API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'
# Markets: h2h (moneyline), h2h_lay (exchange), outrights
# Props are usually under 'alternate' markets or specific keys like 'method_of_victory'
# BUT The Odds API historical endpoint might restrict props access or use different keys.
# Common keys: 'h2h', 'totals', 'spreads'.
# Props like 'method_of_victory' might be available if we request them.
# Let's try requesting 'all' or specific prop markets.
# According to docs, markets param can be 'h2h,spreads,totals,outrights'.
# Props might be behind a paywall or different endpoint.
# Let's try fetching a recent event with 'markets=all' or similar to see what's available.
# Actually, standard plan often includes 'h2h', 'spreads', 'totals'. 
# 'player_props' or 'fight_props' might be separate.

def check_prop_availability():
    print("=== Checking Prop Availability on Odds API ===")
    
    # Check a recent big event (e.g. UFC 309 on Nov 16 2024)
    # or just current odds to see structure
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=us&markets=h2h,totals,spreads&oddsFormat=decimal'
    
    response = requests.get(url)
    data = response.json()
    
    if not data:
        print("No current events found.")
        return
        
    print(f"Found {len(data)} current events.")
    sample = data[0]
    print(f"Sample Event: {sample['sport_title']} - {sample['home_team']} vs {sample['away_team']}")
    
    # Check bookmakers for markets
    for book in sample['bookmakers']:
        print(f"  Bookmaker: {book['title']}")
        for market in book['markets']:
            print(f"    Market: {market['key']}")
            
    # Note: 'totals' usually means Over/Under rounds.
    # If we have 'totals', we can backtest the Round Model (Over/Under).
    # Method props (KO/Sub) are harder to find on standard API tier.

if __name__ == "__main__":
    check_prop_availability()
