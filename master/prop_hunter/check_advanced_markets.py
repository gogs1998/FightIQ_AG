import requests
import json

# API Config
API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'

def check_advanced_markets():
    print("=== Checking Paid API Markets (Props) ===")
    
    # Try to fetch specific prop markets
    # Common keys for paid plans: 
    # 'h2h', 'spreads', 'totals', 'outrights'
    # 'alternate_totals', 'alternate_spreads'
    # 'player_props'? No, usually for team sports.
    # For MMA, 'totals' = Over/Under Rounds.
    
    # Let's try to fetch 'totals' first (Over/Under)
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=us&markets=h2h,totals&oddsFormat=decimal'
    
    print(f"Requesting: {url}")
    response = requests.get(url)
    data = response.json()
    
    if 'message' in data:
        print(f"API Message: {data['message']}")
        
    if isinstance(data, list) and len(data) > 0:
        print(f"Found {len(data)} events.")
        
        # Look for an event with 'totals'
        found_totals = False
        for event in data:
            for book in event['bookmakers']:
                for market in book['markets']:
                    if market['key'] == 'totals':
                        print(f"✅ FOUND TOTALS (Over/Under) for {event['home_team']} vs {event['away_team']}")
                        print(f"   Book: {book['title']}")
                        print(f"   Lines: {[o['point'] for o in market['outcomes']]}")
                        found_totals = True
                        break
                if found_totals: break
            if found_totals: break
            
        if not found_totals:
            print("❌ No 'totals' markets found in current events (might be too early for lines).")
            
    else:
        print("No events found.")

if __name__ == "__main__":
    check_advanced_markets()
