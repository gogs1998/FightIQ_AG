import requests
import json

API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'
# SPORT = 'ufc' # older api might use this, but v4 usually uses mma_mixed_martial_arts

def test_api():
    print(f"Testing API Key: {API_KEY}")
    
    # 1. Check usage/status (via sports endpoint)
    url = f'https://api.the-odds-api.com/v4/sports/?apiKey={API_KEY}'
    try:
        response = requests.get(url)
        print(f"Sports Endpoint Status: {response.status_code}")
        
        if response.status_code == 200:
            print("Key is valid.")
            print(f"Requests Remaining: {response.headers.get('x-requests-remaining')}")
            print(f"Requests Used: {response.headers.get('x-requests-used')}")
        else:
            print(f"Error: {response.text}")
            return

    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # 2. Try to fetch historical odds (sample)
    # We need a date. Let's try a recent past date, e.g., Jan 18, 2025 (UFC 311)
    # Format: YYYY-MM-DDTHH:MM:SSZ
    date = '2025-01-18T12:00:00Z'
    
    hist_url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds-history/?apiKey={API_KEY}&regions=us&markets=h2h&date={date}'
    
    print(f"\nTesting Historical Access for {date}...")
    try:
        response = requests.get(hist_url)
        print(f"Historical Endpoint Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Success! Historical data received.")
            # print(json.dumps(data, indent=2)[:500]) # Print snippet
            print(f"Timestamp: {data.get('timestamp')}")
            print(f"Events found: {len(data.get('data', []))}")
            if data.get('data'):
                print(f"Sample Event: {data['data'][0]['home_team']} vs {data['data'][0]['away_team']}")
        else:
            print(f"Error fetching history: {response.text}")
            
    except Exception as e:
        print(f"Historical Request Error: {e}")

if __name__ == "__main__":
    test_api()
