import requests
import json

API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'

def debug_api():
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=us&markets=h2h&oddsFormat=decimal'
    response = requests.get(url)
    data = response.json()
    
    print(f"Total Events: {len(data)}")
    
    titles = set()
    for e in data:
        titles.add(e['sport_title'])
        
    print("Sport Titles Found:")
    for t in titles:
        print(f" - {t}")
        
    # Print first event
    if data:
        print("\nSample Event:")
        print(json.dumps(data[0], indent=2))

if __name__ == "__main__":
    debug_api()
