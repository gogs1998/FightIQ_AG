import requests
import json

def test_api():
    url = "http://127.0.0.1:8003/predict"
    
    # Use a known matchup from history to ensure features exist
    payload = {
        "f1_name": "Jon Jones",
        "f2_name": "Ciryl Gane",
        "f1_odds": 1.5,
        "f2_odds": 2.5
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\n=== API Response ===")
            print(json.dumps(data, indent=2))
            
            print("\nCheck the API console logs for 'M3 Prediction' output.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Make sure the API is running: python api.py")

if __name__ == "__main__":
    test_api()
