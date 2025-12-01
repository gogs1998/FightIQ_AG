import requests
import json

def test_api():
    url = "http://localhost:8004/predict"
    
    payload = {
        "f1_name": "Jon Jones",
        "f2_name": "Stipe Miocic",
        "f1_odds": 1.5,
        "f2_odds": 2.5
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("\nAPI Response:")
        print(json.dumps(data, indent=2))
        
        # assertions
        assert "winner" in data
        assert "pred_method" in data
        assert "pred_round" in data
        assert "trifecta_prob" in data
        
        print("\nSUCCESS: API returned valid Trifecta prediction.")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        if 'response' in locals():
            print(response.text)

if __name__ == "__main__":
    test_api()
