import requests
import json

def test_api():
    url = "http://127.0.0.1:8003/predict"
    
    payload = {
        "f1_name": "Jon Jones",
        "f2_name": "Tom Aspinall",
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
            
            # Verify Props Fields
            if "pred_method" in data and "min_odds" in data:
                print("\nSUCCESS: Master Props fields found!")
                print(f"Method: {data['pred_method']}")
                print(f"Round: {data['pred_round']}")
                print(f"Trifecta Prob: {data['trifecta_prob']:.1%}")
                print(f"Min Odds: {data['min_odds']}")
            else:
                print("\nFAILURE: Master Props fields missing.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Make sure the API is running: python api.py")

if __name__ == "__main__":
    test_api()
