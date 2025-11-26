import requests
import json

url = "http://localhost:8001/predict"
payload = {
    "f1_name": "Islam Makhachev",
    "f2_name": "Charles Oliveira",
    "f1_odds": 1.3,
    "f2_odds": 3.5
}

try:
    # Check Docs
    print("Checking /docs...")
    resp = requests.get("http://127.0.0.1:8003/docs")
    print(f"/docs Status: {resp.status_code}")

    # Check Predict
    print("Checking /predict...")
    response = requests.post("http://127.0.0.1:8003/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
