import json
with open('fighter_db.json', 'r') as f:
    db = json.load(f)

fighters = ["Jon Jones", "Stipe Miocic", "Conor McGregor", "Islam Makhachev"]
for f in fighters:
    print(f"{f}: {'Found' if f in db else 'Not Found'}")
