import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

# Known URLs for major 2025 events (found via browser)
KNOWN_URLS = {
    "UFC 311": "https://www.bestfightodds.com/events/ufc-311-3596",
    "UFC 312": "https://www.bestfightodds.com/events/ufc-312-3620",
    "UFC 313": "https://www.bestfightodds.com/events/ufc-313-3650",
    "UFC 315": "https://www.bestfightodds.com/events/ufc-315-3708",
    "UFC 316": "https://www.bestfightodds.com/events/ufc-316-3702",
    "UFC 319": "https://www.bestfightodds.com/events/ufc-319-3800",
    "UFC 320": "https://www.bestfightodds.com/events/ufc-320-odds-3779",
    "UFC Fight Night: Dern vs. Ribas 2": "https://www.bestfightodds.com/events/ufc-3277",
    "UFC Fight Night: Adesanya vs. Imavov": "https://www.bestfightodds.com/events/ufc-276-adesanya-vs-cannonier-2478",
    "UFC Fight Night: Vettori vs. Dolidze 2": "https://www.bestfightodds.com/events/ufc-298-3112",
    "UFC Fight Night: Edwards vs. Brady": "https://www.bestfightodds.com/events/ufc-278-usman-vs-edwards-2-2545",
    "UFC Fight Night: Usman vs. Buckley": "https://www.bestfightodds.com/events/ufc-258-usman-vs-burns-2033",
    "UFC 317: Topuria vs. Oliveira": "https://www.bestfightodds.com/events/ufc-3174",
    "UFC Fight Night: Lewis vs. Teixeira": "https://www.bestfightodds.com/events/ufc-fight-night-lewis-vs-spivak-2747",
    "UFC Fight Night: Taira vs. Park": "https://www.bestfightodds.com/events/ufc-fight-night-albazi-vs-taira-odds-3777",
    "UFC Fight Night: Dolidze vs. Hernandez": "https://www.bestfightodds.com/events/ufc-fight-night-dolidze-vs-hernandez-odds-3789",
    "UFC Fight Night: Walker vs. Zhang": "https://www.bestfightodds.com/events/ufc-fight-night-santos-vs-walker-2167",
    # Add more as needed or implement search
}

def clean_name(name):
    return name.strip().lower()

def scrape_event_odds(event_url):
    print(f"Scraping {event_url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(event_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {event_url}: {response.status_code}")
            return {}
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Debug: Print title to confirm page load
        print(f"Page Title: {soup.title.text.strip() if soup.title else 'No Title'}")
        
        odds_data = {}
        
        # Try multiple selectors
        rows = soup.select('table.odds-table tbody tr')
        if not rows:
            print("No rows found with 'table.odds-table tbody tr'")
            # Try finding any table
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables.")
        
        for row in rows:
            fighter_cell = row.select_one('th.t-b-fcc')
            if not fighter_cell:
                # Try finding just a class 't-b-fcc' anywhere in row
                fighter_cell = row.find(class_='t-b-fcc')
            
            if not fighter_cell:
                continue
                
            fighter_name = fighter_cell.text.strip()
            # print(f"Found fighter: {fighter_name}")
            
            odds = []
            for td in row.select('td'):
                text = td.text.strip()
                text = re.sub(r'[^\d\.\+\-]', '', text)
                if text and (text.startswith('+') or text.startswith('-') or text.isdigit()):
                    try:
                        val = float(text)
                        # Filter out unlikely odds (e.g. round numbers like 1, 2, 3 if they are not odds)
                        # BFO odds are usually > 1.0 (decimal) or +/- 100 (moneyline)
                        # If decimal, usually < 100. If moneyline, usually > 100 or < -100.
                        odds.append(val)
                    except:
                        pass
            
            if odds:
                # print(f"  Odds found: {odds}")
                odds_data[clean_name(fighter_name)] = odds[0]
        
        print(f"Extracted odds for {len(odds_data)} fighters.")
        return odds_data
        
    except Exception as e:
        print(f"Error scraping {event_url}: {e}")
        return {}

def update_odds():
    df = pd.read_csv('data/training_data.csv')
    
    # Identify events with missing odds in 2025
    mask_2025 = pd.to_datetime(df['event_date']) >= '2025-01-01'
    missing_mask = mask_2025 & ((df['f_1_odds'].isna()) | (df['f_1_odds'] == 0) | (df['f_2_odds'].isna()) | (df['f_2_odds'] == 0))
    
    events_to_update = df[missing_mask]['event_name'].unique()
    print(f"Found {len(events_to_update)} events with missing odds.")
    
    updated_count = 0
    
    for event_name in events_to_update:
        # Match with known URLs
        url = None
        for key in KNOWN_URLS:
            if key in event_name:
                url = KNOWN_URLS[key]
                break
        
        if not url:
            print(f"Skipping {event_name} (No URL found)")
            continue
            
        odds_map = scrape_event_odds(url)
        if not odds_map:
            continue
            
        # Update dataframe
        event_mask = (df['event_name'] == event_name)
        
        for idx, row in df[event_mask].iterrows():
            f1 = clean_name(row['f_1_name'])
            f2 = clean_name(row['f_2_name'])
            
            # Fuzzy match or direct match
            # Simple direct match for now
            
            if f1 in odds_map:
                df.at[idx, 'f_1_odds'] = odds_map[f1]
                print(f"  Updated odds for {f1}: {odds_map[f1]}")
            if f2 in odds_map:
                df.at[idx, 'f_2_odds'] = odds_map[f2]
                print(f"  Updated odds for {f2}: {odds_map[f2]}")
                
        updated_count += 1
        time.sleep(1) # Be polite
        
    print(f"Updated {updated_count} events.")
    df.to_csv('data/training_data.csv', index=False)
    print("Saved updated training_data.csv")

if __name__ == "__main__":
    update_odds()
