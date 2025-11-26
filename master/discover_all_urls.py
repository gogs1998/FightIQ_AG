import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import json

def clean_name(name):
    return name.strip()

def discover_urls():
    print("Loading data...")
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter 2025 and missing odds
    mask_2025 = df['event_date'] >= '2025-01-01'
    missing_mask = mask_2025 & ((df['f_1_odds'].isna()) | (df['f_1_odds'] == 0))
    
    missing_events = df[missing_mask]['event_name'].unique()
    print(f"Found {len(missing_events)} events with missing odds.")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    discovered_urls = {}
    
    for event_name in missing_events:
        print(f"\nProcessing: {event_name}")
        
        # Get fighters from this event to search for
        # We'll try the main event fighters first (usually the last rows for that event)
        event_fights = df[df['event_name'] == event_name]
        
        # Try a few fighters from the card in case main event is obscure or new
        fighters_to_try = []
        
        # Add main event fighters
        if not event_fights.empty:
            last_fight = event_fights.iloc[-1]
            fighters_to_try.append(last_fight['f_1_name'])
            fighters_to_try.append(last_fight['f_2_name'])
            
            # Add co-main if available
            if len(event_fights) > 1:
                comain = event_fights.iloc[-2]
                fighters_to_try.append(comain['f_1_name'])
                fighters_to_try.append(comain['f_2_name'])
        
        url_found = False
        
        for fighter in fighters_to_try:
            if url_found:
                break
                
            print(f"  Searching via fighter: {fighter}...")
            
            try:
                # 1. Search for fighter
                search_url = f"https://www.bestfightodds.com/search?query={fighter.replace(' ', '+')}"
                resp = requests.get(search_url, headers=headers)
                
                if resp.status_code != 200:
                    print(f"    Failed search request: {resp.status_code}")
                    time.sleep(1)
                    continue
                    
                soup = BeautifulSoup(resp.text, 'html.parser')
                fighter_link = soup.find('a', href=re.compile(r'/fighters/'))
                
                if not fighter_link:
                    print(f"    Profile not found.")
                    time.sleep(0.5)
                    continue
                
                profile_url = f"https://www.bestfightodds.com{fighter_link['href']}"
                
                # 2. Scrape profile
                resp_profile = requests.get(profile_url, headers=headers)
                soup_profile = BeautifulSoup(resp_profile.text, 'html.parser')
                
                # Look for event link
                # Strategy: Look for link containing part of event name or date?
                # BFO event names might differ slightly.
                # Let's try to match the event number (e.g. 317) or a significant word from event name.
                
                # Extract key terms from event name
                # e.g. "UFC Fight Night: Usman vs. Buckley" -> "Usman", "Buckley"
                # e.g. "UFC 317: Topuria vs. Oliveira" -> "317"
                
                keywords = []
                if "UFC" in event_name:
                    # Check for number
                    match = re.search(r'UFC\s+(\d+)', event_name)
                    if match:
                        keywords.append(match.group(1)) # "317"
                    else:
                        # Fight Night, extract names
                        parts = event_name.split(':')
                        if len(parts) > 1:
                            names = parts[1].split(' vs. ')
                            for n in names:
                                keywords.append(n.strip().split()[-1]) # Last name
                else:
                    # Other orgs
                    keywords.append(event_name.split()[0]) # First word e.g. "PFL", "Bellator"
                
                for link in soup_profile.find_all('a', href=re.compile(r'/events/')):
                    href = link['href']
                    text = link.text
                    
                    # Check if any keyword is in the href or text
                    for kw in keywords:
                        if kw.lower() in href.lower() or kw.lower() in text.lower():
                            full_url = f"https://www.bestfightodds.com{href}"
                            print(f"    [SUCCESS] Found URL: {full_url}")
                            discovered_urls[event_name] = full_url
                            url_found = True
                            break
                    if url_found:
                        break
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    Error: {e}")
                
        if not url_found:
            print(f"  [FAILURE] Could not find URL for {event_name}")
            
    # Save results
    print(f"\nDiscovered {len(discovered_urls)} URLs.")
    with open('discovered_urls.json', 'w') as f:
        json.dump(discovered_urls, f, indent=2)
        
    # Also print for copy-paste
    print("\nKNOWN_URLS update:")
    for k, v in discovered_urls.items():
        print(f'    "{k}": "{v}",')

if __name__ == "__main__":
    discover_urls()
