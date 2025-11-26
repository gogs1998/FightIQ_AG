import requests
from bs4 import BeautifulSoup
import re

def find_event_urls():
    # Map Target Event -> Fighter to search
    targets = {
        "UFC 317": "Ilia Topuria",
        "UFC 318": "Max Holloway",
        "UFC 319": "Dricus Du Plessis",
        "UFC 320": "Magomed Ankalaev"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    found_urls = {}
    
    for event_name, fighter_name in targets.items():
        print(f"\nSearching for {event_name} via {fighter_name}...")
        
        # 1. Search for fighter
        search_url = f"https://www.bestfightodds.com/search?query={fighter_name.replace(' ', '+')}"
        try:
            resp = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find fighter link (usually /fighters/Name-ID)
            fighter_link = soup.find('a', href=re.compile(r'/fighters/'))
            
            if not fighter_link:
                print(f"  Could not find profile for {fighter_name}")
                continue
                
            profile_url = f"https://www.bestfightodds.com{fighter_link['href']}"
            print(f"  Found profile: {profile_url}")
            
            # 2. Scrape profile for event
            resp_profile = requests.get(profile_url, headers=headers)
            soup_profile = BeautifulSoup(resp_profile.text, 'html.parser')
            
            # Look for event link
            # Event links are usually /events/ufc-317-...
            # We search for the event number in the link text or href
            
            event_link = soup_profile.find('a', href=re.compile(rf'/events/.*{event_name.replace(" ", "-").lower()}.*'))
            
            if not event_link:
                # Try looser match (just number)
                event_num = event_name.split()[-1]
                event_link = soup_profile.find('a', href=re.compile(rf'/events/.*{event_num}.*'))
            
            if event_link:
                full_event_url = f"https://www.bestfightodds.com{event_link['href']}"
                print(f"  [SUCCESS] Found {event_name}: {full_event_url}")
                found_urls[event_name] = full_event_url
            else:
                print(f"  Could not find {event_name} in fight history.")
                
        except Exception as e:
            print(f"  Error: {e}")
            
    print("\n=== Found URLs ===")
    for k, v in found_urls.items():
        print(f'"{k}": "{v}",')

if __name__ == "__main__":
    find_event_urls()
