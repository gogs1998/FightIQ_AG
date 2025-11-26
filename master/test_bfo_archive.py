import requests
from bs4 import BeautifulSoup
import re

def scrape_archive():
    base_url = "https://www.bestfightodds.com/archive"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    target_event = "UFC 313"
    found = False
    
    for page in range(1, 11):
        url = f"{base_url}?page={page}"
        print(f"Fetching {url}...")
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch page {page}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'/events/'))
            
            print(f"  Found {len(links)} events.")
            
            for link in links:
                text = link.text.strip()
                if target_event in text:
                    print(f"\n[SUCCESS] Found {text} -> {link['href']}")
                    found = True
                    # Don't break immediately, see if there are duplicates
            
            if found:
                break
                
        except Exception as e:
            print(f"Error on page {page}: {e}")
            
    if not found:
        print(f"\n[FAILURE] Could not find {target_event} in first 10 pages.")

if __name__ == "__main__":
    scrape_archive()
