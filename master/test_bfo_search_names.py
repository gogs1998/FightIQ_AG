import requests
from bs4 import BeautifulSoup
import re

def test_search():
    query = "Pereira vs Ankalaev"
    url = f"https://www.bestfightodds.com/search?query={query.replace(' ', '+')}"
    print(f"Fetching {url}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find event links
            links = soup.find_all('a', href=re.compile(r'/events/'))
            
            print(f"Found {len(links)} event links.")
            
            for link in links:
                print(f"{link.text.strip()} -> {link['href']}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
