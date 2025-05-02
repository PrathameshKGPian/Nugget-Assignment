import requests
from bs4 import BeautifulSoup
import time
import re
import json

BASE_URL = "https://www.zomato.com"
LISTING_URL = "https://www.zomato.com/agra"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
}

def get_restaurant_links(listing_url):
    response = requests.get(listing_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "lxml")
    links = set()
    
    # Find all restaurant cards and extract info links
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/agra/') and href.endswith('/info'):
            links.add(BASE_URL + href)
    
    return list(links)

def extract_restaurant_info(info_url):
    response = requests.get(info_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "lxml")
    
    restaurant_info = {
        "name": get_restaurant_name(soup),
        "opening_hours": get_opening_hours(soup),
        "phone_number": get_phone_number(soup),
        "price_per_person": get_price_per_person(soup),
        "address": get_address(soup),
        "url": info_url
    }
    
    return restaurant_info

def get_restaurant_name(soup):
    # Find restaurant name (usually in h1 tag)
    name_tag = soup.find("h1")
    if name_tag:
        return name_tag.text.strip()
    return "Name not found"

def get_opening_hours(soup):
    # Find opening hours span (from your previous code)
    for span in soup.find_all("span"):
        if span.text.strip().endswith("(Today)"):
            return span.text.strip()
    return "Opening hours not found"

def get_phone_number(soup):
    # Look for phone number - typically in a span with a specific class
    # or near text like "Call" or "Phone"
    for a in soup.find_all("a", href=True):
        if "tel:" in a["href"]:
            return a["href"].replace("tel:", "")
    
    # Alternative approach - look for spans with phone number patterns
    phone_pattern = re.compile(r'\+?[0-9][0-9\s-]{7,}[0-9]')
    for span in soup.find_all("span"):
        match = phone_pattern.search(span.text)
        if match:
            return match.group()
    
    return "Phone number not found"

def get_address(soup):
    # Try the exact class combination first
    address_div = soup.find("div", class_="sc-clNaTc ckqQPM")
    if address_div and address_div.text.strip():
        return address_div.text.strip()
    # Fallback: search for the class prefix
    address_div = soup.find("div", class_=lambda x: x and "sc-clNaTc" in x)
    if address_div and address_div.text.strip():
        return address_div.text.strip()
    return "Address not found"

def get_price_per_person(soup):
    # Try the exact class combination first
    price_div = soup.find("div", class_="sc-bEjcJn ePRRqR")
    if price_div and price_div.text.strip():
        return price_div.text.strip()
    # Fallback: search for the class prefix
    price_div = soup.find("div", class_=lambda x: x and "sc-bEjcJn" in x)
    if price_div and price_div.text.strip():
        return price_div.text.strip()
    return "Price per person not found"


if __name__ == "__main__":
    restaurant_links = get_restaurant_links(LISTING_URL)
    print(f"Found {len(restaurant_links)} restaurants.")
    all_restaurant_info = []

    for url in restaurant_links:
        print(f"\nScraping: {url}")
        try:
            info = extract_restaurant_info(url)
            all_restaurant_info.append(info)
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
        time.sleep(1.5)

    # Save as JSON
    with open("restaurants_data.json", "w", encoding="utf-8") as f:
        json.dump(all_restaurant_info, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully scraped and saved {len(all_restaurant_info)} restaurants to restaurants_data.json.")
