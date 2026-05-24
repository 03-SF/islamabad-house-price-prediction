"""
ZAMEEN.COM PROPERTY SCRAPER
===========================

This script scrapes property listings from Zameen.com for Islamabad.
It collects 300-400+ property listings with detailed information.

WHAT THIS SCRAPER DOES:
1. Visits search result pages to find property listing URLs
2. For each property, downloads its detail page and extracts:
   - Price (numeric and text)
   - Area (size of property)
   - City, Location, Property Type
   - Number of bedrooms, bathrooms
   - Built-in year
   - Amenities: parking, servant quarters, store rooms, kitchens, etc.

OUTPUT:
- zameen_results.csv: Excel-friendly file with all properties and their details

HOW TO RUN:
-----------
1. Install required libraries:
       pip install curl_cffi beautifulsoup4 pandas lxml

2. (Optional) Edit CONFIG section (CELL 2) to change:
       - SEARCH_URL: Different city or property type
       - MAX_PAGES: Number of search pages (each page ~100+ listings)
       - MAX_LISTINGS: Total listings to scrape (300-400)

3. Run the script:
       python zameen_scraper.py
       
   Wait for completion. The script will:
   - Show progress for each search page visited
   - Show progress for each property page scraped
   - Save results to zameen_results.csv

NOTES:
- The script uses curl_cffi to bypass Cloudflare protection
- It waits 3-4 seconds between requests to be respectful to the server
- If you get blocked, either wait and retry or use a VPN
"""

# =============================================================================
# CELL 1 — Imports & Setup
# =============================================================================
import re
import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urlparse

try:
    from curl_cffi import requests        # best — bypasses Cloudflare bot detection
    _SESSION = requests.Session(impersonate="chrome124")
    _BACKEND = "curl_cffi  ✅ (Cloudflare bypass active)"
except ImportError:
    import requests as _req_fallback      # fallback — may get blocked on some pages
    _SESSION = _req_fallback.Session()
    _SESSION.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    _BACKEND = "requests (fallback) ⚠️  — run: pip install curl_cffi"

print(f"✅ Imports OK")
print(f"   HTTP backend: {_BACKEND}")

# =============================================================================
# CELL 2 — CONFIG  (edit this block before running)
# =============================================================================

# This is where you configure the scraper
SEARCH_URL   = "https://www.zameen.com/Houses_Property/Islamabad-3-1.html"
# Increase pages to cover more listings; adjust if you hit blocks
MAX_PAGES    = 15     # Number of search pages to visit (each page ~100+ listings)
MAX_LISTINGS = 400    # Total listings to scrape (300-400 as per requirement)
DELAY        = 3.0    # Wait time between requests (seconds) - be respectful to server
JITTER       = 1.0    # Random variation in wait time to look natural
OUTPUT_CSV   = "zameen_results.csv"

print(f"\n📍 Target   : {SEARCH_URL}")
print(f"📄 Pages    : {MAX_PAGES}  |  🏠 Target listings: {MAX_LISTINGS}  |  ⏳ Delay: {DELAY}s")

# =============================================================================
# CELL 3 — HTTP helpers (for fetching web pages)
# =============================================================================

def get_page(url: str) -> BeautifulSoup:
    """
    Download a webpage and return it as a parsed HTML object.
    Raises RuntimeError if blocked by Cloudflare.
    """
    resp = _SESSION.get(url, timeout=20, allow_redirects=True)
    _LISTING_RE = re.compile(r"/Property/", re.I)  # Original regex for context
    html = resp.text
    if "cf-mitigated" in html or ("captcha" in html.lower() and "cloudflare" in html.lower()):
        raise RuntimeError(
            f"Cloudflare blocked the request.\n"
            f"  URL: {url}\n"
            f"  Fix: pip install curl_cffi   (or connect via VPN and retry)"
        )
    return BeautifulSoup(html, "lxml")


def polite_wait():
    """Wait a few seconds so we don't hammer the server with requests."""
    time.sleep(max(0.5, DELAY + random.uniform(-JITTER, JITTER)))


def fix_url(href: str) -> str:
    """Convert a relative URL into a complete web address."""
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://www.zameen.com" + href
    return "https://www.zameen.com/" + href.lstrip("/")


print("✅ HTTP helpers ready")

# =============================================================================
# CELL 4 — Search-page crawler (collects property detail URLs)
# =============================================================================

_LISTING_RE = re.compile(r"/Property/", re.I)


def links_on_page(soup: BeautifulSoup) -> list[str]:
    """
    Find all property listing URLs on a search results page.
    Each URL leads to a detail page for one property.
    """
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Normalize and skip JavaScript/mailto links
        if href.startswith("javascript:") or href.startswith("mailto:"):
            continue

        full = fix_url(href)
        parsed = urlparse(full)

        # Only accept links that are on zameen.com (or relative links resolved)
        if parsed.netloc and "zameen.com" not in parsed.netloc:
            continue

        # Require the path to contain the Property segment and end with .html
        # This avoids social-share wrappers which sometimes include '/Property/' in query strings
        if "/Property/" in parsed.path and parsed.path.lower().endswith('.html'):
            if full not in seen:
                seen.add(full)
                links.append(full)
    return links


def find_next_page(soup: BeautifulSoup, current_url: str) -> Optional[str]:
    """Find the URL for the next page of search results."""
    # Try to find <link rel="next">
    tag = soup.find("link", rel=lambda v: v and "next" in v)
    if tag and tag.get("href"):
        return fix_url(tag["href"])
    
    # Try to find a "Next" button
    for a in soup.find_all("a", href=True):
        label = (a.get("aria-label") or a.get_text(" ")).lower()
        if "next" in label:
            return fix_url(a["href"])
    
    # Try incrementing the page number in URL (e.g., -1.html → -2.html)
    m = re.search(r"-(\d+)\.html$", current_url)
    if m:
        return re.sub(r"-(\d+)\.html$", f"-{int(m.group(1)) + 1}.html", current_url)
    
    return None


def collect_listing_urls(start_url: str, max_pages: int) -> list[str]:
    """
    Visit multiple search result pages and collect URLs of all properties.
    Removes duplicate URLs.
    """
    gathered, page_url = [], start_url
    for page_num in range(1, max_pages + 1):
        print(f"  🔍 Search page {page_num}: {page_url}")
        soup = get_page(page_url)
        found = links_on_page(soup)
        print(f"     → {len(found)} property URLs found on this page")
        gathered.extend(found)
        
        nxt = find_next_page(soup, page_url)
        if not nxt:
            print("     (no next page — stopping)")
            break
        
        polite_wait()
        page_url = nxt
    
    # Remove duplicates while keeping discovery order
    seen, unique = set(), []
    for u in gathered:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    
    return unique


print("✅ Crawler ready")

# =============================================================================
# CELL 5 — Property data model (structure for storing property information)
# =============================================================================

@dataclass
class Property:
    """Stores all information about one property listing"""
    # Basic info
    url:               str                      # Link to the property page
    title:             Optional[str]   = None   # Property name/title
    
    # Price information
    price_text:        Optional[str]   = None   # Price as shown on website (e.g., "PKR 10 Crore")
    price_pkr:         Optional[float] = None   # Price converted to numeric PKR value
    
    # Location
    location:          Optional[str]   = None   # Full address/location description
    city:              Optional[str]   = None   # City name
    
    # Property basic features
    property_type:     Optional[str]   = None   # House, Apartment, Plot, etc.
    bedrooms:          Optional[int]   = None   # Number of bedrooms
    bathrooms:         Optional[int]   = None   # Number of bathrooms
    area:              Optional[str]   = None   # Property size (Kanal, Marla, Sq.ft, etc.)
    built_in_year:     Optional[str]   = None   # Year the property was built
    
    # Amenities
    parking_spaces:    Optional[str]   = None   # Parking availability
    servant_quarters:  Optional[str]   = None   # Servant quarters
    store_rooms:       Optional[str]   = None   # Store rooms
    kitchens:          Optional[str]   = None   # Number of kitchens
    drawing_room:      Optional[str]   = None   # Drawing room present?
    dining_room:       Optional[str]   = None   # Dining room present?
    study_room:        Optional[str]   = None   # Study room present?
    prayer_room:       Optional[str]   = None   # Prayer room present?
    powder_room:       Optional[str]   = None   # Powder room present?
    lounge_or_sitting: Optional[str]   = None   # Lounge or sitting area?
    
    # Additional info
    description:       Optional[str]   = None   # Property description

print("✅ Property model ready")

# =============================================================================
# CELL 6 — Price parser (converts text like "PKR 4.8 Crore" to numbers)
# =============================================================================

_MULTIPLIERS = {
    "crore":    1e7,       # 1 Crore = 10 Million
    "lakh":     1e5,       # 1 Lakh = 100 Thousand
    "million":  1e6,
    "thousand": 1e3,
    "k":        1e3,       # Abbreviation for thousand
    "m":        1e6,       # Abbreviation for million
    "b":        1e9,       # Abbreviation for billion
}


def parse_price(text: str) -> Optional[float]:
    """
    Convert price text to a number.
    Examples:
      'PKR 4.8 Crore' → 48000000.0
      'PKR 50 Lakh' → 5000000.0
    Returns None if the price cannot be parsed.
    """
    if not text:
        return None
    
    text = text.replace(",", "").replace("\xa0", " ")
    m = re.search(r"([\d.]+)\s*(crore|lakh|million|thousand|k\b|m\b|b\b)?", text, re.I)
    if not m:
        return None
    
    try:
        base = float(m.group(1))
    except ValueError:
        return None
    
    suffix = (m.group(2) or "").lower()
    return base * _MULTIPLIERS.get(suffix, 1)


print("✅ Price parser ready")

# =============================================================================
# CELL 7 — Detail page parser (extracts information from a property page)
# =============================================================================

def _text(tag) -> str:
    """Extract and clean text from an HTML element."""
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip() if tag else ""


def _first_int(s: str) -> Optional[int]:
    """Find the first number in a string. Returns None if no number found."""
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def _extract_num_or_yes(txt: str) -> str:
    """If text contains a number, return it. Otherwise return 'Yes'."""
    m = re.search(r"\d+", txt)
    return m.group() if m else "Yes"


def parse_property_page(url: str, city: Optional[str] = None) -> Property:
    """
    Download and parse one property detail page.
    Extracts title, price, location, features, and amenities.
    """
    soup = get_page(url)
    prop = Property(url=url, city=city)

    # Extract title (usually in an H1 heading)
    prop.title = _text(soup.select_one("h1.aea614fd, h1[class*='title']"))

    # Extract location/address
    prop.location = _text(soup.select_one("div.cd230541, div[class*='location']"))

    # Extract details from the info block (bedrooms, bathrooms, area, price, etc.)
    for li in soup.select("div._83bb17d1 ul li, ul._3dc8d08d li"):
        lbl_tag = li.find("span", class_="ed0db22a") or li.find("span", class_=re.compile("label"))
        val_tag = li.find("span", class_="_2fdf7fc5") or li.find("span", class_=re.compile("value"))
        label = _text(lbl_tag).lower()
        value = _text(val_tag) if val_tag else _text(li)

        # Match labels and extract values
        if "price" in label:
            prop.price_text = value
            prop.price_pkr  = parse_price(value)
        elif "area"  in label: 
            prop.area = value
        elif "type"  in label: 
            prop.property_type = value
        elif "bed"   in label: 
            prop.bedrooms = _first_int(value)
        elif "bath"  in label: 
            prop.bathrooms = _first_int(value)

    # If price wasn't found in details, look for it elsewhere on the page
    if not prop.price_text:
        for sel in ["span._105b8a67", "span._2923a568", "div._2923a568"]:
            tag = soup.select_one(sel)
            if tag:
                prop.price_text = _text(tag)
                prop.price_pkr  = parse_price(prop.price_text)
                break

    # Extract amenities (parking, servant quarters, rooms, etc.)
    for section in soup.find_all("div", class_="_83bb17d1"):
        h3 = section.find("h3")
        if not (h3 and "amenit" in h3.get_text().lower()):
            continue
        
        for li in section.find_all("li"):
            txt = _text(li)
            low = txt.lower()
            
            # Look for year (built in year)
            yr = re.search(r"\b(19|20)\d{2}\b", txt)
            if yr and not prop.built_in_year:
                prop.built_in_year = yr.group()
            
            # Check for each amenity
            if "park"    in low: prop.parking_spaces   = _extract_num_or_yes(txt)
            if "servant" in low: prop.servant_quarters = _extract_num_or_yes(txt)
            if "store"   in low: prop.store_rooms      = _extract_num_or_yes(txt)
            if "kitchen" in low: prop.kitchens         = _extract_num_or_yes(txt)
            if "drawing" in low: prop.drawing_room     = "Yes"
            if "dining"  in low: prop.dining_room      = "Yes"
            if "study"   in low: prop.study_room       = "Yes"
            if "prayer"  in low or "masjid" in low: prop.prayer_room = "Yes"
            if "powder"  in low: prop.powder_room      = "Yes"
            if any(w in low for w in ("lounge", "sitting", "living")):
                prop.lounge_or_sitting = "Yes"
        break   # Only process the first amenities section found

    # Extract property description
    desc = soup.select_one("div._3e9c24cd, div._2a806e1f, div._2d2b3f3a")
    if desc:
        prop.description = _text(desc)

    return prop


print("✅ Detail parser ready")

# =============================================================================
# CELL 8 — City name extractor
# =============================================================================

def city_from_url(url: str) -> Optional[str]:
    """Pull the city name from a Zameen search URL (e.g. 'Islamabad')."""
    try:
        seg  = re.sub(r"\.html?$", "", url.rstrip("/").split("/")[-1], flags=re.I)
        name = seg.split("-")[0].strip()
        return name.title() if name else None
    except Exception:
        return None


print(f"✅ City extracted: {city_from_url(SEARCH_URL)}")

# =============================================================================
# CELL 9 — RUN THE SCRAPER (main execution)
# =============================================================================

city = city_from_url(SEARCH_URL)
print(f"\n🏙️  City : {city}")
print("=" * 58)

# ══════════════════════════════════════════════════════════════
# STEP 1: Collect all property listing URLs from search pages
# ══════════════════════════════════════════════════════════════
print("\n📋 STEP 1: Collecting property listing URLs …")
print("(This step visits the search result pages)")
listing_urls = collect_listing_urls(SEARCH_URL, MAX_PAGES)
if MAX_LISTINGS:
    listing_urls = listing_urls[:MAX_LISTINGS]
print(f"\n✅ {len(listing_urls)} listings queued for scraping\n")

# ══════════════════════════════════════════════════════════════
# STEP 2: Visit each property page and extract information
# ══════════════════════════════════════════════════════════════
print("🏠 STEP 2: Scraping individual property pages …")
print("(This step downloads each property's detail page)")
print("=" * 58)

results = []
for i, url in enumerate(listing_urls, 1):
    print(f"  [{i:>3}/{len(listing_urls)}] {url[:72]}")
    try:
        prop = parse_property_page(url, city=city)
        results.append(prop)
        price_str = (f"PKR {prop.price_pkr:,.0f}" if prop.price_pkr
                     else prop.price_text or "(no price)")
        print(f"           ✔ {prop.title or '(no title)'}  |  {price_str}")
    except Exception as err:
        print(f"           ✗ Error: {err}")
    polite_wait()

print(f"\n✅ Finished — {len(results)} properties scraped successfully")

# =============================================================================
# CELL 10 — Build DataFrame & preview results
# =============================================================================

# Convert list of Property objects into a pandas DataFrame (table)
df = pd.DataFrame([asdict(p) for p in results])

df.rename(columns={
    "url":               "URL",
    "title":             "Title",
    "price_text":        "Price (Text)",
    "price_pkr":         "Price (PKR)",
    "location":          "Location",
    "city":              "City",
    "property_type":     "Property Type",
    "bedrooms":          "Bedrooms",
    "bathrooms":         "Bathrooms",
    "area":              "Area",
    "built_in_year":     "Built Year",
    "parking_spaces":    "Parking",
    "servant_quarters":  "Servant Qtrs",
    "store_rooms":       "Store Rooms",
    "kitchens":          "Kitchens",
    "drawing_room":      "Drawing Room",
    "dining_room":       "Dining Room",
    "study_room":        "Study Room",
    "prayer_room":       "Prayer Room",
    "powder_room":       "Powder Room",
    "lounge_or_sitting": "Lounge/Sitting",
    "description":       "Description",
}, inplace=True)

print(f"📊 DataFrame: {df.shape[0]} rows × {df.shape[1]} columns\n")
preview_cols = ["Title", "Price (Text)", "Price (PKR)", "Location", "Bedrooms", "Area"]
available    = [c for c in preview_cols if c in df.columns]
print(df[available].head(10).to_string(index=False))

# =============================================================================
# CELL 11 — Save results to CSV file
# =============================================================================

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n💾 Saved → {OUTPUT_CSV}")

# =============================================================================
# CELL 12 — Display summary statistics about the scraped data
# =============================================================================

print("\n📈 Quick Stats")
print("=" * 45)
print(f"Total listings scraped : {len(df)}")

if "Price (PKR)" in df.columns and df["Price (PKR)"].notna().any():
    print(f"Average price (PKR)    : {df['Price (PKR)'].mean():>15,.0f}")
    print(f"Cheapest (PKR)         : {df['Price (PKR)'].min():>15,.0f}")
    print(f"Most expensive (PKR)   : {df['Price (PKR)'].max():>15,.0f}")

if "Bedrooms" in df.columns and df["Bedrooms"].notna().any():
    print(f"Average bedrooms       : {df['Bedrooms'].mean():.1f}")

if "Property Type" in df.columns and df["Property Type"].notna().any():
    print("\nProperty Types breakdown:")
    print(df["Property Type"].value_counts().to_string())
