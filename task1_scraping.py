"""
Codveda Data Science Internship - Level 1, Task 1: Data Collection and Web Scraping
Author: Thapelo Oatile Tlhomelang

Target site: https://books.toscrape.com/
This site is purpose-built as a public sandbox for practicing web scraping,
so it's a safe and appropriate target for this task.

What this script does:
1. Identifies the target site structure (paginated book listing pages).
2. Uses requests + BeautifulSoup to scrape each page.
3. Handles pagination automatically (follows the "next" button until the last page).
4. Extracts: title, price, star rating, in-stock status, and category.
5. Stores the result as a structured CSV file.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://books.toscrape.com/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url):
    """Fetch a page and return a BeautifulSoup object."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def parse_book_card(card):
    """Extract structured data from a single book card on the listing page."""
    title = card.h3.a["title"]

    price_text = card.select_one(".price_color").get_text(strip=True)
    price = float(price_text.replace("£", "").replace("Â", ""))

    rating_class = card.select_one("p.star-rating")["class"]
    rating_word = [c for c in rating_class if c != "star-rating"][0]
    rating = RATING_WORDS.get(rating_word, None)

    availability = card.select_one(".availability").get_text(strip=True)

    relative_link = card.h3.a["href"]
    book_url = BASE_URL + "catalogue/" + relative_link.replace("../../../", "")

    return {
        "title": title,
        "price_gbp": price,
        "rating_stars": rating,
        "availability": availability,
        "book_url": book_url,
    }


def get_book_details(book_url):
    """Visit a book's detail page to grab its category and its FULL availability
    text (the listing page only shows 'In stock', with no quantity - the actual
    stock count only appears on the detail page). Handles a bit of extra
    navigation depth beyond the flat listing pages."""
    try:
        soup = get_soup(book_url)

        breadcrumb = soup.select("ul.breadcrumb li a")
        # breadcrumb: [Home, Category, Book Title-ish] -> category is index 1
        category = breadcrumb[1].get_text(strip=True) if len(breadcrumb) > 1 else None

        availability_tag = soup.select_one("p.instock.availability")
        availability_detailed = (
            availability_tag.get_text(strip=True) if availability_tag else None
        )

        return category, availability_detailed
    except Exception:
        return None, None


def scrape_all_books(max_pages=None, fetch_details=True):
    """Scrape all book listing pages, following pagination until there is no
    'next' link. Returns a list of dicts."""
    all_books = []
    url = START_URL
    page_num = 1

    while url:
        print(f"Scraping page {page_num}: {url}")
        soup = get_soup(url)

        cards = soup.select("article.product_pod")
        for card in cards:
            book = parse_book_card(card)
            if fetch_details:
                category, availability_detailed = get_book_details(book["book_url"])
                book["category"] = category
                # Overwrite the listing page's bare "In stock" with the detail
                # page's full text (e.g. "In stock (19 available)") so the
                # stock count can actually be extracted downstream.
                if availability_detailed:
                    book["availability"] = availability_detailed
                time.sleep(0.1)  # be polite to the server
            all_books.append(book)

        next_link = soup.select_one("li.next a")
        if next_link and (max_pages is None or page_num < max_pages):
            url = BASE_URL + "catalogue/" + next_link["href"]
            page_num += 1
        else:
            url = None

    return all_books


if __name__ == "__main__":
    # Limiting to 5 pages (~100 books) keeps the demo fast and polite to the
    # server. Set max_pages=None to scrape the entire catalogue (~1000 books).
    books = scrape_all_books(max_pages=5, fetch_details=True)

    df = pd.DataFrame(books)
    df.to_csv("scraped_books_raw.csv", index=False)

    print(f"\nDone. Scraped {len(df)} books.")
    print(df.head())
