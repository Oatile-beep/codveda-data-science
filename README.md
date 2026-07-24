# Codveda Technologies — Data Science Internship

Level 1 (Basic) task submissions for the Codveda Technologies Data Science Internship.

**Intern:** Thapelo Oatile Tlhomelang
**Duration:** 22/07/2026 – 22/08/2026

## Tasks Completed

### Task 1: Data Collection and Web Scraping
Scrapes book data (title, price, rating, availability, stock count, category) from
[books.toscrape.com](https://books.toscrape.com/), a public sandbox site for practicing
web scraping. Handles pagination automatically and saves the result to CSV.

- Script: `task1_scraping.py`
- Output: `scraped_books_raw.csv`
- Tools: Python, requests, BeautifulSoup, pandas

### Task 2: Data Cleaning and Preprocessing
Cleans the raw scraped dataset — handles missing values, removes duplicates and
outliers (IQR method), encodes categorical variables (label + one-hot encoding),
and standardizes numeric features.

- Script: `task2_cleaning.py`
- Output: `scraped_books_cleaned.csv`
- Tools: Python, pandas, scikit-learn

### Task 3: Exploratory Data Analysis (EDA)
Computes summary statistics and generates visualizations (histogram, scatter plot,
box plot, correlation matrix) to explore relationships in the cleaned dataset, plus
a short written insights report.

- Script: `task3_eda.py`
- Output: 4 chart PNGs + `eda_report.md`
- Tools: Python, pandas, matplotlib, seaborn

## How to Run

```bash
pip install requests beautifulsoup4 pandas lxml scikit-learn matplotlib seaborn tabulate

python task1_scraping.py   # produces scraped_books_raw.csv
python task2_cleaning.py   # produces scraped_books_cleaned.csv
python task3_eda.py        # produces charts + eda_report.md
```
