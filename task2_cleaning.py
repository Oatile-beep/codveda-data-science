"""
Codveda Data Science Internship - Level 1, Task 2: Data Cleaning and Preprocessing
Author: Thapelo Oatile Tlhomelang

Input:  scraped_books_raw.csv   (produced by task1_scraping.py)
Output: scraped_books_cleaned.csv

Steps performed:
1. Handle missing data (title, price, rating).
2. Detect and remove outliers (price).
3. Convert categorical variables into numerical format (one-hot encoding for
   category, label encoding for availability tier).
4. Normalize/standardize numerical data (price).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

INPUT_FILE = "scraped_books_raw.csv"
OUTPUT_FILE = "scraped_books_cleaned.csv"


def load_data(path):
    return pd.read_csv(path)


def handle_missing_data(df):
    before = len(df)

    # Titles are essential identifiers - drop rows with no title at all.
    df = df.dropna(subset=["title"]).copy()

    # Price: impute missing values with the median price (robust to outliers).
    median_price = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # Rating: impute missing values with the mode (most common rating).
    mode_rating = df["rating_stars"].mode()[0]
    df["rating_stars"] = df["rating_stars"].fillna(mode_rating)

    print(f"Missing data handled. Rows before: {before}, after: {len(df)}")
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["title", "book_url"]).copy()
    print(f"Duplicates removed: {before - len(df)}")
    return df


def remove_outliers_iqr(df, column):
    """Remove outliers using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)].copy()
    print(f"Outliers removed from '{column}': {before - len(df)} rows "
          f"(valid range: {lower:.2f} to {upper:.2f})")
    return df


def extract_stock_count(availability_text):
    """Pull the numeric stock count out of strings like 'In stock (19 available)'."""
    import re
    match = re.search(r"\((\d+)\s+available\)", str(availability_text))
    return int(match.group(1)) if match else 0


def encode_categoricals(df):
    # Label-encode availability into a simple stock-tier count.
    df["stock_count"] = df["availability"].apply(extract_stock_count)

    le = LabelEncoder()
    df["availability_encoded"] = le.fit_transform(df["availability"])

    # One-hot encode category.
    df = pd.get_dummies(df, columns=["category"], prefix="cat")

    print("Categorical variables encoded (availability: label encoding, "
          "category: one-hot encoding).")
    return df


def normalize_numeric(df):
    scaler = StandardScaler()
    df["price_gbp_scaled"] = scaler.fit_transform(df[["price_gbp"]])
    print("Numeric column 'price_gbp' standardized into 'price_gbp_scaled'.")
    return df


if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}\n")

    df = handle_missing_data(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, "price_gbp")
    df = encode_categoricals(df)
    df = normalize_numeric(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Cleaned dataset saved to {OUTPUT_FILE} ({len(df)} rows, "
          f"{df.shape[1]} columns).")
    print(df.head())
