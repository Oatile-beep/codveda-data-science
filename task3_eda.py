"""
Codveda Data Science Internship - Level 1, Task 3: Exploratory Data Analysis (EDA)
Author: Thapelo Oatile Tlhomelang

Input:  scraped_books_cleaned.csv   (produced by task2_cleaning.py)
Output: 4 PNG charts + a short written insights report (eda_report.md)

Steps performed:
1. Compute summary statistics (mean, median, variance, etc.).
2. Visualize the data using a histogram, a scatter plot, and a box plot.
3. Identify correlations between numerical features using a correlation matrix.
4. Generate a short report summarizing insights from the EDA.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "scraped_books_cleaned.csv"

sns.set_style("whitegrid")


def load_data(path):
    return pd.read_csv(path)


def summary_statistics(df, numeric_cols):
    stats = df[numeric_cols].describe().T
    stats["variance"] = df[numeric_cols].var()
    print("Summary statistics:\n")
    print(stats)
    return stats


def plot_price_histogram(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price_gbp"], bins=20, kde=True, color="#4C72B0")
    plt.title("Distribution of Book Prices")
    plt.xlabel("Price (GBP)")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig("eda_price_histogram.png", dpi=150)
    plt.close()
    print("Saved: eda_price_histogram.png")


def plot_price_vs_stock_scatter(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="stock_count", y="price_gbp",
        hue="rating_stars", palette="viridis", s=60
    )
    plt.title("Price vs. Stock Count (coloured by Rating)")
    plt.xlabel("Stock Count (available units)")
    plt.ylabel("Price (GBP)")
    plt.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("eda_price_vs_stock_scatter.png", dpi=150)
    plt.close()
    print("Saved: eda_price_vs_stock_scatter.png")


def plot_price_by_rating_boxplot(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df, x="rating_stars", y="price_gbp",
        hue="rating_stars", palette="Set2", legend=False
    )
    plt.title("Price Distribution by Star Rating")
    plt.xlabel("Rating (stars)")
    plt.ylabel("Price (GBP)")
    plt.tight_layout()
    plt.savefig("eda_price_by_rating_boxplot.png", dpi=150)
    plt.close()
    print("Saved: eda_price_by_rating_boxplot.png")


def plot_correlation_matrix(df, numeric_cols):
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("eda_correlation_matrix.png", dpi=150)
    plt.close()
    print("Saved: eda_correlation_matrix.png")
    return corr


def generate_report(stats, corr, df):
    lines = []
    lines.append("# Exploratory Data Analysis - Books Dataset\n")
    lines.append(f"Dataset size: {len(df)} books, {df.shape[1]} columns.\n")

    lines.append("## Summary Statistics\n")
    lines.append(stats.round(2).to_markdown())
    lines.append("\n")

    lines.append("## Key Observations\n")

    price_mean = df["price_gbp"].mean()
    price_median = df["price_gbp"].median()
    price_std = df["price_gbp"].std()
    lines.append(
        f"- Average book price is £{price_mean:.2f} (median £{price_median:.2f}), "
        f"with a standard deviation of £{price_std:.2f}, showing a moderate spread "
        f"of prices across the catalogue.\n"
    )

    rating_counts = df["rating_stars"].value_counts().sort_index()
    most_common_rating = rating_counts.idxmax()
    lines.append(
        f"- The most common star rating is {int(most_common_rating)} stars "
        f"({rating_counts.max()} books).\n"
    )

    # Strongest correlation (excluding self-correlation of 1.0)
    corr_unstacked = corr.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0]
    if not corr_unstacked.empty:
        top_pair = corr_unstacked.idxmax()
        top_value = corr[top_pair[0]][top_pair[1]]
        lines.append(
            f"- The strongest correlation observed is between **{top_pair[0]}** "
            f"and **{top_pair[1]}** (r = {top_value:.2f}). "
            f"{'This suggests a meaningful relationship worth investigating further.' if abs(top_value) > 0.3 else 'This is a weak relationship, suggesting these variables are largely independent.'}\n"
        )

    lines.append(
        "- Overall, price does not appear to be strongly driven by rating or "
        "stock count in this dataset - which makes sense, since this is a demo "
        "bookstore with prices assigned independently of these factors.\n"
    )

    report_text = "\n".join(lines)
    with open("eda_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\nSaved: eda_report.md")


if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}\n")

    numeric_cols = ["price_gbp", "rating_stars", "stock_count"]

    stats = summary_statistics(df, numeric_cols)
    plot_price_histogram(df)
    plot_price_vs_stock_scatter(df)
    plot_price_by_rating_boxplot(df)
    corr = plot_correlation_matrix(df, numeric_cols)
    generate_report(stats, corr, df)

    print("\nEDA complete. 4 charts + eda_report.md generated.")
