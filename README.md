# Data Science Portfolio Projects

A collection of self-directed data science projects covering the full pipeline — from data collection to modeling — across three progressive skill levels (Basic, Intermediate, Advanced).

**Author:** Thapelo Oatile Tlhomelang  
**Program:** BSc Computer Science & Statistics, University of Zululand



---

## Level 1 (Basic)

### Task 1: Data Collection and Web Scraping
Scrapes book data (title, price, rating, availability, stock count, category) from [books.toscrape.com](https://books.toscrape.com/), a public sandbox site for practicing web scraping. Handles pagination automatically and saves the result to CSV.

- Script: `task1_scraping.py`
- Output: `scraped_books_raw.csv`
- Tools: Python, requests, BeautifulSoup, pandas

### Task 2: Data Cleaning and Preprocessing
Cleans the raw scraped dataset — handles missing values, removes duplicates and outliers (IQR method), encodes categorical variables (label + one-hot encoding), and standardizes numeric features.

- Script: `task2_cleaning.py`
- Output: `scraped_books_cleaned.csv`
- Tools: Python, pandas, scikit-learn

### Task 3: Exploratory Data Analysis (EDA)
Computes summary statistics and generates visualizations (histogram, scatter plot, box plot, correlation matrix) to explore relationships in the cleaned dataset, plus a short written insights report.

- Script: `task3_eda.py`
- Output: 4 chart PNGs (`eda_*.png`) + `eda_report.md`
- Tools: Python, pandas, matplotlib, seaborn

---

## Level 2 (Intermediate)

### Task 1: Predictive Modeling (Regression)
Predicts crop yield (tons/hectare) from rainfall, fertilizer use, soil quality, and temperature, using a synthetic-but-realistic dataset with a known non-linear relationship (temperature has an optimum, penalizing yield the further it strays). Compares Linear Regression, Decision Tree, and Random Forest.

- Script: `level2_task1_regression.py`
- Output: `crop_yield_data.csv`, `regression_model_comparison.csv`
- Result: Random Forest performed best (R² = 0.90), correctly capturing the non-linear temperature effect that Linear Regression (R² = 0.79) could not.
- Tools: Python, scikit-learn, pandas

### Task 2: Classification with Logistic Regression
Classifies iris flowers into species using the classic Iris dataset (built into scikit-learn). Preprocesses with feature scaling, trains Logistic Regression, and evaluates with accuracy, precision, recall, F1, and a confusion matrix. Compares against Random Forest and SVM.

- Script: `level2_task2_classification.py`
- Output: `classification_model_comparison.csv`
- Result: Logistic Regression reached 93.3% accuracy; SVM performed best overall (96.7%).
- Tools: Python, scikit-learn, pandas

---

## Level 3 (Advanced)

### Task 1: Time Series Analysis
Analyzes and forecasts a synthetic daily sales series (trend + weekly + yearly seasonality + noise) over ~2 years. Decomposes the series, applies a 7-day moving average and Holt-Winters exponential smoothing, and fits a SARIMA model to forecast the final 30 days.

- Script: `level3_task1_timeseries.py`
- Output: `sales_data.csv`, `ts_decomposition.png`, `ts_smoothing.png`, `ts_sarima_forecast.png`
- Result: SARIMA forecast RMSE of 28.71 against held-out actuals.
- Tools: Python, pandas, statsmodels, matplotlib

### Task 2: NLP — Text Classification
Classifies messages as spam or legitimate ("ham") using a synthetic labelled dataset that includes deliberately ambiguous, borderline examples. Preprocesses text (regex tokenization, stopword removal, Porter stemming — fully offline, no external corpus downloads), vectorizes with TF-IDF, and trains Naive Bayes and Logistic Regression classifiers.

- Script: `level3_task2_nlp.py`
- Output: `spam_ham_data.csv`, `nlp_model_comparison.csv`
- Result: Both models reached 91.7% accuracy, with perfect precision but 83% recall on spam — cautious rather than over-eager spam flagging.
- Tools: Python, scikit-learn, nltk, pandas

---

## How to Run

```bash
pip install requests beautifulsoup4 pandas lxml scikit-learn matplotlib seaborn tabulate statsmodels nltk

# Level 1
python task1_scraping.py
python task2_cleaning.py
python task3_eda.py

# Level 2
python level2_task1_regression.py
python level2_task2_classification.py

# Level 3
python level3_task1_timeseries.py
python level3_task2_nlp.py
```

All Level 2 and Level 3 datasets are generated locally by their scripts (no external downloads required), so they run fully offline once the libraries above are installed.
