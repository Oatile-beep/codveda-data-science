# Exploratory Data Analysis - Books Dataset

Dataset size: 100 books, 9 columns.

## Summary Statistics

|              |   count |   mean |   std |   min |   25% |   50% |   75% |   max |   variance |
|:-------------|--------:|-------:|------:|------:|------:|------:|------:|------:|-----------:|
| price_gbp    |     100 |  34.56 | 14.64 | 10.16 |  19.9 | 34.78 | 47.97 | 58.11 |     214.29 |
| rating_stars |     100 |   2.93 |  1.42 |  1    |   2   |  3    |  4    |  5    |       2.03 |
| stock_count  |     100 |  16.86 |  2.83 |  0    |  16   | 16    | 19    | 22    |       8.02 |


## Key Observations

- Average book price is £34.56 (median £34.78), with a standard deviation of £14.64, showing a moderate spread of prices across the catalogue.

- The most common star rating is 1 stars (22 books).

- The strongest correlation observed is between **price_gbp** and **rating_stars** (r = -0.12). This is a weak relationship, suggesting these variables are largely independent.

- Overall, price does not appear to be strongly driven by rating or stock count in this dataset - which makes sense, since this is a demo bookstore with prices assigned independently of these factors.
