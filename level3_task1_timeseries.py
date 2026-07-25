"""
Codveda Data Science Internship - Level 3, Task 1: Time Series Analysis
Author: Thapelo Oatile Tlhomelang

Goal: Analyze and model time-series data to forecast future values (sales).

Note on the dataset: a synthetic but realistic daily sales series is
generated locally (trend + weekly seasonality + yearly seasonality + noise),
so this runs fully offline with a known ground truth to sanity-check results
against.

Steps performed:
1. Plot and decompose the series into trend, seasonality, and residual.
2. Implement a moving average and exponential smoothing.
3. Build a SARIMA model for forecasting.
4. Evaluate using RMSE and visualize the forecast.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided")

RANDOM_STATE = 42


def generate_sales_data(n_days=730, random_state=RANDOM_STATE):
    """Generate ~2 years of synthetic daily sales data with an upward trend,
    weekly seasonality (weekend spikes), yearly seasonality, and noise."""
    rng = np.random.default_rng(random_state)

    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 200 + 0.15 * t
    weekly_seasonality = 30 * np.sin(2 * np.pi * t / 7 + 1.5)
    yearly_seasonality = 50 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 15, n_days)

    sales = (trend + weekly_seasonality + yearly_seasonality + noise).clip(0, None)

    df = pd.DataFrame({"date": dates, "sales": sales.round(2)})
    df.set_index("date", inplace=True)
    return df


def plot_decomposition(df):
    decomposition = seasonal_decompose(df["sales"], model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color="#4C72B0")
    axes[0].set_ylabel("Observed")
    decomposition.trend.plot(ax=axes[1], color="#DD8452")
    axes[1].set_ylabel("Trend")
    decomposition.seasonal.plot(ax=axes[2], color="#55A868")
    axes[2].set_ylabel("Seasonal")
    decomposition.resid.plot(ax=axes[3], color="#C44E52")
    axes[3].set_ylabel("Residual")

    plt.suptitle("Time Series Decomposition of Daily Sales")
    plt.tight_layout()
    plt.savefig("ts_decomposition.png", dpi=150)
    plt.close()
    print("Saved: ts_decomposition.png")


def plot_smoothing(df):
    df_plot = df.copy()
    df_plot["moving_avg_7d"] = df["sales"].rolling(window=7).mean()

    exp_smooth_model = ExponentialSmoothing(
        df["sales"], trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    df_plot["exp_smoothing"] = exp_smooth_model.fittedvalues

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot.index, df_plot["sales"], label="Actual", alpha=0.4)
    plt.plot(df_plot.index, df_plot["moving_avg_7d"], label="7-day Moving Average")
    plt.plot(df_plot.index, df_plot["exp_smoothing"], label="Exponential Smoothing")
    plt.title("Sales: Actual vs. Smoothed")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ts_smoothing.png", dpi=150)
    plt.close()
    print("Saved: ts_smoothing.png")


def sarima_forecast(df, forecast_days=30):
    train = df.iloc[:-forecast_days]
    test = df.iloc[-forecast_days:]

    # SARIMA(p,d,q)(P,D,Q,s): a modest order that captures weekly seasonality
    # (s=7) without being too slow to fit for a demo script.
    model = SARIMAX(
        train["sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    forecast = fitted.forecast(steps=forecast_days)
    rmse = np.sqrt(mean_squared_error(test["sales"], forecast))
    print(f"\nSARIMA forecast RMSE (last {forecast_days} days): {rmse:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(train.index[-90:], train["sales"].iloc[-90:], label="Training data")
    plt.plot(test.index, test["sales"], label="Actual (held out)", color="green")
    plt.plot(test.index, forecast, label="SARIMA forecast", color="red", linestyle="--")
    plt.title(f"SARIMA Forecast vs. Actual (RMSE = {rmse:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ts_sarima_forecast.png", dpi=150)
    plt.close()
    print("Saved: ts_sarima_forecast.png")

    return rmse


if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv("sales_data.csv")
    print(f"Generated synthetic daily sales dataset: {len(df)} days")
    print(df.head())

    plot_decomposition(df)
    plot_smoothing(df)
    rmse = sarima_forecast(df)

    print("\nTime series analysis complete.")
