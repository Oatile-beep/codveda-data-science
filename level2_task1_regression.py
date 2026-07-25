"""
Codveda Data Science Internship - Level 2, Task 1: Predictive Modeling (Regression)
Author: Thapelo Oatile Tlhomelang

Goal: Build and evaluate a regression model to predict a continuous variable
(crop yield, in tons per hectare).

Note on the dataset: rather than relying on an external download (which can
fail depending on your network), this script generates a realistic synthetic
agricultural dataset locally, with a known underlying relationship between
features (rainfall, fertilizer use, soil quality, temperature) and yield.
This keeps the task fully reproducible and offline while still representing
genuine regression mechanics.

Steps performed:
1. Split the dataset into training and testing sets.
2. Train a Linear Regression model using scikit-learn.
3. Evaluate the model using MSE and R-squared.
4. Compare against Decision Tree and Random Forest regressors.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42


def generate_crop_yield_data(n_samples=500, random_state=RANDOM_STATE):
    """Generate a synthetic but realistic crop yield dataset with a known
    underlying relationship plus noise, so results are meaningful to evaluate.

    Features:
      rainfall_mm      - seasonal rainfall in millimetres
      fertilizer_kg_ha - fertilizer applied, kg per hectare
      soil_quality     - index from 0 (poor) to 10 (excellent)
      avg_temp_c       - average growing-season temperature, degrees C
    Target:
      yield_tons_ha     - crop yield, tons per hectare
    """
    rng = np.random.default_rng(random_state)

    rainfall_mm = rng.normal(700, 200, n_samples).clip(100, 1500)
    fertilizer_kg_ha = rng.normal(150, 50, n_samples).clip(0, 350)
    soil_quality = rng.uniform(0, 10, n_samples)
    avg_temp_c = rng.normal(24, 4, n_samples).clip(10, 40)

    # A realistic-ish underlying yield formula:
    # - more rainfall helps (simple positive linear term here)
    # - fertilizer helps
    # - better soil quality helps a lot
    # - yield drops off the further temperature strays from an optimum (~24C)
    temp_penalty = -0.05 * (avg_temp_c - 24) ** 2

    yield_tons_ha = (
        0.004 * rainfall_mm
        + 0.015 * fertilizer_kg_ha
        + 0.6 * soil_quality
        + temp_penalty
        + 1.0
        + rng.normal(0, 0.5, n_samples)
    ).clip(0.1, None)

    df = pd.DataFrame({
        "rainfall_mm": rainfall_mm.round(1),
        "fertilizer_kg_ha": fertilizer_kg_ha.round(1),
        "soil_quality": soil_quality.round(2),
        "avg_temp_c": avg_temp_c.round(1),
        "yield_tons_ha": yield_tons_ha.round(3),
    })
    return df


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f} tons/ha")
    print(f"  R^2:  {r2:.4f}")

    return {"model": name, "mse": mse, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    df = generate_crop_yield_data()
    df.to_csv("crop_yield_data.csv", index=False)
    print(f"Generated synthetic crop yield dataset: {len(df)} rows")
    print(df.head())

    X = df[["rainfall_mm", "fertilizer_kg_ha", "soil_quality", "avg_temp_c"]]
    y = df["yield_tons_ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    results = []
    results.append(train_and_evaluate(
        LinearRegression(), X_train, X_test, y_train, y_test, "Linear Regression"
    ))
    results.append(train_and_evaluate(
        DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=6),
        X_train, X_test, y_train, y_test, "Decision Tree Regressor"
    ))
    results.append(train_and_evaluate(
        RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, max_depth=8),
        X_train, X_test, y_train, y_test, "Random Forest Regressor"
    ))

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    results_df.to_csv("regression_model_comparison.csv", index=False)

    print("\n" + "=" * 50)
    print("Model comparison (best R^2 first):")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {results_df.iloc[0]['model']}")
    print(
        "\nNote: yield has a non-linear relationship with temperature "
        "(a penalty that grows the further temperature strays from the "
        "24C optimum), so tree-based models are expected to capture that "
        "curve better than plain Linear Regression."
    )
