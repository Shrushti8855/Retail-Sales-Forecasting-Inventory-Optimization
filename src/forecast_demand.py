
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "retail_sales.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features using only information available before each day."""

    df = df.sort_values(
        ["store_id", "product_id", "date"]
    ).copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    groups = df.groupby(
        ["store_id", "product_id"]
    )["demand"]

    # Past demand only: never use the current day's target.
    df["lag_1"] = groups.shift(1)
    df["lag_7"] = groups.shift(7)
    df["lag_14"] = groups.shift(14)

    # Shift first so the current day's demand is excluded.
    df["rolling_mean_7"] = groups.transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    df["rolling_mean_28"] = groups.transform(
        lambda s: s.shift(1).rolling(28).mean()
    )

    return df


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    df = create_features(df)

    # Rows without sufficient history cannot be used.
    df = df.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_mean_7",
            "rolling_mean_28",
        ]
    ).copy()

    features = [
        "promotion",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_28",
    ]

    # Chronological split: never randomly mix past and future dates.
    unique_dates = np.sort(df["date"].unique())
    split_index = int(len(unique_dates) * 0.80)
    split_date = pd.Timestamp(unique_dates[split_index])

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    X_train = train[features]
    y_train = train["demand"]
    X_test = test[features]
    y_test = test["demand"]

    print("=" * 55)
    print("RETAIL DEMAND FORECASTING")
    print("=" * 55)
    print(f"Training records: {len(train):,}")
    print(f"Testing records: {len(test):,}")
    print(f"Test period starts: {split_date.date()}")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = np.maximum(
        0,
        model.predict(X_test),
    )

    # Compare against a simple baseline: demand from seven days ago.
    baseline_predictions = test["lag_7"].to_numpy()

    model_mae = mean_absolute_error(
        y_test,
        predictions,
    )
    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    model_r2 = r2_score(y_test, predictions)

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    print("\nTEST RESULTS")
    print(f"Random Forest MAE: {model_mae:.2f}")
    print(f"Random Forest RMSE: {model_rmse:.2f}")
    print(f"Random Forest R²: {model_r2:.4f}")
    print(f"7-day-lag baseline MAE: {baseline_mae:.2f}")

    print("\nMean Absolute Error (MAE):")
    print("On average, how many units the prediction differs by.")

    print("\nRoot Mean Squared Error (RMSE):")
    print("Penalizes large prediction errors more heavily.")

    print("\nR²:")
    print("Measures how well predictions explain demand variation.")

    # Save the trained model and the exact feature order.
    model_path = MODEL_DIR / "demand_forecaster.joblib"

    joblib.dump(
        {
            "model": model,
            "features": features,
        },
        model_path,
    )

    # Save predictions for inspection.
    results = test[
        ["date", "store_id", "product_id", "demand"]
    ].copy()

    results["predicted_demand"] = predictions
    results["baseline_lag_7"] = baseline_predictions

    results.to_csv(
        REPORTS_DIR / "forecast_predictions.csv",
        index=False,
    )

    # Compare total actual and predicted demand per day.
    daily = (
        results.groupby("date")
        .agg(
            actual_demand=("demand", "sum"),
            predicted_demand=("predicted_demand", "sum"),
        )
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        daily.index,
        daily["actual_demand"],
        label="Actual demand",
    )
    ax.plot(
        daily.index,
        daily["predicted_demand"],
        label="Predicted demand",
    )

    ax.set_title("Actual vs Predicted Daily Retail Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total units across stores and products")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "demand_forecast_test.png",
        dpi=150,
    )
    plt.close(fig)

    print(f"\nModel saved to: {model_path}")
    print(f"Predictions saved to: {REPORTS_DIR / 'forecast_predictions.csv'}")
    print(f"Chart saved to: {REPORTS_DIR / 'demand_forecast_test.png'}")


if __name__ == "__main__":
    main()
