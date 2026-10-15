
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "retail_sales.csv"
PREDICTIONS_PATH = ROOT / "reports" / "forecast_predictions.csv"
MODEL_PATH = ROOT / "models" / "demand_forecaster.joblib"
OUTPUT_PATH = ROOT / "reports" / "inventory_recommendations.csv"

LEAD_TIME_DAYS = 5
REVIEW_PERIOD_DAYS = 1
SERVICE_LEVEL_Z = 1.65


def main() -> None:
    # Check that the previous project steps have been completed.
    for path in [DATA_PATH, PREDICTIONS_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the data generator and forecasting scripts first."
            )

    sales = pd.read_csv(DATA_PATH, parse_dates=["date"])
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"],
    )

    # Verify the saved model can be loaded.
    saved_model = joblib.load(MODEL_PATH)
    print("Loaded forecasting model successfully.")

    required_prediction_columns = {
        "date",
        "store_id",
        "product_id",
        "predicted_demand",
    }

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        raise ValueError(
            "Forecast predictions are missing required columns."
        )

    # The latest seven saved forecasts estimate near-term daily demand.
    predictions = predictions.sort_values("date")

    latest_forecasts = (
        predictions.groupby(["store_id", "product_id"])
        .tail(7)
        .groupby(["store_id", "product_id"])
        .agg(
            forecast_daily_demand=("predicted_demand", "mean"),
            forecast_days=("predicted_demand", "size"),
        )
        .reset_index()
    )

    # Obtain the most recent observed inventory for each store-product.
    sales = sales.sort_values("date")

    latest_inventory = (
        sales.groupby(["store_id", "product_id"])
        .tail(1)[
            [
                "store_id",
                "product_id",
                "date",
                "closing_stock",
                "unit_price",
            ]
        ]
        .rename(
            columns={
                "date": "inventory_date",
                "closing_stock": "current_stock",
            }
        )
    )

    # Estimate daily demand variability using recent observations.
    recent_demand = (
        sales.groupby(["store_id", "product_id"])
        .tail(28)
        .groupby(["store_id", "product_id"])
        .agg(
            historical_daily_demand_std=("demand", "std"),
            historical_daily_demand_mean=("demand", "mean"),
        )
        .reset_index()
    )

    recommendations = latest_inventory.merge(
        latest_forecasts,
        on=["store_id", "product_id"],
        how="inner",
    )

    recommendations = recommendations.merge(
        recent_demand,
        on=["store_id", "product_id"],
        how="left",
    )

    if recommendations.empty:
        raise ValueError(
            "No matching forecasts and inventory records found."
        )

    # A missing standard deviation can occur with very little history.
    recommendations["historical_daily_demand_std"] = (
        recommendations["historical_daily_demand_std"].fillna(0)
    )

    # Use nonnegative forecasts.
    recommendations["forecast_daily_demand"] = (
        recommendations["forecast_daily_demand"].clip(lower=0)
    )

    # Approximate safety stock for demand variability during lead time.
    recommendations["safety_stock"] = (
        SERVICE_LEVEL_Z
        * recommendations["historical_daily_demand_std"]
        * np.sqrt(LEAD_TIME_DAYS)
    ).apply(np.ceil).astype(int)

    # Reorder point: forecast demand during lead time + safety stock.
    recommendations["reorder_point"] = (
        recommendations["forecast_daily_demand"]
        * LEAD_TIME_DAYS
        + recommendations["safety_stock"]
    ).apply(np.ceil).astype(int)

    # Target stock covers the lead time plus one review period.
    recommendations["target_stock"] = (
        recommendations["forecast_daily_demand"]
        * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)
        + recommendations["safety_stock"]
    ).apply(np.ceil).astype(int)

    # This first version uses on-hand stock only.
    # Outstanding purchase orders are not included in the source data.
    recommendations["recommended_order_quantity"] = np.where(
        recommendations["current_stock"]
        <= recommendations["reorder_point"],
        np.maximum(
            0,
            recommendations["target_stock"]
            - recommendations["current_stock"],
        ),
        0,
    ).astype(int)

    recommendations["reorder_status"] = np.where(
        recommendations["recommended_order_quantity"] > 0,
        "REORDER",
        "STOCK SUFFICIENT",
    )

    recommendations["estimated_order_cost"] = (
        recommendations["recommended_order_quantity"]
        * recommendations["unit_price"]
    ).round(2)

    # Save a useful, readable output.
    output_columns = [
        "store_id",
        "product_id",
        "inventory_date",
        "forecast_days",
        "forecast_daily_demand",
        "historical_daily_demand_std",
        "current_stock",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order_quantity",
        "reorder_status",
        "unit_price",
        "estimated_order_cost",
    ]

    recommendations = recommendations[output_columns].sort_values(
        ["reorder_status", "store_id", "product_id"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("INVENTORY OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"Store-product combinations: {len(recommendations)}")
    print(
        "Products requiring reorder:",
        int((recommendations["reorder_status"] == "REORDER").sum()),
    )
    print(
        "Total recommended units:",
        int(recommendations["recommended_order_quantity"].sum()),
    )
    print(
        "Estimated order cost:",
        f"{recommendations['estimated_order_cost'].sum():,.2f}",
    )

    print("\nSample recommendations:")
    print(recommendations.head(10).to_string(index=False))

    print(f"\nSaved recommendations to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
