
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "retail_sales.csv"
FORECAST_PATH = PROJECT_ROOT / "reports" / "forecast_predictions.csv"
INVENTORY_PATH = PROJECT_ROOT / "reports" / "inventory_recommendations.csv"


def test_sales_dataset_exists():
    assert DATA_PATH.exists(), "Sales dataset is missing"


def test_sales_dataset_quality():
    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0
    assert df.isna().sum().sum() == 0
    assert not df.duplicated().any()
    assert not df.duplicated(
        subset=["date", "store_id", "product_id"]
    ).any()


def test_sales_quantities_are_valid():
    df = pd.read_csv(DATA_PATH)

    non_negative_columns = [
        "base_demand",
        "demand",
        "opening_stock",
        "units_sold",
        "closing_stock",
        "lost_sales",
        "unit_price",
        "sales_revenue",
        "order_quantity",
    ]

    for column in non_negative_columns:
        assert (df[column] >= 0).all(), (
            f"Negative values found in {column}"
        )

    assert (df["units_sold"] <= df["demand"]).all()
    assert (df["units_sold"] <= df["opening_stock"]).all()
    assert (df["closing_stock"] >= 0).all()



def test_forecast_file_and_predictions():
    assert FORECAST_PATH.exists(), "Forecast file is missing"

    df = pd.read_csv(FORECAST_PATH)

    assert len(df) > 0
    assert {"demand", "predicted_demand", "baseline_lag_7"}.issubset(
        df.columns
    )
    assert df["demand"].notna().all()
    assert df["predicted_demand"].notna().all()
    assert df["baseline_lag_7"].notna().all()

    assert pd.api.types.is_numeric_dtype(df["demand"])
    assert pd.api.types.is_numeric_dtype(df["predicted_demand"])
    assert (df["predicted_demand"] >= 0).all()



def test_inventory_recommendations():
    assert INVENTORY_PATH.exists(), "Inventory recommendations are missing"

    df = pd.read_csv(INVENTORY_PATH)

    assert len(df) == 30
    assert df["recommended_order_quantity"].notna().all()
    assert (df["recommended_order_quantity"] >= 0).all()
    assert (df["safety_stock"] >= 0).all()
    assert (df["reorder_point"] >= 0).all()
    assert (df["target_stock"] >= 0).all()
    assert (df["estimated_order_cost"] >= 0).all()


def test_inventory_cost_calculation():
    df = pd.read_csv(INVENTORY_PATH)

    expected_cost = (
        df["recommended_order_quantity"] * df["unit_price"]
    )

    assert (
        (df["estimated_order_cost"] - expected_cost).abs() < 0.02
    ).all()
