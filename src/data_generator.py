
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def generate_retail_data(
    days: int = 365,
    num_stores: int = 3,
    num_products: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate simulated retail demand and continuous inventory data."""

    if days < 1 or num_stores < 1 or num_products < 1:
        raise ValueError("Days, stores, and products must be positive.")

    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        end="2026-09-30",
        periods=days,
        freq="D",
    )

    stores = [
        f"STORE_{i:03d}"
        for i in range(1, num_stores + 1)
    ]

    products = [
        f"SKU_{i:03d}"
        for i in range(1, num_products + 1)
    ]

    # Typical daily customer demand differs by product.
    product_base_demand = {
        product: float(rng.integers(10, 60))
        for product in products
    }

    # Store demand levels differ.
    store_factor = {
        store: float(rng.uniform(0.75, 1.30))
        for store in stores
    }

    # A product has the same price across stores and dates.
    product_prices = {
        product: round(float(rng.uniform(20, 500)), 2)
        for product in products
    }

    # Initial stock is maintained separately for each store-product pair.
    initial_stock = {
        (store, product): int(rng.integers(80, 151))
        for store in stores
        for product in products
    }

    lead_time = 5
    review_period = 1
    target_days_of_stock = 14

    rows = []

    for store in stores:
        for product in products:
            base_demand = product_base_demand[product]
            multiplier = store_factor[store]
            unit_price = product_prices[product]

            inventory = initial_stock[(store, product)]

            # Maps an arrival date to the quantity arriving that day.
            pending_orders = {}

            for date in dates:
                # Receive replenishment orders due today.
                inventory += pending_orders.pop(date, 0)

                opening_stock = inventory

                # Demand changes with weekends, seasonality and promotions.
                weekend_factor = (
                    1.20 if date.dayofweek >= 5 else 1.0
                )

                seasonal_factor = (
                    1.0
                    + 0.15
                    * np.sin(2 * np.pi * date.dayofyear / 365)
                )

                promotion = int(rng.random() < 0.10)
                promotion_factor = 1.30 if promotion else 1.0

                expected_demand = (
                    base_demand
                    * multiplier
                    * weekend_factor
                    * seasonal_factor
                    * promotion_factor
                )

                # Customer demand is recorded even when inventory is insufficient.
                demand = int(rng.poisson(expected_demand))

                # Sales are limited by stock on hand.
                units_sold = min(demand, opening_stock)
                stockout = int(demand > opening_stock)
                lost_sales = demand - units_sold

                closing_stock = opening_stock - units_sold
                inventory = closing_stock

                # Estimate demand during the lead time.
                expected_daily_demand = (
                    base_demand * multiplier
                )

                lead_time_demand = (
                    expected_daily_demand * lead_time
                )

                # Approximate safety stock for variable daily demand.
                safety_stock = (
                    1.65
                    * np.sqrt(expected_daily_demand * lead_time)
                )

                reorder_point = (
                    lead_time_demand + safety_stock
                )

                # Inventory position includes outstanding orders.
                inventory_position = (
                    inventory + sum(pending_orders.values())
                )

                target_stock = (
                    expected_daily_demand
                    * (lead_time + review_period)
                    + safety_stock
                )

                # Place a replenishment order when needed.
                if inventory_position <= reorder_point:
                    order_quantity = max(
                        0,
                        int(
                            np.ceil(
                                target_stock - inventory_position
                            )
                        ),
                    )

                    if order_quantity > 0:
                        arrival_date = (
                            date + pd.Timedelta(days=lead_time)
                        )

                        pending_orders[arrival_date] = (
                            pending_orders.get(arrival_date, 0)
                            + order_quantity
                        )
                else:
                    order_quantity = 0

                rows.append(
                    {
                        "date": date,
                        "store_id": store,
                        "product_id": product,
                        "base_demand": round(base_demand, 2),
                        "promotion": promotion,
                        "demand": demand,
                        "opening_stock": opening_stock,
                        "units_sold": units_sold,
                        "closing_stock": closing_stock,
                        "stockout": stockout,
                        "lost_sales": lost_sales,
                        "unit_price": unit_price,
                        "sales_revenue": round(
                            units_sold * unit_price, 2
                        ),
                        "order_quantity": order_quantity,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_retail_data()

    output_path = DATA_DIR / "retail_sales.csv"
    df.to_csv(output_path, index=False)

    print("=" * 55)
    print("RETAIL DATA GENERATION COMPLETE")
    print("=" * 55)
    print(f"Saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: {df['date'].min().date()} to "
        f"{df['date'].max().date()}"
    )

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nData quality:")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nInventory metrics:")
    print(f"Total demand: {df['demand'].sum():,}")
    print(f"Total units sold: {df['units_sold'].sum():,}")
    print(f"Estimated lost sales: {df['lost_sales'].sum():,}")
    print(f"Stockout rate: {df['stockout'].mean():.2%}")
    print(f"Replenishment orders placed: {(df['order_quantity'] > 0).sum():,}")


if __name__ == "__main__":
    main()
