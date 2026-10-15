
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "retail_sales.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    # 1. Load the dataset.
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("RETAIL SALES DATA ANALYSIS")
    print("=" * 55)

    # 2. Basic dataset information.
    print("\n1. DATASET OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    print(f"Stores: {df['store_id'].nunique()}")
    print(f"Products: {df['product_id'].nunique()}")

    print("\n2. DATA TYPES")
    print(df.dtypes.to_string())

    # 3. Data quality checks.
    print("\n3. DATA QUALITY")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(
        "Duplicate store-product-date records:",
        df.duplicated(
            subset=["date", "store_id", "product_id"]
        ).sum(),
    )

    print("\n4. NUMERIC SUMMARY")
    print(
        df[[
            "demand",
            "units_sold",
            "opening_stock",
            "closing_stock",
            "unit_price",
            "sales_revenue",
        ]].describe().round(2).to_string()
    )

    # 4. Business metrics.
    total_demand = int(df["demand"].sum())
    total_sold = int(df["units_sold"].sum())
    total_revenue = df["sales_revenue"].sum()
    stockout_rate = df["stockout"].mean()

    print("\n5. BUSINESS METRICS")
    print(f"Total simulated demand: {total_demand:,} units")
    print(f"Total units sold: {total_sold:,}")
    print(f"Estimated unmet demand: {total_demand - total_sold:,}")
    print(f"Total simulated revenue: {total_revenue:,.2f}")
    print(f"Record-level stockout rate: {stockout_rate:.2%}")

    # 5. Compare demand on promotion and non-promotion days.
    print("\n6. PROMOTION ANALYSIS")
    promotion_summary = (
        df.groupby("promotion")
        .agg(
            average_demand=("demand", "mean"),
            average_units_sold=("units_sold", "mean"),
            records=("demand", "size"),
        )
        .round(2)
    )
    print(promotion_summary.to_string())

    # 6. Aggregate by month for a trend chart.
    monthly = (
        df.set_index("date")
        .resample("MS")
        .agg(
            total_demand=("demand", "sum"),
            total_revenue=("sales_revenue", "sum"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(
        monthly.index,
        monthly["total_demand"],
        marker="o",
    )
    axes[0].set_title("Monthly Simulated Retail Demand")
    axes[0].set_ylabel("Units demanded")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        monthly.index,
        monthly["total_revenue"],
        marker="o",
    )
    axes[1].set_title("Monthly Simulated Sales Revenue")
    axes[1].set_ylabel("Revenue (currency units)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    chart_path = REPORTS_DIR / "monthly_trends.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"\nChart saved to: {chart_path}")


if __name__ == "__main__":
    main()
