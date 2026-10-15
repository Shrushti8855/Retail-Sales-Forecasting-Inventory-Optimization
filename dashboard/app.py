
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales & Inventory Analytics",
    page_icon="📦",
    layout="wide",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "retail_sales.csv"
FORECAST_PATH = PROJECT_ROOT / "reports" / "forecast_predictions.csv"
INVENTORY_PATH = PROJECT_ROOT / "reports" / "inventory_recommendations.csv"

st.title("📦 Retail Sales Forecasting & Inventory Optimization")
st.caption(
    "Demand forecasting, sales analytics, stock monitoring, "
    "and data-driven replenishment recommendations."
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    forecasts["date"] = pd.to_datetime(forecasts["date"])
    inventory["inventory_date"] = pd.to_datetime(
        inventory["inventory_date"]
    )

    return sales, forecasts, inventory


try:
    sales, forecasts, inventory = load_data()
except FileNotFoundError as exc:
    st.error(f"Required project file not found: {exc.filename}")
    st.info(
        "Run the data generator, forecasting module, and inventory "
        "optimizer before starting the dashboard."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

store_options = sorted(sales["store_id"].unique())
product_options = sorted(sales["product_id"].unique())

selected_stores = st.sidebar.multiselect(
    "Select stores",
    options=store_options,
    default=store_options,
)

selected_products = st.sidebar.multiselect(
    "Select products",
    options=product_options,
    default=product_options,
)

if not selected_stores or not selected_products:
    st.warning("Select at least one store and one product.")
    st.stop()

filtered_sales = sales[
    sales["store_id"].isin(selected_stores)
    & sales["product_id"].isin(selected_products)
].copy()

filtered_forecasts = forecasts[
    forecasts["store_id"].isin(selected_stores)
    & forecasts["product_id"].isin(selected_products)
].copy()

filtered_inventory = inventory[
    inventory["store_id"].isin(selected_stores)
    & inventory["product_id"].isin(selected_products)
].copy()

if filtered_sales.empty:
    st.warning("No sales records match the selected filters.")
    st.stop()

# --------------------------------------------------
# Key performance indicators
# --------------------------------------------------
total_revenue = filtered_sales["sales_revenue"].sum()
total_units_sold = filtered_sales["units_sold"].sum()
total_demand = filtered_sales["demand"].sum()
total_lost_sales = filtered_sales["lost_sales"].sum()
stockout_rate = filtered_sales["stockout"].mean() * 100

recommended_units = filtered_inventory[
    "recommended_order_quantity"
].sum()

estimated_order_cost = filtered_inventory[
    "estimated_order_cost"
].sum()

st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Units Sold", f"{total_units_sold:,.0f}")
col3.metric("Simulated Demand", f"{total_demand:,.0f}")
col4.metric("Estimated Lost Sales", f"{total_lost_sales:,.0f}")

col5, col6, col7 = st.columns(3)

col5.metric("Record-Level Stockout Rate", f"{stockout_rate:.2f}%")
col6.metric("Recommended Order Units", f"{recommended_units:,.0f}")
col7.metric("Estimated Replenishment Cost", f"₹{estimated_order_cost:,.0f}")

st.caption(
    "These metrics come from a simulated retail dataset. "
    "Lost sales are estimated unmet demand, and stockout rate is "
    "the percentage of records marked as stockouts."
)

# --------------------------------------------------
# Sales trend
# --------------------------------------------------
st.divider()
st.subheader("📈 Sales and Demand Trends")

daily_sales = (
    filtered_sales.groupby("date", as_index=False)
    .agg(
        Revenue=("sales_revenue", "sum"),
        Units_Sold=("units_sold", "sum"),
        Demand=("demand", "sum"),
    )
    .sort_values("date")
)

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    st.markdown("**Daily Sales Revenue**")
    st.line_chart(daily_sales.set_index("date")["Revenue"])

with trend_col2:
    st.markdown("**Daily Demand vs Units Sold**")
    st.line_chart(daily_sales.set_index("date")[["Demand", "Units_Sold"]])

# --------------------------------------------------
# Demand forecasting
# --------------------------------------------------
st.divider()
st.subheader("🔮 Demand Forecast Evaluation")

if not filtered_forecasts.empty:
    forecast_metrics_col1, forecast_metrics_col2 = st.columns(2)

    if {"actual_demand", "predicted_demand"}.issubset(
        filtered_forecasts.columns
    ):
        evaluation = filtered_forecasts.dropna(
            subset=["actual_demand", "predicted_demand"]
        ).copy()

        if not evaluation.empty:
            mae = (
                evaluation["actual_demand"]
                - evaluation["predicted_demand"]
            ).abs().mean()

            forecast_metrics_col1.metric(
                "Filtered Forecast MAE", f"{mae:.2f} units"
            )

            evaluation["date"] = pd.to_datetime(evaluation["date"])

            daily_forecast = (
                evaluation.groupby("date", as_index=False)
                .agg(
                    Actual_Demand=("actual_demand", "sum"),
                    Predicted_Demand=("predicted_demand", "sum"),
                )
                .sort_values("date")
            )

            st.markdown("**Actual vs Predicted Demand (Test Period)**")
            st.line_chart(
                daily_forecast.set_index("date")[
                    ["Actual_Demand", "Predicted_Demand"]
                ]
            )
        else:
            st.info("No valid forecast evaluation rows for these filters.")
    else:
        st.warning(
            "Forecast file must contain actual_demand and predicted_demand "
            "columns to display evaluation."
        )
else:
    st.info("No forecast records match the selected filters.")

# --------------------------------------------------
# Inventory recommendations
# --------------------------------------------------
st.divider()
st.subheader("📦 Inventory Optimization Recommendations")

if not filtered_inventory.empty:
    inventory_view = filtered_inventory.copy()

    inventory_view["estimated_order_cost"] = (
        inventory_view["estimated_order_cost"].round(2)
    )

    reorder_count = (
        inventory_view["reorder_status"] == "REORDER"
    ).sum()

    inv_col1, inv_col2 = st.columns(2)

    inv_col1.metric(
        "Store-Product Combinations",
        len(inventory_view),
    )
    inv_col2.metric(
        "Combinations Flagged for Reorder",
        int(reorder_count),
    )

    st.dataframe(
        inventory_view.sort_values(
            "estimated_order_cost", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = inventory_view.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Inventory Recommendations (CSV)",
        data=csv_bytes,
        file_name="inventory_recommendations_filtered.csv",
        mime="text/csv",
    )
else:
    st.info("No inventory recommendations match the selected filters.")

# --------------------------------------------------
# Product-level analysis
# --------------------------------------------------
st.divider()
st.subheader("🏷️ Product Performance")

product_summary = (
    filtered_sales.groupby("product_id", as_index=False)
    .agg(
        Units_Sold=("units_sold", "sum"),
        Demand=("demand", "sum"),
        Revenue=("sales_revenue", "sum"),
        Lost_Sales=("lost_sales", "sum"),
        Stockout_Records=("stockout", "sum"),
    )
    .sort_values("Revenue", ascending=False)
)

product_col1, product_col2 = st.columns(2)

with product_col1:
    st.markdown("**Revenue by Product**")
    st.bar_chart(product_summary.set_index("product_id")["Revenue"])

with product_col2:
    st.markdown("**Units Sold by Product**")
    st.bar_chart(product_summary.set_index("product_id")["Units_Sold"])

st.dataframe(
    product_summary,
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Portfolio prototype | Python • Pandas • Scikit-learn • Streamlit "
    "| Synthetic data for demonstration, not live retail data."
)
