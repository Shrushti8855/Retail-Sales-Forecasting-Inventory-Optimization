
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Retail Sales Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide",
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "retail_sales.csv"
FORECAST_PATH = ROOT / "reports" / "forecast_predictions.csv"
INVENTORY_PATH = ROOT / "reports" / "inventory_recommendations.csv"


# --------------------------------------------------
# Page title
# --------------------------------------------------
st.title("📦 Retail Sales Forecasting & Inventory Optimization")
st.caption(
    "Demand forecasting, sales analytics, stock monitoring, "
    "and data-driven replenishment recommendations."
)


# --------------------------------------------------
# Load project data
# --------------------------------------------------
@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    inventory = pd.read_csv(
        INVENTORY_PATH,
        parse_dates=["inventory_date"],
    )
    return sales, forecasts, inventory


try:
    sales, forecasts, inventory = load_data()
except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc.filename}")
    st.info(
        "Run the data generator, demand forecasting module, "
        "and inventory optimizer before launching the dashboard."
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

sales = sales[
    sales["store_id"].isin(selected_stores)
    & sales["product_id"].isin(selected_products)
].copy()

forecasts = forecasts[
    forecasts["store_id"].isin(selected_stores)
    & forecasts["product_id"].isin(selected_products)
].copy()

inventory = inventory[
    inventory["store_id"].isin(selected_stores)
    & inventory["product_id"].isin(selected_products)
].copy()

if sales.empty:
    st.warning("No sales records match the selected filters.")
    st.stop()


# --------------------------------------------------
# Key performance indicators
# --------------------------------------------------
st.subheader("📊 Key Performance Indicators")

total_revenue = sales["sales_revenue"].sum()
total_units_sold = sales["units_sold"].sum()
total_demand = sales["demand"].sum()
total_lost_sales = sales["lost_sales"].sum()
stockout_rate = sales["stockout"].mean() * 100

recommended_units = inventory["recommended_order_quantity"].sum()
estimated_order_cost = inventory["estimated_order_cost"].sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Sales Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units_sold:,.0f}")
c3.metric("Simulated Demand", f"{total_demand:,.0f}")
c4.metric("Estimated Lost Sales", f"{total_lost_sales:,.0f}")

c5, c6, c7 = st.columns(3)

c5.metric("Record-Level Stockout Rate", f"{stockout_rate:.2f}%")
c6.metric("Recommended Order Units", f"{recommended_units:,.0f}")
c7.metric("Estimated Replenishment Cost", f"₹{estimated_order_cost:,.0f}")

st.caption(
    "Results use synthetic retail data. Lost sales represent estimated "
    "unmet demand. Stockout rate is the percentage of records marked "
    "as stockouts."
)


# --------------------------------------------------
# Monthly sales and demand trends
# --------------------------------------------------
st.divider()
st.subheader("📈 Sales and Demand Trends")

daily = (
    sales.groupby("date", as_index=False)
    .agg(
        Revenue=("sales_revenue", "sum"),
        Units_Sold=("units_sold", "sum"),
        Demand=("demand", "sum"),
    )
    .sort_values("date")
)

monthly = (
    daily.assign(
        Month=daily["date"].dt.to_period("M").dt.to_timestamp()
    )
    .groupby("Month", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Demand=("Demand", "sum"),
    )
    .sort_values("Month")
)

left, right = st.columns(2)

with left:
    st.markdown("**Monthly Sales Revenue**")
    st.line_chart(
        monthly.set_index("Month")[["Revenue"]],
        x_label="Month",
        y_label="Revenue (₹)",
    )

with right:
    st.markdown("**Monthly Demand vs Units Sold**")
    st.line_chart(
        monthly.set_index("Month")[["Demand", "Units_Sold"]],
        x_label="Month",
        y_label="Units",
    )


# --------------------------------------------------
# Demand forecast evaluation
# --------------------------------------------------
st.divider()
st.subheader("🔮 Demand Forecast Evaluation")

required_columns = {"date", "demand", "predicted_demand"}

if required_columns.issubset(forecasts.columns):
    evaluation = forecasts.dropna(
        subset=["demand", "predicted_demand"]
    ).sort_values("date").copy()

    if not evaluation.empty:
        evaluation["absolute_error"] = (
            evaluation["demand"] - evaluation["predicted_demand"]
        ).abs()

        forecast_mae = evaluation["absolute_error"].mean()

        m1, m2, m3 = st.columns(3)

        m1.metric("Filtered Forecast MAE", f"{forecast_mae:.2f} units")
        m2.metric("Forecast Test Records", f"{len(evaluation):,}")

        if "baseline_lag_7" in evaluation.columns:
            baseline_rows = evaluation.dropna(
                subset=["baseline_lag_7"]
            )

            if not baseline_rows.empty:
                baseline_mae = (
                    baseline_rows["demand"]
                    - baseline_rows["baseline_lag_7"]
                ).abs().mean()

                m3.metric(
                    "7-Day-Lag Baseline MAE",
                    f"{baseline_mae:.2f} units",
                )

        daily_forecast = (
            evaluation.groupby("date", as_index=False)
            .agg(
                Actual_Demand=("demand", "sum"),
                Predicted_Demand=("predicted_demand", "sum"),
            )
            .sort_values("date")
        )

        st.markdown("**Actual vs Predicted Demand — Test Period**")

        st.line_chart(
            daily_forecast.set_index("date")[
                ["Actual_Demand", "Predicted_Demand"]
            ],
            x_label="Date",
            y_label="Demand (units)",
        )

        st.caption(
            "MAE is the mean absolute error. Lower values indicate smaller "
            "average absolute prediction errors. Forecast evaluation uses "
            "the available test-period predictions."
        )
    else:
        st.info("No valid forecast records match the selected filters.")
else:
    st.warning(
        "The forecast file must contain date, demand, and predicted_demand."
    )


# --------------------------------------------------
# Inventory optimization recommendations
# --------------------------------------------------
st.divider()
st.subheader("📦 Inventory Optimization Recommendations")

if not inventory.empty:
    reorder_count = int(
        (inventory["reorder_status"] == "REORDER").sum()
    )

    i1, i2 = st.columns(2)

    i1.metric("Store-Product Combinations", len(inventory))
    i2.metric("Combinations Flagged for Reorder", reorder_count)

    st.dataframe(
        inventory.sort_values(
            "estimated_order_cost",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        label="Download Inventory Recommendations (CSV)",
        data=inventory.to_csv(index=False).encode("utf-8"),
        file_name="inventory_recommendations.csv",
        mime="text/csv",
    )
else:
    st.info("No inventory recommendations match the selected filters.")


# --------------------------------------------------
# Product performance
# --------------------------------------------------
st.divider()
st.subheader("🏷️ Product Performance")

product_summary = (
    sales.groupby("product_id", as_index=False)
    .agg(
        Units_Sold=("units_sold", "sum"),
        Demand=("demand", "sum"),
        Revenue=("sales_revenue", "sum"),
        Lost_Sales=("lost_sales", "sum"),
        Stockout_Records=("stockout", "sum"),
    )
    .sort_values("Revenue", ascending=False)
)

product_left, product_right = st.columns(2)

with product_left:
    st.markdown("**Revenue by Product**")
    st.bar_chart(
        product_summary.set_index("product_id")[["Revenue"]],
        x_label="Product",
        y_label="Revenue (₹)",
    )

with product_right:
    st.markdown("**Units Sold by Product**")
    st.bar_chart(
        product_summary.set_index("product_id")[["Units_Sold"]],
        x_label="Product",
        y_label="Units Sold",
    )

st.dataframe(
    product_summary,
    width="stretch",
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
