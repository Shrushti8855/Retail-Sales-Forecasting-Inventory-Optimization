
# Retail Sales Forecasting & Inventory Optimization System

An end-to-end data science project that uses simulated retail data, machine learning, and inventory planning logic to support demand forecasting and replenishment decisions.

## Project Overview

Retail businesses need to estimate product demand and maintain appropriate inventory levels. Understocking can lead to lost sales, while excess inventory can increase holding costs.

This project combines demand forecasting with a rule-based inventory optimization module and an interactive Streamlit dashboard.

**Note:** The dataset is synthetic and generated for educational and portfolio demonstration. Results do not represent a real retailer.

## Key Features

- Synthetic retail dataset covering 3 stores, 10 products, and 365 days.
- Exploratory data analysis and data-quality checks.
- Random Forest demand forecasting.
- Time-based features, lag features, and rolling demand averages.
- Comparison with a 7-day-lag baseline.
- Safety stock and reorder point calculations.
- Recommended replenishment quantities and estimated order costs.
- Interactive Streamlit dashboard with store and product filters.
- CSV export of inventory recommendations.
- Automated tests using pytest.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Pytest
- Git and GitHub

## Project Structure

```text
Retail-Sales-Forecasting-Inventory-Optimization/
├── dashboard/
│   └── app.py
├── data/
│   └── retail_sales.csv
├── models/
│   └── demand_forecaster.joblib
├── reports/
│   ├── monthly_trends.png
│   ├── demand_forecast_test.png
│   ├── forecast_predictions.csv
│   └── inventory_recommendations.csv
├── src/
│   ├── data_generator.py
│   ├── analyze_data.py
│   ├── forecast_demand.py
│   └── optimize_inventory.py
├── tests/
│   └── test_project.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Retail-Sales-Forecasting-Inventory-Optimization
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Project

### Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal.

### Regenerate the synthetic dataset

```bash
python src/data_generator.py
```

### Analyze the data

```bash
python src/analyze_data.py
```

### Train and evaluate the demand forecasting model

```bash
python src/forecast_demand.py
```

### Generate inventory recommendations

```bash
python src/optimize_inventory.py
```

Run these commands in order when regenerating the project outputs.

## Forecasting Results

Evaluation on the chronological test period produced the following results on the synthetic dataset:

| Metric | Random Forest |
|---|---:|
| Mean Absolute Error (MAE) | 4.87 units |
| Root Mean Squared Error (RMSE) | 6.26 units |
| R² Score | 0.8803 |

The 7-day-lag baseline achieved an MAE of 7.40 units. The Random Forest model reduced MAE by approximately 34.2% relative to this baseline.

These results measure performance on simulated data and should not be interpreted as evidence of equivalent performance on real retail data.

## Inventory Optimization Logic

The inventory module uses forecast daily demand, a five-day lead-time assumption, historical demand variability, and a service-factor parameter to calculate:

- Safety stock
- Reorder point
- Target stock
- Recommended order quantity
- Estimated replenishment cost

The current prototype uses on-hand stock and does not fully account for outstanding purchase orders or real supplier constraints. Its recommendations are illustrative rather than production-ready.

## Automated Testing

Run the tests with:

```bash
python -m pytest tests/test_project.py -v
```

The current test suite validates data quality, quantity constraints, forecast outputs, inventory recommendations, and order-cost calculations.

## Future Improvements

- Validate the forecasting approach on real retail datasets.
- Compare Random Forest with time-series forecasting models.
- Improve stockout and inventory-position calculations.
- Include outstanding orders, supplier constraints, and holding costs.
- Add forecast uncertainty and service-level analysis.
- Deploy the dashboard for public demonstration.

## Skills Demonstrated

Python, data cleaning, exploratory data analysis, feature engineering, supervised machine learning, model evaluation, inventory planning, dashboard development, CSV data handling, and automated testing.

## Disclaimer

This is an educational portfolio project using synthetic data. Inventory recommendations are not intended for direct use in real-world business operations.
