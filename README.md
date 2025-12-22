# ☕ Maven Roasters: AI-Powered Analytics Dashboard

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://px8m8ezwfxgx77kkxwgpje.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)

**A next-generation BI tool integrating historical sales analysis with predictive AI modeling.**

[**🚀 Launch Live Dashboard**](https://dashboardcoffeeshop.streamlit.app/)

</div>

---

## 📖 Overview

This project is a comprehensive **Coffee Shop Analytics Dashboard** built with Python and Streamlit. Unlike traditional dashboards that only look backward, this application integrates an **AI Future Insights Lab** to look forward.

The application analyzes sales transaction data to provide historical insights and uses **Machine Learning models** to predict future revenue, simulate new product launches, and analyze store geographical efficiency.

---

## 🚀 Key Features

### 📊 1. Business Intelligence Dashboard

| Feature | Description |
| :--- | :--- |
| **Net Sales Analysis** | Interactive bar charts visualizing monthly revenue performance across different store locations. |
| **Rush Detection** | Analysis of peak hours with standard deviation metrics to identify rush periods vs. quiet times. |
| **Product Size Analysis** | Stacked bar charts breaking down cup size preferences (Small, Medium, Large) by product category. |
| **Interactive Filters** | Global sidebar filters for Store Location and Product Category that update all visualizations instantly. |

### 🤖 2. AI Future Insights & Lab

> This application utilizes **Scikit-Learn** to power three distinct predictive modules:

#### 📈 Future Revenue Prediction
* **Model:** Polynomial Regression (Pipeline).
* **Function:** Forecasts sales trends 6–12 months into the future.
* **Logic:** Incorporates seasonal metadata (temperature and tourist index) to refine predictions.

#### 🧪 New Product Simulator
* **Model:** Random Forest Regressor.
* **Function:** Simulates the success of a hypothetical product launch.
* **Logic:** Takes into account "Real World Context" (Foot traffic, Office density) to predict unit sales and revenue.


---

## 🛠️ Tech Stack

<div align="center">

| Category | Libraries / Tools |
| :--- | :--- |
| **Frontend** | Streamlit (Custom CSS) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Plotly Graph Objects, Matplotlib |
| **Machine Learning** | Scikit-Learn (RandomForest, PolynomialFeatures) |

</div>

---

## 🧠 Machine Learning Logic

This project demonstrates several advanced data science techniques:

### 1. Data Enrichment
The raw CSV is enriched with synthetic **"Metadata" dictionaries** (e.g., `STORE_CONTEXT`), adding variables like:
* `foot_traffic`
* `office_density`
* `seasonal_data` (Temperature, Tourist Index)

### 2. Model Pipelines

| Pipeline | Composition | Reasoning |
| :--- | :--- | :--- |
| **Forecasting** | `PolynomialFeatures(degree=2)`  | Accounts for the non-linear curve of seasonal sales cycles. |
| **Simulation** | `ColumnTransformer` + `RandomForestRegressor` | Handles categorical data (Product Categories) to predict continuous targets (Sales Qty). |

---

## 📂 File Structure

```text
├── app.py                  # Main application code
├── Coffee Shop Sales.csv   # Dataset
├── README.md               # Documentation
├── packages.txt            # System dependencies
└── requirements.txt        # Python library dependencies
```

---

## 👥 Authors

<div align="center">

| **Hamza Tahboub** | **Majd Igbarea** | **Marysol Karwan** | **Igor Kornev** |
| :---: | :---: | :---: | :---: |
| *(The AI Builder)* | Contributor | Contributor | Contributor |

</div>