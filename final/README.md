☕ Maven Roasters: AI-Powered Analytics Dashboard
📖 Overview
This project is a comprehensive Coffee Shop Analytics Dashboard built with Python and Streamlit. It goes beyond traditional Business Intelligence by integrating an AI Future Insights Lab.

The application analyzes sales transaction data to provide historical insights and uses Machine Learning models to predict future revenue, simulate new product launches, and analyze store geographical efficiency.

dashboard link: https://px8m8ezwfxgx77kkxwgpje.streamlit.app/

🚀 Key Features
📊 1. Business Intelligence Dashboard
Net Sales Analysis: Interactive bar charts visualizing monthly revenue performance across different store locations.

Hourly Traffic & Rush Detection: Analysis of peak hours with standard deviation metrics to identify rush periods vs. quiet times.

Product Size Analysis: Stacked bar charts breaking down cup size preferences (Small, Medium, Large) by product category.

Interactive Filters: Global sidebar filters for Store Location and Product Category that update all visualizations instantly.

🤖 2. AI Future Insights & Lab
This application utilizes Scikit-Learn to power three distinct predictive modules:

📈 Future Revenue Prediction: * Uses Polynomial Regression to forecast sales trends 6-12 months into the future.

Incorporates seasonal data (temperature and tourist index) to refine predictions.

🧪 New Product Simulator: * Uses a Random Forest Regressor to simulate the success of a hypothetical product launch.

Takes into account "Real World Context" (Foot traffic, Office density) to predict unit sales and revenue.

🗺️ Geographical Efficiency Analysis: * Uses Linear Regression to predict the ideal square footage of a store based on its "Bakery vs. Coffee" sales ratio.

Calculates an efficiency score (Sales per Predicted Sq. Ft.) to identify over/under-performing real estate.

🛠️ Tech Stack
Frontend: Streamlit (Custom CSS for UI enhancements)

Data Manipulation: Pandas, NumPy

Visualization: Plotly Graph Objects, Matplotlib

Machine Learning: Scikit-Learn (Linear Regression, Random Forest, Polynomial Features, Pipelines)

🧠 Machine Learning Logic
This project demonstrates several data science techniques:

Data Enrichment:

The raw CSV is enriched with synthetic "Metadata" dictionaries (e.g., STORE_CONTEXT), adding variables like foot_traffic, office_density, and sq_ft to allow for more complex modeling.

Seasonal data (Temperature, Tourist Index) is mapped to transaction dates.

Models Used:

Pipeline 1 (Forecasting): PolynomialFeatures(degree=2) + LinearRegression. Accounts for the non-linear curve of seasonal sales.

Pipeline 2 (Simulation): ColumnTransformer (OneHotEncoding) + RandomForestRegressor. Handles categorical data (Product Categories) to predict continuous targets (Sales Qty).

Pipeline 3 (Real Estate): LinearRegression. Finds the correlation between product mix (Bakery Ratio) and required physical space.

📂 File Structure
├── app.py                  # Main application code
├── Coffee Shop Sales.csv   # Dataset
├── README.md               # Documentation
|-- packages.txt            # Dependencies
└── requirements.txt        # Dependencies
👥 Authors
Hamza Tahboub(The AI builder)

Majd Igbarea

Marysol Karwan

Igor Kornov