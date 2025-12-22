import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 1. SETUP PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Coffee shop Dashboard",
    layout="wide"
)

# --- 2. SETUP MATPLOTLIB (NON-INTERACTIVE) ---
import matplotlib
matplotlib.use('Agg') # Prevents black screen/crashing on servers
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go

# --- 3. APPLY CSS STYLING ---

def apply_glowing_selectbox():
    st.markdown("""
    <style>

    /* =========================
       Base state (NO glow)
       ========================= */

    div[data-baseweb="select"] div[role="combobox"] {
        min-height: 60px;
        border-radius: 14px;
        background-color: #111111;
        border: 2px solid #222c;
        box-shadow: none;
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
    }

    div[data-baseweb="select"] span {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        text-shadow: none;
        transition: text-shadow 0.25s ease;
    }

    label {
        font-size: 20px !important;
        font-weight: 700;
        color: #ffffff;
        text-shadow: none;
        transition: text-shadow 0.25s ease;
    }

    svg {
        filter: none;
        transition: filter 0.25s ease;
    }

    /* =========================
       Hover state (GLOW ONLY)
       ========================= */

    div[data-baseweb="select"] div[role="combobox"]:hover {
        border-color: #00ffff;
        box-shadow: 0 0 22px rgba(0, 255, 255, 0.7);
    }

    div[data-baseweb="select"] div[role="combobox"]:hover span {
        text-shadow:
            0 0 6px #00e5ff,
            0 0 18px #00ffff;
    }

    div[data-baseweb="select"] div[role="combobox"]:hover svg {
        filter: drop-shadow(0 0 6px #00e5ff);
    }

    label:hover {
        text-shadow:
            0 0 6px #00e5ff,
            0 0 18px #00ffff;
    }

    </style>
    """, unsafe_allow_html=True)

# Call once
apply_glowing_selectbox()

# --- 4. CONFIG & ICONS ---
STORE_ICONS = {
    "All Stores": "🏬", "Lower Manhattan": "🏙️", "Hell's Kitchen": "🌆", "Astoria": "🏘️"
}
CATEGORY_ICONS = {
    "All Categories": "📦", "Coffee": "☕", "Tea": "🍵", "Drinking Chocolate": "🍫",
    "Bakery": "🥐", "Flavours": "🧴", "Loose Tea": "🌿", "Coffee beans": "🫘",
    "Packaged Chocolate": "🎁", "Branded": "🏷️"
}

mpl.rcParams.update({
    "text.color": "#ffffff", "axes.labelcolor": "#ffffff", "axes.titlecolor": "#ffffff",
    "xtick.color": "#d1d5db", "ytick.color": "#d1d5db",
})

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# --- 5. DATA LOADING ---
@st.cache_data
def load_data():
    possible_paths = ['final/Coffee Shop Sales.csv', 'Coffee Shop Sales.csv', 'coffee_shop_sales.csv']
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error("❌ Could not find 'Coffee Shop Sales.csv'. Please check your file path.")
    st.stop()
    return None

data_set = load_data()
store_locations = ['All Stores'] + sorted(data_set['store_location'].unique())

# --- 6. AI ENGINE ---
STORE_CONTEXT = {
    "Lower Manhattan": {"sq_ft": 1200, "foot_traffic": 95, "office_density": 0.9},
    "Hell's Kitchen":  {"sq_ft": 800,  "foot_traffic": 85, "office_density": 0.4},
    "Astoria":         {"sq_ft": 1500, "foot_traffic": 60, "office_density": 0.2},
    "All Stores":      {"sq_ft": 3500, "foot_traffic": 80, "office_density": 0.5}
}
TRAINING_DATA_REALITY = {"Lower Manhattan": 1200, "Hell's Kitchen": 800, "Astoria": 1500}
NYC_SEASONAL_DATA = {
    1: {"avg_temp": 33, "tourist_index": 0.8}, 2: {"avg_temp": 35, "tourist_index": 0.8},
    3: {"avg_temp": 42, "tourist_index": 1.0}, 4: {"avg_temp": 53, "tourist_index": 1.1},
    5: {"avg_temp": 62, "tourist_index": 1.2}, 6: {"avg_temp": 72, "tourist_index": 1.4},
    7: {"avg_temp": 77, "tourist_index": 1.5}, 8: {"avg_temp": 75, "tourist_index": 1.5},
    9: {"avg_temp": 68, "tourist_index": 1.3}, 10: {"avg_temp": 58, "tourist_index": 1.2},
    11: {"avg_temp": 48, "tourist_index": 1.1}, 12: {"avg_temp": 38, "tourist_index": 1.6}
}

class AIEngine:
    def __init__(self, df):
        self.df = df.copy()
        # Safe Date Parsing
        self.df['tx_date'] = pd.to_datetime(self.df['transaction_date'], dayfirst=True, errors='coerce')
        self._enrich_data()
        self.qty_model = None
        self.area_model = None

    def _enrich_data(self):
        for store, meta in STORE_CONTEXT.items():
            if store == 'All Stores': continue
            mask = self.df['store_location'] == store
            for key, value in meta.items():
                self.df.loc[mask, key] = value
        
        self.df['month_num'] = self.df['tx_date'].dt.month
        self.df['avg_temp'] = self.df['month_num'].map(lambda x: NYC_SEASONAL_DATA.get(x, {}).get('avg_temp', 50))
        self.df['tourist_index'] = self.df['month_num'].map(lambda x: NYC_SEASONAL_DATA.get(x, {}).get('tourist_index', 1.0))
        self.df.fillna(0, inplace=True)

    def get_future_forecast(self, store_name, days=30):
        """
        Predicts Net Sales for the next N days (default 30) using Polynomial Regression.
        """
        data = self.df.copy()
        if store_name != 'All Stores':
            data = data[data['store_location'] == store_name]
            
        # Group by DAY instead of Month
        data['date_idx'] = data['tx_date'].dt.date
        daily = data.groupby('date_idx').agg({
            'net_sales': 'sum', 
            'avg_temp': 'mean', 
            'tourist_index': 'mean',
            'month_num': 'mean' # Keep track of month to look up weather later
        }).reset_index()
        
        # Convert date to an integer (ordinal) for regression
        daily['date_ordinal'] = pd.to_datetime(daily['date_idx']).map(pd.Timestamp.toordinal)

        if len(daily) < 10: return pd.DataFrame() # Need more daily points for a good curve

        X = daily[['date_ordinal', 'avg_temp', 'tourist_index']]
        y = daily['net_sales']

        # Train Model on Daily Data
        poly_model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])
        poly_model.fit(X, y)
        
        last_date = pd.to_datetime(daily['date_idx'].max())
        future_rows = []
        
        # Loop for next 30 DAYS
        for i in range(1, days + 1):
            next_date = last_date + pd.Timedelta(days=i)
            season = NYC_SEASONAL_DATA.get(next_date.month, {})
            
            future_rows.append({
                'date_ordinal': next_date.toordinal(),
                'avg_temp': season.get('avg_temp', 50),
                'tourist_index': season.get('tourist_index', 1.0),
                'Date': next_date
            })
            
        future_X = pd.DataFrame(future_rows)
        predictions = poly_model.predict(future_X[['date_ordinal', 'avg_temp', 'tourist_index']])
        
        return pd.DataFrame({"Date": future_X['Date'], "Predicted_Sales": predictions})

    def predict_store_area_from_usage(self):
        store_stats = self.df.groupby('store_location').agg(
            total_qty=('transaction_qty', 'sum'),
            bakery_qty=('transaction_qty', lambda x: x[self.df['product_category'] == 'Bakery'].sum())
        ).reset_index()
        store_stats['bakery_ratio'] = store_stats['bakery_qty'] / store_stats['total_qty']
        store_stats['actual_sq_ft'] = store_stats['store_location'].map(TRAINING_DATA_REALITY)
        train_data = store_stats.dropna(subset=['actual_sq_ft'])

        self.area_model = LinearRegression()
        self.area_model.fit(train_data[['total_qty', 'bakery_ratio']], train_data['actual_sq_ft'])
        store_stats['predicted_sq_ft'] = self.area_model.predict(store_stats[['total_qty', 'bakery_ratio']])
        return store_stats[['store_location', 'predicted_sq_ft', 'total_qty']]

    def simulate_new_product(self, category, price, store_name, size_name):
        if not self.qty_model: self.train_scenario_model()
        
        context = STORE_CONTEXT.get(store_name, STORE_CONTEXT["All Stores"])
        input_data = pd.DataFrame({
            'product_category': [category], 'unit_price': [price],
            'foot_traffic': [context['foot_traffic']], 'office_density': [context['office_density']]
        })
        
        # 1. Base Prediction
        avg_qty = self.qty_model.predict(input_data)[0]
        size_factor = 0.8 if size_name.lower() in ['mega', 'huge', 'xl'] else 1.0
        
        estimated_units = avg_qty * 300 * size_factor
        gross_revenue = estimated_units * price
        
        # 2. SMART CANNIBALIZATION LOGIC (Price-Based)
        cannibalization_loss = 0.0
        
        # Get historical data for this category to find the "Reference Price"
        # (e.g., What do people usually pay for Tea?)
        cat_data = self.df[self.df['product_category'] == category]
        
        if not cat_data.empty:
            historical_avg_price = cat_data['unit_price'].mean()
            
            # THE LOGIC: If new price is significantly cheaper (e.g., < 90% of usual), 
            # people will likely trade down.
            if price < (historical_avg_price * 0.9):
                # Calculate the price gap
                price_gap = historical_avg_price - price
                
                # Assume 25% of the new units come from existing customers switching down
                switchers = estimated_units * 0.25
                
                # The loss is the money we WOULD have made if they stayed with the expensive option
                cannibalization_loss = switchers * price_gap

        return estimated_units, gross_revenue, cannibalization_loss, context
            

    def train_scenario_model(self):
        features = ['product_category', 'unit_price', 'foot_traffic', 'office_density']
        target = 'transaction_qty'
        train_data = self.df.dropna(subset=features + [target])
        preprocessor = ColumnTransformer(transformers=[
            ('num', 'passthrough', ['unit_price', 'foot_traffic', 'office_density']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['product_category'])
        ])
        self.qty_model = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=50, random_state=42))])
        self.qty_model.fit(train_data[features], train_data[target])
    def get_model_performance_metrics(self):
        metrics = {}

        # --- CHECK 1: Product Simulator (Projected MONTHLY Accuracy) ---
        if not self.qty_model: self.train_scenario_model()
        
        features = ['product_category', 'unit_price', 'foot_traffic', 'office_density']
        target = 'transaction_qty'
        data = self.df.dropna(subset=features + [target])
        
        X = data[features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        test_model = self.qty_model
        test_model.fit(X_train, y_train)
        
        # 1. Predict Units for single transactions
        pred_qty = test_model.predict(X_test)
        
        # 2. Calculate Revenue Error per Transaction
        actual_revenue = y_test * X_test['unit_price']
        predicted_revenue = pred_qty * X_test['unit_price']
        
        per_transaction_rmse = np.sqrt(mean_squared_error(actual_revenue, predicted_revenue))
        
        # 3. SCALE TO MONTHLY (The Fix)
        # If we are wrong by $1.74 per customer, and we expect ~300 customers/month:
        # We project the total monthly risk.
        metrics['simulator_rmse'] = per_transaction_rmse * 300 
        
        metrics['simulator_r2'] = r2_score(actual_revenue, predicted_revenue)

        # --- CHECK 2: Sales Forecast (Daily Accuracy) ---
        # (This remains unchanged as it is already correct for Daily aggregation)
        data = self.df.copy()
        data['date_idx'] = data['tx_date'].dt.date
        daily = data.groupby('date_idx').agg({
            'net_sales': 'sum', 
            'avg_temp': 'mean', 
            'tourist_index': 'mean'
        }).reset_index()
        daily['date_ordinal'] = pd.to_datetime(daily['date_idx']).map(pd.Timestamp.toordinal)
        
        if len(daily) > 10:
            X = daily[['date_ordinal', 'avg_temp', 'tourist_index']]
            y = daily['net_sales']
            
            poly_model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])
            poly_model.fit(X, y)
            preds = poly_model.predict(X)
            
            metrics['forecast_rmse'] = np.sqrt(mean_squared_error(y, preds))
            metrics['forecast_r2'] = r2_score(y, preds)
        else:
            metrics['forecast_rmse'] = 0
            metrics['forecast_r2'] = 0

        # --- CHECK 3: Store Size (No changes) ---
        store_stats = self.df.groupby('store_location').agg(
            total_qty=('transaction_qty', 'sum'),
            bakery_qty=('transaction_qty', lambda x: x[self.df['product_category'] == 'Bakery'].sum())
        ).reset_index()
        store_stats['bakery_ratio'] = store_stats['bakery_qty'] / store_stats['total_qty']
        store_stats['actual_sq_ft'] = store_stats['store_location'].map(TRAINING_DATA_REALITY)
        check_data = store_stats.dropna(subset=['actual_sq_ft'])
        
        if not check_data.empty:
            model = LinearRegression()
            X = check_data[['total_qty', 'bakery_ratio']]
            y = check_data['actual_sq_ft']
            model.fit(X, y)
            preds = model.predict(X)
            metrics['size_rmse'] = np.sqrt(mean_squared_error(y, preds))
        else:
            metrics['size_rmse'] = 0
            
        return metrics

def render_ai_dashboard(df):
    df = df.copy()
    if 'net_sales' not in df.columns:
        df['net_sales'] = df['transaction_qty'] * df['unit_price']

    ai = AIEngine(df)
    st.title("🤖 AI Future Insights & Lab")
    st.divider()

    st.subheader("1. 📈 Future Revenue Prediction (Next 30 Days)")
    c1, c2 = st.columns([1, 3])
    with c1:
        target_store = st.selectbox("Select Store", store_locations, key="ai_store")
        # Removed the slider for months since we are fixed to 30 days
        st.info("Predicting daily sales for the upcoming month.")
        
    with c2:
        # Call with default days=30
        forecast_df = ai.get_future_forecast(target_store, days=30)
        
        if not forecast_df.empty:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=forecast_df['Date'], 
                y=forecast_df['Predicted_Sales'], 
                mode='lines+markers', 
                name='Daily Forecast', 
                line=dict(color='#00e5ff', width=2)
            ))
            fig_pred.update_layout(
                title=f"30-Day Forecast: {target_store}", 
                template="plotly_dark", 
                height=350,
                xaxis_title="Date",
                yaxis_title="Predicted Sales ($)"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("Not enough daily data to generate a forecast.")

    st.divider()
    st.subheader("2. 🧪 New Product Simulator")

    st.markdown("Predict the impact of adding a **new size** or product type based on store geography.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_store = st.selectbox("Target Store", sorted(STORE_CONTEXT.keys()))
    with col2:
        sim_cat = st.selectbox("Category", ['Tea','Coffee', 'Bakery', 'Drinking Chocolate'])
        sim_size = st.text_input("New Size Name", "Small")
    with col3:
        sim_price = st.number_input("Unit Price ($)", 1.5, 15.0, 2.0)

    if st.button("🔮 Simulate Launch Results"):
        # Unpack the 4 values (including the new 'loss')
        units, gross_rev, loss, ctx = ai.simulate_new_product(sim_cat, sim_price, sim_store, sim_size)
        
        net_revenue = gross_rev - loss
        
        st.markdown("#### 📊 Simulation Results (Net Impact)")
        
        # Row 1: The Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Est. Monthly Units", f"{int(units)}")
        m2.metric("Gross Revenue", f"${gross_rev:,.2f}")
        
        # --- CONDITIONAL UI LOGIC ---
        if loss > 0:
            # Case A: Cannibalization Detected -> Show Red Delta & Warning
            m3.metric(
                "Net Revenue Impact", 
                f"${net_revenue:,.2f}", 
                delta=f"-${loss:,.2f} Cannibalization Risk",
                delta_color="inverse" # Makes the negative delta red
            )
            
            st.warning(
                f"**⚠️ Cannibalization Detected:** "
                f"The model predicts that adding a **{sim_size} {sim_cat}** will cause some existing customers to switch from larger, more expensive sizes.\n\n"
                f"* **Gross Sales:** ${gross_rev:,.2f}\n"
                f"* **Cannibalization Loss:** -${loss:,.2f}\n"
                f"* **Real Business Value:** ${net_revenue:,.2f}"
            )
        else:
            # Case B: No Risk -> Show Clean Metric & Success Message
            m3.metric("Net Revenue Impact", f"${net_revenue:,.2f}")
            
            st.success(
                f"✅ **Safe Launch:** No significant cannibalization risk detected for **{sim_size} {sim_cat}** in {sim_store}."
            )


    st.divider()
    with st.expander("🔬 Model Performance Metrics (RMSE & Accuracy)"):
        st.markdown("""
        **Technical Validation Report:**
        * **RMSE (Root Mean Square Error):** On average, how much is the prediction off? (Lower is better)
        * **R² Score:** How well does the model explain the variance? (100 is perfect, 0 is random guessing)
        """)
        
        # Calculate scores on the fly
        scores = ai.get_model_performance_metrics()
        
        c1, c2 = st.columns(3)
        
        with c1:
            st.markdown("#### 🛒 Product Simulator")
            # CHANGED: Label now clarifies this is "Monthly" error
            st.metric("Proj. Monthly Error", f"±${scores.get('simulator_rmse', 0):,.2f}")
            st.metric("R² Accuracy", f"{scores.get('simulator_r2', 0):.2%}")

        with c2:
            st.markdown("#### 📈 Revenue Forecast")
            st.metric("RMSE (Dollars)", f"${scores.get('forecast_rmse', 0):,.2f}")
            st.metric("Fit Score (R²)", f"{scores.get('forecast_r2', 0):.2%}")
            
# --- 7. MAIN NAVIGATION & DASHBOARD ---
if 'current_page' not in st.session_state: st.session_state['current_page'] = 'Dashboard'

with st.sidebar:
    st.title("Navigation")
    pg = st.radio("Go to:", ['📊 Main Dashboard', '🤖 AI Predictions'], index=0 if st.session_state['current_page'] == 'Dashboard' else 1)
    st.session_state['current_page'] = 'Dashboard' if pg == '📊 Main Dashboard' else 'AI'
    st.divider()

if st.session_state['current_page'] == 'AI':
    render_ai_dashboard(data_set)
else:
    # --- DASHBOARD LOGIC ---
    st.title("📊 Maven Roasters Dashboard")
    st.divider()

    # Sidebar Filters
    if 'selected_store' not in st.session_state: st.session_state['selected_store'] = 'All Stores'
    if 'selected_category' not in st.session_state: st.session_state['selected_category'] = 'All Categories'
    
    cats = ['All Categories'] + data_set.groupby('product_category')['transaction_qty'].sum().sort_values(ascending=False).index.tolist()
    
    with st.sidebar:
        st.markdown("#### Store location:")
        st.session_state['selected_store'] = st.radio(
            '', 
            store_locations, 
            index=store_locations.index(st.session_state['selected_store']), 
            key="store_rad",
            # --- RESTORED ICONS HERE ---
            format_func=lambda x: f"{STORE_ICONS.get(x, '📍')}  {x}"
        )
        
        st.markdown("#### Product category:")
        st.session_state['selected_category'] = st.radio(
            '', 
            cats, 
            index=cats.index(st.session_state['selected_category']), 
            key="cat_rad",
            # --- RESTORED ICONS HERE ---
            format_func=lambda x: f"{CATEGORY_ICONS.get(x, '🔹')}  {x}"
        )

    # Filter Data
    df_filtered = data_set.copy()
    if st.session_state['selected_store'] != 'All Stores':
        df_filtered = df_filtered[df_filtered['store_location'] == st.session_state['selected_store']]
    if st.session_state['selected_category'] != 'All Categories':
        df_filtered = df_filtered[df_filtered['product_category'] == st.session_state['selected_category']]

    # 1. Net Sales Graph
    st.subheader("Net Sales by Month")
    df_sales = data_set.copy()
    if st.session_state['selected_store'] != 'All Stores':
        df_sales = df_sales[df_sales['store_location'] == st.session_state['selected_store']]
    
    # DATE FIX
    df_sales['transaction_date'] = pd.to_datetime(df_sales['transaction_date'], dayfirst=True, errors='coerce')
    df_sales = df_sales.dropna(subset=['transaction_date'])
    df_sales['YearMonth'] = df_sales['transaction_date'].dt.to_period('M').astype(str)
    df_sales['net_sales'] = df_sales['transaction_qty'] * df_sales['unit_price']

    # Get Last 6 Months
    valid_months = sorted(df_sales['YearMonth'].unique())[-6:]
    df_sales = df_sales[df_sales['YearMonth'].isin(valid_months)]
    
    # Prepare data for Bar (Sales)
    summary = df_sales.groupby(['YearMonth', 'store_location'])['net_sales'].sum().unstack(fill_value=0)
    
    # Prepare data for Line (Quantity)
    qty_summary = df_sales.groupby('YearMonth')['transaction_qty'].sum()
    
    # Format X-Axis (Remove Year 2023)
    x_labels = [pd.to_datetime(ym).strftime('%b') for ym in summary.index]

    fig_sales = go.Figure()
    
    # Add Bars
    for col in summary.columns:
        fig_sales.add_trace(go.Bar(x=summary.index, y=summary[col], name=col))
    
    # Add Line
    fig_sales.add_trace(go.Scatter(
        x=summary.index,
        y=qty_summary,
        name='Purchased Items',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='white', width=3)
    ))

    fig_sales.update_layout(
        barmode='group', 
        template="plotly_dark", 
        title="Net Sales & Items Purchased (Last 6 Months)",
        xaxis=dict(tickmode='array', tickvals=summary.index, ticktext=x_labels),
        yaxis=dict(title="Net Sales ($)"),
        yaxis2=dict(title="Items Quantity", overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    # 2. Hourly Graph
    st.subheader("Hourly Transactions (Average over 6 Months)")
    
    df_hourly = df_filtered.copy()
    
    # Ensure date parsing
    if not pd.api.types.is_datetime64_any_dtype(df_hourly['transaction_date']):
         df_hourly['transaction_date'] = pd.to_datetime(df_hourly['transaction_date'], dayfirst=True, errors='coerce')
    
    # Filter for the same 6 months
    df_hourly['YearMonth'] = df_hourly['transaction_date'].dt.to_period('M').astype(str)
    df_hourly = df_hourly[df_hourly['YearMonth'].isin(valid_months)]

    if not df_hourly.empty:
        # Extract Hour and Date
        df_hourly['hour'] = df_hourly['transaction_time'].astype(str).str.split(':').str[0].astype(int)
        df_hourly['date_only'] = df_hourly['transaction_date'].dt.date
        
        # 1. Sum Volume per Hour per Day (e.g., Total coffee sold at 8 AM on Jan 1st)
        daily_hourly_volume = df_hourly.groupby(['date_only', 'hour'])['transaction_qty'].sum().reset_index()
        
        # 2. Average those daily volumes across all days in the 6 months
        hourly_stats = daily_hourly_volume.groupby('hour')['transaction_qty'].agg(['mean', 'std']).reset_index().fillna(0)
        
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(
            x=hourly_stats['hour'], 
            y=hourly_stats['mean'], 
            name="Avg Volume", 
            marker_color='#FF7F50'
        ))
        
        # Optional: Add Standard Deviation Lines to show variability
        fig_hr.add_trace(go.Scatter(
            x=hourly_stats['hour'], 
            y=hourly_stats['mean'] + hourly_stats['std'], 
            name="Deviation (+1 Std)", 
            line=dict(dash='dash', color='#286090')
        ))

        # Deviation (-1 Std) - The Floor
        fig_hr.add_trace(go.Scatter(
            x=hourly_stats['hour'], 
            y=(hourly_stats['mean'] - hourly_stats['std']).clip(lower=0), # Ensures it doesn't go below 0
            name="Deviation (-1 Std)", 
            line=dict(dash='dash', color='#286090'),
            showlegend=True
        ))
        
        fig_hr.update_layout(
            title="Average Items Sold per Hour (Last 6 Months)", 
            xaxis=dict(title="Hour of Day", tickmode='linear', dtick=1),
            yaxis=dict(title="Avg Quantity Sold"),
            template="plotly_dark"
        )
        st.plotly_chart(fig_hr, use_container_width=True)
    else:
        st.info("No data for current filters.")

    # 3. Sizes Analysis
    st.subheader("Cup Sizes per Category")
    df_sizes = df_filtered[df_filtered['product_category'] != 'Coffee beans'].copy()
    
    def get_size(detail):
        d = str(detail).lower()
        if 'small' in d or 'sm' in d: return 'Small'
        if 'medium' in d or 'med' in d: return 'Medium'
        if 'large' in d or 'lg' in d: return 'Large'
        return None
    
    df_sizes['Size'] = df_sizes['product_detail'].apply(get_size)
    df_sizes = df_sizes.dropna(subset=['Size'])
    
    if not df_sizes.empty:
        # Create the pivot table
        size_counts = df_sizes.groupby(['product_category', 'Size'])['transaction_qty'].sum().unstack(fill_value=0)
        
        # --- NEW SORTING LOGIC (ASCENDING) ---
        # 1. Calculate total for each category to determine order
        size_counts['Total_Qty'] = size_counts.sum(axis=1)
        # 2. Sort Ascending (Smallest -> Largest)
        size_counts = size_counts.sort_values('Total_Qty', ascending=True)
        # 3. Remove the helper column so it doesn't get plotted
        size_counts = size_counts.drop(columns=['Total_Qty'])
        # -------------------------------------

        fig_sz = go.Figure()
        colors = {"Small": "#A9CCE3", "Medium": "#5499C7", "Large": "#154360"}
        
        for sz in ['Small', 'Medium', 'Large']:
            if sz in size_counts.columns:
                fig_sz.add_trace(go.Bar(
                    y=size_counts.index, 
                    x=size_counts[sz], 
                    name=sz, 
                    orientation='h', 
                    marker_color=colors[sz]
                ))
        
        fig_sz.update_layout(
            barmode='stack', 
            title="Size Preferences (Ascending Order)", 
            height=400, 
            template="plotly_dark",
            xaxis_title="Total Quantity Sold"
        )
        st.plotly_chart(fig_sz, use_container_width=True)
    else:
        st.info("No size data available.")

    st.markdown("---")
    st.markdown("""
    **Authors:**
    * Hamza Tahboub
    * Majd Igbarea
    * Marysol Karwan
    * Igor Kornev
    """)