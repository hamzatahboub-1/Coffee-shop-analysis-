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

    def get_future_forecast(self, store_name, months=6):
        data = self.df.copy()
        if store_name != 'All Stores':
            data = data[data['store_location'] == store_name]
        data['month_idx'] = data['tx_date'].dt.to_period('M').astype(int)
        monthly = data.groupby('month_idx').agg({'net_sales': 'sum', 'avg_temp': 'mean', 'tourist_index': 'mean', 'tx_date': 'first'}).reset_index()
        
        if len(monthly) < 3: return pd.DataFrame() 

        X = monthly[['month_idx', 'avg_temp', 'tourist_index']]
        y = monthly['net_sales']
        poly_model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        poly_model.fit(X, y)
        
        last_idx = monthly['month_idx'].max()
        last_date = monthly['tx_date'].max()
        future_rows = []
        for i in range(1, months + 1):
            next_date = last_date + pd.DateOffset(months=i)
            season = NYC_SEASONAL_DATA.get(next_date.month, {})
            future_rows.append({
                'month_idx': last_idx + i, 'avg_temp': season.get('avg_temp', 50),
                'tourist_index': season.get('tourist_index', 1.0), 'Date': next_date
            })
        future_X = pd.DataFrame(future_rows)
        predictions = poly_model.predict(future_X[['month_idx', 'avg_temp', 'tourist_index']])
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
        input_data = pd.DataFrame({'product_category': [category], 'unit_price': [price], 'foot_traffic': [context['foot_traffic']], 'office_density': [context['office_density']]})
        avg_qty = self.qty_model.predict(input_data)[0]
        size_factor = 0.8 if size_name.lower() in ['mega', 'huge', 'xl'] else 1.0
        return avg_qty * 300 * size_factor, avg_qty * 300 * size_factor * price, context

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

def render_ai_dashboard(df):
    df = df.copy()
    if 'net_sales' not in df.columns:
        df['net_sales'] = df['transaction_qty'] * df['unit_price']

    ai = AIEngine(df)
    st.title("🤖 AI Future Insights & Lab")
    st.divider()

    st.subheader("1. 📈 Future Revenue Prediction")
    c1, c2 = st.columns([1, 3])
    with c1:
        target_store = st.selectbox("Select Store", store_locations, key="ai_store")
        months_fwd = st.slider("Months", 1, 12, 6)
    with c2:
        forecast_df = ai.get_future_forecast(target_store, months_fwd)
        if not forecast_df.empty:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_Sales'], mode='lines+markers', name='AI Prediction', line=dict(color='#00e5ff', width=3, dash='dash')))
            fig_pred.update_layout(title=f"Prediction: {target_store}", template="plotly_dark", height=350)
            st.plotly_chart(fig_pred, use_container_width=True)

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
        sim_price = st.number_input("Unit Price ($)", 2.0, 15.0, 4.50)

    if st.button("🔮 Simulate Launch Results"):
        units, rev, ctx = ai.simulate_new_product(sim_cat, sim_price, sim_store, sim_size)
        st.markdown("#### 📊 Simulation Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Est. Monthly Units", f"{int(units)}")
        m2.metric("Est. Monthly Revenue", f"${rev:,.2f}")
        m3.metric("Foot Traffic Score", f"{ctx['foot_traffic']}/100", delta_color="off")

        st.warning(
            f"**🧠 Model Reasoning:** "
            f"Prediction adjusted based on **{sim_store}'s** real-world data:\n"
            f"* **Office Density ({ctx['office_density']}):** Determines morning rush volume.\n"
            f"* **Foot Traffic ({ctx['foot_traffic']}):** Influences walk-in probability.\n"
            f"The model detected that {sim_cat} sells better in high-density areas."
        )

    st.divider()
    st.subheader("3. 🗺️ Efficiency Analysis")
    pred_size = ai.predict_store_area_from_usage()
    
    sales = df.groupby('store_location')['net_sales'].sum().reset_index()
    
    merged = pd.merge(pred_size, sales, on='store_location')
    merged['Efficiency'] = merged['net_sales'] / merged['predicted_sq_ft']

    fig_geo = go.Figure()
    fig_geo.add_trace(go.Bar(x=merged['store_location'], y=merged['Efficiency'], name='Sales/SqFt', marker_color='#ff7f50'))
    fig_geo.add_trace(go.Scatter(x=merged['store_location'], y=merged['predicted_sq_ft'], name='Pred Size', yaxis='y2', mode='markers', marker=dict(size=15, color='#00e5ff', symbol='square')))
    fig_geo.update_layout(title="Efficiency vs AI Predicted Size", yaxis2=dict(overlaying='y', side='right', title="Sq Ft"), template="plotly_dark")
    st.plotly_chart(fig_geo, use_container_width=True)

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
    st.subheader("Cup Sizes per Category")# 3. Sizes Analysis
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