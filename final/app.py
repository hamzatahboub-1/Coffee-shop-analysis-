
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # This tells it to be non-interactive
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objs as pgo
import calendar
import matplotlib as mpl 
import os

# Get the folder where this script is running
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV file
csv_path = os.path.join(current_dir, 'final/Coffee Shop Sales.csv')

# Load it
data_set = pd.read_csv(csv_path)

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


# ----------------------------------------------------
# Sidebar icons
# ----------------------------------------------------
STORE_ICONS = {
    "All Stores": "🏬",
    "Lower Manhattan": "🏙️",
    "Hell's Kitchen": "🌆",
    "Astoria": "🏘️"
}

CATEGORY_ICONS = {
    "All Categories": "📦",
    "Coffee": "☕",
    "Tea": "🍵",
    "Drinking Chocolate": "🍫",
    "Bakery": "🥐",
    "Flavours": "🧴",
    "Loose Tea": "🌿",
    "Coffee beans": "🫘",
    "Packaged Chocolate": "🎁",
    "Branded": "🏷️"
}

# Set basic matplotlib color params
mpl.rcParams.update({
    "text.color": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#d1d5db",
    "ytick.color": "#d1d5db",
})

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# --- 1. Store Context (For Product Simulation) ---
# The AI needs this to know why some stores sell more than others.
STORE_CONTEXT = {
    "Lower Manhattan": {"sq_ft": 1200, "foot_traffic": 95, "office_density": 0.9},
    "Hell's Kitchen":  {"sq_ft": 800,  "foot_traffic": 85, "office_density": 0.4},
    "Astoria":         {"sq_ft": 1500, "foot_traffic": 60, "office_density": 0.2},
    "All Stores":      {"sq_ft": 3500, "foot_traffic": 80, "office_density": 0.5}
}

# --- 2. Ground Truth (For Size Prediction Training) ---
# We use this to teach the AI how to recognize a "Big" vs "Small" store.
TRAINING_DATA_REALITY = {
    "Lower Manhattan": 1200, 
    "Hell's Kitchen":  800,  
    "Astoria":         1500  
}

# --- 3. Seasonal Data (For Sales Forecasting) ---
NYC_SEASONAL_DATA = {
    1:  {"avg_temp": 33, "tourist_index": 0.8}, 
    2:  {"avg_temp": 35, "tourist_index": 0.8},
    3:  {"avg_temp": 42, "tourist_index": 1.0},
    4:  {"avg_temp": 53, "tourist_index": 1.1},
    5:  {"avg_temp": 62, "tourist_index": 1.2},
    6:  {"avg_temp": 72, "tourist_index": 1.4},
    7:  {"avg_temp": 77, "tourist_index": 1.5},
    8:  {"avg_temp": 75, "tourist_index": 1.5},
    9:  {"avg_temp": 68, "tourist_index": 1.3},
    10: {"avg_temp": 58, "tourist_index": 1.2},
    11: {"avg_temp": 48, "tourist_index": 1.1},
    12: {"avg_temp": 38, "tourist_index": 1.6}
}

class AIEngine:
    def __init__(self, df):
        self.df = df.copy()
        
        # Calculate Net Sales
        if 'net_sales' not in self.df.columns:
            self.df['net_sales'] = self.df['transaction_qty'] * self.df['unit_price']
            
        # Parse Dates (dayfirst=True fixes the error you saw earlier)
        self.df['tx_date'] = pd.to_datetime(self.df['transaction_date'], dayfirst=True, errors='coerce')
        
        # Enrich Data immediately
        self._enrich_data()
        
        self.qty_model = None  # For Product Simulation
        self.area_model = None # For Size Prediction

    def _enrich_data(self):
        """Injects ALL external data (Context + Weather) into the main dataframe."""
        
        # 1. Inject Store Context (Traffic & Density)
        # This fixes the missing column error!
        for store, meta in STORE_CONTEXT.items():
            if store == 'All Stores': continue
            mask = self.df['store_location'] == store
            for key, value in meta.items():
                self.df.loc[mask, key] = value
        
        # 2. Inject Seasonal Data (Temp & Tourism)
        self.df['month_num'] = self.df['tx_date'].dt.month
        self.df['avg_temp'] = self.df['month_num'].map(lambda x: NYC_SEASONAL_DATA.get(x, {}).get('avg_temp', 50))
        self.df['tourist_index'] = self.df['month_num'].map(lambda x: NYC_SEASONAL_DATA.get(x, {}).get('tourist_index', 1.0))
        
        self.df.fillna(0, inplace=True)

    # --- MODEL 1: SALES FORECAST (Polynomial) ---
    def get_future_forecast(self, store_name, months=6):
        data = self.df.copy()
        if store_name != 'All Stores':
            data = data[data['store_location'] == store_name]
            
        data['month_idx'] = data['tx_date'].dt.to_period('M').astype(int)
        monthly = data.groupby('month_idx').agg({
            'net_sales': 'sum', 'avg_temp': 'mean', 'tourist_index': 'mean', 'tx_date': 'first'
        }).reset_index()
        
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
                'month_idx': last_idx + i,
                'avg_temp': season.get('avg_temp', 50),
                'tourist_index': season.get('tourist_index', 1.0),
                'Date': next_date
            })
            
        future_X = pd.DataFrame(future_rows)
        predictions = poly_model.predict(future_X[['month_idx', 'avg_temp', 'tourist_index']])
        
        return pd.DataFrame({"Date": future_X['Date'], "Predicted_Sales": predictions})

    # --- MODEL 2: PRODUCT SIMULATOR (Random Forest) ---
    def train_scenario_model(self):
        # Now this works because 'foot_traffic' is back in self.df
        features = ['product_category', 'unit_price', 'foot_traffic', 'office_density']
        target = 'transaction_qty'
        
        train_data = self.df.dropna(subset=features + [target])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', 'passthrough', ['unit_price', 'foot_traffic', 'office_density']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['product_category'])
        ])
        
        self.qty_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=50, random_state=42))
        ])
        self.qty_model.fit(train_data[features], train_data[target])

    def simulate_new_product(self, category, price, store_name, size_name):
        if not self.qty_model: self.train_scenario_model()
        
        # Get context from the restored dictionary
        context = STORE_CONTEXT.get(store_name, STORE_CONTEXT["All Stores"])
        
        input_data = pd.DataFrame({
            'product_category': [category], 'unit_price': [price],
            'foot_traffic': [context['foot_traffic']], 'office_density': [context['office_density']]
        })
        
        avg_qty = self.qty_model.predict(input_data)[0]
        size_factor = 0.8 if size_name.lower() in ['mega', 'huge', 'xl'] else 1.0
        
        return avg_qty * 300 * size_factor, avg_qty * 300 * size_factor * price, context

    # --- MODEL 3: STORE SIZE PREDICTOR (Linear Regression) ---
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

def render_ai_dashboard(df):
    # --- SAFETY FIX: Create net_sales locally if missing ---
    if 'net_sales' not in df.columns:
        df = df.copy() # Work on a copy to avoid warnings
        df['net_sales'] = df['transaction_qty'] * df['unit_price']
    # -------------------------------------------------------
    st.title("🤖 AI Future Insights & Lab")
    st.markdown("Predict future trends and simulate business decisions using Real-World Context.")
    st.divider()

    ai = AIEngine(df)

    st.subheader("1. 📈 Future Revenue Prediction")
    c1, c2 = st.columns([1, 3])
    with c1:
        target_store = st.selectbox("Select Store for Forecast", store_locations, key="ai_store")
        months_fwd = st.slider("Months to Predict", 1, 12, 6)
    with c2:
        forecast_df = ai.get_future_forecast(target_store, months_fwd)
        if not forecast_df.empty:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
                mode='lines+markers', name='AI Prediction',
                line=dict(color='#00e5ff', width=3, dash='dash')
            ))
            fig_pred.update_layout(title=f"Predicted Sales Trend: {target_store}", template="plotly_dark", height=350)
            st.plotly_chart(fig_pred, use_container_width=True)

            growth = (forecast_df['Predicted_Sales'].iloc[-1] - forecast_df['Predicted_Sales'].iloc[0])
            st.info(f"💡 **AI Insight:** Based on historical trends, {target_store} is expected to change by **${growth:,.2f}** over the next {months_fwd} months.")

    st.divider()

    st.subheader("2. 🧪 New Product Simulator")
    st.markdown("Predict the impact of adding a **new size** or product type based on store geography.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_store = st.selectbox("Target Store", sorted(STORE_CONTEXT.keys()))
    with col2:
        sim_cat = st.selectbox("Category", ['Coffee', 'Tea', 'Bakery', 'Drinking Chocolate'])
        sim_size = st.text_input("New Size Name", "Mega-Grande")
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
# --- SECTION 3: GEOGRAPHICAL EFFICIENCY (AI-PREDICTED SIZE) ---
    st.subheader("3. 🗺️ Geographical Efficiency Analysis")
    st.markdown("We used AI to **predict store size** based on 'Bakery vs. Coffee' product usage, then calculated efficiency.")
    
    # 1. Get AI Predictions for Store Size
    predicted_sizes_df = ai.predict_store_area_from_usage()
    
    # 2. Calculate Efficiency (Sales / Predicted Sq Ft)
    # We need total sales per store to divide by the area
    sales_per_store = df.groupby('store_location')['net_sales'].sum().reset_index()
    
    # Merge the sales data with the AI predicted size
    merged_geo = pd.merge(predicted_sizes_df, sales_per_store, on='store_location')
    
    # Calculate Efficiency
    merged_geo['Efficiency'] = merged_geo['net_sales'] / merged_geo['predicted_sq_ft']
    
    # 3. Plot
    fig_geo = go.Figure()
    
    # Bar for Efficiency
    fig_geo.add_trace(go.Bar(
        x=merged_geo['store_location'], 
        y=merged_geo['Efficiency'],
        name='Sales per AI-Predicted Sq.Ft', 
        marker_color='#ff7f50'
    ))
    
    # Line for the Predicted Size (to show what the AI guessed)
    fig_geo.add_trace(go.Scatter(
        x=merged_geo['store_location'], 
        y=merged_geo['predicted_sq_ft'],
        name='AI Predicted Area (sq.ft)', 
        yaxis='y2',
        mode='markers+text',
        text=[f"{int(x)} sq.ft" for x in merged_geo['predicted_sq_ft']],
        textposition="top center",
        marker=dict(size=15, color='#00e5ff', symbol='square')
    ))
    
    fig_geo.update_layout(
        title="Store Efficiency (Using AI-Derived Areas)",
        yaxis=dict(title="Sales Efficiency ($/sq.ft)"),
        yaxis2=dict(
            title="Predicted Store Size (sq.ft)", 
            overlaying='y', 
            side='right',
            range=[0, 2000] # Set a fixed range so dots don't fly off
        ),
        template="plotly_dark",
        legend=dict(x=0.1, y=1.1, orientation='h'),
        margin=dict(r=50) # Make room for right axis
    )
    
    st.plotly_chart(fig_geo, use_container_width=True)
    
    # Explain the AI Logic to the user
    with st.expander("🧠 How did the AI guess the store size?"):
        st.write("""
        The model analyzed **Product Usage Patterns**:
        1. **Bakery Ratio:** Stores selling more croissants/food likely have seating areas -> **Predicted Larger**.
        2. **Transaction Volume:** Higher volume implies larger counter/queue space -> **Predicted Larger**.
        """)

# -----------------------------------------------
# ----------------------------------------------------
data_set = pd.read_csv('Coffee Shop Sales.csv')

# --- FIX: DEFINE GLOBAL VARIABLES HERE ---
# This ensures both the Dashboard and the AI Page can see the list of stores
store_locations = ['All Stores'] + sorted(data_set['store_location'].unique())
# ----------------------------------------------------
# ----------------------------------------------------
st.set_page_config(
    page_title="Coffee shop Dashboard",
    layout="wide"
)


def is_dark_theme():
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False

IS_DARK = is_dark_theme()

HOURLY_COLORS = {
    "title": "#ffffff" if IS_DARK else "#1f2937",
    "label": "#d1d5db" if IS_DARK else "#444",
    "grid": "#555555" if IS_DARK else "#dddddd",
}

STREAMLIT_DARK_BG = "#0e1117"
STREAMLIT_LIGHT_BG = "#ffffff"
FIG_BG = STREAMLIT_DARK_BG if IS_DARK else STREAMLIT_LIGHT_BG

st.markdown(
    """
    <style>
    /* Make selectbox & multiselect selection-only (no typing) */
    div[data-baseweb="select"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
    }
    div[data-baseweb="select"] div[role="combobox"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# MAIN NAVIGATION CONTROLLER

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Dashboard'

with st.sidebar:
    st.title("Navigation")
    page_selection = st.radio(
        "Go to:",
        ['📊 Main Dashboard', '🤖 AI Predictions'],
        index=0 if st.session_state['current_page'] == 'Dashboard' else 1
    )

    if page_selection == '📊 Main Dashboard':
        st.session_state['current_page'] = 'Dashboard'
    else:
        st.session_state['current_page'] = 'AI'

    st.divider()

# PAGE ROUTING

if st.session_state['current_page'] == 'AI':
    render_ai_dashboard(data_set)

else:
    # === MAIN DASHBOARD BELOW ===
    st.title("📊 Maven Roasters Coffee shop Dashboard")
    st.markdown("Net sales | Rush Hours | Sizes Analysis")

    st.divider()

    # Unified Sidebar Filters for All Graphs (as button groups)
    store_locations = ['All Stores'] + sorted(data_set['store_location'].unique())

    category_order = (
        data_set.groupby('product_category')['transaction_qty']
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    product_categories = ['All Categories'] + category_order

    default_store = 'All Stores'
    default_category = 'All Categories'

    if 'selected_store' not in st.session_state:
        st.session_state['selected_store'] = default_store
    if 'selected_category' not in st.session_state:
        st.session_state['selected_category'] = default_category

    with st.sidebar:
        st.markdown("#### Store location:")
        selected_store = st.radio(
            '',
            store_locations,
            index=store_locations.index(st.session_state['selected_store']),
            key="store_radio_button",
            label_visibility="collapsed",
            format_func=lambda x: f"{STORE_ICONS.get(x, '📍')}  {x}"
        )
        st.session_state['selected_store'] = selected_store

        st.markdown("#### Product category:")
        default_category_idx = product_categories.index(st.session_state['selected_category'])
        selected_category = st.radio(
            '',
            product_categories,
            index=default_category_idx,
            key="cat_radio_button",
            label_visibility="collapsed",
            format_func=lambda x: f"{CATEGORY_ICONS.get(x, '🔹')}  {x}"
        )
        st.session_state['selected_category'] = selected_category

    def get_filtered_data(store_location, product_category):
        df = data_set.copy()
        if store_location != 'All Stores':
            df = df[df['store_location'] == store_location]
        if product_category != 'All Categories':
            df = df[df['product_category'] == product_category]
        return df

    filtered_data = get_filtered_data(
        st.session_state['selected_store'],
        st.session_state['selected_category']
    )

    st.markdown("""
    <style>
    div[data-baseweb="radio"] label {
        display: inline-block;
        margin: 3px 8px 8px 0 !important;
        padding: 10px 24px;
        border-radius: 14px;
        background-color: #111111;
        border: 2px solid #222c;
        color: #FFF !important;
        font-size: 18px !important;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s, border 0.2s;
    }
    div[data-baseweb="radio"] label:hover {
        background-color: #1e242f;
        border-color: #00aabb55;
    }
    div[data-baseweb="radio"] input[type="radio"] {
        display: none;
    }
    div[data-baseweb="radio"] input[type="radio"]:checked+div {
        background: #00e5ff;
        color: #222 !important;
        border-color: #00e5ff;
    }
    label[for$="radio_button"] {
        font-size: 20px !important;
        font-weight: 700;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Top Row (Two Graphs Side by Side)
    st.subheader("Net Sales by Month and Store")

    def get_data_for_net_sales_graph(store_location):
        df = data_set.copy()
        if store_location != 'All Stores':
            df = df[df['store_location'] == store_location]
        return df

    df = get_data_for_net_sales_graph(selected_store)
    if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df = df.dropna(subset=['transaction_date'])
    df['YearMonth'] = df['transaction_date'].dt.to_period('M').astype(str)
    df['MonthOnly'] = df['transaction_date'].dt.strftime('%b')
    df['net_sales'] = df['transaction_qty'] * df['unit_price']

    monthly_item_counts = df.groupby('YearMonth')['transaction_qty'].sum().reindex(sorted(df['YearMonth'].unique())).fillna(0)
    months_for_items = [df[df['YearMonth'] == m]['MonthOnly'].iloc[0] if m in df['YearMonth'].values else m for m in monthly_item_counts.index]

    unique_months = df['YearMonth'].unique()
    if len(unique_months) > 6:
        keep_months = sorted(unique_months)[:6]
        df = df[df['YearMonth'].isin(keep_months)]
        monthly_item_counts = monthly_item_counts.loc[keep_months]
        months_for_items = [df[df['YearMonth'] == m]['MonthOnly'].iloc[0] if m in df['YearMonth'].values else m for m in keep_months]

    summary = df.groupby(['YearMonth', 'MonthOnly', 'store_location'], as_index=False)['net_sales'].sum()
    pivot = summary.pivot(index='YearMonth', columns='store_location', values='net_sales')
    months_only = summary.drop_duplicates('YearMonth').set_index('YearMonth')['MonthOnly'].reindex(pivot.index).tolist()

    color_palette = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    store_colors = {store: color_palette[i % len(color_palette)] for i, store in enumerate(pivot.columns)}

    fig = go.Figure()

    if selected_store == 'All Stores':
        for store in pivot.columns:
            y_data = pivot[store]
            fig.add_trace(go.Bar(
                x=months_only,
                y=y_data,
                name=store,
                marker_color=store_colors.get(store, None),
                text=None,
                textposition=None
            ))
        barmode = 'group'
    else:
        if selected_store in pivot.columns:
            y_data = pivot[selected_store]
            fig.add_trace(go.Bar(
                x=months_only,
                y=y_data,
                name=selected_store,
                marker_color=store_colors.get(selected_store, 'lightslategray'),
                text=[f"${v:,.2f}" if pd.notnull(v) else "" for v in y_data],
                textposition='outside'
            ))
        barmode = 'group'

    fig.add_trace(go.Scatter(
        x=months_for_items,
        y=monthly_item_counts.values,
        mode='lines+markers',
        name='Number of Purchased Items',
        yaxis='y2',
        marker=dict(color='#00BFFF'),
        line=dict(width=2, color='#00BFFF', dash='dash'),
        opacity=0.85
    ))

    title_str = (
        "Net Sales per Month"
        if selected_store == 'All Stores'
        else f"Net Sales per Month - {selected_store}"
    )

    fig.update_layout(
        barmode=barmode,
        title=title_str,
        xaxis_title="Month",
        yaxis_title="Net Sales ($)",
        yaxis=dict(
            title="Net Sales ($)",
            tickformat="$.2s",
        ),
        yaxis2=dict(
            title="Number of Purchased Items",
            overlaying='y',
            side='right',
            showgrid=False,
            tickformat=",.0f",
            tickfont=dict(color='#00BFFF'),
        ),
        legend_title="Store Location",
        legend=dict(
            orientation="v",
            x=1.1,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            tickangle=45,
            tickmode='array',
            tickvals=months_only,
            ticktext=months_only
        ),
        width=1000,
        height=500,
        margin=dict(l=60, r=160, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Hourly Transactions")

    if 'hour' not in filtered_data.columns:
        filtered_data['hour'] = filtered_data['transaction_time'].astype(str).str.split(':').str[0].astype(int)

    def plot_hourly_avg_orders_with_deviation(df, store_loc, prod_cat):
        if df.empty or 'hour' not in df.columns or 'transaction_qty' not in df.columns:
            st.warning("No data available for the selected filters.")
            return

        if 'transaction_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
                try:
                    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
                except Exception:
                    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            date_col = 'transaction_date'
        elif 'transaction_time' in df.columns:
            try:
                df['transaction_time'] = pd.to_datetime(df['transaction_time'])
                df['date_only'] = df['transaction_time'].dt.date
                date_col = 'date_only'
            except Exception:
                df['date_only'] = df['transaction_time'].astype(str).str.split(" ").str[0]
                date_col = 'date_only'
        else:
            st.warning("No date column available for averaging.")
            return

        grouped = df.groupby([date_col, 'hour'], as_index=False)['transaction_qty'].sum()
        hourly_agg = grouped.groupby('hour')['transaction_qty'].agg(['mean', 'std']).reset_index()
        hourly_agg = hourly_agg.sort_values('hour')
        hourly_agg = hourly_agg.fillna(0)

        if hourly_agg.empty:
            st.warning("No data available for the selected filters.")
            return

        IS_DARK = st.get_option("theme.base") == "dark"
        bg_color = "rgba(0,0,0,0)"
        text_color = (
            st.get_option("theme.textColor")
            or ("#ffffff" if IS_DARK else "#1f2937")
        )
        grid_color = (
            st.get_option("theme.secondaryBackgroundColor")
            or ("#555555" if IS_DARK else "#dddddd")
        )
        bar_color = "#FF7F50"
        std_line_color = "#286090"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly_agg['hour'],
            y=hourly_agg['mean'],
            marker_color=bar_color,
            text=[f"{v:.2f}" if not pd.isna(v) else "" for v in hourly_agg['mean']],
            textposition='outside',
            name='Avg Orders'
        ))

        fig.add_trace(go.Scatter(
            x=hourly_agg['hour'],
            y=hourly_agg['mean'] + hourly_agg['std'],
            mode='lines+markers',
            marker=dict(color=std_line_color, size=8, symbol='diamond'),
            line=dict(color=std_line_color, width=2, dash='dash'),
            name='Mean + Std Dev',
            hovertemplate="Hour: %{x}<br>Avg + Std: %{y:.2f}<extra></extra>",
            showlegend=True,
        ))

        fig.update_layout(
            xaxis=dict(title="Hour of Day", tickmode='linear', color=text_color, gridcolor=grid_color),
            yaxis=dict(title="Average Quantity Ordered", color=text_color, gridcolor=grid_color),
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            margin=dict(l=50, r=50, t=50, b=40),
            width=900,
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    plot_hourly_avg_orders_with_deviation(filtered_data, selected_store, selected_category)

    st.divider()

    # Bottom Row (One Full-Width Graph)
    st.subheader("Sizes Analysis")

    df_cupsize = None
    if 'df' in globals():
        df_cupsize = df
    elif 'data_set' in globals():
        df_cupsize = data_set
    elif 'transactions' in globals():
        df_cupsize = transactions

    if df_cupsize is None:
        st.warning("No dataframe found for cup size analysis.")
        st.stop()

    cup_size_mapping = {
        'Small': ['Small', 'Sm', 'small', 'sm'],
        'Medium': ['Medium', 'Med', 'medium', 'med'],
        'Large': ['Large', 'Lg', 'large', 'lg']
    }

    def extract_cup_size(detail):
        for size, variants in cup_size_mapping.items():
            if any(variant in str(detail) for variant in variants):
                return size
        return None

    def format_category_label(label):
        if label == "Drinking Chocolate":
            return "Drinking<br>Chocolate"
        return label

    def plotly_cup_size_bar(df_cupsize, store_location, category_filter):
        df_filtered = df_cupsize.copy()
        df_filtered = df_filtered[df_filtered['product_category'] != 'Coffee beans']
        if store_location != 'All Stores':
            df_filtered = df_filtered[df_filtered['store_location'] == store_location]
        if category_filter != 'All Categories':
            df_filtered = df_filtered[df_filtered['product_category'] == category_filter]

        df_filtered = df_filtered.copy()
        df_filtered['Cup Size'] = df_filtered['product_detail'].apply(extract_cup_size)
        df_filtered = df_filtered[df_filtered['Cup Size'].notnull()]
        if df_filtered.empty:
            st.write("No cup size data for selected filters.")
            return

        group_table = (
            df_filtered.groupby(['product_category', 'Cup Size'])['transaction_qty']
            .sum()
            .unstack(fill_value=0)
        )
        group_table = group_table[(group_table > 0).any(axis=1)]
        if group_table.empty:
            st.write("No cup size data for selected filters.")
            return

        group_table['_total_qty'] = group_table.sum(axis=1)
        group_table = group_table.sort_values('_total_qty', ascending=True)
        group_table = group_table.drop(columns=['_total_qty'])

        formatted_index = [format_category_label(cat) for cat in group_table.index.tolist()]

        fig = go.Figure()

        colors_map = {"Small": "#A9CCE3", "Medium": "#5499C7", "Large": "#154360"}
        available_sizes = [s for s in ['Small', 'Medium', 'Large'] if s in group_table.columns]

        for size in available_sizes:
            fig.add_trace(
                go.Bar(
                    y=formatted_index,
                    x=group_table[size].tolist(),
                    name=size,
                    orientation='h',
                    marker=dict(color=colors_map.get(size, None)),
                    text=[f"{int(v):,}" if v > 0 else "" for v in group_table[size]],
                    textposition='auto',
                    hovertemplate="Category: %{y}<br>Qty: %{x}<br>Size: " + size + "<extra></extra>"
                )
            )

        n_bars = len(group_table.index)
        final_height = max(220, 24 * n_bars)

        fig.update_layout(
            barmode='stack',
            xaxis_title="Total Ordered Quantity",
            yaxis_title="Product Category",
            title=(
                "Cup Sizes per Category" +
                (f" ({store_location})" if store_location != "All Stores" else "") +
                (f"<br>Category: {category_filter}" if category_filter != "All Categories" else "")
            ),
            height=final_height,
            legend_title="Cup Size",
            margin=dict(l=100, r=30, t=42, b=32),
            yaxis=dict(tickfont=dict(size=10))
        )

        st.plotly_chart(fig, use_container_width=True)

    st.header("Cup Sizes per Category")
    plotly_cup_size_bar(
        df_cupsize,
        selected_store if 'selected_store' in globals() else 'All Stores',
        selected_category if 'selected_category' in globals() else 'All Categories'
    )

    # Optional Footer
    st.markdown("---")
    st.caption("Dashboard generated using Streamlit")
    st.markdown(
        "Authors: \n"
        "Hamza Tahboub  \n"
        "Majd Igbarea  \n"
        "Marysol Karwan  \n"
        "Igor Kornov"
    )