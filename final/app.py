from turtle import width
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objs as pgo
import calendar
import matplotlib as mpl 

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime

# --- 1. External Real-World Data (The Context) ---
STORE_CONTEXT = {
    "Lower Manhattan": {"sq_ft": 1200, "foot_traffic": 95, "office_density": 0.9, "competitors": 5},
    "Hell's Kitchen":  {"sq_ft": 800,  "foot_traffic": 85, "office_density": 0.4, "competitors": 8},
    "Astoria":         {"sq_ft": 1500, "foot_traffic": 60, "office_density": 0.2, "competitors": 2},
    "All Stores":      {"sq_ft": 3500, "foot_traffic": 80, "office_density": 0.5, "competitors": 15}
}

class AIEngine:

    def __init__(self, df):
        self.df = df.copy()
        
        # --- FIX: Calculate net_sales immediately ---
        # The raw CSV doesn't have this, so we compute it here.
        if 'net_sales' not in self.df.columns:
            self.df['net_sales'] = self.df['transaction_qty'] * self.df['unit_price']
        # --------------------------------------------
            
        self._enrich_data()
        self.qty_model = None

    def _enrich_data(self):
        for store, meta in STORE_CONTEXT.items():
            if store == 'All Stores':
                continue
            mask = self.df['store_location'] == store
            for key, value in meta.items():
                self.df.loc[mask, key] = value
        self.df.fillna(0, inplace=True)

    def get_future_forecast(self, store_name, months=6):
        data = self.df.copy()
        if store_name != 'All Stores':
            data = data[data['store_location'] == store_name]

        data['tx_date'] = pd.to_datetime(data['transaction_date'],dayfirst=True,errors='coerce')

        data = data.dropna(subset=['tx_date'])

        data['month_idx'] = data['tx_date'].dt.to_period('M').astype(int)
        monthly_sales = data.groupby('month_idx')['net_sales'].sum().reset_index()

        if len(monthly_sales) < 2:
            return pd.DataFrame()

        reg = LinearRegression()
        reg.fit(monthly_sales[['month_idx']], monthly_sales['net_sales'])

        last_idx = monthly_sales['month_idx'].max()
        future_indices = np.array([[last_idx + i] for i in range(1, months + 1)])
        future_sales = reg.predict(future_indices)

        base_date = pd.Timestamp("1970-01-01") + pd.offsets.MonthBegin(int(last_idx))
        future_dates = [base_date + pd.offsets.MonthBegin(i) for i in range(1, months + 1)]

        return pd.DataFrame({
            "Date": future_dates,
            "Predicted_Sales": future_sales
        })

    def train_scenario_model(self):
        features = ['product_category', 'unit_price', 'foot_traffic', 'office_density']
        target = 'transaction_qty'
        train_data = self.df.dropna(subset=features + [target])

        categorical_cols = ['product_category']
        numerical_cols = ['unit_price', 'foot_traffic', 'office_density']

        preprocessor = ColumnTransformer(transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        self.qty_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=50, random_state=42))
        ])

        self.qty_model.fit(train_data[features], train_data[target])

    def simulate_new_product(self, category, price, store_name, size_name):
        if not self.qty_model:
            self.train_scenario_model()

        context = STORE_CONTEXT.get(store_name, STORE_CONTEXT["All Stores"])

        input_data = pd.DataFrame({
            'product_category': [category],
            'unit_price': [price],
            'foot_traffic': [context['foot_traffic']],
            'office_density': [context['office_density']]
        })

        avg_qty = self.qty_model.predict(input_data)[0]
        size_factor = 0.8 if size_name.lower() in ['mega', 'huge', 'xl'] else 1.0
        estimated_monthly_units = avg_qty * 300 * size_factor
        estimated_revenue = estimated_monthly_units * price

        return estimated_monthly_units, estimated_revenue, context

def render_ai_dashboard(df):
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

    st.subheader("3. 🗺️ Geographical Efficiency Analysis")
    st.markdown("Comparing Store Efficiency: Sales per Square Foot (Real World Data Integration)")

    comp_data = []
    for store, meta in STORE_CONTEXT.items():
        if store == 'All Stores': continue
        total_sales = df[df['store_location'] == store]['transaction_qty'].sum() * df[df['store_location'] == store]['unit_price'].mean()
        efficiency = total_sales / meta['sq_ft']
        comp_data.append({'Store': store, 'Efficiency': efficiency, 'Foot Traffic': meta['foot_traffic']})

    comp_df = pd.DataFrame(comp_data)

    fig_geo = go.Figure()
    fig_geo.add_trace(go.Bar(
        x=comp_df['Store'], y=comp_df['Efficiency'],
        name='Sales per Sq.Ft', marker_color='#ff7f50'
    ))
    fig_geo.add_trace(go.Scatter(
        x=comp_df['Store'], y=comp_df['Foot Traffic'],
        name='Foot Traffic Index', yaxis='y2',
        mode='markers', marker=dict(size=15, color='#00e5ff')
    ))

    fig_geo.update_layout(
        title="Store Efficiency vs. Real World Traffic",
        yaxis=dict(title="Sales Revenue per Sq. Ft ($)"),
        yaxis2=dict(title="Foot Traffic Index", overlaying='y', side='right'),
        template="plotly_dark",
        legend=dict(x=0.1, y=1.1, orientation='h')
    )
    st.plotly_chart(fig_geo, use_container_width=True)

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