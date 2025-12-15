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
# apply_glowing_styles()



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

# ----------------------------------------------------
data_set = pd.read_csv('Coffee Shop Sales.csv')
# ----------------------------------------------------
st.set_page_config(
    page_title="Maven Roasters Coffee Dashboard shop ",
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

STREAMLIT_DARK_BG = "#0e1117"   # official Streamlit dark background
STREAMLIT_LIGHT_BG = "#ffffff"

FIG_BG = STREAMLIT_DARK_BG if IS_DARK else STREAMLIT_LIGHT_BG

# ----------------------------------------------------
# Insert CSS to make selectbox & multiselect selection-only (no typing)
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
# ----------------------------------------------------
# Dashboard Title
# ----------------------------------------------------
st.title("📊 Maven Roasters Coffee shop Dashboard")
st.markdown("Net sales | Rush Hours |Trendy categories")

st.divider()

# ----------------------------------------------------
# Unified Sidebar Filters for All Graphs (Filters as button groups instead of selectboxes)
# ----------------------------------------------------
store_locations = ['All Stores'] + sorted(data_set['store_location'].unique())

# --- Sort product categories list by most-ordered descending ---
category_order = (
    data_set.groupby('product_category')['transaction_qty']
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)
product_categories = ['All Categories'] + category_order

default_store = 'All Stores'
default_category = 'All Categories'

import streamlit as st

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

# Remove any sidebar glowing/radio glow effect, just set simple style
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
div[data-baseweb="radio"] input[type="radio"]:not(:checked)+div {
}
label[for$="radio_button"] {
    font-size: 20px !important;
    font-weight: 700;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Top Row (Two Graphs Side by Side)
# ----------------------------------------------------

st.subheader("Net Sales by Month and Store")

# For Graph 1: Do NOT apply product category filter (always use all categories)
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

unique_months = df['YearMonth'].unique()
if len(unique_months) > 6:
    keep_months = sorted(unique_months)[:6]
    df = df[df['YearMonth'].isin(keep_months)]

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
    # Grouped bars for all stores, no labels (for clarity)
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
    # Single store: show bar labels
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
    legend_title="Store Location",
    legend=dict(
        orientation="v",
        x=1.02,
        y=1,
        bgcolor='rgba(0,0,0,0)', # transparent legend background
        bordercolor='rgba(0,0,0,0)', # set legend frame color to transparent
        borderwidth=1,
        font=dict(size=11)
    ),
    xaxis=dict(
        tickangle=45,
        tickmode='array',
        tickvals=months_only,
        ticktext=months_only
    ),
    width=900,
    height=500,
    margin=dict(l=60, r=60, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)
# --- Normal text title for Hourly Transactions ---
st.subheader("Hourly Transactions")

if 'hour' not in filtered_data.columns:
    filtered_data['hour'] = filtered_data['transaction_time'].astype(str).str.split(':').str[0].astype(int)

import plotly.graph_objects as go

def plot_hourly_transactions_plotly(df, store_loc, prod_cat):
    if df.empty or 'hour' not in df.columns or 'transaction_qty' not in df.columns:
        st.warning("No data available for the selected filters.")
        return

    hourly_summary = df.groupby('hour', as_index=False)['transaction_qty'].sum().sort_values('hour')

    if hourly_summary.empty:
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
    bar_color = "#FF7F50"  # bright coral-like color

    fig = go.Figure(go.Bar(
        x=hourly_summary['hour'],
        y=hourly_summary['transaction_qty'],
        marker_color=bar_color,
        text=hourly_summary['transaction_qty'],
        textposition='outside'
    ))

    fig.update_layout(
        xaxis=dict(title="Hour of Day", tickmode='linear', color=text_color, gridcolor=grid_color),
        yaxis=dict(title="Total Quantity Ordered", color=text_color, gridcolor=grid_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=50, r=50, t=50, b=40),
        width=900,
        height=450
    )

    # Removed legend glow

    st.plotly_chart(fig, use_container_width=True)

# Usage
plot_hourly_transactions_plotly(filtered_data, selected_store, selected_category)

st.divider()

# ----------------------------------------------------
# Bottom Row (One Full-Width Graph)
# ----------------------------------------------------
st.subheader("Monthly Orders per Product Category")

category_colors = {
    'Coffee':'#4C6058',
    'Tea':'#7A4419',
    'Drinking Chocolate':'#924C9D',
    'Bakery':'#E3AC4E',
    'Flavours':'#D94C4C',
    'Loose Tea':'#32A69B',
    'Coffee beans':'#86624E',
    'Packaged Chocolate':'#8376B6',
    'Branded':'#E37C96'
}

def get_monthly_category_summary(df, store_location, product_category):
    df_filtered = df.copy()
    if df_filtered.empty:
        return pd.DataFrame()
    if not pd.api.types.is_string_dtype(df_filtered['transaction_date']):
        df_filtered['transaction_date'] = df_filtered['transaction_date'].astype(str)
    if df_filtered['transaction_date'].str.contains('/').any():
        date_format = '%d/%m/%Y'
    elif df_filtered['transaction_date'].str.contains('-').any():
        date_format = '%Y-%m-%d'
    else:
        date_format = None

    try:
        df_filtered['transaction_date'] = pd.to_datetime(df_filtered['transaction_date'], format=date_format)
    except Exception:
        df_filtered['transaction_date'] = pd.to_datetime(df_filtered['transaction_date'], errors='coerce')

    df_filtered['month'] = df_filtered['transaction_date'].dt.to_period('M').astype(str)

    if product_category == 'All Categories':
        summary = (
            df_filtered
            .groupby(['product_category', 'month'], as_index=False)['transaction_qty']
            .sum()
            .rename(columns={'transaction_qty': 'Total Ordered Quantity'})
        )
        cat_totals = summary.groupby('product_category')['Total Ordered Quantity'].sum().sort_values(ascending=False).head(7)
        top_cats = cat_totals.index.tolist()
        summary = summary[summary['product_category'].isin(top_cats)]
        group_col = 'product_category'
    else:
        summary = (
            df_filtered.groupby(['month'], as_index=False)['transaction_qty']
            .sum()
            .rename(columns={'transaction_qty': 'Total Ordered Quantity'})
        )
        summary['product_category'] = product_category
        group_col = 'product_category'

    months = sorted(summary['month'].unique())
    all_groups = summary[group_col].unique()
    complete_idx = pd.MultiIndex.from_product([all_groups, months], names=[group_col, 'month'])
    summary = summary.set_index([group_col, 'month']).reindex(complete_idx, fill_value=0).reset_index()
    summary = summary.sort_values([group_col, 'month'])
    return summary

def _remove_2023_from_title(title):
    import re
    new_title = re.sub(r'(2023[/-]?)', '', title)
    new_title = re.sub(r'\(*2023\)*', '', new_title)
    new_title = re.sub(r'[,\s]*2023[,\s]*', '', new_title)
    new_title = re.sub(r'\(\s*,\s*\)', '', new_title)
    return new_title.strip()

def st_plot_category_by_month(df, store_location, product_category):
    summary = get_monthly_category_summary(df, store_location, product_category)
    if summary.empty:
        st.warning("No data available for the selected filters.")
        return

    if product_category == 'All Categories':
        pivot = summary.pivot(index='month', columns='product_category', values='Total Ordered Quantity').fillna(0)
        cat_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
        pivot = pivot[cat_order]
        columns_to_plot = list(pivot.columns)
        legend_title = "Product Category"
    else:
        pivot = summary.pivot(index='month', columns='product_category', values='Total Ordered Quantity').fillna(0)
        columns_to_plot = [product_category]
        legend_title = "Product Category"

    months = list(pivot.index)

    def format_month_label(month):
        if isinstance(month, str) and (month.startswith('2023-') or month.startswith('2023/')):
            return month[5:]
        return month

    months_no_year = [format_month_label(m) for m in months]

    def get_month_abbr(m):
        if isinstance(m, str):
            if '-' in m:
                part = m.split('-')[-1]
            elif '/' in m:
                part = m.split('/')[-1]
            else:
                part = m
            try:
                m_int = int(part)
                return calendar.month_abbr[m_int]
            except Exception:
                return m
        return str(m)

    months_abbr = [get_month_abbr(m) for m in months]

    fig = go.Figure()
    for i, col in enumerate(columns_to_plot):
        color = category_colors.get(col, f'rgba(60,60,{(i*40)%255},0.97)')
        fig.add_trace(go.Scatter(
            x=months_abbr,
            y=pivot[col],
            mode='lines+markers',
            name=col,
            line=dict(width=2, color=color),
            marker=dict(size=8, color=color),
            hovertemplate='%{x}<br>%{y:,d}<br>%{fullData.name}<extra></extra>',
            opacity=0.7
        ))

    axis_ticktext = months_abbr
    raw_title = f"Monthly Orders per Product Category<br><sup>({store_location}, {product_category})</sup>"
    filtered_title = _remove_2023_from_title(raw_title)
    fig.update_layout(
        width=850, height=420,
        title=filtered_title,
        xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=months_abbr,
            ticktext=axis_ticktext
        ),
        yaxis=dict(title='Total Ordered Quantity', tickformat=',d'),
        legend=dict(title=legend_title, x=1.03, y=1, font=dict(size=10), traceorder="normal"),
        margin=dict(l=60, r=160, t=70, b=50),
        hovermode='x unified',
    )

    # Removed legend glow

    st.plotly_chart(fig, use_container_width=True)

st.header("")
st_plot_category_by_month(filtered_data, selected_store, selected_category)

# ----------------------------------------------------
# Optional Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("Dashboard generated using Streamlit")
st.markdown(
    "Authors  \n"
    "Hamza Tahboub  \n"
    "Majd igbarea  \n"
    "Marysol karwan  \n"
    "Igor Kornov"
)