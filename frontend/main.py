# frontend/main.py - ENHANCED VERSION WITH FORECAST DRIVERS
# ----------------
# Streamlit frontend for AI Demand Forecasting

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import io
from datetime import datetime

# ---- PAGE STATE ----
if "show_app" not in st.session_state:
    st.session_state.show_app = False

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"

# Country mapping
COUNTRY_MAP = {
    "India": "IN",
    "United States": "US",
    "United Kingdom": "UK"
}

st.set_page_config(
    page_title="Expedition Co. Demand Forecasting",
    layout="wide",
    page_icon="📈"
)

st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

# --- ENHANCED CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f9;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0.5rem;
    }
    
    .card-box {
        background: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 15px;
    }
    
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
    }
    
    h1 {
        color: #1a237e !important;
        font-weight: 700;
    }

    .insight-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #e0f2f1 100%);
        border-left: 4px solid #2196f3;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
            
    .insight-title {
        color: #1565c0;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    
    .insight-text {
        color: #0d47a1;
        font-size: 1.05em;
        line-height: 1.6;
    }
    
    /* ===== FORECAST DRIVERS SECTION ===== */
    .forecast-drivers-container {
        background: white;
        padding: 26px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-top: 28px;
        margin-bottom: 28px;
    }
    
    .drivers-header {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 10px;
    }
    
    .drivers-subtitle {
        font-size: 0.96rem;
        color: #666;
        margin-bottom: 22px;
    }
    
    .driver-card {
        background: #fafafa;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .driver-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transform: translateX(2px);
    }
    
    .driver-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .festival-awareness-card {
        background: #fafafa;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #9c27b0;
    }
            
    .festival-awareness-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 4px;
    }
            
    .festival-awareness-date {
        font-size: 0.85rem;
        color: #666;
    }
            
    .indicator-positive {
        background: #4caf50;
        box-shadow: 0 0 8px rgba(76, 175, 80, 0.4);
    }
    
    .indicator-negative {
        background: #f44336;
        box-shadow: 0 0 8px rgba(244, 67, 54, 0.4);
    }
    
    .indicator-neutral {
        background: #ff9800;
        box-shadow: 0 0 8px rgba(255, 152, 0, 0.4);
    }
    
    .indicator-none {
        background: #9e9e9e;
    }
    
    .driver-content {
        flex: 1;
    }
    
    .driver-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 2px;
    }
    
    .driver-description {
        font-size: 0.85rem;
        color: #666;
        line-height: 1.4;
    }
    
    .seasonal-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border-left: 4px solid #9c27b0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }
    
    .seasonal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .seasonal-name {
        font-size: 0.98rem;
        font-weight: 600;
        color: #1a237e;
        flex: 1;
    }
    
    .seasonal-impact {
        font-size: 0.92rem;
        font-weight: 700;
        color: #7b1fa2;
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 4px 12px;
        border-radius: 14px;
        white-space: nowrap;
        margin-left: 12px;
    }

    
    .seasonal-period {
        font-size: 0.86rem;
        color: #666;
        display: flex;
        align-items: center;
        gap: 6px;
    }
            
    .seasonal-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateX(2px);
    }
    
    .no-factors-message {
        text-align: center;
        padding: 32px 24px;
        color: #999;
        font-size: 0.95rem;
        background: #fafafa;
        border-radius: 10px;
        border: 1px dashed #ddd;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a237e;
        margin: 0;
    }
    
    .section-subtitle {
        font-size: 0.95rem;
        color: #666;
        margin-top: 6px;
        margin-left: 48px;
    }
            
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        background: white;
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 20px;
        margin-top: 24px;
    }
    
    .section-icon {
        font-size: 1.8rem;
        line-height: 1;
    }
            
    .column-title {
        font-size: 1.12rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 18px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stButton>button {
        background-color: #2563eb !important;   
        color: #ffffff !important;              
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #1e40af !important;  
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        cursor: pointer;
    }

    .stButton>button:active {
        background-color: #1e3a8a !important; 
        transform: translateY(1px);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.25);
    }
            
    div[data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
        cursor: pointer;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #1e40af !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

def find_col(keywords: list, columns: list) -> str:
    """Auto-detect column based on keywords."""
    for col in columns:
        if any(keyword in col.lower() for keyword in keywords):
            return col
    return columns[0] if columns else None


def create_forecast_chart(history_data: list, forecast_data: list, category: str):
    """Generate Plotly chart with historical data, forecast, and confidence interval."""
    history_df = pd.DataFrame(history_data)
    forecast_df = pd.DataFrame(forecast_data)
    
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    
    fig = go.Figure()
    
    # Confidence Interval
    if "Upper_Bound" in forecast_df.columns and "Lower_Bound" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            name='Upper Bound',
            x=forecast_df["Date"],
            y=forecast_df["Upper_Bound"],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            name='Confidence Interval',
            x=forecast_df["Date"],
            y=forecast_df["Lower_Bound"],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(41, 98, 255, 0.2)',
            line=dict(width=0)
        ))
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=history_df['Date'],
        y=history_df['Actual_Units'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#00C853', width=3),
        marker=dict(size=6)
    ))
    
    # Forecasted Data
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecasted_Units'],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#2962FF', width=3, dash='dot'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # Connect last historical to first forecast
    if not history_df.empty and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=[history_df['Date'].iloc[-1], forecast_df['Date'].iloc[0]],
            y=[history_df['Actual_Units'].iloc[-1], forecast_df['Forecasted_Units'].iloc[0]],
            mode='lines',
            line=dict(color='#9e9e9e', width=2, dash='dash'),
            showlegend=False
        ))

    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=f'📊 Demand Forecast: {category}',
            font=dict(size=20)
        ),
        xaxis_title="Date (Monthly)",
        yaxis_title="Units Sold",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    return fig


def get_impact_class(factor_name: str, external_factors: dict) -> str:
    """Determine impact indicator color based on factor type."""
    
    # Positive impact factors (green)
    positive_factors = ["upcoming_promotion", "marketing_campaign"]
    if factor_name in positive_factors:
        return "indicator-positive"
    
    # Negative impact factors (red)
    negative_factors = [
        "supply_chain_disruption", 
        "availability_issues", 
        "logistics_constraints",
        "regulatory_changes"
    ]
    if factor_name in negative_factors:
        return "indicator-negative"
    
    # Neutral/uncertain factors (yellow)
    neutral_factors = ["new_product_launch", "economic_uncertainty"]
    if factor_name in neutral_factors:
        if factor_name == "new_product_launch":
            return "indicator-neutral"
        if factor_name == "economic_uncertainty":
            uncertainty_level = external_factors.get("economic_uncertainty", "None")
            if uncertainty_level in ["Medium", "High"]:
                return "indicator-negative"
            return "indicator-neutral"
    
    # Price change - depends on direction
    if factor_name == "price_change":
        price_direction = external_factors.get("price_change", "Same")
        if price_direction == "Decrease":
            return "indicator-positive"
        elif price_direction == "Increase":
            return "indicator-negative"
    
    return "indicator-none"


def get_factor_description(factor_name: str, external_factors: dict) -> str:
    """Get human-readable description for each factor."""
    
    descriptions = {
        "upcoming_promotion": "Promotional campaign expected to boost demand through price incentives",
        "marketing_campaign": "Active marketing initiatives expanding brand awareness and consideration",
        "new_product_launch": "New SKU introduction - may expand category or cannibalize existing products",
        "availability_issues": "Inventory constraints may limit ability to meet demand",
        "supply_chain_disruption": "Supply chain constraints may cap fulfillment capacity",
        "logistics_constraints": "Transportation limitations may delay product availability",
        "regulatory_changes": "Compliance requirements may impact product availability or pricing",
        "economic_uncertainty": f"{external_factors.get('economic_uncertainty', 'Low')} economic volatility affecting consumer spending patterns",
        "price_change": f"Price {external_factors.get('price_change', 'Same').lower()} may influence purchase decisions"
    }
    
    return descriptions.get(factor_name, "Market condition affecting demand")

def render_forecast_drivers(external_factors: dict, festivals: list, seasonality: dict, data_months: int):
    """
    Render the Forecast Drivers section with external and seasonal factors.
    Now requires data_months parameter to validate seasonal impact display.
    """
    
    st.markdown("""
    <div class="forecast-drivers-container">
        <div class="drivers-header">📊 Forecast Drivers</div>
        <div class="drivers-subtitle">This forecast is influenced by external conditions and seasonal demand patterns.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_external, col_seasonal = st.columns(2)
    
    # === LEFT COLUMN: External Factors ===
    with col_external:
        st.markdown('<div class="column-title">🌍 External Factors Impact</div>', unsafe_allow_html=True)
        
        # [External factors code remains exactly the same - DO NOT MODIFY]
        active_factors = []
        
        if external_factors:
            factor_mapping = {
                "upcoming_promotion": "Promotional Campaign",
                "marketing_campaign": "Marketing Initiative",
                "new_product_launch": "New Product Launch",
                "availability_issues": "Inventory Constraints",
                "price_change": "Price Adjustment",
                "supply_chain_disruption": "Supply Chain Risk",
                "logistics_constraints": "Logistics Limitation",
                "regulatory_changes": "Regulatory Changes",
                "economic_uncertainty": "Economic Uncertainty"
            }
            
            for key, display_name in factor_mapping.items():
                if key == "price_change":
                    if external_factors.get(key) and external_factors.get(key) != "Same":
                        active_factors.append({
                            "name": f"{display_name} ({external_factors.get(key)})",
                            "key": key,
                            "description": get_factor_description(key, external_factors)
                        })
                elif key == "economic_uncertainty":
                    if external_factors.get(key) and external_factors.get(key) != "None":
                        active_factors.append({
                            "name": f"{display_name} ({external_factors.get(key)})",
                            "key": key,
                            "description": get_factor_description(key, external_factors)
                        })
                else:
                    if external_factors.get(key):
                        active_factors.append({
                            "name": display_name,
                            "key": key,
                            "description": get_factor_description(key, external_factors)
                        })
        
        if active_factors:
            for factor in active_factors:
                impact_class = get_impact_class(factor["key"], external_factors)
                st.markdown(f"""
                <div class="driver-card">
                    <div class="driver-indicator {impact_class}"></div>
                    <div class="driver-content">
                        <div class="driver-name">{factor["name"]}</div>
                        <div class="driver-description">{factor["description"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-factors-message">
                No external factors specified.<br>
                Forecast is based on historical patterns only.
            </div>
            """, unsafe_allow_html=True)
    
    # === RIGHT COLUMN: Seasonal Factors (UPDATED LOGIC) ===
    with col_seasonal:
        st.markdown('<div class="column-title">🗓️ Seasonal Factors Impact</div>', unsafe_allow_html=True)
        
        seasonal_items = []
        
        # CRITICAL: Only show seasonal impact if we have 24+ months of data
        if data_months >= 24 and seasonality:
            yearly_strength = seasonality.get("yearly_seasonality_strength", 0)
            
            # Only show if seasonal pattern is meaningful (>15% threshold)
            if yearly_strength > 15:
                seasonal_items.append({
                    "name": "Yearly Seasonal Pattern",
                    "period": "Throughout the year",
                    "impact": f"{yearly_strength:.0f}% variance",
                    "impact_value": yearly_strength
                })
        
        # Display seasonal items or insufficient data message
        if seasonal_items:
            for item in seasonal_items:
                st.markdown(f"""
                <div class="seasonal-card">
                    <div class="seasonal-header">
                        <div class="seasonal-name">{item["name"]}</div>
                        <div class="seasonal-impact">{item["impact"]}</div>
                    </div>
                    <div class="seasonal-period">📅 {item["period"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show honest message about why seasonal impact isn't available
            if data_months < 24:
                st.markdown(f"""
                <div class="no-factors-message">
                    <strong>No significant seasonal impact detected.</strong><br><br>
                    Limited historical data available ({data_months} months). At least 24 months of historical data are required to reliably identify seasonal patterns and calculate their impact on demand.<br><br>
                    As more data becomes available, seasonal trends will be automatically detected and displayed here.
                </div>
                """, unsafe_allow_html=True)
            else:
                # Data is sufficient but no strong seasonality found
                st.markdown("""
                <div class="no-factors-message">
                    No significant seasonal patterns detected for the selected period.<br><br>
                    Demand appears relatively stable throughout the year.
                </div>
                """, unsafe_allow_html=True)


# === NEW SECTION: Festivals & Holidays (Below Forecast Drivers) ===
def render_festivals_awareness(festivals: list):
    """
    Render festivals as contextual awareness only (not forecast drivers).
    Simple, clean display with dates.
    """
    
    if not festivals or len(festivals) == 0:
        return  # Don't show section if no festivals
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📅</div>
        <div>
            <div class="section-title">Upcoming Festivals & Holidays</div>
        </div>
    </div>
    <div class="section-subtitle">Events occurring in the forecast period (for reference only)</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
    
    # Display festivals with dates
    for festival in festivals[:4]:  # Limit to 4 most relevant
        # Parse festival string: "Festival Name (Month Year)"
        if '(' in festival:
            festival_name = festival.split('(')[0].strip()
            festival_date = festival.split('(')[1].replace(')', '').strip()
        else:
            festival_name = festival
            festival_date = "Date TBD"
        
        st.markdown(f"""
        <div style="background: white; padding: 14px 18px; border-radius: 10px; margin-bottom: 10px; 
                    border-left: 4px solid #9c27b0; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
            <div style="font-size: 1rem; font-weight: 600; color: #1a237e; margin-bottom: 6px;">
                {festival_name}
            </div>
            <div style="font-size: 0.88rem; color: #666; display: flex; align-items: center; gap: 6px;">
                📅 {festival_date}
            </div>
        </div>
        """, unsafe_allow_html=True)


# --- MAIN APP ---

if not st.session_state.show_app:
    st.markdown("""
    <style>
        .header-card {
            background: white;
            padding: 22px 28px;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
            max-width: 1500px;
            width: 100%; 
            margin-top: 0px;
            margin-left: 0px;
            margin-right: auto; 
        }
        .header-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a8a;
        }
        .header-subtitle {
            font-size: 1.05rem;
            color: #555;
            margin-top: 6px;
        }
    </style>
    """, unsafe_allow_html=True)

    page = st.container()
    with page:
        st.markdown("""
            <div class="header-card">
                <div class="header-title">📈 AI Demand Forecasting</div>
                <div class="header-subtitle">
                    Powered by advanced time-series AI with contextual market analysis
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:52vh'></div>", unsafe_allow_html=True)
        st.markdown("<div style='padding-bottom:20px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        if st.button("🚀 Generate New Forecast"):
            st.session_state.show_app = True
            st.rerun()

    st.stop()

st.title("📈 Generate Demand Forecast")
st.caption("Turn historical sales data into accurate demand forecasts")

# Initialize session state
if 'forecast_result' not in st.session_state:
    st.session_state['forecast_result'] = None
if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = None
if 'validation_result' not in st.session_state:
    st.session_state['validation_result'] = None
if 'selected_external_factors' not in st.session_state:
    st.session_state['selected_external_factors'] = {}

# --- AI INSIGHT BOX ---
if st.session_state['forecast_result']:
    res = st.session_state['forecast_result']
    insight = res.get('ai_insight', 'AI analysis pending...')
    category = st.session_state.get('selected_category', 'Product')
    
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">✨ AI Insight: {category}</div>
        <div class="insight-text">{insight}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("📤 Upload your sales data to begin forecast generation.")

# --- MAIN LAYOUT ---
col_chart, col_info = st.columns([3, 1])

with col_chart:
    # Data Upload Section with consistent header
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📂</div>
        <div>
            <div class="section-title">Data Upload & Column Mapping</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Sales Data (CSV)",
        type="csv",
        help="Upload a CSV file containing your historical sales data",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str)
        cols = df.columns.tolist()
        
        with st.expander("👀 Preview Uploaded Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total rows: {len(df)} | Columns: {len(cols)}")
        
        # Column Mapping
        with st.expander("⚙️ Column Mapping (Configure your file)", expanded=True):
            st.info("Map your CSV columns to the required fields")
            
            c1, c2, c3 = st.columns(3)
            
            default_date = find_col(['date', 'time', 'period', 'day', 'ts'], cols)
            default_cat = find_col(['cat', 'prod', 'type', 'item', 'store', 'sku', 'product'], cols)
            default_units = find_col(['unit', 'qty', 'sold', 'number', 'sales', 'amount', 'quantity'], cols)
            
            date_col = c1.selectbox("📅 Date Column", cols, index=cols.index(default_date) if default_date in cols else 0)
            category_col = c2.selectbox("📦 Category/Product Column", cols, index=cols.index(default_cat) if default_cat in cols else 0)
            units_col = c3.selectbox("🔢 Units Sold Column", cols, index=cols.index(default_units) if default_units in cols else 0)
        
        st.divider()
        
        # Product/Category Selection with consistent header
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📦</div>
            <div>
                <div class="section-title">Select Product / Category</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if category_col and category_col in df.columns:
            unique_cats = df[category_col].dropna().unique().tolist()
            try:
                unique_cats.sort()
            except TypeError:
                pass
            
            sel_cat = st.selectbox(
                "Select Product/Category to Forecast", 
                unique_cats, 
                key="sel_cat", 
                label_visibility="collapsed"
            )
        else:
            st.error("❌ Category column not found or selected.")
            sel_cat = None
        
        # Validate button
        if sel_cat and st.button("🔍 Validate Data"):
            with st.spinner("Validating data..."):
                temp_df = df.rename(columns={date_col: "Date", category_col: "Category", units_col: "Units_Sold"})
                
                buffer = io.StringIO()
                temp_df.to_csv(buffer, index=False)
                buffer.seek(0)
                
                files = {"file": ("data.csv", buffer.getvalue(), "text/csv")}
                data = {"category": str(sel_cat), "date_col": "Date", "category_col": "Category", "units_col": "Units_Sold"}
                
                try:
                    response = requests.post(f"{API_URL}/validate-data", files=files, data=data, timeout=30)
                    
                    if response.status_code == 200:
                        st.session_state['validation_result'] = response.json()
                        st.success("✅ Data validated successfully!")
                        st.rerun()
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Validation Failed: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend server")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Forecast Configuration
        if st.session_state.get('validation_result'):
            validation = st.session_state['validation_result']
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">⚙️</div>
                <div>
                    <div class="section-title">Forecast Configuration</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Region & Forecast Period
            col_region, col_horizon = st.columns(2)
            
            with col_region:
                region = st.selectbox(
                    "🌍 Region",
                    list(COUNTRY_MAP.keys()),
                    index=0,
                    help="Select your target market region"
                )
                country_code = COUNTRY_MAP[region]
            
            with col_horizon:
                available_horizons = validation.get('available_horizons', [])
                if not available_horizons:
                    st.error("❌ No forecast horizons available. Insufficient data.")
                    selected_horizon = None
                else:
                    selected_horizon = st.radio(
                        "Forecast Period",
                        available_horizons,
                        format_func=lambda x: f"{x} Month{'s' if x > 1 else ''}",
                        horizontal=True
                    )
            
            if selected_horizon:
                data_months = validation['data_summary']['num_months']
                disabled_messages = []
                if 3 not in available_horizons and data_months < 12:
                    disabled_messages.append(f"ℹ️ 3-month forecast requires 12+ months (you have {data_months})")
                if 6 not in available_horizons and data_months < 24:
                    disabled_messages.append(f"ℹ️ 6-month forecast requires 24+ months (you have {data_months})")
                
                if disabled_messages:
                    for msg in disabled_messages:
                        st.info(msg)
                
                # External Factors Section with consistent header
                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">🌍</div>
                    <div>
                        <div class="section-title">External Factors (Optional)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Business Factors**")
                col_left, col_right = st.columns(2)
                with col_left:
                    with st.container():
                        st.markdown('<div class="card-box">', unsafe_allow_html=True)
                        upcoming_promotion = st.checkbox("Upcoming Promotion", help="Planned promotional campaign or sale event")
                        marketing_campaign = st.checkbox("Marketing Campaign", help="Active marketing or advertising initiative")
                        new_product_launch = st.checkbox("New Product Launch", help="New product introduction scheduled")
                        availability_issues = st.checkbox("Availability Issues", help="Expected inventory or stock constraints")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                with col_right:
                    with st.container():
                        st.markdown('<div class="card-box">', unsafe_allow_html=True)
                        price_change = st.selectbox("Price Change", ["Same", "Increase", "Decrease"], help="Planned pricing adjustments")
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("**Risk Factors**")
                col_left, col_right = st.columns(2)
                with col_left:
                    with st.container():
                        st.markdown('<div class="card-box">', unsafe_allow_html=True)
                        supply_chain_disruption = st.checkbox("Supply Chain Disruption", help="Supply chain or sourcing risks")
                        regulatory_changes = st.checkbox("Regulatory Changes", help="Regulatory or compliance changes expected")
                        logistics_constraints = st.checkbox("Logistics Constraints", help="Transportation or delivery constraints")
                        st.markdown('</div>', unsafe_allow_html=True)

                with col_right:
                    with st.container():
                        st.markdown('<div class="card-box">', unsafe_allow_html=True)
                        economic_uncertainty = st.selectbox("Economic Uncertainty", ["None", "Low", "Medium", "High"], help="Economic volatility level")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Generate Forecast Button
                generate_clicked = st.button("🚀 Generate Forecast", use_container_width=True)
                
                if generate_clicked:
                    status_placeholder = st.empty()
                    status_placeholder.info(f"⏳ Preparing data for **{sel_cat}**...")
                    
                    temp_df = df.rename(columns={date_col: "Date", category_col: "Category", units_col: "Units_Sold"})
                    
                    buffer = io.StringIO()
                    temp_df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    
                    files = {"file": ("data.csv", buffer.getvalue(), "text/csv")}
                    data = {
                        "category": str(sel_cat),
                        "date_col": "Date",
                        "category_col": "Category",
                        "units_col": "Units_Sold",
                        "horizon": selected_horizon,
                        "upcoming_promotion": str(upcoming_promotion).lower(),
                        "marketing_campaign": str(marketing_campaign).lower(),
                        "new_product_launch": str(new_product_launch).lower(),
                        "availability_issues": str(availability_issues).lower(),
                        "price_change": price_change,
                        "supply_chain_disruption": str(supply_chain_disruption).lower(),
                        "regulatory_changes": str(regulatory_changes).lower(),
                        "logistics_constraints": str(logistics_constraints).lower(),
                        "economic_uncertainty": economic_uncertainty,
                        "region": region,
                        "country": country_code
                    }
                    
                    # Store selected external factors
                    st.session_state['selected_external_factors'] = {
                        "upcoming_promotion": upcoming_promotion,
                        "marketing_campaign": marketing_campaign,
                        "new_product_launch": new_product_launch,
                        "availability_issues": availability_issues,
                        "price_change": price_change,
                        "supply_chain_disruption": supply_chain_disruption,
                        "regulatory_changes": regulatory_changes,
                        "logistics_constraints": logistics_constraints,
                        "economic_uncertainty": economic_uncertainty
                    }
                    
                    try:
                        status_placeholder.info("⚙️ Running AI model & generating insights...")
                        
                        api_response = requests.post(f"{API_URL}/forecast/upload", files=files, data=data, timeout=120)
                        
                        if api_response.status_code == 200:
                            result = api_response.json()
                            st.session_state['forecast_result'] = result
                            st.session_state['selected_category'] = sel_cat
                            status_placeholder.empty()
                            st.rerun()
                        else:
                            status_placeholder.empty()
                            error_detail = api_response.json().get("detail", "Unknown error")
                            st.error(f"❌ Forecast Failed: {error_detail}")
                            
                    except requests.exceptions.ConnectionError:
                        status_placeholder.empty()
                        st.error("❌ Cannot connect to backend server")
                    except Exception as e:
                        status_placeholder.empty()
                        st.error(f"❌ Error: {str(e)}")
    
    # Display Results
    if st.session_state['forecast_result']:
        res = st.session_state['forecast_result']
        category = st.session_state.get('selected_category', 'Product')
        
        st.divider()
        st.markdown("### 📊 Forecast Results")
        
        # Metrics Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        col_m1.metric(
            label="🎯 Forecasted Units",
            value=f"{res['forecasted_units']:,}",
            delta=f"{res['mom_change_percent']:+.1f}% MoM" if res.get('mom_change_percent') else None,
            delta_color="normal" if res.get('mom_change_percent', 0) >= 0 else "inverse"
        )
        
        col_m2.metric(label="📈 Trend", value=res["trend"], delta=f"{res['confidence']} Confidence", delta_color="off")
        col_m3.metric(label="📅 Historical Data", value=f"{res['data_months']} months", delta="✓ Sufficient" if res['data_months'] >= 12 else "⚠ Limited", delta_color="off")
        col_m4.metric(label="📊 Confidence Range", value=f"{res.get('lower_bound', 0):,} - {res.get('upper_bound', 0):,}", delta="95% interval", delta_color="off")
        
        st.write("")
        
        # Chart
        fig = create_forecast_chart(res["history_data"], res["forecast_data"], category)
        st.plotly_chart(fig, use_container_width=True)
        
        # === FORECAST DRIVERS SECTION ===
        render_forecast_drivers(
            external_factors=st.session_state.get('selected_external_factors', {}),
            festivals=res.get('festivals', []),
            seasonality=res.get('seasonality', {}),
            data_months=res.get('data_months', 0)
        )

        render_festivals_awareness(festivals=res.get('festivals', []))

# Right Column: Data Quality Info
with col_info:
    if st.session_state.get('validation_result'):
        validation = st.session_state['validation_result']
        
        st.subheader("📊 Data Quality")
        data_summary = validation.get("data_summary", {})
        
        if data_summary:
            st.markdown(f"**Date Range:**")
            st.markdown(f"{data_summary.get('date_range_start', 'N/A')}")
            st.markdown(f"to {data_summary.get('date_range_end', 'N/A')}")
            st.write("")
            st.markdown(f"**Data Points:** {data_summary.get('num_months', 0)} months")
            st.markdown(f"**Avg Monthly:** {data_summary.get('avg_monthly_units', 0):,.0f} units")
            st.markdown(f"**Total Units:** {data_summary.get('total_units', 0):,}")
        
        st.markdown("---")
        
        if validation.get('ready_for_forecast'):
            st.success("✅ Ready for forecast")
        else:
            st.warning("⚠️ More data needed")
    
    elif st.session_state['forecast_result']:
        res = st.session_state['forecast_result']
        
        st.subheader("📊 Data Quality")
        data_summary = res.get("data_summary", {})
        
        if data_summary:
            st.markdown(f"**Date Range:**")
            st.markdown(f"{data_summary.get('date_range_start', 'N/A')}")
            st.markdown(f"to {data_summary.get('date_range_end', 'N/A')}")
            st.write("")
            st.markdown(f"**Avg Monthly:** {data_summary.get('avg_monthly_units', 0):,.0f} units")
            st.markdown(f"**Total Units:** {data_summary.get('total_units', 0):,}")
    
    else:
        st.subheader("💡 Tips")
        st.markdown("""
        - Upload **6+ months** minimum
        - **12+ months** for seasonal patterns
        - **24+ months** for best accuracy
        - Ensure CSV has date, category, units columns
        """)

# Footer
st.divider()
st.caption("🚀 Expedition Co. | AI-Powered Supply Chain Management | Demand Forecasting Module v1.0")
