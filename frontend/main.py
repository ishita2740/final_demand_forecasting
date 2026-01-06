# frontend/main.py
# ----------------
# Streamlit frontend for AI Demand Forecasting

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import io
from datetime import datetime

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Expedition Co. Demand Forecasting",
    layout="wide",
    page_icon="📈"
)

# --- CUSTOM CSS ---
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
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .data-quality-good {
        color: #2e7d32;
        font-weight: bold;
    }
    
    .data-quality-warning {
        color: #f57c00;
        font-weight: bold;
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
    """
    Generate Plotly chart with historical data, forecast, and confidence interval.
    """
    history_df = pd.DataFrame(history_data)
    forecast_df = pd.DataFrame(forecast_data)
    
    # Convert dates
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    
    fig = go.Figure()
    
    # 1. Confidence Interval (Shaded Area)
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
    
    # 2. Historical Data (Solid Line)
    fig.add_trace(go.Scatter(
        x=history_df['Date'],
        y=history_df['Actual_Units'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#00C853', width=3),
        marker=dict(size=6)
    ))
    
    # 3. Forecasted Data (Dotted Line)
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecasted_Units'],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#2962FF', width=3, dash='dot'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # 4. Connect last historical point to first forecast
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


# --- MAIN APP ---

st.title("📈 AI Demand Forecasting")
st.caption("Powered by Advanced Time-Series AI with Contextual Market Analysis")

# Initialize session state
if 'forecast_result' not in st.session_state:
    st.session_state['forecast_result'] = None
if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = None

# --- AI INSIGHT BOX (Conditional) ---
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
    st.info("📤 Upload your sales data and generate a forecast to see AI-powered insights.")

# --- MAIN LAYOUT ---
col_chart, col_info = st.columns([3, 1])

with col_chart:
    st.subheader("📂 Data Upload & Forecast Generation")
    
    uploaded_file = st.file_uploader(
        "Upload Sales Data (CSV)",
        type="csv",
        help="Upload a CSV file containing your historical sales data"
    )
    
    if uploaded_file:
        # Read the file
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str)
        cols = df.columns.tolist()
        
        # Show data preview
        with st.expander("👀 Preview Uploaded Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total rows: {len(df)} | Columns: {len(cols)}")
        
        # Column Mapping
        with st.expander("⚙️ Column Mapping (Configure your file)", expanded=True):
            st.info("Map your CSV columns to the required fields")
            
            c1, c2, c3 = st.columns(3)
            
            # Auto-detect columns
            default_date = find_col(['date', 'time', 'period', 'day', 'ts'], cols)
            default_cat = find_col(['cat', 'prod', 'type', 'item', 'store', 'sku', 'product'], cols)
            default_units = find_col(['unit', 'qty', 'sold', 'number', 'sales', 'amount', 'quantity'], cols)
            
            date_col = c1.selectbox(
                "📅 Date Column",
                cols,
                index=cols.index(default_date) if default_date in cols else 0
            )
            category_col = c2.selectbox(
                "📦 Category/Product Column",
                cols,
                index=cols.index(default_cat) if default_cat in cols else 0
            )
            units_col = c3.selectbox(
                "🔢 Units Sold Column",
                cols,
                index=cols.index(default_units) if default_units in cols else 0
            )
        
        st.divider()
        
        # Forecast Parameters
        st.markdown("### 🎯 Forecast Parameters")
        c_param_left, c_param_right = st.columns(2)
        
        # Get unique categories
        if category_col and category_col in df.columns:
            unique_cats = df[category_col].dropna().unique().tolist()
            try:
                unique_cats.sort()
            except TypeError:
                pass
            
            sel_cat = c_param_left.selectbox(
                "📦 Select Product/Category to Forecast",
                unique_cats,
                key="sel_cat"
            )
        else:
            st.error("❌ Category column not found or selected.")
            sel_cat = None
        
        horizon = c_param_right.selectbox(
            "📆 Forecast Horizon",
            options=[1, 3, 6],
            format_func=lambda x: f"{x} Month{'s' if x > 1 else ''}",
            key="horizon"
        )
        
        st.write("")
        
        # Generate Button
        col_btn1, col_btn2 = st.columns([1, 3])
        generate_clicked = col_btn1.button(
            "🚀 Generate Forecast",
            type="primary",
            use_container_width=True
        )
        
        if generate_clicked and sel_cat:
            status_placeholder = st.empty()
            status_placeholder.info(f"⏳ Preparing data for **{sel_cat}**...")
            
            # Prepare data for API
            temp_df = df.rename(columns={
                date_col: "Date",
                category_col: "Category",
                units_col: "Units_Sold"
            })
            
            buffer = io.StringIO()
            temp_df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            files = {"file": ("data.csv", buffer.getvalue(), "text/csv")}
            data = {
                "category": str(sel_cat),
                "date_col": "Date",
                "category_col": "Category",
                "units_col": "Units_Sold",
                "horizon": horizon
            }
            
            try:
                status_placeholder.info("⚙️ Running AI model & generating insights...")
                
                api_response = requests.post(
                    f"{API_URL}/forecast/upload",
                    files=files,
                    data=data,
                    timeout=120
                )
                
                if api_response.status_code == 200:
                    st.session_state['forecast_result'] = api_response.json()
                    st.session_state['selected_category'] = sel_cat
                    status_placeholder.empty()
                    st.rerun()
                else:
                    status_placeholder.empty()
                    error_detail = api_response.json().get("detail", "Unknown error")
                    st.error(f"❌ Forecast Failed: {error_detail}")
                    st.session_state['forecast_result'] = None
                    
            except requests.exceptions.ConnectionError:
                status_placeholder.empty()
                st.error("❌ Cannot connect to backend server. Make sure it's running on http://127.0.0.1:8000")
            except Exception as e:
                status_placeholder.empty()
                st.error(f"❌ Error: {str(e)}")
    
    # --- DISPLAY FORECAST RESULTS ---
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
            delta=f"{res['mom_change_percent']:+.1f}% MoM",
            delta_color="normal" if res['mom_change_percent'] >= 0 else "inverse"
        )
        
        col_m2.metric(
            label="📈 Trend",
            value=res["trend"],
            delta=f"{res['confidence']} Confidence",
            delta_color="off"
        )
        
        col_m3.metric(
            label="📅 Historical Data",
            value=f"{res['data_months']} months",
            delta="✓ Sufficient" if res['data_months'] >= 12 else "⚠ Limited",
            delta_color="off"
        )
        
        col_m4.metric(
            label="📊 Confidence Range",
            value=f"{res.get('lower_bound', 0):,} - {res.get('upper_bound', 0):,}",
            delta="95% interval",
            delta_color="off"
        )
        
        st.write("")
        
        # Chart
        fig = create_forecast_chart(
            res["history_data"],
            res["forecast_data"],
            category
        )
        st.plotly_chart(fig, use_container_width=True)

# --- RIGHT COLUMN: Data Quality Info ---
with col_info:
    if st.session_state['forecast_result']:
        res = st.session_state['forecast_result']
        
        st.subheader("📊 Data Quality")
        data_summary = res.get("data_summary", {})
        
        if data_summary:
            st.markdown(f"**Date Range:** {data_summary.get('date_range_start', 'N/A')} to {data_summary.get('date_range_end', 'N/A')}")
            st.markdown(f"**Avg Monthly Sales:** {data_summary.get('avg_monthly_units', 0):,.0f} units")
            st.markdown(f"**Total Units:** {data_summary.get('total_units', 0):,}")
    
    else:
        st.subheader("💡 Tips")
        st.markdown("""
        - Upload **12+ months** of data for best results
        - Ensure your CSV has **date**, **category**, and **units** columns
        - The AI automatically detects seasonal patterns
        """)

# --- FOOTER ---
st.divider()
st.caption("🚀 Expedition Co. | AI-Powered Supply Chain Management | Demand Forecasting Module v1.0")