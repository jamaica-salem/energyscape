"""
ENERGYSCAPE: Multi-Seasonal Mathematical-Computational Framework for Predictive Energy Management and Carbon Reduction
Main Streamlit Application — Restructured to Student Mock Design Architecture
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import custom modules
from modules.data_processing import (
    load_historical_bills, load_appliance_loads, load_seasonal_data, validate_dataset
)
from modules.historical_analysis import calculate_historical_metrics
from modules.seasonal_analysis import calculate_seasonal_metrics, DEFAULT_DRY_MONTHS, DEFAULT_WET_MONTHS
from modules.load_analysis import calculate_appliance_loads, get_load_summary
from modules.forecasting import fit_ets_forecast
from modules.validation import calculate_mape, calculate_rmse, interpret_mape, verify_computational_consistency
from modules.carbon import calculate_carbon_emissions, calculate_carbon_summary
from modules.scenarios import calculate_bau_baseline, simulate_conservation_scenarios
from modules.optimization import optimize_conservation_target, monitor_target_consumption, calculate_sensitivity_analysis
from modules.recommendations import generate_appliance_recommendations, generate_executive_summary_recommendation
from utils.formatting import format_currency, format_kwh, format_co2, format_pct, format_num

# Streamlit Page Config
st.set_page_config(
    page_title="ENERGYSCAPE — Energy & Carbon Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# COMPREHENSIVE THEME CSS & MOCK DESIGN STYLING
# ----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Body & Background (Bankio Light Warm Gray Theme) */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F3F4F6 !important;
        color: #111827 !important;
    }

    .stApp {
        background-color: #F3F4F6 !important;
    }

    /* Top Streamlit Header Bar (Deploy & Menu) in Pure White #FFFFFF with Bottom Border #EAECF0 */
    header[data-testid="stHeader"],
    div[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #EAECF0 !important;
    }
    header[data-testid="stHeader"] *,
    div[data-testid="stHeader"] * {
        color: #111827 !important;
        fill: #111827 !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Headings Typography (Bankio Clean Sans) */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    .page-title {
        color: #0B4F46 !important;
    }

    /* Top Greeting & Header Bar */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.25rem;
        padding-bottom: 0.25rem;
    }

    /* Card Containers & Interactive Hover (Bankio Crisp White Cards) */
    .ui-card {
        background-color: #FFFFFF !important;
        border: 1px solid #EAECF0 !important;
        border-radius: 18px !important;
        padding: 1.5rem 1.75rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05) !important;
        box-sizing: border-box !important;
        overflow: visible !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .ui-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px -3px rgba(16, 24, 40, 0.08) !important;
    }

    /* Clean Minimal Cards (No top colored borders, pure Bankio white style) */
    .card-emerald,
    .card-teal,
    .card-blue,
    .card-cyan {
        border-top: 1px solid #EAECF0 !important;
        background: #FFFFFF !important;
    }

    /* Hero Highlight Card (Bankio Solid Flat Deep Emerald Card #0B4F46) */
    .hero-consumption-card {
        background-color: #0B4F46 !important;
        border-radius: 20px !important;
        padding: 1.75rem 2.25rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 12px 32px -5px rgba(11, 79, 70, 0.3) !important;
        margin-bottom: 1.75rem !important;
        position: relative !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }
    .hero-card-label {
        font-size: 0.82rem !important;
        color: #A7F3D0 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    .hero-card-title {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em !important;
    }
    .hero-metric-val {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.03em !important;
        line-height: 1.1 !important;
        margin: 0.2rem 0 !important;
    }
    .hero-subtext {
        font-size: 0.85rem !important;
        color: #D1FAE5 !important;
    }

    /* Soothing Matcha Pill Badges (No Bright/Neon Greens) */
    .pill-badge-blue,
    .pill-badge-teal,
    .pill-badge-green {
        background-color: #E8F5E9 !important;
        color: #1B5E20 !important;
        padding: 0.3rem 0.8rem !important;
        border-radius: 9999px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        white-space: nowrap !important;
    }

    /* Metric Display */
    .kpi-label {
        font-size: 0.8rem !important;
        color: #6B7280 !important;
        font-weight: 600 !important;
        margin-bottom: 0.15rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    .kpi-val {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        letter-spacing: -0.02em !important;
        margin: 0.15rem 0 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Sidebar Navigation (Deep Emerald Green Theme #0B4F46 - Seamless Borderless) */
    section[data-testid="stSidebar"] {
        background-color: #0B4F46 !important;
        border: none !important;
        border-right: none !important;
        box-shadow: none !important;
    }

    /* Sidebar Collapse/Close Button Color Fix (#FFFFFF) */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapseButton"] path,
    button[aria-label="Close sidebar"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Close sidebar"] svg,
    button[aria-label="Collapse sidebar"] svg,
    button[aria-label="Close sidebar"] path,
    button[aria-label="Collapse sidebar"] path,
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] button[kind="header"] svg,
    section[data-testid="stSidebar"] button[kind="header"] path,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] svg,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] path {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    /* Strictly Scoped Nav Content Typography */
    .sidebar-brand,
    .sidebar-section-header,
    div[role="radiogroup"] label {
        font-family: 'Inter', sans-serif !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 1.5rem;
        padding-bottom: 0.85rem;
        border-bottom: none !important;
        letter-spacing: -0.02em;
    }
    .sidebar-section-header {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.5rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Hide Radio Circle Dots Completely for Clean Sidebar Menu Buttons */
    div[role="radiogroup"] label input[type="radio"],
    div[role="radiogroup"] label [data-testid="stRadioButtonCustomIcon"],
    div[role="radiogroup"] label div[aria-hidden="true"],
    div[role="radiogroup"] label svg {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Navigation Radio Items (Deep Emerald Background with White Text) */
    div[role="radiogroup"] label,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span,
    div[role="radiogroup"] label div,
    div[role="radiogroup"] label * {
        color: #E6F4EA !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
    }
    div[role="radiogroup"] label {
        background-color: transparent !important;
        padding: 10px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        transform: translateX(3px) !important;
    }
    div[role="radiogroup"] label:hover * {
        color: #FFFFFF !important;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label[data-checked="true"] p,
    div[role="radiogroup"] label[data-checked="true"] span,
    div[role="radiogroup"] label[data-checked="true"] div,
    div[role="radiogroup"] label[data-checked="true"] * {
        color: #0B4F46 !important;
        font-weight: 800 !important;
        font-size: 1.02rem !important;
    }

    /* Sidebar Widgets & Labels White Styling */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
    }

    /* Widget Labels & Inputs (Bankio Pure White Rounded Inputs) */
    label[data-testid="stWidgetLabel"] p {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    /* Form Inputs, Selectboxes & Dropdown Outer Containers */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div,
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stMultiSelect"] > div > div,
    button[data-testid*="stNumberInputStep"],
    input[type="number"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #EAECF0 !important;
        border-color: #EAECF0 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-baseweb="select"]:hover > div,
    div[data-baseweb="select"]:focus-within > div,
    div[data-testid="stSelectbox"] > div > div:hover,
    div[data-testid="stSelectbox"] > div > div:focus-within,
    div[data-testid="stMultiSelect"] > div > div:hover,
    div[data-testid="stMultiSelect"] > div > div:focus-within {
        border-color: #0B4F46 !important;
        box-shadow: 0 0 0 1px #0B4F46 !important;
        outline: none !important;
    }

    /* ABSOLUTE CLEAN SEARCH BAR - NO DOUBLE BOX / NO WHITE RECTANGLE */
    div[data-testid="stTextInput"],
    div[data-testid="stTextInput"] > div,
    div[data-testid="stTextInput"] [data-baseweb="input"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="base-input"],
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        border: 1px solid #EAECF0 !important;
        border-color: #EAECF0 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] *:hover,
    div[data-testid="stTextInput"] *:focus,
    div[data-testid="stTextInput"] *:focus-within,
    div[data-testid="stTextInput"] *:active {
        border-color: #EAECF0 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Multiselect Tag Pills (100% Solid Emerald Green #0B4F46) */
    [data-baseweb="tag"],
    span[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: #0B4F46 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-color: #0B4F46 !important;
        border-radius: 8px !important;
        padding: 3px 10px !important;
        margin: 2px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] p,
    [data-baseweb="tag"] div,
    [data-baseweb="tag"] button,
    [data-baseweb="tag"] [role="button"],
    [data-baseweb="tag"] svg,
    [data-baseweb="tag"] path {
        background-color: transparent !important;
        background: transparent !important;
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Action Buttons (Bankio Deep Teal Pill Button #0B4F46) */
    div.stButton > button,
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #0B4F46 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(11, 79, 70, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #063B34 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(11, 79, 70, 0.35) !important;
    }
    div.stButton > button * {
        color: #FFFFFF !important;
    }

    /* Minimal Flat Custom File Uploader Styling (Centered Layout) */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #0B4F46 !important;
        border-radius: 18px !important;
        padding: 1.5rem 2rem !important;
        box-shadow: 0 2px 10px rgba(11, 79, 70, 0.05) !important;
        transition: all 0.25s ease !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #10B981 !important;
        background-color: #F9FAFB !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.1) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
    div[data-testid="stFileUploader"] label {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        width: 100% !important;
        margin: 0 auto 0.75rem auto !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] *,
    div[data-testid="stFileUploader"] label * {
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        display: flex !important;
        width: 100% !important;
        margin: 0 auto !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #0B4F46 !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploader"] section > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        gap: 8px !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploader"] section button {
        background-color: #0B4F46 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.75rem !important;
        box-shadow: 0 4px 12px rgba(11, 79, 70, 0.2) !important;
        margin: 0 auto !important;
    }
    div[data-testid="stFileUploader"] small {
        text-align: center !important;
        display: block !important;
        margin-top: 0.5rem !important;
        color: #6B7280 !important;
    }

    /* Table Header Styling (Matching IMG_3512.jpeg - Deep Emerald Top Column Header) */
    div[data-testid="stTable"] table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border: 1px solid #EAECF0 !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        width: 100% !important;
    }
    div[data-testid="stTable"] table thead tr th,
    div[data-testid="stTable"] th,
    .ui-table thead tr th {
        background-color: #0B4F46 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em !important;
        padding: 12px 16px !important;
        border-bottom: 2px solid #063B34 !important;
        border-top: none !important;
        text-align: left !important;
    }
    div[data-testid="stTable"] table thead tr th *,
    div[data-testid="stTable"] th * {
        color: #FFFFFF !important;
    }
    div[data-testid="stTable"] table tbody tr td,
    div[data-testid="stTable"] td,
    .ui-table tbody tr td {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-size: 0.88rem !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #F3F4F6 !important;
    }
    div[data-testid="stTable"] table tbody tr:hover td {
        background-color: #F9FAFB !important;
    }

    /* Bankio Table Custom Styling (IMG_3512.jpeg Specification: Emerald Top Header & First Column) */
    .bankio-table-container {
        border: 1px solid #EAECF0 !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.03) !important;
        margin-top: 0.75rem !important;
        margin-bottom: 1.25rem !important;
        background-color: #FFFFFF !important;
    }
    .bankio-table {
        width: 100% !important;
        border-collapse: collapse !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
    }
    .bankio-table thead tr {
        background-color: #0B4F46 !important;
    }
    .bankio-table thead tr th {
        background-color: #0B4F46 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 14px 18px !important;
        text-align: left !important;
        border: none !important;
    }
    .bankio-table tbody tr {
        border-bottom: 1px solid #F3F4F6 !important;
        transition: background-color 0.15s ease !important;
    }
    .bankio-table tbody tr:last-child {
        border-bottom: none !important;
    }
    .bankio-table tbody tr:hover {
        background-color: #F9FAFB !important;
    }
    .bankio-table tbody tr td {
        padding: 12px 18px !important;
        color: #111827 !important;
        vertical-align: middle !important;
    }
    /* Table Styling: First Row (Header) Deep Emerald Green Only */
    .bankio-table tbody tr td.first-col-bold {
        font-weight: 700 !important;
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

def filter_dataframe_by_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filters any Pandas DataFrame across all string and numeric columns matching the search query.
    Case-insensitive search.
    """
    if not query or df is None or df.empty:
        return df
    
    q = str(query).strip().lower()
    if not q:
        return df
    
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        col_str = df[col].astype(str).str.lower()
        mask = mask | col_str.str.contains(q, regex=False, na=False)
        
    return df[mask]

def search_entire_system(query: str, historical_df: pd.DataFrame, appliance_df: pd.DataFrame, seasonal_df: pd.DataFrame, apps_processed: pd.DataFrame) -> dict:
    """
    Scans all 10 system pages and returns a dictionary mapping page names to match status.
    """
    if not query or not str(query).strip():
        return {}
    
    q = str(query).strip().lower()
    matches = {}
    
    # 1. Dashboard
    dash_m = len(filter_dataframe_by_search(appliance_df, q))
    if dash_m > 0 or any(w in q for w in ["dashboard", "consumption", "baseline"]):
        matches["Dashboard"] = max(dash_m, 1)
        
    # 2. Data Input
    di_m = (
        len(filter_dataframe_by_search(appliance_df, q))
        + len(filter_dataframe_by_search(historical_df, q))
        + len(filter_dataframe_by_search(seasonal_df, q))
    )
    if di_m > 0 or any(w in q for w in ["data input", "appliance", "billing", "inventory"]):
        matches["Data Input"] = max(di_m, 1)
        
    # 3. Season
    if any(m_word in q for m_word in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "dry", "wet", "season", "thermal", "surge", "peak"]):
        matches["Season"] = 2
        
    # 4. Energy L.
    el_m = len(filter_dataframe_by_search(apps_processed, q))
    if el_m > 0 or any(w in q for w in ["energy l.", "air conditioner", "computer", "lighting", "fan", "refrigerator", "watt", "kwh"]):
        matches["Energy L."] = max(el_m, 1)
        
    # 5. Forecast
    if any(w in q for w in ["forecast", "ets", "projection", "mape", "expenditure", "upper", "lower", "bill"]):
        matches["Forecast"] = 10
        
    # 6. Carbon
    if any(w in q for w in ["carbon", "co2", "emission", "footprint", "greenhouse", "bau", "baseline"]):
        matches["Carbon"] = 4
        
    # 7. Scenario
    if any(w in q for w in ["scenario", "reduction", "conservation", "intervention", "5%", "10%", "15%"]):
        matches["Scenario"] = 4
        
    # 8. Optimization
    if any(w in q for w in ["optimization", "goal", "linear", "constraint", "1945", "target", "savings"]):
        matches["Optimization"] = 5
        
    # 9. Impact
    if any(w in q for w in ["impact", "la paz", "an-anaao", "benchmark", "sensitivity", "elasticity"]):
        matches["Impact"] = 6
        
    # 10. Reports
    if any(w in q for w in ["report", "executive", "comparative", "methodology", "audit"]):
        matches["Reports"] = 9
        
    return matches

def render_bankio_table(df: pd.DataFrame, first_col_green: bool = False, search_query: str = ""):
    """
    Renders a custom HTML table matching the Bankio UI design in IMG_3512.jpeg.
    - TOP HEADER ROW ONLY: Deep Emerald Green (#0B4F46) background with White text (#FFFFFF)
    - Data rows: Crisp white background with dark charcoal text (#111827).
    - Automatically filters rows when a search query is active!
    """
    if df is None or df.empty:
        return
    
    display_df = df.copy()
    
    # Apply global search query filtering if active
    active_search = search_query if search_query else st.session_state.get("global_search_term", "")
    if active_search:
        display_df = filter_dataframe_by_search(display_df, active_search)
        
    if display_df.empty:
        st.markdown(f'<div style="padding: 1rem 1.25rem; color: #6B7280; font-size: 0.88rem; font-style: italic; background: #FFFFFF; border: 1px solid #EAECF0; border-radius: 12px; margin: 0.75rem 0;">No matching records found for "{active_search}".</div>', unsafe_allow_html=True)
        return
    
    html = ['<div class="bankio-table-container">']
    html.append('<table class="bankio-table">')
    
    # Top Header Row (Solid Emerald Green)
    html.append('<thead><tr>')
    for col in display_df.columns:
        col_title = str(col).replace('_', ' ').title()
        html.append(f'<th>{col_title}</th>')
    html.append('</tr></thead>')
    
    # Body Rows
    html.append('<tbody>')
    for _, row in display_df.iterrows():
        html.append('<tr>')
        for i, val in enumerate(row):
            if isinstance(val, (int, float)):
                if isinstance(val, float):
                    val_str = f"{val:,.2f}"
                else:
                    val_str = f"{val:,}"
            else:
                val_str = str(val)
                
            if i == 0:
                html.append(f'<td class="first-col-bold">{val_str}</td>')
            elif str(val_str).lower() in ['very high', 'high', 'pass']:
                html.append(f'<td><span class="pill-badge-green">{val_str}</span></td>')
            elif str(val_str).lower() in ['moderate', 'low']:
                html.append(f'<td><span class="pill-badge-green" style="opacity: 0.85;">{val_str}</span></td>')
            else:
                html.append(f'<td>{val_str}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    
    st.markdown(''.join(html), unsafe_allow_html=True)

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_ORDER = {month: idx + 1 for idx, month in enumerate(MONTH_NAMES)}

def get_dataset_schools(*dfs: pd.DataFrame) -> list:
    """Return unique school names in the order they appear in active datasets."""
    schools = []
    for df in dfs:
        if df is None or df.empty or "school" not in df.columns:
            continue
        for school in df["school"].dropna().astype(str):
            if school and school not in schools:
                schools.append(school)
    return schools

def get_forecast_school(selected_school: str, historical_schools: list) -> str:
    """Forecasting requires a school with historical billing rows."""
    if selected_school in historical_schools:
        return selected_school
    return historical_schools[0] if historical_schools else selected_school

def get_seasonal_source(seasonal_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer the active seasonal dataset; fall back to historical bills if needed."""
    if seasonal_df is not None and not seasonal_df.empty and "consumption_kwh" in seasonal_df.columns:
        return seasonal_df
    return historical_df

def build_monthly_season_summary(source_df: pd.DataFrame, selected_school: str) -> tuple:
    """Build month-level seasonal chart data from either kWh seasonal data or bill data."""
    if source_df is None or source_df.empty:
        return pd.DataFrame(), "bill_php", "Avg Monthly Bill (₱)"

    data = source_df.copy()
    if selected_school and "school" in data.columns:
        data = data[data["school"] == selected_school]

    value_col = "consumption_kwh" if "consumption_kwh" in data.columns else "bill_php"
    value_label = "Avg Monthly Consumption (kWh)" if value_col == "consumption_kwh" else "Avg Monthly Bill (₱)"
    needed_cols = ["month", value_col]
    data = data.dropna(subset=[col for col in needed_cols if col in data.columns])
    if data.empty:
        return pd.DataFrame(), value_col, value_label

    monthly_summary = data.groupby("month")[value_col].mean().reset_index()
    monthly_summary["month_num"] = monthly_summary["month"].map(MONTH_ORDER)
    monthly_summary = monthly_summary.dropna(subset=["month_num"]).sort_values("month_num")
    monthly_summary["month_num"] = monthly_summary["month_num"].astype(int)
    monthly_summary["month_name"] = monthly_summary["month_num"].apply(lambda m: MONTH_ABBR[m - 1])
    monthly_summary["full_month_name"] = monthly_summary["month_num"].apply(lambda m: MONTH_NAMES[m - 1])
    overall_mean = monthly_summary[value_col].mean() if not monthly_summary.empty else 1.0
    monthly_summary["seasonal_index"] = monthly_summary[value_col] / overall_mean if overall_mean > 0 else 1.0
    return monthly_summary, value_col, value_label

def summarize_school_for_comparison(school: str, historical_df: pd.DataFrame, appliance_df: pd.DataFrame, seasonal_source_df: pd.DataFrame) -> dict:
    """Calculate all comparative metrics for a school from active datasets."""
    apps = calculate_appliance_loads(appliance_df, electricity_rate, school)
    load_sum = get_load_summary(apps, electricity_rate)
    seasonal = calculate_seasonal_metrics(seasonal_source_df, DEFAULT_DRY_MONTHS, DEFAULT_WET_MONTHS, school)
    bau = calculate_bau_baseline(load_sum.get("total_kwh", 0.0), electricity_rate, emission_factor)
    opt = optimize_conservation_target(appliance_df=apps, electricity_rate=electricity_rate, emission_factor=emission_factor)
    forecast = fit_ets_forecast(historical_df, school)
    forecast_df = forecast["forecast_df"]
    peak = max(seasonal["monthly_averages"], key=seasonal["monthly_averages"].get) if seasonal.get("monthly_averages") else "N/A"
    top_load = f"{load_sum.get('top_appliance', 'N/A')} ({load_sum.get('top_kwh', 0):,.0f} kWh)"
    return {
        "school": school,
        "load_summary": load_sum,
        "seasonal": seasonal,
        "bau": bau,
        "opt": opt,
        "forecast": forecast,
        "forecast_df": forecast_df,
        "peak": peak,
        "top_load": top_load,
    }

# ----------------------------------------------------
# APPLICATION ENTRY & WELCOME LANDING STATE
# ----------------------------------------------------
if "entered_app" not in st.session_state:
    st.session_state["entered_app"] = False
if "close_requested" not in st.session_state:
    st.session_state["close_requested"] = False

def enter_application():
    st.session_state["entered_app"] = True
    st.session_state["nav_selection"] = "Dashboard"

def close_application():
    st.session_state["close_requested"] = True

if st.session_state["close_requested"]:
    components.html(
        """
        <script>
            [window.top, window.parent, window].forEach(function (target) {
                try {
                    target.close();
                } catch (error) {}
            });
            document.body.innerHTML = '<p style="font-family: sans-serif; text-align: center; margin-top: 2rem;">You can close this tab.</p>';
        </script>
        """,
        height=100,
    )
    st.stop()

# FULL SCREEN WELCOME LANDING PAGE (EXACT MATCH TO REFERENCE IMAGE)
if not st.session_state["entered_app"]:
    import base64
    welcome_bg_path = Path(__file__).parent / "assets" / "welcome_bg.png"
    bg_b64 = ""
    if welcome_bg_path.exists():
        with open(welcome_bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@700;900&family=Inter:wght@800;900&display=swap');

        .stApp {{
            background-image: url('data:image/png;base64,{bg_b64}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{
            display: none !important;
        }}
        div[data-testid="stAppViewContainer"] {{
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            height: 100vh !important;
            min-height: 100vh !important;
            overflow: hidden !important;
        }}
        section[data-testid="stMain"] {{
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            height: 100vh !important;
            min-height: 100vh !important;
            width: 100% !important;
            overflow: hidden !important;
            padding: 0 !important;
        }}
        .main {{
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            min-height: 100vh !important;
            height: 100vh !important;
            width: 100% !important;
        }}
        .main .block-container {{
            padding: 0 1.5rem !important;
            max-width: 960px !important;
            width: 100% !important;
            margin: auto !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        /* Bubbly Title with Thick White Outline matching reference image */
        .welcome-ref-title {{
            font-family: 'Fredoka', 'Outfit', sans-serif !important;
            font-size: 5.8rem !important;
            font-weight: 900 !important;
            color: #0B4F46 !important;
            -webkit-text-fill-color: #0B4F46 !important;
            text-shadow: 
                -4px -4px 0 #FFFFFF, 
                 4px -4px 0 #FFFFFF, 
                -4px  4px 0 #FFFFFF, 
                 4px  4px 0 #FFFFFF,
                -6px  0px 0 #FFFFFF,
                 6px  0px 0 #FFFFFF,
                 0px -6px 0 #FFFFFF,
                 0px  6px 0 #FFFFFF,
                 0 12px 30px rgba(0, 0, 0, 0.45) !important;
            letter-spacing: 0.05em !important;
            margin: 0 0 0.5rem 0 !important;
            line-height: 1.05 !important;
            text-align: center !important;
        }}

        /* Bubbly Subtitle with Thick White Outline matching reference image */
        .welcome-ref-subtitle {{
            font-family: 'Fredoka', 'Inter', sans-serif !important;
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #0B4F46 !important;
            -webkit-text-fill-color: #0B4F46 !important;
            text-shadow: 
                -2.5px -2.5px 0 #FFFFFF, 
                 2.5px -2.5px 0 #FFFFFF, 
                -2.5px  2.5px 0 #FFFFFF, 
                 2.5px  2.5px 0 #FFFFFF,
                -3.5px  0px 0 #FFFFFF,
                 3.5px  0px 0 #FFFFFF,
                 0px -3.5px 0 #FFFFFF,
                 0px  3.5px 0 #FFFFFF,
                 0 6px 18px rgba(0, 0, 0, 0.35) !important;
            letter-spacing: 0.06em !important;
            margin-bottom: 2.5rem !important;
            text-transform: uppercase !important;
            text-align: center !important;
            max-width: 850px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            line-height: 1.25 !important;
        }}

        /* Capsule Buttons matching reference image proportions */
        div[data-testid="stButton"] {{
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }}
        div[data-testid="stButton"] button {{
            background: linear-gradient(180deg, #0B4F46 0%, #194D40 100%) !important;
            color: #FFFFFF !important;
            font-family: 'Fredoka', 'Outfit', sans-serif !important;
            font-size: 1.75rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.08em !important;
            padding: 0.75rem 2rem !important;
            border-radius: 9999px !important;
            border: 4px solid #FFFFFF !important;
            box-shadow: 0 10px 24px rgba(11, 79, 70, 0.5), 0 4px 10px rgba(0,0,0,0.3) !important;
            transition: all 0.25s ease !important;
            width: 320px !important;
            max-width: 320px !important;
            margin: 0 auto !important;
            display: block !important;
            text-shadow: -2px -2px 0 #0B4F46, 2px -2px 0 #0B4F46, -2px 2px 0 #0B4F46, 2px 2px 0 #0B4F46 !important;
        }}
        div[data-testid="stButton"] button:hover {{
            transform: scale(1.06) !important;
            background: linear-gradient(180deg, #194D40 0%, #0B4F46 100%) !important;
            box-shadow: 0 14px 32px rgba(11, 79, 70, 0.7), 0 6px 16px rgba(0,0,0,0.4) !important;
            border-color: #FFFFFF !important;
        }}
    </style>

    <div style="text-align: center; margin-bottom: 2.5rem;">
        <div class="welcome-ref-title">ENERGYSCAPE</div>
        <div class="welcome-ref-subtitle">A MATHEMATICAL-COMPUTATIONAL DECISION SUPPORT SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    # STACKED CAPSULE BUTTONS (ENTER & EXIT) MATCHING REFERENCE IMAGE EXACTLY
    col_w1, col_w2, col_w3 = st.columns([1, 1.3, 1])
    with col_w2:
        st.button("ENTER", key="btn_welcome_enter_main", on_click=enter_application, use_container_width=True)
        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)
        st.button("EXIT", key="btn_welcome_exit_main", on_click=close_application, use_container_width=True)

    st.stop()

# ----------------------------------------------------
# SIDEBAR NAVIGATION (EXACT 10 ITEMS - NO EMOJIS)
# ----------------------------------------------------
NAV_OPTIONS = [
    "Dashboard",
    "Data Input",
    "Season",
    "Energy L.",
    "Forecast",
    "Carbon",
    "Scenario",
    "Optimization",
    "Impact",
    "Reports"
]

if "nav_selection" not in st.session_state or st.session_state["nav_selection"] not in NAV_OPTIONS:
    st.session_state["nav_selection"] = "Dashboard"

def navigate_to_page(target_page: str):
    st.session_state["nav_selection"] = target_page

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ <span>ENERGYSCAPE</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-header">NAVIGATION VIEWS</div>', unsafe_allow_html=True)
    navigation_option = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="nav_selection",
        label_visibility="collapsed"
    )

# System default parameters
electricity_rate = 11.00
emission_factor = 0.70
forecast_horizon = 12
uploaded_bills = st.session_state.get("uploaded_bills", None)
uploaded_loads = st.session_state.get("uploaded_loads", None)
uploaded_seasonal = st.session_state.get("uploaded_seasonal", None)

# ----------------------------------------------------
# DATA INGESTION
# ----------------------------------------------------
try:
    historical_df = load_historical_bills(uploaded_bills)
    appliance_df = load_appliance_loads(uploaded_loads)
    seasonal_df = load_seasonal_data(uploaded_seasonal)
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

historical_schools = get_dataset_schools(historical_df)
appliance_schools = get_dataset_schools(appliance_df)
seasonal_schools = get_dataset_schools(seasonal_df)
available_schools = get_dataset_schools(historical_df, appliance_df, seasonal_df)

if not available_schools:
    st.error("No school records were found in the active datasets.")
    st.stop()

current_school = st.session_state.get("school_selection")
school_index = available_schools.index(current_school) if current_school in available_schools else 0

with st.sidebar:
    st.markdown('<div class="sidebar-section-header">ACTIVE DATASET</div>', unsafe_allow_html=True)
    school_selection = st.selectbox(
        "Institution",
        available_schools,
        index=school_index,
        key="school_selection",
        label_visibility="collapsed"
    )

target_school = school_selection
forecast_school = get_forecast_school(target_school, historical_schools)
seasonal_source_df = get_seasonal_source(seasonal_df, historical_df)

# ----------------------------------------------------
# SOOTHING MATCHA & DARK FOREST GREEN PALETTE (NO BRIGHT/NEON GREENS)
# ----------------------------------------------------
GREEN_PALETTE = ["#0B4F46", "#194D40", "#286654", "#3C826D", "#5A9E87", "#7CAF9B", "#A0CFC0", "#C8E6DC"]
GREEN_MONO_PALETTE = GREEN_PALETTE
BLUE_PALETTE = GREEN_PALETTE
BLUE_GREEN_PALETTE = GREEN_PALETTE

def apply_green_theme(fig, title=""):
    layout_kwargs = dict(
        font=dict(family="Inter", color="#6B7280"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        xaxis=dict(
            title="",
            gridcolor="#F3F4F6", 
            showline=True, 
            linecolor="#EAECF0", 
            tickfont=dict(color="#6B7280", size=11, weight="bold")
        ),
        yaxis=dict(
            title="",
            gridcolor="#F3F4F6", 
            showline=True, 
            linecolor="#EAECF0", 
            tickfont=dict(color="#6B7280", size=11, weight="bold")
        ),
        legend=dict(
            title="",
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1, 
            font=dict(color="#111827", size=11, weight="bold")
        )
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(family="Inter", size=13, color="#111827", weight="bold"), x=0, xanchor="left", y=0.98, yanchor="top")
    else:
        layout_kwargs["title"] = dict(text="")
    fig.update_layout(**layout_kwargs)
    return fig

apply_blue_theme = apply_green_theme

PAGE_HEADER_TITLES = {
    "Welcome": "WELCOME TO ENERGYSCAPE",
    "Dashboard": "ENERGYSCAPE",
    "Data Input": "DATA INPUT",
    "Season": "MULTI-SEASONAL ANALYSIS",
    "Energy L.": "ENERGY LOAD INVENTORY",
    "Forecast": "ELECTRICITY FORECAST",
    "Carbon": "CARBON FOOTPRINT AUDIT",
    "Scenario": "CONSERVATION SCENARIO MODELING",
    "Optimization": "OPTIMIZATION",
    "Impact": "IMPACT ASSESSMENT",
    "Reports": "REPORTS & COMPARATIVE ANALYSIS"
}

# ----------------------------------------------------
# TOP HEADER BAR WITH SEARCH INPUT
# ----------------------------------------------------
current_page_title = PAGE_HEADER_TITLES.get(navigation_option, "ENERGYSCAPE")
top_c1, top_c2 = st.columns([2.2, 1])
with top_c1:
    greeting_sub = '<p style="font-size: 0.88rem; color: #6B7280; margin: 0.2rem 0 0.5rem 0;">Welcome, Administrator! Real-time energy insights & carbon analytics</p>' if navigation_option == "Dashboard" else ''
    badge_html = '<span class="pill-badge-green">Decision Support System</span>' if navigation_option == "Dashboard" else ''
    st.markdown(f'<div style="margin-bottom: 0.5rem;"><div style="display: flex; align-items: center; gap: 12px;"><h1 class="page-title" style="font-size: 1.75rem; font-weight: 700; color: #111827; margin: 0; letter-spacing: -0.02em;">{current_page_title}</h1>{badge_html}</div>{greeting_sub}</div>', unsafe_allow_html=True)
with top_c2:
    search_term = st.text_input("Search", placeholder="🔍 Search dashboard metrics...", label_visibility="collapsed", key="global_search_input")
    st.session_state["global_search_term"] = search_term

def clear_search_callback():
    st.session_state["global_search_input"] = ""
    st.session_state["global_search_term"] = ""

# Render Search Active Status Banner if user enters a search term
if search_term and search_term.strip():
    apps_proc = calculate_appliance_loads(appliance_df, electricity_rate, target_school)
    system_matches = search_entire_system(search_term, historical_df, appliance_df, seasonal_df, apps_proc)
    
    c_s1, c_s2 = st.columns([4, 1])
    with c_s1:
        st.markdown(f"""
        <div class="ui-card" style="padding: 0.75rem 1.25rem !important; margin-bottom: 0.5rem !important; border-left: 4px solid #0B4F46 !important; background-color: #E8F5E9 !important;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 0.85rem; font-weight: 700; color: #1B5E20;">🔍 SYSTEM SEARCH ACTIVE:</span>
                    <span style="font-size: 0.9rem; font-weight: 800; color: #0B4F46; margin-left: 8px;">"{search_term.strip()}"</span>
                </div>
                <span style="font-size: 0.8rem; color: #1B5E20;">Scanning all 10 pages in system</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_s2:
        st.button("CLEAR SEARCH", key="btn_clear_search", on_click=clear_search_callback, use_container_width=True)
        
    if system_matches:
        st.markdown('<div style="font-size: 0.82rem; font-weight: 700; color: #111827; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em;">SYSTEM MATCHES FOUND — CLICK TO JUMP TO PAGE:</div>', unsafe_allow_html=True)
        match_cols = st.columns(len(system_matches))
        for idx, (p_name, count) in enumerate(system_matches.items()):
            with match_cols[idx]:
                btn_label = f"CURRENT: {p_name}" if p_name == navigation_option else f"GO TO: {p_name}"
                st.button(
                    btn_label,
                    key=f"btn_search_nav_{p_name}",
                    on_click=navigate_to_page,
                    args=(p_name,),
                    use_container_width=True
                )
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

# ----------------------------------------------------
# NAVIGATION VIEWS IMPLEMENTATION
# ----------------------------------------------------

hist_metrics = calculate_historical_metrics(historical_df, target_school)
apps_processed = calculate_appliance_loads(appliance_df, electricity_rate, target_school)
load_summary = get_load_summary(apps_processed, electricity_rate)
bau_base = calculate_bau_baseline(load_summary.get("total_kwh", 0.0), electricity_rate, emission_factor)
scenarios_sim = simulate_conservation_scenarios(bau_base)
opt_res = optimize_conservation_target(appliance_df=apps_processed, electricity_rate=electricity_rate, emission_factor=emission_factor)
ets_res = fit_ets_forecast(historical_df, forecast_school, forecast_horizon=forecast_horizon)

# --- 1. DASHBOARD (MAIN MOCK GRID LAYOUT) ---
if navigation_option == "Dashboard":
    # 1. CURRENT CONSUMPTION (Full-Width Hero Section)
    st.markdown(f"""
    <div class="hero-consumption-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem;">
            <div>
                <div class="hero-card-label">Baseline Operational Audit</div>
                <div class="hero-card-title" style="margin-bottom: 0 !important;">CURRENT CONSUMPTION</div>
            </div>
            <span class="pill-badge-teal" style="font-size: 0.85rem; padding: 0.35rem 0.85rem;">{target_school}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.75rem; align-items: center; background: rgba(255, 255, 255, 0.08); padding: 1.25rem 1.5rem; border-radius: 14px;">
            <div>
                <div class="hero-subtext">Monthly Energy Load</div>
                <div class="hero-metric-val">{format_kwh(load_summary.get("total_kwh", 0.0))}</div>
            </div>
            <div>
                <div class="hero-subtext">Monthly Bill (BAU)</div>
                <div class="hero-metric-val">{format_currency(bau_base["monthly_cost_php"])}</div>
            </div>
            <div>
                <div class="hero-subtext">Annual Carbon Footprint</div>
                <div class="hero-metric-val">{bau_base["annual_co2_kg"]/1000:.2f} t CO₂e</div>
            </div>
            <div>
                <div class="hero-subtext">Emission Factor Baseline</div>
                <div class="hero-metric-val" style="font-size: 1.8rem;">{emission_factor:.2f} kg/kWh</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Middle Row: PRIORITY LOAD & FORECAST (2 Columns: 1.1 : 1)
    col_p, col_f = st.columns([1.1, 1])
    
    with col_p:
        st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 0.25rem; margin-bottom: 0.85rem;">PRIORITY LOAD</h3>', unsafe_allow_html=True)
        top_apps = apps_processed.sort_values(by='monthly_kwh', ascending=False).head(3)
        
        for idx, row in top_apps.iterrows():
            st.markdown(f"""
            <div class="ui-card" style="margin-bottom: 0.85rem !important; padding: 1.1rem 1.35rem !important;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 1rem; color: #111827;">{row['appliance']}</div>
                        <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.25rem;">
                            <strong>{format_kwh(row['monthly_kwh'])}</strong> ({row['percentage_share']:.1f}% share) | {format_currency(row['monthly_cost_php'])}/mo
                        </div>
                    </div>
                    <span class="pill-badge-green" style="font-size: 0.78rem; padding: 0.3rem 0.75rem;">{row['priority']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_f:
        st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 0.25rem; margin-bottom: 0.85rem;">FORECAST SUMMARY</h3>', unsafe_allow_html=True)
        fc_df = ets_res["forecast_df"]
        avg_fc_bill = fc_df['forecast_bill'].mean()
        
        st.markdown(f"""
        <div class="ui-card" style="margin-bottom: 0.85rem !important; padding: 1.1rem 1.35rem !important;">
            <div class="kpi-label">Projected Monthly Avg ({forecast_horizon} Months)</div>
            <div class="kpi-val" style="font-size: 1.6rem; color: #0B4F46;">{format_currency(avg_fc_bill)}</div>
            <div style="margin-top: 0.4rem; font-size: 0.82rem; color: #6B7280;">
                MAPE Accuracy: <span style="color: #047857; font-weight: 800;">{ets_res['val_mape']:.2f}%</span> | RMSE: <strong>{format_currency(ets_res['val_rmse'])}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ui-card" style="margin-bottom: 0.85rem !important; padding: 1.1rem 1.35rem !important;">
            <div class="kpi-label">Confidence Interval Range</div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #047857; margin-top: 0.2rem;">{format_currency(fc_df['lower_bound'].mean())} – {format_currency(fc_df['upper_bound'].mean())}</div>
            <div style="margin-top: 0.4rem; font-size: 0.82rem; color: #6B7280;">
                Exponential Smoothing (ETS) Baseline Projection
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Bottom Row: ELECTRICITY TREND (Full-Width Section)
    st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 1.5rem; margin-bottom: 0.85rem;">ELECTRICITY TREND</h3>', unsafe_allow_html=True)
    plot_df = historical_df[historical_df['bill_php'].notna()]
    plot_df = plot_df[plot_df['school'] == target_school]
        
    fig_tr = px.line(
        plot_df, 
        x="date_dt", 
        y="bill_php", 
        color=None,
        color_discrete_sequence=BLUE_GREEN_PALETTE,
        markers=True,
        height=360
    )
    fig_tr = apply_blue_theme(fig_tr, f"Historical Monthly Electricity Expenditure — {target_school} (₱)")
    fig_tr.update_traces(line=dict(width=3, color="#0B4F46"), hovertemplate="<b>%{x|%b %Y}</b><br>Bill: ₱%{y:,.2f}<extra></extra>")
    st.plotly_chart(fig_tr, use_container_width=True)
    
    sy_min = historical_df['school_year'].min() if 'school_year' in historical_df.columns and not historical_df['school_year'].empty else "2021–2022"
    sy_max = historical_df['school_year'].max() if 'school_year' in historical_df.columns and not historical_df['school_year'].empty else "2025–2026"
    sy_cov_str = f"Coverage: SY {sy_min}–{sy_max}" if sy_min != sy_max else f"Coverage: SY {sy_min}"
    
    st.markdown(f"""
    <div class="ui-card" style="margin-top: 0.75rem; padding: 1.1rem 1.5rem !important;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="kpi-label">Historical Average Bill</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #111827;">{format_currency(hist_metrics.get("avg_bill", 0))}</div>
            </div>
            <div>
                <div class="kpi-label">Historical Peak Bill</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #0B4F46;">{format_currency(hist_metrics.get("max_bill", 0))}</div>
            </div>
            <span class="pill-badge-green" style="font-size: 0.82rem; padding: 0.35rem 0.85rem;">{sy_cov_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PROCEED BUTTON TO DATA INPUT
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_dashboard", on_click=navigate_to_page, args=("Data Input",), use_container_width=True)

# --- 2. DATA INPUT ---
elif navigation_option == "Data Input":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Electrical Billing Records & Appliance Load Inventories</p>', unsafe_allow_html=True)
    
    # Institution Badge
    st.markdown(f"""
    <div class="ui-card" style="padding: 1rem 1.25rem !important; margin-bottom: 1rem !important;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="kpi-label">SELECTED INSTITUTION</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #0F766E;">{target_school}</div>
            </div>
            <span class="pill-badge-teal">Active Dataset</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive File Drop Uploader Section matching Wireframe 2
    uploaded_file = st.file_uploader(
        "DROP FILE HERE OR CLICK TO UPLOAD",
        type=["csv"],
        key="data_input_uploader",
        help="Drag and drop or click to upload CSV electrical billing records or appliance load inventories."
    )
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            custom_df = pd.read_csv(uploaded_file)
            cols = set(str(c).lower() for c in custom_df.columns)
            if "bill_php" in cols or "date" in cols or "school_year" in cols:
                uploaded_file.seek(0)
                if st.session_state.get("uploaded_bills_filename") != uploaded_file.name:
                    st.session_state["uploaded_bills"] = uploaded_file
                    st.session_state["uploaded_bills_filename"] = uploaded_file.name
                    st.success(f"✓ Historical Billing Dataset '{uploaded_file.name}' loaded & applied system-wide! ({len(custom_df)} rows imported)")
                    st.rerun()
                else:
                    st.success(f"✓ Active Historical Billing Dataset: '{uploaded_file.name}' ({len(custom_df)} rows)")
            elif "appliance" in cols or "power_watts" in cols or "quantity" in cols:
                uploaded_file.seek(0)
                if st.session_state.get("uploaded_loads_filename") != uploaded_file.name:
                    st.session_state["uploaded_loads"] = uploaded_file
                    st.session_state["uploaded_loads_filename"] = uploaded_file.name
                    st.success(f"✓ Appliance Inventory Dataset '{uploaded_file.name}' loaded & applied system-wide! ({len(custom_df)} rows imported)")
                    st.rerun()
                else:
                    st.success(f"✓ Active Appliance Inventory Dataset: '{uploaded_file.name}' ({len(custom_df)} rows)")
            elif "consumption_kwh" in cols or "season" in cols:
                uploaded_file.seek(0)
                if st.session_state.get("uploaded_seasonal_filename") != uploaded_file.name:
                    st.session_state["uploaded_seasonal"] = uploaded_file
                    st.session_state["uploaded_seasonal_filename"] = uploaded_file.name
                    st.success(f"✓ Seasonal Consumption Dataset '{uploaded_file.name}' loaded & applied system-wide! ({len(custom_df)} rows imported)")
                    st.rerun()
                else:
                    st.success(f"✓ Active Seasonal Consumption Dataset: '{uploaded_file.name}' ({len(custom_df)} rows)")
            else:
                st.info(f"File '{uploaded_file.name}' imported ({len(custom_df)} rows).")
        except Exception as ex:
            st.error(f"Error parsing uploaded file: {ex}")
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1.25rem; margin-bottom: 0.75rem;">Appliance Electrical Load Inventory</h3>', unsafe_allow_html=True)
    app_display = appliance_df.copy()
    if 'school' in app_display.columns:
        app_display = app_display[app_display['school'] == target_school]
    if search_term:
        app_display = app_display[app_display['appliance'].str.contains(search_term, case=False)]
    render_bankio_table(app_display)

    if not seasonal_df.empty:
        st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1rem; margin-bottom: 0.75rem;">Seasonal Consumption Records</h3>', unsafe_allow_html=True)
        seasonal_display = seasonal_df.copy()
        if 'school' in seasonal_display.columns:
            seasonal_display = seasonal_display[seasonal_display['school'] == target_school]
        render_bankio_table(seasonal_display)
    
    # Data Validity Checklist Section matching Bankio Minimal Style
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1.5rem; margin-bottom: 0.75rem;">DATA VALIDITY AUDIT CHECKLIST</h3>', unsafe_allow_html=True)
    
    hist_val = validate_dataset(historical_df, "historical")
    app_val = validate_dataset(appliance_df, "appliance")
    seasonal_val = validate_dataset(seasonal_df, "seasonal")

    total_missing = int(historical_df.isna().sum().sum() + appliance_df.isna().sum().sum() + seasonal_df.isna().sum().sum())
    negative_values = int(
        (historical_df.select_dtypes(include="number") < 0).sum().sum()
        + (appliance_df.select_dtypes(include="number") < 0).sum().sum()
        + (seasonal_df.select_dtypes(include="number") < 0).sum().sum()
    )
    invalid_ranges = int(
        (appliance_df["quantity"] <= 0).sum()
        + (appliance_df["power_watts"] <= 0).sum()
        + ((appliance_df["hours_per_day"] <= 0) | (appliance_df["hours_per_day"] > 24)).sum()
        + ((appliance_df["operating_days"] <= 0) | (appliance_df["operating_days"] > 31)).sum()
    )

    expected_dates = 0
    observed_dates = 0
    for _, school_data in historical_df.groupby("school"):
        valid_dates = school_data["date_dt"].dropna().drop_duplicates()
        if not valid_dates.empty:
            expected_dates += len(pd.date_range(valid_dates.min(), valid_dates.max(), freq="MS"))
            observed_dates += len(valid_dates)
    sequence_pct = round((observed_dates / expected_dates) * 100) if expected_dates else 0

    bill_values = historical_df["bill_php"].dropna()
    if len(bill_values) >= 4:
        first_quartile = bill_values.quantile(0.25)
        third_quartile = bill_values.quantile(0.75)
        upper_fence = third_quartile + 1.5 * (third_quartile - first_quartile)
        outlier_count = int((bill_values > upper_fence).sum())
    else:
        outlier_count = 0

    missing_status = "PASS" if total_missing == 0 else "REVIEW"
    nonnegative_status = "PASS" if negative_values == 0 else "REVIEW"
    range_status = "PASS" if invalid_ranges == 0 else "REVIEW"
    sequence_status = "PASS" if sequence_pct == 100 else "REVIEW"
    outlier_status = "PASS" if outlier_count == 0 else "REVIEW"
    
    st.markdown(f"""
    <div class="ui-card" style="padding: 1.25rem 1.5rem !important;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.9rem; color: #374151;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="color: #047857; margin-right: 8px;">✓</span> NO MISSING VALUES</span>
                <span class="pill-badge-green">{missing_status} ({total_missing} Missing)</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="color: #047857; margin-right: 8px;">✓</span> NON-NEGATIVE VALUES</span>
                <span class="pill-badge-green">{nonnegative_status} ({negative_values} Invalid)</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="color: #047857; margin-right: 8px;">✓</span> VALID RANGES (Hours 1-24, Days 1-31)</span>
                <span class="pill-badge-green">{range_status} ({invalid_ranges} Invalid)</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="color: #047857; margin-right: 8px;">✓</span> COMPLETE DATE TIMESTAMPS</span>
                <span style="color: #047857; font-weight: 800;">{sequence_status} ({sequence_pct}% Sequence)</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; grid-column: span 2; padding-top: 4px;">
                <span><span style="color: #0B4F46; margin-right: 8px;">✓</span> POTENTIAL OUTLIERS</span>
                <span class="pill-badge-green">{outlier_status} ({outlier_count} Potential Peaks)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_data_input", on_click=navigate_to_page, args=("Season",), use_container_width=True)

# --- 3. SEASON ---
elif navigation_option == "Season":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.5rem;">Multi-Seasonal Load Comparison & Climate Dynamics</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<h4 style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">Season Classification Parameters</h4>', unsafe_allow_html=True)
        dry_months = st.multiselect(
            "Select Dry Season Months",
            options=MONTH_NAMES,
            default=DEFAULT_DRY_MONTHS,
            label_visibility="collapsed"
        )
        wet_months = [m for m in MONTH_NAMES if m not in dry_months]
        
    s_metrics = calculate_seasonal_metrics(
        seasonal_source_df, 
        dry_months, 
        wet_months, 
        school_name=target_school
    )
    
    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)
    
    col_sea_left, col_sea_right = st.columns([1.6, 1])
    
    with col_sea_left:
        monthly_summary, season_value_col, season_value_label = build_monthly_season_summary(seasonal_source_df, target_school)
        format_season_value = format_kwh if season_value_col == "consumption_kwh" else format_currency
        
        # Filter monthly_summary dynamically based on chosen months in dry_months dropdown
        if dry_months:
            filtered_summary = monthly_summary[monthly_summary['full_month_name'].isin(dry_months)].copy()
            if filtered_summary.empty:
                filtered_summary = monthly_summary.copy()
        else:
            filtered_summary = monthly_summary.copy()
            
        overall_mean = filtered_summary[season_value_col].mean() if not filtered_summary.empty else 1.0
        filtered_summary['seasonal_index'] = filtered_summary[season_value_col] / overall_mean if overall_mean > 0 else 1.0

        # Determine Peak and Lowest Period dynamically from chosen months
        if not filtered_summary.empty:
            max_idx = filtered_summary[season_value_col].idxmax()
            min_idx = filtered_summary[season_value_col].idxmin()
            
            peak_month_str = filtered_summary.loc[max_idx, 'full_month_name'].upper()
            peak_val = filtered_summary.loc[max_idx, season_value_col]
            peak_s_index = filtered_summary.loc[max_idx, 'seasonal_index']
            lowest_month_str = filtered_summary.loc[min_idx, 'full_month_name'].upper()
            lowest_val = filtered_summary.loc[min_idx, season_value_col]
        else:
            peak_month_str = "N/A"
            peak_val = 0.0
            peak_s_index = 1.0
            lowest_month_str = "N/A"
            lowest_val = 0.0

        peak_pct_str = f"({(peak_s_index - 1.0) * 100:+.0f}% Peak)" if peak_s_index != 1.0 else "(Baseline)"
        
        # Bankio Minimal Green Palette: Emerald for Peak Month, Dark Teal for other Chosen Months
        peak_full = peak_month_str.title()
        bar_colors = ["#0B4F46" if m == peak_full else "#286654" for m in filtered_summary['full_month_name']]
        
        st.markdown(f'<h3 style="font-size: 1.05rem; font-weight: 700; color: #111827; margin-bottom: 0.75rem;">Filtered Monthly {season_value_label} & Seasonal Index Trend — {target_school}</h3>', unsafe_allow_html=True)
        fig_sea = go.Figure()
        fig_sea.add_trace(go.Bar(
            x=filtered_summary['month_name'],
            y=filtered_summary[season_value_col],
            name=season_value_label,
            marker_color=bar_colors,
            hovertemplate=f"Month: %{{x}}<br>{season_value_label}: %{{y:,.2f}}<extra></extra>"
        ))
        fig_sea.add_trace(go.Scatter(
            x=filtered_summary['month_name'],
            y=filtered_summary['seasonal_index'] * overall_mean,
            name="Seasonal Trend Index",
            mode="lines+markers",
            line=dict(color="#047857", width=3.5),
            hovertemplate="Month: %{x}<br>Index: %{text:.2f}<extra></extra>",
            text=filtered_summary['seasonal_index']
        ))
        fig_sea = apply_blue_theme(fig_sea, "")
        fig_sea.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_sea, use_container_width=True)
        
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        
        # Bottom Left Metrics - NOW DYNAMIC
        st.markdown(f"""
        <div class="ui-card" style="padding: 1rem 1.25rem !important;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                <div>
                    <div class="kpi-label">PEAK PERIOD</div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #0B4F46;">{peak_month_str}</div>
                </div>
                <div>
                    <div class="kpi-label">SEASONAL INDEX</div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #111827;">{peak_s_index:.2f} {peak_pct_str}</div>
                </div>
                <div>
                    <div class="kpi-label">LOWEST PERIOD</div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #047857;">{lowest_month_str}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sea_right:
        dry_avg = s_metrics.get("dry_avg", 0.0)
        wet_avg = s_metrics.get("wet_avg", 0.0)
        pct_diff = s_metrics.get("percentage_difference", 0.0)
        
        if dry_avg >= wet_avg:
            variance_html = f"Dry Season monthly average (<strong>{format_season_value(dry_avg)}</strong>) exceeds Wet Season baseline (<strong>{format_season_value(wet_avg)}</strong>) by approximately <strong>{abs(pct_diff):.2f}%</strong>."
        else:
            variance_html = f"Wet Season monthly average (<strong>{format_season_value(wet_avg)}</strong>) exceeds Dry Season baseline (<strong>{format_season_value(dry_avg)}</strong>) by approximately <strong>{abs(pct_diff):.2f}%</strong>."
            
        dry_months_formatted = ", ".join(dry_months) if dry_months else "None selected"

        st.markdown(f"""
        <div class="ui-card" style="height: 100%; min-height: 420px;">
            <h3 style="font-size: 1.1rem; font-weight: 800; color: #111827; margin-bottom: 0.75rem;">INTERPRETATION:</h3>
            <p style="font-size: 0.88rem; color: #374151; line-height: 1.6; margin-bottom: 0.75rem;">
                <strong>Seasonal Load Peak:</strong> Electricity use peaks during <strong>{peak_month_str.title()}</strong> (Average: <strong>{format_season_value(peak_val)}</strong>) driven by institutional load demands and seasonal climate variations.
            </p>
            <p style="font-size: 0.88rem; color: #374151; line-height: 1.6; margin-bottom: 0.75rem;">
                <strong>Seasonal Variance:</strong> {variance_html}
            </p>
            <p style="font-size: 0.88rem; color: #374151; line-height: 1.6; margin: 0;">
                <strong>Operational Action:</strong> Targeted energy conservation and load duty-cycle management during classified Dry season months (<em>{dry_months_formatted}</em>) and peak period <strong>{peak_month_str.title()}</strong> offers maximum potential for load curtailment.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # PROCEED BUTTON
    st.markdown('<div style="margin-top: 1.75rem;"></div>', unsafe_allow_html=True)
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_season", on_click=navigate_to_page, args=("Energy L.",), use_container_width=True)

# --- 4. ENERGY L. ---
elif navigation_option == "Energy L.":
    st.markdown('<p style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.25rem;">Appliance Electrical Load & Consumption Breakdown</p>', unsafe_allow_html=True)
    
    # Horizontal Bar Chart for Appliance Load Characterization
    apps_filtered = filter_dataframe_by_search(apps_processed, search_term)
    apps_chart_df = apps_filtered.sort_values(by='monthly_kwh', ascending=True)
    
    # Soothing Matcha & Dark Forest Green Palette (No Bright/Neon Greens)
    green_bar_palette = ["#063B34", "#0B4F46", "#194D40", "#286654", "#3C826D", "#5A9E87", "#7CAF9B", "#A0CFC0"]
    
    fig_hbar = px.bar(
        apps_chart_df,
        y='appliance',
        x='monthly_kwh',
        orientation='h',
        color='appliance',
        text='monthly_kwh',
        color_discrete_sequence=green_bar_palette,
        height=380
    )
    fig_hbar = apply_blue_theme(fig_hbar, f"Appliance Monthly Energy Load Breakdown — {target_school} (kWh/month)")
    fig_hbar.update_traces(texttemplate='%{text:,.1f} kWh', textposition='outside', hovertemplate="<b>%{y}</b><br>Monthly Load: %{x:,.2f} kWh<extra></extra>")
    fig_hbar.update_layout(showlegend=False)
    st.plotly_chart(fig_hbar, use_container_width=True)
    
    # Bottom Metrics Card matching Bankio Style
    top2_text = f"{load_summary['top_appliance']} + {load_summary['second_appliance']}" if load_summary.get('second_appliance') else load_summary.get('top_appliance', 'Primary Load')
    st.markdown(f"""
    <div class="ui-card" style="margin-top: 0.5rem; padding: 1.25rem 1.5rem !important;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <div class="kpi-label">CONTRIBUTION</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #111827;">{load_summary['top2_combined_share']:.1f}% ESTIMATED</div>
                <div style="font-size: 0.78rem; color: #6B7280;">Top 2 Combined Load ({top2_text})</div>
            </div>
            <div>
                <div class="kpi-label">CONSUMPTION</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0B4F46;">{load_summary['total_kwh']:.2f} KWH</div>
                <div style="font-size: 0.78rem; color: #6B7280;">Total Campus Baseline Monthly Load</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1rem; margin-bottom: 0.75rem;">Appliance Load Inventory Matrix</h3>', unsafe_allow_html=True)
    if search_term:
        apps_view = apps_processed[apps_processed['appliance'].str.contains(search_term, case=False)]
    else:
        apps_view = apps_processed
    render_bankio_table(apps_view[['appliance', 'quantity', 'power_watts', 'hours_per_day', 'operating_days', 'monthly_kwh', 'percentage_share', 'monthly_cost_php', 'priority']])

    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_energy_l", on_click=navigate_to_page, args=("Forecast",), use_container_width=True)

# --- 5. FORECAST ---
elif navigation_option == "Forecast":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Holt-Winters Exponential Smoothing (ETS) Projections</p>', unsafe_allow_html=True)
    
    fc_df = ets_res["forecast_df"]
    
    # 3-Line Forecast Chart matching Bankio Green Theme
    fig_fc_line = go.Figure()
    fig_fc_line.add_trace(go.Scatter(
        x=fc_df['date_str'],
        y=fc_df['lower_bound'],
        name="Lower Confidence Bound",
        mode="lines",
        line=dict(color="rgba(11, 79, 70, 0.3)", width=1, dash="dash"),
        hovertemplate="Lower Bound: ₱%{y:,.2f}<extra></extra>",
        showlegend=False
    ))
    fig_fc_line.add_trace(go.Scatter(
        x=fc_df['date_str'],
        y=fc_df['upper_bound'],
        name="95% Confidence Interval Band",
        mode="lines",
        fill='tonexty',
        fillcolor="rgba(11, 79, 70, 0.12)",
        line=dict(color="rgba(11, 79, 70, 0.3)", width=1, dash="dash"),
        hovertemplate="Upper Bound: ₱%{y:,.2f}<extra></extra>"
    ))
    fig_fc_line.add_trace(go.Scatter(
        x=fc_df['date_str'],
        y=fc_df['forecast_bill'],
        name="ETS Forecasted Bill (₱)",
        mode="lines+markers",
        line=dict(color="#0B4F46", width=3.5),
        marker=dict(size=7, color="#063B34"),
        hovertemplate="<b>%{x}</b><br>Forecast: ₱%{y:,.2f}<extra></extra>"
    ))
    fig_fc_line = apply_blue_theme(fig_fc_line, f"Forecasted Electricity Bills — {target_school} ({forecast_horizon} Months)")
    fig_fc_line.update_layout(height=360)
    st.plotly_chart(fig_fc_line, use_container_width=True)
    
    # Calculate MAE & Annual Metrics
    mae_val = ets_res.get("val_mae", 0.0)
    ann_kwh = (fc_df['forecast_bill'].sum() / electricity_rate)
    lower_ann_kwh = (fc_df['lower_bound'].sum() / electricity_rate)
    upper_ann_kwh = (fc_df['upper_bound'].sum() / electricity_rate)
    
    # Bottom Metrics Card matching Bankio Style
    st.markdown(f"""
    <div class="ui-card" style="margin-top: 0.5rem; padding: 1.25rem 1.5rem !important;">
        <div style="display: grid; grid-template-columns: 1fr 1.4fr; gap: 24px;">
            <div>
                <div class="kpi-label">MODEL PERFORMANCE</div>
                <div style="font-size: 0.9rem; color: #374151; line-height: 1.6; margin-top: 0.4rem;">
                    <strong>MAE:</strong> {format_currency(mae_val)}<br>
                    <strong>RMSE:</strong> {format_currency(ets_res["val_rmse"])}<br>
                    <strong>MAPE:</strong> <span style="color: #047857; font-weight: 800;">{ets_res["val_mape"]:.2f}%</span> ({interpret_mape(ets_res["val_mape"])})
                </div>
            </div>
            <div>
                <div class="kpi-label">FORECASTED ANNUAL CONSUMPTION</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #111827; margin-top: 0.2rem;">{ann_kwh:,.0f} KWH ({format_currency(fc_df['forecast_bill'].sum())})</div>
                <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.4rem;">
                    <strong>PREDICTION INTERVAL:</strong> {lower_ann_kwh:,.0f} KWH – {upper_ann_kwh:,.0f} KWH
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1rem; margin-bottom: 0.75rem;">Projected Monthly Expenditure Table</h3>', unsafe_allow_html=True)
    render_bankio_table(fc_df[['date_str', 'month', 'forecast_bill', 'lower_bound', 'upper_bound']])

    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_forecast", on_click=navigate_to_page, args=("Carbon",), use_container_width=True)

# --- 6. CARBON ---
elif navigation_option == "Carbon":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Scope 2 Carbon Footprint Quantification & Projections</p>', unsafe_allow_html=True)
    
    bau = calculate_bau_baseline(load_summary.get("total_kwh", 0.0), electricity_rate, emission_factor)
    fc_df = ets_res["forecast_df"]
    fc_annual_kwh = (fc_df['forecast_bill'].sum() / electricity_rate)
    fc_annual_co2 = fc_annual_kwh * emission_factor
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">BASELINE</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #111827; margin-top: 0.3rem;">{bau['monthly_co2_kg']:,.2f} kg CO₂e</div>
            <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.4rem;">
                Monthly Baseline Footprint ({bau['annual_co2_kg']/1000:.2f} t CO₂e / Year)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_c2:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">FORECAST</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #0B4F46; margin-top: 0.3rem;">{(fc_annual_co2/12):,.2f} kg CO₂e</div>
            <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.4rem;">
                Projected Monthly Average ({fc_annual_co2/1000:.2f} t CO₂e / Year)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Projected Annual CO2 Banner
    st.markdown(f"""
    <div class="ui-card" style="margin-top: 0.5rem; padding: 1.25rem 1.5rem !important;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #111827;">Projected Annual CO₂</div>
                <div style="font-size: 0.82rem; color: #6B7280;">Calculated with Grid Emission Factor = {emission_factor:.2f} kg CO₂e/kWh</div>
            </div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #0B4F46;">{bau['annual_co2_kg']:,.0f} kg CO₂e</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    fc_df['forecast_co2_kg'] = (fc_df['forecast_bill'] / electricity_rate) * emission_factor
    fig_carbon = px.bar(
        fc_df,
        x='month',
        y='forecast_co2_kg',
        color_discrete_sequence=["#0B4F46"],
        text='forecast_co2_kg',
        height=340
    )
    fig_carbon = apply_blue_theme(fig_carbon, f"Projected Monthly Scope 2 Carbon Footprint — {target_school} (kg CO₂e)")
    fig_carbon.update_traces(texttemplate='%{text:,.1f} kg', textposition='outside', hovertemplate="Month: %{x}<br>Carbon Footprint: %{y:,.2f} kg CO₂e<extra></extra>")
    st.plotly_chart(fig_carbon, use_container_width=True)
        
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #0F172A; margin-top: 1rem; margin-bottom: 0.75rem;">Business-as-Usual (BAU) Benchmark Table</h3>', unsafe_allow_html=True)
    bau_table = pd.DataFrame([
        {"Indicator": "Monthly Electricity Consumption", "Value": format_kwh(bau["monthly_kwh"])},
        {"Indicator": "Annual Electricity Consumption", "Value": format_kwh(bau["annual_kwh"])},
        {"Indicator": "Monthly Electricity Cost", "Value": format_currency(bau["monthly_cost_php"])},
        {"Indicator": "Annual Electricity Cost", "Value": format_currency(bau["annual_cost_php"])},
        {"Indicator": "Monthly Carbon Emissions", "Value": format_co2(bau["monthly_co2_kg"])},
        {"Indicator": "Annual Carbon Emissions", "Value": format_co2(bau["annual_co2_kg"])},
    ])
    render_bankio_table(bau_table)

    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_carbon", on_click=navigate_to_page, args=("Scenario",), use_container_width=True)

# --- 7. SCENARIO ---
elif navigation_option == "Scenario":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Adjust Appliance Duty-Cycles & Simulate Energy Savings Scenarios</p>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 0.25rem; margin-bottom: 0.75rem;">ADJUST INTERVENTION LEVELS:</h3>', unsafe_allow_html=True)
    
    top_load_rows = apps_processed.sort_values(by='monthly_kwh', ascending=False).head(2)
    top1_name = load_summary.get("top_appliance", "Primary Load")
    top2_name = load_summary.get("second_appliance") or "Secondary Load"
    col_sl1, col_sl2, col_sl3 = st.columns(3)
    with col_sl1:
        top1_red = st.slider(f"{top1_name} Intervention (%)", min_value=0, max_value=100, value=15, step=5)
    with col_sl2:
        top2_red = st.slider(f"{top2_name} Intervention (%)", min_value=0, max_value=100, value=15, step=5)
    with col_sl3:
        other_red = st.slider("Remaining Loads (%)", min_value=0, max_value=100, value=10, step=5)
        
    top1_share = float(top_load_rows.iloc[0]['percentage_share'] / 100.0) if len(top_load_rows) >= 1 else 0.0
    top2_share = float(top_load_rows.iloc[1]['percentage_share'] / 100.0) if len(top_load_rows) >= 2 else 0.0
    other_share = max(0.0, 1.0 - top1_share - top2_share)

    avg_red_pct = (top1_red * top1_share + top2_red * top2_share + other_red * other_share)
    
    col_sim1, col_sim2, col_sim3 = st.columns([1, 1.5, 1])
    with col_sim2:
        run_sim = st.button("SIMULATE SCENARIO", key="btn_run_sim", use_container_width=True)
        
    base_kwh = load_summary.get("total_kwh", 0.0)
    sim_kwh = base_kwh * (1.0 - (avg_red_pct / 100.0))
    kwh_saved = base_kwh - sim_kwh
    cost_saved_m = kwh_saved * electricity_rate
    cost_saved_y = cost_saved_m * 12
    co2_avoided_m = kwh_saved * emission_factor
    co2_avoided_y = co2_avoided_m * 12
    
    if run_sim:
        st.toast(f"⚡ Scenario simulated ({top1_red:.0f}% {top1_name}, {top2_red:.0f}% {top2_name}, {other_red:.0f}% Other loads)!")
        st.markdown(f"""
        <div style="background-color: #E6F4EA; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.75rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #047857; font-size: 0.98rem;">SIMULATION APPLIED</div>
                <div style="font-size: 0.85rem; color: #065F46;">Projected Load: <b>{sim_kwh:,.2f} kWh/month</b> ({avg_red_pct:.1f}% Energy Reduction). Annual Savings: <b>{format_currency(cost_saved_y)}</b></div>
            </div>
            <span class="pill-badge-green">SIMULATED</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 1.5rem; margin-bottom: 0.75rem;">SCENARIO RESULT</h3>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="ui-card" style="padding: 1.25rem 1.5rem !important;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
            <div>
                <div class="kpi-label">BASELINE SCENARIO</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #111827;">{base_kwh:,.2f} KWH</div>
                <div style="font-size: 0.78rem; color: #6B7280;">Monthly Baseline</div>
            </div>
            <div>
                <div class="kpi-label">PROJECTED SCENARIO</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #0B4F46;">{sim_kwh:,.2f} KWH</div>
                <div style="font-size: 0.78rem; color: #6B7280;">Simulated Monthly Target</div>
            </div>
            <div>
                <div class="kpi-label">ENERGY SAVED REDUCTION</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #047857;">{kwh_saved:,.2f} KWH</div>
                <div style="font-size: 0.78rem; color: #047857; font-weight: 700;">{avg_red_pct:.2f}% REDUCTION</div>
            </div>
            <div>
                <div class="kpi-label">COST SAVED & CO₂ AVOIDED</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0B4F46;">{format_currency(cost_saved_m)}/mo</div>
                <div style="font-size: 0.78rem; color: #047857; font-weight: 700;">{co2_avoided_m:,.1f} KG CO₂e / month</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    active_reduction_rate = avg_red_pct / 100.0
    scenarios_df = simulate_conservation_scenarios(bau_base, [active_reduction_rate])
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1rem; margin-bottom: 0.75rem;">Simulated Conservation Scenarios Comparison</h3>', unsafe_allow_html=True)
    render_bankio_table(scenarios_df)
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        fig_sc_kwh = px.bar(scenarios_df, x="Scenario", y="Projected Monthly kWh", color="Scenario", text="Projected Monthly kWh", color_discrete_sequence=GREEN_PALETTE, height=320)
        fig_sc_kwh = apply_blue_theme(fig_sc_kwh, f"Simulated Monthly Load — {target_school}")
        fig_sc_kwh.update_traces(texttemplate='%{text:,.1f} kWh', textposition='outside', hovertemplate="Scenario: %{x}<br>Projected Load: %{y:,.2f} kWh<extra></extra>")
        st.plotly_chart(fig_sc_kwh, use_container_width=True)
    with col_sc2:
        fig_sc_co2 = px.bar(scenarios_df, x="Scenario", y="Annual Avoided CO₂e (kg)", color="Scenario", text="Annual Avoided CO₂e (kg)", color_discrete_sequence=GREEN_PALETTE, height=320)
        fig_sc_co2 = apply_blue_theme(fig_sc_co2, f"Avoided Annual CO₂ Emissions — {target_school}")
        fig_sc_co2.update_traces(texttemplate='%{text:,.1f} kg', textposition='outside', hovertemplate="Scenario: %{x}<br>Avoided CO₂: %{y:,.2f} kg<extra></extra>")
        st.plotly_chart(fig_sc_co2, use_container_width=True)

    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_scenario", on_click=navigate_to_page, args=("Optimization",), use_container_width=True)

# --- 8. OPTIMIZATION ---
elif navigation_option == "Optimization":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Linear Goal Programming Optimization & Operational Constraints</p>', unsafe_allow_html=True)
    constraint_top1 = load_summary.get("top_appliance", "Primary Load")
    constraint_top2 = load_summary.get("second_appliance") or "Secondary Load"
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 0.25rem; margin-bottom: 0.75rem;">OBJECTIVE FUNCTION</h3>', unsafe_allow_html=True)
    opt_goal = st.selectbox(
        "Optimization Objective",
        [
            "MINIMIZE ELECTRICITY + COST + CO₂",
            "MINIMIZE ELECTRICITY LOAD (kWh)",
            "MINIMIZE OPERATIONAL EXPENDITURE (₱)",
            "MINIMIZE GREENHOUSE GAS EMISSIONS (kg CO₂e)"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1rem; margin-bottom: 0.75rem;">OPERATIONAL CONSTRAINTS:</h3>', unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        max_ac_limit = st.slider(f"Max {constraint_top1} Reduction (%)", min_value=5, max_value=35, value=15, step=1, key="opt_limit_ac")
    with col_c2:
        max_comp_limit = st.slider(f"Max {constraint_top2} Reduction (%)", min_value=5, max_value=35, value=15, step=1, key="opt_limit_comp")
    with col_c3:
        max_other_limit = st.slider("Max Remaining Loads Reduction (%)", min_value=1, max_value=25, value=10, step=1, key="opt_limit_other")

    # Run Scipy Linear Programming solver dynamically on constraints and selected objective
    opt_res = optimize_conservation_target(
        appliance_df=apps_processed,
        electricity_rate=electricity_rate,
        emission_factor=emission_factor,
        max_ac_red=max_ac_limit / 100.0,
        max_comp_red=max_comp_limit / 100.0,
        max_other_red=max_other_limit / 100.0,
        objective=opt_goal
    )
        
    col_r1, col_r2, col_r3 = st.columns([1, 1.5, 1])
    with col_r2:
        run_opt = st.button("RUN OPTIMIZATION", key="btn_run_opt", use_container_width=True)
        
    if run_opt:
        st.toast(f"⚡ Linear Goal Programming Optimization executed for {target_school}!")
        st.markdown(f"""
        <div style="background-color: #E6F4EA; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.75rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #047857; font-size: 0.98rem;">OPTIMIZATION COMPLETE (LINEAR GOAL PROGRAMMING)</div>
                <div style="font-size: 0.85rem; color: #065F46;">Optimal Target: <b>{format_kwh(opt_res['optimized_monthly_kwh'])}</b> ({opt_res['reduction_percentage']:.1f}% Reduction). Annual Savings: <b>{format_currency(opt_res['annual_cost_savings_php'])}</b></div>
            </div>
            <span class="pill-badge-green">OPTIMIZED</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)
        
    op1, op2, op3, op4 = st.columns(4)
    with op1:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Optimal Strategy</div>
            <div class="kpi-val" style="font-size: 1.2rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{opt_res["selected_scenario"]}</div>
            <div style="font-size: 0.78rem; color: #6B7280;">{opt_res.get("strategy_focus", "Linear Goal Programming")}</div>
        </div>
        """, unsafe_allow_html=True)
    with op2:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Optimized Target</div>
            <div class="kpi-val">{format_kwh(opt_res["optimized_monthly_kwh"])}</div>
            <div style="font-size: 0.78rem; color: #6B7280;">Monthly target load</div>
        </div>
        """, unsafe_allow_html=True)
    with op3:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Annual Cost Savings</div>
            <div class="kpi-val">{format_currency(opt_res["annual_cost_savings_php"])}</div>
            <div style="font-size: 0.78rem; color: #6B7280;">Financial budget relief</div>
        </div>
        """, unsafe_allow_html=True)
    with op4:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Annual Avoided CO₂</div>
            <div class="kpi-val">{format_co2(opt_res["annual_avoided_co2_kg"])}</div>
            <div style="font-size: 0.78rem; color: #6B7280;">Greenhouse reduction</div>
        </div>
        """, unsafe_allow_html=True)
        
    opt_chart_df = pd.DataFrame([
        {"Strategy": "BAU Baseline", "Monthly Load (kWh)": opt_res["bau_monthly_kwh"]},
        {"Strategy": "Optimized Target", "Monthly Load (kWh)": opt_res["optimized_monthly_kwh"]}
    ])
    fig_opt_bar = px.bar(
        opt_chart_df,
        x="Strategy",
        y="Monthly Load (kWh)",
        color="Strategy",
        text="Monthly Load (kWh)",
        color_discrete_sequence=["#286654", "#0B4F46"],
        height=280
    )
    fig_opt_bar = apply_blue_theme(fig_opt_bar, f"Baseline vs Linear Goal Programming Target — {target_school}")
    fig_opt_bar.update_traces(texttemplate='%{text:,.1f} kWh', textposition='outside', hovertemplate="Strategy: %{x}<br>Monthly Load: %{y:,.2f} kWh<extra></extra>")
    fig_opt_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_opt_bar, use_container_width=True)

    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 0.5rem; margin-bottom: 0.75rem;">Optimized Baseline vs Target Comparison Table</h3>', unsafe_allow_html=True)
    opt_table = pd.DataFrame([
        {"Indicator": "Monthly Electricity Consumption", "BAU/Current": format_kwh(opt_res["bau_monthly_kwh"]), "Optimized Target": format_kwh(opt_res["optimized_monthly_kwh"]), "Reduction": format_kwh(opt_res["monthly_kwh_savings"])},
        {"Indicator": "Annual Electricity Consumption", "BAU/Current": format_kwh(opt_res["bau_monthly_kwh"] * 12), "Optimized Target": format_kwh(opt_res["optimized_monthly_kwh"] * 12), "Reduction": format_kwh(opt_res["annual_kwh_savings"])},
        {"Indicator": "Monthly Electricity Cost", "BAU/Current": format_currency(opt_res["bau_monthly_kwh"] * electricity_rate), "Optimized Target": format_currency(opt_res["optimized_monthly_kwh"] * electricity_rate), "Reduction": format_currency(opt_res["monthly_cost_savings_php"])},
        {"Indicator": "Annual Electricity Cost", "BAU/Current": format_currency(opt_res["bau_monthly_kwh"] * 12 * electricity_rate), "Optimized Target": format_currency(opt_res["optimized_monthly_kwh"] * 12 * electricity_rate), "Reduction": format_currency(opt_res["annual_cost_savings_php"])},
        {"Indicator": "Reduction Percentage", "BAU/Current": "0%", "Optimized Target": f"{opt_res['reduction_percentage']:.0f}%", "Reduction": f"{opt_res['reduction_percentage']:.0f}%"}
    ])
    render_bankio_table(opt_table)
    
    st.markdown('<h4 style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-top: 1.25rem; margin-bottom: 0.25rem;">Operational Target Monitor Input</h4>', unsafe_allow_html=True)
    actual_default = float(round(load_summary.get("total_kwh", 0.0), 2))
    target_default = float(round(opt_res.get("optimized_monthly_kwh", 0.0), 2))
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        actual_input = st.number_input("Actual Monthly Electricity Consumption (kWh)", min_value=0.0, max_value=20000.0, value=actual_default, step=25.0)
    with col_t2:
        target_input = st.number_input("Target Consumption Benchmark (kWh)", min_value=0.0, max_value=20000.0, value=target_default, step=25.0)
        
    mon_res = monitor_target_consumption(actual_input, target_input)
    if mon_res["is_on_target"]:
        st.markdown(f"""
        <div style="background-color: #E6F4EA; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1.25rem 1.5rem; display: flex; align-items: center; justify-content: space-between; margin-top: 0.5rem; margin-bottom: 1.25rem;">
            <div>
                <h4 style="color: #047857; font-size: 1.05rem; font-weight: 700; margin: 0;">STATUS: COMPLIANT WITH ENERGY TARGET</h4>
                <p style="color: #047857; font-size: 0.88rem; margin: 0.25rem 0 0 0;">Actual consumption ({format_kwh(mon_res['actual_kwh'])}) is below target ceiling ({format_kwh(mon_res['target_kwh'])}).</p>
            </div>
            <span class="pill-badge-green" style="font-size: 0.95rem; padding: 0.4rem 1rem;">COMPLIANT</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 12px; padding: 1.25rem 1.5rem; display: flex; align-items: center; justify-content: space-between; margin-top: 0.5rem; margin-bottom: 1.25rem;">
            <div>
                <h4 style="color: #991B1B; font-size: 1.05rem; font-weight: 700; margin: 0;">STATUS: EXCEEDS ENERGY TARGET (ACTION REQUIRED)</h4>
                <p style="color: #B91C1C; font-size: 0.88rem; margin: 0.25rem 0 0 0;">Actual consumption ({format_kwh(mon_res['actual_kwh'])}) exceeds target benchmark by {format_kwh(mon_res['difference_kwh'])}.</p>
            </div>
            <span class="pill-badge-red" style="font-size: 0.95rem; padding: 0.4rem 1rem;">ACTION REQUIRED</span>
        </div>
        """, unsafe_allow_html=True)

    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_opt", on_click=navigate_to_page, args=("Impact",), use_container_width=True)

# --- 9. IMPACT ---
elif navigation_option == "Impact":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Institutional Impact, Baseline vs. Optimized Savings, & Sensitivity Elasticity</p>', unsafe_allow_html=True)
    
    kwh_savings_annual = opt_res["annual_kwh_savings"]
    cost_savings_annual = opt_res["annual_cost_savings_php"]
    co2_savings_annual = opt_res["annual_avoided_co2_kg"]
    
    col_imp1, col_imp2, col_imp3 = st.columns(3)
    with col_imp1:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Energy kWh Saved</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #047857; margin-top: 0.3rem;">{kwh_savings_annual:,.2f} KWH</div>
            <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.4rem;">
                Annual Saved Energy ({opt_res['monthly_kwh_savings']:,.2f} kWh/mo)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_imp2:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Cost ₱ Saved</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #111827; margin-top: 0.3rem;">{format_currency(cost_savings_annual)}</div>
            <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.4rem;">
                Annual Budget Relief ({format_currency(opt_res['monthly_cost_savings_php'])}/mo)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_imp3:
        st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Carbon kg Avoided</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #0B4F46; margin-top: 0.3rem;">{co2_savings_annual:,.2f} KG</div>
            <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.4rem;">
                Annual CO₂e Reduction ({opt_res['monthly_kwh_savings']*emission_factor:,.1f} kg/mo)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 1rem; margin-bottom: 0.75rem;">BASELINE VS OPTIMIZED IMPACT TABLE</h3>', unsafe_allow_html=True)
    imp_table = pd.DataFrame([
        {"Impact Category": "Monthly Electricity (kWh)", "Baseline (BAU)": format_kwh(opt_res["bau_monthly_kwh"]), "Optimized": format_kwh(opt_res["optimized_monthly_kwh"]), "Annual Net Impact": f"-{kwh_savings_annual:,.2f} kWh Saved/Year"},
        {"Impact Category": "Monthly Expenditure (₱)", "Baseline (BAU)": format_currency(opt_res["bau_monthly_kwh"] * electricity_rate), "Optimized": format_currency(opt_res["optimized_monthly_kwh"] * electricity_rate), "Annual Net Impact": f"-{format_currency(cost_savings_annual)} Saved/Year"},
        {"Impact Category": "Monthly Carbon Footprint (kg CO₂e)", "Baseline (BAU)": f"{opt_res['bau_monthly_kwh']*emission_factor:,.2f} kg", "Optimized": f"{opt_res['optimized_monthly_kwh']*emission_factor:,.2f} kg", "Annual Net Impact": f"-{co2_savings_annual:,.2f} kg CO₂e Avoided/Year"},
    ])
    render_bankio_table(imp_table)
    
    m_an = calculate_historical_metrics(historical_df, "An-anaao Integrated School")
    m_lp = calculate_historical_metrics(historical_df, "La Paz Integrated School")
    fc_an = fit_ets_forecast(historical_df, "An-anaao Integrated School")
    fc_lp = fit_ets_forecast(historical_df, "La Paz Integrated School")
    
    fc_an_mean = fc_an["forecast_df"]["forecast_bill"].mean()
    fc_lp_mean = fc_lp["forecast_df"]["forecast_bill"].mean()
    
    comp_df = pd.DataFrame([
        {"Indicator": "Historical Total Bills (₱)", "An-anaao Integrated School": format_currency(m_an.get("total_bill")), "La Paz Integrated School": format_currency(m_lp.get("total_bill")), "Difference": format_currency(m_lp.get("total_bill", 0) - m_an.get("total_bill", 0))},
        {"Indicator": "Historical Avg Monthly Bill (₱)", "An-anaao Integrated School": format_currency(m_an.get("avg_bill")), "La Paz Integrated School": format_currency(m_lp.get("avg_bill")), "Difference": format_currency(m_lp.get("avg_bill", 0) - m_an.get("avg_bill", 0))},
        {"Indicator": "Highest Historical Bill (₱)", "An-anaao Integrated School": format_currency(m_an.get("max_bill")), "La Paz Integrated School": format_currency(m_lp.get("max_bill")), "Difference": format_currency(m_lp.get("max_bill", 0) - m_an.get("max_bill", 0))},
        {"Indicator": "Average Forecasted Bill (₱)", "An-anaao Integrated School": format_currency(fc_an_mean), "La Paz Integrated School": format_currency(fc_lp_mean), "Difference": format_currency(fc_lp_mean - fc_an_mean)},
    ])
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1.25rem; margin-bottom: 0.75rem;">Comparative School Benchmark Matrix</h3>', unsafe_allow_html=True)
    render_bankio_table(comp_df)
    
    sens_df = calculate_sensitivity_analysis(bau_base["monthly_kwh"])
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 1.25rem; margin-bottom: 0.75rem;">Sensitivity Ratios & Rate Elasticity Table</h3>', unsafe_allow_html=True)
    render_bankio_table(sens_df)

    # PROCEED BUTTON
    col_proc1, col_proc2, col_proc3 = st.columns([1, 1.5, 1])
    with col_proc2:
        st.button("PROCEED →", key="btn_proceed_impact", on_click=navigate_to_page, args=("Reports",), use_container_width=True)

# --- 10. REPORTS ---
elif navigation_option == "Reports":
    st.markdown('<p style="font-size: 0.88rem; color: #6B7280; margin-bottom: 1.25rem;">Multi-Institutional Comparative Benchmark & Executive Summary Report Generator</p>', unsafe_allow_html=True)
    
    # 1. COMPARATIVE ANALYSIS SECTION (Wireframe 1)
    st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 0.25rem; margin-bottom: 0.75rem;">COMPARATIVE ANALYSIS</h3>', unsafe_allow_html=True)
    
    apps_an = calculate_appliance_loads(appliance_df, electricity_rate, "An-anaao Integrated School")
    apps_lp = calculate_appliance_loads(appliance_df, electricity_rate, "La Paz Integrated School")
    
    sum_an = get_load_summary(apps_an, electricity_rate)
    sum_lp = get_load_summary(apps_lp, electricity_rate)
    
    s_an = calculate_seasonal_metrics(historical_df, DEFAULT_DRY_MONTHS, DEFAULT_WET_MONTHS, "An-anaao Integrated School")
    s_lp = calculate_seasonal_metrics(historical_df, DEFAULT_DRY_MONTHS, DEFAULT_WET_MONTHS, "La Paz Integrated School")
    
    bau_an = calculate_bau_baseline(sum_an.get("total_kwh", 0), electricity_rate, emission_factor)
    bau_lp = calculate_bau_baseline(sum_lp.get("total_kwh", 0), electricity_rate, emission_factor)
    
    scen_an = simulate_conservation_scenarios(bau_an)
    scen_lp = simulate_conservation_scenarios(bau_lp)
    
    opt_an = optimize_conservation_target(appliance_df=apps_an, electricity_rate=electricity_rate, emission_factor=emission_factor)
    opt_lp = optimize_conservation_target(appliance_df=apps_lp, electricity_rate=electricity_rate, emission_factor=emission_factor)
    
    fc_an = fit_ets_forecast(historical_df, "An-anaao Integrated School")
    fc_lp = fit_ets_forecast(historical_df, "La Paz Integrated School")
    fc_an_df = fc_an["forecast_df"]
    fc_lp_df = fc_lp["forecast_df"]
    
    peak_an_str = "December (Peak)" if not s_an.get("monthly_averages") else f"{max(s_an['monthly_averages'], key=s_an['monthly_averages'].get)}"
    peak_lp_str = "December (Peak)" if not s_lp.get("monthly_averages") else f"{max(s_lp['monthly_averages'], key=s_lp['monthly_averages'].get)}"
    
    kwh_diff = sum_lp.get("total_kwh", 0) - sum_an.get("total_kwh", 0)
    top_an_str = f"{sum_an.get('top_appliance', 'Air Conditioner')} ({sum_an.get('top_kwh', 0):,.0f} kWh)"
    top_lp_str = f"{sum_lp.get('top_appliance', 'Air Conditioner')} ({sum_lp.get('top_kwh', 0):,.0f} kWh)"
    
    comp_analysis_df = pd.DataFrame([
        {"Metric": "AVERAGE KWH", "An-anaao Integrated School": f"{sum_an.get('total_kwh', 0):,.2f} kWh / mo", "La Paz Integrated School": f"{sum_lp.get('total_kwh', 0):,.2f} kWh / mo", "Variance": f"{kwh_diff:+,.2f} kWh / mo"},
        {"Metric": "PEAK SEASON", "An-anaao Integrated School": peak_an_str, "La Paz Integrated School": peak_lp_str, "Variance": "Seasonal Peak Variant"},
        {"Metric": "HIGHEST LOAD", "An-anaao Integrated School": top_an_str, "La Paz Integrated School": top_lp_str, "Variance": "Dominant Appliance Load"},
        {"Metric": "FORECAST (Avg Bill)", "An-anaao Integrated School": format_currency(fc_an_df['forecast_bill'].mean()), "La Paz Integrated School": format_currency(fc_lp_df['forecast_bill'].mean()), "Variance": format_currency(fc_lp_df['forecast_bill'].mean() - fc_an_df['forecast_bill'].mean())},
        {"Metric": "FORECAST (Interval)", "An-anaao Integrated School": f"{format_currency(fc_an_df['lower_bound'].mean())} - {format_currency(fc_an_df['upper_bound'].mean())}", "La Paz Integrated School": f"{format_currency(fc_lp_df['lower_bound'].mean())} - {format_currency(fc_lp_df['upper_bound'].mean())}", "Variance": "Baseline Confidence Range"},
        {"Metric": "MAPE ACCURACY", "An-anaao Integrated School": f"{fc_an['val_mape']:.2f}% ({interpret_mape(fc_an['val_mape'])})", "La Paz Integrated School": f"{fc_lp['val_mape']:.2f}% ({interpret_mape(fc_lp['val_mape'])})", "Variance": f"{fc_lp['val_mape'] - fc_an['val_mape']:+.2f}% Error Diff"},
        {"Metric": "OPTIMIZED TARGET", "An-anaao Integrated School": f"{opt_an['optimized_monthly_kwh']:,.2f} kWh / mo", "La Paz Integrated School": f"{opt_lp['optimized_monthly_kwh']:,.2f} kWh / mo", "Variance": f"{opt_an['reduction_percentage']:.0f}% Target Ceiling"},
        {"Metric": "ENERGY REDUCTION", "An-anaao Integrated School": f"{opt_an['monthly_kwh_savings']:,.2f} kWh / mo ({opt_an['reduction_percentage']:.0f}%)", "La Paz Integrated School": f"{opt_lp['monthly_kwh_savings']:,.2f} kWh / mo ({opt_lp['reduction_percentage']:.0f}%)", "Variance": "Multi-tier Duty Cycle"},
        {"Metric": "CO₂ REDUCTION", "An-anaao Integrated School": f"{opt_an['annual_avoided_co2_kg']:,.2f} kg CO₂e / yr", "La Paz Integrated School": f"{opt_lp['annual_avoided_co2_kg']:,.2f} kg CO₂e / yr", "Variance": "Scope 2 Emission Avoidance"},
    ])
    render_bankio_table(comp_analysis_df)
    
    col_r1, col_r2, col_r3 = st.columns([1, 1.5, 1])
    with col_r2:
        run_comp = st.button("RUN COMPARATIVE ANALYSIS", key="btn_run_comp", use_container_width=True)
        
    if run_comp:
        st.toast("⚡ Comparative Analysis completed across all institutional datasets!")
        st.markdown("""
        <div style="background-color: #E6F4EA; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.75rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #047857; font-size: 0.98rem;">COMPARATIVE ANALYSIS EXECUTED</div>
                <div style="font-size: 0.85rem; color: #065F46;">Institutional benchmark matrices updated for An-anaao Integrated School vs La Paz Integrated School.</div>
            </div>
            <span class="pill-badge-green">UPDATED</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='margin: 2rem 0; border: 0; border-top: 1px solid #EAECF0;'>", unsafe_allow_html=True)
    
    # 2. EXECUTIVE REPORT SECTION (Wireframe 2)
    st.markdown('<h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 0.75rem;">EXECUTIVE REPORT</h3>', unsafe_allow_html=True)
    
    report_school = st.selectbox("Selected School", ["An-anaao Integrated School", "La Paz Integrated School"], index=0, key="report_school_select")
    
    # Dynamic calculations for Executive Report Box
    rep_apps = calculate_appliance_loads(appliance_df, electricity_rate, report_school)
    rep_sum = get_load_summary(rep_apps, electricity_rate)
    rep_s = calculate_seasonal_metrics(historical_df, DEFAULT_DRY_MONTHS, DEFAULT_WET_MONTHS, report_school)
    rep_bau = calculate_bau_baseline(rep_sum.get("total_kwh", 0), electricity_rate, emission_factor)
    rep_scen = simulate_conservation_scenarios(rep_bau)
    rep_opt = optimize_conservation_target(appliance_df=rep_apps, electricity_rate=electricity_rate, emission_factor=emission_factor)
    rep_fc = fit_ets_forecast(historical_df, report_school)
    rep_fc_df = rep_fc["forecast_df"]
    
    rep_peak_month = "December" if not rep_s.get("monthly_averages") else max(rep_s['monthly_averages'], key=rep_s['monthly_averages'].get)
    rep_peak_idx = (max(rep_s['monthly_averages'].values()) / rep_s['overall_avg']) if rep_s.get('overall_avg', 0) > 0 else 1.21
    
    report_html = f"""<div class="ui-card" style="padding: 1.5rem 1.75rem !important;">
<div style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">
School: <span style="color: #0B4F46;">{report_school}</span>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem;">
<div style="background: #F9FAFB; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #EAECF0;">
<div class="kpi-label">SEASONAL FINDING</div>
<div style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-top: 0.3rem;">
Peak consumption: <span style="color: #0B4F46;">{rep_peak_month} (Seasonal Index: {rep_peak_idx:.2f})</span>
</div>
</div>
<div style="background: #F9FAFB; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #EAECF0;">
<div class="kpi-label">ENERGY LOAD</div>
<div style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-top: 0.3rem;">
Priority load: <span style="color: #0B4F46;">{rep_sum['top_appliance']} ({rep_sum['top_kwh']:,.2f} kWh/mo, {rep_sum['top_share']:.1f}% share)</span>
</div>
</div>
<div style="background: #F9FAFB; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #EAECF0;">
<div class="kpi-label">FORECAST</div>
<div style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-top: 0.3rem;">
Expected consumption: <span style="color: #0B4F46;">{format_currency(rep_fc_df['forecast_bill'].mean())} / month (MAPE: {rep_fc['val_mape']:.2f}%)</span>
</div>
</div>
<div style="background: #F9FAFB; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #EAECF0;">
<div class="kpi-label">CARBON</div>
<div style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-top: 0.3rem;">
Projected emissions: <span style="color: #0B4F46;">{rep_bau['annual_co2_kg']:,.2f} kg CO₂e / year</span>
</div>
</div>
<div style="background: #F9FAFB; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #EAECF0; grid-column: span 2;">
<div class="kpi-label">OPTIMIZATION</div>
<div style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-top: 0.3rem;">
Recommended strategy: <span style="color: #0B4F46;">{rep_opt['selected_scenario']} ({format_kwh(rep_opt['optimized_monthly_kwh'])} target)</span>
</div>
</div>
<div style="background: #E6F4EA; padding: 1.2rem 1.25rem; border-radius: 12px; border: 1px solid #A7F3D0; grid-column: span 2;">
<div class="kpi-label" style="color: #047857 !important;">IMPACT & SAVINGS</div>
<div style="font-size: 0.95rem; font-weight: 700; color: #047857; margin-top: 0.4rem; line-height: 1.6;">
Energy saved: <strong>{format_kwh(rep_opt['annual_kwh_savings'])} / year</strong><br>
Cost saved: <strong>{format_currency(rep_opt['annual_cost_savings_php'])} / year</strong><br>
CO₂ avoided: <strong>{format_co2(rep_opt['annual_avoided_co2_kg'])} / year</strong>
</div>
</div>
</div>
</div>"""
    st.markdown(report_html, unsafe_allow_html=True)
    
    col_gen1, col_gen2, col_gen3 = st.columns([1, 1.5, 1])
    with col_gen2:
        gen_rep = st.button("GENERATE REPORT", key="btn_gen_rep", use_container_width=True)
        
    if gen_rep:
        st.success(f"🎉 Executive Energy Audit Report for '{report_school}' successfully generated and ready for export!")
        
    st.markdown("<hr style='margin: 2rem 0; border: 0; border-top: 1px solid #EAECF0;'>", unsafe_allow_html=True)
    
    # 3. COMPUTATIONAL CONSISTENCY AUDIT
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-bottom: 0.75rem;">Systemic Computational Consistency Audit</h3>', unsafe_allow_html=True)
    val_table = verify_computational_consistency(bau_base["monthly_kwh"], electricity_rate)
    render_bankio_table(val_table)
