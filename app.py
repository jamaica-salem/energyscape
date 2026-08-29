"""
ENERGYSCAPE: Multi-Seasonal Mathematical-Computational Framework for Predictive Energy Management and Carbon Reduction
Main Streamlit Application — Custom Light Blue Design System (Clean Typography without Emojis)
"""

import streamlit as st
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
# COMPREHENSIVE LIGHT BLUE THEME CSS FIXES
# ----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Body & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Remove ugly black Streamlit header & decoration */
    header[data-testid="stHeader"] {
        background-color: #F8FAFC !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Headings Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
    }

    /* Top Greeting Banner */
    .user-greeting-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
    }
    .greeting-title {
        font-size: 0.95rem !important;
        color: #64748B !important;
        margin: 0 !important;
    }
    .greeting-name {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
    }

    /* Card Containers */
    .ui-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04) !important;
        margin-bottom: 1.5rem !important;
    }

    /* Dark Featured Savings Card (Deep Blue Theme) */
    .dark-featured-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%) !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px -4px rgba(30, 58, 138, 0.35) !important;
        margin-bottom: 1.5rem !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .dark-card-label {
        font-size: 0.85rem !important;
        color: #93C5FD !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .dark-card-value {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0.25rem 0 !important;
        letter-spacing: -0.02em !important;
    }
    .dark-card-subtitle {
        font-size: 0.85rem !important;
        color: #BFDBFE !important;
        margin-top: 0.5rem !important;
    }

    /* Pill Badges */
    .pill-badge-blue {
        background-color: #DBEAFE !important;
        color: #1E40AF !important;
        padding: 0.3rem 0.75rem !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    .pill-badge-green {
        background-color: #DCFCE7 !important;
        color: #166534 !important;
        padding: 0.3rem 0.75rem !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    .pill-badge-red {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        padding: 0.3rem 0.75rem !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        display: inline-flex !important;
        align-items: center !important;
    }

    /* Metric Display */
    .kpi-label {
        font-size: 0.85rem !important;
        color: #64748B !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem !important;
    }
    .kpi-val {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }

    /* Sidebar Fixes */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 800;
        color: #1E3A8A !important;
        margin-bottom: 1.5rem;
    }
    .sidebar-section-header {
        font-size: 0.75rem !important;
        font-weight: 800 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Radio button / Navigation list fix */
    div[role="radiogroup"] label {
        background-color: transparent !important;
        color: #334155 !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease !important;
    }
    div[role="radiogroup"] label:hover {
        background-color: #F1F5F9 !important;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        font-weight: 800 !important;
    }

    /* Widget Labels & Inputs Fix */
    label[data-testid="stWidgetLabel"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    input[type="number"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }

    /* Legend Row */
    .legend-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.875rem;
    }
    .legend-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS (NO EMOJIS)
# ----------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ <span>ENERGYSCAPE</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-header">MAIN MENU</div>', unsafe_allow_html=True)
    navigation_option = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Historical Analysis",
            "Seasonal Analysis",
            "Energy Load Analysis",
            "Forecasting",
            "Carbon & BAU",
            "Conservation Scenarios",
            "Optimization",
            "School Comparison",
            "Target Monitor",
            "Validation & Sensitivity",
            "Methodology"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="sidebar-section-header">SETTINGS & PARAMETERS</div>', unsafe_allow_html=True)
    
    school_selection = st.selectbox(
        "Selected Institution",
        ["An-anaao Integrated School", "La Paz Integrated School", "Both"]
    )
    
    electricity_rate = st.number_input(
        "Electricity Rate (₱/kWh)",
        min_value=1.0, max_value=50.0, value=11.00, step=0.50
    )
    
    emission_factor = st.number_input(
        "Emission Factor (kg CO₂e/kWh)",
        min_value=0.10, max_value=2.00, value=0.70, step=0.05
    )
    
    forecast_horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=3, max_value=24, value=12, step=1
    )
    
    st.markdown('<div class="sidebar-section-header">DATA SOURCE</div>', unsafe_allow_html=True)
    use_project_dataset = st.checkbox("Use Project Datasets", value=True)
    
    uploaded_bills = None
    uploaded_loads = None
    if not use_project_dataset:
        uploaded_bills = st.file_uploader("Upload Historical Bills CSV", type=["csv"])
        uploaded_loads = st.file_uploader("Upload Appliance Loads CSV", type=["csv"])

# ----------------------------------------------------
# DATA INGESTION
# ----------------------------------------------------
try:
    historical_df = load_historical_bills(uploaded_bills)
    appliance_df = load_appliance_loads(uploaded_loads)
    seasonal_df = load_seasonal_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# ----------------------------------------------------
# BLUE THEME PLOTLY STYLING HELPER
# ----------------------------------------------------
BLUE_PALETTE = ["#0F4C81", "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]

def apply_blue_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Plus Jakarta Sans", size=15, color="#0F172A")),
        font=dict(family="Plus Jakarta Sans", color="#475569"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            title="",
            gridcolor="#F1F5F9", 
            showline=True, 
            linecolor="#E2E8F0", 
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            title="",
            gridcolor="#F1F5F9", 
            showline=True, 
            linecolor="#E2E8F0", 
            tickfont=dict(color="#475569")
        ),
        legend=dict(
            title="",
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1, 
            font=dict(color="#0F172A")
        )
    )
    return fig

# ----------------------------------------------------
# MODULE IMPLEMENTATIONS
# ----------------------------------------------------

# --- 1. DASHBOARD ---
if navigation_option == "Dashboard":
    st.markdown("""
    <div class="user-greeting-banner">
        <div>
            <p class="greeting-title">Welcome back, Energy Administrator!</p>
            <h1 class="greeting-name">Energy Insights Dashboard</h1>
        </div>
        <div>
            <span class="pill-badge-blue">Abra, Philippines</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    target_school = "An-anaao Integrated School" if school_selection == "Both" else school_selection
    hist_metrics = calculate_historical_metrics(historical_df, target_school)
    apps_processed = calculate_appliance_loads(appliance_df, electricity_rate, target_school)
    load_summary = get_load_summary(apps_processed, electricity_rate)
    
    bau_base = calculate_bau_baseline(load_summary.get("total_kwh", 2289.10), electricity_rate, emission_factor)
    scenarios_sim = simulate_conservation_scenarios(bau_base)
    opt_res = optimize_conservation_target(scenarios_sim)
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns([1, 1, 1.2])
    
    with col_kpi1:
        st.markdown(f"""
        <div class="ui-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="kpi-label">Historical Average Bill</div>
                    <div class="kpi-val">{format_currency(hist_metrics.get("avg_bill", 0))}</div>
                </div>
                <span class="pill-badge-blue">Last 4 SY</span>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748B;">
                Range: {format_currency(hist_metrics.get("min_bill", 0))} – {format_currency(hist_metrics.get("max_bill", 0))}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ui-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="kpi-label">BAU Monthly Cost</div>
                    <div class="kpi-val">{format_currency(bau_base["monthly_cost_php"])}</div>
                </div>
                <span class="pill-badge-red">Baseline</span>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748B;">
                Estimated Load: {format_kwh(load_summary.get("total_kwh", 0))}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="ui-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="kpi-label">Primary Load Share</div>
                    <div class="kpi-val">{load_summary.get("top_share", 0):.1f}%</div>
                </div>
                <span class="pill-badge-blue">{load_summary.get("top_appliance", "Air Conditioner")}</span>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748B;">
                Top 2 Combined: {load_summary.get("top2_combined_share", 0):.1f}% of total load
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ui-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="kpi-label">BAU Carbon Footprint</div>
                    <div class="kpi-val">{format_co2(bau_base["monthly_co2_kg"])}</div>
                </div>
                <span class="pill-badge-blue">Monthly</span>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748B;">
                Annual: {format_co2(bau_base["annual_co2_kg"])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="dark-featured-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="dark-card-label">Optimized Target Savings</span>
                <span class="pill-badge-green">15% Reduction</span>
            </div>
            <div class="dark-card-value">{format_kwh(opt_res["optimized_monthly_kwh"])}</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #60A5FA;">
                Annual Cost Savings: {format_currency(opt_res["annual_cost_savings_php"])}
            </div>
            <div class="dark-card-subtitle">
                Avoided Carbon: <strong>{format_co2(opt_res["annual_avoided_co2_kg"])}</strong>/year
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_chart_left, col_chart_right = st.columns([1, 1.2])
    
    with col_chart_left:
        st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #0F172A; margin-bottom: 0.75rem;">Appliance Load Category Breakdown</h3>', unsafe_allow_html=True)
        
        if not apps_processed.empty:
            fig_donut = px.pie(
                apps_processed, names="appliance", values="monthly_kwh",
                hole=0.6, color_discrete_sequence=BLUE_PALETTE,
                height=280
            )
            fig_donut.update_traces(textinfo="percent", hoverinfo="label+value+percent")
            fig_donut.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=10),
                annotations=[dict(text="Energy Load", x=0.5, y=0.5, font=dict(size=14, family="Plus Jakarta Sans", color="#0F172A"), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
            st.markdown('<div style="margin-top: 0.5rem;">', unsafe_allow_html=True)
            for idx, row in apps_processed.head(4).iterrows():
                color_dot = BLUE_PALETTE[idx % len(BLUE_PALETTE)]
                st.markdown(f"""
                <div class="legend-row">
                    <div class="legend-left">
                        <div class="legend-dot" style="background-color: {color_dot};"></div>
                        <span style="color: #0F172A; font-weight: 600;">{row['appliance']}</span>
                    </div>
                    <div>
                        <strong style="color: #0F172A;">{format_kwh(row['monthly_kwh'])}</strong> 
                        <span style="color: #64748B; margin-left: 6px;">({row['percentage_share']:.1f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col_chart_right:
        st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; color: #0F172A; margin-bottom: 0.75rem;">Monthly Electricity Billing Patterns</h3>', unsafe_allow_html=True)
        
        plot_df = historical_df[historical_df['bill_php'].notna()]
        if school_selection != "Both":
            plot_df = plot_df[plot_df['school'] == school_selection]
            
        fig_bar = px.bar(
            plot_df.tail(12), x="month", y="bill_php", color="school",
            color_discrete_sequence=["#1D4ED8", "#60A5FA"],
            height=380
        )
        fig_bar = apply_blue_theme(fig_bar)
        fig_bar.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig_bar, use_container_width=True)

    exec_rec = generate_executive_summary_recommendation(
        load_summary.get("top_appliance", "Air Conditioner"),
        load_summary.get("top_share", 34.60),
        opt_res["optimized_monthly_kwh"]
    )
    st.markdown(f"""
    <div class="ui-card" style="border-left: 6px solid #1D4ED8; margin-top: 1.5rem;">
        <h3 style="font-size: 1.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem;">Executive Management Recommendation</h3>
        <p style="font-size: 0.95rem; color: #334155; line-height: 1.6; margin: 0;">{exec_rec}</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. HISTORICAL ANALYSIS ---
elif navigation_option == "Historical Analysis":
    st.title("Historical Electricity Bill Analysis")
    st.write("Multi-year monthly electricity billing records in Philippine Pesos (₱).")
    
    metrics = calculate_historical_metrics(historical_df, school_selection)
    
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Historical Expenditure", format_currency(metrics["total_bill"]))
        c2.metric("Average Monthly Bill", format_currency(metrics["avg_bill"]))
        c3.metric("Highest Monthly Bill", format_currency(metrics["max_bill"]))
        c4.metric("Lowest Monthly Bill", format_currency(metrics["min_bill"]))
        
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Monthly Trend Line", "Yearly Totals", "Monthly Averages"])
        
        with tab1:
            plot_df = historical_df[historical_df['bill_php'].notna()]
            if school_selection != "Both":
                plot_df = plot_df[plot_df['school'] == school_selection]
            fig = px.line(plot_df, x="date", y="bill_php", color="school", markers=True, color_discrete_sequence=["#1D4ED8", "#3B82F6"])
            fig = apply_blue_theme(fig, "Monthly Bill Trend")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.dataframe(metrics["yearly_summary"], use_container_width=True)
            
        with tab3:
            st.dataframe(metrics["monthly_summary"], use_container_width=True)

# --- 3. SEASONAL ANALYSIS ---
elif navigation_option == "Seasonal Analysis":
    st.title("Seasonal Consumption Analysis")
    st.write("Dry (Dec–May) vs Wet (Jun–Nov) seasonal consumption patterns.")
    
    dry_config = st.multiselect("Dry Season Months Configuration", options=[
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
    ], default=DEFAULT_DRY_MONTHS)
    
    wet_config = [m for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"] if m not in dry_config]
    s_metrics = calculate_seasonal_metrics(seasonal_df, dry_config, wet_config, school_selection)
    
    if s_metrics:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Dry Season Avg", format_kwh(s_metrics["dry_avg"]))
        sc2.metric("Wet Season Avg", format_kwh(s_metrics["wet_avg"]))
        sc3.metric("Seasonal Difference", format_kwh(s_metrics["seasonal_difference"]))
        sc4.metric("Percentage Difference", format_pct(s_metrics["percentage_difference"]))
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_s = px.bar(seasonal_df, x="month", y="consumption_kwh", color="season", color_discrete_map={"Dry": "#1D4ED8", "Wet": "#93C5FD"})
            fig_s = apply_blue_theme(fig_s, "Monthly Seasonal Energy Consumption (kWh)")
            st.plotly_chart(fig_s, use_container_width=True)
            
        with col_s2:
            s_idx_df = pd.DataFrame(list(s_metrics["seasonal_indices"].items()), columns=["Month", "Seasonal Index"])
            fig_idx = px.line(s_idx_df, x="Month", y="Seasonal Index", markers=True, color_discrete_sequence=["#1D4ED8"])
            fig_idx.add_hline(y=1.0, line_dash="dash", line_color="#94A3B8")
            fig_idx = apply_blue_theme(fig_idx, "Monthly Seasonal Index")
            st.plotly_chart(fig_idx, use_container_width=True)

# --- 4. ENERGY LOAD ANALYSIS ---
elif navigation_option == "Energy Load Analysis":
    st.title("Electrical Load Quantification")
    st.write("Appliance energy consumption, monthly financial cost, and priority ranking.")
    
    apps_processed = calculate_appliance_loads(appliance_df, electricity_rate, school_selection)
    load_sum = get_load_summary(apps_processed, electricity_rate)
    
    if not apps_processed.empty:
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Total Monthly Load", format_kwh(load_sum["total_kwh"]))
        lc2.metric("Total Monthly Cost", format_currency(load_sum["total_cost_php"]))
        lc3.metric("Top Appliance", load_sum["top_appliance"])
        lc4.metric("Top 2 Share", format_pct(load_sum["top2_combined_share"]))
        
        st.markdown("### Ranked Appliance Load Inventory")
        st.dataframe(apps_processed[['rank', 'appliance', 'quantity', 'power_watts', 'hours_per_day', 'operating_days', 'monthly_kwh', 'monthly_cost_php', 'percentage_share', 'priority']], use_container_width=True)
        
        st.markdown("### Pareto Energy Concentration")
        apps_sorted = apps_processed.sort_values(by='monthly_kwh', ascending=False)
        apps_sorted['cum_share'] = apps_sorted['percentage_share'].cumsum()
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=apps_sorted['appliance'], y=apps_sorted['monthly_kwh'], name="Monthly kWh", marker_color="#1D4ED8"))
        fig_p.add_trace(go.Scatter(x=apps_sorted['appliance'], y=apps_sorted['cum_share'], name="Cumulative Share (%)", yaxis="y2", mode="lines+markers", line=dict(color="#2563EB", width=3)))
        
        fig_p.update_layout(
            title=dict(text="Pareto Appliance Load Analysis", font=dict(color="#0F172A")),
            yaxis=dict(title="Energy (kWh)", gridcolor="#F1F5F9", tickfont=dict(color="#475569")),
            yaxis2=dict(title="Cumulative Share (%)", overlaying="y", side="right", range=[0, 105], tickfont=dict(color="#475569")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#0F172A"))
        )
        st.plotly_chart(fig_p, use_container_width=True)

# --- 5. FORECASTING ---
elif navigation_option == "Forecasting":
    st.title("Exponential Smoothing (ETS) Forecasting")
    st.write("Point forecasts and confidence intervals computed via ExponentialSmoothing.")
    
    target_school = "An-anaao Integrated School" if school_selection == "Both" else school_selection
    fc_res = fit_ets_forecast(historical_df, target_school, forecast_horizon=forecast_horizon)
    
    hist_df = historical_df[historical_df['school'] == target_school].dropna(subset=['bill_php']).sort_values(by='date_dt')
    fc_df = fc_res["forecast_df"]
    
    val_mape = calculate_mape(fc_res["val_actuals"], fc_res["val_predictions"])
    val_rmse = calculate_rmse(fc_res["val_actuals"], fc_res["val_predictions"])
    
    f1, f2, f3 = st.columns(3)
    f1.metric("Calculated MAPE", format_pct(val_mape))
    f2.metric("Calculated RMSE", format_currency(val_rmse))
    f3.metric("Forecast Quality", interpret_mape(val_mape))
    
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist_df['date_dt'], y=hist_df['bill_php'], mode='lines+markers', name='Historical Bill (₱)', line=dict(color='#1D4ED8', width=2)))
    fig_fc.add_trace(go.Scatter(x=fc_df['date_dt'], y=fc_df['forecast_bill'], mode='lines+markers', name='ETS Forecast (₱)', line=dict(color='#2563EB', width=3, dash='dash')))
    fig_fc.add_trace(go.Scatter(x=fc_df['date_dt'], y=fc_df['upper_bound'], mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
    fig_fc.add_trace(go.Scatter(x=fc_df['date_dt'], y=fc_df['lower_bound'], mode='lines', name='95% Interval', fill='tonexty', fillcolor='rgba(37, 99, 235, 0.15)', line=dict(width=0)))
    
    fig_fc = apply_blue_theme(fig_fc, f"Electricity Bill Forecast for {target_school}")
    st.plotly_chart(fig_fc, use_container_width=True)
    
    st.dataframe(fc_df[['date_str', 'month', 'forecast_bill', 'lower_bound', 'upper_bound']], use_container_width=True)

# --- 6. CARBON & BAU ---
elif navigation_option == "Carbon & BAU":
    st.title("Carbon Footprint & BAU Baseline")
    st.write("Baseline Business-as-Usual metrics and greenhouse gas emissions.")
    
    apps_processed = calculate_appliance_loads(appliance_df, electricity_rate, school_selection)
    load_sum = get_load_summary(apps_processed, electricity_rate)
    bau = calculate_bau_baseline(load_sum.get("total_kwh", 2289.10), electricity_rate, emission_factor)
    
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("BAU Monthly kWh", format_kwh(bau["monthly_kwh"]))
    b2.metric("BAU Monthly Cost", format_currency(bau["monthly_cost_php"]))
    b3.metric("BAU Monthly CO₂e", format_co2(bau["monthly_co2_kg"]))
    b4.metric("BAU Annual CO₂e", format_co2(bau["annual_co2_kg"]))
    
    bau_table = pd.DataFrame([
        {"Indicator": "Monthly Electricity Consumption", "Value": format_kwh(bau["monthly_kwh"])},
        {"Indicator": "Annual Electricity Consumption", "Value": format_kwh(bau["annual_kwh"])},
        {"Indicator": "Monthly Electricity Cost", "Value": format_currency(bau["monthly_cost_php"])},
        {"Indicator": "Annual Electricity Cost", "Value": format_currency(bau["annual_cost_php"])},
        {"Indicator": "Monthly Carbon Emissions", "Value": format_co2(bau["monthly_co2_kg"])},
        {"Indicator": "Annual Carbon Emissions", "Value": format_co2(bau["annual_co2_kg"])},
    ])
    st.dataframe(bau_table, use_container_width=True)

# --- 7. CONSERVATION SCENARIOS ---
elif navigation_option == "Conservation Scenarios":
    st.title("Conservation Scenarios (5%, 10%, 15%)")
    st.write("Simulated scenario comparison against Business-as-Usual.")
    
    apps_processed = calculate_appliance_loads(appliance_df, electricity_rate, school_selection)
    load_sum = get_load_summary(apps_processed, electricity_rate)
    bau = calculate_bau_baseline(load_sum.get("total_kwh", 2289.10), electricity_rate, emission_factor)
    scenarios_df = simulate_conservation_scenarios(bau)
    
    st.dataframe(scenarios_df, use_container_width=True)
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        fig_sc_kwh = px.bar(scenarios_df, x="Scenario", y="Projected Monthly kWh", color="Scenario", color_discrete_sequence=["#0F4C81", "#1D4ED8", "#2563EB", "#3B82F6"])
        fig_sc_kwh = apply_blue_theme(fig_sc_kwh, "Projected Monthly Consumption (kWh)")
        st.plotly_chart(fig_sc_kwh, use_container_width=True)
        
    with col_sc2:
        fig_sc_co2 = px.bar(scenarios_df, x="Scenario", y="Annual Avoided CO₂e (kg)", color="Scenario", color_discrete_sequence=["#0F4C81", "#1D4ED8", "#2563EB", "#3B82F6"])
        fig_sc_co2 = apply_blue_theme(fig_sc_co2, "Annual Avoided CO₂ Emissions (kg)")
        st.plotly_chart(fig_sc_co2, use_container_width=True)

# --- 8. OPTIMIZATION ---
elif navigation_option == "Optimization":
    st.title("Mathematical Optimization Target")
    st.write("Feasible energy reduction target under operational constraints.")
    
    apps_processed = calculate_appliance_loads(appliance_df, electricity_rate, school_selection)
    load_sum = get_load_summary(apps_processed, electricity_rate)
    bau = calculate_bau_baseline(load_sum.get("total_kwh", 2289.10), electricity_rate, emission_factor)
    scenarios_df = simulate_conservation_scenarios(bau)
    opt_res = optimize_conservation_target(scenarios_df)
    
    op1, op2, op3, op4 = st.columns(4)
    op1.metric("Optimal Strategy", opt_res["selected_scenario"])
    op2.metric("Optimized Target", format_kwh(opt_res["optimized_monthly_kwh"]))
    op3.metric("Annual Cost Savings", format_currency(opt_res["annual_cost_savings_php"]))
    op4.metric("Annual Avoided CO₂", format_co2(opt_res["annual_avoided_co2_kg"]))
    
    opt_table = pd.DataFrame([
        {"Indicator": "Monthly Electricity Consumption", "BAU/Current": format_kwh(opt_res["bau_monthly_kwh"]), "Optimized Target": format_kwh(opt_res["optimized_monthly_kwh"]), "Reduction": format_kwh(opt_res["monthly_kwh_savings"])},
        {"Indicator": "Annual Electricity Consumption", "BAU/Current": format_kwh(opt_res["bau_monthly_kwh"] * 12), "Optimized Target": format_kwh(opt_res["optimized_monthly_kwh"] * 12), "Reduction": format_kwh(opt_res["annual_kwh_savings"])},
        {"Indicator": "Monthly Electricity Cost", "BAU/Current": format_currency(opt_res["bau_monthly_kwh"] * electricity_rate), "Optimized Target": format_currency(opt_res["optimized_monthly_kwh"] * electricity_rate), "Reduction": format_currency(opt_res["monthly_cost_savings_php"])},
        {"Indicator": "Annual Electricity Cost", "BAU/Current": format_currency(opt_res["bau_monthly_kwh"] * 12 * electricity_rate), "Optimized Target": format_currency(opt_res["optimized_monthly_kwh"] * 12 * electricity_rate), "Reduction": format_currency(opt_res["annual_cost_savings_php"])},
        {"Indicator": "Reduction Percentage", "BAU/Current": "0%", "Optimized Target": f"{opt_res['reduction_percentage']:.0f}%", "Reduction": f"{opt_res['reduction_percentage']:.0f}%"}
    ])
    st.dataframe(opt_table, use_container_width=True)

# --- 9. SCHOOL COMPARISON ---
elif navigation_option == "School Comparison":
    st.title("Comparative School Analysis")
    st.write("Direct computational comparison between An-anaao and La Paz Integrated Schools.")
    
    m_an = calculate_historical_metrics(historical_df, "An-anaao Integrated School")
    m_lp = calculate_historical_metrics(historical_df, "La Paz Integrated School")
    fc_an = fit_ets_forecast(historical_df, "An-anaao Integrated School")
    fc_lp = fit_ets_forecast(historical_df, "La Paz Integrated School")
    
    comp_df = pd.DataFrame([
        {"Indicator": "Historical Total Bills (₱)", "An-anaao Integrated School": format_currency(m_an.get("total_bill")), "La Paz Integrated School": format_currency(m_lp.get("total_bill")), "Difference": format_currency(m_lp.get("total_bill", 0) - m_an.get("total_bill", 0))},
        {"Indicator": "Historical Avg Monthly Bill (₱)", "An-anaao Integrated School": format_currency(m_an.get("avg_bill")), "La Paz Integrated School": format_currency(m_lp.get("avg_bill")), "Difference": format_currency(m_lp.get("avg_bill", 0) - m_an.get("avg_bill", 0))},
        {"Indicator": "Highest Historical Bill (₱)", "An-anaao Integrated School": format_currency(m_an.get("max_bill")), "La Paz Integrated School": format_currency(m_lp.get("max_bill")), "Difference": format_currency(m_lp.get("max_bill", 0) - m_an.get("max_bill", 0))},
        {"Indicator": "Average Forecasted Bill (₱)", "An-anaao Integrated School": format_currency(fc_an["forecast_df"]["forecast_bill"].mean()), "La Paz Integrated School": format_currency(fc_lp["forecast_df"]["forecast_bill"].mean()), "Difference": format_currency(fc_lp["forecast_df"]["forecast_bill"].mean() - fc_an["forecast_df"]["forecast_bill"].mean())},
    ])
    st.dataframe(comp_df, use_container_width=True)

# --- 10. TARGET MONITOR ---
elif navigation_option == "Target Monitor":
    st.title("Electricity Target Monitor")
    st.write("Interactive decision-support comparator for actual vs target monthly consumption.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        actual_input = st.number_input("Actual Monthly Electricity Consumption (kWh)", min_value=0.0, max_value=10000.0, value=1800.0, step=25.0)
    with col_t2:
        target_input = st.number_input("Target Consumption Benchmark (kWh)", min_value=0.0, max_value=10000.0, value=1945.74, step=25.0)
        
    mon_res = monitor_target_consumption(actual_input, target_input)
    
    tm1, tm2, tm3 = st.columns(3)
    tm1.metric("Actual Consumption", format_kwh(mon_res["actual_kwh"]))
    tm2.metric("Target Benchmark", format_kwh(mon_res["target_kwh"]))
    tm3.metric("Variance", format_kwh(mon_res["difference_kwh"]), delta=f"{mon_res['percentage_difference']:.2f}%", delta_color="inverse")
    
    if mon_res["is_on_target"]:
        st.markdown('<span class="pill-badge-green" style="font-size: 1rem; padding: 0.5rem 1.25rem;">STATUS: AT OR BELOW TARGET (COMPLIANT)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-badge-red" style="font-size: 1rem; padding: 0.5rem 1.25rem;">STATUS: EXCEEDS TARGET (ACTION REQUIRED)</span>', unsafe_allow_html=True)

# --- 11. VALIDATION & SENSITIVITY ---
elif navigation_option == "Validation & Sensitivity":
    st.title("Computational Validation & Sensitivity")
    st.write("Verifies internal mathematical consistency and tests sensitivity ratios.")
    
    st.markdown("### Sensitivity Analysis")
    sens_df = calculate_sensitivity_analysis()
    st.dataframe(sens_df, use_container_width=True)
    
    st.markdown("### Computational Validation Table")
    val_table = verify_computational_consistency()
    st.dataframe(val_table, use_container_width=True)

# --- 12. METHODOLOGY ---
elif navigation_option == "Methodology":
    st.title("ENERGYSCAPE Mathematical Methodology & Handbook")
    st.markdown("""
    ### Mathematical Formulation Handbook
    
    #### 1. Appliance Electrical Load Quantification
    $$\\text{Monthly Energy Consumption (kWh)} = \\frac{P \\times Q \\times H \\times D}{1000}$$
    
    #### 2. Carbon Footprint Model
    $$\\text{CO}_2\\text{e (kg)} = \\text{Electricity Consumption (kWh)} \\times \\text{Emission Factor (0.70 kg CO}_2\\text{e/kWh)}$$
    
    #### 3. Forecast Error Metrics
    $$\\text{MAPE} = \\frac{100}{n} \\sum_{i=1}^n \\left| \\frac{A_i - F_i}{A_i} \\right|, \\quad \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (A_i - F_i)^2}$$
    
    #### 4. Conservation Scenario & Optimization Model
    $$\\text{Scenario Consumption} = \\text{BAU} \\times (1 - r)$$
    """)
