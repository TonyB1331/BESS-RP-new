# bess_dashboard.py / bess_dash4.py
import datetime
import glob
import io
import os
import pickle
import re
import warnings
import numpy as np
import pandas as pd
import requests
import plotly.express as px            
import plotly.graph_objects as go      
from plotly.subplots import make_subplots
import streamlit as st
import sys
from pathlib import Path

# Import tomorrow's simulation engine cleanly from the scripts package
from scripts.run_daily_sim import run_daily_sim

from db_reader2 import fetch_ipto_bess_data_db, fetch_entsoe_actuals_db
from db_reader2 import fetch_henex_dam_db, fetch_ipto_reserve_prices_db

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

from dam_enginev2 import run_bess_optimization
from dam_predictor import generate_dam_forecast
from reserves_prices_predictor import generate_reserve_forecasts


st.set_page_config(layout="wide", page_title="BESS Asset & Market Dashboard")

# ==========================================
# AUTHENTICATION LOGIN GUARD GATE (ADMIN & VIEW-ONLY)
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["is_admin"] = False

    if st.session_state["password_correct"]:
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.write("")
        st.markdown("### 🔒 Authorized Asset Portal Login")
        user_password = st.text_input("Enter Dashboard Security Passphrase", type="password")
        
        if user_password == "tonyb":
            st.session_state["password_correct"] = True
            st.session_state["is_admin"] = True
            st.rerun()
        elif user_password in ["client123", "guest", "viewonly"]:
            st.session_state["password_correct"] = True
            st.session_state["is_admin"] = False
            st.rerun()
        elif user_password != "":
            st.error("❌ Access Denied. Invalid token passphrase matching profile sequence.")
            
    return False

if not check_password():
    st.stop()

is_admin = st.session_state.get("is_admin", False)

# Helper functions for permanent disk persistence
def save_persistent_state(filename, data):
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass

def load_persistent_state(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetricValue"] { color: #00ffcc; font-family: monospace; }
        div[data-testid="stMetricLabel"] { color: #aaaaaa; }
    </style>
""", unsafe_allow_html=True)

st.title("🔋 BESS Optimization & Market Intelligence Dashboard")
st.markdown("---")

ASSET_CAPACITIES_MW = {
    "STO_BZ1_01_SP": 17.0,
    "STO_BZ1_02_SP": 16.0,
    "MYT_BZ1_01_SB": 35.0,
    "MYT_BZ1_02_SB": 300.0,
    "SUS_BZ1_01_SB": 22.0,
    "ENE_BZ1_01_SB": 49.0,
    "SUS_BZ1_02_SB": 30.0,
    "SUS_BZ1_03_SB": 20.0,
    "PPC_BZ1_01_SB": 1.0,
    "PPC_BZ1_02_SB": 1.0,
    "STO_BZ1_03_SP": 1.0,
    "OPT_BZ1_01_SB": 49.9,
}

@st.cache_data(show_spinner=False)
def get_cached_dam_forecast():
    return generate_dam_forecast()

@st.cache_data(show_spinner=False)
def get_cached_reserve_forecast():
    return generate_reserve_forecasts()

@st.cache_data
def get_cached_optimization(duration, pc_max, pd_max, cycles, n_c, n_d, prediction_file):
    return run_bess_optimization(
        duration=duration, pc_max_input=pc_max, pd_max_input=pd_max,
        Cs=cycles, n_c_input=n_c, n_d_input=n_d, prediction_file_path=prediction_file
    )

@st.cache_data
def get_cached_optimization_df(duration, pc_max, pd_max, cycles, n_c, n_d, _df_market_input):
    return run_bess_optimization(
        duration=duration, pc_max_input=pc_max, pd_max_input=pd_max,
        Cs=cycles, n_c_input=n_c, n_d_input=n_d, df_market_input=_df_market_input
    )

@st.cache_data
def get_cached_ipto_intel(start_str, end_str, asset_name):
    return fetch_ipto_bess_data_db(time_from=start_str, time_to=end_str, asset_name=asset_name)

@st.cache_data
def get_cached_entsoe_actuals(start_str, end_str):
    return fetch_entsoe_actuals_db(time_from=start_str, time_to=end_str)

@st.cache_data
def get_cached_henex_range(start_date, end_date):
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dfs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for single_date in date_list:
        d_str = single_date.strftime("%Y%m%d")
        urls_to_try = [
            f"https://www.enexgroup.gr/documents/20126/200106/{d_str}_EL-DAM_Results_EN_v01.xlsx",
            f"https://www.enexgroup.gr/documents/20126/200106/{d_str}_EL-DAM_Results_EN_v02.xlsx",
            f"https://www.enexgroup.gr/documents/20126/366820/{d_str}_EL-DAM_ResultsSummary_EN_v01.xlsx"
        ]
        for url in urls_to_try:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200 and len(r.content) > 1000:
                    df = pd.read_excel(io.BytesIO(r.content), sheet_name=0)
                    df.columns = [str(c).strip() for c in df.columns]
                    time_col = 'DELIVERY_MTU' if 'DELIVERY_MTU' in df.columns else ('MTU' if 'MTU' in df.columns else None)
                    if time_col and 'MCP' in df.columns:
                        df_mcp = df.drop_duplicates(subset=[time_col], keep='first').copy()
                        df_mcp[time_col] = pd.to_datetime(df_mcp[time_col])
                        all_dfs.append(df_mcp[[time_col, 'MCP']].rename(columns={'MCP': 'value', time_col: 'Date/Hour'}))
                        break
            except Exception:
                continue
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True).sort_values('Date/Hour').drop_duplicates(subset=['Date/Hour']).reset_index(drop=True)
    return pd.DataFrame(columns=['Date/Hour', 'value'])

@st.cache_data
def get_cached_historical_reserve_prices():
    files_map = {
        'aFRR_Down_Price_EUR_MW': 'Reserve_Prices_aFRR_Down*.xlsx',
        'aFRR_Up_Price_EUR_MW': 'Reserve_Prices_aFRR_Up*.xlsx',
        'mFRR_Down_Price_EUR_MW': 'Reserve_Prices_mFRR_Down*.xlsx',
        'mFRR_Up_Price_EUR_MW': 'Reserve_Prices_mFRR_Up*.xlsx',
        'FCR_Down_Price_EUR_MW': 'Reserve_Prices_FCR_Down*.xlsx',
        'FCR_Up_Price_EUR_MW': 'Reserve_Prices_FCR_Up*.xlsx'
    }
    merged_df = None
    for col_name, pattern in files_map.items():
        matched_files = glob.glob(pattern)
        if matched_files:
            try:
                df = pd.read_excel(matched_files[0], sheet_name=0)
                val_col = [c for c in df.columns if c != 'Date/Hour'][0]
                df['dt'] = pd.to_datetime(df['Date/Hour'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
                df_sub = df[['dt', val_col]].rename(columns={val_col: col_name})
                merged_df = df_sub if merged_df is None else pd.merge(merged_df, df_sub, on='dt', how='outer')
            except Exception:
                continue
    return merged_df

# SIDEBAR CONTROLS
st.sidebar.header("🕹️ Optimization Control Knobs")
bess_duration = st.sidebar.slider("Duration (hours - h)", 0.5, 8.0, 2.0, 0.5, disabled=not is_admin)

pc_max_val = st.sidebar.number_input("Max Charging Power (Pc max - MW)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1, disabled=not is_admin)
pd_max_val = st.sidebar.number_input("Max Discharging Power (Pd max - MW)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1, disabled=not is_admin)

cycle_limit = st.sidebar.slider("Daily Cycle Space Bound (Cs)", 1.0, 3.0, 1.5, 0.1, disabled=not is_admin)

col_nc, col_nd = st.sidebar.columns(2)
with col_nc:
    nc_val = st.number_input("Nc (Charge Eff.)", min_value=0.50, max_value=1.00, value=0.95, step=0.01, disabled=not is_admin)
with col_nd:
    nd_val = st.number_input("Nd (Discharge Eff.)", min_value=0.50, max_value=1.00, value=0.95, step=0.01, disabled=not is_admin)

if is_admin:
    if st.sidebar.button("Run Simulation Engine", type="primary"):
        st.session_state["optimized"] = True
        get_cached_optimization.clear()
        get_cached_optimization_df.clear()
else:
    st.sidebar.info("👁️ View-Only Mode Active")

st.sidebar.markdown("---")
st.sidebar.header("🏛️ IPTO Reporting Asset")

def sync_asset_capacity():
    selected_unit = st.session_state.get("bess_unit_profile_select")
    if selected_unit and selected_unit in ASSET_CAPACITIES_MW:
        st.session_state["bess_asset_capacity_readonly"] = float(ASSET_CAPACITIES_MW[selected_unit])

if "bess_unit_profile_select" not in st.session_state:
    st.session_state["bess_unit_profile_select"] = list(ASSET_CAPACITIES_MW.keys())[0]

if "bess_asset_capacity_readonly" not in st.session_state:
    init_unit = st.session_state["bess_unit_profile_select"]
    st.session_state["bess_asset_capacity_readonly"] = float(ASSET_CAPACITIES_MW[init_unit])

selected_asset = st.sidebar.selectbox(
    "Select BESS Storage Unit Profile",
    options=list(ASSET_CAPACITIES_MW.keys()),
    key="bess_unit_profile_select",
    on_change=sync_asset_capacity
)

asset_capacity_input = st.sidebar.number_input(
    f"Capacity for {selected_asset} (MW)", 
    value=st.session_state["bess_asset_capacity_readonly"], 
    disabled=True,
    key="bess_asset_capacity_readonly"
)

# XBID Simulation Control Knobs in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("⚡ XBID Simulation Controls")
xbid_scenarios = st.sidebar.slider("Monte-Carlo Scenarios", min_value=50, max_value=500, value=200, step=50, disabled=not is_admin)
xbid_conservative = st.sidebar.checkbox("Conservative Mode (Prudent Lower Bound)", value=False, disabled=not is_admin)
xbid_reopt_dam = st.sidebar.checkbox("Enable DAM Re-optimization", value=False, disabled=not is_admin)
xbid_min_edge = st.sidebar.number_input("Min Edge Trigger (€/MWh)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, disabled=not is_admin)

st.sidebar.markdown("---")
st.sidebar.header("📅 Historical Telemetry Window")

min_avail_date = pd.Timestamp("2026-01-01").date()
current_date = pd.Timestamp.now().date()

default_telemetry_start = current_date - pd.Timedelta(days=2)
default_telemetry_end = current_date

telemetry_range = st.sidebar.date_input(
    "Live Market / Telemetry Range",
    value=(default_telemetry_start, default_telemetry_end),
    min_value=min_avail_date,
    max_value=current_date,
    key="telemetry_calendar"
)

if isinstance(telemetry_range, tuple) and len(telemetry_range) == 2:
    start_telemetry, end_telemetry = telemetry_range
else:
    start_telemetry, end_telemetry = default_telemetry_start, default_telemetry_end

ipto_start_str = start_telemetry.strftime("%Y-%m-%d")
ipto_end_str = end_telemetry.strftime("%Y-%m-%d")

# Sidebar Data Refresh Control
st.sidebar.markdown("---")
if is_admin:
    if st.sidebar.button("🔄 Reload Market Database"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ IPTO Operational Report",    
    "☀️ RES & Load Report",            
    "🔮 DA Price Predictions",          
    "🔋 BESS Optimization & Backtest",  
    "⚡ XBID Intraday Trading"                     
])

# ----------------------------------------------------
# TAB 1: IPTO REPORTING
# ----------------------------------------------------
with tab1:
    st.subheader(f"🏛️ IPTO Storage Unit Schedule: {selected_asset} ({ASSET_CAPACITIES_MW[selected_asset]} MW) [{start_telemetry} to {end_telemetry}]")
    
    with st.spinner(f"Querying IPTO storage schedule for {selected_asset}..."):
        ipto_payload = get_cached_ipto_intel(ipto_start_str, ipto_end_str, selected_asset)
    
    if ipto_payload and "error" not in ipto_payload:
        df_req = ipto_payload.get("requirements", pd.DataFrame())
        df_res = ipto_payload.get("results", pd.DataFrame())
        df_rev = ipto_payload.get("revenue", pd.DataFrame())
        df_non_realized_rev = ipto_payload.get("non_realized_revenue", pd.DataFrame())
        
        df_xbid_buy_vol = ipto_payload.get("xbid_buy_vol", pd.DataFrame())
        df_xbid_buy_vwap = ipto_payload.get("xbid_buy_vwap", pd.DataFrame())
        df_xbid_sell_vol = ipto_payload.get("xbid_sell_vol", pd.DataFrame())
        df_xbid_sell_vwap = ipto_payload.get("xbid_sell_vwap", pd.DataFrame())
        
        capacity_div = asset_capacity_input if asset_capacity_input > 0 else 1.0
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        if not df_rev.empty:
            df_rev["value"] = pd.to_numeric(df_rev["value"], errors="coerce").fillna(0.0)
            total_ipto_revenue = df_rev["value"].sum()
            revenue_per_mw = total_ipto_revenue / capacity_div
            
            with m_col1:
                st.metric(label="Total Realized DAM Revenue", value=f"€{total_ipto_revenue:,.2f}")
            with m_col2:
                st.metric(label="Realized Revenue per MW", value=f"€{revenue_per_mw:,.2f}/MW")

        if not df_non_realized_rev.empty:
            df_non_realized_rev["value"] = pd.to_numeric(df_non_realized_rev["value"], errors="coerce").fillna(0.0)
            total_non_realized_revenue = df_non_realized_rev["value"].sum()
            non_realized_revenue_per_mw = total_non_realized_revenue / capacity_div
            
            with m_col3:
                st.metric(label="Total Non-Realized DAM Revenue", value=f"€{total_non_realized_revenue:,.2f}")
            with m_col4:
                st.metric(label="Non-Realized Revenue per MW", value=f"€{non_realized_revenue_per_mw:,.2f}/MW")

        st.markdown("---")
        
        st.markdown(f"### ⚖️ ISP Committed Requirements vs Realized Entity Production ({selected_asset})")
        fig_ipto_ops = go.Figure()
        if not df_req.empty:
            df_req["value"] = pd.to_numeric(df_req["value"], errors="coerce").fillna(0.0)
            fig_ipto_ops.add_trace(go.Scatter(x=df_req["timestamp"], y=df_req["value"], name="ISP Mandatory Requirements (MW)", line=dict(color="#f43f5e", width=2, dash="dash")))
        if not df_res.empty:
            df_res["value"] = pd.to_numeric(df_res["value"], errors="coerce").fillna(0.0)
            fig_ipto_ops.add_trace(go.Scatter(x=df_res["timestamp"], y=df_res["value"], name="Realized Entity Production (MW)", line=dict(color="#3b82f6", width=2)))
        fig_ipto_ops.update_layout(hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.15), yaxis_title="Power Headroom (MW)")
        st.plotly_chart(fig_ipto_ops, use_container_width=True)

        st.markdown("---")

        st.markdown("### 💰 Realized vs Non-Realized Day-Ahead Revenue Stream")
        fig_ipto_rev = go.Figure()
        if not df_rev.empty:
            fig_ipto_rev.add_trace(go.Scatter(x=df_rev["timestamp"], y=df_rev["value"], name="Realized DAM Revenue (€)", line=dict(color="#06b6d4", width=2)))
        if not df_non_realized_rev.empty:
            fig_ipto_rev.add_trace(go.Scatter(x=df_non_realized_rev["timestamp"], y=df_non_realized_rev["value"], name="Non-Realized ISP Revenue (€)", line=dict(color="#f43f5e", width=1.5, dash="dash")))

        fig_ipto_rev.update_layout(hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.15), yaxis_title="Revenue (€)")
        st.plotly_chart(fig_ipto_rev, use_container_width=True)

        st.markdown("---")
        
        st.markdown("### 🔀 Continuous Intraday XBID Performance (Storage Assets Pool)")
        
        start_dt_xbid = pd.Timestamp(start_telemetry)
        end_dt_xbid = pd.Timestamp(end_telemetry) + pd.Timedelta(days=1)
        xbid_timeline = pd.date_range(start=start_dt_xbid, end=end_dt_xbid, freq='15min', inclusive='left')
        
        df_xbid_master = pd.DataFrame({'timestamp': xbid_timeline})
        df_xbid_master["Date_Hour"] = df_xbid_master["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
        
        ticks_xbid = df_xbid_master[(df_xbid_master["timestamp"].dt.hour.isin([0, 12])) & (df_xbid_master["timestamp"].dt.minute == 0)]
        tx_vals = ticks_xbid["Date_Hour"].tolist()
        tx_txts = ticks_xbid["timestamp"].dt.strftime("%b %d<br>%H:%M").tolist()

        df_sell_merged = pd.merge(df_xbid_master, df_xbid_sell_vol, on='timestamp', how='left').rename(columns={'value': 'vol'})
        df_sell_merged = pd.merge(df_sell_merged, df_xbid_sell_vwap, on='timestamp', how='left').rename(columns={'value': 'vwap'})
        df_sell_merged['vol'] = df_sell_merged['vol'].fillna(0.0)
        df_sell_merged['vwap'] = df_sell_merged['vwap'].fillna(0.0)

        df_buy_merged = pd.merge(df_xbid_master, df_xbid_buy_vol, on='timestamp', how='left').rename(columns={'value': 'vol'})
        df_buy_merged = pd.merge(df_buy_merged, df_xbid_buy_vwap, on='timestamp', how='left').rename(columns={'value': 'vwap'})
        df_buy_merged['vol'] = df_buy_merged['vol'].fillna(0.0)
        df_buy_merged['vwap'] = df_buy_merged['vwap'].fillna(0.0)

        st.markdown("#### 🟥 Intraday XBID Commercial Sell Dispatch & Value")
        fig_xbid_sell = make_subplots(specs=[[{"secondary_y": True}]])
        fig_xbid_sell.add_trace(go.Bar(x=df_sell_merged["Date_Hour"], y=df_sell_merged["vol"], name="XBID Sell Volume (MW)", marker_color="#ff3344", opacity=0.90, marker_line_width=0), secondary_y=False)
        fig_xbid_sell.add_trace(go.Scatter(x=df_sell_merged["Date_Hour"], y=df_sell_merged["vwap"], name="XBID Sell VWAP (€/MWh)", line=dict(color="#f59e0b", width=1.5)), secondary_y=True)
        fig_xbid_sell.update_layout(barmode="overlay", bargap=0.05, hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.15))
        fig_xbid_sell.update_xaxes(tickmode='array', tickvals=tx_vals, ticktext=tx_txts, gridcolor='rgba(255,255,255,0.08)')
        fig_xbid_sell.update_yaxes(title_text="Traded Volume (MW)", secondary_y=False, gridcolor='rgba(255,255,255,0.08)')
        fig_xbid_sell.update_yaxes(title_text="Clearing Value (€/MWh)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_xbid_sell, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 🟩 Intraday XBID Commercial Buy Dispatch & Value")
        fig_xbid_buy = make_subplots(specs=[[{"secondary_y": True}]])
        fig_xbid_buy.add_trace(go.Bar(x=df_buy_merged["Date_Hour"], y=df_buy_merged["vol"], name="XBID Buy Volume (MW)", marker_color="#10b981", opacity=0.90, marker_line_width=0), secondary_y=False)
        fig_xbid_buy.add_trace(go.Scatter(x=df_buy_merged["Date_Hour"], y=df_buy_merged["vwap"], name="XBID Buy VWAP (€/MWh)", line=dict(color="#f59e0b", width=1.5)), secondary_y=True)
        fig_xbid_buy.update_layout(barmode="overlay", bargap=0.05, hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.15))
        fig_xbid_buy.update_xaxes(tickmode='array', tickvals=tx_vals, ticktext=tx_txts, gridcolor='rgba(255,255,255,0.08)')
        fig_xbid_buy.update_yaxes(title_text="Traded Volume (MW)", secondary_y=False, gridcolor='rgba(255,255,255,0.08)')
        fig_xbid_buy.update_yaxes(title_text="Clearing Value (€/MWh)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_xbid_buy, use_container_width=True)

    else:
        err_msg = ipto_payload.get("error") if ipto_payload else "Unknown Error"
        st.error(f"❌ Failed loading DuckDB market layer: {err_msg}")

# ----------------------------------------------------
# TAB 2: RES & LOAD REPORT
# ----------------------------------------------------
with tab2:
    st.subheader(f"☀️ ENTSO-E Actual RES Generation & System Load ({start_telemetry} to {end_telemetry})")
    
    with st.spinner("Fetching ENTSO-E actual generation and load metrics from DuckDB..."):
        df_entsoe = get_cached_entsoe_actuals(ipto_start_str, ipto_end_str)
        
    if not df_entsoe.empty:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Peak System Load", f"{df_entsoe['actual_load_mw'].max():,.1f} MW")
        with m_col2:
            st.metric("Peak Solar Production", f"{df_entsoe['solar_mw'].max():,.1f} MW")
        with m_col3:
            st.metric("Peak Wind Onshore", f"{df_entsoe['wind_onshore_mw'].max():,.1f} MW")
        with m_col4:
            st.metric("Max Total RES", f"{df_entsoe['total_res_mw'].max():,.1f} MW")
            
        st.markdown("---")
        
        st.markdown("### 📈 Actual Generation & Grid Load Timeline")
        fig_entsoe = go.Figure()
        fig_entsoe.add_trace(go.Scatter(x=df_entsoe["timestamp"], y=df_entsoe["actual_load_mw"], name="Actual System Load (MW)", line=dict(color="#f43f5e", width=2)))
        fig_entsoe.add_trace(go.Scatter(x=df_entsoe["timestamp"], y=df_entsoe["total_res_mw"], name="Total RES Generation (MW)", line=dict(color="#10b981", width=2)))
        fig_entsoe.add_trace(go.Scatter(x=df_entsoe["timestamp"], y=df_entsoe["solar_mw"], name="Solar Generation (MW)", line=dict(color="#f59e0b", width=1.5, dash="dot")))
        fig_entsoe.add_trace(go.Scatter(x=df_entsoe["timestamp"], y=df_entsoe["wind_onshore_mw"], name="Wind Onshore Generation (MW)", line=dict(color="#06b6d4", width=1.5, dash="dot")))
        
        fig_entsoe.update_layout(
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=-0.15),
            yaxis_title="Power (MW)"
        )
        st.plotly_chart(fig_entsoe, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Filtered Records Data Table")
        
        df_display = df_entsoe.copy()
        df_display = df_display.rename(columns={
            "timestamp": "Timestamp",
            "solar_mw": "Solar Production (MW)",
            "wind_onshore_mw": "Wind Onshore Production (MW)",
            "total_res_mw": "Total Actual RES (MW)",
            "actual_load_mw": "Actual System Load (MW)"
        })
        
        st.dataframe(df_display, use_container_width=True, height=350)
        
        csv_entsoe = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Export RES & Load Report (.csv)",
            data=csv_entsoe,
            file_name=f"entsoe_res_and_load_{ipto_start_str}_to_{ipto_end_str}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No ENTSO-E RES & Load records found for the selected date range in DuckDB. Please run `python db_sync.py` to sync this window.")

# ----------------------------------------------------
# TAB 3: DA PRICE PREDICTIONS
# ----------------------------------------------------
with tab3:
    st.subheader("🔮 LightGBM Day-Ahead Energy & Reserve Capacity Price Predictions")
    st.markdown("Trains dedicated LightGBM gradient boosted models utilizing real-time API grid forecasts and historical lag signals to output tomorrow's ($D+1$) 96 market periods.")

    forecast_saved = load_persistent_state("latest_forecast.pkl")
    run_forecast_clicked = False
    
    if is_admin:
        if st.button("Generate Tomorrow's Forecasts", type="primary"):
            run_forecast_clicked = True
    else:
        if forecast_saved is None:
            st.info("ℹ️ Predictions will be visible here once generated by the administrator.")

    if run_forecast_clicked:
        with st.spinner("⏳ Fetching features & training LightGBM models for DAM MCP and all 6 Balancing Reserves..."):
            try:
                df_dam_preds, target_date = get_cached_dam_forecast()
                df_res_preds, _ = get_cached_reserve_forecast()
                save_persistent_state("latest_forecast.pkl", {
                    "df_dam_preds": df_dam_preds,
                    "target_date": target_date,
                    "df_res_preds": df_res_preds
                })
                forecast_saved = {
                    "df_dam_preds": df_dam_preds,
                    "target_date": target_date,
                    "df_res_preds": df_res_preds
                }
            except Exception as e:
                st.error(f"❌ Prediction Engine Failed: {e}")

    if forecast_saved is not None:
        try:
            df_dam_preds = forecast_saved["df_dam_preds"]
            target_date = forecast_saved["target_date"]
            df_res_preds = forecast_saved["df_res_preds"]

            st.markdown(f"### 📊 Predictions Overview for Tomorrow ({target_date})")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Average DAM Price", f"€{df_dam_preds['Price_Prediction_EUR_MWh'].mean():.2f}/MWh")
            m_col2.metric("Peak DAM Price", f"€{df_dam_preds['Price_Prediction_EUR_MWh'].max():.2f}/MWh")
            m_col3.metric("Minimum DAM Price", f"€{df_dam_preds['Price_Prediction_EUR_MWh'].min():.2f}/MWh")
            
            st.markdown("---")
            
            st.markdown("### 📈 Day-Ahead Market Clearing Price (MCP) Forecast")
            fig_dam = go.Figure()
            fig_dam.add_trace(go.Scatter(x=df_dam_preds.index, y=df_dam_preds['Price_Prediction_EUR_MWh'], name="Predicted DAM MCP (€/MWh)", line=dict(color="#00ffcc", width=2.5)))
            fig_dam.update_layout(hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Price Forecast (€/MWh)", xaxis_title="Delivery Interval")
            st.plotly_chart(fig_dam, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown("#### 🛡️ Balancing Reserve Capacity Prices Forecast (€/MW)")
            fig_res = go.Figure()
            colors = {
                'FCR_Up_Price_EUR_MW': '#f59e0b', 
                'FCR_Down_Price_EUR_MW': '#06b6d4', 
                'aFRR_Up_Price_EUR_MW': '#10b981', 
                'aFRR_Down_Price_EUR_MW': '#8b5cf6', 
                'mFRR_Up_Price_EUR_MW': '#ec4899', 
                'mFRR_Down_Price_EUR_MW': '#3b82f6'
            }
            for col in df_res_preds.columns:
                fig_res.add_trace(go.Scatter(x=df_res_preds.index, y=df_res_preds[col], name=col.replace('_Price_EUR_MW', ''), line=dict(color=colors.get(col, '#ffffff'), width=2)))
            fig_res.update_layout(hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Reserve Price Forecast (€/MW)", xaxis_title="Delivery Interval")
            st.plotly_chart(fig_res, use_container_width=True)

            st.markdown("---")

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_dam_export = df_dam_preds.reset_index()
                df_dam_export.to_excel(writer, sheet_name="DAM_Energy_Forecast", index=False)

                df_res_export = df_res_preds.reset_index()
                df_res_export.to_excel(writer, sheet_name="Reserve_Capacity_Forecast", index=False)

            buffer.seek(0)
            file_name_export = f"Greece_Market_Price_Forecasts_{target_date}.xlsx"

            st.download_button(
                label=f"⬇️ Download Tomorrow's Forecast File ({file_name_export})",
                data=buffer,
                file_name=file_name_export,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        except Exception as e:
            st.error(f"❌ Forecast Rendering Error: {e}")

# ----------------------------------------------------
# TAB 4: BESS OPTIMIZATION & BACKTEST
# ----------------------------------------------------
with tab4:
    st.subheader("🔋 BESS Multi-Market Optimization Engine")

    # 1. Mode Selector
    opt_mode = st.radio(
        "Select Optimization Mode",
        options=["🔮 Tomorrow's Forecast (D+1 Horizon)", "📜 Historical Backtest (DuckDB Stream)"],
        horizontal=True
    )

    df_q, df_d = None, None
    opt_saved = load_persistent_state("latest_opt_results.pkl")

    # --- MODE A: D+1 Tomorrow Forecast ---
    if opt_mode == "🔮 Tomorrow's Forecast (D+1 Horizon)":
        latest_pred_file = st.session_state.get("latest_pred_file", None)
        st.info("Uses ML price predictions generated in Tab 3 or the latest forecast Excel file.")
        
        if is_admin:
            if st.button("🚀 Run Tomorrow's Forecast Optimization", type="primary"):
                with st.spinner("⏳ Running linear programming optimization for D+1..."):
                    df_q, df_d = run_bess_optimization(
                        duration=bess_duration,
                        Cs=cycle_limit,
                        pc_max_input=pc_max_val,
                        pd_max_input=pd_max_val,
                        n_c_input=nc_val,
                        n_d_input=nd_val,
                        prediction_file_path=latest_pred_file
                    )
                    if df_q is not None:
                        st.session_state["opt_results_q"] = df_q
                        st.session_state["opt_results_d"] = df_d
                        save_persistent_state("latest_opt_results.pkl", {"df_q": df_q, "df_d": df_d})

    # --- MODE B: Historical Backtest ---
    else:
        st.markdown("#### 📅 Select Historical Backtest Window")
        current_date = pd.Timestamp.now().date()
        
        hist_cols = st.columns([2, 1])
        with hist_cols[0]:
            hist_range = st.date_input(
                "Historical Date Range",
                value=(current_date - pd.Timedelta(days=7), current_date - pd.Timedelta(days=1)),
                max_value=current_date,
                key="bess_backtest_picker"
            )

        if isinstance(hist_range, tuple) and len(hist_range) == 2:
            h_start, h_end = hist_range
            if is_admin:
                if st.button("🚀 Run Historical Backtest Optimization", type="primary"):
                    with st.spinner(f"⏳ Querying DuckDB & solving backtest from {h_start} to {h_end}..."):
                        
                        df_dam = fetch_henex_dam_db(str(h_start), str(h_end))

                        if df_dam.empty:
                            st.error(f"❌ No Day-Ahead Market records found in database between {h_start} and {h_end}.")
                        else:
                            df_res_raw = fetch_ipto_reserve_prices_db(str(h_start), str(h_end), reserve_type="ALL")
                            
                            if not df_res_raw.empty:
                                df_res_raw = df_res_raw.rename(columns={"timestamp": "Delivery_Interval"})
                                df_up = df_res_raw.pivot(index='Delivery_Interval', columns='reserve_type', values='price_up')
                                df_down = df_res_raw.pivot(index='Delivery_Interval', columns='reserve_type', values='price_down')

                                rename_up = {col: f"{col}_Up_Price_EUR_MW" for col in df_up.columns}
                                rename_down = {col: f"{col}_Down_Price_EUR_MW" for col in df_down.columns}
                                
                                df_res_pivoted = pd.concat([
                                    df_up.rename(columns=rename_up),
                                    df_down.rename(columns=rename_down)
                                ], axis=1).reset_index()

                                df_hist_market = pd.merge(df_dam, df_res_pivoted, on='Delivery_Interval', how='left')
                            else:
                                df_hist_market = df_dam.copy()

                            reserve_cols = [
                                "FCR_Up_Price_EUR_MW", "FCR_Down_Price_EUR_MW",
                                "aFRR_Up_Price_EUR_MW", "aFRR_Down_Price_EUR_MW",
                                "mFRR_Up_Price_EUR_MW", "mFRR_Down_Price_EUR_MW"
                            ]
                            for c in reserve_cols:
                                if c not in df_hist_market.columns:
                                    df_hist_market[c] = 0.0
                                else:
                                    df_hist_market[c] = df_hist_market[c].fillna(0.0)

                            df_q, df_d = run_bess_optimization(
                                duration=bess_duration,
                                Cs=cycle_limit,
                                pc_max_input=pc_max_val,
                                pd_max_input=pd_max_val,
                                n_c_input=nc_val,
                                n_d_input=nd_val,
                                df_market_input=df_hist_market
                            )
                            if df_q is not None:
                                st.session_state["opt_results_q"] = df_q
                                st.session_state["opt_results_d"] = df_d
                                save_persistent_state("latest_opt_results.pkl", {"df_q": df_q, "df_d": df_d})

    # --- Session State & Disk Fallback ---
    if df_q is None:
        if "opt_results_q" in st.session_state:
            df_q = st.session_state["opt_results_q"]
            df_d = st.session_state["opt_results_d"]
        elif opt_saved is not None:
            df_q = opt_saved.get("df_q")
            df_d = opt_saved.get("df_d")

    # --- RENDER DASHBOARD RESULTS & EXCEL DOWNLOAD ---
    if df_q is not None and df_d is not None:
        st.markdown("---")
        st.markdown("### 💰 Financial Breakdown")

        # 1. Directional Capacity Profit Calculations
        up_capacity_profit = (
            (df_d['FCR_Up_Profit'].iloc[0] if 'FCR_Up_Profit' in df_d.columns else 0.0) +
            (df_d['aFRR_Up_Profit'].iloc[0] if 'aFRR_Up_Profit' in df_d.columns else 0.0) +
            (df_d['mFRR_Up_Profit'].iloc[0] if 'mFRR_Up_Profit' in df_d.columns else 0.0)
        )
        
        down_capacity_profit = (
            (df_d['FCR_Dn_Profit'].iloc[0] if 'FCR_Dn_Profit' in df_d.columns else 0.0) +
            (df_d['aFRR_Dn_Profit'].iloc[0] if 'aFRR_Dn_Profit' in df_d.columns else 0.0) +
            (df_d['mFRR_Dn_Profit'].iloc[0] if 'mFRR_Dn_Profit' in df_d.columns else 0.0)
        )

        total_capacity_profit = up_capacity_profit + down_capacity_profit
        dam_profit = df_d['DAM_Arbitrage_Profit'].iloc[0] if 'DAM_Arbitrage_Profit' in df_d.columns else 0.0
        total_profit = df_d['Total_Daily_Profit'].iloc[0] if 'Total_Daily_Profit' in df_d.columns else (dam_profit + total_capacity_profit)

        # 2. Key Metric Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("DAM Arbitrage Profit", f"€{dam_profit:,.2f}")
        k2.metric("Total Capacities Profit", f"€{total_capacity_profit:,.2f}")
        k3.metric("Upwards Capacity Profit", f"€{up_capacity_profit:,.2f}")
        k4.metric("Downwards Capacity Profit", f"€{down_capacity_profit:,.2f}")
        k5.metric("Total Daily Net Profit", f"€{total_profit:,.2f}")

        st.markdown("---")

        # =========================================================
        # 📊 GRAPH 1: 15-MINUTE DISPATCH & CAPACITY BARS
        # =========================================================
        st.markdown("### ⚡ Operational Actions & Capacity Allocation (Quarterly)")

        time_x = df_q['Date_Hour'] if 'Date_Hour' in df_q.columns else (
            df_q['Delivery_Interval'] if 'Delivery_Interval' in df_q.columns else df_q.index
        )

        fig_actions = go.Figure()

        def get_clean_series(df, col_aliases, negative=False):
            matched_col = next((c for c in col_aliases if c in df.columns), None)
            if matched_col is None:
                return None
            s = pd.to_numeric(df[matched_col], errors='coerce').fillna(0.0)
            if s.abs().sum() == 0:
                return None
            s = -s if negative else s
            return s.replace(0, np.nan)

        dam_dh = get_clean_series(df_q, ['DAM_Discharge_MW', 'Discharge_MW'], negative=False)
        dam_ch = get_clean_series(df_q, ['DAM_Charge_MW', 'Charge_MW'], negative=True)

        if dam_dh is not None:
            fig_actions.add_trace(go.Bar(
                x=time_x, y=dam_dh,
                name="DAM Discharge (MW)", marker_color="#ef4444", opacity=0.9
            ))
        if dam_ch is not None:
            fig_actions.add_trace(go.Bar(
                x=time_x, y=dam_ch,
                name="DAM Charge (MW)", marker_color="#10b981", opacity=0.9
            ))

        up_cap_configs = [
            (['FCR_Up_Award_MW', 'FCR_Up_Capacity_MW', 'fcr_up'], 'FCR Up Capacity', '#f59e0b'),
            (['aFRR_Up_Award_MW', 'aFRR_Up_Capacity_MW', 'afrr_up'], 'aFRR Up Capacity', '#8b5cf6'),
            (['mFRR_Up_Award_MW', 'mFRR_Up_Capacity_MW', 'mfrr_up'], 'mFRR Up Capacity', '#3b82f6')
        ]

        for aliases, label, color in up_cap_configs:
            series = get_clean_series(df_q, aliases, negative=False)
            if series is not None:
                fig_actions.add_trace(go.Bar(
                    x=time_x, y=series, name=label,
                    marker_color=color, opacity=0.75
                ))

        dn_cap_configs = [
            (['FCR_Dn_Award_MW', 'FCR_Down_Capacity_MW', 'fcr_dn', 'FCR_Dn_Capacity_MW'], 'FCR Down Capacity', '#06b6d4'),
            (['aFRR_Dn_Award_MW', 'aFRR_Down_Capacity_MW', 'afrr_dn', 'aFRR_Dn_Capacity_MW'], 'aFRR Down Capacity', '#ec4899'),
            (['mFRR_Dn_Award_MW', 'mFRR_Down_Capacity_MW', 'mfrr_dn', 'mFRR_Dn_Capacity_MW'], 'mFRR Down Capacity', '#14b8a6')
        ]

        for aliases, label, color in dn_cap_configs:
            series = get_clean_series(df_q, aliases, negative=True)
            if series is not None:
                fig_actions.add_trace(go.Bar(
                    x=time_x, y=series, name=label,
                    marker_color=color, opacity=0.75
                ))

        fig_actions.update_layout(
            barmode="relative",
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=-0.22),
            yaxis_title="Power Headroom / Capacity (MW)  [ + Up / - Down ]",
            xaxis_title="Delivery Interval"
        )
        st.plotly_chart(fig_actions, use_container_width=True)

        st.markdown("---")

        # =========================================================
        # 📈 GRAPH 2: BESS STATE OF CHARGE (SOC / SOE)
        # =========================================================
        st.markdown("### 🔋 BESS State of Charge Progression (SOE / SOC)")

        soe_col = next((c for c in ['Expected_Energy_MWh', 'SOE_Capacity_MWh', 'Energy_MWh', 'soe'] if c in df_q.columns), None)

        fig_soc = go.Figure()
        if soe_col:
            fig_soc.add_trace(go.Scatter(
                x=time_x, y=df_q[soe_col],
                name="Stored Energy (MWh)",
                line=dict(color="#00ffcc", width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 204, 0.12)'
            ))

        fig_soc.update_layout(
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Energy Stored (MWh)",
            xaxis_title="Delivery Interval"
        )
        st.plotly_chart(fig_soc, use_container_width=True)

        st.markdown("---")

        # Detailed Table
        st.markdown("#### 📋 Operational Dispatch Schedule")
        st.dataframe(df_q, use_container_width=True, height=300)

        # Excel Export (15-Column Layout)
        df_schedule_export = df_q.rename(columns={
            "Date_Hour": "Delivery_Interval",
            "Price": "DAM_Price_EUR_MWh",
            "DAM_Charge_MW": "DAM_Charge_MW",
            "DAM_Discharge_MW": "DAM_Discharge_MW",
            "Physical_Charge_MW": "Physical_Charge_MW",
            "Physical_Discharge_MW": "Physical_Discharge_MW",
            "Expected_Energy_MWh": "SOE_Capacity_MWh",
            "Total_Capacity_Up_MW": "Total_Capacity_Up_MW",
            "Total_Capacity_Dn_MW": "Total_Capacity_Dn_MW",
            "mFRR_Up_Award_MW": "mFRR_Up_Capacity_MW",
            "mFRR_Dn_Award_MW": "mFRR_Down_Capacity_MW",
            "FCR_Up_Award_MW": "FCR_Up_Capacity_MW",
            "FCR_Dn_Award_MW": "FCR_Down_Capacity_MW",
            "aFRR_Up_Award_MW": "aFRR_Up_Capacity_MW",
            "aFRR_Dn_Award_MW": "aFRR_Down_Capacity_MW"
        })

        target_export_cols = [
            "Delivery_Interval", "DAM_Price_EUR_MWh", "DAM_Charge_MW", "DAM_Discharge_MW",
            "Physical_Charge_MW", "Physical_Discharge_MW", "SOE_Capacity_MWh",
            "Total_Capacity_Up_MW", "Total_Capacity_Dn_MW",
            "mFRR_Up_Capacity_MW", "mFRR_Down_Capacity_MW",
            "FCR_Up_Capacity_MW", "FCR_Down_Capacity_MW",
            "aFRR_Up_Capacity_MW", "aFRR_Down_Capacity_MW"
        ]
        
        for col in target_export_cols:
            if col not in df_schedule_export.columns:
                df_schedule_export[col] = 0.0

        df_schedule_export = df_schedule_export[target_export_cols]

        try:
            first_dt = pd.to_datetime(df_schedule_export["Delivery_Interval"].iloc[0])
            last_dt = pd.to_datetime(df_schedule_export["Delivery_Interval"].iloc[-1])
            date_str = f"{first_dt.strftime('%d_%m')}_to_{last_dt.strftime('%d_%m')}" if first_dt.date() != last_dt.date() else first_dt.strftime("%d_%m")
        except Exception:
            date_str = "D+1"

        output_filename = f"BESS_Optimized_Schedule_{date_str}.xlsx"

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_schedule_export.to_excel(writer, sheet_name="BESS_Schedule", index=False)
            df_d.to_excel(writer, sheet_name="Daily_Financial_Summary", index=False)
        buffer.seek(0)

        st.download_button(
            label=f"⬇️ Download Optimization Results ({output_filename})",
            data=buffer,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    elif not is_admin and opt_saved is None:
        st.info("ℹ️ Optimization results will be visible here once solved by the administrator.")

# ----------------------------------------------------
# TAB 5: XBID TRADING (Tomorrow's Simulation)
# ----------------------------------------------------
with tab5:
    st.subheader("⚡ Tomorrow's XBID Intraday Monte-Carlo Simulation")
    st.markdown("Simulates intraday trading against a calibrated Greek XBID market across multiple Monte-Carlo scenarios for tomorrow's schedule.")

    schedule_files = glob.glob("BESS_Optimized_Schedule_*.xlsx")
    if schedule_files:
        schedule_files.sort(key=os.path.getmtime, reverse=True)
        schedule_file = schedule_files[0]
    else:
        schedule_file = st.session_state.get("latest_pred_file", None)

    xbid_saved = load_persistent_state("latest_xbid_results.pkl")

    if not schedule_file or not os.path.exists(schedule_file):
        if xbid_saved is None:
            st.warning("⚠️ No `BESS_Optimized_Schedule_DD_MM.xlsx` schedule file found. Please run BESS Optimization in Tab 4 first.")
    else:
        st.info(f"📂 Selected schedule file: `{schedule_file}`")
        if is_admin:
            if st.button("🚀 Run Tomorrow's XBID Simulation", type="primary", key="btn_run_xbid_sim_tab5"):
                with st.spinner(f"⏳ Running {xbid_scenarios} Monte-Carlo XBID market draws for `{schedule_file}`..."):
                    try:
                        sim_out = run_daily_sim(
                            schedule_path=schedule_file,
                            power_mw=pc_max_val,
                            energy_mwh=pc_max_val * bess_duration,
                            n_scenarios=xbid_scenarios,
                            conservative=xbid_conservative,
                            reoptimize_dam=xbid_reopt_dam,
                            min_edge_eur=xbid_min_edge
                        )
                        st.session_state["xbid_sim_out"] = sim_out
                        save_persistent_state("latest_xbid_results.pkl", sim_out)
                    except Exception as e:
                        st.error(f"❌ XBID Simulation Failed: {e}")

    # Fallback to persistent disk cache
    if "xbid_sim_out" not in st.session_state and xbid_saved is not None:
        st.session_state["xbid_sim_out"] = xbid_saved

    # Render Results if available
    if "xbid_sim_out" in st.session_state:
        res = st.session_state["xbid_sim_out"]

        st.markdown("---")
        st.markdown(f"### 💰 Financial Value Decomposition — Delivery Date: `{res['day']}`")

        # 1. Headline Value Metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Committed DAM Revenue", f"€{res['dam_energy_revenue']:,.2f}")
        kpi2.metric("Committed Capacity Revenue", f"€{res['reserve_revenue']:,.2f}")
        kpi3.metric("Expected XBID Intraday PnL", f"€{res['xbid_pnl_median']:,.2f}")
        kpi4.metric("Total Expected Uplift", f"{res['uplift_median']:+.2f}%", 
                   delta_color="normal", help="P10 to P90 range")

        st.markdown("---")

        # 2. Risk Band Summary
        st.markdown("#### 🛡️ Risk Band Analysis (P10 — Median — P90)")
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("P10 Lower Bound Value", f"€{res['total_value_p10']:,.2f}", f"{res['uplift_p10']:+.2f}%")
        r_col2.metric("Median Expected Value", f"€{res['total_value_median']:,.2f}", f"{res['uplift_median']:+.2f}%")
        r_col3.metric("P90 Upper Bound Value", f"€{res['total_value_p90']:,.2f}", f"{res['uplift_p90']:+.2f}%")

        # 3. Representative Order Blotter
        rep = res.get("representative", {})
        orders = rep.get("orders", [])
        if orders:
            st.markdown("---")
            st.markdown(f"### 📋 Representative Scenario Order Blotter ({len(orders)} trades executed)")
            df_orders = pd.DataFrame(orders)
            st.dataframe(df_orders, use_container_width=True, height=300)
    elif not is_admin and xbid_saved is None:
        st.info("ℹ️ XBID Simulation results will be visible here once run by the administrator.")