# db_reader2.py
from pathlib import Path
import os
import requests
import duckdb
import db_connection
import pandas as pd
import streamlit as st

# Dynamically resolve absolute path to repository root
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = str(BASE_DIR / "greece_energy_market2.duckdb")

# Fallback direct download URL from your GitHub repository (main branch)
GITHUB_LFS_RAW_URL = "https://raw.githubusercontent.com/TonyB1331/BESS-Full/main/greece_energy_market2.duckdb"

BESS_STORAGE_UNITS = [
    "STO_BZ1_01_SP", "STO_BZ1_02_SP", "MYT_BZ1_01_SB", "MYT_BZ1_02_SB",
    "SUS_BZ1_01_SB", "ENE_BZ1_01_SB", "SUS_BZ1_02_SB", "SUS_BZ1_03_SB",
    "PPC_BZ1_01_SB", "PPC_BZ1_02_SB", "STO_BZ1_03_SP", "OPT_BZ1_01_SB"
]

def ensure_database_ready():
    """No-op: the database now lives in MotherDuck (see db_connection.py).
    Kept so existing call sites keep working."""
    return


@st.cache_data(ttl=1800)
def fetch_entsoe_actuals_db(time_from: str, time_to: str):
    """Fetches ENTSO-E Actual Solar, Wind, Total RES, and Load directly from DuckDB."""
    ensure_database_ready()
    try:
        with db_connection.connect(read_only=True) as conn:
            query = f"""
                SELECT timestamp, solar_mw, wind_onshore_mw, total_res_mw, actual_load_mw
                FROM entsoe_actuals
                WHERE timestamp >= '{time_from} 00:00:00' AND timestamp <= '{time_to} 23:45:00'
                ORDER BY timestamp ASC
            """
            return conn.execute(query).df()
    except Exception:
        return pd.DataFrame(columns=['timestamp', 'solar_mw', 'wind_onshore_mw', 'total_res_mw', 'actual_load_mw'])

@st.cache_data(ttl=1800)
def fetch_ipto_system_forecasts_db(time_from: str, time_to: str):
    """Fetches ISP1 System Forecasts (Non-Dispatchable Load & RES) directly from DuckDB."""
    ensure_database_ready()
    try:
        with db_connection.connect(read_only=True) as conn:
            query = f"""
                SELECT timestamp, category, value_mw
                FROM ipto_schedules
                WHERE unit_id = 'SYSTEM_GR'
                  AND timestamp >= '{time_from} 00:00:00' AND timestamp <= '{time_to} 23:45:00'
                ORDER BY timestamp ASC
            """
            return conn.execute(query).df()
    except Exception:
        return pd.DataFrame(columns=['timestamp', 'category', 'value_mw'])

@st.cache_data(ttl=1800)
def fetch_ipto_bess_data_db(time_from: str, time_to: str, asset_name: str = "STO_BZ1_01_SP"):
    """
    Fetches IPTO BESS Schedules, DAM Realized/Non-Realized Revenue, and XBID Trades from DuckDB.
    """
    ensure_database_ready()
    try:
        with db_connection.connect(read_only=True) as conn:
            start_ts = f"{time_from} 00:00:00"
            end_ts = f"{time_to} 23:45:00"
            target_unit = asset_name if asset_name in BESS_STORAGE_UNITS else BESS_STORAGE_UNITS[0]

            df_schedules = conn.execute(f"""
                SELECT timestamp, unit_id, category, value_mw AS value
                FROM ipto_schedules
                WHERE unit_id = '{target_unit}'
                  AND timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
                ORDER BY timestamp
            """).df()

            df_results = df_schedules[df_schedules['category'] == 'ISP2_RESULTS_V1'][['timestamp', 'value']].reset_index(drop=True)

            df_req_all = df_schedules[df_schedules['category'].isin(['ISP1_REQUIREMENTS_V2', 'ISP1_REQUIREMENTS_V1'])].copy()
            if not df_req_all.empty:
                df_req_all['v_rank'] = df_req_all['category'].apply(lambda x: 1 if x == 'ISP1_REQUIREMENTS_V2' else 2)
                df_requirements = df_req_all.sort_values(by=['timestamp', 'v_rank']).drop_duplicates(subset=['timestamp'], keep='first')[['timestamp', 'value']].reset_index(drop=True)
            else:
                df_requirements = pd.DataFrame(columns=['timestamp', 'value'])

            df_dam = conn.execute(f"""
                SELECT timestamp, mcp_eur_mwh AS value
                FROM henex_dam
                WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
                ORDER BY timestamp
            """).df()

            if not df_results.empty and not df_dam.empty:
                df_rev = pd.merge(df_results, df_dam, on='timestamp', suffixes=('_results', '_dam'))
                df_rev['value'] = (df_rev['value_results'] * df_rev['value_dam']) / 4.0
                df_revenue = df_rev[['timestamp', 'value']]
            else:
                df_revenue = pd.DataFrame(columns=['timestamp', 'value'])

            if not df_requirements.empty and not df_dam.empty:
                df_non_rev = pd.merge(df_requirements, df_dam, on='timestamp', suffixes=('_req', '_dam'))
                df_non_rev['value'] = (df_non_rev['value_req'] * df_non_rev['value_dam']) / 4.0
                df_non_realized_revenue = df_non_rev[['timestamp', 'value']]
            else:
                df_non_realized_revenue = pd.DataFrame(columns=['timestamp', 'value'])

            df_xbid = conn.execute(f"""
                SELECT timestamp, side, vwap_eur_mwh, total_trades_mw
                FROM henex_xbid
                WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
                ORDER BY timestamp
            """).df()

            if not df_xbid.empty:
                df_buy = df_xbid[df_xbid['side'].str.upper().str.contains('BUY', na=False)]
                df_sell = df_xbid[df_xbid['side'].str.upper().str.contains('SELL', na=False)]

                df_xbid_buy_vol = df_buy[['timestamp', 'total_trades_mw']].rename(columns={'total_trades_mw': 'value'})
                df_xbid_buy_vwap = df_buy[['timestamp', 'vwap_eur_mwh']].rename(columns={'vwap_eur_mwh': 'value'})
                df_xbid_sell_vol = df_sell[['timestamp', 'total_trades_mw']].rename(columns={'total_trades_mw': 'value'})
                df_xbid_sell_vwap = df_sell[['timestamp', 'vwap_eur_mwh']].rename(columns={'vwap_eur_mwh': 'value'})
            else:
                empty_df = pd.DataFrame(columns=['timestamp', 'value'])
                df_xbid_buy_vol, df_xbid_buy_vwap = empty_df, empty_df
                df_xbid_sell_vol, df_xbid_sell_vwap = empty_df, empty_df

            return {
                'results': df_results,
                'requirements': df_requirements,
                'revenue': df_revenue,
                'non_realized_revenue': df_non_realized_revenue,
                'xbid_buy_vol': df_xbid_buy_vol,
                'xbid_buy_vwap': df_xbid_buy_vwap,
                'xbid_sell_vol': df_xbid_sell_vol,
                'xbid_sell_vwap': df_xbid_sell_vwap
            }
    except Exception as e:
        return {"error": f"Database file missing or locked: {e}"}

@st.cache_data(ttl=1800)
def fetch_ipto_reserve_prices_db(time_from: str, time_to: str, reserve_type: str = "ALL"):
    """Fetches ISP Reserve Prices (FCR, aFRR, mFRR) directly from DuckDB."""
    ensure_database_ready()
    try:
        with db_connection.connect(read_only=True) as conn:
            type_filter = f"AND reserve_type = '{reserve_type}'" if reserve_type != "ALL" else ""
            query = f"""
                SELECT timestamp, reserve_type, price_up, price_down
                FROM ipto_reserve_prices
                WHERE timestamp >= '{time_from} 00:00:00' 
                  AND timestamp <= '{time_to} 23:45:00' {type_filter}
                ORDER BY timestamp ASC
            """
            return conn.execute(query).df()
    except Exception:
        return pd.DataFrame(columns=['timestamp', 'reserve_type', 'price_up', 'price_down'])


# Add this function to db_reader2.py

@st.cache_data(ttl=1800)
def fetch_henex_dam_db(time_from: str, time_to: str) -> pd.DataFrame:
    """Fetches Day-Ahead Market Clearing Prices (MCP) from henex_dam in DuckDB."""
    ensure_database_ready()
    try:
        with db_connection.connect(read_only=True) as conn:
            query = f"""
                SELECT timestamp AS Delivery_Interval, mcp_eur_mwh AS DAM_Price
                FROM henex_dam
                WHERE timestamp >= '{time_from} 00:00:00' 
                  AND timestamp <= '{time_to} 23:45:00'
                ORDER BY timestamp ASC
            """
            return conn.execute(query).df()
    except Exception:
        return pd.DataFrame(columns=['Delivery_Interval', 'DAM_Price'])