# db_sync.py
import io
import re
import requests
import warnings
import duckdb
import os
import db_connection
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient

# Suppress openpyxl warnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Target database file name
DB_FILE = "greece_energy_market2.duckdb"

# API & Endpoint Credentials
ENTSOE_API_TOKEN = os.environ.get("ENTSOE_API_TOKEN", "")
IPTO_PUBLIC_API_URL = "https://www.admie.gr/getOperationMarketFilewRange"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

# Target BESS Units
BESS_UNITS = [
    "STO_BZ1_01_SP", "STO_BZ1_02_SP", "MYT_BZ1_01_SB", "MYT_BZ1_02_SB",
    "SUS_BZ1_01_SB", "ENE_BZ1_01_SB", "SUS_BZ1_02_SB", "SUS_BZ1_03_SB",
    "PPC_BZ1_01_SB", "PPC_BZ1_02_SB", "STO_BZ1_03_SP", "OPT_BZ1_01_SB"
]
BESS_UNITS_UPPER = set(u.upper().strip() for u in BESS_UNITS)

# System Forecast Targets in ISP Requirements Files
SYSTEM_FORECAST_TARGETS = {
    "NON-DISPATCHEBLE LOAD": "ISP1_LOAD_NON_DISPATCHABLE",
    "NON-DISPATCHABLE RES": "ISP1_RES_NON_DISPATCHABLE",
    "RESFIT PORTFOLIO": "ISP1_RES_RESFIT_PORTFOLIO",
    "NON-DISPATCHEBLE LOSSES": "ISP1_LOAD_LOSSES"
}

# ==========================================
# 1. DATABASE SCHEMA INITIALIZATION
# ==========================================
def init_db():
    """Initializes DuckDB tables with unique indexes for fast upserts."""
    conn = db_connection.connect()
    
    # 1. HEnEx Day-Ahead Market (DAM)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS henex_dam (
            timestamp TIMESTAMP PRIMARY KEY,
            mcp_eur_mwh DOUBLE
        );
    """)
    
    # 2. HEnEx Continuous Intraday (XBID) Storage Trades
    conn.execute("""
        CREATE TABLE IF NOT EXISTS henex_xbid (
            timestamp TIMESTAMP,
            side VARCHAR,
            classification VARCHAR,
            vwap_eur_mwh DOUBLE,
            total_trades_mw DOUBLE,
            PRIMARY KEY (timestamp, side, classification)
        );
    """)
    
    # 3. IPTO Operational Schedules & System Forecasts
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ipto_schedules (
            timestamp TIMESTAMP,
            unit_id VARCHAR,
            category VARCHAR,
            value_mw DOUBLE,
            PRIMARY KEY (timestamp, unit_id, category)
        );
    """)

    # 4. ENTSO-E Actuals (Solar, Wind, Total RES & Load)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entsoe_actuals (
            timestamp TIMESTAMP PRIMARY KEY,
            solar_mw DOUBLE,
            wind_onshore_mw DOUBLE,
            total_res_mw DOUBLE,
            actual_load_mw DOUBLE
        );
    """)

    # 5. IPTO ISP Reserve Capacity Prices (FCR / aFRR / mFRR)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ipto_reserve_prices (
           timestamp TIMESTAMP,
           reserve_type VARCHAR, -- 'FCR', 'aFRR', 'mFRR'
           price_up DOUBLE,
           price_down DOUBLE,
           PRIMARY KEY (timestamp, reserve_type)
         );
    """)
    
    conn.close()
    print("✅ Database schema initialized successfully.")

# ==========================================
# 2. DATA FETCHER & UPSERT HELPERS
# ==========================================

# --- A. ENTSO-E Actuals Ingestion ---
def sync_entsoe_actuals(start_date: str, end_date: str):
    """Fetches Actual Solar, Wind, Total RES, and Load from ENTSO-E API and upserts into DuckDB."""
    print(f"🌍 Syncing ENTSO-E Actuals ({start_date} to {end_date})...")
    client = EntsoePandasClient(api_key=ENTSOE_API_TOKEN)
    start_ts = pd.Timestamp(start_date, tz="Europe/Athens")
    end_ts = pd.Timestamp(end_date, tz="Europe/Athens") + pd.Timedelta(days=1)
    
    try:
        # 1. Fetch Actual Generation
        df_gen = client.query_generation("GR", start=start_ts, end=end_ts)
        
        solar_col = [c for c in df_gen.columns if "Solar" in str(c)]
        wind_col = [c for c in df_gen.columns if "Wind Onshore" in str(c) or "Wind" in str(c)]
        
        df_res = pd.DataFrame(index=df_gen.index)
        df_res["solar_mw"] = df_gen[solar_col].sum(axis=1) if solar_col else 0.0
        df_res["wind_onshore_mw"] = df_gen[wind_col].sum(axis=1) if wind_col else 0.0
        df_res["total_res_mw"] = df_res["solar_mw"] + df_res["wind_onshore_mw"]
        
        # 2. Fetch Actual Load
        df_load = client.query_load("GR", start=start_ts, end=end_ts)
        if isinstance(df_load, pd.Series):
            df_load = df_load.to_frame(name="actual_load_mw")
        elif isinstance(df_load, pd.DataFrame):
            df_load.columns = ["actual_load_mw"]
            
        # 3. Combine Generation & Load
        df_combined = df_res.join(df_load[["actual_load_mw"]], how="outer")
        
        # Convert timezone & handle DST duplicate timestamps in October
        df_combined.index = df_combined.index.tz_convert("Europe/Athens").tz_localize(None)
        df_combined = df_combined.groupby(level=0).mean()  # Deduplicate clock-change timestamps
        df_combined = df_combined.resample("15min").ffill().reset_index()
        df_combined = df_combined.rename(columns={"index": "timestamp"})
        
        df_final = df_combined[["timestamp", "solar_mw", "wind_onshore_mw", "total_res_mw", "actual_load_mw"]].dropna(subset=["timestamp"])
        for col in ["solar_mw", "wind_onshore_mw", "total_res_mw", "actual_load_mw"]:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce").fillna(0.0)
            
        conn = db_connection.connect()
        conn.execute("""
            INSERT INTO entsoe_actuals 
            SELECT * FROM df_final 
            ON CONFLICT (timestamp) DO UPDATE SET 
                solar_mw = EXCLUDED.solar_mw,
                wind_onshore_mw = EXCLUDED.wind_onshore_mw,
                total_res_mw = EXCLUDED.total_res_mw,
                actual_load_mw = EXCLUDED.actual_load_mw;
        """)
        conn.close()
        print(f"✅ ENTSO-E Actuals synced: {len(df_final)} records inserted/updated.")
    except Exception as e:
        print(f"❌ Failed syncing ENTSO-E Actuals: {e}")

# --- B. HEnEx DAM Ingestion ---
def fetch_henex_dam_mcp_for_date(target_date_str):
    urls_to_try = [
        f"https://www.enexgroup.gr/documents/20126/200106/{target_date_str}_EL-DAM_Results_EN_v01.xlsx",
        f"https://www.enexgroup.gr/documents/20126/200106/{target_date_str}_EL-DAM_Results_EN_v02.xlsx",
        f"https://www.enexgroup.gr/documents/20126/366820/{target_date_str}_EL-DAM_ResultsSummary_EN_v01.xlsx"
    ]
    for url in urls_to_try:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and len(r.content) > 1000:
                file_bytes = io.BytesIO(r.content)
                try:
                    df = pd.read_excel(file_bytes, sheet_name=0)
                except Exception:
                    file_bytes.seek(0)
                    df = pd.read_csv(file_bytes)
                
                df.columns = [str(c).strip() for c in df.columns]
                time_col = 'DELIVERY_MTU' if 'DELIVERY_MTU' in df.columns else ('MTU' if 'MTU' in df.columns else None)
                if time_col and 'MCP' in df.columns:
                    df_mcp = df.drop_duplicates(subset=[time_col], keep='first').copy()
                    df_mcp[time_col] = pd.to_datetime(df_mcp[time_col])
                    return df_mcp[[time_col, 'MCP']].rename(columns={'MCP': 'mcp_eur_mwh', time_col: 'timestamp'})
        except Exception:
            continue
    return pd.DataFrame()

def sync_henex_dam(start_date, end_date):
    print(f"🌐 Syncing HEnEx DAM Prices ({start_date} to {end_date})...")
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dfs = []
    
    for single_date in date_list:
        df_day = fetch_henex_dam_mcp_for_date(single_date.strftime("%Y%m%d"))
        if not df_day.empty:
            all_dfs.append(df_day)
            
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        conn = db_connection.connect()
        conn.execute("""
            INSERT INTO henex_dam 
            SELECT * FROM df_combined 
            ON CONFLICT (timestamp) DO UPDATE SET mcp_eur_mwh = EXCLUDED.mcp_eur_mwh;
        """)
        conn.close()
        print(f"✅ HEnEx DAM synced: {len(df_combined)} records inserted/updated.")

# --- C. HEnEx XBID Ingestion ---
def fetch_henex_xbid_storage_for_date(target_date_str):
    urls_to_try = [
        f"https://www.enexgroup.gr/documents/20126/1550281/{target_date_str}_EL-XBID_Results_EN_v01.xlsx",
        f"https://www.enexgroup.gr/documents/20126/1550281/{target_date_str}_EL-XBID_Results_EN_v02.xlsx"
    ]
    for url in urls_to_try:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and len(r.content) > 1000:
                file_bytes = io.BytesIO(r.content)
                df = pd.read_excel(file_bytes, sheet_name=0)
                df.columns = [str(c).strip() for c in df.columns]
                
                classification_col = next((c for c in df.columns if 'CLASSIFICATION' in c.upper()), None)
                side_col = next((c for c in df.columns if 'SIDE_DESCR' in c.upper()), None) or next((c for c in df.columns if 'SIDE' in c.upper()), None)
                vwap_col = next((c for c in df.columns if 'VWAP' in c.upper()), None)
                trades_col = next((c for c in df.columns if 'TOTAL_TRADES' in c.upper()), None)
                dt_col = next((c for c in df.columns if 'DELIVERY' in c.upper()), None)
                
                if classification_col and vwap_col and trades_col and dt_col:
                    df_storage = df[df[classification_col].astype(str).str.strip().str.upper() == 'STORAGE'].copy()
                    if not df_storage.empty:
                        df_clean = df_storage[[dt_col, side_col, classification_col, vwap_col, trades_col]].copy()
                        df_clean.columns = ['timestamp', 'side', 'classification', 'vwap_eur_mwh', 'total_trades_mw']
                        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
                        df_clean['side'] = df_clean['side'].astype(str).str.strip().str.upper()
                        df_clean['classification'] = df_clean['classification'].astype(str).str.strip().str.upper()
                        df_clean['vwap_eur_mwh'] = pd.to_numeric(df_clean['vwap_eur_mwh'], errors='coerce').fillna(0.0)
                        df_clean['total_trades_mw'] = pd.to_numeric(df_clean['total_trades_mw'], errors='coerce').fillna(0.0)
                        return df_clean
        except Exception:
            continue
    return pd.DataFrame()

def sync_henex_xbid(start_date, end_date):
    print(f"🌐 Syncing HEnEx XBID Storage Trades ({start_date} to {end_date})...")
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dfs = []
    
    for single_date in date_list:
        df_day = fetch_henex_xbid_storage_for_date(single_date.strftime("%Y%m%d"))
        if not df_day.empty:
            all_dfs.append(df_day)
            
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        conn = db_connection.connect()
        conn.execute("""
            INSERT INTO henex_xbid 
            SELECT * FROM df_combined 
            ON CONFLICT (timestamp, side, classification) DO UPDATE 
            SET vwap_eur_mwh = EXCLUDED.vwap_eur_mwh, total_trades_mw = EXCLUDED.total_trades_mw;
        """)
        conn.close()
        print(f"✅ HEnEx XBID synced: {len(df_combined)} records inserted/updated.")

# --- D. IPTO Ingestion ---
def is_primary_schedule_sheet(sheet_name: str) -> bool:
    s = sheet_name.strip().upper()
    excluded_keywords = ["FCR", "AFRR", "MFRR", "RR", "GENERICCONSTRAINTS", "ACTIVATED", "CCGT", "OFFERS"]
    return not any(k in s for k in excluded_keywords)

def extract_ipto_q_dataframe(file_url: str, filename: str, category_label: str) -> pd.DataFrame:
    if not file_url.startswith("http"):
        file_url = f"https://www.admie.gr{file_url}"

    try:
        res = requests.get(file_url, headers=HEADERS, timeout=60)
        res.raise_for_status()
        file_stream = io.BytesIO(res.content)
    except Exception:
        return pd.DataFrame()

    date_match = re.search(r"^(\d{4})(\d{2})(\d{2})_", filename)
    file_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else ""
    if not file_date:
        return pd.DataFrame()

    extracted_rows = []
    try:
        xls = pd.ExcelFile(file_stream)
        target_sheets = [s for s in xls.sheet_names if is_primary_schedule_sheet(s)] or [xls.sheet_names[0]]

        for sheet_name in target_sheets:
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            if df_raw.empty:
                continue

            for row_idx, row in df_raw.iterrows():
                for col_idx, cell_val in enumerate(row.values):
                    cell_str = str(cell_val).strip().upper()
                    
                    matched_unit = None
                    matched_category = category_label

                    # 1. Check for target BESS units
                    if cell_str in BESS_UNITS_UPPER:
                        matched_unit = cell_str
                    # 2. Check for System Forecast targets in ISP Requirements
                    elif cell_str in SYSTEM_FORECAST_TARGETS:
                        matched_unit = "SYSTEM_GR"
                        matched_category = SYSTEM_FORECAST_TARGETS[cell_str]

                    if matched_unit:
                        time_col_indices = []
                        for h_idx in range(max(0, row_idx - 5), row_idx):
                            h_vals = [str(v).strip() for v in df_raw.iloc[h_idx].values]
                            t_indices = [i for i, v in enumerate(h_vals) if re.match(r"^\d{2}:\d{2}(:\d{2})?$", v)]
                            if len(t_indices) >= 24:
                                time_col_indices = t_indices[:96]
                                break

                        base_dt = pd.Timestamp(file_date)
                        for q_idx in range(1, 97):
                            ts = base_dt + pd.Timedelta(minutes=15 * (q_idx - 1))

                            if time_col_indices and (q_idx - 1) < len(time_col_indices):
                                c_i = time_col_indices[q_idx - 1]
                                val = row.iloc[c_i]
                            else:
                                c_i = col_idx + q_idx
                                val = row.iloc[c_i] if c_i < len(row) else 0.0

                            try:
                                parsed_val = float(val) if pd.notna(val) and str(val).strip() != "" else 0.0
                            except (ValueError, TypeError):
                                parsed_val = 0.0

                            extracted_rows.append({
                                'timestamp': ts,
                                'unit_id': matched_unit,
                                'category': matched_category,
                                'value_mw': parsed_val
                            })
                        break
    except Exception:
        pass

    return pd.DataFrame(extracted_rows)

def extract_ipto_reserve_prices(file_url: str, filename: str) -> pd.DataFrame:
    if not file_url.startswith("http"):
        file_url = f"https://www.admie.gr{file_url}"

    try:
        res = requests.get(file_url, headers=HEADERS, timeout=60)
        res.raise_for_status()
        file_stream = io.BytesIO(res.content)
    except Exception:
        return pd.DataFrame()

    date_match = re.search(r"^(\d{4})(\d{2})(\d{2})_", filename)
    file_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else ""
    if not file_date:
        return pd.DataFrame()

    extracted_rows = []
    try:
        xls = pd.ExcelFile(file_stream)
        
        for sheet_name in xls.sheet_names:
            s_upper = sheet_name.upper()
            res_type = None
            if "FCR" in s_upper and "AFRR" not in s_upper and "MFRR" not in s_upper:
                res_type = "FCR"
            elif "AFRR" in s_upper:
                res_type = "aFRR"
            elif "MFRR" in s_upper:
                res_type = "mFRR"

            if not res_type:
                continue

            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            if df_raw.empty:
                continue

            # Row 4: Price Up, Row 5: Price Down
            row_up_idx, row_dn_idx = None, None
            for row_idx, row in df_raw.iterrows():
                r_str = " ".join([str(v).strip().upper() for v in row.values if pd.notna(v)])
                if "PRICE UP" in r_str:
                    row_up_idx = row_idx
                elif "PRICE DOWN" in r_str or "PRICE DN" in r_str:
                    row_dn_idx = row_idx

            if row_up_idx is None and row_dn_idx is None:
                row_up_idx, row_dn_idx = 4, 5

            base_dt = pd.Timestamp(file_date)
            for q_idx in range(1, 97):
                ts = base_dt + pd.Timedelta(minutes=15 * (q_idx - 1))
                
                try:
                    val_up = float(df_raw.iloc[row_up_idx, q_idx]) if row_up_idx is not None else 0.0
                except (ValueError, TypeError, IndexError):
                    val_up = 0.0

                try:
                    val_dn = float(df_raw.iloc[row_dn_idx, q_idx]) if row_dn_idx is not None else 0.0
                except (ValueError, TypeError, IndexError):
                    val_dn = 0.0

                extracted_rows.append({
                    'timestamp': ts,
                    'reserve_type': res_type,
                    'price_up': val_up,
                    'price_down': val_dn
                })
    except Exception:
        pass

    return pd.DataFrame(extracted_rows)

# --- Update in db_sync2.py ---

def sync_ipto(start_date, end_date):
    print(f"🏛️ Syncing IPTO Schedules & Reserve Prices ({start_date} to {end_date})...")
    
    # Updated category dictionary to pull v2 Requirements
    categories = {
        "ISP2_RESULTS_V1": "ISP2ISPResults",
        "ISP1_REQUIREMENTS_V2": "ISP1Requirements"  # IPTO API category endpoint for v2
    }
    
    all_rows = []
    reserve_rows = []
    
    for label, cat_code in categories.items():
        params = {"dateStart": start_date, "dateEnd": end_date, "FileCategory": cat_code}
        try:
            res = requests.get(IPTO_PUBLIC_API_URL, params=params, headers=HEADERS, timeout=30)
            records = res.json() if res.status_code == 200 and isinstance(res.json(), list) else []
            for record in records:
                file_url = record.get("file_path") or record.get("url") or record.get("fileUrl")
                if not file_url:
                    continue
                filename = file_url.split("/")[-1]
                
                # Check for version suffix: _02 (v2) or fallback to _01 if _02 isn't published yet
                match = re.search(r"_(\d+)\.(xlsx|xls|csv)$", filename, re.IGNORECASE)
                version_num = int(match.group(1)) if match else 1
                
                # Filter for v2 files (_02) for Requirements, keeping _01 for Results
                if (label == "ISP1_REQUIREMENTS_V2" and version_num == 2) or (label == "ISP2_RESULTS_V1" and version_num == 1):
                    df_parsed = extract_ipto_q_dataframe(file_url, filename, label)
                    if not df_parsed.empty:
                        all_rows.append(df_parsed)

                    if label == "ISP2_RESULTS_V1":
                        df_res_prices = extract_ipto_reserve_prices(file_url, filename)
                        if not df_res_prices.empty:
                            reserve_rows.append(df_res_prices)
        except Exception as e:
            print(f"⚠️ IPTO Sync Error [{label}]: {e}")

    conn = db_connection.connect()
    if all_rows:
        df_combined = pd.concat(all_rows, ignore_index=True)
        conn.execute("""
            INSERT INTO ipto_schedules 
            SELECT * FROM df_combined 
            ON CONFLICT (timestamp, unit_id, category) DO UPDATE SET value_mw = EXCLUDED.value_mw;
        """)

    if reserve_rows:
        df_res_combined = pd.concat(reserve_rows, ignore_index=True).drop_duplicates(subset=['timestamp', 'reserve_type'])
        conn.execute("""
            INSERT INTO ipto_reserve_prices 
            SELECT * FROM df_res_combined 
            ON CONFLICT (timestamp, reserve_type) DO UPDATE SET 
                price_up = EXCLUDED.price_up, 
                price_down = EXCLUDED.price_down;
        """)
        print(f"✅ IPTO Reserve Prices Synced: {len(df_res_combined)} records inserted/updated.")

    conn.close()
# ==========================================
# 3. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    init_db()
    
    # Dynamically calculate yesterday (D-1), today (D), and tomorrow (D+1)
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    
    start_str = yesterday.strftime("%Y-%m-%d")
    end_str = tomorrow.strftime("%Y-%m-%d")
    
    print(f"🚀 Running daily rolling database sync pipeline for interval: {start_str} to {end_str}\n")
    
    # Execute synchronization functions
    sync_entsoe_actuals(start_str, end_str)
    sync_henex_dam(start_str, end_str)
    sync_henex_xbid(start_str, end_str)
    sync_ipto(start_str, end_str)
    
    print("\n🎉 Daily database sync completed successfully!")
