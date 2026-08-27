import streamlit as st
import pandas as pd
import duckdb
import db_connection

st.set_page_config(layout="wide", page_title="DuckDB Data Explorer")

st.title("🗃️ Local DuckDB Time-Series Explorer")
st.caption("Search, filter, and inspect exact values stored inside 'greece_energy_market2.duckdb'")
st.markdown("---")

ASSET_CAPACITIES_MW = {
    "STO_BZ1_01_SP": 17.0, "STO_BZ1_02_SP": 16.0, "MYT_BZ1_01_SB": 35.0,
    "MYT_BZ1_02_SB": 300.0, "SUS_BZ1_01_SB": 22.0, "ENE_BZ1_01_SB": 49.0,
    "SUS_BZ1_02_SB": 30.0, "SUS_BZ1_03_SB": 20.0, "PPC_BZ1_01_SB": 1.0,
    "PPC_BZ1_02_SB": 1.0, "STO_BZ1_03_SP": 1.0, "OPT_BZ1_01_SB": 49.9,
}

# 1. Selection Controls
col_tbl, col_d1, col_d2 = st.columns([1.5, 1, 1])

with col_tbl:
    selected_table = st.selectbox(
        "Select Table / Stream", 
        ["henex_dam", "henex_xbid", "ipto_reserve_prices", "ipto_schedules", "ipto_system_forecasts", "entsoe_actuals"],
        key="standalone_table_select"
    )

with col_d1:
    search_start = st.date_input("Search From", value=pd.Timestamp("2026-08-01"), key="standalone_search_start")
with col_d2:
    search_end = st.date_input("Search To", value=pd.Timestamp.now().date(), key="standalone_search_end")

# 2. Dynamic Table Filters
if selected_table == "ipto_reserve_prices":
    st.subheader("🛡️ ISP Reserve Capacity Clearing Prices")
    
    # Separate tabs for each reserve type
    tab_fcr, tab_afrr, tab_mfrr, tab_all = st.tabs(["⚡ FCR Prices", "🔄 aFRR Prices", "📊 mFRR Prices", "📋 All Reserves"])
    
    res_configs = [
        ("FCR", tab_fcr),
        ("aFRR", tab_afrr),
        ("mFRR", tab_mfrr),
        ("ALL", tab_all)
    ]
    
    for r_type, tab_obj in res_configs:
        with tab_obj:
            where_clause = f"AND reserve_type = '{r_type}'" if r_type != "ALL" else ""
            query = f"""
                SELECT timestamp AS "Timestamp",
                       reserve_type AS "Reserve Type",
                       price_up AS "Price Up (€/MW)",
                       price_down AS "Price Down (€/MW)"
                FROM ipto_reserve_prices
                WHERE timestamp >= '{search_start} 00:00:00' 
                  AND timestamp <= '{search_end} 23:45:00' {where_clause}
                ORDER BY timestamp ASC, reserve_type ASC
            """
            try:
                conn = db_connection.connect(read_only=True)
                df_res = conn.execute(query).df()
                conn.close()

                if not df_res.empty:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total MTUs", f"{len(df_res):,}")
                    m2.metric("Avg Price Up", f"€{df_res['Price Up (€/MW)'].mean():.2f}/MW")
                    m3.metric("Avg Price Down", f"€{df_res['Price Down (€/MW)'].mean():.2f}/MW")

                    st.dataframe(df_res, use_container_width=True, height=450)

                    csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"⬇️ Export {r_type} Reserve Prices (.csv)",
                        data=csv_bytes,
                        file_name=f"ISP_Reserve_Prices_{r_type}_{search_start}_to_{search_end}.csv",
                        mime="text/csv",
                        key=f"btn_dl_{r_type}"
                    )
                else:
                    st.info(f"No {r_type} price records found for the selected date range.")
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

else:
    with st.container(border=True):
        if selected_table == "henex_dam":
            min_p, max_p = st.slider("MCP Range (€/MWh)", -100.0, 500.0, (-100.0, 500.0), key="standalone_dam_slider")
            query = f"""
                SELECT timestamp, mcp_eur_mwh 
                FROM henex_dam 
                WHERE timestamp >= '{search_start} 00:00:00' 
                  AND timestamp <= '{search_end} 23:45:00'
                  AND mcp_eur_mwh BETWEEN {min_p} AND {max_p}
                ORDER BY timestamp ASC
            """

        elif selected_table == "ipto_schedules":
            sel_unit = st.selectbox("Select Storage Unit", options=list(ASSET_CAPACITIES_MW.keys()), key="standalone_ipto_unit")
            query = f"""
                SELECT timestamp, unit_id, category, value_mw 
                FROM ipto_schedules 
                WHERE unit_id = '{sel_unit}'
                  AND timestamp >= '{search_start} 00:00:00' 
                  AND timestamp <= '{search_end} 23:45:00'
                ORDER BY timestamp ASC
            """

        elif selected_table == "ipto_system_forecasts":
            sel_fc = st.selectbox(
                "Select Forecast Series", 
                options=["ALL", "ISP1_LOAD_NON_DISPATCHABLE", "ISP1_RES_NON_DISPATCHABLE", "ISP1_RES_RESFIT_PORTFOLIO", "ISP1_LOAD_LOSSES"],
                key="standalone_ipto_fc"
            )
            cat_clause = f"AND category = '{sel_fc}'" if sel_fc != "ALL" else ""
            query = f"""
                SELECT timestamp, unit_id, category, value_mw 
                FROM ipto_schedules 
                WHERE unit_id = 'SYSTEM_GR'
                  AND timestamp >= '{search_start} 00:00:00' 
                  AND timestamp <= '{search_end} 23:45:00' {cat_clause}
                ORDER BY timestamp ASC
            """

        elif selected_table == "henex_xbid":
            sel_side = st.radio("Side", ["ALL", "BUY", "SELL"], horizontal=True, key="standalone_xbid_side")
            side_clause = f"AND side = '{sel_side}'" if sel_side != "ALL" else ""
            query = f"""
                SELECT timestamp, side, classification, vwap_eur_mwh, total_trades_mw 
                FROM henex_xbid 
                WHERE timestamp >= '{search_start} 00:00:00' 
                  AND timestamp <= '{search_end} 23:45:00' {side_clause}
                ORDER BY timestamp ASC
            """

        else:  # entsoe_actuals
            query = f"""
                SELECT timestamp, solar_mw, wind_onshore_mw, total_res_mw, actual_load_mw 
                FROM entsoe_actuals 
                WHERE timestamp >= '{search_start} 00:00:00' 
                  AND timestamp <= '{search_end} 23:45:00'
                ORDER BY timestamp ASC
            """

    try:
        conn = db_connection.connect(read_only=True)
        df_search_results = conn.execute(query).df()
        conn.close()

        col_meta, col_btn = st.columns([3, 1])
        with col_meta:
            st.markdown(f"**Found `{len(df_search_results):,}` matching records**")
        with col_btn:
            if not df_search_results.empty:
                csv_bytes = df_search_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Export Filtered Query (.csv)",
                    data=csv_bytes,
                    file_name=f"{selected_table}_export_{search_start}_to_{search_end}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="standalone_dl_btn"
                )

        st.dataframe(df_search_results, use_container_width=True, height=550)

    except Exception as e:
        st.error(f"❌ Failed executing DuckDB query: {e}")