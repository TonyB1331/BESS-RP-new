import datetime
import duckdb
import db_connection
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DB_FILE = "greece_energy_market2.duckdb"


def fetch_reserve_market_data_db(start_date="2025-01-01"):
    """
    Fetches DAM prices, grid fundamentals (Load/RES actuals & forecasts), 
    and all 6 reserve capacity price time series directly from DuckDB.
    """
    conn = db_connection.connect(read_only=True)
    
    today = datetime.date.today()
    day_after_tomorrow = today + datetime.timedelta(days=2)
    start_ts = f"{start_date} 00:00:00"
    end_ts = f"{day_after_tomorrow.strftime('%Y-%m-%d')} 23:45:00"

    # 1. Fetch HEnEx Day-Ahead Market Clearing Prices (MCP)
    df_dam = conn.execute(f"""
        SELECT timestamp, mcp_eur_mwh AS Greece_DAM_MCP_15min
        FROM henex_dam
        WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp ASC
    """).df()

    # 2. Fetch ENTSO-E Actual RES & Load
    df_entsoe = conn.execute(f"""
        SELECT 
            timestamp, 
            total_res_mw AS "ENTSOE Actual RES (Solar+Wind) Greece Production",
            actual_load_mw AS ENTSOE_Actual_Load_GR
        FROM entsoe_actuals
        WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp ASC
    """).df()

    # 3. Fetch IPTO System Forecasts (ISP1 Non-Dispatchable Load & RES)
    df_ipto_fc = conn.execute(f"""
        SELECT timestamp, category, value_mw
        FROM ipto_schedules
        WHERE unit_id = 'SYSTEM_GR'
          AND category IN ('ISP1_LOAD_NON_DISPATCHABLE', 'ISP1_RES_NON_DISPATCHABLE')
          AND timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp ASC
    """).df()

    if not df_ipto_fc.empty:
        df_sys_fc = df_ipto_fc.pivot(index='timestamp', columns='category', values='value_mw').reset_index()
        df_sys_fc = df_sys_fc.rename(columns={
            'ISP1_LOAD_NON_DISPATCHABLE': 'ISP_Requirements_GREECE_15M_System_Load',
            'ISP1_RES_NON_DISPATCHABLE': 'ISP_Requirements_GREECE_15M_System_RES_Forecast'
        })
    else:
        df_sys_fc = pd.DataFrame(columns=['timestamp', 'ISP_Requirements_GREECE_15M_System_Load', 'ISP_Requirements_GREECE_15M_System_RES_Forecast'])

    # 4. Fetch IPTO Reserve Capacity Prices (FCR, aFRR, mFRR Up/Down)
    df_res_prices = conn.execute(f"""
        SELECT timestamp, reserve_type, price_up, price_down
        FROM ipto_reserve_prices
        WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp ASC
    """).df()

    conn.close()

    # Pivot reserve prices into individual target columns
    if not df_res_prices.empty:
        piv_up = df_res_prices.pivot(index='timestamp', columns='reserve_type', values='price_up')
        piv_dn = df_res_prices.pivot(index='timestamp', columns='reserve_type', values='price_down')
        
        piv_up.columns = [f"ISP_Results_GREECE_15M_Reserve_Prices_{c}_Up" for c in piv_up.columns]
        piv_dn.columns = [f"ISP_Results_GREECE_15M_Reserve_Prices_{c}_Down" for c in piv_dn.columns]
        
        df_res_pivoted = pd.concat([piv_up, piv_dn], axis=1).reset_index()
    else:
        df_res_pivoted = pd.DataFrame()

    # Create master 15-minute time continuum
    timeline = pd.date_range(start=start_ts, end=end_ts, freq='15min')
    df_master = pd.DataFrame({'timestamp': timeline})

    # Merge all datasets into master DataFrame
    for df_sub in [df_dam, df_entsoe, df_sys_fc, df_res_pivoted]:
        if not df_sub.empty and 'timestamp' in df_sub.columns:
            df_master = pd.merge(df_master, df_sub, on='timestamp', how='left')

    df_master = df_master.set_index('timestamp').sort_index()

    # Ensure all target and baseline columns exist
    expected_cols = [
        'Greece_DAM_MCP_15min',
        'ISP_Requirements_GREECE_15M_System_Load',
        'ISP_Requirements_GREECE_15M_System_RES_Forecast',
        'ENTSOE Actual RES (Solar+Wind) Greece Production',
        'ENTSOE_Actual_Load_GR',
        'ISP_Results_GREECE_15M_Reserve_Prices_FCR_Up',
        'ISP_Results_GREECE_15M_Reserve_Prices_FCR_Down',
        'ISP_Results_GREECE_15M_Reserve_Prices_aFRR_Up',
        'ISP_Results_GREECE_15M_Reserve_Prices_aFRR_Down',
        'ISP_Results_GREECE_15M_Reserve_Prices_mFRR_Up',
        'ISP_Results_GREECE_15M_Reserve_Prices_mFRR_Down'
    ]
    
    for c in expected_cols:
        if c not in df_master.columns:
            df_master[c] = np.nan

    df_master = df_master.interpolate(method='linear').ffill().bfill()
    return df_master


def build_reserve_features(df):
    """Generates feature sets including grid fundamentals, temporal signals, and target-specific lags."""
    df = df.copy()
    
    f_load = 'ISP_Requirements_GREECE_15M_System_Load'
    f_res = 'ISP_Requirements_GREECE_15M_System_RES_Forecast'
    a_res = 'ENTSOE Actual RES (Solar+Wind) Greece Production'
    a_load = 'ENTSOE_Actual_Load_GR'
    
    # Grid Fundamental Features
    df['forecast_net_load'] = df[f_load] - df[f_res]
    df['forecast_res_ramp'] = df[f_res].diff()
    
    df['actual_res_lag_192'] = df[a_res].shift(192)
    df['actual_load_lag_192'] = df[a_load].shift(192)
    df['actual_res_lag_288'] = df[a_res].shift(288)
    df['actual_load_lag_288'] = df[a_load].shift(288)
    
    # Calendar / Temporal Position
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Define Target Columns
    targets = [
        'ISP_Results_GREECE_15M_Reserve_Prices_FCR_Up',
        'ISP_Results_GREECE_15M_Reserve_Prices_FCR_Down',
        'ISP_Results_GREECE_15M_Reserve_Prices_aFRR_Up',
        'ISP_Results_GREECE_15M_Reserve_Prices_aFRR_Down',
        'ISP_Results_GREECE_15M_Reserve_Prices_mFRR_Up',
        'ISP_Results_GREECE_15M_Reserve_Prices_mFRR_Down'
    ]
    
    # Construct target-specific historical lags (24h, 48h, 7d)
    for tgt in targets + ['Greece_DAM_MCP_15min']:
        df[f'{tgt}_lag_96'] = df[tgt].shift(96)
        df[f'{tgt}_lag_192'] = df[tgt].shift(192)
        df[f'{tgt}_lag_672'] = df[tgt].shift(672)
    
    return df, targets


def generate_reserve_forecasts():
    """Trains 6 dedicated LightGBM regressors to output tomorrow's 15-minute reserve capacity price predictions."""
    raw_df = fetch_reserve_market_data_db(start_date="2025-01-01")
    df_feat, target_cols = build_reserve_features(raw_df)
    
    # Base feature set shared across models
    base_features = [
        'ISP_Requirements_GREECE_15M_System_Load', 
        'ISP_Requirements_GREECE_15M_System_RES_Forecast',
        'forecast_net_load', 'forecast_res_ramp',
        'actual_res_lag_192', 'actual_load_lag_192', 
        'actual_res_lag_288', 'actual_load_lag_288',
        'hour', 'minute', 'day_of_week', 'month', 'is_weekend',
        'Greece_DAM_MCP_15min_lag_96', 'Greece_DAM_MCP_15min_lag_192', 'Greece_DAM_MCP_15min_lag_672'
    ]
    
    tomorrow_date = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_start = pd.Timestamp(tomorrow_date)
    tomorrow_end = tomorrow_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    
    df_history = df_feat[df_feat.index < tomorrow_start]
    df_tomorrow = df_feat.loc[tomorrow_start:tomorrow_end]
    
    if len(df_tomorrow) == 0:
        raise ValueError(f"No records found for tomorrow ({tomorrow_date}) in DuckDB.")
        
    df_tomorrow_clean = df_tomorrow.ffill().bfill()
    predictions_dict = {}
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1
    }
    
    # Loop and train 6 LightGBM models (1 per reserve product)
    for target in target_cols:
        target_features = base_features + [
            f'{target}_lag_96', f'{target}_lag_192', f'{target}_lag_672'
        ]
        
        df_train_clean = df_history.dropna(subset=[target] + target_features)
        
        X = df_train_clean[target_features]
        y = df_train_clean[target]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        X_tomorrow = df_tomorrow_clean[target_features]
        preds = model.predict(X_tomorrow)
        
        # Format display column name
        col_name = target.replace('ISP_Results_GREECE_15M_Reserve_Prices_', '') + '_Price_EUR_MW'
        predictions_dict[col_name] = np.maximum(0, preds)  # Floor negative predictions at zero
        
    df_output = pd.DataFrame(predictions_dict, index=df_tomorrow_clean.index)
    return df_output, tomorrow_date


if __name__ == "__main__":
    print("Running standalone reserve capacity price prediction test via DuckDB...")
    try:
        df_out, target_dt = generate_reserve_forecasts()
        print(f"Success! Generated {len(df_out)} predictions for all 6 reserves on {target_dt}.\n")
        print(df_out.head())
    except Exception as e:
        print(f"Error during execution: {e}")