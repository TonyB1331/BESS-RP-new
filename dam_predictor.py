# dam_predictor.py
import datetime
import duckdb
import db_connection
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DB_FILE = "greece_energy_market2.duckdb"

def fetch_energy_data(start_date="2025-01-01"):
    """
    Fetches DAM MCP, ENTSO-E Actual Load & RES, and IPTO ISP Forecast Load & RES
    directly from local DuckDB database (greece_energy_market2.duckdb).
    """
    conn = db_connection.connect(read_only=True)
    
    today = datetime.date.today()
    day_after_tomorrow = today + datetime.timedelta(days=2)
    end_date_str = day_after_tomorrow.strftime('%Y-%m-%d')
    
    start_ts = f"{start_date} 00:00:00"
    end_ts = f"{end_date_str} 23:45:00"

    # 1. Fetch HEnEx Day-Ahead Market Clearing Prices (MCP)
    df_dam = conn.execute(f"""
        SELECT timestamp, mcp_eur_mwh AS Greece_DAM_MCP_15min
        FROM henex_dam
        WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp
    """).df()

    # 2. Fetch ENTSO-E Actual Generation (Solar + Wind) & Load
    df_entsoe = conn.execute(f"""
        SELECT timestamp, 
               total_res_mw AS "ENTSOE Actual RES (Solar+Wind) Greece Production",
               actual_load_mw AS ENTSOE_Actual_Load_GR
        FROM entsoe_actuals
        WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp
    """).df()

    # 3. Fetch IPTO ISP Requirements Forecasts (System Load & System RES)
    df_ipto_fc = conn.execute(f"""
        SELECT timestamp, category, value_mw
        FROM ipto_schedules
        WHERE unit_id = 'SYSTEM_GR'
          AND timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
        ORDER BY timestamp
    """).df()

    conn.close()

    # Pivot IPTO Forecasts into columns
    if not df_ipto_fc.empty:
        df_ipto_pivot = df_ipto_fc.pivot(index='timestamp', columns='category', values='value_mw').reset_index()
    else:
        df_ipto_pivot = pd.DataFrame(columns=['timestamp', 'ISP1_LOAD_NON_DISPATCHABLE', 'ISP1_RES_NON_DISPATCHABLE'])

    # Map categories to predictor feature column names
    rename_map = {}
    if 'ISP1_LOAD_NON_DISPATCHABLE' in df_ipto_pivot.columns:
        rename_map['ISP1_LOAD_NON_DISPATCHABLE'] = 'ISP_Requirements_GREECE_15M_System_Load'
    if 'ISP1_RES_NON_DISPATCHABLE' in df_ipto_pivot.columns:
        rename_map['ISP1_RES_NON_DISPATCHABLE'] = 'ISP_Requirements_GREECE_15M_System_RES_Forecast'

    df_ipto_pivot = df_ipto_pivot.rename(columns=rename_map)

    # Clean missing forecast columns if table didn't have data
    for col in ['ISP_Requirements_GREECE_15M_System_Load', 'ISP_Requirements_GREECE_15M_System_RES_Forecast']:
        if col not in df_ipto_pivot.columns:
            df_ipto_pivot[col] = np.nan

    # 4. Merge all streams into a master time-series index
    df_merged = pd.merge(df_dam, df_entsoe, on='timestamp', how='outer')
    df_merged = pd.merge(df_merged, df_ipto_pivot, on='timestamp', how='outer')

    df_merged = df_merged.set_index('timestamp').sort_index()

    # Ensure all required target/feature columns exist
    required_cols = [
        'Greece_DAM_MCP_15min',
        'ISP_Requirements_GREECE_15M_System_Load',
        'ISP_Requirements_GREECE_15M_System_RES_Forecast',
        'ENTSOE Actual RES (Solar+Wind) Greece Production',
        'ENTSOE_Actual_Load_GR'
    ]
    for col in required_cols:
        if col not in df_merged.columns:
            df_merged[col] = np.nan

    # Linear interpolation for continuous time-series feature stability
    df_merged = df_merged[required_cols].interpolate(method='linear').ffill().bfill()

    return df_merged

def build_features(df):
    """Generates feature set, dynamic forecast deltas, calendar dimensions, and midday price signals."""
    df = df.copy()
    target = 'Greece_DAM_MCP_15min'
    f_load = 'ISP_Requirements_GREECE_15M_System_Load'
    f_res = 'ISP_Requirements_GREECE_15M_System_RES_Forecast'
    a_res = 'ENTSOE Actual RES (Solar+Wind) Greece Production'
    a_load = 'ENTSOE_Actual_Load_GR'
    
    # 1. Base Net Load & Solar Penetration Ratio
    df['forecast_net_load'] = df[f_load] - df[f_res]
    df['actual_net_load_lag_192'] = df[a_load].shift(192) - df[a_res].shift(192)
    df['res_penetration_ratio'] = df[f_res] / (df[f_load] + 1e-5)
    
    # 2. Dynamic Forecast Deltas
    # Delta vs Yesterday (1-day / 96 steps prior)
    df['delta_res_vs_actual_lag96'] = df[f_res] - df[a_res].shift(96)
    df['delta_load_vs_actual_lag96'] = df[f_load] - df[a_load].shift(96)
    df['delta_net_load_vs_actual_lag96'] = df['forecast_net_load'] - (df[a_load].shift(96) - df[a_res].shift(96))
    
    # Delta vs 2 Days Ago (2-day / 192 steps prior)
    df['delta_res_vs_actual_lag192'] = df[f_res] - df[a_res].shift(192)
    df['delta_load_vs_actual_lag192'] = df[f_load] - df[a_load].shift(192)
    df['delta_net_load_vs_actual_lag192'] = df['forecast_net_load'] - df['actual_net_load_lag_192']
    
    # 3. Base Target Lags
    df['target_lag_96'] = df[target].shift(96)     # Yesterday (Current day price signal)
    df['target_lag_192'] = df[target].shift(192)   # 2 days ago
    df['target_lag_672'] = df[target].shift(672)   # 1 week ago
    
    # 4. Actual Historical Lags
    df['actual_res_lag_192'] = df[a_res].shift(192)
    df['actual_load_lag_192'] = df[a_load].shift(192)
    df['actual_res_lag_288'] = df[a_res].shift(288)
    df['actual_load_lag_288'] = df[a_load].shift(288)
    
    # 5. Calendar Dimensions & Noon Indicators
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Noon Window Flag (10:00 AM to 16:45 PM solar cannibalization peak)
    df['is_noon_window'] = df['hour'].isin(range(10, 17)).astype(int)
    
    # 6. Yesterday's Midday & Low-Price Signals (Lag 96 Focus)
    # 1-hour rolling min & mean around yesterday's same time slot
    df['target_lag96_roll_min_1h'] = df['target_lag_96'].rolling(4, center=True).min()
    df['target_lag96_roll_mean_1h'] = df['target_lag_96'].rolling(4, center=True).mean()
    
    # Minimum price yesterday during midday window (forward-filled across the date)
    noon_mask = df['hour'].isin(range(10, 17))
    df['yesterday_noon_min_price'] = np.nan
    df.loc[noon_mask, 'yesterday_noon_min_price'] = df.loc[noon_mask, 'target_lag_96']
    df['yesterday_noon_min_price'] = df['yesterday_noon_min_price'].groupby(df.index.date).transform('min')
    
    # Indicator for Zero or Near-Zero prices yesterday
    df['target_lag96_is_zero_or_neg'] = (df['target_lag_96'] <= 0.1).astype(int)
    
    # Noon Price & Penetration Interaction Terms
    df['noon_x_target_lag96'] = df['target_lag_96'] * df['is_noon_window']
    df['noon_x_res_penetration'] = df['res_penetration_ratio'] * df['is_noon_window']
    
    return df, target

def generate_dam_forecast():
    """Trains a LightGBM regressor to output tomorrow's 15-minute price predictions."""
    raw_df = fetch_energy_data(start_date="2025-01-01")
    df_feat, target_col = build_features(raw_df)
    
    features = [
        'ISP_Requirements_GREECE_15M_System_Load', 
        'ISP_Requirements_GREECE_15M_System_RES_Forecast',
        'forecast_net_load', 
        'res_penetration_ratio',
        
        # Target Price Lags
        'target_lag_96', 
        'target_lag_192', 
        'target_lag_672',
        
        # Yesterday Midday & Low-Price Features
        'is_noon_window',
        'target_lag96_roll_min_1h',
        'target_lag96_roll_mean_1h',
        'yesterday_noon_min_price',
        'target_lag96_is_zero_or_neg',
        'noon_x_target_lag96',
        'noon_x_res_penetration',
        
        # Forecast Delta Features
        'delta_res_vs_actual_lag96', 'delta_load_vs_actual_lag96', 'delta_net_load_vs_actual_lag96',
        'delta_res_vs_actual_lag192', 'delta_load_vs_actual_lag192', 'delta_net_load_vs_actual_lag192',
        
        # Actual Lags
        'actual_res_lag_192', 'actual_load_lag_192', 
        'actual_res_lag_288', 'actual_load_lag_288',
        
        # Calendar Features
        'hour', 'minute', 'day_of_week', 'month', 'is_weekend'
    ]
    
    tomorrow_date = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_start = pd.Timestamp(tomorrow_date)
    tomorrow_end = tomorrow_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    
    df_history = df_feat[df_feat.index < tomorrow_start]
    df_train_clean = df_history.dropna(subset=[target_col] + features)
    
    X = df_train_clean[features]
    y = df_train_clean[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # Sample Weighting: Give 2.0x weight to midday hours during training
    train_weights = np.where(X_train['is_noon_window'] == 1, 2.0, 1.0)
    val_weights = np.where(X_val['is_noon_window'] == 1, 2.0, 1.0)
    
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
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val, weight=val_weights, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    df_tomorrow = df_feat.loc[tomorrow_start:tomorrow_end]
    if len(df_tomorrow) == 0:
        raise ValueError(f"No records found for tomorrow ({tomorrow_date}) in the dataset.")
        
    df_tomorrow_clean = df_tomorrow.ffill().bfill()
    X_tomorrow = df_tomorrow_clean[features]
    
    tomorrow_preds = model.predict(X_tomorrow)
    
    df_output = pd.DataFrame({
        'Price_Prediction_EUR_MWh': tomorrow_preds
    }, index=df_tomorrow_clean.index)
    
    return df_output, tomorrow_date

# Standalone execution check
if __name__ == "__main__":
    print("Running standalone prediction test using DuckDB...")
    try:
        df_out, target_dt = generate_dam_forecast()
        print(f"Success! Generated {len(df_out)} predictions for {target_dt}.")
        
        # Exporting predictions to Excel
        excel_filename = f"Greece_DAM_Price_Forecast_{target_dt}.xlsx"
        df_out.to_excel(excel_filename, index_label="Timestamp")
        print(f"Successfully exported predictions to '{excel_filename}'.")
        
        print("\nFirst 5 rows of output:")
        print(df_out.head())
    except Exception as e:
        print(f"Error during standalone execution: {e}")