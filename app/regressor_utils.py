import pandas as pd
import numpy as np

def generate_features_for_forecast(hist_json, forecast_start, forecast_end, wellness_score):
    """
    Generate features for the requested forecast window.
    Uses historical earnings to create lags and rolling stats.
    Returns a DataFrame with one row per day.
    """
    # Create date range for prediction
    date_range = pd.date_range(start=forecast_start, end=forecast_end, freq='D')

    # Start from last known values
    df_hist = pd.DataFrame(hist_json)
    if not df_hist.empty:
        df_hist["day"] = pd.to_datetime(df_hist["day"], format="%Y-%m-%d")
        df_hist.set_index("day", inplace=True)
        df_hist.sort_index(inplace=True)
        # These are not used later in the function as provided, but good practice to handle
        # last_known_row = df_hist.iloc[-1].copy()
        # last_known_date = df_hist.index[-1]
    
    # print(df_hist) # Original print statement

    # Initialize new dataframe
    df_pred = pd.DataFrame(index=date_range)
    df_pred.index.name = 'timestamp'

    df_pred['earnings'] = np.nan 
    # Generate time-based features
    df_pred['day_of_week'] = df_pred.index.dayofweek
    df_pred['is_weekend'] = df_pred['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_pred["wellness_score"] = wellness_score
    
    # Initialize placeholder for earnings and feature columns to NaN
    df_pred['rolling_mean_7'] = np.nan
    df_pred['rolling_std_7'] = np.nan
    df_pred['rolling_mean_14'] = np.nan
    for lag in range(1, 15):
        df_pred[f'lag_{lag}'] = np.nan

    history = df_hist.copy() # Use a copy of the historical data

    # Cache the minimum historical date if history is not empty
    min_hist_date = history.index[0] if not history.empty else None

    for idx in df_pred.index:
        # --- Rolling features calculation ---
        if not history.empty:
            all_historical_earnings = history['total_earnings'] # This is a Series

            # Rolling mean/std for 7 days
            if len(all_historical_earnings) >= 7:
                df_pred.loc[idx, 'rolling_mean_7'] = np.mean(all_historical_earnings.iloc[-7:])
                df_pred.loc[idx, 'rolling_std_7'] = np.std(all_historical_earnings.iloc[-7:])
            # Else, they remain np.nan (initialized earlier)

            # Rolling mean for 14 days
            if len(all_historical_earnings) >= 14:
                df_pred.loc[idx, 'rolling_mean_14'] = np.mean(all_historical_earnings.iloc[-14:])
            # Else, it remains np.nan (initialized earlier)
        # If history is empty, rolling features remain np.nan as initialized.

        # --- Lag features calculation (Corrected) ---
        for lag in range(1, 15):
            col_name = f'lag_{lag}'
            val = np.nan # Default to NaN

            if not history.empty and min_hist_date is not None:
                lag_idx = idx - pd.Timedelta(days=lag)
                
                # Check if the lag_idx is before the earliest date in history
                if lag_idx >= min_hist_date:
                    # Filter history for dates less than or equal to lag_idx
                    lag_values_series = history.loc[history.index <= lag_idx, 'total_earnings']
                    if not lag_values_series.empty:
                        val = lag_values_series.iloc[-1] # Get the latest value up to lag_idx
                    # Else (lag_values_series is empty): val remains np.nan
                    # This case means lag_idx is >= min_hist_date but the slice is empty (e.g., gaps in history).
                # Else (lag_idx < min_hist_date): val remains np.nan
            # Else (history is empty or min_hist_date is None): val remains np.nan
            
            df_pred.loc[idx, col_name] = val
            
    return df_pred