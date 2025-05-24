import pandas as pd
import numpy as np

def generate_features_for_forecast(df_hist, forecast_start, forecast_end, driver_id):
    """
    Generate features for the requested forecast window.
    Uses historical earnings to create lags and rolling stats.
    Returns a DataFrame with one row per day.
    """
    # Create date range for prediction
    date_range = pd.date_range(start=forecast_start, end=forecast_end, freq='h')

    # Start from last known values
    last_known_row = df_hist.iloc[-1].copy()
    last_known_date = df_hist.index[-1]

    # Initialize new dataframe
    df_pred = pd.DataFrame(index=date_range)
    df_pred.index.name = 'timestamp'

    # Generate time-based features
    # df_pred['month'] = df_pred.index.month
    df_pred['day_of_week'] = df_pred.index.dayofweek
    df_pred['hour_of_day'] = df_pred.index.hour
    df_pred['is_weekend'] = df_pred['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Copy historical earnings into new column
    df_pred['earnings'] = np.nan  # Placeholder for predicted values

    # Rolling features
    df_pred['rolling_mean_7'] = np.nan
    df_pred['rolling_std_7'] = np.nan
    df_pred['rolling_mean_14'] = np.nan

    # Fill lagged features using historical + predicted values
    for lag in range(1, 15):  # up to lag_14
        col_name = f'lag_{lag}'
        df_pred[col_name] = df_pred.index.map(
            lambda dt: df_hist.loc[df_hist.index <= dt - pd.Timedelta(days=lag), 'earnings'].iloc[-1]
            if (dt - pd.Timedelta(days=lag)) >= df_hist.index[0] else np.nan
        )

    # Simulate walk-forward prediction
    history = df_hist.copy()

    for idx in df_pred.index:
        # Get previous 14 days of earnings
        past_values = history.loc[history.index <= idx, 'earnings'].values[-14:]
        if len(past_values) >= 7:
            df_pred.loc[idx, 'rolling_mean_7'] = np.mean(past_values[-7:])
            df_pred.loc[idx, 'rolling_std_7'] = np.std(past_values[-7:])
        if len(past_values) >= 14:
            df_pred.loc[idx, 'rolling_mean_14'] = np.mean(past_values[-14:])

        # Set lag features manually based on past earnings
        for lag in range(1, 15):
            lag_idx = idx - pd.Timedelta(days=lag)
            val = history.loc[history.index <= lag_idx, 'earnings'].iloc[-1] if not history.empty else np.nan
            df_pred.loc[idx, f'lag_{lag}'] = val

    return df_pred