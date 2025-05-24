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

def make_prompt(query: str) -> str:
    """
    Generates a context-aware prompt based on user query.
    Matches query intent to predefined welfare scenarios.
    """
    query = query.lower()

    # --- Welfare Scenarios ---
    if "insurance" in query or "coverage" in query or "claim" in query:
        return """
        You are a helpful assistant providing Gojek driver insurance information.
        Answer the following question clearly and accurately:
        Question: {query}

        Provide:
        - What is typically covered by Gojek driver insurance
        - Steps to file a claim
        - Contact information or where to find more help
        """.strip().format(query=query)

    elif "tired" in query or "fatigue" in query or "sleepy" in query or "rest" in query:
        return """
        You are a helpful assistant giving fatigue prevention tips to drivers.
        Answer the following question clearly and compassionately:
        Question: {query}

        Include:
        - Signs of fatigue while driving
        - What to do immediately if tired
        - Preventive strategies for future trips
        """.strip().format(query=query)

    elif "save money" in query or "financial" in query or "budget" in query or "money" in query:
        return """
        You are a helpful assistant offering financial advice to Gojek drivers.
        Answer the following question clearly and practically:
        Question: {query}

        Include:
        - Practical steps to save money from daily earnings
        - Budgeting tips for gig workers
        - Recommended tools or apps (optional)
        """.strip().format(query=query)

    elif "weather" in query or "rain" in query or "storm" in query or "drive" in query:
        return """
        You are a helpful assistant giving weather-related driving advice to Gojek drivers.
        Answer the following question clearly and safely:
        Question: {query}

        Include:
        - Safety tips for driving in bad weather
        - When to consider stopping work
        - How to stay visible and safe
        """.strip().format(query=query)

    elif "traffic" in query or "jam" in query or "macet" in query:
        return """
        You are a helpful assistant giving traffic management advice to Gojek drivers.
        Answer the following question clearly:
        Question: {query}

        Include:
        - Tips for navigating heavy traffic
        - Best times to avoid peak congestion
        - Apps or tools to monitor real-time traffic
        """.strip().format(query=query)

    elif "health" in query or "sick" in query or "medical" in query:
        return """
        You are a helpful assistant giving health & wellness advice to Gojek drivers.
        Answer the following question clearly and empathetically:
        Question: {query}

        Include:
        - General guidance for staying healthy as a driver
        - What to do if feeling unwell while working
        - Resources available through Gojek or local services
        """.strip().format(query=query)

    elif "license" in query or "documents" in query or "SIM" in query or "KTP" in query:
        return """
        You are a helpful assistant guiding drivers on document requirements and renewals.
        Answer the following question clearly:
        Question: {query}

        Include:
        - Required documents for ride-hailing drivers
        - Where and how to renew expired licenses
        - What to do if documents are lost or stolen
        """.strip().format(query=query)

    elif "bike maintenance" in query or "motor maintenance" in query or "service" in query:
        return """
        You are a helpful assistant giving motorcycle maintenance tips to Gojek drivers.
        Answer the following question clearly:
        Question: {query}

        Include:
        - Basic maintenance schedule
        - Signs that something might be wrong
        - Where to get reliable service
                """.strip().format(query=query)

    else:
        # Default prompt for unknown or general questions
        return """
        You are a helpful assistant supporting Gojek drivers with welfare and safety information.
        Answer the following question politely and usefully:
        Question: {query}

        Try to provide actionable advice, even if it's general.
        """.strip().format(query=query)