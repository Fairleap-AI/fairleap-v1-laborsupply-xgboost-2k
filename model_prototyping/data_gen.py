import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define possible values
locations = ['Jakarta', 'Bandung', 'Surabaya', 'Semarang', 'Banten']
drivers = [f'driver_{i}' for i in range(1, 11)]  # 10 drivers

# --- MODIFICATION: Dictionary to track the last timestamp for each driver ---
last_driver_timestamps = {}

# Initialize last_driver_timestamps with a starting point for each driver
initial_start_time = datetime.now() - timedelta(days=30)
for driver_id in drivers:
    last_driver_timestamps[driver_id] = initial_start_time

# Generate dummy data
data = []
for driver_id in drivers:
    # current_time for this driver starts from their last known timestamp
    # or the initial start if it's their first set of sessions
    current_driver_session_time = last_driver_timestamps[driver_id]

    for _ in range(50):  # Each driver has 50 sessions
        # --- MODIFICATION: Ensure new timestamp is after the driver's last session ---
        # Add a random number of minutes (and seconds for more granularity)
        # to the driver's last recorded session time
        time_increment_minutes = random.randint(30, 240) # Sessions can be 30 mins to 4 hours apart
        time_increment_seconds = random.randint(1, 59)
        current_driver_session_time += timedelta(minutes=time_increment_minutes, seconds=time_increment_seconds)

        ts = current_driver_session_time  # This is now the unique timestamp for this session
        hour_of_day = ts.hour
        day_of_week = ts.weekday() # Monday=0, Sunday=6
        location = random.choice(locations)

        # Generate realistic features
        # Ensure hours_worked is plausible given the time between sessions
        # For simplicity, we'll keep it random but bounded.
        # A more complex model could base hours_worked on the next session's start time.
        hours_worked = round(random.uniform(1, 8), 2) # Max 8 hours per session for this model
        
        # Ensure hours_worked doesn't exceed time until next *potential* session start
        # This is a simplification; in reality, you'd know the session end time.
        # For now, we'll assume a session doesn't last longer than the typical minimum gap.
        hours_worked = min(hours_worked, (time_increment_minutes / 60) * 0.8) # e.g., work at most 80% of the gap
        hours_worked = round(max(0.5, hours_worked), 2) # Min 0.5 hours


        rides = random.randint(max(1, int(hours_worked * 1)), max(2, int(hours_worked * 5))) # 1-5 rides per hour
        
        # Earnings: base per ride + bonus for hours_worked (e.g. longer engagement)
        base_ride_earning = random.uniform(10000, 25000) # 10k-25k per ride
        hourly_bonus_factor = 1 + (hours_worked / 10) # Small bonus for longer hours
        earnings = round(rides * base_ride_earning * hourly_bonus_factor / 1000) * 1000 # Round to nearest 1000

        avg_ride_duration_minutes = 0
        if rides > 0:
            avg_ride_duration_minutes = round((hours_worked * 60) / rides, 2)
            
        wellness_score = int(random.uniform(0,100))

        data.append([
            driver_id,
            ts.strftime('%Y-%m-%d %H:%M:%S'),  # With seconds
            day_of_week,
            hour_of_day,
            location,
            hours_worked,
            rides,
            earnings,
            wellness_score,
            random.choice(locations),  # preferred_location (can be different from current)
            avg_ride_duration_minutes # In minutes for better readability
        ])
    
    # --- MODIFICATION: Update the global last timestamp for this driver ---
    last_driver_timestamps[driver_id] = current_driver_session_time


# Create DataFrame
columns = [
    'driver_id', 'timestamp', 'day_of_week', 'hour_of_day',
    'location_cluster', 'hours_worked', 'rides_completed',
    'earnings', 'wellness_score', 'preferred_location', 'avg_ride_duration_minutes'
]
df = pd.DataFrame(data, columns=columns)

# Convert to datetime and sort by driver + time (already implicitly sorted by driver, but good practice)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['driver_id', 'timestamp']).reset_index(drop=True)

# --- Verification (Optional) ---
def check_timestamp_uniqueness_per_driver(dataframe):
    all_unique = True
    for did, group in dataframe.groupby('driver_id'):
        if not group['timestamp'].is_unique:
            print(f"Timestamps are NOT unique for driver {did}")
            all_unique = False
        # Check if timestamps are sorted
        if not group['timestamp'].is_monotonic_increasing:
            print(f"Timestamps are NOT strictly increasing for driver {did}")
            all_unique = False # Or a different flag for sorted
    return all_unique

if check_timestamp_uniqueness_per_driver(df):
    print("✅ Timestamps are unique and strictly increasing per driver.")
else:
    print("⚠️ Issue with timestamp uniqueness or order per driver.")


# Save to CSV
df.to_csv('synthetic_driver_data.csv', index=False)

print("✅ Synthetic data generated with unique and sequential timestamps per driver, saved to 'synthetic_driver_data_v2.csv'")
print(df.head())