import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://api.openf1.org/v1"

def get_current_session():
    """Get the current F1 session information."""
    today = datetime.now().strftime("%Y-%m-%d")
    response = requests.get(f"{BASE_URL}/sessions", params={"date": today})
    if response.status_code == 200:
        sessions = response.json()
        if sessions:
            return sessions[0]  # Return the first session of the day
    return None

def get_timing_data(session_key):
    """Get real-time timing data for the current session."""
    response = requests.get(f"{BASE_URL}/timing_data", params={"session_key": session_key})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()

def get_car_data(session_key):
    """Get real-time car data for the current session."""
    response = requests.get(f"{BASE_URL}/car_data", params={"session_key": session_key})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()

def get_race_control(session_key):
    """Get race control messages."""
    response = requests.get(f"{BASE_URL}/race_control", params={"session_key": session_key})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()

def get_weather_data(session_key):
    """Get weather data for the current session."""
    response = requests.get(f"{BASE_URL}/weather", params={"session_key": session_key})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame() 