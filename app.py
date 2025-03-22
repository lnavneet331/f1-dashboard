import streamlit as st
import pandas as pd
import plotly.express as px
import time
from utils import (
    get_current_session,
    get_timing_data,
    get_car_data,
    get_race_control,
    get_weather_data
)

st.set_page_config(page_title="F1 Live Dashboard", layout="wide")

st.title("🏎️ Formula 1 Live Dashboard")

# Sidebar for session information
with st.sidebar:
    st.header("Session Information")
    current_session = get_current_session()
    if current_session:
        st.write(f"**Session:** {current_session.get('session_name', 'N/A')}")
        st.write(f"**Type:** {current_session.get('session_type', 'N/A')}")
        st.write(f"**Status:** {current_session.get('session_status', 'N/A')}")
    else:
        st.warning("No active session found")

# Main dashboard layout
if current_session:
    session_key = current_session.get('session_key')
    
    # Create three columns for the main metrics
    col1, col2, col3 = st.columns(3)
    
    # Timing Data
    with col1:
        st.subheader("Timing Data")
        timing_df = get_timing_data(session_key)
        if not timing_df.empty:
            st.dataframe(timing_df[['driver_number', 'lap_number', 'last_lap', 'gap_to_leader']].head(10))
    
    # Car Data
    with col2:
        st.subheader("Car Data")
        car_df = get_car_data(session_key)
        if not car_df.empty:
            st.dataframe(car_df[['driver_number', 'speed', 'rpm', 'gear']].head(10))
    
    # Weather Data
    with col3:
        st.subheader("Weather Conditions")
        weather_df = get_weather_data(session_key)
        if not weather_df.empty:
            latest_weather = weather_df.iloc[-1]
            st.metric("Track Temperature", f"{latest_weather.get('track_temperature', 'N/A')}°C")
            st.metric("Air Temperature", f"{latest_weather.get('air_temperature', 'N/A')}°C")
    
    # Race Control Messages
    st.subheader("Race Control Messages")
    race_control_df = get_race_control(session_key)
    if not race_control_df.empty:
        st.dataframe(race_control_df[['message', 'timestamp']].tail(5))
    
    # Auto-refresh every 5 seconds
    st.empty()
    time.sleep(5)
    st.experimental_rerun()
else:
    st.warning("No active F1 session is currently running. Please check back during race weekends.") 