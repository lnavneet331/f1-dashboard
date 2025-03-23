import streamlit as st
import requests
import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://api.openf1.org/v1"
CURRENT_YEAR = 2024  # Focus on 2024 season

# Cache for API responses
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_meetings(year: int) -> List[Dict]:
    """Fetch meetings (races) for a specific year with caching."""
    try:
        logger.info(f"Fetching meetings for year {year}")
        response = requests.get(
            f"{BASE_URL}/meetings",
            params={
                "year": year,
            }
        )
        response.raise_for_status()
        meetings = response.json()
        logger.info(f"Found {len(meetings)} meetings in schedule")
        if meetings:
            logger.info(f"First meeting data: {meetings[0]}")
        return meetings
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching meetings for year {year}: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_seasons() -> List[int]:
    """Fetch available F1 seasons (hardcoded to 2024 for now)."""
    try:
        logger.info("Fetching available seasons")
        # Hardcoded to 2024 as the API doesn't provide a seasons endpoint
        return [2024]
    except Exception as e:
        logger.error(f"Error fetching seasons: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_rounds(year: int) -> List[Dict]:
    """Fetch all rounds (meetings) for a specific season, sorted by date."""
    try:
        logger.info(f"Fetching rounds for year {year}")
        meetings = fetch_meetings(year)
        if meetings:
            # Sort meetings by date_start and assign round numbers
            sorted_meetings = sorted(meetings, key=lambda x: x.get('date_start', ''))
            for i, meeting in enumerate(sorted_meetings, 1):
                meeting['round'] = i  # Add inferred round number
            logger.info(f"Successfully fetched {len(sorted_meetings)} rounds")
            return sorted_meetings
        else:
            logger.error(f"No meetings found for year {year}")
            return []
    except Exception as e:
        logger.error(f"Error fetching rounds: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_session_types(meeting_key: int) -> List[str]:
    """Fetch available session types for a specific meeting."""
    try:
        logger.info(f"Fetching session types for meeting_key {meeting_key}")
        response = requests.get(
            f"{BASE_URL}/sessions",
            params={
                "meeting_key": meeting_key  # Removed category parameter
            }
        )
        response.raise_for_status()
        
        if response.status_code == 200:
            sessions = response.json()
            session_types = sorted(list(set(session['session_name'] for session in sessions)))
            logger.info(f"Successfully fetched {len(session_types)} session types")
            return session_types
        else:
            logger.error(f"Failed to fetch session types. Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching session types: {str(e)}")
        return []

def main():
    st.title("Formula 1 Dashboard")
    
    # Create sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Add a test button to check API connectivity
        if st.button("Test API Connection"):
            try:
                response = requests.get(
                    f"{BASE_URL}/meetings",
                    params={
                        "year": CURRENT_YEAR,
                    }
                )
                if response.status_code == 200:
                    st.success("API connection successful!")
                    data = response.json()
                    st.write(f"Number of meetings found: {len(data)}")
                    if data:
                        st.write("Sample meeting data:")
                        st.json(data[0])  # Show first meeting data
                else:
                    st.error(f"API connection failed with status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"API connection error: {str(e)}")

        # Fetch and display seasons dropdown
        with st.spinner("Loading seasons..."):
            seasons = fetch_seasons()
        
        if not seasons:
            st.error("Failed to fetch seasons. Please try again later.")
            st.write("Debug information:")
            try:
                response = requests.get(
                    f"{BASE_URL}/meetings",
                    params={"year": CURRENT_YEAR}
                )
                st.write(f"Response status code: {response.status_code}")
                st.write(f"Response content: {response.text[:200]}...")  # Show first 200 chars
            except Exception as e:
                st.write(f"Error details: {str(e)}")
            return

        selected_season = st.selectbox("Select Season", seasons)

        # Fetch and display rounds dropdown
        with st.spinner("Loading rounds..."):
            rounds = fetch_rounds(selected_season)
        
        if not rounds:
            st.error("Failed to fetch rounds. Please try again later.")
            return

        round_options = {f"Round {r.get('round', 'N/A')} - {r.get('meeting_name', 'Unknown')}": r.get('meeting_key', 0) for r in rounds}
        selected_round = st.selectbox("Select Round", list(round_options.keys()))
        meeting_key = round_options[selected_round]

        # Fetch and display session types dropdown
        with st.spinner("Loading session types..."):
            session_types = fetch_session_types(meeting_key)
        
        if not session_types:
            st.error("Failed to fetch session types. Please try again later.")
            return

        selected_session = st.selectbox("Select Session Type", session_types)

    # Main content area
    st.write("### Selected Options:")
    st.write(f"- Season: {selected_season}")
    st.write(f"- Round: {selected_round}")
    st.write(f"- Session Type: {selected_session}")
    
    # Add placeholder for future content
    st.write("### Dashboard Content")
    st.write("Select options from the sidebar to view F1 data")

if __name__ == "__main__":
    main()