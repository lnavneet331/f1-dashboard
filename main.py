import streamlit as st
import requests
import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime
import time
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://api.openf1.org/v1"
CURRENT_YEAR = 2000  # Focus on 2024 season
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(filename: str) -> str:
    """Get the full path for a cache file."""
    return os.path.join(CACHE_DIR, filename)

def save_to_cache(data: any, filename: str):
    """Save data to cache file."""
    cache_path = get_cache_path(filename)
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def load_from_cache(filename: str) -> any:
    """Load data from cache file if it exists."""
    cache_path = get_cache_path(filename)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def fetch_meetings(year: int) -> List[Dict]:
    """Fetch meetings (races) for a specific year with persistent caching."""
    try:
        cache_filename = f"meetings_{year}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading meetings for year {year} from cache")
            return cached_data

        logger.info(f"Fetching meetings for year {year}")
        response = requests.get(
            f"{BASE_URL}/meetings",
            params={"year": year}
        )
        response.raise_for_status()
        meetings = response.json()
        logger.info(f"Found {len(meetings)} meetings in schedule")
        if meetings:
            logger.info(f"First meeting data: {meetings[0]}")
            save_to_cache(meetings, cache_filename)
        return meetings
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching meetings for year {year}: {str(e)}")
        return []

@st.cache_resource
def fetch_seasons() -> List[int]:
    """Fetch available F1 seasons with persistent caching."""
    try:
        cache_filename = "seasons.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info("Loading seasons from cache")
            return cached_data

        logger.info("Fetching available seasons")
        # Try fetching meetings for a range of years to find available seasons
        current_year = datetime.now().year
        available_seasons = []
        
        # Check last 10 years and next 2 years
        for year in range(current_year - 10, current_year + 3):
            try:
                response = requests.get(
                    f"{BASE_URL}/meetings",
                    params={"year": year}
                )
                if response.status_code == 200 and response.json():
                    available_seasons.append(year)
                    logger.info(f"Found data for season {year}")
            except Exception as e:
                logger.warning(f"Could not fetch data for year {year}: {str(e)}")
                continue
        
        if not available_seasons:
            logger.warning("No seasons found, falling back to current year")
            available_seasons = [current_year]
        
        # Sort seasons in descending order (newest first)
        available_seasons.sort(reverse=True)
        save_to_cache(available_seasons, cache_filename)
        return available_seasons
    except Exception as e:
        logger.error(f"Error fetching seasons: {str(e)}")
        return [CURRENT_YEAR]  # Fallback to current year if there's an error

@st.cache_resource
def fetch_rounds(year: int) -> List[Dict]:
    """Fetch all rounds (meetings) for a specific season with persistent caching."""
    try:
        cache_filename = f"rounds_{year}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading rounds for year {year} from cache")
            return cached_data

        logger.info(f"Fetching rounds for year {year}")
        meetings = fetch_meetings(year)
        if meetings:
            # Sort meetings by date_start and assign round numbers
            sorted_meetings = sorted(meetings, key=lambda x: x.get('date_start', ''))
            for i, meeting in enumerate(sorted_meetings, 1):
                meeting['round'] = i  # Add inferred round number
            logger.info(f"Successfully fetched {len(sorted_meetings)} rounds")
            save_to_cache(sorted_meetings, cache_filename)
            return sorted_meetings
        else:
            logger.error(f"No meetings found for year {year}")
            return []
    except Exception as e:
        logger.error(f"Error fetching rounds: {str(e)}")
        return []

@st.cache_resource
def fetch_session_types(meeting_key: int) -> List[str]:
    """Fetch available session types for a specific meeting with persistent caching."""
    try:
        cache_filename = f"session_types_{meeting_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading session types for meeting_key {meeting_key} from cache")
            return cached_data

        logger.info(f"Fetching session types for meeting_key {meeting_key}")
        response = requests.get(
            f"{BASE_URL}/sessions",
            params={"meeting_key": meeting_key}
        )
        response.raise_for_status()
        
        if response.status_code == 200:
            sessions = response.json()
            session_types = sorted(list(set(session['session_name'] for session in sessions)))
            logger.info(f"Successfully fetched {len(session_types)} session types")
            save_to_cache(session_types, cache_filename)
            return session_types
        else:
            logger.error(f"Failed to fetch session types. Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching session types: {str(e)}")
        return []

@st.cache_resource
def fetch_session_key(meeting_key: int, session_name: str) -> int:
    """Fetch the session key for a specific meeting and session name."""
    try:
        cache_filename = f"session_key_{meeting_key}_{session_name}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading session key from cache for meeting {meeting_key} and session {session_name}")
            return cached_data

        logger.info(f"Fetching session key for meeting {meeting_key} and session {session_name}")
        response = requests.get(
            f"{BASE_URL}/sessions",
            params={
                "meeting_key": meeting_key,
                "session_name": session_name
            }
        )
        response.raise_for_status()
        sessions = response.json()
        if sessions:
            session_key = sessions[0]['session_key']
            save_to_cache(session_key, cache_filename)
            return session_key
        return None
    except Exception as e:
        logger.error(f"Error fetching session key: {str(e)}")
        return None

@st.cache_resource
def fetch_drivers(session_key: int) -> List[Dict]:
    """Fetch driver information for a specific session."""
    try:
        cache_filename = f"drivers_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading drivers from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching drivers for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/drivers",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        drivers = response.json()
        save_to_cache(drivers, cache_filename)
        return drivers
    except Exception as e:
        logger.error(f"Error fetching drivers: {str(e)}")
        return []

@st.cache_resource
def fetch_positions(session_key: int) -> List[Dict]:
    """Fetch position data for a specific session."""
    try:
        cache_filename = f"positions_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading positions from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching positions for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/position",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        positions = response.json()
        save_to_cache(positions, cache_filename)
        return positions
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        return []

@st.cache_resource
def fetch_lap_times(session_key: int) -> List[Dict]:
    """Fetch lap times for a specific session."""
    try:
        cache_filename = f"lap_times_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading lap times from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching lap times for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/laps",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        lap_times = response.json()
        save_to_cache(lap_times, cache_filename)
        return lap_times
    except Exception as e:
        logger.error(f"Error fetching lap times: {str(e)}")
        return []

def get_latest_positions(positions: List[Dict]) -> Dict[int, int]:
    """Get the latest position for each driver."""
    latest_positions = {}
    for pos in positions:
        latest_positions[pos['driver_number']] = pos['position']
    return latest_positions

def get_finishing_times(lap_times: List[Dict]) -> Dict[int, float]:
    """Get the finishing time for each driver."""
    finishing_times = {}
    for lap in lap_times:
        driver_number = lap['driver_number']
        # Only consider valid lap times
        if 'lap_duration' in lap and lap['lap_duration'] is not None and lap['lap_duration'] > 0:
            if driver_number not in finishing_times or lap['lap_number'] > finishing_times[driver_number]['lap_number']:
                finishing_times[driver_number] = {
                    'time': lap['lap_duration'],
                    'lap_number': lap['lap_number']
                }
    return {k: v['time'] for k, v in finishing_times.items()}

def calculate_gaps(finishing_times: Dict[int, float]) -> Dict[int, str]:
    """Calculate time gaps to the leader for each driver."""
    if not finishing_times:
        return {}
    
    # Find the fastest time (leader)
    leader_time = min(finishing_times.values())
    gaps = {}
    
    for driver_number, time in finishing_times.items():
        if time == leader_time:
            gaps[driver_number] = "Leader"
        else:
            gap = time - leader_time
            gaps[driver_number] = f"+{format_time(gap)}"
    
    return gaps

def get_position_changes(positions: List[Dict]) -> Dict[int, int]:
    """Calculate position changes for each driver."""
    position_changes = {}
    driver_positions = {}
    
    # Sort positions by date to track changes
    sorted_positions = sorted(positions, key=lambda x: x['date'])
    
    for pos in sorted_positions:
        driver_number = pos['driver_number']
        if driver_number not in driver_positions:
            driver_positions[driver_number] = pos['position']
        else:
            # Calculate change from starting position
            position_changes[driver_number] = driver_positions[driver_number] - pos['position']
    
    return position_changes

def format_position_change(change: int) -> str:
    """Format position change with + or - sign."""
    if change > 0:
        return f"+{change}"
    elif change < 0:
        return str(change)
    return "0"

def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS.mmm format."""
    if seconds is None or seconds <= 0:
        return 'DNF'
    try:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    except (TypeError, ValueError):
        return 'DNF'

def main():
    st.title("Formula 1 Dashboard")
    
    # Create sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Add navigation buttons
        st.write("### Navigation")
        view = st.radio(
            "Select View",
            ["🏁 Sessions", "👤 Driver Details"],
            index=0,
            horizontal=True
        )

        # Fetch and display seasons dropdown
        with st.spinner("Loading seasons..."):
            seasons = fetch_seasons()
        
        if not seasons:
            st.error("Failed to fetch seasons. Please try again later.")
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
    if view == "🏁 Sessions":
        # Session View
        st.write(f"### {selected_season} {selected_round.split(' - ')[1]} · {selected_session}")
        
        # Fetch and display leaderboard
        with st.spinner("Loading leaderboard..."):
            session_key = fetch_session_key(meeting_key, selected_session)
            if session_key:
                drivers = fetch_drivers(session_key)
                positions = fetch_positions(session_key)
                
                if drivers and positions:
                    latest_positions = get_latest_positions(positions)
                    
                    # Create a DataFrame for the leaderboard
                    leaderboard_data = []
                    lap_times = fetch_lap_times(session_key)
                    finishing_times = get_finishing_times(lap_times)
                    gaps = calculate_gaps(finishing_times)
                    position_changes = get_position_changes(positions)
                    
                    # Check if this is a race or sprint session
                    is_race_session = selected_session.lower() in ['race', 'sprint']
                    
                    for driver in drivers:
                        driver_number = driver['driver_number']
                        entry = {
                            'Position': latest_positions.get(driver_number, 'DNF'),
                            'Driver': f"{driver['driver_number']} - {driver['name_acronym']}",
                            'Time': format_time(finishing_times.get(driver_number)),
                            'Gap': gaps.get(driver_number, 'DNF')
                        }
                        if is_race_session:
                            entry['Pos Change'] = format_position_change(position_changes.get(driver_number, 0))
                        leaderboard_data.append(entry)
                    
                    # Sort by position and display
                    df = pd.DataFrame(leaderboard_data)
                    df = df.sort_values('Position')
                    
                    st.write("### Leaderboard")
                    
                    # Create a dictionary of driver numbers to team colors
                    driver_colors = {driver['driver_number']: f"#{driver['team_colour']}" 
                                   for driver in drivers}
                    
                    # Apply styling to the DataFrame
                    styled_df = df.style
                    
                    # Apply team colors to driver cells
                    def color_team(val):
                        """Apply team color to driver cell with appropriate text color."""
                        for driver_number, team_color in driver_colors.items():
                            if val.startswith(f"{driver_number} -"):
                                # Convert hex to RGB to determine if background is light or dark
                                hex_color = team_color.lstrip('#')
                                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                # Calculate relative luminance
                                luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                                # Use white text for dark backgrounds, black for light backgrounds
                                text_color = 'white' if luminance < 0.5 else 'black'
                                return f'background-color: {team_color}; color: {text_color};'
                        return ''
                    
                    # Apply team colors to all driver cells
                    styled_df = styled_df.applymap(color_team, subset=['Driver'])
                    
                    # Apply position change colors for race/sprint sessions
                    def color_position_change(val):
                        if val.startswith('+'):
                            return 'color: green'
                        elif val.startswith('-'):
                            return 'color: red'
                        return ''
                    
                    if is_race_session:
                        styled_df = styled_df.applymap(color_position_change, subset=['Pos Change'])
                    
                    # Display the styled DataFrame
                    st.dataframe(styled_df, hide_index=True)
                else:
                    st.error("Failed to load leaderboard data. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
    
    else:  # Driver Details View
        st.write("### Driver Details")
        session_key = fetch_session_key(meeting_key, selected_session)
        if session_key:
            drivers = fetch_drivers(session_key)
            if drivers:
                # Create a grid of driver cards
                cols = st.columns(3)  # 3 columns for the grid
                for idx, driver in enumerate(drivers):
                    with cols[idx % 3]:  # Cycle through columns
                        # Create a card with team color background
                        team_color = f"#{driver['team_colour']}"
                        hex_color = team_color.lstrip('#')
                        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                        text_color = 'white' if luminance < 0.5 else 'black'
                        
                        # Create a custom HTML card with team color
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {team_color};
                                color: {text_color};
                                padding: 20px;
                                border-radius: 10px;
                                margin-bottom: 20px;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <h3 style="margin: 0;">{driver['driver_number']} - {driver['name_acronym']}</h3>
                                <p style="margin: 10px 0;">{driver['full_name']}</p>
                                <p style="margin: 5px 0;">{driver['team_name']}</p>
                                <p style="margin: 5px 0;">{driver['country_code']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Add driver image if available
                        if 'headshot_url' in driver:
                            st.image(driver['headshot_url'], width=150)
            else:
                st.error("Failed to load driver details. Please try again later.")
        else:
            st.error("Failed to fetch session key. Please try again later.")

if __name__ == "__main__":
    main()