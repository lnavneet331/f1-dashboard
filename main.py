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
CURRENT_YEAR = datetime.now().year  # Use current year instead of hardcoded value
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# DRS status mapping
DRS_STATUS = {
    0: "DRS off",
    1: "DRS off",
    8: "DRS eligible",
    10: "DRS on",
    12: "DRS on",
    14: "DRS on"
}

# Session type priority for sorting
SESSION_PRIORITY = {
    'Race': 1,
    'Sprint': 2,
    'Qualifying': 3,
    'Practice': 4
}

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

@st.cache_resource
def fetch_weather(session_key: int) -> Dict:
    """Fetch weather data for a specific session."""
    try:
        cache_filename = f"weather_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading weather data from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching weather data for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/weather",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        weather_data = response.json()
        if weather_data:
            # Get the latest weather data
            latest_weather = weather_data[-1]
            save_to_cache(latest_weather, cache_filename)
            return latest_weather
        return None
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return None

@st.cache_resource
def fetch_car_data(session_key: int, driver_number: int) -> List[Dict]:
    """Fetch car telemetry data for a specific driver and session."""
    try:
        cache_filename = f"car_data_{session_key}_{driver_number}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading car data from cache for session {session_key} and driver {driver_number}")
            return cached_data

        logger.info(f"Fetching car data for session {session_key} and driver {driver_number}")
        response = requests.get(
            f"{BASE_URL}/car_data",
            params={
                "session_key": session_key,
                "driver_number": driver_number
            }
        )
        response.raise_for_status()
        car_data = response.json()
        if car_data:
            save_to_cache(car_data, cache_filename)
        return car_data
    except Exception as e:
        logger.error(f"Error fetching car data: {str(e)}")
        return []

@st.cache_resource
def fetch_race_control(session_key: int) -> List[Dict]:
    """Fetch race control messages for a specific session."""
    try:
        cache_filename = f"race_control_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading race control data from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching race control data for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/race_control",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        race_control_data = response.json()
        if race_control_data:
            save_to_cache(race_control_data, cache_filename)
        return race_control_data
    except Exception as e:
        logger.error(f"Error fetching race control data: {str(e)}")
        return []

@st.cache_resource
def fetch_team_radio(session_key: int, driver_number: int) -> List[Dict]:
    """Fetch team radio messages for a specific driver and session."""
    try:
        cache_filename = f"team_radio_{session_key}_{driver_number}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading team radio data from cache for session {session_key} and driver {driver_number}")
            return cached_data

        logger.info(f"Fetching team radio data for session {session_key} and driver {driver_number}")
        response = requests.get(
            f"{BASE_URL}/team_radio",
            params={
                "session_key": session_key,
                "driver_number": driver_number
            }
        )
        response.raise_for_status()
        radio_data = response.json()
        if radio_data:
            save_to_cache(radio_data, cache_filename)
        return radio_data
    except Exception as e:
        logger.error(f"Error fetching team radio data: {str(e)}")
        return []

@st.cache_resource
def fetch_car_telemetry(session_key: int, driver_number: int) -> List[Dict]:
    """Fetch car telemetry data for a specific driver and session."""
    try:
        cache_filename = f"car_telemetry_{session_key}_{driver_number}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading car telemetry from cache for session {session_key} and driver {driver_number}")
            return cached_data

        logger.info(f"Fetching car telemetry for session {session_key} and driver {driver_number}")
        response = requests.get(
            f"{BASE_URL}/car_data",
            params={
                "session_key": session_key,
                "driver_number": driver_number
            }
        )
        response.raise_for_status()
        telemetry_data = response.json()
        if telemetry_data:
            save_to_cache(telemetry_data, cache_filename)
        return telemetry_data
    except Exception as e:
        logger.error(f"Error fetching car telemetry: {str(e)}")
        return []

@st.cache_resource
def fetch_pit_stops(session_key: int) -> List[Dict]:
    """Fetch pit stop data for a specific session."""
    try:
        cache_filename = f"pit_stops_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading pit stops from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching pit stops for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/pit",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        pit_stops = response.json()
        if pit_stops:
            save_to_cache(pit_stops, cache_filename)
        return pit_stops
    except Exception as e:
        logger.error(f"Error fetching pit stops: {str(e)}")
        return []

@st.cache_resource
def fetch_stints(session_key: int) -> List[Dict]:
    """Fetch stint data for a specific session."""
    try:
        cache_filename = f"stints_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading stints from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching stints for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/stints",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        stints = response.json()
        if stints:
            save_to_cache(stints, cache_filename)
        return stints
    except Exception as e:
        logger.error(f"Error fetching stints: {str(e)}")
        return []

@st.cache_resource
def fetch_location_data(session_key: int, driver_number: int) -> List[Dict]:
    """Fetch location data for a specific driver and session."""
    try:
        cache_filename = f"location_{session_key}_{driver_number}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading location data from cache for session {session_key} and driver {driver_number}")
            return cached_data

        logger.info(f"Fetching location data for session {session_key} and driver {driver_number}")
        response = requests.get(
            f"{BASE_URL}/location",
            params={
                "session_key": session_key,
                "driver_number": driver_number
            }
        )
        response.raise_for_status()
        location_data = response.json()
        if location_data:
            save_to_cache(location_data, cache_filename)
        return location_data
    except Exception as e:
        logger.error(f"Error fetching location data: {str(e)}")
        return []

def get_fastest_lap_data(session_key: int, lap_times: List[Dict], drivers: List[Dict]) -> Dict:
    """Get the fastest lap data for each driver and overall."""
    if not lap_times:
        return {}
    
    # Convert lap times to DataFrame
    df = pd.DataFrame(lap_times)
    
    # Get fastest lap for each driver
    fastest_laps = {}
    for driver in drivers:
        driver_laps = df[df['driver_number'] == driver['driver_number']]
        if not driver_laps.empty:
            fastest_lap = driver_laps.loc[driver_laps['lap_duration'].idxmin()]
            fastest_laps[driver['driver_number']] = {
                'lap_number': fastest_lap['lap_number'],
                'lap_duration': fastest_lap['lap_duration'],
                'driver_name': f"{driver['name_acronym']} ({driver['driver_number']})",
                'team_name': driver['team_name']
            }
    
    # Get overall fastest lap
    overall_fastest = df.loc[df['lap_duration'].idxmin()]
    fastest_laps['overall'] = {
        'lap_number': overall_fastest['lap_number'],
        'lap_duration': overall_fastest['lap_duration'],
        'driver_name': next((f"{d['name_acronym']} ({d['driver_number']})" 
                           for d in drivers if d['driver_number'] == overall_fastest['driver_number']), 'Unknown'),
        'team_name': next((d['team_name'] 
                          for d in drivers if d['driver_number'] == overall_fastest['driver_number']), 'Unknown')
    }
    
    return fastest_laps

def create_track_map(location_data: List[Dict], driver_name: str, team_color: str) -> str:
    """Create an HTML visualization of the track map with car position."""
    if not location_data:
        return ""
    
    # Extract x and y coordinates
    x_coords = [point['x'] for point in location_data]
    y_coords = [point['y'] for point in location_data]
    
    # Create the track map using plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create figure
    fig = go.Figure()
    
    # Add track line
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color='#666666', width=2),
        name='Track'
    ))
    
    # Add car position
    fig.add_trace(go.Scatter(
        x=[x_coords[-1]],
        y=[y_coords[-1]],
        mode='markers',
        marker=dict(
            size=15,
            color=team_color,
            symbol='circle'
        ),
        name=driver_name
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Track Map - {driver_name}",
        showlegend=True,
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Convert to HTML
    return fig.to_html(full_html=False)

def process_telemetry_data(telemetry_data: List[Dict]) -> pd.DataFrame:
    """Process raw telemetry data into a pandas DataFrame."""
    if not telemetry_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(telemetry_data)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add DRS status text
    df['drs_status'] = df['drs'].map(DRS_STATUS)
    
    return df

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

def get_latest_session_data() -> tuple:
    """Get the latest available session data."""
    try:
        # Get the most recent season
        seasons = fetch_seasons()
        if not seasons:
            return None, None, None
        
        latest_season = seasons[0]  # Seasons are already sorted in descending order
        
        # Get rounds for the latest season
        rounds = fetch_rounds(latest_season)
        if not rounds:
            return None, None, None
        
        # Get the latest round
        latest_round = rounds[-1]  # Rounds are sorted by date_start
        meeting_key = latest_round.get('meeting_key')
        
        # Get session types for the latest round
        session_types = fetch_session_types(meeting_key)
        if not session_types:
            return None, None, None
        
        # Prioritize session types (Race > Qualifying > Practice)
        session_priority = {'Race': 1, 'Sprint': 2, 'Qualifying': 3, 'Practice': 4}
        latest_session = min(session_types, key=lambda x: session_priority.get(x, 999))
        
        return latest_season, latest_round, latest_session
    except Exception as e:
        logger.error(f"Error getting latest session data: {str(e)}")
        return None, None, None

def process_pit_stops(pit_stops: List[Dict], drivers: List[Dict]) -> pd.DataFrame:
    """Process pit stop data into a pandas DataFrame with driver information."""
    if not pit_stops:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(pit_stops)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add driver information
    driver_info = {d['driver_number']: d for d in drivers}
    df['driver_name'] = df['driver_number'].map(lambda x: f"{driver_info[x]['name_acronym']} ({x})")
    df['team_name'] = df['driver_number'].map(lambda x: driver_info[x]['team_name'])
    
    return df

def process_stints(stints: List[Dict], drivers: List[Dict]) -> pd.DataFrame:
    """Process stint data into a pandas DataFrame with driver information."""
    if not stints:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(stints)
    
    # Add driver information
    driver_info = {d['driver_number']: d for d in drivers}
    df['driver_name'] = df['driver_number'].map(lambda x: f"{driver_info[x]['name_acronym']} ({x})")
    df['team_name'] = df['driver_number'].map(lambda x: driver_info[x]['team_name'])
    
    # Calculate stint length
    df['stint_length'] = df['lap_end'] - df['lap_start'] + 1
    
    return df

def main():
    st.set_page_config(layout="wide", page_title="F1 Dashboard")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-nav-button {
            background-color: #1E1E1E;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .main-nav-button:hover {
            background-color: #2E2E2E;
            transform: translateY(-2px);
        }
        .sub-nav-button {
            background-color: #2E2E2E;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .sub-nav-button:hover {
            background-color: #3E3E3E;
            transform: translateY(-2px);
        }
        .active-button {
            background-color: #FF1801 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main Navigation
    st.title("Formula 1 Dashboard")
    
    # Create three columns for main navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🏁 Sessions", key="sessions_nav", use_container_width=True):
            st.session_state.current_view = "sessions"
            st.session_state.current_subview = "info"
    
    with nav_col2:
        if st.button("👤 Drivers", key="drivers_nav", use_container_width=True):
            st.session_state.current_view = "drivers"
    
    with nav_col3:
        if st.button("🏎️ Teams", key="teams_nav", use_container_width=True):
            st.session_state.current_view = "teams"
    
    # Initialize session state if not exists
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "sessions"
        st.session_state.current_subview = "info"
    
    # Fetch and display seasons dropdown
    with st.spinner("Loading seasons..."):
        seasons = fetch_seasons()
    
    if not seasons:
        st.error("Failed to fetch seasons. Please try again later.")
        return

    # Get latest session data for default values
    latest_season, latest_round, latest_session = get_latest_session_data()
    
    # Create default options
    default_season = latest_season if latest_season else seasons[0]
    default_round = f"Round {latest_round.get('round', 'N/A')} - {latest_round.get('meeting_name', 'Unknown')}" if latest_round else None
    default_session = latest_session if latest_session else None

    # Main content area
    if st.session_state.current_view == "sessions":
        # Sessions View
        st.write("### Session Selection")
        
        # Create seven columns for sub-navigation
        sub_nav_col1, sub_nav_col2, sub_nav_col3, sub_nav_col4, sub_nav_col5, sub_nav_col6, sub_nav_col7 = st.columns(7)
        
        with sub_nav_col1:
            if st.button("📊 Session Info", key="session_info_nav", use_container_width=True):
                st.session_state.current_subview = "info"
        
        with sub_nav_col2:
            if st.button("🚨 Race Control", key="race_control_nav", use_container_width=True):
                st.session_state.current_subview = "race_control"
        
        with sub_nav_col3:
            if st.button("📻 Team Radio", key="team_radio_nav", use_container_width=True):
                st.session_state.current_subview = "team_radio"
            
        with sub_nav_col4:
            if st.button("📈 Telemetry", key="telemetry_nav", use_container_width=True):
                st.session_state.current_subview = "telemetry"
            
        with sub_nav_col5:
            if st.button("🛑 Pit Stops", key="pit_stops_nav", use_container_width=True):
                st.session_state.current_subview = "pit_stops"
            
        with sub_nav_col6:
            if st.button("🔄 Tyre Stints", key="stints_nav", use_container_width=True):
                st.session_state.current_subview = "stints"
            
        with sub_nav_col7:
            if st.button("🗺️ Track Map", key="track_map_nav", use_container_width=True):
                st.session_state.current_subview = "track_map"
        
        # Session selection controls
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_season = st.selectbox("Select Season", seasons, 
                                         index=seasons.index(default_season) if default_season in seasons else 0)
        
        # Fetch rounds for the selected season
        with st.spinner("Loading rounds..."):
            rounds = fetch_rounds(selected_season)
        
        if not rounds:
            st.error("Failed to fetch rounds. Please try again later.")
            return

        # Create round options dictionary
        round_options = {f"Round {r.get('round', 'N/A')} - {r.get('meeting_name', 'Unknown')}": r.get('meeting_key', 0) for r in rounds}
        
        with col2:
            selected_round = st.selectbox("Select Round", list(round_options.keys()), 
                                        index=list(round_options.keys()).index(default_round) if default_round in round_options else 0)
            meeting_key = round_options[selected_round]

        # Fetch session types for the selected round
        with st.spinner("Loading session types..."):
            session_types = fetch_session_types(meeting_key)
        
        if not session_types:
            st.error("Failed to fetch session types. Please try again later.")
            return

        with col3:
            selected_session = st.selectbox("Select Session Type", session_types,
                                          index=session_types.index(default_session) if default_session in session_types else 0)

        # Check if we have valid session data
        session_key = fetch_session_key(meeting_key, selected_session)
        if not session_key:
            st.warning("No session data available for the selected options.")
            return

        # Display sub-view content
        if st.session_state.current_subview == "info":
            st.write(f"### {selected_season} {selected_round.split(' - ')[1]} · {selected_session}")
            
            # Display weather information
            weather_data = fetch_weather(session_key)
            if weather_data:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Air Temperature", f"{weather_data['air_temperature']}°C")
                with col2:
                    st.metric("Track Temperature", f"{weather_data['track_temperature']}°C")
                with col3:
                    st.metric("Humidity", f"{weather_data['humidity']}%")
                with col4:
                    st.metric("Wind Speed", f"{weather_data['wind_speed']} m/s")
            
            # Display leaderboard
            with st.spinner("Loading leaderboard..."):
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
        
        elif st.session_state.current_subview == "race_control":
            st.write("### Race Control Messages")
            if session_key:
                race_control_data = fetch_race_control(session_key)
                if race_control_data:
                    # Group messages by category
                    messages_by_category = {}
                    for msg in race_control_data:
                        category = msg.get('category', 'Other')
                        if category not in messages_by_category:
                            messages_by_category[category] = []
                        messages_by_category[category].append(msg)
                    
                    # Display messages by category
                    for category, messages in messages_by_category.items():
                        st.write(f"#### {category}")
                        for msg in messages:
                            # Create a colored box based on flag type
                            flag_color = {
                                'RED': '#FF0000',
                                'YELLOW': '#FFFF00',
                                'GREEN': '#00FF00',
                                'BLACK': '#000000',
                                'BLUE': '#0000FF',
                                'WHITE': '#FFFFFF',
                                'CHEQUERED': '#000000'
                            }.get(msg.get('flag', ''), '#808080')
                            
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: {flag_color};
                                    color: {'white' if flag_color in ['#000000', '#0000FF'] else 'black'};
                                    padding: 10px;
                                    border-radius: 5px;
                                    margin: 5px 0;
                                ">
                                    <strong>{msg.get('flag', 'Message')}</strong><br>
                                    {msg.get('message', 'No message')}<br>
                                    <small>Lap {msg.get('lap_number', 'N/A')}</small>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No race control messages available for this session.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "team_radio":
            st.write("### Team Radio Messages")
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    # Create a grid of driver cards with radio messages
                    cols = st.columns(3)
                    for idx, driver in enumerate(drivers):
                        with cols[idx % 3]:
                            # Create a card with team color background
                            team_color = f"#{driver['team_colour']}"
                            hex_color = team_color.lstrip('#')
                            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                            text_color = 'white' if luminance < 0.5 else 'black'
                            
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
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Add team radio messages
                            radio_messages = fetch_team_radio(session_key, driver['driver_number'])
                            if radio_messages:
                                for msg in radio_messages:
                                    st.audio(msg['recording_url'])
                            else:
                                st.info("No radio messages available")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "telemetry":
            st.write("### Car Telemetry")
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    # Create a dropdown for driver selection
                    driver_options = {f"{driver['driver_number']} - {driver['name_acronym']}": driver['driver_number'] 
                                    for driver in drivers}
                    selected_driver = st.selectbox("Select Driver", list(driver_options.keys()))
                    driver_number = driver_options[selected_driver]
                    
                    # Fetch and process telemetry data
                    telemetry_data = fetch_car_telemetry(session_key, driver_number)
                    if telemetry_data:
                        df = process_telemetry_data(telemetry_data)
                        
                        # Create columns for different metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("#### Speed")
                            st.line_chart(df.set_index('date')['speed'])
                            
                            st.write("#### RPM")
                            st.line_chart(df.set_index('date')['rpm'])
                            
                        with col2:
                            st.write("#### Throttle & Brake")
                            throttle_brake = df[['date', 'throttle', 'brake']].set_index('date')
                            st.line_chart(throttle_brake)
                            
                            st.write("#### DRS Status")
                            drs_data = df[['date', 'drs_status']].set_index('date')
                            st.line_chart(drs_data)
                        
                        # Display gear selection
                        st.write("#### Gear Selection")
                        gear_data = df[['date', 'n_gear']].set_index('date')
                        st.line_chart(gear_data)
                    else:
                        st.info("No telemetry data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "pit_stops":
            st.write("### Pit Stop Analysis")
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    pit_stops = fetch_pit_stops(session_key)
                    if pit_stops:
                        df = process_pit_stops(pit_stops, drivers)
                        
                        # Display overall statistics
                        st.write("#### Overall Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Pit Stops", len(df))
                        with col2:
                            st.metric("Average Stop Time", f"{df['pit_duration'].mean():.1f}s")
                        with col3:
                            st.metric("Fastest Stop", f"{df['pit_duration'].min():.1f}s")
                        
                        # Display pit stop timeline
                        st.write("#### Pit Stop Timeline")
                        timeline_data = df[['date', 'driver_name', 'pit_duration', 'lap_number']].copy()
                        timeline_data['date'] = timeline_data['date'].dt.strftime('%H:%M:%S')
                        st.dataframe(timeline_data, hide_index=True)
                        
                        # Display team-wise analysis
                        st.write("#### Team Analysis")
                        team_stats = df.groupby('team_name').agg({
                            'pit_duration': ['count', 'mean', 'min']
                        }).round(2)
                        team_stats.columns = ['Total Stops', 'Average Duration', 'Fastest Stop']
                        st.dataframe(team_stats)
                        
                        # Display driver-wise analysis
                        st.write("#### Driver Analysis")
                        driver_stats = df.groupby('driver_name').agg({
                            'pit_duration': ['count', 'mean', 'min']
                        }).round(2)
                        driver_stats.columns = ['Total Stops', 'Average Duration', 'Fastest Stop']
                        st.dataframe(driver_stats)
                    else:
                        st.info("No pit stop data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "stints":
            st.write("### Tyre Stint Analysis")
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    stints = fetch_stints(session_key)
                    if stints:
                        df = process_stints(stints, drivers)
                        
                        # Display overall statistics
                        st.write("#### Overall Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Stints", len(df))
                        with col2:
                            st.metric("Average Stint Length", f"{df['stint_length'].mean():.1f} laps")
                        with col3:
                            st.metric("Longest Stint", f"{df['stint_length'].max()} laps")
                        
                        # Display stint timeline
                        st.write("#### Stint Timeline")
                        timeline_data = df[['driver_name', 'compound', 'lap_start', 'lap_end', 'stint_length']].copy()
                        timeline_data = timeline_data.sort_values('lap_start')
                        st.dataframe(timeline_data, hide_index=True)
                        
                        # Display compound analysis
                        st.write("#### Compound Analysis")
                        compound_stats = df.groupby('compound').agg({
                            'stint_length': ['count', 'mean', 'max']
                        }).round(2)
                        compound_stats.columns = ['Total Stints', 'Average Length', 'Longest Stint']
                        st.dataframe(compound_stats)
                        
                        # Display team-wise analysis
                        st.write("#### Team Analysis")
                        team_stats = df.groupby('team_name').agg({
                            'stint_length': ['count', 'mean', 'max']
                        }).round(2)
                        team_stats.columns = ['Total Stints', 'Average Length', 'Longest Stint']
                        st.dataframe(team_stats)
                        
                        # Display driver-wise analysis
                        st.write("#### Driver Analysis")
                        driver_stats = df.groupby('driver_name').agg({
                            'stint_length': ['count', 'mean', 'max']
                        }).round(2)
                        driver_stats.columns = ['Total Stints', 'Average Length', 'Longest Stint']
                        st.dataframe(driver_stats)
                    else:
                        st.info("No stint data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "track_map":
            st.write("### Track Map - Fastest Laps")
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    # Get lap times
                    lap_times = fetch_lap_times(session_key)
                    if lap_times:
                        # Get fastest lap data
                        fastest_laps = get_fastest_lap_data(session_key, lap_times, drivers)
                        
                        # Display overall fastest lap
                        st.write("#### Overall Fastest Lap")
                        overall = fastest_laps['overall']
                        st.write(f"**Driver:** {overall['driver_name']}")
                        st.write(f"**Team:** {overall['team_name']}")
                        st.write(f"**Lap:** {overall['lap_number']}")
                        st.write(f"**Time:** {format_time(overall['lap_duration'])}")
                        
                        # Get location data for the fastest lap
                        driver_number = next(d['driver_number'] for d in drivers 
                                          if f"{d['name_acronym']} ({d['driver_number']})" == overall['driver_name'])
                        location_data = fetch_location_data(session_key, driver_number)
                        
                        # Create and display track map
                        if location_data:
                            team_color = next(d['team_colour'] for d in drivers 
                                            if d['driver_number'] == driver_number)
                            track_map_html = create_track_map(location_data, overall['driver_name'], f"#{team_color}")
                            st.components.v1.html(track_map_html, height=600)
                        else:
                            st.info("No location data available for this session.")
                        
                        # Display all drivers' fastest laps
                        st.write("#### All Drivers' Fastest Laps")
                        fastest_laps_df = pd.DataFrame([
                            {k: v for k, v in data.items() if k != 'driver_name' and k != 'team_name'}
                            for data in fastest_laps.values()
                            if data['driver_name'] != overall['driver_name']
                        ])
                        fastest_laps_df = fastest_laps_df.sort_values('lap_duration')
                        st.dataframe(fastest_laps_df, hide_index=True)
                    else:
                        st.info("No lap time data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
    
    elif st.session_state.current_view == "drivers":
        st.write("### Driver Standings")
        # TODO: Implement driver standings view
        st.info("Driver standings view coming soon!")
    
    else:  # Teams view
        st.write("### Constructor Standings")
        # TODO: Implement constructor standings view
        st.info("Constructor standings view coming soon!")

if __name__ == "__main__":
    main()