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

@st.cache_resource
def fetch_session_details(session_key: int) -> Dict:
    """Fetch session details including start and end times."""
    try:
        cache_filename = f"session_details_{session_key}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading session details from cache for session {session_key}")
            return cached_data

        logger.info(f"Fetching session details for session {session_key}")
        response = requests.get(
            f"{BASE_URL}/sessions",
            params={"session_key": session_key}
        )
        response.raise_for_status()
        sessions = response.json()
        if sessions:
            session_details = sessions[0]
            save_to_cache(session_details, cache_filename)
            return session_details
        return None
    except Exception as e:
        logger.error(f"Error fetching session details: {str(e)}")
        return None

@st.cache_resource
def fetch_driver_standings(season: int = None) -> List[Dict]:
    """Fetch driver championship standings for a specific season."""
    try:
        if not season:
            season = datetime.now().year  # Default to current year
            
        cache_filename = f"driver_standings_{season}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading driver standings from cache for season {season}")
            return cached_data

        logger.info(f"Fetching driver standings for season {season}")
        
        # We'll need to calculate this from race results since OpenF1 doesn't provide standings directly
        # Get all races for the season
        rounds = fetch_rounds(season)
        if not rounds:
            return []
        
        # Initialize driver points dictionary
        driver_points = {}
        driver_info = {}
        
        # Process each race to accumulate points
        for round_info in rounds:
            meeting_key = round_info.get('meeting_key')
            if not meeting_key:
                continue
                
            # Get race session key (if exists)
            session_types = fetch_session_types(meeting_key)
            race_session = None
            
            # Find race or sprint session for points
            for session_priority in ['Race', 'Sprint']:
                if session_priority in session_types:
                    race_session = session_priority
                    session_key = fetch_session_key(meeting_key, race_session)
                    
                    # Get drivers and results
                    drivers = fetch_drivers(session_key)
                    positions = fetch_positions(session_key)
                    
                    if not drivers or not positions:
                        continue
                    
                    # Get final positions
                    latest_positions = get_latest_positions(positions)
                    
                    # Points for regular race positions (1st to 10th)
                    race_points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] if race_session == 'Race' else [8, 7, 6, 5, 4, 3, 2, 1]
                    
                    # Award points based on positions
                    for driver in drivers:
                        driver_number = driver['driver_number']
                        position = latest_positions.get(driver_number)
                        
                        # Store driver info
                        if driver_number not in driver_info:
                            driver_info[driver_number] = {
                                'name': driver['full_name'],
                                'name_acronym': driver['name_acronym'],
                                'team': driver['team_name'],
                                'team_color': f"#{driver['team_colour']}"
                            }
                        
                        # Calculate points
                        points = 0
                        if position and 1 <= position <= len(race_points):
                            points = race_points[position-1]
                            
                        # Add to driver's total
                        if driver_number not in driver_points:
                            driver_points[driver_number] = 0
                        driver_points[driver_number] += points
        
        # Compile standings
        standings = []
        for driver_number, points in driver_points.items():
            if driver_number in driver_info:
                standings.append({
                    'driver_number': driver_number,
                    'name': driver_info[driver_number]['name'],
                    'name_acronym': driver_info[driver_number]['name_acronym'],
                    'team': driver_info[driver_number]['team'],
                    'team_color': driver_info[driver_number]['team_color'],
                    'points': points
                })
        
        # Sort by points (descending)
        standings = sorted(standings, key=lambda x: x['points'], reverse=True)
        
        # Add position information
        for i, driver in enumerate(standings, 1):
            driver['position'] = i
        
        # Cache the results
        save_to_cache(standings, cache_filename)
        return standings
    except Exception as e:
        logger.error(f"Error fetching driver standings: {str(e)}")
        return []

@st.cache_resource
def fetch_constructor_standings(season: int = None) -> List[Dict]:
    """Fetch constructor championship standings for a specific season."""
    try:
        if not season:
            season = datetime.now().year  # Default to current year
            
        cache_filename = f"constructor_standings_{season}.json"
        cached_data = load_from_cache(cache_filename)
        if cached_data:
            logger.info(f"Loading constructor standings from cache for season {season}")
            return cached_data

        logger.info(f"Fetching constructor standings for season {season}")
        
        # Get driver standings first
        driver_standings = fetch_driver_standings(season)
        if not driver_standings:
            return []
        
        # Group points by team
        team_points = {}
        team_colors = {}
        team_drivers = {}
        
        for driver in driver_standings:
            team = driver['team']
            points = driver['points']
            
            if team not in team_points:
                team_points[team] = 0
                team_colors[team] = driver['team_color']
                team_drivers[team] = []
                
            team_points[team] += points
            team_drivers[team].append({
                'name': driver['name'],
                'name_acronym': driver['name_acronym'],
                'driver_number': driver['driver_number'],
                'points': driver['points']
            })
        
        # Compile standings
        standings = []
        for team, points in team_points.items():
            standings.append({
                'team': team,
                'points': points,
                'team_color': team_colors[team],
                'drivers': team_drivers[team]
            })
        
        # Sort by points (descending)
        standings = sorted(standings, key=lambda x: x['points'], reverse=True)
        
        # Add position information
        for i, team in enumerate(standings, 1):
            team['position'] = i
        
        # Cache the results
        save_to_cache(standings, cache_filename)
        return standings
    except Exception as e:
        logger.error(f"Error fetching constructor standings: {str(e)}")
        return []

def process_telemetry_data(telemetry_data: List[Dict], session_key: int, resampling_interval: str = '1S') -> pd.DataFrame:
    """Process raw telemetry data into a pandas DataFrame and trim to session duration."""
    if not telemetry_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(telemetry_data)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get session start and end times
    session_details = fetch_session_details(session_key)
    if session_details:
        session_start = pd.to_datetime(session_details['date_start'])
        session_end = pd.to_datetime(session_details['date_end'])
        
        # Filter data to only include points within session duration
        df = df[(df['date'] >= session_start) & (df['date'] <= session_end)]
    
    # Add DRS status text
    df['drs_status'] = df['drs'].map(DRS_STATUS)
    
    # Set date as index for resampling
    df = df.set_index('date')
    
    # Resample data to reduce points while preserving trends
    # For numeric columns, use mean
    numeric_cols = ['speed', 'rpm', 'throttle', 'brake', 'n_gear']
    resampled_numeric = df[numeric_cols].resample(resampling_interval).mean()
    
    # For DRS status, use mode (most common value)
    resampled_drs = df['drs_status'].resample(resampling_interval).agg(lambda x: x.mode().iloc[0] if not x.empty else None)
    
    # Combine resampled data
    df_resampled = pd.concat([resampled_numeric, resampled_drs], axis=1)
    
    # Forward fill any missing values
    df_resampled = df_resampled.fillna(method='ffill')
    
    # Reset index to get date back as a column
    df_resampled = df_resampled.reset_index()
    
    return df_resampled

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

def is_valid_hex_color(hex_color: str) -> bool:
    """Check if a string is a valid hexadecimal color code."""
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    return len(hex_color) == 6 and all(c in '0123456789ABCDEFabcdef' for c in hex_color)

def safe_hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hexadecimal color code to an RGB tuple, with validation."""
    if not is_valid_hex_color(hex_color):
        return (128, 128, 128)  # Default to grey if invalid
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def display_driver_standings(driver_standings: List[Dict]):
    """Display driver standings in a styled table."""
    if not driver_standings:
        st.info("No driver standings data available.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(driver_standings)
    
    # Prepare data for display
    display_data = []
    for _, row in df.iterrows():
        display_data.append({
            'Position': row['position'],
            'Driver': f"{row['name_acronym']} ({row['driver_number']})",
            'Name': row['name'],
            'Team': row['team'],
            'Points': row['points']
        })
    
    # Create DataFrame for display
    display_df = pd.DataFrame(display_data)
    
    # Apply styling
    styled_df = display_df.style
    
    # Apply position styling
    def color_position(val):
        if val == 1:
            return 'background-color: gold; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
        elif val == 2:
            return 'background-color: silver; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
        elif val == 3:
            return 'background-color: #CD7F32; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
        return 'text-align: center;'
    
    styled_df = styled_df.applymap(color_position, subset=['Position'])
    
    # Apply team colors to team cells
    def color_team(val, teams):
        for team_data in teams:
            if val == team_data['team']:
                team_color = team_data['team_color']
                rgb = safe_hex_to_rgb(team_color)
                luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                text_color = 'white' if luminance < 0.5 else 'black'
                return f'background-color: {team_color}; color: {text_color}; font-weight: bold; border-radius: 4px; padding: 8px 12px;'
        return ''
    
    # Create a lookup dictionary for team colors
    team_data = [{'team': row['team'], 'team_color': row['team_color']} for _, row in df.iterrows()]
    
    # Apply team color function
    styled_df = styled_df.applymap(lambda x: color_team(x, team_data), subset=['Team'])
    
    # Apply points styling
    def color_points(val):
        if val > 0:
            return 'font-weight: bold;'
        return ''
    
    styled_df = styled_df.applymap(color_points, subset=['Points'])
    
    # Set table properties
    styled_df = styled_df.set_table_attributes('class="f1-table"')
    
    # Display the table
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

def display_constructor_standings(constructor_standings: List[Dict]):
    """Display constructor standings in a styled table."""
    if not constructor_standings:
        st.info("No constructor standings data available.")
        return
    
    # Create DataFrame
    display_data = []
    for team in constructor_standings:
        display_data.append({
            'Position': team['position'],
            'Team': team['team'],
            'Drivers': ', '.join([f"{driver['name_acronym']}" for driver in team['drivers']]),
            'Points': team['points']
        })
    
    # Create DataFrame for display
    display_df = pd.DataFrame(display_data)
    
    # Apply styling
    styled_df = display_df.style
    
    # Apply position styling
    def color_position(val):
        if val == 1:
            return 'background-color: gold; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
        elif val == 2:
            return 'background-color: silver; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
        elif val == 3:
            return 'background-color: #CD7F32; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
        return 'text-align: center;'
    
    styled_df = styled_df.applymap(color_position, subset=['Position'])
    
    # Apply team colors to team cells
    def color_team(val, teams):
        for team_data in teams:
            if val == team_data['team']:
                team_color = team_data['team_color']
                rgb = safe_hex_to_rgb(team_color)
                luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                text_color = 'white' if luminance < 0.5 else 'black'
                return f'background-color: {team_color}; color: {text_color}; font-weight: bold; border-radius: 4px; padding: 8px 12px;'
        return ''
    
    # Apply team color function
    styled_df = styled_df.applymap(lambda x: color_team(x, constructor_standings), subset=['Team'])
    
    # Apply points styling
    def color_points(val):
        if val > 0:
            return 'font-weight: bold;'
        return ''
    
    styled_df = styled_df.applymap(color_points, subset=['Points'])
    
    # Set table properties
    styled_df = styled_df.set_table_attributes('class="f1-table"')
    
    # Display the table
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

def display_team_details(constructor_standings: List[Dict]):
    """Display detailed information for each team."""
    if not constructor_standings:
        st.info("No constructor data available.")
        return
    
    # Display each team in a card with its drivers
    for team in constructor_standings:
        # Create a card with team color
        team_color = team['team_color']
        rgb = safe_hex_to_rgb(team_color)
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
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0;">{team['position']}. {team['team']}</h3>
                    <h3 style="margin: 0;">{team['points']} pts</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create a table for the team's drivers
        driver_data = []
        for driver in team['drivers']:
            driver_data.append({
                'Driver': f"{driver['name_acronym']} ({driver['driver_number']})",
                'Name': driver['name'],
                'Points': driver['points']
            })
        
        # Display driver table
        driver_df = pd.DataFrame(driver_data)
        styled_df = driver_df.style.set_table_attributes('class="f1-table"')
        st.dataframe(styled_df, hide_index=True, use_container_width=True)

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
        
        /* Table styling improvements */
        .dataframe {
            width: 100% !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        .dataframe th {
            background-color: #FF1801 !important;
            color: white !important;
            font-weight: bold !important;
            text-align: left !important;
            padding: 12px 15px !important;
            border-bottom: 2px solid #ddd !important;
            font-size: 1.1em !important;
        }
        .dataframe td {
            padding: 10px 15px !important;
            border-bottom: 1px solid #ddd !important;
            font-size: 1em !important;
            transition: background-color 0.3s ease !important;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f2f2f2 !important;
        }
        .dataframe tr:nth-child(odd) {
            background-color: #ffffff !important;
        }
        .dataframe tr:hover td {
            background-color: #e6e6e6 !important;
        }
        .dataframe tr:last-child td {
            border-bottom: 0 !important;
        }
        
        /* Responsive table for mobile */
        @media screen and (max-width: 768px) {
            .dataframe th, .dataframe td {
                padding: 8px 10px !important;
                font-size: 0.9em !important;
            }
        }
        
        /* Timeline table specific styles */
        .timeline-table {
            width: 100% !important;
        }
        .timeline-table td {
            vertical-align: middle !important;
        }
        
        /* Custom styling for F1 themed tables */
        .f1-table th {
            background-color: #15151E !important; 
            color: #FFFFFF !important;
            border-bottom: 3px solid #FF1801 !important;
        }
        .f1-table tr:nth-child(even) {
            background-color: #F8F8F8 !important;
        }
        .f1-table tr:nth-child(odd) {
            background-color: #FFFFFF !important;
        }
        
        /* Section headers */
        .section-header {
            color: #FF1801;
            font-weight: bold;
            border-bottom: 2px solid #FF1801;
            padding-bottom: 8px;
            margin-top: 25px;
            margin-bottom: 15px;
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
        
        # Create six columns for sub-navigation (removed track map nav)
        sub_nav_col1, sub_nav_col2, sub_nav_col3, sub_nav_col4, sub_nav_col5, sub_nav_col6 = st.columns(6)
        
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
                        
                        st.markdown('<p class="section-header">LEADERBOARD</p>', unsafe_allow_html=True)
                        
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
                                    return f'background-color: {team_color}; color: {text_color}; font-weight: bold; border-radius: 4px; padding: 8px 12px;'
                            return ''
                        
                        # Apply team colors to all driver cells
                        styled_df = styled_df.applymap(color_team, subset=['Driver'])
                        
                        # Apply position change colors for race/sprint sessions
                        def color_position_change(val):
                            if val.startswith('+'):
                                return 'color: green; font-weight: bold;'
                            elif val.startswith('-'):
                                return 'color: red; font-weight: bold;'
                            return ''
                        
                        if is_race_session:
                            styled_df = styled_df.applymap(color_position_change, subset=['Pos Change'])
                        
                        # Apply position styling
                        def color_position(val):
                            if val == 1:
                                return 'background-color: gold; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
                            elif val == 2:
                                return 'background-color: silver; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
                            elif val == 3:
                                return 'background-color: #CD7F32; color: black; font-weight: bold; text-align: center; border-radius: 4px;'
                            return 'text-align: center;'
                        
                        styled_df = styled_df.applymap(color_position, subset=['Position'])
                        
                        # Apply gap styling
                        def color_gap(val):
                            if val == 'Leader':
                                return 'color: gold; font-weight: bold;'
                            return ''
                        
                        styled_df = styled_df.applymap(color_gap, subset=['Gap'])
                        
                        # Set table properties
                        styled_df = styled_df.set_table_attributes('class="f1-table"')
                        
                        # Display the styled DataFrame
                        st.dataframe(styled_df, hide_index=True, use_container_width=True)
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
                    
                    # Add resampling interval control
                    st.write("#### Data Resolution Control")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        resampling_interval = st.slider(
                            "Adjust data resolution",
                            min_value=1,
                            max_value=120,  # Increased to 2 minutes
                            value=5,
                            step=1,
                            help="Lower values show more detail but may be slower to render. Higher values show smoother trends. Maximum is 2 minutes."
                        )
                    with col2:
                        st.write(f"Current interval: {resampling_interval:.1f}s")
                        if resampling_interval >= 60:
                            st.write(f"({resampling_interval/60:.1f} minutes)")
                    
                    # Fetch and process telemetry data
                    telemetry_data = fetch_car_telemetry(session_key, driver_number)
                    if telemetry_data:
                        df = process_telemetry_data(telemetry_data, session_key, f"{resampling_interval:.1f}S")
                        
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
                        
                        # Display data point count
                        st.info(f"Showing {len(df)} data points (original: {len(telemetry_data)})")
                    else:
                        st.info("No telemetry data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "pit_stops":
            st.markdown('<p class="section-header">PIT STOP ANALYSIS</p>', unsafe_allow_html=True)
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    pit_stops = fetch_pit_stops(session_key)
                    if pit_stops:
                        df = process_pit_stops(pit_stops, drivers)
                        
                        # Display overall statistics
                        st.markdown('<p class="section-header">OVERALL STATISTICS</p>', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Pit Stops", len(df))
                        with col2:
                            st.metric("Average Stop Time", f"{df['pit_duration'].mean():.1f}s")
                        with col3:
                            st.metric("Fastest Stop", f"{df['pit_duration'].min():.1f}s")
                        
                        # Display pit stop timeline
                        st.markdown('<p class="section-header">PIT STOP TIMELINE</p>', unsafe_allow_html=True)
                        timeline_data = df[['date', 'driver_name', 'pit_duration', 'lap_number']].copy()
                        timeline_data['date'] = timeline_data['date'].dt.strftime('%H:%M:%S')
                        
                        # Style timeline table
                        styled_timeline = timeline_data.style.set_table_attributes('class="timeline-table f1-table"')
                        
                        # Apply styling to pit duration based on speed
                        def color_pit_duration(val):
                            """Color code pit stop duration - faster stops are greener"""
                            if val < 2.5:  # Extraordinary stop
                                return 'background-color: #00FF00; color: black; font-weight: bold;'
                            elif val < 3.0:  # Very fast stop
                                return 'background-color: #66FF66; color: black; font-weight: bold;'
                            elif val < 4.0:  # Good stop
                                return 'background-color: #99FF99; color: black;'
                            elif val > 6.0:  # Slow stop
                                return 'background-color: #FFCCCC; color: black;'
                            return ''
                        
                        styled_timeline = styled_timeline.applymap(color_pit_duration, subset=['pit_duration'])
                        
                        # Add driver team colors
                        driver_colors = {f"{d['name_acronym']} ({d['driver_number']})": f"#{d['team_colour']}" 
                                       for d in drivers}
                        
                        def color_driver_cell(val):
                            """Apply team color to driver name cell"""
                            for driver_name, team_color in driver_colors.items():
                                if val == driver_name:
                                    # Convert hex to RGB to determine if background is light or dark
                                    hex_color = team_color.lstrip('#')
                                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                    # Calculate relative luminance
                                    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                                    # Use white text for dark backgrounds, black for light backgrounds
                                    text_color = 'white' if luminance < 0.5 else 'black'
                                    return f'background-color: {team_color}; color: {text_color}; font-weight: bold; border-radius: 4px;'
                            return ''
                        
                        styled_timeline = styled_timeline.applymap(color_driver_cell, subset=['driver_name'])
                        
                        st.dataframe(styled_timeline, hide_index=True, use_container_width=True)
                        
                        # Display team-wise analysis
                        st.markdown('<p class="section-header">TEAM ANALYSIS</p>', unsafe_allow_html=True)
                        team_stats = df.groupby('team_name').agg({
                            'pit_duration': ['count', 'mean', 'min']
                        }).round(2)
                        team_stats.columns = ['Total Stops', 'Average Duration', 'Fastest Stop']
                        
                        # Style team analysis table
                        styled_team = team_stats.style.set_table_attributes('class="f1-table"')
                        
                        # Apply background color based on fastest pit stop
                        def highlight_fastest(s, prop='background-color', color='#FFFF00'):
                            """Highlight the fastest pit stop"""
                            is_min = s == s.min()
                            return [f'{prop}: {color}; color: black; font-weight: bold;' if v else '' for v in is_min]
                        
                        styled_team = styled_team.apply(highlight_fastest, axis=0, subset=['Fastest Stop'])
                        
                        # Apply background color based on pit stop count
                        def highlight_most_stops(s, prop='background-color', color='#AADDFF'):
                            """Highlight team with most pit stops"""
                            is_max = s == s.max()
                            return [f'{prop}: {color}; color: black; font-weight: bold;' if v else '' for v in is_max]
                        
                        styled_team = styled_team.apply(highlight_most_stops, axis=0, subset=['Total Stops'])
                        
                        st.dataframe(styled_team, use_container_width=True)
                        
                        # Display driver-wise analysis
                        st.markdown('<p class="section-header">DRIVER ANALYSIS</p>', unsafe_allow_html=True)
                        driver_stats = df.groupby('driver_name').agg({
                            'pit_duration': ['count', 'mean', 'min']
                        }).round(2)
                        driver_stats.columns = ['Total Stops', 'Average Duration', 'Fastest Stop']
                        
                        # Style driver analysis table
                        styled_driver = driver_stats.style.set_table_attributes('class="f1-table"')
                        
                        # Apply highlighting to driver table
                        styled_driver = styled_driver.apply(highlight_fastest, axis=0, subset=['Fastest Stop'])
                        styled_driver = styled_driver.apply(highlight_most_stops, axis=0, subset=['Total Stops'])
                        
                        st.dataframe(styled_driver, use_container_width=True)
                    else:
                        st.info("No pit stop data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
        
        elif st.session_state.current_subview == "stints":
            st.markdown('<p class="section-header">TYRE STINT ANALYSIS</p>', unsafe_allow_html=True)
            if session_key:
                drivers = fetch_drivers(session_key)
                if drivers:
                    stints = fetch_stints(session_key)
                    if stints:
                        df = process_stints(stints, drivers)
                        
                        # Display overall statistics
                        st.markdown('<p class="section-header">OVERALL STATISTICS</p>', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Stints", len(df))
                        with col2:
                            st.metric("Average Stint Length", f"{df['stint_length'].mean():.1f} laps")
                        with col3:
                            st.metric("Longest Stint", f"{df['stint_length'].max()} laps")
                        
                        # Display stint timeline
                        st.markdown('<p class="section-header">STINT TIMELINE</p>', unsafe_allow_html=True)
                        timeline_data = df[['driver_name', 'compound', 'lap_start', 'lap_end', 'stint_length']].copy()
                        timeline_data = timeline_data.sort_values('lap_start')
                        
                        # Style timeline table
                        styled_timeline = timeline_data.style.set_table_attributes('class="timeline-table f1-table"')
                        
                        # Apply styling to compounds
                        def color_compound(val):
                            """Color code tyre compounds"""
                            compounds = {
                                'SOFT': '#FF3333',
                                'MEDIUM': '#FFCC33',
                                'HARD': '#FFFFFF',
                                'INTERMEDIATE': '#33CC33',
                                'WET': '#3333FF',
                                'C1': '#FFFFFF',
                                'C2': '#FFFFCC',
                                'C3': '#FFCC33',
                                'C4': '#FF9933',
                                'C5': '#FF3333'
                            }
                            if val.upper() in compounds:
                                bg_color = compounds[val.upper()]
                                text_color = 'black' if val.upper() not in ['WET'] else 'white'
                                return f'background-color: {bg_color}; color: {text_color}; font-weight: bold; border-radius: 4px; text-align: center;'
                            return ''
                        
                        styled_timeline = styled_timeline.applymap(color_compound, subset=['compound'])
                        
                        # Apply styling to stint length
                        def color_stint_length(val):
                            """Color code stint length - longer stints are darker green"""
                            if val > 30:  # Very long stint
                                return 'background-color: #006600; color: white; font-weight: bold;'
                            elif val > 20:  # Long stint
                                return 'background-color: #009900; color: white; font-weight: bold;'
                            elif val > 10:  # Medium stint
                                return 'background-color: #33CC33; color: black;'
                            elif val < 5:  # Very short stint
                                return 'background-color: #FFCCCC; color: black;'
                            return ''
                        
                        styled_timeline = styled_timeline.applymap(color_stint_length, subset=['stint_length'])
                        
                        # Add driver team colors
                        driver_colors = {f"{d['name_acronym']} ({d['driver_number']})": f"#{d['team_colour']}" 
                                       for d in drivers}
                        
                        def color_driver_cell(val):
                            """Apply team color to driver name cell"""
                            for driver_name, team_color in driver_colors.items():
                                if val == driver_name:
                                    # Convert hex to RGB to determine if background is light or dark
                                    hex_color = team_color.lstrip('#')
                                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                    # Calculate relative luminance
                                    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                                    # Use white text for dark backgrounds, black for light backgrounds
                                    text_color = 'white' if luminance < 0.5 else 'black'
                                    return f'background-color: {team_color}; color: {text_color}; font-weight: bold; border-radius: 4px;'
                            return ''
                        
                        styled_timeline = styled_timeline.applymap(color_driver_cell, subset=['driver_name'])
                        
                        st.dataframe(styled_timeline, hide_index=True, use_container_width=True)
                        
                        # Display compound analysis
                        st.markdown('<p class="section-header">COMPOUND ANALYSIS</p>', unsafe_allow_html=True)
                        compound_stats = df.groupby('compound').agg({
                            'stint_length': ['count', 'mean', 'max']
                        }).round(2)
                        compound_stats.columns = ['Total Stints', 'Average Length', 'Longest Stint']
                        
                        # Style compound analysis table
                        styled_compound = compound_stats.style.set_table_attributes('class="f1-table"')
                        
                        # Apply background color based on stint counts and length
                        def highlight_most_used(s, prop='background-color', color='#AADDFF'):
                            """Highlight most used compound"""
                            is_max = s == s.max()
                            return [f'{prop}: {color}; color: black; font-weight: bold;' if v else '' for v in is_max]
                        
                        styled_compound = styled_compound.apply(highlight_most_used, axis=0, subset=['Total Stints'])
                        styled_compound = styled_compound.apply(highlight_most_used, axis=0, subset=['Longest Stint'])
                        
                        st.dataframe(styled_compound, use_container_width=True)
                        
                        # Display team-wise analysis
                        st.markdown('<p class="section-header">TEAM ANALYSIS</p>', unsafe_allow_html=True)
                        team_stats = df.groupby('team_name').agg({
                            'stint_length': ['count', 'mean', 'max']
                        }).round(2)
                        team_stats.columns = ['Total Stints', 'Average Length', 'Longest Stint']
                        
                        # Style team analysis table
                        styled_team = team_stats.style.set_table_attributes('class="f1-table"')
                        styled_team = styled_team.apply(highlight_most_used, axis=0, subset=['Total Stints'])
                        styled_team = styled_team.apply(highlight_most_used, axis=0, subset=['Longest Stint'])
                        
                        st.dataframe(styled_team, use_container_width=True)
                        
                        # Display driver-wise analysis
                        st.markdown('<p class="section-header">DRIVER ANALYSIS</p>', unsafe_allow_html=True)
                        driver_stats = df.groupby('driver_name').agg({
                            'stint_length': ['count', 'mean', 'max']
                        }).round(2)
                        driver_stats.columns = ['Total Stints', 'Average Length', 'Longest Stint']
                        
                        # Style driver analysis table
                        styled_driver = driver_stats.style.set_table_attributes('class="f1-table"')
                        styled_driver = styled_driver.apply(highlight_most_used, axis=0, subset=['Total Stints'])
                        styled_driver = styled_driver.apply(highlight_most_used, axis=0, subset=['Longest Stint'])
                        
                        st.dataframe(styled_driver, use_container_width=True)
                    else:
                        st.info("No stint data available for this session.")
                else:
                    st.error("Failed to load driver details. Please try again later.")
            else:
                st.error("Failed to fetch session key. Please try again later.")
    
    elif st.session_state.current_view == "drivers":
        st.markdown('<p class="section-header">DRIVER CHAMPIONSHIP STANDINGS</p>', unsafe_allow_html=True)
        
        # Add season selector
        seasons = fetch_seasons()
        selected_season = st.selectbox("Select Season", seasons, index=0)
        
        # Fetch driver standings for the selected season
        with st.spinner("Loading driver standings..."):
            driver_standings = fetch_driver_standings(selected_season)
        
        if driver_standings:
            # Display championship leader info
            if len(driver_standings) > 0:
                leader = driver_standings[0]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            padding: 20px;
                            border-radius: 10px;
                            background-color: gold;
                            color: black;
                            text-align: center;
                            margin-bottom: 20px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <h2 style="margin: 0;">Championship Leader</h2>
                            <h1 style="margin: 10px 0;">{leader['name']}</h1>
                            <p style="font-size: 1.2em; margin: 0;">{leader['points']} Points</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col2:
                    team_color = leader['team_color']
                    rgb = safe_hex_to_rgb(team_color)
                    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                    text_color = 'white' if luminance < 0.5 else 'black'
                    
                    st.markdown(
                        f"""
                        <div style="
                            padding: 20px;
                            border-radius: 10px;
                            background-color: {team_color};
                            color: {text_color};
                            text-align: center;
                            margin-bottom: 20px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <h2 style="margin: 0;">Team</h2>
                            <h1 style="margin: 10px 0;">{leader['team']}</h1>
                            <p style="font-size: 1.2em; margin: 0;">Driver #{leader['driver_number']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display points gap visualization for top 5
            if len(driver_standings) >= 5:
                st.markdown('<p class="section-header">CHAMPIONSHIP BATTLE - TOP 5</p>', unsafe_allow_html=True)
                
                top5 = driver_standings[:5]
                gap_data = {
                    'Driver': [f"{d['name_acronym']}" for d in top5],
                    'Points': [d['points'] for d in top5],
                    'Gap': [0] + [top5[0]['points'] - d['points'] for d in top5[1:]]
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create points bar chart
                    points_chart = pd.DataFrame(gap_data)
                    st.bar_chart(data=points_chart.set_index('Driver')['Points'])
                
                with col2:
                    # Create points gap table
                    gap_df = pd.DataFrame({
                        'Pos': [d['position'] for d in top5],
                        'Driver': [f"{d['name_acronym']} ({d['driver_number']})" for d in top5],
                        'Points': [d['points'] for d in top5],
                        'Gap to Leader': ['-'] + [f"+{top5[0]['points'] - d['points']}" for d in top5[1:]]
                    })
                    
                    styled_gap = gap_df.style.set_table_attributes('class="f1-table"')
                    st.dataframe(styled_gap, hide_index=True, use_container_width=True)
            
            # Display full standings
            st.markdown('<p class="section-header">FULL DRIVER STANDINGS</p>', unsafe_allow_html=True)
            display_driver_standings(driver_standings)
        else:
            st.error("Failed to load driver standings data.")
    
    elif st.session_state.current_view == "teams":
        st.markdown('<p class="section-header">CONSTRUCTOR CHAMPIONSHIP STANDINGS</p>', unsafe_allow_html=True)
        
        # Add season selector
        seasons = fetch_seasons()
        selected_season = st.selectbox("Select Season", seasons, index=0)
        
        # Fetch constructor standings for the selected season
        with st.spinner("Loading constructor standings..."):
            constructor_standings = fetch_constructor_standings(selected_season)
        
        if constructor_standings:
            # Display championship leader info
            if len(constructor_standings) > 0:
                leader = constructor_standings[0]
                
                team_color = leader['team_color']
                rgb = safe_hex_to_rgb(team_color)
                luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                text_color = 'white' if luminance < 0.5 else 'black'
                
                st.markdown(
                    f"""
                    <div style="
                        padding: 20px;
                        border-radius: 10px;
                        background-color: {team_color};
                        color: {text_color};
                        text-align: center;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        <h2 style="margin: 0;">Leading Constructor</h2>
                        <h1 style="margin: 10px 0;">{leader['team']}</h1>
                        <p style="font-size: 1.2em; margin: 5px 0;">{leader['points']} Points</p>
                        <p style="font-size: 1.1em; margin: 5px 0;">Drivers: {', '.join([driver['name_acronym'] for driver in leader['drivers']])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display points gap visualization for top teams
            if len(constructor_standings) >= 3:
                st.markdown('<p class="section-header">CONSTRUCTORS BATTLE</p>', unsafe_allow_html=True)
                
                # Create points bar chart
                chart_data = pd.DataFrame({
                    'Team': [team['team'] for team in constructor_standings],
                    'Points': [team['points'] for team in constructor_standings]
                })
                
                st.bar_chart(data=chart_data.set_index('Team')['Points'])
            
            # Display full standings
            st.markdown('<p class="section-header">FULL CONSTRUCTOR STANDINGS</p>', unsafe_allow_html=True)
            display_constructor_standings(constructor_standings)
            
            # Display team details
            st.markdown('<p class="section-header">TEAM DETAILS</p>', unsafe_allow_html=True)
            display_team_details(constructor_standings)
        else:
            st.error("Failed to load constructor standings data.")
    
if __name__ == "__main__":
    main()