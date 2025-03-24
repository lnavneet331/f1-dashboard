import main
import logging
import time
import sys
import os
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cache_prefetcher.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CachePrefetcher")

def ensure_cache_dir_exists():
    """Make sure the cache directory exists."""
    if not os.path.exists(main.CACHE_DIR):
        os.makedirs(main.CACHE_DIR, exist_ok=True)
        logger.info(f"Created cache directory at {main.CACHE_DIR}")

def prefetch_seasons():
    """Prefetch available seasons data."""
    logger.info("Prefetching seasons data...")
    all_seasons = main.fetch_seasons()
    
    # Filter to only include 2023-2025
    seasons = [s for s in all_seasons if 2023 <= s <= 2025]
    
    logger.info(f"Found {len(all_seasons)} total seasons, filtered to {len(seasons)} seasons (2023-2025): {seasons}")
    return seasons

def prefetch_driver_standings(seasons):
    """Prefetch driver standings for all seasons."""
    logger.info("Prefetching driver standings for all seasons...")
    for season in tqdm(seasons, desc="Driver Standings"):
        main.fetch_driver_standings(season)
        time.sleep(1)  # Slight delay to avoid hitting API rate limits

def prefetch_constructor_standings(seasons):
    """Prefetch constructor standings for all seasons."""
    logger.info("Prefetching constructor standings for all seasons...")
    for season in tqdm(seasons, desc="Constructor Standings"):
        main.fetch_constructor_standings(season)
        time.sleep(1)  # Slight delay to avoid hitting API rate limits

def prefetch_rounds(seasons):
    """Prefetch rounds for all seasons."""
    logger.info("Prefetching rounds data for all seasons...")
    all_rounds = {}
    
    for season in tqdm(seasons, desc="Rounds"):
        rounds = main.fetch_rounds(season)
        all_rounds[season] = rounds
        logger.info(f"Found {len(rounds)} rounds for season {season}")
        time.sleep(1)  # Slight delay to avoid hitting API rate limits
        
    return all_rounds

def prefetch_session_types(rounds_by_season):
    """Prefetch session types for all rounds."""
    logger.info("Prefetching session types for all rounds...")
    all_session_types = {}
    
    for season, rounds in rounds_by_season.items():
        for round_info in tqdm(rounds, desc=f"Session Types {season}"):
            meeting_key = round_info.get('meeting_key')
            if meeting_key:
                session_types = main.fetch_session_types(meeting_key)
                all_session_types[meeting_key] = session_types
                time.sleep(0.5)  # Slight delay to avoid hitting API rate limits
    
    return all_session_types

def prefetch_session_data(rounds_by_season, session_types_by_meeting):
    """Prefetch session data including drivers, positions, etc."""
    logger.info("Prefetching session data...")
    
    for season, rounds in rounds_by_season.items():
        for round_info in rounds:
            meeting_key = round_info.get('meeting_key')
            if not meeting_key or meeting_key not in session_types_by_meeting:
                continue
                
            round_name = round_info.get('meeting_name', 'Unknown')
            session_types = session_types_by_meeting[meeting_key]
            
            for session_type in tqdm(session_types, desc=f"{season} {round_name}"):
                try:
                    # Fetch session key
                    session_key = main.fetch_session_key(meeting_key, session_type)
                    if not session_key:
                        logger.warning(f"No session key found for {season} {round_name} {session_type}")
                        continue
                        
                    logger.info(f"Prefetching data for {season} {round_name} {session_type} (session_key: {session_key})")
                    
                    # Fetch basic session data
                    main.fetch_session_details(session_key)
                    weather_data = main.fetch_weather(session_key)
                    race_control = main.fetch_race_control(session_key)
                    lap_times = main.fetch_lap_times(session_key)
                    positions = main.fetch_positions(session_key)
                    pit_stops = main.fetch_pit_stops(session_key)
                    stints = main.fetch_stints(session_key)
                    
                    # Log counts to verify data was fetched
                    logger.info(f"  - Weather data: {'Success' if weather_data else 'No data'}")
                    logger.info(f"  - Race control messages: {len(race_control) if race_control else 0}")
                    logger.info(f"  - Lap times records: {len(lap_times) if lap_times else 0}")
                    logger.info(f"  - Position records: {len(positions) if positions else 0}")
                    logger.info(f"  - Pit stops: {len(pit_stops) if pit_stops else 0}")
                    logger.info(f"  - Stint records: {len(stints) if stints else 0}")
                    
                    # Fetch driver data
                    drivers = main.fetch_drivers(session_key)
                    if not drivers:
                        logger.warning(f"No drivers found for {season} {round_name} {session_type}")
                        continue
                        
                    logger.info(f"  - Found {len(drivers)} drivers")
                    
                    # Process all drivers for recent races, limit to top drivers for older races
                    # to balance comprehensive data with API load
                    driver_limit = 10 if season >= (datetime.now().year - 2) else 5
                    
                    for i, driver in enumerate(drivers):
                        if i >= driver_limit:
                            logger.info(f"  - Limiting to {driver_limit} drivers for this session")
                            break
                            
                        driver_number = driver['driver_number']
                        driver_name = driver.get('name_acronym', str(driver_number))
                        
                        logger.info(f"  - Prefetching data for driver {driver_name} ({driver_number})")
                        
                        # Fetch telemetry data
                        car_data = main.fetch_car_data(session_key, driver_number)
                        logger.info(f"    - Car telemetry points: {len(car_data) if car_data else 0}")
                        
                        # Fetch team radio
                        team_radio = main.fetch_team_radio(session_key, driver_number)
                        logger.info(f"    - Team radio messages: {len(team_radio) if team_radio else 0}")
                        
                        # Fetch location data
                        location_data = main.fetch_location_data(session_key, driver_number)
                        logger.info(f"    - Location data points: {len(location_data) if location_data else 0}")
                        
                        # Break if this is a very old race with no telemetry/radio
                        if not car_data and not team_radio and not location_data and season < (datetime.now().year - 3):
                            logger.info(f"  - No detailed data found for this driver in older session, skipping remaining drivers")
                            break
                        
                        time.sleep(0.75)  # Slightly longer delay to respect API limits
                    
                    time.sleep(1.5)  # Increased delay between sessions
                
                except Exception as e:
                    logger.error(f"Error prefetching data for {season} {round_name} {session_type}: {str(e)}")
                    time.sleep(3)  # Longer delay after an error to recover
                    continue
            
            # Add a progress indicator between rounds
            logger.info(f"Completed prefetching for {season} {round_name}")

def prefetch_specific_session(season, round_number, session_type=None):
    """Prefetch data for a specific session or all sessions in a round."""
    # Validate season is within our range
    if season < 2023 or season > 2025:
        logger.error(f"Season {season} is outside the supported range (2023-2025)")
        return
        
    logger.info(f"Prefetching specific data for season {season}, round {round_number}")
    
    # Ensure cache directory exists
    ensure_cache_dir_exists()
    
    # Fetch rounds for the season
    rounds = main.fetch_rounds(season)
    if not rounds:
        logger.error(f"No rounds found for season {season}")
        return
    
    # Find the specified round
    target_round = None
    for r in rounds:
        if r.get('round') == round_number:
            target_round = r
            break
    
    if not target_round:
        logger.error(f"Round {round_number} not found for season {season}")
        return
    
    meeting_key = target_round.get('meeting_key')
    if not meeting_key:
        logger.error(f"No meeting key found for round {round_number}")
        return
    
    # Fetch session types
    session_types = main.fetch_session_types(meeting_key)
    if not session_types:
        logger.error(f"No session types found for round {round_number}")
        return
    
    # If a specific session type is requested, filter to just that one
    if session_type:
        if session_type in session_types:
            session_types = [session_type]
        else:
            logger.error(f"Session type {session_type} not found. Available types: {session_types}")
            return
    
    # Create a dictionary structure to match the expected format for prefetch_session_data
    rounds_by_season = {season: [target_round]}
    session_types_by_meeting = {meeting_key: session_types}
    
    # Prefetch the data
    prefetch_session_data(rounds_by_season, session_types_by_meeting)
    
    logger.info(f"Completed prefetching specific data for season {season}, round {round_number}")

def main_prefetch():
    """Main function to prefetch all cache data."""
    start_time = time.time()
    logger.info("Starting cache prefetch process...")
    
    # Ensure cache directory exists
    ensure_cache_dir_exists()
    
    # Prefetch basic data
    seasons = prefetch_seasons()
    
    # Prefetch standings
    prefetch_driver_standings(seasons)
    prefetch_constructor_standings(seasons)
    
    # Prefetch rounds and session types
    rounds_by_season = prefetch_rounds(seasons)
    session_types_by_meeting = prefetch_session_types(rounds_by_season)
    
    # Prefetch detailed session data
    prefetch_session_data(rounds_by_season, session_types_by_meeting)
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Cache prefetch completed in {total_time:.2f} seconds")
    logger.info(f"Cache directory: {main.CACHE_DIR}")

if __name__ == "__main__":
    try:
        # Check if specific parameters were provided
        if len(sys.argv) > 2:
            # Format: python cache_prefetcher.py season round [session_type]
            season = int(sys.argv[1])
            
            # Validate season is within range
            if season < 2023 or season > 2025:
                logger.error(f"Season {season} is outside the supported range (2023-2025)")
                sys.exit(1)
                
            round_number = int(sys.argv[2])
            session_type = sys.argv[3] if len(sys.argv) > 3 else None
            
            logger.info(f"Running in specific mode: Season {season}, Round {round_number}, Session: {session_type}")
            prefetch_specific_session(season, round_number, session_type)
        else:
            # Run the full prefetch
            logger.info("Running full prefetch")
            main_prefetch()
            
    except KeyboardInterrupt:
        logger.info("Cache prefetch interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during cache prefetch: {str(e)}")
        sys.exit(1)
