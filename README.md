# F1 Live Dashboard

A real-time Formula 1 dashboard built with Streamlit that displays live race data using the OpenF1 API.

## Features

- Real-time timing data
- Live car telemetry
- Weather conditions
- Race control messages
- Auto-refreshing dashboard

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

The dashboard will automatically connect to the OpenF1 API and display real-time data during F1 sessions. The data refreshes every 5 seconds.

Note: The dashboard will only show data during active F1 sessions (Practice, Qualifying, or Race sessions).

## Data Sources

This dashboard uses the OpenF1 API (https://api.openf1.org) to fetch real-time Formula 1 data. The API is free to use and doesn't require authentication. 