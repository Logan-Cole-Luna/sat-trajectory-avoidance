import numpy as np
import requests
from sgp4.api import Satrec
from sgp4.api import jday
import plotly.graph_objs as go
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Function to fetch TLE data from a URL
def fetch_tle_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        tle_data = response.text.splitlines()
        return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    else:
        raise Exception(f"Error fetching TLE data: {response.status_code}")

# Convert epoch year and days to a timezone-aware datetime object
def convert_epoch_to_datetime(epochyr, epochdays):
    year = int(epochyr)
    if year < 57:  # Handling for years below 57 (assumed to be post-2000, as per SGP4 standard)
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epochdays - 1)  # Make it timezone-aware
    return epoch

# Check if the TLE data is outdated (e.g., older than 30 days)
def tle_is_outdated(epochyr, epochdays):
    tle_datetime = convert_epoch_to_datetime(epochyr, epochdays)
    days_old = (datetime.now(timezone.utc) - tle_datetime).days  # Use timezone-aware UTC datetime
    return days_old > 30  # Example: treat as outdated if older than 30 days

# Function to calculate satellite positions using TLE data
def calculate_orbit_positions(tle_group, time_range):
    name, line1, line2 = tle_group
    satellite = Satrec.twoline2rv(line1, line2)

    # Check if the TLE data is outdated
    if tle_is_outdated(satellite.epochyr, satellite.epochdays):
        print(f"Skipping outdated satellite: {name}")
        return None

    positions = []
    for t in np.linspace(0, time_range, 500):  # Increase the number of points for smoothness
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            positions.append(r)
        else:
            print(f"Error {e}: skipping {name}")
            return None  # Skip this satellite if there's an error
    return positions

# Function to dynamically adjust axis limits based on data
def get_dynamic_limits(positions):
    all_positions = np.concatenate(positions)
    max_val = np.max(np.abs(all_positions))
    return [-max_val, max_val]

# Function to add a transparent Earth to the plot
def add_earth_to_plot(fig):
    # Create a transparent Earth for visual reference
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    r = 6371  # Approximate Earth radius in kilometers
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # Add surface for Earth
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'rgba(0, 0, 0, 0.1)'], [1, 'rgba(0, 0, 0, 0.1)']],  # Transparent Earth
        showscale=False
    ))

# Function to plot orbits and check for collisions with Plotly
def plot_orbits_and_collisions_plotly(active_positions, debris_positions, trajectory_length=10):
    fig = go.Figure()

    # Plot smooth trajectories of active satellites (without dots for past trajectory)
    for positions in active_positions:
        x_vals, y_vals, z_vals = zip(*positions[:-1])  # Trajectory excluding the last point
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',  # Line only for trajectory
            line=dict(color='blue', width=2),
            name='Satellite Trajectory'
        ))

        # Plot the current position as a single dot (last point)
        current_pos = positions[-1]
        fig.add_trace(go.Scatter3d(
            x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
            mode='markers',
            marker=dict(size=6, color='cyan', symbol='circle'),
            name='Current Satellite Position'
        ))

    # Plot smooth trajectories of debris (without dots for past trajectory)
    for positions_debris in debris_positions:
        x_vals_debris, y_vals_debris, z_vals_debris = zip(*positions_debris[:-1])  # Trajectory excluding the last point
        fig.add_trace(go.Scatter3d(
            x=x_vals_debris, y=y_vals_debris, z=z_vals_debris,
            mode='lines',
            line=dict(color='red', width=2),
            name='Debris Trajectory'
        ))

        # Plot the current position of debris as a single dot (last point)
        current_pos_debris = positions_debris[-1]
        fig.add_trace(go.Scatter3d(
            x=[current_pos_debris[0]], y=[current_pos_debris[1]], z=[current_pos_debris[2]],
            mode='markers',
            marker=dict(size=6, color='yellow', symbol='circle'),
            name='Current Debris Position'
        ))

    # Adjust layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', showgrid=True),  # Show grid
            yaxis=dict(title='Y', showgrid=True),  # Show grid
            zaxis=dict(title='Z', showgrid=True),  # Show grid
            bgcolor='lightgray',  # Optional background color
        ),
        title='3D Orbits with Current Positions and Trajectories',
        showlegend=True,
    )

    fig.show()

# Parallel processing for orbit calculations
def calculate_orbits_parallel(tle_groups, time_range):
    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(lambda tle_group: calculate_orbit_positions(tle_group, time_range), tle_groups), total=len(tle_groups)))

# Fetch TLE data from the provided URLs
tle_urls = {
    "Last 30 Days' Launches": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle',
    "Active Satellites": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
    "Russian ASAT Test Debris (COSMOS 1408)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle',
    "Chinese ASAT Test Debris (FENGYUN 1C)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle',
    "IRIDIUM 33 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle',
    "COSMOS 2251 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle'
}

# Time range for simulation (e.g., 5 days)
time_range = 86400 * 5

# Fetch and calculate orbits in parallel
active_sats_positions = []
debris_positions = []

for name, url in tle_urls.items():
    tle_groups = fetch_tle_data(url)  # Fetch TLE data for this group
    positions = calculate_orbits_parallel(tle_groups[:100], time_range)  # Limit to 100 objects per group for faster processing
    if 'debris' in name.lower():  # Classify debris vs active satellites
        debris_positions.extend(filter(None, positions))
    else:
        active_sats_positions.extend(filter(None, positions))

# Now create the interactive 3D plot using Plotly
plot_orbits_and_collisions_plotly(active_sats_positions, debris_positions, trajectory_length=15)