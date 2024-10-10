import numpy as np
import requests
from sgp4.api import Satrec
from sgp4.api import jday
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # For the progress bar

# Function to fetch TLE data from a URL
def fetch_tle_data(url):
    print(f"Fetching TLE data from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        tle_data = response.text.splitlines()
        return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    else:
        raise Exception(f"Error fetching TLE data: {response.status_code}")

# Convert epoch year and days to a timezone-aware datetime object in UTC
def convert_epoch_to_datetime(epochyr, epochdays):
    year = int(epochyr)
    if year < 57:  # Handling for years below 57 (assumed to be post-2000, as per SGP4 standard)
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1) + timedelta(days=epochdays - 1)
    return epoch.replace(tzinfo=timezone.utc)  # Set timezone to UTC

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
    for t in np.linspace(0, time_range, 10):  # Reduced points for clarity
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            positions.append((r, v))  # Store both position and velocity
        else:
            print(f"Error {e}: skipping {name}")
            return None
    return positions

# Function to plot key locations and short trajectories
def plot_orbits_and_trajectories(active_positions, debris_positions, ax):
    collision_points = []  # Store collision points

    # Plot active satellites as points with short trajectories
    for positions in active_positions:
        pos_vals, vel_vals = zip(*positions)
        x_vals, y_vals, z_vals = zip(*[pos for pos, vel in positions])
        ax.scatter(x_vals, y_vals, z_vals, color='blue', alpha=0.7)  # Plot positions as dots

        # Plot short trajectories based on velocity
        for (pos, vel) in positions:
            ax.plot([pos[0], pos[0] + vel[0]], [pos[1], pos[1] + vel[1]], [pos[2], pos[2] + vel[2]], color='blue', alpha=0.5)

    # Plot debris as points with short trajectories
    for positions_debris in debris_positions:
        pos_vals_debris, vel_vals_debris = zip(*positions_debris)
        x_vals_debris, y_vals_debris, z_vals_debris = zip(*[pos for pos, vel in positions_debris])
        ax.scatter(x_vals_debris, y_vals_debris, z_vals_debris, color='red', alpha=0.7)  # Plot positions as dots

        # Plot short trajectories based on velocity
        for (pos, vel) in positions_debris:
            ax.plot([pos[0], pos[0] + vel[0]], [pos[1], pos[1] + vel[1]], [pos[2], pos[2] + vel[2]], color='red', alpha=0.5)

    # Check for collisions and highlight collision points
    for positions in active_positions:
        for debris_positions in debris_positions:
            for active_pos, _ in positions:
                for debris_pos, _ in debris_positions:
                    if np.linalg.norm(np.array(active_pos) - np.array(debris_pos)) < 10:  # Collision threshold
                        collision_points.append(debris_pos)

    if collision_points:
        collision_points = np.array(collision_points)
        ax.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2], color='yellow', s=50, label="Collision Points")

    # Adjust axis limits for the plot
    ax.set_xlim([-50000, 50000])
    ax.set_ylim([-50000, 50000])
    ax.set_zlim([-50000, 50000])

# Function to create a single plot showing orbits and collisions
def create_orbit_plot(active_positions, debris_positions, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{title} - Orbits with Short Trajectories and Collision Points")

    plot_orbits_and_trajectories(active_positions, debris_positions, ax)

    plt.show()

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

# Calculate orbits for each TLE group
active_sats_positions = []
debris_positions = []

print("Starting to process TLE groups...")

for name, url in tqdm(tle_urls.items(), desc="Processing TLE groups", unit="group"):
    tle_groups = fetch_tle_data(url)  # Fetch TLE data for this group

    for tle_group in tqdm(tle_groups[:10], desc=f"Processing {name}", unit="satellite"):  # Limit to 10 objects per group for faster processing
        positions = calculate_orbit_positions(tle_group, time_range)
        if positions:
            if 'debris' in name.lower():  # Classify debris vs active satellites
                debris_positions.append(positions)
            else:
                active_sats_positions.append(positions)
        else:
            print(f"Skipping satellite due to error: {name}")

# Now create the plot showing orbits and collisions
create_orbit_plot(active_sats_positions, debris_positions, "3D Orbits with Short Trajectories and Collision Points")