import numpy as np
import requests
from sgp4.api import Satrec
from sgp4.api import jday
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D

# Function to fetch TLE data from a URL
def fetch_tle_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        tle_data = response.text.splitlines()
        return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    else:
        raise Exception(f"Error fetching TLE data: {response.status_code}")

# Convert epoch year and days to a datetime object
def convert_epoch_to_datetime(epochyr, epochdays):
    year = int(epochyr)
    if year < 57:  # Handling for years below 57 (assumed to be post-2000, as per SGP4 standard)
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1) + timedelta(days=epochdays - 1)
    return epoch

# Check if the TLE data is outdated (e.g., older than 30 days)
def tle_is_outdated(epochyr, epochdays):
    tle_datetime = convert_epoch_to_datetime(epochyr, epochdays)
    days_old = (datetime.utcnow() - tle_datetime).days
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
    for t in np.linspace(0, time_range, 100):
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            positions.append(r)
        else:
            print(f"Error {e}: skipping {name}")
            return None  # Skip this satellite if there's an error
    return positions

# Function to plot orbits and check for collisions, collision tracks in red
def plot_orbits_and_collisions(active_positions, debris_positions, ax1, ax2):
    collision_points = []  # Store collision points

    # Plot active satellite orbits
    for positions in active_positions:
        x_vals, y_vals, z_vals = zip(*positions)
        ax1.plot(x_vals, y_vals, z_vals, color='blue', alpha=0.7)

    # Plot debris orbits and check for collisions
    for positions_debris in debris_positions:
        x_vals_debris, y_vals_debris, z_vals_debris = zip(*positions_debris)
        ax1.plot(x_vals_debris, y_vals_debris, z_vals_debris, color='red', alpha=0.7)

        # Check for collisions (where positions are close)
        for active_pos in active_positions:
            for debris_pos in positions_debris:
                if np.linalg.norm(np.array(active_pos) - np.array(debris_pos)) < 10:  # Collision threshold
                    collision_points.append(debris_pos)

    # Highlight collision points in the main plot
    if collision_points:
        collision_points = np.array(collision_points)
        ax1.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2], color='yellow', s=50, label="Collision Points")
        # Plot collision points in the second subplot
        ax2.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2], color='red', s=50)

    # Adjust axis limits for both plots
    ax1.set_xlim([-50000, 50000])
    ax1.set_ylim([-50000, 50000])
    ax1.set_zlim([-50000, 50000])
    ax2.set_xlim([-50000, 50000])
    ax2.set_ylim([-50000, 50000])
    ax2.set_zlim([-50000, 50000])

# Function to create a side-by-side plot
def create_side_by_side_plot(active_positions, debris_positions, title):
    fig = plt.figure(figsize=(16, 8))

    # Left: Orbits and collisions
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"{title} - Orbits")

    # Right: Collisions only
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"{title} - Collision Points")

    plot_orbits_and_collisions(active_positions, debris_positions, ax1, ax2)

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

for name, url in tle_urls.items():
    tle_groups = fetch_tle_data(url)  # Fetch TLE data for this group

    for tle_group in tle_groups[:10]:  # Limit to 100 objects per group for faster processing
        positions = calculate_orbit_positions(tle_group, time_range)
        if positions:
            if 'debris' in name.lower():  # Classify debris vs active satellites
                debris_positions.append(positions)
            else:
                active_sats_positions.append(positions)
        else:
            print(f"Skipping satellite due to error: {name}")

# Now create the side-by-side plot showing orbits and collisions
create_side_by_side_plot(active_sats_positions, debris_positions, "3D Orbits with Collision Points Highlighted")
