# implement_satellite_model.py

import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from stable_baselines3 import PPO
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
from eval.satellite_avoidance_env import SatelliteAvoidanceEnv
from astropy.constants import G
from astropy import units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from poliastro.bodies import Earth, Moon

EARTH_RADIUS = Earth.R.to(u.m).value  # Earth's radius in meters
MOON_RADIUS = Moon.R.to(u.m).value  # Moon's radius in meters


# TLE data URLs and local file paths
TLE_URLS = {
    "Last 30 Days' Launches": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle', 'tle_data/Last_30_Days_Launches.tle'),
    "Active Satellites": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle', 'tle_data/Active_Satellites.tle'),
    "Russian ASAT Test Debris (COSMOS 1408)": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle', 'tle_data/Russian_ASAT_Test_Debris_(COSMOS_1408).tle')
}

def fetch_tle_data(url, local_file_path):
    # Use only local file
    with open(local_file_path, 'r') as file:
        tle_data = file.read().splitlines()
        return [tle_data[i:i + 3] for i in range(0, len(tle_data), 3)]

def convert_epoch_to_datetime(epochyr, epochdays):
    """Convert epoch year and days to a timezone-aware datetime object."""
    year = int(epochyr)
    if year < 57:
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epochdays - 1)
    return epoch

def tle_is_outdated(epochyr, epochdays):
    """Check if the TLE data is outdated."""
    tle_datetime = convert_epoch_to_datetime(epochyr, epochdays)
    days_old = (datetime.now(timezone.utc) - tle_datetime).days
    return days_old > 30  # Treat as outdated if older than 30 days

# Function to calculate the average distance from Earth for a given set of positions
def average_distance_from_earth(positions):
    """Calculate the average distance from Earth given a list of 3D positions."""
    earth_center = np.array([0, 0, 0])  # Earth is centered at the origin
    distances = [np.linalg.norm(pos - earth_center) for pos in positions]
    return np.mean(distances)

# Function to scale orbit duration based on inverse distance
def dynamic_orbit_duration_inverse(distance, min_duration=50, max_duration=500, scaling_factor=5000):
    """Dynamically scale the number of orbit points based on inverse distance from Earth."""
    # Closer objects get shorter durations, farther objects get longer durations
    if distance < scaling_factor:
        duration = int(max_duration - (distance / scaling_factor) * (max_duration - min_duration))
    else:
        duration = max_duration
    return duration

# Function to create a 3D Earth model with a colorscale (instead of texture)
def create_earth_model():
    # Define spherical coordinates for the Earth model
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))  # Earth radius ~6371 km
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create a surface plot for the Earth model using a predefined colorscale (earth tones)
    earth_model = go.Surface(
        x=x, y=y, z=z,
        colorscale='earth',  # Use 'earth' colorscale to simulate the Earth texture
        cmin=0, cmax=1,
        showscale=False,
        hoverinfo='skip',
        opacity=1
    )

    return earth_model


# Function to plot orbits and check for collisions with Plotly
def plot_orbits_and_collisions_plotly(active_positions, debris_positions, model_trajectory, use_dynamic_scaling=True, scaling_factor=5000):
    fig = go.Figure()

    # Plot smooth trajectories of active satellites (dynamic or full path based on use_dynamic_scaling)
    for positions in active_positions:
        if use_dynamic_scaling:
            avg_distance = average_distance_from_earth(positions)
            orbit_duration = dynamic_orbit_duration_inverse(avg_distance, scaling_factor=scaling_factor)
            x_vals, y_vals, z_vals = zip(*positions[:orbit_duration])
        else:
            x_vals, y_vals, z_vals = zip(*positions)
            
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.5)', width=3),
            name=f'Satellite Orbit'
        ))

        current_pos = positions[0]
        fig.add_trace(go.Scatter3d(
            x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
            mode='markers',
            marker=dict(size=6, color='cyan', symbol='circle'),
            name='Current Satellite Position'
        ))

    # Plot smooth trajectories of debris
    for positions_debris in debris_positions:
        if use_dynamic_scaling:
            avg_distance = average_distance_from_earth(positions_debris)
            orbit_duration = dynamic_orbit_duration_inverse(avg_distance, scaling_factor=scaling_factor)
            x_vals_debris, y_vals_debris, z_vals_debris = zip(*positions_debris[:orbit_duration])
        else:
            x_vals_debris, y_vals_debris, z_vals_debris = zip(*positions_debris)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals_debris, y=y_vals_debris, z=z_vals_debris,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
            name=f'Debris Orbit'
        ))

        current_pos_debris = positions_debris[0]
        fig.add_trace(go.Scatter3d(
            x=[current_pos_debris[0]], y=[current_pos_debris[1]], z=[current_pos_debris[2]],
            mode='markers',
            marker=dict(size=6, color='yellow', symbol='circle'),
            name='Current Debris Position'
        ))

    # Plot model satellite path if available
    if model_trajectory:
        model_x_vals, model_y_vals, model_z_vals = zip(*model_trajectory)
        fig.add_trace(go.Scatter3d(
            x=model_x_vals, y=model_y_vals, z=model_z_vals,
            mode='lines+markers',
            line=dict(color='rgba(0, 255, 0, 0.7)', width=5),
            marker=dict(size=4, color='lime'),
            name='Model Satellite Path'
        ))

    # Create the Earth model
    earth_model = create_earth_model()

        # Create the Moon model (approximate position and scale)
    moon_distance = 384400 * 1000  # Average distance to the Moon in meters
    moon_x = moon_distance / 1000  # Simple placement along x-axis for visualization
    moon_u = np.linspace(0, 2 * np.pi, 100)
    moon_v = np.linspace(0, np.pi, 100)
    moon_x_vals = (MOON_RADIUS / 1000) * np.outer(np.cos(moon_u), np.sin(moon_v)) + moon_x
    moon_y_vals = (MOON_RADIUS / 1000) * np.outer(np.sin(moon_u), np.sin(moon_v))
    moon_z_vals = (MOON_RADIUS / 1000) * np.outer(np.ones(np.size(moon_u)), np.cos(moon_v))
    fig.add_trace(go.Surface(
        x=moon_x_vals, y=moon_y_vals, z=moon_z_vals,
        colorscale='gray',
        cmin=0, cmax=1,
        showscale=False,
        opacity=0.9,
        hoverinfo='skip',
        name='Moon'
    ))

    # Add Earth model to the figure
    fig.add_trace(earth_model)

    # Update layout for 3D plot with a space-like background
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode="data",
            bgcolor="black"
        ),
        title='3D Orbits with Earth and Debris',
        showlegend=True
    )

    fig.show()

def plot_orbits_gravitational(active_positions, debris_positions, model_trajectory):
    import plotly.graph_objects as go
    import numpy as np
    from astropy import units as u
    from poliastro.bodies import Earth, Moon, Sun
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time

    EARTH_RADIUS = Earth.R.to(u.km).value  # Earth's radius in km
    MOON_RADIUS = Moon.R.to(u.km).value    # Moon's radius in km

    # Get Moon position relative to Earth
    t0 = Time('2024-10-09 12:00:00', scale='utc')

    with solar_system_ephemeris.set('jpl'):
        earth_pos = get_body_barycentric('earth', t0)
        moon_pos = get_body_barycentric('moon', t0)

    moon_pos_rel = (moon_pos - earth_pos).get_xyz().to(u.km).value.flatten()

    # Determine plot boundaries based on maximum extent of the positions
    max_extent = max(
        np.max([np.max(np.abs(pos)) for pos in active_positions] +
               [np.max(np.abs(pos)) for pos in debris_positions] +
               [np.max(np.abs(moon_pos_rel))]),
        5 * EARTH_RADIUS  # Ensure a minimum size
    )

    # Create the figure
    fig = go.Figure()

    # Plot the Earth
    u_vals = np.linspace(0, 2 * np.pi, 100)
    v_vals = np.linspace(0, np.pi, 100)
    x_earth = EARTH_RADIUS * np.outer(np.cos(u_vals), np.sin(v_vals))
    y_earth = EARTH_RADIUS * np.outer(np.sin(u_vals), np.sin(v_vals))
    z_earth = EARTH_RADIUS * np.outer(np.ones(np.size(u_vals)), np.cos(v_vals))
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='earth',
        cmin=0, cmax=1,
        showscale=False,
        opacity=1,
        hoverinfo='skip',
        name='Earth'
    ))

    # Plot the Moon
    moon_scale_factor = 1  # Keep actual size for Moon
    x_moon = MOON_RADIUS * moon_scale_factor * np.outer(np.cos(u_vals), np.sin(v_vals)) + moon_pos_rel[0]
    y_moon = MOON_RADIUS * moon_scale_factor * np.outer(np.sin(u_vals), np.sin(v_vals)) + moon_pos_rel[1]
    z_moon = MOON_RADIUS * moon_scale_factor * np.outer(np.ones(np.size(u_vals)), np.cos(v_vals)) + moon_pos_rel[2]
    fig.add_trace(go.Surface(
        x=x_moon, y=y_moon, z=z_moon,
        colorscale='gray',
        cmin=0, cmax=1,
        showscale=False,
        opacity=1,
        hoverinfo='skip',
        name='Moon'
    ))

    # Plot gravitational waves for Earth and Moon originating from their positions
    grid_size = 500  # Increase grid size for better coverage
    wave_amplitude_earth = 0.5 * EARTH_RADIUS
    wave_amplitude_moon = 0.5 * MOON_RADIUS

    # Generate a larger grid to cover more space
    x_grid = np.linspace(-max_extent, max_extent, grid_size)
    y_grid = np.linspace(-max_extent, max_extent, grid_size)
    x_wave, y_wave = np.meshgrid(x_grid, y_grid)

    # Earth wave pattern
    dist_earth = np.sqrt((x_wave - 0)**2 + (y_wave - 0)**2)
    wave_earth = wave_amplitude_earth * np.sin(dist_earth * 2 * np.pi / EARTH_RADIUS)
    fig.add_trace(go.Surface(
        x=x_wave, y=y_wave, z=wave_earth,
        colorscale='Blues',
        opacity=0.1,
        showscale=False,
        name='Earth Wave'
    ))

    # Moon wave pattern
    x_moon_wave = np.linspace(-max_extent + moon_pos_rel[0], max_extent + moon_pos_rel[0], grid_size)
    y_moon_wave = np.linspace(-max_extent + moon_pos_rel[1], max_extent + moon_pos_rel[1], grid_size)
    x_moon_grid, y_moon_grid = np.meshgrid(x_moon_wave, y_moon_wave)
    dist_moon = np.sqrt((x_moon_grid - moon_pos_rel[0])**2 + (y_moon_grid - moon_pos_rel[1])**2)
    wave_moon = wave_amplitude_moon * np.sin(dist_moon * 2 * np.pi / MOON_RADIUS)
    fig.add_trace(go.Surface(
        x=x_moon_grid, y=y_moon_grid, z=wave_moon + moon_pos_rel[2],
        colorscale='Greens',
        opacity=0.1,
        showscale=False,
        name='Moon Wave'
    ))

    # Plot satellite trajectories
    for positions in active_positions:
        x_vals, y_vals, z_vals = zip(*positions)
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color='blue', width=1),
            name='Satellite Orbit'
        ))

    # Plot debris trajectories
    for positions in debris_positions:
        x_vals, y_vals, z_vals = zip(*positions)
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color='red', width=1),
            name='Debris Orbit'
        ))

    # Plot the model satellite trajectory
    if model_trajectory:
        x_vals, y_vals, z_vals = zip(*model_trajectory)
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=2, color='lime'),
            name='Model Satellite Path'
        ))

    # Update layout to center on Earth and adjust scale
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (km)', showbackground=False),
            yaxis=dict(title='Y (km)', showbackground=False),
            zaxis=dict(title='Z (km)', showbackground=False),
            aspectmode='data',
            bgcolor='black',
            camera=dict(
                eye=dict(x=2.5, y=2.5, z=2.5)  # Adjust the camera position for better view
            )
        ),
        title='Satellite Orbits with Earth and Moon Waves',
        showlegend=True
    )

    fig.show()

# Function to calculate satellite positions using TLE data
MAX_DISTANCE = 3e5  # Maximum allowed distance from Earth in meters

def calculate_orbit_positions(tle_group, time_range):
    name, line1, line2 = tle_group
    satellite = Satrec.twoline2rv(line1, line2)

    # Check if the TLE data is outdated
    if tle_is_outdated(satellite.epochyr, satellite.epochdays):
        print(f"Skipping outdated satellite: {name}")
        return None

    positions = []
    for t in np.linspace(0, time_range, 1000):  # Increase the number of points for smoothness
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            distance = np.linalg.norm(r)
            if distance <= MAX_DISTANCE:
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

# Parallel processing for orbit calculations
def calculate_orbits_parallel(tle_groups, time_range):
    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(lambda tle_group: calculate_orbit_positions(tle_group, time_range), tle_groups), total=len(tle_groups)))


# Example usage when fetching TLE data
tle_urls = {
    "Last 30 Days' Launches": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle', 'tle_data/Last_30_Days_Launches.tle'),
    "Active Satellites": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle', 'tle_data/Active_Satellites.tle'),
    "Russian ASAT Test Debris (COSMOS 1408)": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle', 'tle_data/Russian_ASAT_Test_Debris_(COSMOS_1408).tle'),
    "Chinese ASAT Test Debris (FENGYUN 1C)": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle', 'tle_data/Chinese_ASAT_Test_Debris_(FENGYUN_1C).tle'),
    "IRIDIUM 33 Debris": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle', 'tle_data/IRIDIUM_33_Debris.tle'),
    "COSMOS 2251 Debris": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle', 'tle_data/COSMOS_2251_Debris.tle')
}

# Time range for simulation (e.g., 5 days)
time_range = 86400 * 5

# Fetch and calculate orbits in parallel
active_sats_positions = []
debris_positions = []

for name, (url, local_file_path) in tle_urls.items():
    tle_groups = fetch_tle_data(url, local_file_path)  # Fetch TLE data for this group
    positions = calculate_orbits_parallel(tle_groups[:100], time_range)  # Limit to 100 objects per group
    if 'debris' in name.lower():
        debris_positions.extend(filter(None, positions))
    else:
        active_sats_positions.extend(filter(None, positions))

# Use the trained PPO model if set to True
use_model = True

if use_model:
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    env = SatelliteAvoidanceEnv(debris_positions_sample, satellite_distance=100000000e3)

    # Load the saved PPO model
    model = PPO.load("models/satellite_avoidance_model_advanced")

    # Test the model and collect positions for plotting
    model_trajectory = []
    obs, _ = env.reset()  # Here, no info is expected, so only capture obs
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        model_trajectory.append(env.satellite_position.tolist())
        if done:
            print("[DEBUG] Episode finished, resetting environment.")
            obs, _ = env.reset()
else:
    model_trajectory = None

# Check if a collision was detected or if the satellite avoided it successfully
collision_avoided = all(np.linalg.norm(env.satellite_position - debris) >= 10e3 for debris in env.debris_positions)

if collision_avoided:
    print("Satellite avoided all collisions.")
else:
    print("Satellite was on route to collide, collision detected.")


# Example usage when fetching TLE data
use_dynamic_scaling = True  # Set to True for dynamic orbit plotting, False for full orbits

# Convert model trajectory positions from meters to kilometers
model_trajectory_km = [(np.array(pos) / 1000).tolist() for pos in model_trajectory]

plot_orbits_gravitational(
    active_sats_positions,
    debris_positions,
    model_trajectory=model_trajectory_km
)

# Now create the interactive 3D plot using Plotly
plot_orbits_and_collisions_plotly(
    active_sats_positions,
    debris_positions,
    model_trajectory=model_trajectory_km,
    use_dynamic_scaling=use_dynamic_scaling,  # Control whether to dynamically scale or plot full paths
    scaling_factor=500  # Adjust this value to fine-tune dynamic scaling
)



