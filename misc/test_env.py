# implement_combined_satellite_model.py

import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from stable_baselines3 import PPO
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
from utils.eval.satellite_avoidance_env import SatelliteAvoidanceEnv
from astropy.constants import G
from astropy import units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from poliastro.bodies import Earth, Moon
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

# Constants
EARTH_RADIUS = Earth.R.to(u.m).value  # Earth's radius in meters
MOON_RADIUS = Moon.R.to(u.m).value    # Moon's radius in meters

# Use JPL ephemeris for accurate positions
solar_system_ephemeris.set('de430')

# TLE data URLs and local file paths
TLE_URLS = {
    "Last 30 Days' Launches": 'tle_data/Last_30_Days_Launches.tle',
    "Active Satellites": 'tle_data/Active_Satellites.tle',
    "Russian ASAT Test Debris (COSMOS 1408)": 'tle_data/Russian_ASAT_Test_Debris_(COSMOS_1408).tle',
    "Chinese ASAT Test Debris (FENGYUN 1C)": 'tle_data/Chinese_ASAT_Test_Debris_(FENGYUN_1C).tle',
    "IRIDIUM 33 Debris": 'tle_data/IRIDIUM_33_Debris.tle',
    "COSMOS 2251 Debris": 'tle_data/COSMOS_2251_Debris.tle'
}

def fetch_tle_data(local_file_path):
    """Load TLE data from local file."""
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

# Add constant before calculate_orbit_positions function
MAX_DISTANCE = 3e5  # Maximum allowed distance from Earth in meters (300,000m)

def calculate_orbit_positions(tle_group, time_range):
    """Calculate satellite positions using TLE data."""
    name, line1, line2 = tle_group
    satellite = Satrec.twoline2rv(line1, line2)

    # Check if the TLE data is outdated
    if tle_is_outdated(satellite.epochyr, satellite.epochdays):
        return None

    positions = []
    for t in np.linspace(0, time_range, 1000):  # Increase the number of points for smoothness
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            distance = np.linalg.norm(r)
            if distance <= MAX_DISTANCE:  # Add distance check
                positions.append(r)
        else:
            return None  # Skip this satellite if there's an error
    return positions

def calculate_orbits_parallel(tle_groups, time_range):
    """Parallel processing for orbit calculations."""
    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(lambda tle_group: calculate_orbit_positions(tle_group, time_range), tle_groups), total=len(tle_groups)))

def average_distance_from_earth(positions):
    """Calculate the average distance from Earth given a list of 3D positions."""
    earth_center = np.array([0, 0, 0])  # Earth is centered at the origin
    distances = [np.linalg.norm(pos - earth_center) for pos in positions]
    return np.mean(distances)

def dynamic_orbit_duration_inverse(distance, min_duration=50, max_duration=500, scaling_factor=5000):
    """Dynamically scale the number of orbit points based on inverse distance from Earth."""
    if distance < scaling_factor:
        duration = int(max_duration - (distance / scaling_factor) * (max_duration - min_duration))
    else:
        duration = max_duration
    return duration

def create_earth_model():
    """Create a 3D Earth model."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))  # Earth radius ~6371 km
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))

    earth_model = go.Surface(
        x=x, y=y, z=z,
        colorscale='earth',
        cmin=0, cmax=1,
        showscale=False,
        hoverinfo='skip',
        opacity=1
    )

    return earth_model

def create_moon_model():
    # Create the Moon model (approximate position and scale)
    moon_distance = 384400 * 1000  # Average distance to the Moon in meters
    moon_x = moon_distance / 1000  # Simple placement along x-axis for visualization
    moon_u = np.linspace(0, 2 * np.pi, 100)
    moon_v = np.linspace(0, np.pi, 100)
    moon_x_vals = (MOON_RADIUS / 1000) * np.outer(np.cos(moon_u), np.sin(moon_v)) + moon_x
    moon_y_vals = (MOON_RADIUS / 1000) * np.outer(np.sin(moon_u), np.sin(moon_v))
    moon_z_vals = (MOON_RADIUS / 1000) * np.outer(np.ones(np.size(moon_u)), np.cos(moon_v))
    moon_model = go.Surface(
        x=moon_x_vals, y=moon_y_vals, z=moon_z_vals,
        colorscale='gray',
        cmin=0, cmax=1,
        showscale=False,
        opacity=0.9,
        hoverinfo='skip',
        name='Moon'
    )

    return moon_model

def plot_orbits_and_collisions_plotly(
    active_positions,
    debris_positions,
    model_trajectories,
    use_dynamic_scaling=True,
    scaling_factor=5000,
    export_animation=False,
    export_path_html="animation.html",
    export_path_gif="animation.gif"
):
    """Plot orbits and optionally export an animated HTML and GIF."""
    import os
    from PIL import Image
    import plotly.io as pio

    fig = go.Figure()

    # Number of frames for the animation
    print("Creating animation frames...")
    num_frames = 10

    # Initialize frames list
    frames = []

    # Generate frames
    for frame_num in range(num_frames):
        frame_data = []

        # Calculate normalized time
        t = frame_num / num_frames

        # Active satellites
        for positions in active_positions:
            idx = int(t * (len(positions) - 1))
            current_pos = positions[idx]
            frame_data.append(go.Scatter3d(
                x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
                mode='markers',
                marker=dict(size=4, color='cyan'),
                name='Active Satellite'
            ))

        # Debris
        for positions_debris in debris_positions:
            idx = int(t * (len(positions_debris) - 1))
            current_pos = positions_debris[idx]
            frame_data.append(go.Scatter3d(
                x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
                mode='markers',
                marker=dict(size=2, color='yellow'),
                name='Debris'
            ))

        # Model satellites
        for idx_model, model_trajectory in enumerate(model_trajectories):
            idx = int(t * (len(model_trajectory) - 1))
            current_pos = model_trajectory[idx]
            frame_data.append(go.Scatter3d(
                x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
                mode='markers',
                marker=dict(size=5, color='lime'),
                name=f'Model Satellite {idx_model+1}'
            ))

        # Add Earth model to the frame
        earth_model = create_earth_model()
        frame_data.append(earth_model)

        frames.append(go.Frame(data=frame_data, name=f'frame{frame_num}'))

    # Add initial data to the figure
    fig.add_traces(frames[0].data)

    # Configure layout for animation
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode="data",
            bgcolor="black"
        ),
        title='3D Orbits with Earth, Debris, and Model Satellites',
        showlegend=True,
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f'frame{k}'], {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                label=f'{k}'
            ) for k in range(num_frames)],
            active=0,
            transition={'duration': 0},
            x=0, y=0,
            currentvalue=dict(
                font=dict(size=12),
                prefix='Frame: ',
                visible=True,
                xanchor='center'
            ),
            len=1.0
        )]
    )

    # Assign frames to the figure
    fig.frames = frames

    if export_animation:
        # Create directory for frame images
        os.makedirs("frames", exist_ok=True)
        images = []

        # Generate and save images for each frame
        for i, frame in enumerate(frames):
            fig.update(data=frame.data)
            filename = f"frames/frame_{i:03d}.png"
            fig.write_image(filename)
            images.append(Image.open(filename))

        # Save images as a GIF
        images[0].save(
            export_path_gif,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )

        # Save animation as HTML
        pio.write_html(fig, file=export_path_html, auto_play=False)

    # Display the figure
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


# Main execution
if __name__ == '__main__':
    # Time range for simulation (e.g., 5 days)
    time_range = 86400 * 5  # 5 days in seconds

    # Fetch and calculate orbits in parallel
    active_sats_positions = []
    debris_positions = []

    for name, local_file_path in TLE_URLS.items():
        tle_groups = fetch_tle_data(local_file_path)  # Load TLE data from file
        positions = calculate_orbits_parallel(tle_groups[:100], time_range)  # Limit to 100 objects per group
        if 'debris' in name.lower():
            debris_positions.extend(filter(None, positions))
        else:
            active_sats_positions.extend(filter(None, positions))

    # Use the trained PPO model
    use_model = False

    # Create multiple environments for different satellites with unique heights and rotation angles
    satellite_configs = [
        {'distance': 700e3, 'angle': 45},
        {'distance': 10000e3, 'angle': 30},
        {'distance': 60000e3, 'angle': 60},
    ]

    environments = []
    for config in satellite_configs:
        # Remove random debris generation and use actual TLE debris data
        env = SatelliteAvoidanceEnv(
            debris_positions=debris_positions[:100],  # Use real TLE debris data
            max_debris=100,
            satellite_distance=config['distance'],
            init_angle=config['angle']
        )
        environments.append(env)

    # Load the saved PPO model
    model = PPO.load("models/satellite_avoidance_model_combined")

    # Initialize lists to store data for plotting and metrics
    all_model_trajectories = []
    performance_metrics = []  # List to store performance metrics for each satellite

    num_steps = 1000  # Number of steps to simulate

    # Test the model for each environment and collect positions for plotting
    for i, env in enumerate(environments):
        model_trajectory = []
        actions = []
        collision_occurred = False
        min_distance_to_debris = float('inf')
        obs, _ = env.reset()
        cumulative_reward = 0

        for _ in range(num_steps):
            action, _states = model.predict(obs)
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if 'collision_occurred' in info and info['collision_occurred']:
                collision_occurred = True
            model_trajectory.append((env.satellite_position / 1000).tolist())  # Convert to km
            # Calculate distance to nearest debris
            distances = [np.linalg.norm(env.satellite_position - debris) for debris in env.debris_positions]
            min_distance = min(distances)
            if min_distance < min_distance_to_debris:
                min_distance_to_debris = min_distance
            cumulative_reward += reward
            if done:
                break  # Exit the loop if done

        # Store the model trajectory
        all_model_trajectories.append(model_trajectory)

        # Compute performance metrics
        num_steps_taken = len(actions)
        total_delta_v = sum(np.linalg.norm(a) for a in actions)

        # Store performance metrics
        metrics = {
            'satellite_id': i + 1,
            'num_steps_taken': num_steps_taken,
            'total_delta_v': total_delta_v,
            'cumulative_reward': cumulative_reward,
            'collision_occurred': collision_occurred,
            'min_distance_to_debris': min_distance_to_debris
        }
        performance_metrics.append(metrics)

        # Print out performance metrics
        print(f"Satellite {i + 1} Performance Metrics:")
        print(f"  Total steps taken: {num_steps_taken}")
        print(f"  Total delta-v used: {total_delta_v:.4f} m/s")
        print(f"  Total cumulative reward: {cumulative_reward:.4f}")
        print(f"  Collision occurred: {collision_occurred}")
        print(f"  Minimum distance to debris: {min_distance_to_debris:.2f} meters")
        print()

    # Use dynamic scaling for plotting
    use_dynamic_scaling = True  # Set to True for dynamic orbit plotting, False for full orbits

    # Create the interactive 3D plot using Plotly for all satellites
    plot_orbits_and_collisions_plotly(
        active_positions=active_sats_positions,
        debris_positions=debris_positions,
        model_trajectories=all_model_trajectories,
        use_dynamic_scaling=use_dynamic_scaling,
        scaling_factor=50000,
        export_animation=True,
        export_path_html="orbit_animation.html",
        export_path_gif="orbit_animation.gif"
    )
