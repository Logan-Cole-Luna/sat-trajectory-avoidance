import numpy as np
import requests
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import Time
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Add imageio for creating MP4 videos
from utils.eval.satellite_avoidance_env import SatelliteAvoidanceEnv

# Ensure that imageio's ffmpeg backend is installed for MP4 export:
# Run the following command in your terminal:
# pip install imageio[ffmpeg]

# Constants for orbit and gravitational force
G = 6.67430e-11  # Gravitational constant in m^3 kg^−1 s^−2
M = 5.972e24     # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters
MAX_DISTANCE = 10000  # Maximum allowed distance from Earth in meters for debris

# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100, satellite_distance=4000e3, init_angle=0, collision_course=False):
        super(SatelliteAvoidanceEnv, self).__init__()

        # Action: changes in velocity in x, y, z directions
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(3,), dtype=np.float32)

        # Set maximum number of debris objects (debris_positions can be variable)
        self.max_debris = max_debris
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

        # Observation space: (3 for satellite position + 3 * max_debris for debris + 1 for dummy scalar)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + self.max_debris * 3 + 1,), dtype=np.float32
        )

        # Set initial orbit with configurable altitude and angle
        self.initial_orbit = Orbit.circular(Earth, alt=satellite_distance * u.m, inc=init_angle * u.deg)
        self.satellite_position = self.initial_orbit.r.to(u.m).value
        self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value

        # Calculate the orbital period
        self.orbital_period = self.initial_orbit.period.to(u.s).value  # Orbital period in seconds

        # Collision course flag
        self.collision_course = collision_course

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset to the initial circular orbit with angle
        self.satellite_position = self.initial_orbit.r.to(u.m).value
        self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value

        # Dynamically set a new number of debris
        num_debris = np.random.randint(1, self.max_debris + 1)
        self.debris_positions = [np.random.randn(3) * 10000 for _ in range(num_debris)]

        self.elapsed_time = 0.0
        # Calculate time increment per step
        self.time_increment = self.orbital_period / 500  # Adjust the denominator to control steps per orbit

        if self.collision_course:
            # Find the nearest debris
            distances = [np.linalg.norm(self.satellite_position - debris) for debris in self.debris_positions]
            min_index = np.argmin(distances)
            nearest_debris = self.debris_positions[min_index]
            # Calculate direction towards debris
            direction_to_debris = nearest_debris - self.satellite_position
            direction_to_debris_normalized = direction_to_debris / np.linalg.norm(direction_to_debris)
            # Adjust the satellite's velocity to point towards the debris
            speed = np.linalg.norm(self.satellite_velocity)
            # Increase speed slightly to ensure collision
            speed *= 1.05  # Increase speed by 5%
            self.satellite_velocity = direction_to_debris_normalized * speed

        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten debris positions
        debris_flat = np.array(self.debris_positions).flatten()

        # Truncate if there are more debris than max_debris
        if len(self.debris_positions) > self.max_debris:
            debris_flat = debris_flat[:self.max_debris * 3]
        # Pad debris positions if fewer than max_debris
        elif len(debris_flat) < self.max_debris * 3:
            debris_flat = np.pad(debris_flat, (0, self.max_debris * 3 - len(debris_flat)), mode='constant')

        # Return satellite position, debris positions, and a dummy scalar to match the model's observation size
        return np.concatenate([self.satellite_position, debris_flat, [0.0]])

    def _apply_gravitational_force(self):
        # Calculate the distance from the Earth's center
        r = np.linalg.norm(self.satellite_position)

        # Newton's law of universal gravitation
        force_magnitude = G * M / r**2
        force_direction = -self.satellite_position / r  # Direction towards Earth's center
        gravitational_force = force_magnitude * force_direction

        # Update velocity (F = ma, assume satellite mass = 1 for simplicity)
        self.satellite_velocity += gravitational_force * self.time_increment

    def step(self, action):
        # Apply action to adjust velocity (scaled for more realistic impact)
        self.satellite_velocity += action * 1.0  # Scale action to have a significant but realistic effect

        # Apply gravitational force
        self._apply_gravitational_force()

        # Update satellite's position based on velocity and time increment
        self.satellite_position += self.satellite_velocity * self.time_increment

        # Update the elapsed time
        self.elapsed_time += self.time_increment

        # Initialize flags
        done = False
        collision_occurred = False

        # Calculate reward
        reward = -np.linalg.norm(self.satellite_velocity - self.initial_orbit.v.to(u.m / u.s).value)  # Penalize deviation from stable orbit

        # Check for collisions with debris
        for debris in self.debris_positions:
            distance = np.linalg.norm(self.satellite_position - debris)
            if distance < 10e3:  # Collision threshold (10 km)
                reward -= 1000  # Large penalty for collision
                done = True
                collision_occurred = True
                break  # Exit loop if collision occurs

        # Penalty for getting too close to Earth
        distance_from_earth_center = np.linalg.norm(self.satellite_position)
        if distance_from_earth_center < EARTH_RADIUS + 100e3:  # 100 km buffer above Earth's surface
            reward -= 500  # Penalty for entering atmosphere buffer
            done = True

        # Optionally, set 'done' to True after a full orbit
        if self.elapsed_time >= self.orbital_period:
            done = True

        reward -= 0.1  # Small time penalty to incentivize efficiency

        return self._get_obs(), reward, done, False, {'collision_occurred': collision_occurred}  # Add truncated=False

    def render(self, mode='human'):
        print(f"Satellite position: {self.satellite_position}")

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

# Simplified plot function without model trajectory
def plot_orbits_and_collisions_plotly(active_positions, debris_positions, model_trajectories=None, use_dynamic_scaling=True, scaling_factor=5000, export_animation=False, export_path_html="animation.html", export_path_gif="animation.gif"):
    fig = go.Figure()

    # Initialize traces for Earth
    earth_model = create_earth_model()
    fig.add_trace(earth_model)

    # Filter debris trajectories that are too far
    filtered_debris_positions = []
    for debris in debris_positions:
        if debris:  # Check if debris is not None
            avg_distance = average_distance_from_earth(debris)
            if avg_distance <= MAX_DISTANCE:
                filtered_debris_positions.append(debris)

    # Initialize trajectory lines
    satellite_lines = []
    debris_lines = []
    for i in range(len(active_positions)):
        line = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='lines',
            line=dict(color='cyan', width=2),
            name=f'Satellite {i+1} Trajectory'
        )
        satellite_lines.append(line)
        fig.add_trace(line)
    
    for j in range(len(filtered_debris_positions)):
        line = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='lines',
            line=dict(color='yellow', width=1),
            name=f'Debris {j+1} Trajectory'
        )
        debris_lines.append(line)
        fig.add_trace(line)

    # Create frames for animation
    frames = []
    num_frames = 500
    max_allowed_frames = 50
    frame_step = max(1, num_frames // max_allowed_frames)
    
    # Initialize list to store images for GIF
    images = []
    if export_animation:
        import os
        from PIL import Image
        os.makedirs("frames", exist_ok=True)

    # Pre-calculate complete trajectory paths
    trajectory_data = []
    
    # Add complete satellite trajectories (static paths)
    for i, positions in enumerate(active_positions):
        trajectory_data.append(go.Scatter3d(
            x=[pos[0]/1000 for pos in positions],
            y=[pos[1]/1000 for pos in positions],
            z=[pos[2]/1000 for pos in positions],
            mode='lines',
            line=dict(color='rgba(0,0,255,0.5)', width=1),  # Use rgba color for transparency
            name=f'Satellite {i+1} Path',
            showlegend=True
        ))

    # Add complete debris trajectories (static paths)
    for j, debris in enumerate(filtered_debris_positions):
        trajectory_data.append(go.Scatter3d(
            x=[pos[0]/1000 for pos in debris],
            y=[pos[1]/1000 for pos in debris],
            z=[pos[2]/1000 for pos in debris],
            mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=1),  # Use rgba color for transparency
            name=f'Debris {j+1} Path',
            showlegend=True
        ))

    # Add static paths to figure
    for trace in trajectory_data:
        fig.add_trace(trace)

    # Add model trajectories to static paths
    colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'brown']
    if model_trajectories:
        for idx, model_trajectory in enumerate(model_trajectories):
            color = colors[idx % len(colors)]
            trajectory_data.append(go.Scatter3d(
                x=[pos[0]/1000 for pos in model_trajectory],
                y=[pos[1]/1000 for pos in model_trajectory],
                z=[pos[2]/1000 for pos in model_trajectory],
                mode='lines',
                line=dict(color=f'rgba(0,255,0,0.5)', width=2),
                name=f'Model Satellite {idx+1} Path',
                showlegend=True
            ))

    for frame in tqdm(range(0, num_frames, frame_step), desc="Creating animation frames"):
        frame_data = []
        # Add moving satellite markers
        for i, positions in enumerate(active_positions):
            idx = int((frame / num_frames) * (len(positions) - 1))
            frame_data.append(go.Scatter3d(
                x=[positions[idx][0]/1000],
                y=[positions[idx][1]/1000],
                z=[positions[idx][2]/1000],
                mode='markers',
                marker=dict(size=6, color='cyan', symbol='circle'),
                name=f'Satellite {i+1}',
                showlegend=False
            ))

        # Add moving debris markers
        for j, debris in enumerate(filtered_debris_positions):
            idx = int((frame / num_frames) * (len(debris) - 1))
            frame_data.append(go.Scatter3d(
                x=[debris[idx][0]/1000],
                y=[debris[idx][1]/1000],
                z=[debris[idx][2]/1000],
                mode='markers',
                marker=dict(size=4, color='yellow', symbol='circle'),
                name=f'Debris {j+1}',
                showlegend=False
            ))

        # Add moving model satellite markers
        if model_trajectories:
            for idx, trajectory in enumerate(model_trajectories):
                model_idx = int((frame / num_frames) * (len(trajectory) - 1))
                frame_data.append(go.Scatter3d(
                    x=[trajectory[model_idx][0]/1000],
                    y=[trajectory[model_idx][1]/1000],
                    z=[trajectory[model_idx][2]/1000],
                    mode='markers',
                    marker=dict(size=8, color='lime', symbol='diamond'),
                    name=f'Model Satellite {idx+1}',
                    showlegend=False
                ))

        # Export frame as image if animation export is enabled
        if export_animation:
            fig_temp = go.Figure(data=trajectory_data + frame_data)
            fig_temp.update_layout(
                scene=dict(
                    aspectmode="data",
                    bgcolor="black"
                ),
                title='',        # Remove the title in exported frames
                showlegend=False # Hide the legend in exported frames
            )
            temp_image_path = f"frames/frame_{frame}.png"
            fig_temp.write_image(temp_image_path)
            images.append(imageio.imread(temp_image_path))

        frames.append(go.Frame(data=frame_data, name=f'frame{frame}'))

    # Add initial marker positions
    fig.add_traces(frames[0].data)

    # Update layout for animation
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, {"frame": {"duration": 50, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}])]
        )],
        scene=dict(
            aspectmode="data",
            bgcolor="black"
        ),
        title='',        # Remove the title
        showlegend=False # Hide the legend
    )

    if export_animation:
        # Convert numpy arrays to PIL Images before saving
        pil_images = []
        for img_array in images:
            pil_image = Image.fromarray(img_array)
            pil_images.append(pil_image)

        # Save images as GIF using PIL images
        pil_images[0].save(
            export_path_gif,
            save_all=True,
            append_images=pil_images[1:],
            duration=100,
            loop=0
        )
        print(f"Animation exported as GIF to {export_path_gif}")
        
        # Clean up frame images
        import shutil
        shutil.rmtree("frames")
        
        # Save HTML version
        fig.write_html(export_path_html)
        print(f"Animation exported to {export_path_html}")
        
    fig.show()

# Simplified function to read TLE data from local file only
def fetch_tle_data(local_file_path):
    """Load TLE data from local file."""
    try:
        with open(local_file_path, 'r') as file:
            tle_data = file.read().splitlines()
            return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    except Exception as e:
        raise Exception(f"Error reading local TLE file '{local_file_path}': {e}")

# Convert epoch year and days to a timezone-aware datetime object
def convert_epoch_to_datetime(epochyr, epochdays):
    year = int(epochyr)
    if year < 57:  # Handling for years below 57 (assumed to be post-2000, as per SGP4 standard)
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epochdays - 1)  # Make it timezone-aware
    return epoch

# Check if the TLE data is outdated
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
        return None

    positions = []
    is_debris = 'debris' in name.lower()  # Determine if the group is debris
    for t in np.linspace(0, time_range, 1000):  # Increase the number of points for smoothness
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            distance = np.linalg.norm(r)
            if is_debris:
                if distance <= MAX_DISTANCE:  # Apply distance check only for debris
                    positions.append(r)
            else:
                positions.append(r)  # No distance check for active satellites
        else:
            return None  # Skip this satellite if there's an error
    return positions if positions else None  # Return None if no valid positions were found

# Parallel processing for orbit calculations
def calculate_orbits_parallel(tle_groups, time_range):
    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(lambda tle_group: calculate_orbit_positions(tle_group, time_range), tle_groups), total=len(tle_groups)))

# Update tle_urls to include only local file paths
tle_files = [
    'tle_data/Last_30_Days_Launches.tle',
    'tle_data/Active_Satellites.tle',
    'tle_data/Russian_ASAT_Test_Debris_(COSMOS_1408).tle',
    'tle_data/Chinese_ASAT_Test_Debris_(FENGYUN_1C).tle',
    'tle_data/IRIDIUM_33_Debris.tle',
    'tle_data/COSMOS_2251_Debris.tle'
]

# Simplified main execution
if __name__ == '__main__':
    # First load the model to check its observation space
    model = PPO.load("models/satellite_avoidance_model_combined")
    expected_obs_size = model.observation_space.shape[0]  # Should be 307
    max_debris = (expected_obs_size - 3) // 3  # Calculate required max_debris from model's observation space
    
    print(f"Model expects observation size: {expected_obs_size}")
    print(f"Calculated max_debris: {max_debris}")

    # Time range for simulation
    time_range = 86400 * 5

    # Fetch and calculate orbits from local files only
    active_sats_positions = []
    debris_positions = []

    for local_file_path in tle_files:
        tle_groups = fetch_tle_data(local_file_path)
        positions = calculate_orbits_parallel(tle_groups[:100], time_range)
        if 'debris' in local_file_path.lower():
            debris_positions.extend(filter(None, positions))
        else:
            active_sats_positions.extend(filter(None, positions))

    # Pad or truncate debris_positions to match exactly what the model expects
    if len(debris_positions) < max_debris:
        # Pad with copies of existing debris if we don't have enough
        while len(debris_positions) < max_debris:
            debris_positions.append(debris_positions[0])
    else:
        # Truncate if we have too many
        debris_positions = debris_positions[:max_debris]

    # Create multiple environments for different satellites
    satellite_configs = [
        {'distance': 700e3, 'angle': 45},
        {'distance': 10000e3, 'angle': 30},
        {'distance': 60000e3, 'angle': 60},
    ]

    # Initialize environments with the exact number of debris points
    environments = []
    for config in satellite_configs:
        env = SatelliteAvoidanceEnv(
            debris_positions=debris_positions,  # Pass exactly max_debris positions
            max_debris=max_debris,
            satellite_distance=config['distance'],
            init_angle=config['angle']
        )
        environments.append(env)

    print(f"Model observation space: {model.observation_space.shape}")
    print(f"Environment observation space: {environments[0].observation_space.shape}")

    # Verify observation spaces match
    assert model.observation_space.shape == environments[0].observation_space.shape, \
        f"Observation space mismatch: model {model.observation_space.shape} != env {environments[0].observation_space.shape}"

    # Simulate model trajectories
    all_model_trajectories = []
    num_steps = 1000

    for i, env in enumerate(environments):
        print(f"Simulating trajectory for Model Satellite {i+1}...")
        model_trajectory = []
        obs, _ = env.reset()
        
        for _ in range(num_steps):
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            model_trajectory.append((env.satellite_position / 1000).tolist())
            if done:
                break
        
        all_model_trajectories.append(model_trajectory)
        print(f"Completed simulation for Model Satellite {i+1}")

    # Create the visualization with model trajectories
    plot_orbits_and_collisions_plotly(
        active_sats_positions,
        debris_positions,
        model_trajectories=all_model_trajectories,
        use_dynamic_scaling=True,
        scaling_factor=500,
        export_animation=True,
        export_path_html="orbit_animation.html",
        export_path_gif="orbit_animation.gif"
    )
