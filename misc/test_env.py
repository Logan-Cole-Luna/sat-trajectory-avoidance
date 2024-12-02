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
import imageio  # Add imageio for creating MP4 videos

# Ensure that imageio's ffmpeg backend is installed for MP4 export:
# Run the following command in your terminal:
# pip install imageio[ffmpeg]

# Constants for orbit and gravitational force
G = 6.67430e-11  # Gravitational constant in m^3 kg^−1 s^−2
M = 5.972e24     # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters

# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100, satellite_distance=4000e3, init_angle=0, collision_course=False):
        super(SatelliteAvoidanceEnv, self).__init__()

        # Action: changes in velocity in x, y, z directions
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(3,), dtype=np.float32)

        # Set maximum number of debris objects (debris_positions can be variable)
        self.max_debris = max_debris
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

        # Observation space: (3 for satellite position + 3 * max_debris for debris)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + self.max_debris * 3,), dtype=np.float32
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

        # Return satellite position concatenated with debris positions
        return np.concatenate([self.satellite_position, debris_flat])

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

        return self._get_obs(), reward, done, {'collision_occurred': collision_occurred}

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

# Function to plot orbits and check for collisions with Plotly
def plot_orbits_and_collisions_plotly(active_positions, debris_positions, model_trajectories, use_dynamic_scaling=True, scaling_factor=5000, export_animation=False, export_path_html="animation.html", export_path_mp4="animation.mp4"):
    fig = go.Figure()

    # Initialize traces for Earth
    earth_model = create_earth_model()
    fig.add_trace(earth_model)

    # Initialize trajectory lines
    satellite_lines = []
    debris_lines = []
    model_traj_lines = []  # Initialize model trajectory lines
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
    
    for j in range(len(debris_positions)):
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
    
    # Define a list of bright colors for model trajectories
    bright_colors = ['magenta', 'lime', 'orange', 'cyan', 'yellow', 'purple', 'green', 'pink', 'gold', 'turquoise']
    


    # Create frames for animation
    frames = []
    num_frames = max(len(pos) for pos in active_positions + debris_positions + model_trajectories)
    max_allowed_frames = 50  # Reduce maximum number of frames
    frame_step = max(1, num_frames // max_allowed_frames)  # Calculate step to sample frames
    
    # Initialize list to store images for MP4
    images = []
    if export_animation:
        import os
        from PIL import Image
        os.makedirs("frames", exist_ok=True)  # Create a directory to store frame images

    for frame in range(0, num_frames, frame_step):
        # Print progress every 50 frames
        if frame % 50 == 0:
            print(f"Frame loaded: {frame}")
        
        frame_data = []
        # Update trajectory lines
        for i, positions in enumerate(active_positions):
            traj = go.Scatter3d(
                x=[pos[0]/1000 for pos in positions[:frame+1]],
                y=[pos[1]/1000 for pos in positions[:frame+1]],
                z=[pos[2]/1000 for pos in positions[:frame+1]],
                mode='lines',
                line=dict(color='cyan', width=2),
                showlegend=False
            )
            frame_data.append(traj)
        for j, debris in enumerate(debris_positions):
            traj = go.Scatter3d(
                x=[pos[0]/1000 for pos in debris[:frame+1]],
                y=[pos[1]/1000 for pos in debris[:frame+1]],
                z=[pos[2]/1000 for pos in debris[:frame+1]],
                mode='lines',
                line=dict(color='yellow', width=1),
                showlegend=False
            )
            frame_data.append(traj)
        # Update satellite and debris positions
        for i, positions in enumerate(active_positions):
            if frame < len(positions):
                frame_data.append(go.Scatter3d(
                    x=[positions[frame][0]/1000],
                    y=[positions[frame][1]/1000],
                    z=[positions[frame][2]/1000],
                    mode='markers',
                    marker=dict(size=6, color='cyan', symbol='circle'),
                    name=f'Satellite {i+1}'
                ))
            else:
                frame_data.append(go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode='markers',
                    marker=dict(size=6, color='cyan', symbol='circle'),
                    name=f'Satellite {i+1}'
                ))
        for j, debris in enumerate(debris_positions):
            if frame < len(debris):
                frame_data.append(go.Scatter3d(
                    x=[debris[frame][0]/1000],
                    y=[debris[frame][1]/1000],
                    z=[debris[frame][2]/1000],
                    mode='markers',
                    marker=dict(size=4, color='yellow', symbol='circle'),
                    name=f'Debris {j+1}'
                ))
            else:
                frame_data.append(go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode='markers',
                    marker=dict(size=4, color='yellow', symbol='circle'),
                    name=f'Debris {j+1}'
                ))

        # Export frame as image for MP4
        if export_animation:
            fig_temp = go.Figure(data=frame_data)
            fig_temp.update_layout(
                scene=dict(
                    aspectmode="data",
                    bgcolor="black"
                ),
                title='Animated 3D Orbits with Earth, Debris, and Model Satellites',
                showlegend=False
            )
            temp_image_path = f"frames/frame_{frame}.png"
            fig_temp.write_image(temp_image_path)
            images.append(imageio.imread(temp_image_path))

    fig.frames = frames

    # Update layout for animation with dynamic axis ranges
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
        title='Animated 3D Orbits with Earth, Debris, and Model Satellites',
        showlegend=True
    )

    # Export the animation if requested
    if export_animation:
        # Create MP4 video from saved frames
        imageio.mimsave(export_path_mp4, images, fps=10)
        print(f"Animation exported as MP4 to {export_path_mp4}")
        # Optionally, clean up frame images
        import shutil
        shutil.rmtree("frames")
    
    # Export the animation as HTML if requested
    if export_animation:
        fig.write_html(export_path_html)
        print(f"Animation exported to {export_path_html}")
        
    fig.show()

# Function to fetch TLE data from a URL or fallback to a local file
def fetch_tle_data(url, local_file_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            tle_data = response.text.splitlines()
            return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
        else:
            print(f"Error fetching TLE data: {response.status_code}, switching to local file.")
    except Exception as e:
        print(f"Error fetching TLE data from URL: {e}, switching to local file.")

    # Fallback to local file if fetching fails
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
    for t in np.linspace(0, time_range, 1000):  # Increase the number of points for smoothness
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range
        e, r, v = satellite.sgp4(jd, fr)  # Get position (r) and velocity (v)
        if e == 0:  # Only add positions if no error occurred
            positions.append(r)
        else:
            return None  # Skip this satellite if there's an error
    return positions

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

# Use the trained PPO model
use_model = True

# Create multiple environments for different satellites with unique heights and rotation angles
satellite_configs = [
    {'distance': 70, 'angle': 45},
    {'distance': 10, 'angle': 30},
    {'distance': 600, 'angle': 60},
]

environments = []
for config in satellite_configs:
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    env = SatelliteAvoidanceEnv(
        debris_positions_sample, 
        satellite_distance=config['distance'], 
        init_angle=config['angle'], 
        collision_course=False)
    environments.append(env)

# Load the saved PPO model
model = PPO.load("models/satellite_avoidance_model_ext_best")

# Initialize lists to store data for alternative visualizations and metrics
all_model_trajectories = []
all_satellite_positions = []
all_satellite_velocities = []
all_distances_to_debris = []
all_rewards = []
all_cumulative_rewards_over_time = []
performance_metrics = []  # List to store performance metrics for each satellite

num_steps = 1000  # Number of steps to simulate

# Test the model for each environment and collect positions for plotting
for i, env in enumerate(environments):
    model_trajectory = []
    satellite_positions = []
    satellite_velocities = []
    distances_to_nearest_debris = []
    rewards = []
    cumulative_rewards_over_time = []
    cumulative_reward = 0
    actions = []
    collision_occurred = False
    min_distance_to_debris = float('inf')
    obs, _ = env.reset()
    for _ in range(num_steps):
        action, _states = model.predict(obs)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        if 'collision_occurred' in info and info['collision_occurred']:
            collision_occurred = True
        model_trajectory.append(env.satellite_position.tolist())
        satellite_positions.append(env.satellite_position.tolist())
        satellite_velocities.append(env.satellite_velocity.tolist())
        # Calculate distance to nearest debris
        distances = [np.linalg.norm(env.satellite_position - debris) for debris in env.debris_positions]
        min_distance = min(distances)
        distances_to_nearest_debris.append(min_distance)
        if min_distance < min_distance_to_debris:
            min_distance_to_debris = min_distance
        rewards.append(reward)
        cumulative_reward += reward
        cumulative_rewards_over_time.append(cumulative_reward)
        if done:
            break  # Exit the loop if done

    # Convert model trajectory positions from meters to kilometers
    model_trajectory_km = [(np.array(pos)).tolist() for pos in model_trajectory]
    all_model_trajectories.append(model_trajectory_km)
    # Store other data for plotting
    all_satellite_positions.append(satellite_positions)
    all_satellite_velocities.append(satellite_velocities)
    all_distances_to_debris.append(distances_to_nearest_debris)
    all_rewards.append(rewards)
    all_cumulative_rewards_over_time.append(cumulative_rewards_over_time)

    # Compute performance metrics
    num_steps_taken = len(actions)
    total_delta_v = sum(np.linalg.norm(a) for a in actions)
    # total_delta_v is in m/s

    # Store performance metrics
    metrics = {
        'satellite_id': i+1,
        'num_steps_taken': num_steps_taken,
        'total_delta_v': total_delta_v,
        'cumulative_reward': cumulative_reward,
        'collision_occurred': collision_occurred,
        'min_distance_to_debris': min_distance_to_debris
    }
    performance_metrics.append(metrics)

    # Print out performance metrics
    print(f"Satellite {i+1} Performance Metrics:")
    print(f"  Total steps taken: {num_steps_taken}")
    print(f"  Total delta-v used: {total_delta_v:.4f} m/s")
    print(f"  Total cumulative reward: {cumulative_reward:.4f}")
    print(f"  Collision occurred: {collision_occurred}")
    print(f"  Minimum distance to debris: {min_distance_to_debris:.2f} meters")
    print()

# Use dynamic scaling for plotting
use_dynamic_scaling = True  # Set to True for dynamic orbit plotting, False for full orbits

# Update the model trajectories conversion before plotting
model_trajectories_m = [[(np.array(pos)).tolist() for pos in traj] for traj in all_model_trajectories]

# Create the interactive 3D plot using Plotly for all satellites
plot_orbits_and_collisions_plotly(
    active_sats_positions,
    debris_positions,
    model_trajectories=model_trajectories_m,
    use_dynamic_scaling=use_dynamic_scaling,  # Control whether to dynamically scale or plot full paths
    scaling_factor=500,  # Adjust this value to fine-tune dynamic scaling,
    export_animation=True
)

# Alternative Visualizations
'''
# Plot distance to nearest debris over time for each satellite
for i, distances in enumerate(all_distances_to_debris):
    plt.figure()
    plt.plot(distances)
    plt.xlabel('Time Step')
    plt.ylabel('Distance to Nearest Debris (meters)')
    plt.title(f'Satellite {i+1} Distance to Nearest Debris Over Time')
    plt.grid(True)
    plt.show()

# Plot altitude over time for each satellite
for i, positions in enumerate(all_satellite_positions):
    altitudes = [np.linalg.norm(pos) - EARTH_RADIUS for pos in positions]
    plt.figure()
    plt.plot(altitudes)
    plt.xlabel('Time Step')
    plt.ylabel('Altitude above Earth Surface (meters)')
    plt.title(f'Satellite {i+1} Altitude Over Time')
    plt.grid(True)
    plt.show()

# Plot cumulative reward over time for each satellite
for i, cumulative_rewards in enumerate(all_cumulative_rewards_over_time):
    plt.figure()
    plt.plot(cumulative_rewards)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Satellite {i+1} Cumulative Reward Over Time')
    plt.grid(True)
    plt.show()

# Plot speed over time for each satellite
for i, velocities in enumerate(all_satellite_velocities):
    speeds = [np.linalg.norm(v) for v in velocities]
    plt.figure()
    plt.plot(speeds)
    plt.xlabel('Time Step')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Satellite {i+1} Speed Over Time')
    plt.grid(True)
    plt.show()

# Plot 2D projections (X vs Y) for each satellite
for i, positions in enumerate(all_satellite_positions):
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title(f'Satellite {i+1} Orbit Projection (X vs Y)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Plot 3D orbits for each satellite
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, positions in enumerate(all_satellite_positions):
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]
    ax.plot(x_vals, y_vals, z_vals, label=f'Satellite {i+1}')
ax.set_xlabel('X Position (meters)')
ax.set_ylabel('Y Position (meters)')
ax.set_zlabel('Z Position (meters)')
ax.set_title('3D Orbits of Satellites')
ax.legend()
plt.show()
'''

def plot_orbits_and_collisions_plotly_animated(active_positions, debris_positions, model_trajectories, num_steps=100):
    """Create an animated 3D plot of active satellites, debris, and model predictions."""
    fig = go.Figure()

    # Initialize traces for Earth
    earth_model = create_earth_model()
    fig.add_trace(earth_model)

    # Initialize traces for Moon if needed
    # moon_model = create_moon_model()
    # fig.add_trace(moon_model)

    # Initialize satellite traces
    satellite_traces = []
    for i, positions in enumerate(active_positions):
        trace = go.Scatter3d(
            x=[positions[0][0]], y=[positions[0][1]], z=[positions[0][2]],
            mode='markers',
            marker=dict(size=6, color='cyan'),
            name=f'Active Satellite {i+1}'
        )
        satellite_traces.append(trace)
        fig.add_trace(trace)

    # Initialize debris traces
    debris_traces = []
    for i, debris in enumerate(debris_positions):
        trace = go.Scatter3d(
            x=[debris[0][0]], y=[debris[0][1]], z=[debris[0][2]],
            mode='markers',
            marker=dict(size=4, color='yellow'),
            name=f'Debris {i+1}'
        )
        debris_traces.append(trace)
        fig.add_trace(trace)

    # Initialize model prediction traces
    model_traces = []
    for i, model_trajectory in enumerate(model_trajectories):
        trace = go.Scatter3d(
            x=[model_trajectory[0][0]], y=[model_trajectory[0][1]], z=[model_trajectory[0][2]],
            mode='markers',
            marker=dict(size=6, color='magenta'),
            name=f'Model Satellite {i+1}'
        )
        model_traces.append(trace)
        fig.add_trace(trace)

    frames = []
    for step in range(1, num_steps):
        frame_data = []
        for i, positions in enumerate(active_positions):
            frame_data.append(go.Scatter3d(
                x=[positions[step][0]],
                y=[positions[step][1]],
                z=[positions[step][2]],
                mode='markers',
                marker=dict(size=6, color='cyan'),
                name=f'Active Satellite {i+1}'
            ))
        for i, debris in enumerate(debris_positions):
            frame_data.append(go.Scatter3d(
                x=[debris[step][0]],
                y=[debris[step][1]],
                z=[debris[step][2]],
                mode='markers',
                marker=dict(size=4, color='yellow'),
                name=f'Debris {i+1}'
            ))
        for i, model_trajectory in enumerate(model_traces):
            frame_data.append(go.Scatter3d(
                x=[model_trajectories[i][step][0]],
                y=[model_trajectories[i][step][1]],
                z=[model_trajectories[i][step][2]],
                mode='markers',
                marker=dict(size=6, color='magenta'),
                name=f'Model Satellite {i+1}'
            ))
        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

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
            xaxis=dict(range=[-10000, 10000], backgroundcolor="black"),
            yaxis=dict(range=[-10000, 10000], backgroundcolor="black"),
            zaxis=dict(range=[-10000, 10000], backgroundcolor="black"),
            aspectmode='data'
        ),
        title='Animated 3D Orbits: Active Satellites, Debris, and Model Predictions',
        showlegend=True
    )

    fig.show()

# After collecting all_model_trajectories and other data

# Create the animated plot
plot_orbits_and_collisions_plotly_animated(
    active_positions=all_satellite_positions,
    debris_positions=all_debris_positions,
    model_trajectories=all_model_trajectories,
    num_steps=100  # Adjust based on simulation steps
)