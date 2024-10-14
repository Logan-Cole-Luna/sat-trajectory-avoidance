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

# Custom Satellite Avoidance Environment
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100):
        super(SatelliteAvoidanceEnv, self).__init__()

        # Action: changes in velocity in x, y, z directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Set maximum number of debris objects (debris_positions can be variable)
        self.max_debris = max_debris
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

        # Observation space: (3 for satellite position + 3 * max_debris for debris)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + self.max_debris * 3,), dtype=np.float32
        )

        # Satellite initial position and velocity
        self.satellite_position = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
        self.satellite_velocity = np.zeros(3, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.satellite_position = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
        self.satellite_velocity = np.zeros(3, dtype=np.float64)
        return self._get_obs()

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

    def step(self, action):
        # Update satellite's velocity and position
        self.satellite_velocity += action
        self.satellite_position += self.satellite_velocity

        # Calculate reward
        reward = -np.linalg.norm(self.satellite_velocity)  # Penalize speed
        done = False

        # Check for collisions with debris
        for debris in self.debris_positions:
            distance = np.linalg.norm(self.satellite_position - debris)
            if distance < 10:  # Collision threshold
                reward -= 100  # Penalty for collision
                done = True

        reward -= 0.1  # Time penalty for efficiency
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Satellite position: {self.satellite_position}")


# Function to plot orbits and check for collisions with Plotly
def plot_orbits_and_collisions_plotly(active_positions, debris_positions, model_trajectory, plot_full_path=True, trajectory_length=10):
    fig = go.Figure()

    # Plot smooth trajectories of active satellites (without dots for past trajectory)
    for positions in active_positions:
        if plot_full_path:
            x_vals, y_vals, z_vals = zip(*positions[:-1])  # Full Trajectory excluding the last point
        else:
            x_vals, y_vals, z_vals = zip(*positions[-trajectory_length:])  # Short Trajectory (last N points)

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
        if plot_full_path:
            x_vals_debris, y_vals_debris, z_vals_debris = zip(*positions_debris[:-1])  # Full Trajectory excluding the last point
        else:
            x_vals_debris, y_vals_debris, z_vals_debris = zip(*positions_debris[-trajectory_length:])  # Short Trajectory (last N points)

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

    # Plot the model's satellite path in green with a thicker line and ensure it's rendered last (on top)
    if model_trajectory:
        if plot_full_path:
            model_x_vals, model_y_vals, model_z_vals = zip(*model_trajectory)  # Full model path
        else:
            model_x_vals, model_y_vals, model_z_vals = zip(*model_trajectory[-trajectory_length:])  # Short model path

        # Adding the model's trajectory with a thicker line
        fig.add_trace(go.Scatter3d(
            x=model_x_vals, y=model_y_vals, z=model_z_vals,
            mode='lines+markers',  # Line with markers for better visibility
            line=dict(color='green', width=6),  # Thicker green line for the model's path
            marker=dict(size=4, color='lime'),  # Small markers on the path
            name='Model Satellite Path'
        ))

    # Adjust layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', showgrid=True),  # Show grid
            yaxis=dict(title='Y', showgrid=True),  # Show grid
            zaxis=dict(title='Z', showgrid=True),  # Show grid
            bgcolor='lightgray',  # Optional background color
        ),
        title='3D Orbits with Model Satellite Path and Debris',
        showlegend=True,
    )

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

# Parallel processing for orbit calculations
def calculate_orbits_parallel(tle_groups, time_range):
    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(lambda tle_group: calculate_orbit_positions(tle_group, time_range), tle_groups), total=len(tle_groups)))

# Example usage when fetching TLE data
tle_urls = {
    "Last 30 Days' Launches": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle', 'tle_data/Last_30_Days\'_Launches.tle'),
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
    positions = calculate_orbits_parallel(tle_groups[:100], time_range)  # Limit to 100 objects per group for faster processing
    if 'debris' in name.lower():  # Classify debris vs active satellites
        debris_positions.extend(filter(None, positions))
    else:
        active_sats_positions.extend(filter(None, positions))

# Use the trained PPO model if set to True
use_model = True  # Set this to False to skip using the model

if use_model:
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]  # Example debris positions
    env = SatelliteAvoidanceEnv(debris_positions_sample)  # Create the environment

    # Load the saved PPO model
    model = PPO.load("satellite_avoidance_model")

    # Test the model and collect positions for plotting
    model_trajectory = []
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        model_trajectory.append(env.satellite_position.tolist())  # Collect satellite position
        if done:
            obs = env.reset()
else:
    model_trajectory = None

# Now create the interactive 3D plot using Plotly
# Set `plot_full_path` to False to plot only a short trajectory (e.g., last 10 points), or True to plot full paths
plot_orbits_and_collisions_plotly(
    active_sats_positions,
    debris_positions,
    model_trajectory=model_trajectory,
    trajectory_length=15,  # You can adjust the number of points for short trajectories
    plot_full_path=True   # Set to True for full orbit paths, or False for shorter trajectories
)
