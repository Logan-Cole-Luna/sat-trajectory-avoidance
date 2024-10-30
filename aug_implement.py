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

# Constants for orbit and gravitational force
G = 6.67430e-11  # Gravitational constant in m^3 kg^−1 s^−2
M = 5.972e24  # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters

# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100, satellite_distance=400e3):  # Default to 400 km altitude
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

        # Set initial orbit with configurable altitude for more realistic orbital mechanics
        self.initial_orbit = Orbit.circular(Earth, alt=satellite_distance * u.m)
        self.satellite_position = self.initial_orbit.r.to(u.m).value  # Satellite position in meters
        self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value  # Satellite velocity in meters per second

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset to the initial circular orbit
        self.initial_orbit = Orbit.circular(Earth, alt=400 * u.km)
        self.satellite_position = self.initial_orbit.r.to(u.m).value  # Reset to 400 km altitude
        self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value  # Velocity for circular orbit

        # Dynamically set a new number of debris
        num_debris = np.random.randint(1, self.max_debris + 1)
        self.debris_positions = [np.random.randn(3) * 10000 for _ in range(num_debris)]

        print(f"[DEBUG] Reset satellite position: {self.satellite_position}", end="\r")
        print(f"[DEBUG] Reset satellite velocity: {self.satellite_velocity}", end="\r")

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
        self.satellite_velocity += gravitational_force * 10  # Simulate for 10 seconds

        print(f"[DEBUG] Gravitational force applied: {gravitational_force}", end="\r")
        print(f"[DEBUG] Updated satellite velocity after gravity: {self.satellite_velocity}", end="\r")
        print(f"[DEBUG] Distance from Earth's center: {r}", end="\r")
        print(f"[DEBUG] Gravitational force magnitude: {force_magnitude}", end="\r")

    def step(self, action):
        # Apply action to adjust velocity (scaled for more realistic impact)
        self.satellite_velocity += action * 1.0  # Scale action to have a significant but realistic effect

        print(f"[DEBUG] Action applied to velocity: {action}", end="\r")
        print(f"[DEBUG] Updated satellite velocity after action: {self.satellite_velocity}", end="\r")

        # Apply gravitational force (adjusts velocity based on distance to Earth)
        self._apply_gravitational_force()

        # Update satellite's position based on velocity
        self.satellite_position += self.satellite_velocity * 10  # Simulate for 10 seconds

        print(f"[DEBUG] Updated satellite position: {self.satellite_position}", end="\r")

        # Calculate reward
        reward = -np.linalg.norm(self.satellite_velocity - self.initial_orbit.v.to(u.m / u.s).value)  # Penalize deviation from stable orbit
        done = False

        # Check for collisions with debris
        for debris in self.debris_positions:
            distance = np.linalg.norm(self.satellite_position - debris)
            if distance < 10e3:  # Collision threshold (10 km)
                reward -= 1000  # Large penalty for collision
                done = True
                print(f"[DEBUG] Collision detected with debris at distance: {distance}", end="\r")

        # Penalty for getting too close to Earth
        distance_from_earth_center = np.linalg.norm(self.satellite_position)
        if distance_from_earth_center < EARTH_RADIUS + 100e3:  # 100 km buffer above Earth's surface
            reward -= 500  # Penalty for entering atmosphere buffer
            done = True
            print(f"[DEBUG] Satellite too close to Earth. Distance: {distance_from_earth_center}", end="\r")

        reward -= 0.1  # Small time penalty to incentivize efficiency
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Satellite position: {self.satellite_position}", end="\r")

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
    for t in np.linspace(0, time_range, 1000):  # Increase the number of points for smoothness
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
    model = PPO.load("satellite_avoidance_model_ext")

    # Test the model and collect positions for plotting
    model_trajectory = []
    obs, _ = env.reset()  # Here, no info is expected, so only capture obs
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
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

# Now create the interactive 3D plot using Plotly
plot_orbits_and_collisions_plotly(
    active_sats_positions,
    debris_positions,
    model_trajectory=model_trajectory_km,
    use_dynamic_scaling=use_dynamic_scaling,  # Control whether to dynamically scale or plot full paths
    scaling_factor=500  # Adjust this value to fine-tune dynamic scaling
)

