
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
def plot_orbits_and_collisions_plotly(active_positions, debris_positions, model_trajectories, use_dynamic_scaling=True, scaling_factor=5000):
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
            name=f'Active Satellite Orbit'
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

    # Plot model satellite paths if available
    if model_trajectories:
        for model_trajectory in model_trajectories:
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
        title='3D Orbits with Earth, Debris, and Model Satellites',
        showlegend=True
    )

    fig.show()

# Function to fetch TLE data from a URL or fallback to a local file
def fetch_tle_data(url, local_file_path):
    # Use only local file
    with open(local_file_path, 'r') as file:
        tle_data = file.read().splitlines()
        return [tle_data[i:i + 3] for i in range(0, len(tle_data), 3)]

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
MAX_DISTANCE = 2e5  # Maximum allowed distance from Earth in meters (300,000m)

def calculate_orbit_positions(tle_group, time_range):
    name, line1, line2 = tle_group
    satellite = Satrec.twoline2rv(line1, line2)

    # Check if the TLE data is outdated
    if tle_is_outdated(satellite.epochyr, satellite.epochdays):
        return None

    positions = []
    for t in np.linspace(0, time_range, 1000):        
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)  # Adjust based on time range