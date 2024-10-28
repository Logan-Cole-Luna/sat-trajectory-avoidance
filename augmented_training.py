import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv  # For parallel environments
from gymnasium import spaces
import requests
from sgp4.api import Satrec
from sgp4.api import jday
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os
import platform
import psutil
import torch
import time

# Gravitational constant and central mass (e.g., Earth)
G = 6.67430e-11  # Gravitational constant in m^3 kg^−1 s^−2
M = 5.972e24  # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters
MAX_VELOCITY = 10e3  # Max velocity cap to prevent explosion of values (e.g., 10 km/s)

class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100):
        super(SatelliteAvoidanceEnv, self).__init__()

        self.max_debris = max_debris

        # Action: changes in velocity in the x, y, z directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: satellite position + max number of debris (zero-padded if fewer)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + self.max_debris * 3,), dtype=np.float32)

        # Initial satellite position and velocity (elliptical orbit approximation)
        self.satellite_position = np.array([7000e3, 0.0, 0.0], dtype=np.float64)  # 7000 km from Earth center
        self.satellite_velocity = np.array([0.0, 7.12e3, 0.0], dtype=np.float64)  # Velocity for low Earth orbit
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

    def _get_obs(self):
        debris_flat = np.concatenate(self.debris_positions)
        # Pad debris positions if fewer than max_debris
        if len(self.debris_positions) < self.max_debris:
            debris_flat = np.pad(debris_flat, (0, (self.max_debris - len(self.debris_positions)) * 3), mode='constant')
        return np.concatenate([self.satellite_position, debris_flat])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.satellite_position = np.array([7000e3, 0.0, 0.0], dtype=np.float64)  # Reset to starting position
        self.satellite_velocity = np.array([0.0, 7.12e3, 0.0], dtype=np.float64)  # Reset to starting velocity

        # Dynamically set a new number of debris
        num_debris = np.random.randint(1, self.max_debris + 1)  # Random number of debris
        self.debris_positions = [np.random.randn(3) * 10000 for _ in range(num_debris)]

        return self._get_obs(), {}

    def _apply_gravitational_force(self):
        # Calculate the distance from the Earth's center
        r = np.linalg.norm(self.satellite_position)

        # Newton's law of universal gravitation
        force_magnitude = G * M / r**2
        force_direction = -self.satellite_position / r  # Direction towards Earth center
        gravitational_force = force_magnitude * force_direction

        # Update velocity (F = ma, assume satellite mass = 1 for simplicity)
        self.satellite_velocity += gravitational_force

    def step(self, action):
        # Normalize and apply action to adjust velocity
        action = np.clip(action, -1.0, 1.0)  # Ensure action is bounded between -1 and 1
        self.satellite_velocity += action * 0.1  # Scale the action to avoid large changes in velocity

        # Apply velocity cap to avoid unrealistic speeds
        velocity_magnitude = np.linalg.norm(self.satellite_velocity)
        if velocity_magnitude > MAX_VELOCITY:
            self.satellite_velocity = (self.satellite_velocity / velocity_magnitude) * MAX_VELOCITY

        # Apply gravitational force (adjusts velocity based on distance to Earth)
        self._apply_gravitational_force()

        # Update satellite's position based on velocity
        self.satellite_position += self.satellite_velocity

        # Calculate the distance from the center of the Earth
        distance_from_earth_center = np.linalg.norm(self.satellite_position)

        # Initialize reward and done flag
        reward = -np.linalg.norm(self.satellite_velocity)  # Penalize for high velocity
        done = False

        # Enforce elliptical orbit via reward (penalty for deviation from elliptical path)
        apogee_altitude = 9000e3  # Example elliptical orbit apogee at 9000 km altitude
        perigee_altitude = 7000e3  # Example perigee at 7000 km altitude
        apogee_distance = EARTH_RADIUS + apogee_altitude
        perigee_distance = EARTH_RADIUS + perigee_altitude

        # Penalize for deviation from elliptical orbit bounds
        if distance_from_earth_center > apogee_distance or distance_from_earth_center < perigee_distance:
            reward -= 200  # Penalty for orbit deviation

        # Earth radius buffer for entering atmosphere (100 km buffer above Earth's surface)
        buffer_altitude = 100e3  # 100 km above Earth's surface

        # Penalty for crashing into the Earth or entering the atmosphere buffer
        if distance_from_earth_center < EARTH_RADIUS + buffer_altitude:
            if distance_from_earth_center < EARTH_RADIUS:
                reward -= 1000  # Large penalty for crashing into Earth
            else:
                reward -= 500  # Penalty for entering atmosphere buffer
            done = True

        # Check for collisions with debris
        for debris in self.debris_positions:
            distance_to_debris = np.linalg.norm(self.satellite_position - debris)
            if distance_to_debris < 10:  # Collision threshold
                reward -= 100  # Large penalty for collisions
                done = True

        reward -= 0.1  # Small time penalty to incentivize efficiency
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"Satellite position: {self.satellite_position}")

# Fetch TLE data from URL, fallback to local file if 403
def fetch_tle_data(url, local_file_path, cache_duration=7200):
    # Check cache
    if os.path.exists(local_file_path):
        file_age = time.time() - os.path.getmtime(local_file_path)
        if file_age < cache_duration:
            print(f"Using cached TLE data from '{local_file_path}'.")
            return read_local_tle_file(local_file_path)
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with open(local_file_path, 'w') as file:
                file.write(response.text)
            print(f"Fetched and cached TLE data to '{local_file_path}'.")
            tle_data = response.text.splitlines()
            return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 lines (Name, Line1, Line2)
        elif response.status_code == 403:
            print(f"403 error, switching to local file: {local_file_path}")
            return read_local_tle_file(local_file_path)
        else:
            raise Exception(f"Error fetching TLE data: {response.status_code}")
    except Exception as e:
        print(f"Error fetching TLE data: {e}, switching to local file.")
        return read_local_tle_file(local_file_path)

# Function to read TLE data from a local file
def read_local_tle_file(local_file_path):
    try:
        with open(local_file_path, 'r') as file:
            tle_data = file.read().splitlines()
            return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 lines (Name, Line1, Line2)
    except FileNotFoundError:
        print(f"Local TLE file '{local_file_path}' not found. Skipping this entry.")
        return []
    except Exception as e:
        raise Exception(f"Error reading local TLE file '{local_file_path}': {e}")

# Convert epoch year and days to a timezone-aware datetime object
def convert_epoch_to_datetime(epochyr, epochdays):
    year = int(epochyr)
    if year < 57:
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epochdays - 1)
    return epoch

# Check if TLE data is outdated (older than 30 days)
def tle_is_outdated(epochyr, epochdays):
    tle_datetime = convert_epoch_to_datetime(epochyr, epochdays)
    days_old = (datetime.now(timezone.utc) - tle_datetime).days
    return days_old > 30

# Calculate satellite positions using TLE data
def calculate_orbit_positions(tle_group, time_range):
    name, line1, line2 = tle_group
    satellite = Satrec.twoline2rv(line1, line2)

    if tle_is_outdated(satellite.epochyr, satellite.epochdays):
        print(f"Skipping outdated satellite: {name}")
        return None

    positions = []
    for t in np.linspace(0, time_range, 100):
        jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)
        e, r, v = satellite.sgp4(jd, fr)
        if e == 0:
            positions.append(r)
        else:
            print(f"Error {e}: skipping {name}")
            return None
    return positions

# Example debris positions (replace with real data from TLE)
def get_debris_positions(tle_urls, time_range):
    debris_positions = []
    for name, url_info in tle_urls.items():
        if isinstance(url_info, tuple):
            url, local_file_path = url_info  # Unpack the tuple
        else:
            url = url_info
            local_file_path = f'tle_data/{name}.tle'  # Provide a default local path if only a URL is provided

        tle_groups = fetch_tle_data(url, local_file_path)
        if tle_groups:  # Ensure we have valid TLE groups before proceeding
            for tle_group in tle_groups[:10]:  # Limit to 10 objects per group
                positions = calculate_orbit_positions(tle_group, time_range)
                if positions:
                    debris_positions.append(positions[-1])  # Append the last position of each object
    return debris_positions

# Function to print system hardware info
def print_system_info():
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {psutil.cpu_count(logical=True)}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    
    # Check GPU information if CUDA is available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("GPU: No CUDA-compatible GPU found")
    print("==========================\n")

# Parallel environment for training
def make_env():
    return SatelliteAvoidanceEnv(debris_positions)

if __name__ == '__main__':
    print_system_info()

    tle_urls = {
        "Last 30 Days' Launches": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle',
        "Active Satellites": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
        "Russian ASAT Test Debris (COSMOS 1408)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle',
        "Chinese ASAT Test Debris (FENGYUN 1C)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle',
        "IRIDIUM 33 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle',
        "COSMOS 2251 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle'
    }

    time_range = 86400 * 5  # 5 days in seconds

    # Get debris positions
    debris_positions = get_debris_positions(tle_urls, time_range)

    env = SubprocVecEnv([make_env for _ in range(4)])  # Run four environments in parallel

    # Train PPO model
    model = PPO('MlpPolicy', env, verbose=1, device='cuda')  # Ensure it uses the GPU
    model.learn(total_timesteps=100000)  # Train for 100k steps

    # Save the trained model
    model.save("satellite_avoidance_model_ext")

    # Test and retrieve the satellite's trajectory from one of the environments
    env = SatelliteAvoidanceEnv(debris_positions)  # Create a new instance of the environment to test the trained model
    obs, _ = env.reset()  # Reset the environment

    satellite_positions_history = []  # To store the trajectory of the satellite

    # Simulate the model for 1000 steps after training to see its performance
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        satellite_positions_history.append(env.satellite_position.tolist())  # Store the satellite's positions

        if done or truncated:  # Either done or truncated ends the episode
            obs, _ = env.reset()

    print("Training and testing complete.")
