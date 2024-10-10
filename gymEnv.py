import gymnasium as gym  # Updated import for Gymnasium
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces  # Import from Gymnasium instead of Gym
import requests
from sgp4.api import Satrec
from sgp4.api import jday
from datetime import datetime, timedelta, timezone


# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions):
        super(SatelliteAvoidanceEnv, self).__init__()
        
        # Action: changes in velocity in the x, y, z directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # Velocity changes
        
        # Observation: satellite position and debris positions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + len(debris_positions) * 3,), dtype=np.float32)
        
        # Initial satellite position as float64 to match velocity data type
        self.satellite_position = np.array([7000.0, 0.0, 0.0], dtype=np.float64)  # Example starting position
        self.satellite_velocity = np.zeros(3, dtype=np.float64)
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state and handle random seeding
        super().reset(seed=seed)
        self.satellite_position = np.array([7000.0, 0.0, 0.0], dtype=np.float64)  # Reset to initial satellite position
        self.satellite_velocity = np.zeros(3, dtype=np.float64)
        return self._get_obs(), {}

    def _get_obs(self):
        # Return current satellite position and debris positions
        return np.concatenate([self.satellite_position] + self.debris_positions)

    def step(self, action):
        # Update the satellite's velocity based on action
        self.satellite_velocity += action
        self.satellite_position += self.satellite_velocity  # Update position with the new velocity

        # Reward and done flag
        done = False
        reward = -np.linalg.norm(self.satellite_velocity)  # Penalize higher velocities to encourage fuel efficiency

        # Check for collisions
        for debris in self.debris_positions:
            distance = np.linalg.norm(self.satellite_position - debris)
            if distance < 10:  # Collision threshold
                reward -= 100  # Large penalty for collisions
                done = True

        reward -= 0.1  # Time penalty for efficiency

        # Return observation, reward, done flag, truncated flag, and additional info
        return self._get_obs(), reward, done, False, {}  # False indicates no truncation

    def render(self, mode='human'):
        # Simple render function to print satellite position
        print(f"Satellite position: {self.satellite_position}")


# Function to fetch TLE data from a URL
def fetch_tle_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        tle_data = response.text.splitlines()
        return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    else:
        raise Exception(f"Error fetching TLE data: {response.status_code}")

# Convert epoch year and days to a timezone-aware datetime object
def convert_epoch_to_datetime(epochyr, epochdays):
    year = int(epochyr)
    if year < 57:  # Handling for years below 57 (assumed to be post-2000, as per SGP4 standard)
        year += 2000
    else:
        year += 1900
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epochdays - 1)
    return epoch

# Check if the TLE data is outdated (e.g., older than 30 days)
def tle_is_outdated(epochyr, epochdays):
    tle_datetime = convert_epoch_to_datetime(epochyr, epochdays)
    days_old = (datetime.now(timezone.utc) - tle_datetime).days
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

# Example debris positions (replace with real data from TLE)
def get_debris_positions(tle_urls, time_range):
    debris_positions = []

    for name, url in tle_urls.items():
        tle_groups = fetch_tle_data(url)  # Fetch TLE data for this group
        for tle_group in tle_groups[:10]:  # Limit to 10 objects per group for faster processing
            positions = calculate_orbit_positions(tle_group, time_range)
            if positions:
                debris_positions.append(positions[-1])  # Use final position as debris position

    return debris_positions

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

# Get debris positions from TLE data
debris_positions = get_debris_positions(tle_urls, time_range)

# Create environment and train RL model
env = SatelliteAvoidanceEnv(debris_positions)

# Train the model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)  # Training the model with 100k steps

# Save the trained model for future use
model.save("satellite_avoidance_model")
