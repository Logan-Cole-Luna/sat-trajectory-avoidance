import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
import requests
from sgp4.api import Satrec
from sgp4.api import jday
from datetime import datetime, timedelta, timezone


# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions):
        super(SatelliteAvoidanceEnv, self).__init__()
        
        # Action: changes in velocity in the x, y, z directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: satellite position and debris positions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + len(debris_positions) * 3,), dtype=np.float32)
        
        # Initial satellite position for launch from Cape Canaveral, Florida (example coordinates)
        self.satellite_position = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
        self.satellite_velocity = np.zeros(3, dtype=np.float64)
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.satellite_position = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
        self.satellite_velocity = np.zeros(3, dtype=np.float64)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.satellite_position] + self.debris_positions)

    def step(self, action):
        self.satellite_velocity += action
        self.satellite_position += self.satellite_velocity

        # Reward and done flag
        done = False
        reward = -np.linalg.norm(self.satellite_velocity)

        # Check for collisions with debris
        for debris in self.debris_positions:
            distance = np.linalg.norm(self.satellite_position - debris)
            if distance < 10:  # Collision threshold
                reward -= 100  # Large penalty for collisions
                done = True

        reward -= 0.1  # Time penalty for efficiency

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        print(f"Satellite position: {self.satellite_position}")


# Function to fetch TLE data from a URL with a custom User-Agent
def fetch_tle_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        tle_data = response.text.splitlines()
        return [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]  # Group into 3 lines (Name, Line1, Line2)
    else:
        raise Exception(f"Error fetching TLE data: {response.status_code}")

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
    for name, url in tle_urls.items():
        tle_groups = fetch_tle_data(url)
        for tle_group in tle_groups[:10]:  # Limit to 10 objects per group
            positions = calculate_orbit_positions(tle_group, time_range)
            if positions:
                debris_positions.append(positions[-1])
    return debris_positions

# Fetch TLE data
tle_urls = {
    "Last 30 Days' Launches": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle',
    "Active Satellites": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
    "Russian ASAT Test Debris (COSMOS 1408)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle',
    "Chinese ASAT Test Debris (FENGYUN 1C)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle',
    "IRIDIUM 33 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle',
    "COSMOS 2251 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle'
}

time_range = 86400 * 5  # 5 days

# Get debris positions
debris_positions = get_debris_positions(tle_urls, time_range)

# Create environment and train PPO model
env = SatelliteAvoidanceEnv(debris_positions)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)  # Train for 100k steps

# Save the trained model
model.save("satellite_avoidance_model")