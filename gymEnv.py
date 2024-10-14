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
import matplotlib.animation as animation

# Gravitational constant and central mass (e.g., Earth)
G = 6.67430e-11  # Gravitational constant in m^3 kg^−1 s^−2
M = 5.972e24  # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters

class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100):
        super(SatelliteAvoidanceEnv, self).__init__()

        self.max_debris = max_debris

        # Action: changes in velocity in the x, y, z directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: satellite position + max number of debris (zero-padded if fewer)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + self.max_debris * 3,), dtype=np.float32)

        # Initial satellite position and velocity (circular orbit approximation)
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

        print(f"Reset: Satellite Position = {self.satellite_position}, Debris Count = {num_debris}")
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
        # Apply action to adjust velocity
        self.satellite_velocity += action

        # Apply gravitational force (adjusts velocity based on distance to Earth)
        self._apply_gravitational_force()

        # Update satellite's position based on velocity
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
        truncated = False
        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"Satellite position: {self.satellite_position}")

# Fetch TLE data from URL, fallback to local file if 403
def fetch_tle_data(url, local_file_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
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

# Function to animate the model's training (optional)
def animate_training(satellite_positions_history, debris_positions, save_animation=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    debris_x = [pos[0] for pos in debris_positions]
    debris_y = [pos[1] for pos in debris_positions]
    debris_z = [pos[2] for pos in debris_positions]
    ax.scatter(debris_x, debris_y, debris_z, color='red', label='Debris', s=50)

    def update(num, satellite_positions_history, line):
        ax.clear()
        ax.scatter(debris_x, debris_y, debris_z, color='red', label='Debris', s=50)
        satellite_positions = satellite_positions_history[:num]
        sat_x = [pos[0] for pos in satellite_positions]
        sat_y = [pos[1] for pos in satellite_positions]
        sat_z = [pos[2] for pos in satellite_positions]
        line.set_data(sat_x, sat_y)
        line.set_3d_properties(sat_z)
        return line,

    line, = ax.plot([], [], [], color='blue', lw=2, label='Satellite')
    ani = animation.FuncAnimation(fig, update, frames=len(satellite_positions_history),
                                  fargs=(satellite_positions_history, line), interval=50, blit=False)

    if save_animation:
        print("Saving animation to satellite_training.mp4")
        ani.save('satellite_training.mp4', writer='ffmpeg')

    plt.show()

# Fetch TLE data
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

# Parallel environment for training
def make_env():
    return SatelliteAvoidanceEnv(debris_positions)

if __name__ == '__main__':
    env = SubprocVecEnv([make_env for _ in range(4)])  # Run four environments in parallel

    # Train PPO model
    model = PPO('MlpPolicy', env, verbose=1, device='cuda')  # Ensure it uses the GPU
    model.learn(total_timesteps=100000)  # Train for 100k steps

    # Save the trained model
    model.save("satellite_avoidance_model")

    # Test and retrieve the satellite's trajectory from one of the environments
    env = SatelliteAvoidanceEnv(debris_positions)  # Create a new instance of the environment to test the trained model
    obs, _ = env.reset()  # Reset the environment

    satellite_positions_history = []  # To store the trajectory of the satellite

    # Option to save animation and data
    save_animation = False  # Set to True if you want to save animation (slows down training)

    # Simulate the model for 1000 steps after training to see its performance
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        satellite_positions_history.append(env.satellite_position.tolist())  # Store the satellite's positions

        if done or truncated:  # Either done or truncated ends the episode
            obs, _ = env.reset()

    # After training, visualize the results
    if save_animation:
        animate_training(satellite_positions_history, env.debris_positions, save_animation=save_animation)
    else:
        print("Skipping animation to avoid slowing down the process.")
