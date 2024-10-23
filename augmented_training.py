import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym import spaces
import requests
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from poliastro.twobody.orbit import Orbit
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from astropy import units as u
from astropy.time import Time
import os
import time
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import solve_ivp

# Constants
EARTH_RADIUS = 6371.0  # km

# Debris class
class Debris:
    def __init__(self, orbit):
        self.orbit = orbit

# Custom Satellite Avoidance Environment
class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_orbits, max_debris=100):
        super(SatelliteAvoidanceEnv, self).__init__()

        self.max_debris = max_debris
        self.training = True  # Flag to indicate training mode
        self.time_step = 60.0  # seconds

        # Action: thrust acceleration in x, y, z directions
        self.max_thrust = 0.0001  # km/s^2, adjust based on propulsion capabilities
        self.action_space = spaces.Box(
            low=-self.max_thrust,
            high=self.max_thrust,
            shape=(3,),
            dtype=np.float32
        )

        # Observation: satellite position and velocity + debris relative positions and velocities
        # Each debris contributes 6 values: relative position (3), relative velocity (3)
        obs_size = 6 + self.max_debris * 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Randomize initial orbit parameters
        self.initial_altitude = 700.0  # km
        self.initial_inclination = 0.0  # degrees

        self.debris_orbits = debris_orbits  # List of debris orbits
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.max_steps = int(86400 / self.time_step)  # Simulate for one day
        self.step_count = 0

        # Satellite physical properties
        self.satellite_area = 10.0  # m^2
        self.satellite_mass = 1000.0  # kg

        # For reward shaping and curriculum learning
        self.training_stage = 1  # Start at stage 1
        self.performance_metric = []
        self.current_reward = 0.0
        self.threshold_stage_1 = -1000.0  # Adjust as appropriate
        self.threshold_stage_2 = -500.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize initial orbit within small bounds
        alt_variation = np.random.uniform(-50, 50)  # +/-50 km variation
        inc_variation = np.random.uniform(-5, 5)    # +/-5 degrees variation
        self.initial_orbit = Orbit.circular(
            Earth,
            alt=(self.initial_altitude + alt_variation) * u.km,
            inc=np.abs(self.initial_inclination + inc_variation) * u.deg,  # Use absolute value to keep inclination positive
            epoch=Time.now()
        )
        self.satellite_orbit = self.initial_orbit
        self.current_time = self.initial_orbit.epoch
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        # Get initial orbital elements
        self.initial_a = self.satellite_orbit.a.to(u.km).value
        self.initial_ecc = self.satellite_orbit.ecc.value

        # Randomly select debris orbits for this episode
        num_debris = min(len(self.debris_orbits), self.max_debris)
        indices = np.random.choice(len(self.debris_orbits), num_debris, replace=False)
        self.selected_debris = [Debris(self.debris_orbits[i]) for i in indices]

        # Get initial positions and velocities
        self.selected_debris_positions = np.array([debris.orbit.r.to(u.km).value for debris in self.selected_debris])
        self.selected_debris_velocities = np.array([debris.orbit.v.to(u.km / u.s).value for debris in self.selected_debris])

        return self._get_obs(), {}  # Return observation and empty info dictionary (required by Gymnasium)

    def _get_obs(self):
        # Get satellite position and velocity in ECI frame
        r_sat = self.satellite_orbit.r.to(u.km).value  # Position in km
        v_sat = self.satellite_orbit.v.to(u.km / u.s).value  # Velocity in km/s

        # Debris relative positions and velocities
        debris_relative_states = []
        for r_debris, v_debris in zip(self.selected_debris_positions, self.selected_debris_velocities):
            relative_position = (r_debris - r_sat) / 10000.0  # Normalize
            relative_velocity = (v_debris - v_sat) / 8.0      # Normalize
            debris_relative_states.extend(relative_position)
            debris_relative_states.extend(relative_velocity)

        # Pad if fewer debris
        max_debris = self.max_debris
        current_debris = len(self.selected_debris)
        if current_debris < max_debris:
            padding = [0.0] * ((max_debris - current_debris) * 6)
            debris_relative_states.extend(padding)

        # Normalize satellite position and velocity
        r_norm = r_sat / 10000.0
        v_norm = v_sat / 8.0

        # Construct observation
        obs = np.concatenate([
            r_norm,
            v_norm,
            np.array(debris_relative_states)
        ])

        return obs.astype(np.float32)

    def step(self, action):
        # Clip action to max thrust
        action = np.clip(action, -self.max_thrust, self.max_thrust)

        # Add action noise during training for robustness
        if self.training:
            action += np.random.normal(0, 0.00001, size=3)

        # Compute acceleration due to thrust
        thrust_acceleration = action * u.km / u.s**2

        # Include perturbations
        j2_acceleration = self.compute_j2_acceleration()
        drag_acceleration = self.compute_drag_acceleration()

        # Total acceleration
        total_acceleration = thrust_acceleration + j2_acceleration + drag_acceleration

        # Propagate orbit with total acceleration
        self.satellite_orbit = self.propagate_orbit(self.satellite_orbit, total_acceleration)

        # Update time
        self.current_time += self.time_step * u.s

        # Get current position and orbital elements
        r = self.satellite_orbit.r.to(u.km).value
        v = self.satellite_orbit.v.to(u.km / u.s).value

        current_a = self.satellite_orbit.a.to(u.km).value
        current_ecc = self.satellite_orbit.ecc.value

        # Reward shaping
        delta_v_magnitude = np.linalg.norm(action) * self.time_step  # Approximate delta-v over time step
        action_change = np.linalg.norm(action - self.previous_action)

        # Deviation from initial orbital elements
        a_diff = abs(current_a - self.initial_a)
        ecc_diff = abs(current_ecc - self.initial_ecc)

        # Reward components
        reward = 0.0

        # Stage-based rewards
        if self.training_stage == 1:
            # Early stage: Focus on maintaining orbit
            reward += self.reward_for_orbit_maintenance()
        elif self.training_stage == 2:
            # Mid stage: Introduce penalties for fuel usage
            reward += self.reward_for_orbit_maintenance()
            reward += self.reward_for_fuel_efficiency(delta_v_magnitude)
        elif self.training_stage == 3:
            # Advanced stage: Full reward function
            reward += self.full_reward_function(delta_v_magnitude, action_change, a_diff, ecc_diff)

        # Check for collisions and other penalties
        collision, collision_penalty = self.check_collision(r, v)
        reward += collision_penalty
        if collision:
            done = True
        else:
            done = False

        # Update previous action
        self.previous_action = action

        # Update debris positions
        self.update_debris_positions()

        # Check if maximum steps reached
        self.step_count += 1
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            # At the end of the episode, penalize the distance from starting point
            final_r = self.satellite_orbit.r.to(u.km).value
            start_r = self.initial_orbit.r.to(u.km).value
            position_diff = np.linalg.norm(final_r - start_r)
            reward -= position_diff * 0.01  # Penalize distance from starting point

        # Update training stage based on performance
        self.current_reward = reward
        self.update_training_stage()

        return self._get_obs(), reward, done, truncated, {}

    def compute_j2_acceleration(self):
        # Constants
        mu = Earth.k.to(u.km**3 / u.s**2).value  # Earth's gravitational parameter
        J2 = 1.08263e-3  # Earth's second zonal harmonic coefficient
        R = Earth.R.to(u.km).value  # Earth's mean radius

        # Satellite position
        r_vec = self.satellite_orbit.r.to(u.km).value
        r = np.linalg.norm(r_vec)
        x, y, z = r_vec

        # Compute J2 acceleration
        factor = 1.5 * J2 * mu * R**2 / r**5
        f = factor * np.array([
            x * (5 * (z**2 / r**2) - 1),
            y * (5 * (z**2 / r**2) - 1),
            z * (5 * (z**2 / r**2) - 3)
        ])
        return f * u.km / u.s**2

    def compute_drag_acceleration(self):
        # Only significant in LEO (e.g., below 1000 km)
        altitude = np.linalg.norm(self.satellite_orbit.r.to(u.km).value) - EARTH_RADIUS
        #print("Alt: ", altitude)  # Debugging output
        
        if (altitude < 1000).any():  # Change this based on the logic you need
            # Constants for drag calculation
            rho = self.get_atmospheric_density()
            Cd = 2.2  # Drag coefficient
            A = self.satellite_area  # Cross-sectional area in m^2
            m = self.satellite_mass  # Mass in kg

            v_rel = self.satellite_orbit.v.to(u.m / u.s).value  # Assume atmosphere co-rotating with Earth
            v_rel_mag = np.linalg.norm(v_rel)

            drag_acc = -0.5 * Cd * A / m * rho * v_rel_mag * v_rel  # Acceleration in m/s^2
            return (drag_acc * u.m / u.s**2).to(u.km / u.s**2)
        else:
            return np.zeros(3) * u.km / u.s**2
        
    def get_atmospheric_density(self):
        # Calculate altitude based on the first value of h
        h = (self.satellite_orbit.r.to(u.km).value[0] - EARTH_RADIUS) * 1000  # Altitude in meters

        # Check if altitude is below 1000 km
        if h < 1000 * 1000:  # Below 1000 km
            rho0 = 1.225  # kg/m^3 at sea level
            scale_height = 8500  # m
            rho = rho0 * np.exp(-h / scale_height)
            return rho
        else:
            return 0.0

    def propagate_orbit(self, orbit, acceleration):
        def equations_of_motion(t, state_vector):
            # Unpack state vector
            x, y, z, vx, vy, vz = state_vector
            r_vec = np.array([x, y, z])
            v_vec = np.array([vx, vy, vz])
            r = np.linalg.norm(r_vec)

            # Two-body acceleration
            mu = Earth.k.to(u.km**3 / u.s**2).value
            gravity_acceleration = -mu * r_vec / r**3

            # Total acceleration
            total_acceleration = gravity_acceleration + acceleration.to(u.km / u.s**2).value

            # Return derivatives
            return [vx, vy, vz, *total_acceleration]

        # Initial state vector
        r0 = orbit.r.to(u.km).value
        v0 = orbit.v.to(u.km / u.s).value
        state_vector = np.concatenate((r0, v0))

        # Time span
        t_span = (0, self.time_step)

        # Integrate equations of motion
        sol = solve_ivp(equations_of_motion, t_span, state_vector, method='RK45', rtol=1e-8)

        # Extract final state
        xf, yf, zf, vxf, vyf, vzf = sol.y[:, -1]
        rf = np.array([xf, yf, zf]) * u.km
        vf = np.array([vxf, vyf, vzf]) * u.km / u.s

        # Create new orbit
        new_orbit = Orbit.from_vectors(Earth, rf, vf, epoch=orbit.epoch + self.time_step * u.s)

        return new_orbit

    def update_debris_positions(self):
        new_positions = []
        new_velocities = []
        for debris in self.selected_debris:
            # Propagate debris orbit
            debris.orbit = debris.orbit.propagate(self.time_step * u.s)
            # Get new position and velocity
            r = debris.orbit.r.to(u.km).value
            v = debris.orbit.v.to(u.km / u.s).value
            new_positions.append(r)
            new_velocities.append(v)
        self.selected_debris_positions = np.array(new_positions)
        self.selected_debris_velocities = np.array(new_velocities)

    def check_collision(self, r_sat, v_sat):
        collision = False
        collision_penalty = 0.0
        for debris_pos, debris_vel in zip(self.selected_debris_positions, self.selected_debris_velocities):
            relative_position = r_sat - debris_pos
            relative_velocity = v_sat - debris_vel
            rel_speed = np.linalg.norm(relative_velocity)
            if rel_speed == 0:
                continue
            miss_distance = np.linalg.norm(np.cross(relative_position, relative_velocity)) / rel_speed
            time_to_close_approach = -np.dot(relative_position, relative_velocity) / (rel_speed**2)
            # Check if time to close approach is within next time step
            if 0 <= time_to_close_approach <= self.time_step:
                if miss_distance < 1.0:  # Collision threshold in km
                    collision = True
                    collision_penalty -= 1000.0  # Large penalty
                    break
        return collision, collision_penalty

    def reward_for_orbit_maintenance(self):
        # Reward the agent for staying close to the initial orbit
        current_a = self.satellite_orbit.a.to(u.km).value
        current_ecc = self.satellite_orbit.ecc.value
        a_diff = abs(current_a - self.initial_a)
        ecc_diff = abs(current_ecc - self.initial_ecc)
        reward = -a_diff * 0.01 - ecc_diff * 100.0
        return reward

    def reward_for_fuel_efficiency(self, delta_v_magnitude):
        # Penalize delta-v usage
        reward = -delta_v_magnitude * 1000.0
        return reward

    def full_reward_function(self, delta_v_magnitude, action_change, a_diff, ecc_diff):
        reward = -delta_v_magnitude * 1000.0
        reward -= action_change * 500.0
        reward -= a_diff * 0.01
        reward -= ecc_diff * 100.0
        return reward

    def update_training_stage(self):
        # Update performance metric
        self.performance_metric.append(self.current_reward)
        # Check if agent's performance meets criteria to advance
        if self.training_stage == 1 and len(self.performance_metric) >= 100:
            if np.mean(self.performance_metric[-100:]) > self.threshold_stage_1:
                self.training_stage = 2
        elif self.training_stage == 2 and len(self.performance_metric) >= 100:
            if np.mean(self.performance_metric[-100:]) > self.threshold_stage_2:
                self.training_stage = 3

    def render(self, mode='human'):
        pass  # Implement visualization if needed

    def set_training(self, training):
        self.training = training

# Function to read TLE data from a local file
def read_local_tle_file(local_file_path):
    try:
        with open(local_file_path, 'r') as file:
            tle_data = file.read().splitlines()
            # Remove any empty lines
            tle_data = [line.strip() for line in tle_data if line.strip()]
            # Ensure the data length is a multiple of 3
            num_groups = len(tle_data) // 3
            tle_data = tle_data[:num_groups * 3]
            tle_groups = [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]
            return [group for group in tle_groups if len(group) == 3]
    except FileNotFoundError:
        print(f"Local TLE file '{local_file_path}' not found.")
        return []
    except Exception as e:
        raise Exception(f"Error reading local TLE file '{local_file_path}': {e}")

# Function to fetch TLE data with caching
def fetch_tle_data_with_cache(url, local_file_path, cache_duration=7200):
    # Check if local file exists and is recent enough
    if os.path.exists(local_file_path):
        file_age = time.time() - os.path.getmtime(local_file_path)
        if file_age < cache_duration:
            print(f"Using cached TLE data from '{local_file_path}'.")
            return read_local_tle_file(local_file_path)

    # Fetch data from the internet
    try:
        headers = {'User-Agent': 'MySatelliteSimulator/1.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with open(local_file_path, 'w') as file:
                file.write(response.text)
            print(f"Fetched and cached TLE data to '{local_file_path}'.")
            tle_data = response.text.splitlines()
            # Remove any empty lines
            tle_data = [line.strip() for line in tle_data if line.strip()]
            # Ensure the data length is a multiple of 3
            num_groups = len(tle_data) // 3
            tle_data = tle_data[:num_groups * 3]
            tle_groups = [tle_data[i:i+3] for i in range(0, len(tle_data), 3)]
            return [group for group in tle_groups if len(group) == 3]
        else:
            print(f"Error fetching TLE data: {response.status_code}.")
            return read_local_tle_file(local_file_path)
    except Exception as e:
        print(f"Exception during TLE data fetch: {e}.")
        return read_local_tle_file(local_file_path)

# Function to calculate orbit from TLE
def calculate_orbit(tle_group):
    if len(tle_group) != 3:
        return None  # Skip invalid TLE groups

    name, line1, line2 = tle_group
    # Create Orbit object
    try:
        orbit = Orbit.from_tle(line1, line2)
        return orbit
    except Exception as e:
        return None

# Parallel processing for orbit calculations
def calculate_orbits_parallel(tle_groups):
    valid_tle_groups = [tg for tg in tle_groups if len(tg) == 3]
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(calculate_orbit, valid_tle_groups), total=len(valid_tle_groups)))
    return results

# Fetch debris orbits
def get_debris_orbits(tle_urls):
    debris_orbits = []
    for name, (url, local_file_path) in tle_urls.items():
        tle_groups = fetch_tle_data_with_cache(url, local_file_path)
        if tle_groups:
            orbits_list = calculate_orbits_parallel(tle_groups)
            for orbit in orbits_list:
                if orbit:
                    debris_orbits.append(orbit)
    return debris_orbits

# Visualization function
def plot_trajectory(satellite_positions_history, debris_positions):
    fig = go.Figure()

    # Plot the satellite trajectory
    sat_positions = np.array(satellite_positions_history)
    fig.add_trace(go.Scatter3d(
        x=sat_positions[:, 0], y=sat_positions[:, 1], z=sat_positions[:, 2],
        mode='lines',
        name='Satellite Trajectory',
        line=dict(color='blue', width=2)
    ))

    # Plot debris positions
    debris_positions = np.array(debris_positions)
    fig.add_trace(go.Scatter3d(
        x=debris_positions[:, 0], y=debris_positions[:, 1], z=debris_positions[:, 2],
        mode='markers',
        name='Debris',
        marker=dict(size=2, color='red')
    ))

    # Add Earth model
    u_vals = np.linspace(0, 2 * np.pi, 100)
    v_vals = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u_vals, v_vals)
    x = EARTH_RADIUS * np.cos(u) * np.sin(v)
    y = EARTH_RADIUS * np.sin(u) * np.sin(v)
    z = EARTH_RADIUS * np.cos(v)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Earth',
        cmin=0, cmax=1,
        showscale=False,
        opacity=0.5,
        name='Earth'
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor="black"),
            yaxis=dict(title='Y (km)', backgroundcolor="black"),
            zaxis=dict(title='Z (km)', backgroundcolor="black"),
            bgcolor='black'
        ),
        title='Satellite Trajectory and Debris',
        showlegend=True
    )

    fig.show()

if __name__ == '__main__':
    # TLE URLs
    tle_urls = {
        "Active Satellites": (
            'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
            'tle_data/Active_Satellites.tle'
        ),
        "Space Debris": (
            'https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=tle',
            'tle_data/Space_Debris.tle'
        )
    }

    # Fetch debris orbits
    debris_orbits = get_debris_orbits(tle_urls)

    # Increase debris density by adding more random debris orbits
    # For simplicity, generate random orbits near the Earth's orbit
    for _ in range(5000):
        alt = np.random.uniform(200, 2000) * u.km
        inc = np.random.uniform(0, 180) * u.deg
        raan = np.random.uniform(0, 360) * u.deg
        argp = np.random.uniform(0, 360) * u.deg
        nu = np.random.uniform(0, 360) * u.deg
        epoch = Time.now()
        orbit = Orbit.from_classical(Earth, alt + Earth.R, 0 * u.one, inc, raan, argp, nu, epoch=epoch)
        debris_orbits.append(orbit)

    # Prepare the environment
    def make_env():
        env = SatelliteAvoidanceEnv(debris_orbits, max_debris=1000)  # Increase max_debris
        env.set_training(True)
        return env

    num_envs = 8  # Increase number of parallel environments
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Create the PPO model with adjusted hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        learning_rate=1e-4,         # Lower learning rate
        n_steps=2048,               # Increase number of steps per update
        batch_size=64,              # Adjust batch size
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,                # Increase number of epochs
        clip_range=0.2,
        ent_coef=0.01               # Encourage exploration
    )

    # Train the model
    total_timesteps = 1000000  # Increase training time
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("satellite_avoidance_model")

    # Testing the trained model
    test_env = SatelliteAvoidanceEnv(debris_orbits, max_debris=1000)
    test_env.set_training(False)
    obs = test_env.reset()

    satellite_positions_history = []

    for _ in range(test_env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated = test_env.step(action)
        r = test_env.satellite_orbit.r.to(u.km).value
        satellite_positions_history.append(r)
        if done or truncated:
            break

    # Visualization
    # Collect final debris positions for visualization
    debris_positions = test_env.selected_debris_positions
    plot_trajectory(satellite_positions_history, debris_positions)