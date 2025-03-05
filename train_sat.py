"""
This module provides functionality to fetch satellite TLE data, simulate orbital dynamics, 
and run a training loop using PPO for satellite avoidance. It includes functions to generate 
plots using Plotly and a custom Gym environment.
"""

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
from stable_baselines3.common.vec_env import SubprocVecEnv
import psutil
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os
import tensorboard
from typing import Dict
from utils.eval.satellite_avoidance_env import SatelliteAvoidanceEnv

# Constants for orbit and gravitational force
G = 6.67430e-11  # Gravitational constant in m^3 kg^−1 s^−2
M = 5.972e24  # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters

# Function to fetch TLE data from a URL or fallback to a local file
def fetch_tle_data(url, local_file_path):
    """
    Fetch Three-Line Element (TLE) data from a URL with a fallback to a local file.
    
    Parameters:
        url (str): The URL to fetch TLE data from.
        local_file_path (str): The local file path to read TLE data if fetching fails.
    
    Returns:
        List[List[str]]: A list of TLE groups where each group contains 3 lines (Name, Line1, Line2).
    
    Raises:
        Exception: If neither network nor local file access is successful.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            tle_data = response.text.splitlines()
            return [tle_data[i:i + 3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
        else:
            print(f"Error fetching TLE data: {response.status_code}, switching to local file.")
    except Exception as e:
        print(f"Error fetching TLE data from URL: {e}, switching to local file.")

    # Fallback to local file if fetching fails
    try:
        with open(local_file_path, 'r') as file:
            tle_data = file.read().splitlines()
            return [tle_data[i:i + 3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    except Exception as e:
        raise Exception(f"Error reading local TLE file '{local_file_path}': {e}")

# Function to calculate orbits in parallel
def calculate_orbits_parallel(tle_groups, time_range):
    """
    Calculate satellite orbits in parallel using TLE data.
    
    Parameters:
        tle_groups (List[List[str]]): List of TLE groups.
        time_range (float): Total simulation time in seconds.
    
    Returns:
        List: A list with each element being the calculated positions for a satellite.
    """
    def calculate_single_orbit(tle_group):
        """
        Calculate the orbit for a single TLE group.
        
        Parameters:
            tle_group (List[str]): A single TLE data group.
        
        Returns:
            List: A list of positions or None if calculation fails.
        """
        name, line1, line2 = tle_group
        satellite = Satrec.twoline2rv(line1, line2)
        positions = []
        for t in np.linspace(0, time_range, 1000):
            jd, fr = jday(2024, 10, 9, 12, 0, 0 + t)
            e, r, v = satellite.sgp4(jd, fr)
            if e == 0:
                positions.append(r)
        return positions if len(positions) > 0 else None

    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(calculate_single_orbit, tle_groups), total=len(tle_groups)))

# Function to calculate the average distance from Earth for a given set of positions
def average_distance_from_earth(positions):
    """
    Calculate the average distance from Earth given a list of 3D positions.
    
    Parameters:
        positions (List[array-like]): List of position vectors.
    
    Returns:
        float: The average distance from the Earth's center.
    """
    earth_center = np.array([0, 0, 0])  # Earth is centered at the origin
    distances = [np.linalg.norm(pos - earth_center) for pos in positions]
    return np.mean(distances)

# Function to scale orbit duration based on inverse distance
def dynamic_orbit_duration_inverse(distance, min_duration=50, max_duration=500, scaling_factor=5000):
    """
    Dynamically scale the number of orbit points based on inverse distance from Earth.
    
    Parameters:
        distance (float): Average distance from Earth.
        min_duration (int): Minimum orbit duration.
        max_duration (int): Maximum orbit duration.
        scaling_factor (float): Factor to scale the duration change.
    
    Returns:
        int: The number of orbit points to display.
    """
    # Closer objects get shorter durations, farther objects get longer durations
    if distance < scaling_factor:
        duration = int(max_duration - (distance / scaling_factor) * (max_duration - min_duration))
    else:
        duration = max_duration
    return duration

# Function to create a 3D Earth model with a colorscale (instead of texture)
def create_earth_model():
    """
    Create a 3D model of Earth with a predefined colorscale.
    
    Returns:
        go.Surface: A Plotly Surface object representing Earth.
    """
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
    """
    Plot orbits and collisions using Plotly.
    
    Parameters:
        active_positions (List[List[array-like]]): List of active satellite positions.
        debris_positions (List[List[array-like]]): List of debris positions.
        model_trajectory (List[array-like] or None): Trajectory from a model, if available.
        use_dynamic_scaling (bool): Whether to use dynamic scaling for plot durations.
        scaling_factor (float): Scaling factor for dynamic orbit duration.
    """
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


# Main code for fetching TLE data and calculating orbits
tle_urls = {
    "Last 30 Days' Launches": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle', 'tle_data/Last_30_Days_Launches.tle'),
    "Active Satellites": ('https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle', 'tle_data/Active_Satellites.tle'),
}

time_range = 86400 * 5  # 5 days

# Fetch and calculate orbits in parallel
active_sats_positions = []
debris_positions = []

for name, (url, local_file_path) in tle_urls.items():
    tle_groups = fetch_tle_data(url, local_file_path)
    positions = calculate_orbits_parallel(tle_groups[:100], time_range)  # Limit to 100 objects per group
    if 'debris' in name.lower():
        debris_positions.extend(filter(None, positions))
    else:
        active_sats_positions.extend(filter(None, positions))

# Use the trained PPO model if set to True
use_model = False

if use_model:
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    env = SatelliteAvoidanceEnv(debris_positions_sample)

    # Load the saved PPO model
    model = PPO.load("satellite_avoidance_model_ext")

    # Test the model and collect positions for plotting
    model_trajectory = []
    obs, _ = env.reset()  # Here, no info is expected, so only capture obs
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        model_trajectory.append(env.satellite_position.tolist())
        if done:
            obs, _ = env.reset()  # Only reset the observation
else:
    model_trajectory = None

#plot_orbits_and_collisions_plotly(
#    active_sats_positions,
#    debris_positions,
#    model_trajectory=model_trajectory
#)

# Training code
class MetricsCallback(BaseCallback):
    """
    Custom callback to log training metrics during PPO training.
    
    Attributes:
        training_metrics (dict): Dictionary to store metrics of the training process.
    """
    def __init__(self, verbose=0):
        """
        Initialize the MetricsCallback.
        
        Parameters:
            verbose (int): Verbosity level.
        """
        super().__init__(verbose)
        self.training_metrics = {
            'collision_count': 0,
            'rewards': [],
            'delta_v': [],
            'min_distances': [],
            'successful_avoidance': 0,
            'total_episodes': 0
        }

    def _on_step(self) -> bool:
        """
        Called at each step to record metrics.
        
        Returns:
            bool: True to indicate training should continue.
        """
        # Get current env info
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        
        # Update metrics
        if info.get('collision_occurred', False):
            self.training_metrics['collision_count'] += 1
        else:
            self.training_metrics['successful_avoidance'] += 1
            
        self.training_metrics['rewards'].append(reward)
        self.training_metrics['delta_v'].append(np.linalg.norm(self.locals['actions'][0]))
        
        # Log to tensorboard
        self.logger.record("train/collision_rate", 
                         self.training_metrics['collision_count'] / max(1, self.n_calls))
        self.logger.record("train/mean_reward", np.mean(self.training_metrics['rewards'][-100:]))
        self.logger.record("train/mean_delta_v", np.mean(self.training_metrics['delta_v'][-100:]))
        
        return True

if __name__ == '__main__':
    # Create directories for logs and models
    log_dir = "results/logs/"
    model_dir = "models/train/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/best_model", exist_ok=True)

    num_envs = 6
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    
    # Create training environment
    env = SubprocVecEnv([lambda: SatelliteAvoidanceEnv(debris_positions_sample) for _ in range(num_envs)])
    
    # Create separate evaluation environment
    eval_env = SatelliteAvoidanceEnv(debris_positions_sample)

    # Initialize callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model/",
        log_path=f"{log_dir}/eval/",
        eval_freq=2048,
        deterministic=True,
        render=False
    )
    
    metrics_callback = MetricsCallback()

    # Assign process to specific logical CPUs
    #psutil.Process().cpu_affinity(list(range(psutil.cpu_count(logical=True))))

    # Initialize and train PPO model with callbacks
    model = PPO(
        'MlpPolicy',      # Uses a Multi-Layer Perceptron (neural network) policy
        env,              # The environment to train on
        verbose=1,        # Level of logging output (0=none, 1=training info, 2=debug)
        tensorboard_log=log_dir,
        device='cuda',     
        n_steps=2048,     # Number of steps to run for each environment per update
        batch_size=1024,  # Number of samples per optimization step
        n_epochs=10       # Number of times to optimize on the same data
    )
    # clip_range=0.2 from PPO to stabilize updates.

    print("\nTo monitor training progress, run:")
    print(f"tensorboard --logdir={log_dir}")
    print("Then open http://localhost:6006\n")

    # Train with both callbacks
    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, metrics_callback]
    )

    # Save the final model
    model.save(f"{model_dir}/satellite_avoidance_model_final")

    # Print final metrics
    print("\nTraining Metrics:")
    print(f"Total Episodes: {metrics_callback.training_metrics['total_episodes']}")
    print(f"Collision Rate: {metrics_callback.training_metrics['collision_count'] / max(1, metrics_callback.training_metrics['total_episodes']):.2%}")
    print(f"Average Reward: {np.mean(metrics_callback.training_metrics['rewards']):.2f}")
    print(f"Average Delta-V: {np.mean(metrics_callback.training_metrics['delta_v']):.2f} m/s")
    
    # Test the final model
    test_env = SatelliteAvoidanceEnv(debris_positions_sample)
    obs, _ = test_env.reset()
    satellite_positions_history = []

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = test_env.step(action)
        satellite_positions_history.append(test_env.satellite_position.tolist())
        if done or truncated:
            obs, _ = test_env.reset()

    print("\nTesting complete.")