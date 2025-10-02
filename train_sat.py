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
import gymnasium as gym
from gymnasium import spaces
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import Time
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import psutil
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
import os
import yaml
import tensorboard
from typing import Dict
import json
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


"""
Note: We only use local TLE files in this script. Use `tle_export.py` to refresh data.
The heavy TLE orbit plotting is disabled by default to keep training fast.
"""

# Optional: Load and process local TLEs only if explicitly enabled
ENABLE_TLE_PROCESSING = os.environ.get("ENABLE_TLE_PROCESSING", "0") == "1"
active_sats_positions = []
debris_positions = []
if ENABLE_TLE_PROCESSING:
    tle_local_files = {
        "Last 30 Days' Launches": ('tle_data/Last_30_Days_Launches.tle'),
        "Active Satellites": ('tle_data/Active_Satellites.tle'),
    }
    time_range = 86400 * 5  # 5 days
    for name, local_file_path in tle_local_files.items():
        # Read local file directly without network
        try:
            with open(local_file_path, 'r') as f:
                tle_data = f.read().splitlines()
                tle_groups = [tle_data[i:i + 3] for i in range(0, len(tle_data), 3)]
        except Exception as e:
            print(f"Failed to read TLE file {local_file_path}: {e}")
            tle_groups = []
        positions = calculate_orbits_parallel(tle_groups[:100], time_range)
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
class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning by progressively increasing difficulty.
    """
    def __init__(self, curriculum_config: Dict, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.stages = curriculum_config.get('stages', [])
        self.current_stage = 0
        self.stage_start_timestep = 0
        
    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"Starting curriculum learning with {len(self.stages)} stages")
        return True
        
    def _on_step(self) -> bool:
        # Check if we should advance to next curriculum stage
        if self.current_stage < len(self.stages):
            stage = self.stages[self.current_stage]
            stage_timesteps = stage.get('timesteps', 100000)
            
            if self.num_timesteps - self.stage_start_timestep >= stage_timesteps:
                if self.current_stage + 1 < len(self.stages):
                    self.current_stage += 1
                    self.stage_start_timestep = self.num_timesteps
                    next_stage = self.stages[self.current_stage]
                    
                    # Update environment parameters
                    if hasattr(self.training_env, 'envs'):
                        for env in self.training_env.envs:
                            if hasattr(env, 'env'):  # Unwrap Monitor
                                actual_env = env.env
                                if hasattr(actual_env, 'collision_course_probability'):
                                    actual_env.collision_course_probability = next_stage.get('collision_probability', 0.8)
                                if hasattr(actual_env, 'debris_radius_m'):
                                    actual_env.debris_radius_m = next_stage.get('debris_radius_m', 0.0)
                    
                    if self.verbose > 0:
                        print(f"\nAdvancing to curriculum stage {self.current_stage + 1}: {next_stage['name']}")
                        print(f"  Collision probability: {next_stage.get('collision_probability', 0.8)}")
                        print(f"  Debris radius: {next_stage.get('debris_radius_m', 0.0)} m")
        
        return True

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
        self._episode_rewards = None

    def _on_training_start(self) -> None:
        # Initialize episode reward trackers once env is available
        num_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1
        self._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    def _on_step(self) -> bool:
        """
        Called at each step to record metrics.
        
        Returns:
            bool: True to indicate training should continue.
        """
        rewards = self.locals['rewards']
        infos = self.locals['infos']
        dones = self.locals['dones']
        actions = self.locals['actions']

        # Accumulate step rewards per env
        if self._episode_rewards is None:
            num_envs = len(rewards) if hasattr(rewards, '__len__') else 1
            self._episode_rewards = np.zeros(num_envs, dtype=np.float64)
        self._episode_rewards += rewards

        # Track delta-v magnitude per step (mean across envs)
        self.training_metrics['delta_v'].append(float(np.mean(np.linalg.norm(actions, axis=1))))

        # Handle episode terminations per env
        for env_idx, done in enumerate(dones):
            if done:
                self.training_metrics['total_episodes'] += 1
                ep_info = infos[env_idx]
                if ep_info.get('collision_occurred', False):
                    self.training_metrics['collision_count'] += 1
                else:
                    self.training_metrics['successful_avoidance'] += 1

                # Record episode reward
                self.training_metrics['rewards'].append(self._episode_rewards[env_idx])
                self._episode_rewards[env_idx] = 0.0

        # Log to tensorboard (episode-level metrics)
        total_eps = max(1, self.training_metrics['total_episodes'])
        self.logger.record("train/collision_rate", self.training_metrics['collision_count'] / total_eps)
        if len(self.training_metrics['rewards']) > 0:
            self.logger.record("train/mean_ep_reward", np.mean(self.training_metrics['rewards'][-min(100, len(self.training_metrics['rewards'])):]))
        self.logger.record("train/mean_delta_v", np.mean(self.training_metrics['delta_v'][-min(100, len(self.training_metrics['delta_v'])):]))

        return True

if __name__ == '__main__':
    # Create directories for logs and models
    log_dir = "results/logs/"
    model_dir = "models/train/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/best_model", exist_ok=True)

    # Load YAML config (configs/train.yaml) with env fallback
    cfg_path = os.environ.get("TRAIN_CONFIG", "configs/train.yaml")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: failed to load {cfg_path}: {e}")

    # Configurable knobs via config/env vars
    num_envs = int(cfg.get("num_envs", os.environ.get("NUM_ENVS", "6")))
    disable_eval = bool(cfg.get("train", {}).get("disable_eval", os.environ.get("DISABLE_EVAL", "0") == "1"))
    eval_freq = int(cfg.get("train", {}).get("eval_freq", os.environ.get("EVAL_FREQ", "2048")))
    device_opt = str(cfg.get("device", os.environ.get("DEVICE", "cpu")))
    checkpoint_freq = int(cfg.get("train", {}).get("checkpoint_freq", os.environ.get("CHECKPOINT_FREQ", "100000")))
    # Vectorized env type: 'subproc' or 'dummy' (default to 'dummy' on Windows for stability)
    import sys
    default_vec = "dummy" if sys.platform.startswith("win") else "subproc"
    vec_cfg = str(cfg.get("vec_env", "auto")).lower()
    if vec_cfg == "auto":
        vec_env_type = default_vec
    else:
        vec_env_type = vec_cfg
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    # Training collision parameters
    def _get_bool_env(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        train_collision_prob = float(cfg.get("train", {}).get("collision", {}).get("probability", os.environ.get("TRAIN_COLLISION_PROB", "0.7")))
    except ValueError:
        train_collision_prob = 0.7
    train_guarantee_collision = bool(cfg.get("train", {}).get("collision", {}).get("guarantee", _get_bool_env("TRAIN_GUARANTEE_COLLISION", False)))
    try:
        tcs = cfg.get("train", {}).get("collision", {}).get("ttc_steps", os.environ.get("TRAIN_COLLISION_TTC_STEPS"))
        train_collision_ttc_steps = int(tcs) if tcs is not None else None
    except ValueError:
        train_collision_ttc_steps = None
    try:
        debris_radius_m = float(cfg.get("train", {}).get("collision", {}).get("debris_radius_m", os.environ.get("TRAIN_DEBRIS_RADIUS_M", "0.0")))
    except ValueError:
        debris_radius_m = 0.0
    
    # Physical parameters
    try:
        max_thrust = float(cfg.get("train", {}).get("max_thrust", os.environ.get("TRAIN_MAX_THRUST", "0.05")))
    except ValueError:
        max_thrust = 0.05
    
    # Test-specific parameters (for final evaluation)
    try:
        test_collision_prob = float(cfg.get("test", {}).get("collision_probability", train_collision_prob))
    except (ValueError, TypeError):
        test_collision_prob = train_collision_prob
    try:
        test_debris_radius_m = float(cfg.get("test", {}).get("debris_radius_m", debris_radius_m))
    except (ValueError, TypeError):
        test_debris_radius_m = debris_radius_m
    try:
        test_collision_ttc_steps = cfg.get("test", {}).get("ttc_steps", train_collision_ttc_steps)
        test_collision_ttc_steps = int(test_collision_ttc_steps) if test_collision_ttc_steps is not None else None
    except (ValueError, TypeError):
        test_collision_ttc_steps = train_collision_ttc_steps
    
    # Load reward parameters from config
    reward_cfg = cfg.get("train", {}).get("reward", {})
    reward_params = {
        'collision_penalty': float(reward_cfg.get('collision_penalty', 500.0)),
        'survival_reward': float(reward_cfg.get('survival_reward', 0.02)),
        'distance_shaping_coeff': float(reward_cfg.get('distance_shaping_coeff', 0.0002)),
        'coast_bonus': float(reward_cfg.get('coast_bonus', 0.05)),
        'delta_v_linear_cost': float(reward_cfg.get('delta_v_linear_cost', 6.0)),
        'delta_v_quadratic_cost': float(reward_cfg.get('delta_v_quadratic_cost', 0.5)),
        'large_burn_penalty': float(reward_cfg.get('large_burn_penalty', 3.0)),
        'smoothness_penalty': float(reward_cfg.get('smoothness_penalty', 0.005)),
        'cumulative_dv_penalty': float(reward_cfg.get('cumulative_dv_penalty', 0.2)),
        'projected_miss_reward': float(reward_cfg.get('projected_miss_reward', 0.0002)),
        'milestone_reward': float(reward_cfg.get('milestone_reward', 0.4)),
    }
    
    # Create training environment
    if vec_env_type == "subproc" and num_envs > 1:
        env = SubprocVecEnv([
            lambda: Monitor(SatelliteAvoidanceEnv(
                debris_positions_sample,
                collision_course_probability=train_collision_prob,
                guarantee_collision=train_guarantee_collision,
                collision_ttc_steps=train_collision_ttc_steps,
                debris_radius_m=debris_radius_m,
                max_thrust=max_thrust,
                enable_quantitative_metrics=True,
                **reward_params  # Pass configurable reward parameters
            )) for _ in range(num_envs)
        ])
    else:
        # Fallback to DummyVecEnv for Windows/num_envs==1
        env = DummyVecEnv([
            lambda: Monitor(SatelliteAvoidanceEnv(
                debris_positions_sample,
                collision_course_probability=train_collision_prob,
                guarantee_collision=train_guarantee_collision,
                collision_ttc_steps=train_collision_ttc_steps,
                debris_radius_m=debris_radius_m,
                max_thrust=max_thrust,
                enable_quantitative_metrics=True,
                **reward_params  # Pass configurable reward parameters
            )) for _ in range(num_envs)
        ])
    
    # Create separate evaluation environment (match vectorization type for stability)
    if vec_env_type == "subproc" and num_envs > 1:
        eval_env = SubprocVecEnv([
            lambda: Monitor(SatelliteAvoidanceEnv(
                debris_positions_sample,
                collision_course_probability=train_collision_prob,
                guarantee_collision=False,
                collision_ttc_steps=train_collision_ttc_steps,
                debris_radius_m=debris_radius_m,
                max_thrust=max_thrust,
                enable_quantitative_metrics=True,
                **reward_params  # Pass configurable reward parameters
            ))
        ])
    else:
        eval_env = DummyVecEnv([
            lambda: Monitor(SatelliteAvoidanceEnv(
                debris_positions_sample,
                collision_course_probability=train_collision_prob,
                guarantee_collision=False,
                collision_ttc_steps=train_collision_ttc_steps,
                debris_radius_m=debris_radius_m,
                max_thrust=max_thrust,
                enable_quantitative_metrics=True,
                **reward_params  # Pass configurable reward parameters
            ))
        ])

    # Initialize callbacks
    callbacks = []
    
    # Add curriculum learning callback if enabled
    if cfg.get('curriculum', {}).get('enabled', False):
        curriculum_callback = CurriculumCallback(cfg['curriculum'], verbose=1)
        callbacks.append(curriculum_callback)
    
    if not disable_eval:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{model_dir}/best_model/",
            log_path=f"{log_dir}/eval/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)

    # Periodic checkpoint saving
    if checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=checkpoint_freq // max(1, num_envs),
            save_path=model_dir,
            name_prefix="ppo_checkpoint"
        )
        callbacks.append(ckpt_cb)

    # Assign process to specific logical CPUs
    #psutil.Process().cpu_affinity(list(range(psutil.cpu_count(logical=True))))

    # Enhanced PPO model with improved hyperparameters
    # Learning rate schedule
    lr_schedule = cfg.get("train", {}).get("lr_schedule", "constant")
    if lr_schedule == "linear":
        def linear_schedule(initial_value: float):
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value
            return func
        learning_rate = linear_schedule(float(cfg.get("train", {}).get("learning_rate", 0.0001)))
    else:
        learning_rate = float(cfg.get("train", {}).get("learning_rate", 0.0001))
    
    # Check if resuming from checkpoint
    resume_config = cfg.get("train", {}).get("resume", {})
    resume_enabled = resume_config.get("enabled", False)
    resume_model_path = resume_config.get("model_path", "")
    start_timestep = resume_config.get("start_timestep", 0)
    
    if resume_enabled and os.path.exists(resume_model_path):
        print(f"Resuming training from: {resume_model_path}")
        print(f"Starting from timestep: {start_timestep}")
        
        # Load existing model
        model = PPO.load(
            resume_model_path,
            env=env,
            device=device_opt,
            tensorboard_log=log_dir
        )
        
        # Update any hyperparameters that might have changed
        model.learning_rate = learning_rate
        model.ent_coef = float(cfg.get("train", {}).get("ent_coef", 0.005))
        model.vf_coef = float(cfg.get("train", {}).get("vf_coef", 0.5))
        model.max_grad_norm = float(cfg.get("train", {}).get("max_grad_norm", 0.5))
        model.gamma = float(cfg.get("train", {}).get("gamma", 0.999))
        model.gae_lambda = float(cfg.get("train", {}).get("gae_lambda", 0.95))
        # For clip_range, we need to handle it as either a float or callable
        clip_range_val = float(cfg.get("train", {}).get("clip_range", 0.2))
        model.clip_range = lambda _: clip_range_val  # Make it callable
        model.target_kl = float(cfg.get("train", {}).get("target_kl", 0.01))
        
        print("Model loaded successfully for resuming")
    else:
        print("Creating new PPO model")
        model = PPO(
            'MlpPolicy',      # Uses a Multi-Layer Perceptron (neural network) policy
            env,              # The environment to train on
            verbose=1,        # Level of logging output (0=none, 1=training info, 2=debug)
            tensorboard_log=log_dir,
            device=device_opt,
            # Improved hyperparameters for orbital mechanics
            n_steps=int(cfg.get("train", {}).get("n_steps", 2048)),
            batch_size=int(cfg.get("train", {}).get("batch_size", 256)),
            n_epochs=int(cfg.get("train", {}).get("n_epochs", 10)),
            learning_rate=learning_rate,
            ent_coef=float(cfg.get("train", {}).get("ent_coef", 0.005)),
            vf_coef=float(cfg.get("train", {}).get("vf_coef", 0.5)),
            max_grad_norm=float(cfg.get("train", {}).get("max_grad_norm", 0.5)),
            gamma=float(cfg.get("train", {}).get("gamma", 0.999)),
            gae_lambda=float(cfg.get("train", {}).get("gae_lambda", 0.95)),
            clip_range=float(cfg.get("train", {}).get("clip_range", 0.2)),
            target_kl=float(cfg.get("train", {}).get("target_kl", 0.01)),
            policy_kwargs=dict(
                net_arch=cfg.get("train", {}).get("net_arch", [256, 256, 128])
            )
        )

    print("\nTo monitor training progress, run:")
    print(f"tensorboard --logdir={log_dir}")
    print("Then open http://localhost:6006\n")

    # Calculate remaining timesteps for training
    total_timesteps = int(cfg.get("train", {}).get("total_timesteps", os.environ.get("TOTAL_TIMESTEPS", "500000")))
    
    if resume_enabled and start_timestep > 0:
        remaining_timesteps = total_timesteps - start_timestep
        print(f"Resuming training:")
        print(f"  Total target timesteps: {total_timesteps:,}")
        print(f"  Already completed: {start_timestep:,}")
        print(f"  Remaining timesteps: {remaining_timesteps:,}")
        
        if remaining_timesteps <= 0:
            print("Training already completed!")
            exit(0)
            
        actual_timesteps = remaining_timesteps
    else:
        actual_timesteps = total_timesteps
        print(f"Starting fresh training for {actual_timesteps:,} timesteps")

    # Train with callbacks
    model.learn(
        total_timesteps=actual_timesteps,
        callback=callbacks,
        reset_num_timesteps=not resume_enabled  # Don't reset if resuming
    )

    # Save the final model
    model.save(f"{model_dir}/satellite_avoidance_model")

    # Print final metrics
    print("\nTraining Metrics:")
    print(f"Total Episodes: {metrics_callback.training_metrics['total_episodes']}")
    print(f"Collision Rate: {metrics_callback.training_metrics['collision_count'] / max(1, metrics_callback.training_metrics['total_episodes']):.2%}")
    print(f"Average Reward: {np.mean(metrics_callback.training_metrics['rewards']):.2f}")
    print(f"Average Delta-V: {np.mean(metrics_callback.training_metrics['delta_v']):.2f} m/s")
    # Save metrics to JSON
    os.makedirs('results/evaluation', exist_ok=True)
    summary = {
        'total_episodes': int(metrics_callback.training_metrics['total_episodes']),
        'collision_rate': float(metrics_callback.training_metrics['collision_count'] / max(1, metrics_callback.training_metrics['total_episodes'])),
        'avg_reward': float(np.mean(metrics_callback.training_metrics['rewards']) if len(metrics_callback.training_metrics['rewards']) else 0.0),
        'avg_delta_v': float(np.mean(metrics_callback.training_metrics['delta_v']) if len(metrics_callback.training_metrics['delta_v']) else 0.0),
        'timesteps': int(total_timesteps),
        'vec_env': 'DummyVecEnv' if isinstance(env, DummyVecEnv) else 'SubprocVecEnv',
        'eval_freq': int(eval_freq),
        'num_envs': int(num_envs),
        'device': str(device_opt)
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/evaluation/training_summary_{ts}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Test the final model
    test_env = SatelliteAvoidanceEnv(
        debris_positions_sample,
        collision_course_probability=test_collision_prob,
        guarantee_collision=False,
        collision_ttc_steps=test_collision_ttc_steps,
        debris_radius_m=test_debris_radius_m,
        max_thrust=max_thrust,
        enable_quantitative_metrics=True,
        **reward_params  # Pass configurable reward parameters
    )
    obs, _ = test_env.reset()
    satellite_positions_history = []

    test_steps = int(cfg.get("test", {}).get("steps", os.environ.get("TEST_STEPS", "1000")))
    for _ in range(test_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = test_env.step(action)
        satellite_positions_history.append(test_env.satellite_position.tolist())
        if done or truncated:
            obs, _ = test_env.reset()

    print("\nTesting complete.")

    # Diagnostic: No-thrust collision check and plot
    def simulate_no_thrust_collision(env: SatelliteAvoidanceEnv, steps: int = 1500):
        import plotly.graph_objects as go
        obs, _ = env.reset()
        # Explicitly set a collision course: pick a debris and aim directly at it
        if len(env.debris_positions) == 0:
            print("No debris available for diagnostic.")
            return None, False
        target = env.debris_positions[0]
        # Place satellite outside collision threshold and point toward debris
        direction = (target - env.satellite_position)
        if np.linalg.norm(direction) == 0:
            direction = np.array([1.0, 0.0, 0.0])
        direction = direction / np.linalg.norm(direction)
        start_offset = 15000.0  # meters (> D_COLLIDE)
        env.satellite_position = target - direction * start_offset
        env.satellite_velocity = direction * 60.0  # m/s toward debris
        env.time_increment = 1.0  # seconds for finer stepping
        # Disable gravity during diagnostic to ensure straight-line approach
        original_gravity = env._apply_gravitational_force
        env._apply_gravitational_force = lambda: None
        pos_traj = []
        collided = False
        for _ in range(steps):
            zero_action = np.zeros(3, dtype=np.float32)
            obs, reward, done, truncated, info = env.step(zero_action)
            pos_traj.append(env.satellite_position.copy())
            if info.get('collision_occurred', False):
                collided = True
            if done or truncated:
                break
        # Restore gravity
        env._apply_gravitational_force = original_gravity

        # Build simple 3D plot of satellite path and debris
        fig = go.Figure()
        if len(pos_traj) > 1:
            x, y, z = np.array(pos_traj).T
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Satellite (no thrust)'))
        for d in env.debris_positions:
            fig.add_trace(go.Scatter3d(x=[d[0]], y=[d[1]], z=[d[2]], mode='markers', marker=dict(size=3, color='red'), name='Debris'))

        # Earth sphere
        u_lin = np.linspace(0, 2*np.pi, 50)
        v_lin = np.linspace(0, np.pi, 50)
        xs = (EARTH_RADIUS*np.outer(np.cos(u_lin), np.sin(v_lin))).astype(float)
        ys = (EARTH_RADIUS*np.outer(np.sin(u_lin), np.sin(v_lin))).astype(float)
        zs = (EARTH_RADIUS*np.outer(np.ones_like(u_lin), np.cos(v_lin))).astype(float)
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.3, showscale=False, name='Earth'))
        fig.update_layout(scene=dict(aspectmode='data'), title=f'No-Thrust Diagnostic (collided={collided})')
        os.makedirs('results/evaluation', exist_ok=True)
        out_html = 'results/evaluation/no_thrust_collision_diagnostic.html'
        fig.write_html(out_html)
        return out_html, collided

    diag_env = SatelliteAvoidanceEnv(debris_positions_sample)
    out_plot, collided = simulate_no_thrust_collision(diag_env, steps=2000)
    print(f"No-thrust diagnostic saved to: {out_plot}. Collided={collided}")