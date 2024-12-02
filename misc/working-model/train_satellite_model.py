# train_satellite_model.py

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from utils.eval.satellite_avoidance_env import SatelliteAvoidanceEnv, EARTH_RADIUS
import os

def create_training_envs(num_envs, debris_positions, collision_course_probability=1.0):
    """
    Create multiple parallel training environments with collision courses enabled.

    Args:
        num_envs (int): Number of parallel environments.
        debris_positions (list of array-like): Initial positions of debris objects in meters.
        collision_course_probability (float): Probability to set the satellite on a collision course each episode.

    Returns:
        SubprocVecEnv: Vectorized environments for parallel training.
    """
    env_fns = [
        lambda: SatelliteAvoidanceEnv(
            debris_positions=debris_positions,
            satellite_distance=None,  # Not set to ensure collision course
            init_angle=None,          # Not set to ensure collision course
            collision_course_probability=collision_course_probability,
            enable_quantitative_metrics=False,
            enable_qualitative_evaluation=False,
            enable_advanced_evaluation=False,
            enable_visualization_tools=False,
            enable_comparative_analysis=False,
            enable_robustness_testing=False
        )
        for _ in range(num_envs)
    ]
    return SubprocVecEnv(env_fns)

def main():
    """
    Main function to train the PPO model with TensorBoard integration.
    """
    num_envs = 4  # Number of parallel environments for training

    # Generate debris positions in LEO (160 km to 2000 km altitude)
    debris_positions_sample = []
    for _ in range(100):
        altitude = np.random.uniform(160e3, 2000e3)  # meters
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = EARTH_RADIUS + altitude
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        debris_positions_sample.append(np.array([x, y, z]))

    # Collision course probability (e.g., 90% of episodes start on collision course)
    collision_course_probability = 0.9

    # Create TensorBoard log directory
    log_dir = "tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create training environments
    train_env = create_training_envs(
        num_envs=num_envs,
        debris_positions=debris_positions_sample,
        collision_course_probability=collision_course_probability
    )

    # Initialize PPO model with TensorBoard support
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        device='cuda',  # Change to 'cuda' if GPU is available
        tensorboard_log=log_dir
    )

    # Define evaluation environment for EvalCallback
    eval_env = SatelliteAvoidanceEnv(
        debris_positions=debris_positions_sample,
        collision_course_probability=1.0,  # Always start on collision course during evaluation
        enable_quantitative_metrics=True,
        enable_qualitative_evaluation=True,
        enable_visualization_tools=False  # Disable visualization during training for efficiency
    )

    # Create EvalCallback to evaluate the model periodically during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/evaluation/',
        eval_freq=5000,  # Evaluate every 5000 timesteps
        deterministic=True,
        render=False
    )

    # Train the model with EvalCallback
    print("Starting training with TensorBoard monitoring...")
    # Inform the user to launch TensorBoard
    print("\nTo monitor training progress, run the following command in your terminal:")
    print(f"tensorboard --logdir={log_dir} --port=6006")
    print("Then open your browser and navigate to http://localhost:6006/ to view TensorBoard.")
    model.learn(
        total_timesteps=500_000,
        callback=eval_callback
    )
    print("Training complete.")

    # Save the trained model
    model.save("satellite_avoidance_model_combined")
    print("Model saved as 'satellite_avoidance_model_combined.zip'.")

    # Close training and evaluation environments
    train_env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
