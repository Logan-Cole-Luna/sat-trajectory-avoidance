# train_combined_satellite_model.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from satellite_avoidance_env import SatelliteAvoidanceEnv

if __name__ == '__main__':
    num_envs = 4  # Number of parallel environments

    # Use real debris positions for training
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]

    # Flag to always train on collision course
    collision_course_training = True

    env_fns = [
        lambda: SatelliteAvoidanceEnv(
            debris_positions=debris_positions_sample,
            satellite_distance=700e3,
            init_angle=45,
            collision_course=collision_course_training
        )
        for _ in range(num_envs)
    ]

    env = SubprocVecEnv(env_fns)

    model = PPO('MlpPolicy', env, verbose=1, device='cpu')  # Use 'cuda' if GPU is available

    # Train the model
    model.learn(total_timesteps=50_000)

    # Save the trained model
    model.save("satellite_avoidance_model_combined")

    print("Training complete. Model saved as 'satellite_avoidance_model_combined'.")
