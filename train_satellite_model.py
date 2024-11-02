# train_satellite_model.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from satellite_avoidance_env import SatelliteAvoidanceEnv

if __name__ == '__main__':
    num_envs = 4  # Number of parallel environments
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    env_fns = [lambda: SatelliteAvoidanceEnv(debris_positions_sample) for _ in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    model = PPO('MlpPolicy', env, verbose=1, device='cuda')

    # Train the model
    model.learn(total_timesteps=50_000)

    # Save the trained model
    model.save("satellite_avoidance_model_advanced")

    print("Training complete. Model saved as 'satellite_avoidance_model_updated'.")
