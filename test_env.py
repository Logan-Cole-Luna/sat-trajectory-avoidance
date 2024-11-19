# test_environment.py

from eval.satellite_avoidance_env import SatelliteAvoidanceEnv
import numpy as np

def main():
    # Initialize environment with collision_course_probability=1.0
    env = SatelliteAvoidanceEnv(
        debris_positions=[],  # Will be overridden in reset
        collision_course_probability=1.0,
        enable_quantitative_metrics=True,
        enable_qualitative_evaluation=True,
        enable_visualization_tools=False  # Disable visualization for faster testing
    )

    num_test_episodes = 5

    for episode in range(num_test_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            # Simple policy: do nothing (no thrust)
            action = np.zeros(3)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            print(f"Episode {episode+1}, Step {step}, Reward: {reward}, Done: {done}")
        print(f"Episode {episode+1} ended after {step} steps.\n")

    env.close()

if __name__ == '__main__':
    main()
