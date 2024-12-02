# evaluate_model.py

import numpy as np
from stable_baselines3 import PPO
from satellite_avoidance_env import SatelliteAvoidanceEnv
import logging
import os
import json
import csv

def create_evaluation_env(debris_positions, collision_course_probability=1.0):
    """
    Create a single evaluation environment with all evaluation features enabled.

    Args:
        debris_positions (list of array-like): Initial positions of debris objects in meters.
        collision_course_probability (float): Probability to set the satellite on a collision course each episode.

    Returns:
        SatelliteAvoidanceEnv: Environment configured for evaluation.
    """
    return SatelliteAvoidanceEnv(
        debris_positions=debris_positions,
        satellite_distance=None,  # Optional: Not set to ensure collision course
        init_angle=None,          # Optional: Not set to ensure collision course
        collision_course_probability=collision_course_probability,
        enable_quantitative_metrics=True,
        enable_qualitative_evaluation=True,
        enable_advanced_evaluation=True,
        enable_visualization_tools=True,
        enable_comparative_analysis=True,
        enable_robustness_testing=True
    )

def save_metrics_to_folder(metrics, folder_path, model_name):
    """
    Save evaluation metrics to a specified folder.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
        folder_path (str): Path to the folder where metrics will be saved.
        model_name (str): Name of the model (e.g., PPO, Baseline).
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save as JSON
    json_path = os.path.join(folder_path, f"{model_name}_metrics.json")
    with open(json_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)
    print(f"Metrics saved to {json_path}")

    # Optionally, save as CSV for easier analysis
    csv_path = os.path.join(folder_path, f"{model_name}_metrics.csv")
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    print(f"Metrics saved to {csv_path}")

def evaluate_model(model, eval_env, num_episodes=100, visualize=False):
    """
    Evaluate the trained PPO model on the evaluation environment.

    Args:
        model (stable_baselines3.PPO): Trained PPO model.
        eval_env (SatelliteAvoidanceEnv): Environment configured for evaluation.
        num_episodes (int): Number of evaluation episodes.
        visualize (bool): If True, generates trajectory visualizations.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    collision_count = 0
    total_rewards = []
    total_steps = []
    total_delta_v = []
    min_distances = []
    fuel_usage = []  # New metric
    trajectories = []

    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        cumulative_reward = 0.0
        steps = 0
        delta_v_total = 0.0
        min_distance = np.inf
        initial_fuel = eval_env.fuel_mass  # Track initial fuel
        trajectory = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            cumulative_reward += reward
            steps += 1

            # Track delta-v usage (assuming action scaled to delta-v)
            delta_v = np.linalg.norm(action * eval_env.time_increment)
            delta_v_total += delta_v

            # Track minimum distance to debris
            distances = [np.linalg.norm(eval_env.satellite_position - debris) for debris in eval_env.debris_positions]
            current_min_distance = min(distances)
            if current_min_distance < min_distance:
                min_distance = current_min_distance

            # Track fuel usage
            current_fuel = eval_env.fuel_mass
            fuel_consumed = initial_fuel - current_fuel
            fuel_usage.append(fuel_consumed)

            # Record trajectory
            if eval_env.enable_qualitative_evaluation:
                trajectory.append(eval_env.satellite_position.copy())

            if done and reward <= -1000.0:
                collision_count += 1

        total_rewards.append(cumulative_reward)
        total_steps.append(steps)
        total_delta_v.append(delta_v_total)
        min_distances.append(min_distance)
        trajectories.append(trajectory)

        # Save trajectory for visualization
        if visualize:
            eval_env.save_trajectory(episode_num=episode+1)
            eval_env.run_visualization(episode_num=episode+1)

    # Calculate metrics
    collision_rate = collision_count / num_episodes * 100
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)
    average_delta_v = np.mean(total_delta_v)
    average_min_distance = np.mean(min_distances)
    average_fuel_consumed = np.mean(fuel_usage)

    # Prepare metrics dictionary
    metrics = {
        "collision_rate": collision_rate,
        "average_cumulative_reward": average_reward,
        "average_steps_taken": average_steps,
        "average_delta_v_used_m_s": average_delta_v,
        "average_min_distance_to_debris_meters": average_min_distance,
        "average_fuel_consumed_kg": average_fuel_consumed
    }

    # Print detailed metrics
    print(f"\nEvaluation of PPO-trained model over {num_episodes} episodes:")
    print(f"Collision Rate: {collision_rate}%")
    print(f"Average Cumulative Reward: {average_reward:.2f}")
    print(f"Average Steps Taken: {average_steps}")
    print(f"Average Delta-v Used: {average_delta_v:.4f} m/s")
    print(f"Average Minimum Distance to Debris: {average_min_distance:.2f} meters")
    print(f"Average Fuel Consumed: {average_fuel_consumed:.2f} kg")

    return metrics

def evaluate_baseline(env, num_episodes=100):
    """
    Evaluate the baseline (rule-based) policy.

    Args:
        env (SatelliteAvoidanceEnv): Environment configured for evaluation.
        num_episodes (int): Number of episodes to run for baseline evaluation.

    Returns:
        dict: Dictionary containing baseline evaluation metrics.
    """
    collision_count = 0
    total_rewards = []
    total_steps = []
    total_delta_v = []
    min_distances = []
    fuel_usage = []  # New metric
    trajectories = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0.0
        steps = 0
        delta_v_total = 0.0
        min_distance = np.inf
        initial_fuel = env.fuel_mass  # Track initial fuel
        trajectory = []

        while not done:
            # Baseline action
            action = env._baseline_policy(obs)
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            steps += 1

            # Track delta-v usage
            delta_v = np.linalg.norm(action * env.time_increment)
            delta_v_total += delta_v

            # Track minimum distance to debris
            distances = [np.linalg.norm(env.satellite_position - debris) for debris in env.debris_positions]
            current_min_distance = min(distances)
            if current_min_distance < min_distance:
                min_distance = current_min_distance

            # Track fuel usage
            current_fuel = env.fuel_mass
            fuel_consumed = initial_fuel - current_fuel
            fuel_usage.append(fuel_consumed)

            # Record trajectory
            if env.enable_qualitative_evaluation:
                trajectory.append(env.satellite_position.copy())

            if done and reward <= -1000.0:
                collision_count += 1

        total_rewards.append(cumulative_reward)
        total_steps.append(steps)
        total_delta_v.append(delta_v_total)
        min_distances.append(min_distance)
        trajectories.append(trajectory)

        # Save trajectory for visualization
        if env.enable_visualization_tools:
            env.save_trajectory(episode_num=episode+1)
            env.run_visualization(episode_num=episode+1)

    # Calculate metrics
    collision_rate = collision_count / num_episodes * 100
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)
    average_delta_v = np.mean(total_delta_v)
    average_min_distance = np.mean(min_distances)
    average_fuel_consumed = np.mean(fuel_usage)

    # Prepare metrics dictionary
    metrics = {
        "collision_rate": collision_rate,
        "average_cumulative_reward": average_reward,
        "average_steps_taken": average_steps,
        "average_delta_v_used_m_s": average_delta_v,
        "average_min_distance_to_debris_meters": average_min_distance,
        "average_fuel_consumed_kg": average_fuel_consumed
    }

    # Print detailed metrics
    print(f"\nBaseline Evaluation over {num_episodes} episodes:")
    print(f"Collision Rate: {collision_rate}%")
    print(f"Average Cumulative Reward: {average_reward:.2f}")
    print(f"Average Steps Taken: {average_steps}")
    print(f"Average Delta-v Used: {average_delta_v:.4f} m/s")
    print(f"Average Minimum Distance to Debris: {average_min_distance:.2f} meters")
    print(f"Average Fuel Consumed: {average_fuel_consumed:.2f} kg")

    return metrics

def main():
    """
    Main function to evaluate the trained PPO model.
    """
    # Configuration
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    collision_course_probability = 1.0  # Always start on collision course

    # Create evaluation environment
    eval_env = create_evaluation_env(
        debris_positions=debris_positions_sample,
        collision_course_probability=collision_course_probability
    )

    # Load the trained model
    model = PPO.load("satellite_avoidance_model_combined")

    # Define output folder
    output_folder = "evaluation_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Perform Quantitative Performance Metrics Evaluation
    print("\nEvaluating PPO-trained model...")
    ppo_metrics = evaluate_model(model, eval_env, num_episodes=100, visualize=True)
    save_metrics_to_folder(ppo_metrics, output_folder, "PPO_trained_model")

    # Perform Comparative Analysis with Baseline Models
    print("\nEvaluating Baseline (Rule-Based) policy...")
    baseline_metrics = evaluate_baseline(eval_env, num_episodes=100)
    save_metrics_to_folder(baseline_metrics, output_folder, "Baseline_policy")

    # Perform Robustness and Generalization Testing
    print("\nPerforming Robustness and Generalization Testing...")
    robustness_metrics = eval_env.evaluate_robustness(num_episodes=100)
    save_metrics_to_folder(robustness_metrics, output_folder, "Robustness_testing")

    # Perform Advanced Evaluation Techniques (e.g., Noise Injection)
    print("\nPerforming Advanced Evaluation Techniques (Noise Injection)...")
    advanced_metrics = eval_env.run_advanced_evaluation(num_episodes=100)
    save_metrics_to_folder(advanced_metrics, output_folder, "Advanced_evaluation")

    # Close evaluation environment
    eval_env.close()

    # ===========================
    # End of Evaluation Phase
    # ===========================

    print("\nAll evaluations completed. Results are saved in the 'evaluation_results' folder.")

if __name__ == '__main__':
    main()
