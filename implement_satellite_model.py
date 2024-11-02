# implement_satellite_model.py

from stable_baselines3 import PPO
import numpy as np
from satellite_avoidance_env import SatelliteAvoidanceEnv
import plotly.graph_objects as go
from astropy.constants import G
from poliastro.bodies import Earth

EARTH_RADIUS = Earth.R.to(u.m).value  # Earth's radius in meters

def plot_trajectory(trajectory):
    x_vals, y_vals, z_vals = zip(*trajectory)
    fig = go.Figure()

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = (EARTH_RADIUS / 1000) * np.outer(np.cos(u), np.sin(v))
    y = (EARTH_RADIUS / 1000) * np.outer(np.sin(u), np.sin(v))
    z = (EARTH_RADIUS / 1000) * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='earth',
        cmin=0, cmax=1,
        showscale=False,
        opacity=1,
        hoverinfo='skip'
    ))

    # Plot satellite trajectory
    fig.add_trace(go.Scatter3d(
        x=np.array(x_vals)/1000, y=np.array(y_vals)/1000, z=np.array(z_vals)/1000,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=2, color='red'),
        name='Satellite Trajectory'
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(title='X (km)'),
        yaxis=dict(title='Y (km)'),
        zaxis=dict(title='Z (km)'),
        aspectmode='data',
        bgcolor='black'
    ),
    title='Satellite Trajectory',
    showlegend=True)

    fig.show()

if __name__ == '__main__':
    debris_positions_sample = [np.random.randn(3) * 10000 for _ in range(100)]
    env = SatelliteAvoidanceEnv(debris_positions_sample)

    # Load the trained model
    model = PPO.load("satellite_avoidance_model_updated")

    obs, _ = env.reset()
    trajectory = []

    # Run the simulation
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        trajectory.append(env.satellite_position.copy())
        if done or truncated:
            break

    # Plot the trajectory
    plot_trajectory(trajectory)
