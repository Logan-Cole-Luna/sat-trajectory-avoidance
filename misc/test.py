import plotly.graph_objects as go
import numpy as np

# Function to simulate debris and satellite movements over time
def simulate_debris_movement(num_steps, debris_positions, satellite_positions):
    # Lists to store the positions over time
    debris_trajectories = []
    satellite_trajectories = []

    # Simulate movement for each debris and satellite
    for t in range(num_steps):
        # Simulating some small random movement (you could replace this with real data)
        debris_step = [pos + np.random.uniform(-0.01, 0.01, 3) for pos in debris_positions]
        satellite_step = [sat_pos + np.random.uniform(-0.005, 0.005, 3) for sat_pos in satellite_positions]

        debris_trajectories.append(debris_step)
        satellite_trajectories.append(satellite_step)

    return debris_trajectories, satellite_trajectories

# Earth data (a simple textured globe)
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(theta), np.sin(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.ones(100), np.cos(phi))

# Initial debris and satellite positions (replace these with actual data)
num_debris = 100
num_satellites = 5
debris_positions = [np.random.uniform(-1.5, 1.5, 3) for _ in range(num_debris)]
satellite_positions = [np.array([0, 0, 0])] * num_satellites  # Start at origin for simplicity

# Simulate debris and satellite movement over time (e.g., 100 steps)
num_steps = 100
debris_trajectories, satellite_trajectories = simulate_debris_movement(num_steps, debris_positions, satellite_positions)

# Create Plotly figure
fig = go.Figure()

# Add initial Earth globe
earth_surface = go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.5, name='Earth')
fig.add_trace(earth_surface)

# Plot the initial positions of debris and satellites
for debris_pos in debris_trajectories[0]:
    fig.add_trace(go.Scatter3d(
        x=[debris_pos[0]], y=[debris_pos[1]], z=[debris_pos[2]],
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Debris'
    ))

for satellite_pos in satellite_trajectories[0]:
    fig.add_trace(go.Scatter3d(
        x=[satellite_pos[0]], y=[satellite_pos[1]], z=[satellite_pos[2]],
        mode='markers',
        marker=dict(size=5, color='green'),
        name='Satellite'
    ))

# Create animation frames
frames = []
for step in range(1, num_steps):
    frame_data = []
    
    # Keep the Earth visible in every frame
    frame_data.append(earth_surface)  # Earth plot must be added to every frame
    
    # Update debris positions
    for debris_pos in debris_trajectories[step]:
        frame_data.append(go.Scatter3d(
            x=[debris_pos[0]], y=[debris_pos[1]], z=[debris_pos[2]],
            mode='markers',
            marker=dict(size=2, color='red'),
            name='Debris'
        ))
    
    # Update satellite positions
    for satellite_pos in satellite_trajectories[step]:
        frame_data.append(go.Scatter3d(
            x=[satellite_pos[0]], y=[satellite_pos[1]], z=[satellite_pos[2]],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Satellite'
        ))
    
    frames.append(go.Frame(data=frame_data, name=str(step)))

# Define layout for the animation
fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False),
        bgcolor="black"
    ),
    title="Space Debris and Satellite Simulation",
    showlegend=True,
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True, "mode": "immediate"}]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}])])]
)

# Add the frames to the figure
fig.frames = frames

# Show the animated plot
fig.show()