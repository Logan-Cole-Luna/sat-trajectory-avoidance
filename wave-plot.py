import numpy as np
import plotly.graph_objects as go

# Create a grid
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
x, y = np.meshgrid(x, y)

# Positions and sizes of Earth, Moon, and Sun
earth_position = np.array([2, 2])
moon_position = np.array([-2, -2])
sun_position = np.array([0, 0])
earth_radius = 1.0  # Proportional size for Earth
moon_radius = 0.3   # Proportional size for Moon
sun_radius = 2.0    # Proportional size for Sun

# Calculate distances to simulate wave propagation
dist_earth = np.sqrt((x - earth_position[0])**2 + (y - earth_position[1])**2)
dist_moon = np.sqrt((x - moon_position[0])**2 + (y - moon_position[1])**2)
dist_sun = np.sqrt((x - sun_position[0])**2 + (y - sun_position[1])**2)

# Create wave patterns with a sinusoidal function and limit the height
wave_earth = 0.02 * np.sin(dist_earth * 2 * np.pi / (2 * earth_radius))
wave_earth = np.clip(wave_earth, -earth_radius, earth_radius)  # Limit wave height

wave_moon = 0.02 * np.sin(dist_moon * 2 * np.pi / (2 * moon_radius))
wave_moon = np.clip(wave_moon, -moon_radius, moon_radius)  # Limit wave height

wave_sun = 0.01 * np.sin(dist_sun * 2 * np.pi / (2 * sun_radius))  # Smaller amplitude for the Sun to avoid overpowering
wave_sun = np.clip(wave_sun, -sun_radius, sun_radius)  # Limit wave height

# Plotting with Plotly
fig = go.Figure(data=[
    go.Surface(z=wave_earth, x=x, y=y, colorscale='Blues', opacity=0.7, name='Earth Wave'),
    go.Surface(z=wave_moon, x=x, y=y, colorscale='Greys', opacity=0.7, name='Moon Wave'),
    go.Surface(z=wave_sun, x=x, y=y, colorscale='Oranges', opacity=0.5, name='Sun Wave'),
    go.Scatter3d(
        x=[earth_position[0]],
        y=[earth_position[1]],
        z=[0],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Earth'
    ),
    go.Scatter3d(
        x=[moon_position[0]],
        y=[moon_position[1]],
        z=[0],
        mode='markers',
        marker=dict(size=5, color='gray'),
        name='Moon'
    ),
    go.Scatter3d(
        x=[sun_position[0]],
        y=[sun_position[1]],
        z=[0],
        mode='markers',
        marker=dict(size=15, color='yellow'),
        name='Sun'
    )
])

fig.update_layout(
    title='Gravitational Wave Pattern of Earth, Moon, and Sun',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Wave Intensity',
        zaxis=dict(range=[-1.5, 1.5])  # Set Z-axis range to limit height
    )
)

fig.show()