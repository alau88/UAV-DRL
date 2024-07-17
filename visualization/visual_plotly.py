import numpy as np
import plotly.graph_objects as go
from core.uav_env import UAVEnv

# Initialize environment
env = UAVEnv(num_users=20, num_uavs=3)

# Reset environment
obs = env.reset()

# Define a simple movement policy for demonstration (random movements)
num_steps = 50
user_positions_history = []
uav_positions_history = []

for step in range(num_steps):
    action = np.random.uniform(-1, 1, (env.num_uavs, 2))
    obs, reward, done, info = env.step(action)

    # Extract user and UAV positions
    user_positions = obs[:env.num_users]
    uav_positions = obs[env.num_uavs:]

    user_positions_history.append(user_positions)
    uav_positions_history.append(uav_positions)

# Create the plotly figure
fig = go.Figure()

# Add initial user and UAV positions
fig.add_trace(go.Scatter(
    x=user_positions_history[0][:, 0],
    y=user_positions_history[0][:, 1],
    mode='markers',
    marker=dict(color='blue'),
    name='Users'
))

fig.add_trace(go.Scatter(
    x=uav_positions_history[0][:, 0],
    y=uav_positions_history[0][:, 1],
    mode='markers',
    marker=dict(color='red'),
    name='UAVs'
))

# Create frames for each step
frames = []
for step in range(num_steps):
    frames.append(go.Frame(data=[
        go.Scatter(
            x=user_positions_history[step][:, 0],
            y=user_positions_history[step][:, 1],
            mode='markers',
            marker=dict(color='blue'),
            name='Users'
        ),
        go.Scatter(
            x=uav_positions_history[step][:, 0],
            y=uav_positions_history[step][:, 1],
            mode='markers',
            marker=dict(color='red'),
            name='UAVs'
        )
    ], name=str(step)))

fig.frames = frames

# Add slider
sliders = [dict(
    steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=300, redraw=True), transition=dict(duration=0))], label=str(k)) for k in range(num_steps)],
    transition=dict(duration=0),
    x=0.1,
    xanchor='left',
    y=0,
    yanchor='top'
)]

# Update layout with slider
fig.update_layout(
    title='UAV and User Positions Over Time',
    xaxis_title='X',
    yaxis_title='Y',
    sliders=sliders,
    updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True, transition=dict(duration=0))]), dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])])]
)

# Show the plot
fig.show()
