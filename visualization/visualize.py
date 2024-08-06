import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

def visualize_uav_positions(all_user_positions, all_uav_positions):
    fig = go.Figure()

    # Determine the structure of the inputs
    if isinstance(all_user_positions, list) and isinstance(all_user_positions[0], list):
        # Original function input format (list of lists of arrays)
        num_steps = len(all_user_positions[0])
        user_positions_func = lambda step: all_user_positions[0][step]
        uav_positions_func = lambda step: all_uav_positions[0][step]
    elif isinstance(all_user_positions, np.ndarray):
        # New function input format (arrays)
        num_steps = 1  # Only one time step
        user_positions_func = lambda step: all_user_positions
        uav_positions_func = lambda step: all_uav_positions
    else:
        raise ValueError("Unrecognized input format for positions data.")

    for step in range(num_steps):
        user_positions = user_positions_func(step)
        uav_positions = uav_positions_func(step)

        fig.add_trace(go.Scatter(
            x=user_positions[:, 0],
            y=user_positions[:, 1],
            mode='markers',
            marker=dict(color='blue'),
            name=f'Users (Step {step + 1})',
            visible=True if step == 0 else 'legendonly'
        ))

        fig.add_trace(go.Scatter(
            x=uav_positions[:, 0],
            y=uav_positions[:, 1],
            mode='markers',
            marker=dict(color='red'),
            name=f'UAVs (Step {step + 1})',
            visible=True if step == 0 else 'legendonly'
        ))

    if num_steps > 1:
        # Create frames for animation
        frames = []
        for step in range(num_steps):
            user_positions = user_positions_func(step)
            uav_positions = uav_positions_func(step)
            frames.append(go.Frame(data=[
                go.Scatter(
                    x=user_positions[:, 0],
                    y=user_positions[:, 1],
                    mode='markers',
                    marker=dict(color='blue'),
                    name='Users'
                ),
                go.Scatter(
                    x=uav_positions[:, 0],
                    y=uav_positions[:, 1],
                    mode='markers',
                    marker=dict(color='red'),
                    name='UAVs'
                )
            ], name=str(step)))

        fig.frames = frames

        # Add slider
        sliders = [dict(
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=300, redraw=True),
                                                               transition=dict(duration=0))], label=str(k)) for k in
                   range(num_steps)],
            transition=dict(duration=0),
            x=0.1,
            xanchor='left',
            y=0,
            yanchor='top'
        )]

        # Update layout with slider and animation buttons
        fig.update_layout(
            title='UAV and User Positions Over Time',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            sliders=sliders,
            updatemenus=[dict(type='buttons', showactive=False, buttons=[
                dict(label='Play', method='animate', args=[None, dict(
                    frame=dict(duration=300, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0))]),
                dict(label='Pause', method='animate', args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode='immediate',
                    transition=dict(duration=0))])
            ])]
        )
    else:
        # Static layout for single step
        fig.update_layout(
            title='Initial Positions of UAVs and Users',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            width=1000,  # Set the width of the plot
            height=1000,  # Set the height of the plot
            showlegend=True,
            legend=dict(x=0.99, y=0.01, xanchor='right', yanchor='bottom')
        )

    pio.show(fig)