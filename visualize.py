import plotly.graph_objects as go
import plotly.io as pio

def visualize_uav_positions(all_user_positions, all_uav_positions):
    fig = go.Figure()
    num_steps = len(all_user_positions[0])

    for step in range(num_steps):
        user_positions = all_user_positions[0][step]
        uav_positions = all_uav_positions[0][step]
        
        fig.add_trace(go.Scatter(
            x=user_positions[:, 0],
            y=user_positions[:, 1],
            mode='markers',
            marker=dict(color='blue'),
            name=f'Users (Step {step+1})',
            visible=True if step == 0 else 'legendonly'
        ))

        fig.add_trace(go.Scatter(
            x=uav_positions[:, 0],
            y=uav_positions[:, 1],
            mode='markers',
            marker=dict(color='red'),
            name=f'UAVs (Step {step+1})',
            visible=True if step == 0 else 'legendonly'
        ))

    # Create frames for animation
    frames = []
    for step in range(num_steps):
        user_positions = all_user_positions[0][step]
        uav_positions = all_uav_positions[0][step]
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

    pio.show(fig)
