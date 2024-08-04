import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def save_animation(all_user_positions, all_uav_positions, filename='animation.mp4'):
    fig, ax = plt.subplots()
    num_steps = len(all_user_positions[0])

    def update(step):
        ax.clear()
        user_positions = all_user_positions[0][step]
        uav_positions = all_uav_positions[0][step]
        ax.plot(user_positions[:, 0], user_positions[:, 1], 'bo', label='Users')
        ax.plot(uav_positions[:, 0], uav_positions[:, 1], 'ro', label='UAVs')
        ax.legend()
        ax.set_title(f"Step {step + 1}")
        ax.set_xlim(0, 100)  # Assuming your area_size is (100, 100)
        ax.set_ylim(0, 100)

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50)
    
    # Specify the writer as ffmpeg
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(filename, writer=writer)
