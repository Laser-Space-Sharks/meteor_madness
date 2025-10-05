import plotly.graph_objects as go
import numpy as np

# Generate some sample data for animation
num_frames = 50
num_points = 100
frames_data = []


for i in range(num_frames):
    x = np.sin(np.linspace(0, 2 * np.pi, num_points) + i * 0.1) * (1 + i * 0.05)
    y = np.cos(np.linspace(0, 2 * np.pi, num_points) + i * 0.1) * (1 + i * 0.05)
    z = np.linspace(0, 10, num_points) + np.sin(i * 0.2) * 2
    frames_data.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, opacity=0.8)))

# Create the initial figure
fig = go.Figure(data=frames_data[0])

# Define the animation frames
frames = [go.Frame(data=[frame_data], name=str(i)) for i, frame_data in enumerate(frames_data)]
fig.frames = frames

# Add a slider and play/pause buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}])],
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(method="animate", args=[[f.name], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}], label=str(i))
                for i, f in enumerate(fig.frames)
            ]
        )
    ]
)



fig.show()
