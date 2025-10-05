import plotly.graph_objects as go
import numpy as np

# 1. Define the path of the sphere
# Create a spiral path in 3D using NumPy
t = np.linspace(0, 10 * np.pi, 250)
x = np.cos(t) * (1 + 0.1 * t)
y = np.sin(t) * (1 + 0.1 * t)
z = t

# Create the figure
fig = go.Figure(
    data=[
        # Trace 1: The moving sphere (a single marker)
        go.Scatter3d(
            x=[x[0]], 
            y=[y[0]], 
            z=[z[0]], 
            mode='markers',
            marker=dict(size = 12, color='red'),
            name='Sphere'
        ),
        # Trace 2: The trail (a line)
        go.Scatter3d(
            x=[], 
            y=[], 
            z=[], 
            mode='lines',
            line=dict(color='orange', width=4),
            name='Trail'
        )
    ],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(range=[-12, 12]),
            yaxis=dict(range=[-12, 12]),
            zaxis=dict(range=[-1, 35])
        ),
        title_text="Moving Sphere with a Trail",
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                )]
            )
        ],
        sliders=[dict(
            steps=[dict(method='animate', args=[[f'{k}'], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate", "transition": 
            {"duration": 0}}], label=str(k)) for k in range(len(t))],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(prefix="Frame: ", visible = True)
        )]
    )
)

# 2. Define the frames for the animation
frames = [
    go.Frame(
        name=str(k),
        data=[
            # Update the sphere's position
            go.Scatter3d(x=[x[k]], y=[y[k]], z=[z[k]]),
            # Update the trail, adding the new point
            go.Scatter3d(x=x[:k+1], y=y[:k+1], z=z[:k+1])
        ]
    ) for k in range(len(t))
]

# 3. Add the frames to the figure and display
fig.frames = frames
fig.show()

