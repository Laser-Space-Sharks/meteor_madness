
import plotly.graph_objects as go
import numpy as np

def generate_sphere_data(radius=1.0, resolution=40):
    """
    Generate coordinates for a sphere using spherical coordinates.
    """
    u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    return x, y, z

x_base, y_base, z_base = generate_sphere_data()


# Number of animation frames
num_frames = 100
frames = []

for i in range(num_frames):
    angle = 2 * np.pi * i / num_frames
    
    # Create a rotation matrix for the z-axis
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation to the sphere's coordinates
    x_rotated = x_base * rotation_matrix[0, 0] + y_base * rotation_matrix[0, 1]
    y_rotated = x_base * rotation_matrix[1, 0] + y_base * rotation_matrix[1, 1]
    z_rotated = z_base # No change for z-axis rotation
    
    # Append the rotated sphere as a frame
    frames.append(
        go.Frame(
            data=[go.Surface(x=x_rotated, y=y_rotated, z=z_rotated, colorscale='viridis')],
            name=str(i)
        )
    )


# Create the initial figure and layout
fig = go.Figure(
    data=[go.Surface(x=x_base, y=y_base, z=z_base, colorscale='viridis')],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis=dict(range=[-1.5, 1.5]),
            zaxis=dict(range=[-1.5, 1.5]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(label='Play', method='animate', args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}]),
                    dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])
                ]
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        args=[[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        label=f'{i}',
                        method='animate'
                    ) for i, f in enumerate(frames)
                ],
                transition={'duration': 0},
                x=0.1, y=0, pad={'t': 50}
            )
        ]
    ),
    frames=frames
)

fig.show()
