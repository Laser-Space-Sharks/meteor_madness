import plotly.graph_objects as go
import numpy as np
from PIL import Image

# Color scale that convers our greyscale image into a colored image of the earth
colorscale =[
    [0.0, "rgb(30, 59, 117)"  ],
    [0.1, "rgb(46, 68, 21)"   ],
    [0.2, "rgb(74, 96, 28)"   ],
    [0.3, "rgb(115,141,90)"   ],
    [0.4, "rgb(122, 126, 75)" ],
    [0.6, "rgb(122, 126, 75)" ],
    [0.7, "rgb(141,115,96)"   ],
    [0.8, "rgb(223, 197, 170)"],
    [0.9, "rgb(237,214,183)"  ],
    [1.0, "rgb(255, 255, 255)"]
]

#Arbituary radius value of the earth
radius = 10

# Number of animation frames
num_frames = 100
frames = []

#Function that maps our textured image to a sphere
def sphere(size, texture): 
    N_lat = int(texture.shape[0])
    N_lon = int(texture.shape[1])
    theta = np.linspace(0,2*np.pi,N_lat)
    phi = np.linspace(0,np.pi,N_lon)
    
    # Set up coordinates for points on the sphere
    x0 = size * np.outer(np.cos(theta),np.sin(phi))
    y0 = size * np.outer(np.sin(theta),np.sin(phi))
    z0 = size * np.outer(np.ones(N_lat),np.cos(phi))
    
    # Set up trace
    return x0,y0,z0

texture = np.asarray(Image.open("rocky.jpeg")).T

x,y,z = sphere(radius,texture)
surf = go.Surface(x = x, y = y, z = z, surfacecolor = texture, colorscale = colorscale, showscale = False)    

for i in range(num_frames):
    angle = 2 * np.pi * i / num_frames
    
    # Create a rotation matrix for the z-axis
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation to the sphere's coordinates
    x_rotated = x * rotation_matrix[0, 0] + y * rotation_matrix[0, 1]
    y_rotated = x * rotation_matrix[1, 0] + y * rotation_matrix[1, 1]
    z_rotated = z # No change for z-axis rotation
    
    # Append the rotated sphere as a frame
    frames.append(
        go.Frame(
            data=[go.Surface(x=x_rotated, y=y_rotated, z=z_rotated, colorscale='viridis')],
            name=str(i)
        )
    )

#layout = go.Layout(scene = dict(aspectratio= dict(x = 1, y = 1, z = 1)))

#fig = go.Figure(data = [surf], layout = layout)

# Create the initial figure and layout
fig = go.Figure( data=[surf],
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

#Colors the surroundings to be dark and hides grid lines
fig.update_layout(template='plotly_dark')
fig.update_scenes(
    xaxis_showgrid = False,
    xaxis_visible  = False,
    yaxis_showgrid = False,
    yaxis_visible  = False,
    zaxis_showgrid = False,
    zaxis_visible  = False
)

fig.show()
