import plotly.graph_objects as go
import numpy as np
from PIL import Image
import sys
sys.path.insert(1, "./kepler")
from kepler import *

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
gray_colorscale = [[0, "rgb(128, 128, 128)"], [1, "rgb(128, 128, 128)"]]
# Radius of the earth in km
radius = 6378.1

try:
    texture = np.asarray(Image.open("earth4.jpeg"))
except:
    texture = np.asarray(Image.open("plotly_plotting/earth4.jpeg"))

r = len(texture)
c = len(texture[0])
texture_fixed = np.empty((r, c))
for i in range(r):
    for j in range(c):
        texture_fixed[i][j] = int(texture[i][j][0])
texture_fixed = texture_fixed.T
#High definition image
#texture = np.asarray(Image.open("earth.jpeg")).T

#Function that maps our textured image to a sphere
def sphere(size, resolution): 
    theta = np.linspace(0, 2*np.pi, 2*resolution)
    phi   = np.linspace(0,   np.pi,   resolution)
    
    # Set up coordinates for points on the sphere
    x0 = size * np.outer(np.cos(theta),np.sin(phi))
    y0 = size * np.outer(np.sin(theta),np.sin(phi))
    z0 = size * np.outer(np.ones(resolution*2),np.cos(phi))
    return x0,y0,z0

x, y, z = sphere(radius, int(texture_fixed.shape[1]))
#
surf = go.Surface(x = x, y = y, z = z, surfacecolor = texture_fixed, colorscale = colorscale, showscale = False)

# Trace 2: The trail left by the asteroid
line = go.Scatter3d(x = [], y = [], z = [], mode = "lines", line = dict(color = "#FAB387", width = 3))

# Call Tristan's code to pick starting point for asteroid
def graph_conic(impact_conic):
    length = len(impact_conic[0])

    # Trace 1, the asteroid point
    point = go.Scatter3d(x = [impact_conic[0][0]], y = [impact_conic[1][0]], z = [impact_conic[2][0]], 
    mode = "markers", marker = dict(size = 5, color = "grey"))

    layout = go.Layout(
        scene = dict(aspectmode = "data"),
        updatemenus = [dict(
            type = "buttons", 
            showactive = False, 
            buttons = [
                dict(label="Play", method="animate",
                args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause",method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                ]
            )],
        sliders = [dict(
            steps = [dict(
                method = "animate", 
                args = [[f'{k}'], {
                    "frame": {"duration": 50, "redraw": True}, 
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                label = str(k)
            ) for k in range(length)], 
            transition = dict(duration = 0), 
            x = 0, 
            y = 0,
            currentvalue = dict(prefix = "Frame: ", visible = True)
        )]
    )
    
    fig = go.Figure(data = [surf, point, line], layout = layout)

    frames = [go.Frame(name = str(k), traces = [1, 2],
        data = [go.Scatter3d(x = [impact_conic[0][k]], y = [impact_conic[1][k]],  z = [impact_conic[2][k]]), 
                go.Scatter3d(x = impact_conic[0][:k+1],y = impact_conic[1][:k+1], z = impact_conic[2][:k+1])]) for k in range(length)
    ]

    #Colors the surroundings to be dark and hides grid lines
    fig.update_layout(template = "plotly_dark")
    fig.update_scenes(
        xaxis_showgrid = False,
        xaxis_visible  = False,
        yaxis_showgrid = False,
        yaxis_visible  = False,
        zaxis_showgrid = False,
        zaxis_visible  = False
    )
    fig.frames = frames

    return(fig)


fig = graph_conic(conic_from_impact(40.7128, -74.0068, np.array([3, 12, -2]), G*Me, 1000))
fig.show()
