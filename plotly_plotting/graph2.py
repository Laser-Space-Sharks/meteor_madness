import plotly.graph_objects as go
import numpy as np
from PIL import Image
import kepler as kp

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

texture = np.asarray(Image.open("earth4.jpeg"))
r = len(texture)
c = len(texture[0])

#High definition image
#texture = np.asarray(Image.open("earth.jpeg")).T

impact_conic = kp.conic_from_impact(40.7128, -74.0068, np.array([3, 12, -2]), kp.G*kp.Me, 1000)
print(impact_conic[0][0])

texture_fixed = np.empty((r, c))
for i in range(r):
    for j in range(c):
        texture_fixed[i][j] = int(texture[i][j][0])
texture_fixed = texture_fixed.T

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
# Call Tristan's code to pick starting point for asteroid
# Picking and arbituary spot for now

# Some radius given from website, picking random value for now
surf = go.Surface(x = x, y = y, z = z, surfacecolor = texture_fixed, colorscale = colorscale, showscale = False)
point = go.Scatter3d(x = [10000], y = [0], z = [0], mode = "markers", marker = dict(size = 5, color = "grey"))

# Trace 2: The trail (a line)
line = go.Scatter3d(x = [], y = [], z = [], mode = "lines", line = dict(color = "#FAB387", width = 3))

layout = go.Layout(
    scene = dict(aspectmode = "data"), 
    updatemenus = [dict(
        type = "buttons", 
        showactive = False, 
        buttons = [dict(
            label = "Play", 
            method = "animate",
            args = [None, {
            "frame": {"duration": 50, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 0}
            }]
        )]
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
        ) for k in range(10)], 
        transition = dict(duration = 0), 
        x = 0, 
        y = 0,
        currentvalue = dict(prefix = "Frame: ", visible = True)
    )]
)
    
fig = go.Figure(data = [surf, point, line], layout = layout)

frames = [go.Frame(traces = [1, 2],
    data = [go.Scatter3d(x = impact_conic[0], y = impact_conic[1], z = impact_conic[2]), 
            go.Scatter3d(x = np.linspace(10000, 10000-k*100, 100), y = np.zeros(100), z = np.zeros(100))]) for k in range(100)
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
fig.show()

