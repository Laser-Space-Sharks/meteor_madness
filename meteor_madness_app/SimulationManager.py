import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

import numpy as np
from PIL import Image
import plotly.graph_objects as go

bp = Blueprint('SimulationManager', __name__, url_prefix='/SimulationManager')

@bp.route('/trajectory', methods=('GET', 'POST'))
def trajectory():
    print(os.getcwd())
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
    try:
        texture = np.asarray(Image.open("earth.jpeg")).T
    except:
        texture = np.asarray(Image.open("plotly_plotting/earth.jpeg")).T

    x,y,z = sphere(radius,texture)
    surf = go.Surface(x = x, y = y, z = z, surfacecolor = texture, colorscale = colorscale, showscale = False)    

    layout = go.Layout(scene = dict(aspectratio= dict(x = 1, y = 1, z = 1)))
    fig = go.Figure(data = [surf], layout = layout)

    #Colors the surroundings to be dark and hides grid lines
    fig.update_layout(
        template='plotly_dark',
        width=1000,
        height=500
        )
    fig.update_scenes(
        xaxis_showgrid = False,
        xaxis_visible  = False,
        yaxis_showgrid = False,
        yaxis_visible  = False,
        zaxis_showgrid = False,
        zaxis_visible  = False
)

    if request.method == 'POST':
        smaxis = request.form['smaxis']
        ecc = request.form['ecc']
        argp = request.form['argp']
        lasc = request.form['lasc']
        inc = request.form['inc']
        nu = request.form['nu']

        error = None
        print(smaxis)

        if not smaxis: 
            error = 'smaxis is required.'
        elif not ecc:
            error = 'ecc is required.'
        elif not argp:
            error = 'argp is required.'
        elif not lasc:
            error = 'lasc is required.'
        elif not inc:
            error = 'inc is required.'
        elif not nu:
            error = 'nu is required.'
        
        if error is None:
            return render_template('SimulationManager/trajectory.html', trajectory_plot=fig.to_html(full_html=False))
        else:
            flash(error)

    return render_template('SimulationManager/trajectory.html', trajectory_plot=fig.to_html(full_html=False))

# def plotData():
#     exec(open("../plotly_plotting.py").read())
#     plot_data = {"trajectory_plot":fig.to_html(full_html=False)}
#     return render_template('SimulationManager/trajectory.html', trajectory_plot=plot_data)