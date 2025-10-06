import os
import numpy as np
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import kepler
from graph2 import graph_conic
from math import radians

bp = Blueprint('SimulationManager', __name__, url_prefix='/SimulationManager')

def generate_graph(smaxis, ecc, argp, lasc, inc, nu, mu):
    session['keplerian_params'] = kepler.KeplerElement(
        smaxis,
        ecc, 
        argp, 
        lasc, 
        inc, 
        nu, 
        mu)
    session['data'] = kepler.kepler_conic(session.get('keplerian_params'), 500)
    session['latlon'] = kepler.latlon_from_cartesian(
        np.array([session.get('data')[0][-1], 
        session.get('data')[1][-1],
        session.get('data')[2][-1]])
    )
    return graph_conic(session.get('data'))

@bp.route('/trajectory', methods=('GET', 'POST'))
def trajectory():
    smaxis, ecc, argp, lasc, inc, nu, mu = kepler.KEPLER_DEFAULT
    fig = generate_graph(smaxis, ecc, argp, lasc, inc, nu, mu)

    if request.method == 'POST':
        smaxis = float(request.form['smaxis'])
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
        
        if float(ecc) > 1 and smaxis > 0:
            smaxis = -smaxis

        fig = generate_graph(
            float(smaxis), 
            float(ecc), 
            radians(float(argp)), 
            radians(float(lasc)), 
            radians(float(inc)), 
            radians(float(nu)), 
            radians(float(mu))
        )

        if error is None:
            return render_template('SimulationManager/trajectory.html', trajectory_plot=fig.to_html(full_html=False))
        else:
            flash(error)

    return render_template('SimulationManager/trajectory.html', trajectory_plot=fig.to_html(full_html=False))

@bp.route('/impact', methods=('GET', 'POST'))
def impact():
    print(session.get('latlon')[0], session.get('latlon'))
    return render_template('SimulationManager/impact.html', 
        lat=session.get('latlon')[0], 
        lon=session.get('latlon')[1],
        deathzone=50000,
        evac_zone=100000,
        crater=500,
    )