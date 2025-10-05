import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import kepler
from graph2 import graph_conic

bp = Blueprint('SimulationManager', __name__, url_prefix='/SimulationManager')

@bp.route('/trajectory', methods=('GET', 'POST'))
def trajectory():
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
        
        session['keplerian_params'] = kepler.KeplerElement(
            smaxis,
            ecc, 
            argp, 
            lasc, 
            inc, 
            nu, 
            kepler.g * kepler.Me)
        session['data'] = kepler.kepler_conic(session.get('keplerian_params'), 500)
        fig = graph_conic(session.get('data'))
        
        if error is None:
            return render_template('SimulationManager/trajectory.html', trajectory_plot=fig.to_html(full_html=False))
        else:
            flash(error)

    return render_template('SimulationManager/trajectory.html', trajectory_plot=fig.to_html(full_html=False))

@bp.route('/impact', methods=('GET', 'POST'))
def impact():
    return render_template('SimulationManager/impact.html')