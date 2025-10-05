from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

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
            return render_template('SimulationManager/trajectory.html')
        else:
            flash(error)

    return render_template('SimulationManager/trajectory.html')

