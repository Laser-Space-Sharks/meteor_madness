import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('action', __name__, url_prefix='/action')

def calcEvacCost(pop, daysAway, onLand):
    cost = (461.37 + 265.9)*pop*daysAway # (average cost + economic loss)
    if onLand:
        cost += 20000000000
    return cost  

@bp.route('/mitigation', methods=('GET', 'POST'))
def mitigation():
    defaultDaysAway = 40
    # evac_cost_calc = calcEvacCost(session.get('affected_pop'), defaultDaysAway, onLand])

    # if error is None:
    #     return render_template('action/mitigation.html')
    # else:
    #     flash(error)

    return render_template('action/mitigation.html', evac_cost=evac_cost_calc)

