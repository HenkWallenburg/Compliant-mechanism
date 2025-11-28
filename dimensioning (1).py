# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:38:13 2025

@author: atabe
"""

import numpy as np
from math import sin, cos
from scipy.optimize import minimize

# Constants
l11 = 8.0
l12 = 2.25
dot_theta2 = 0.1047
eps = 1e-6   # strict inequality margin

def residuals_array(x):
    # Unpack variables
    l1, l2, l3, l4, l5, l6, l10, l13 = x[0:8]
    th2, th4, th5, th8, th9, th10, th11 = x[8:15]
    dth5, dth4, dth8, dth9, dl1 = x[15:20]
    
    r = np.zeros(17)

    r[0]  = l2*cos(th2) + l6*cos(th5) - l10*cos(th10) + l12*cos(th8) - l12*cos(th9)
    r[1]  = l2*sin(th2) + l6*sin(th5) - l10*sin(th10) + l12*sin(th8) - l12*sin(th9)

    r[2]  = l1 + l4*cos(th4) + l5*cos(th5) - l11*cos(th11)
    r[3]  = l3 + l4*sin(th4) + l5*sin(th5) - (l11*sin(th11) + l12)

    r[4]  = l2*cos(th2) + l12*cos(th8) + l11*cos(th5) - l11*cos(th11)
    r[5]  = l2*sin(th2) + l12*sin(th8) + l11*sin(th5) - (l11*sin(th11) + l12)

    r[6]  = l2*dot_theta2*cos(th2) + l6*dth5*cos(th5)
    r[7]  = l2*dot_theta2*sin(th2) + l6*dth5*sin(th5)

    r[8]  = dl1 + l4*dth4*cos(th4) + l5*dth5*cos(th5)
    r[9]  = l4*dth4*sin(th4) + l5*dth5*sin(th5)

    r[10] = l2*dot_theta2*cos(th2) + l12*dth8*cos(th8) + l11*dth5*cos(th5)
    r[11] = l2*dot_theta2*sin(th2) + l12*dth8*sin(th8) + l11*dth5*sin(th5)
    
    r[12] = l10*cos(th10) + l12*cos(th9) + l13*cos(th5) - l11*cos(th11)
    r[13] = l10*sin(th10) + l12*sin(th9) + l13*sin(th5) - l11*sin(th11) - l12
    
    r[14] = l12*dth9*cos(th9) + l13*dth5*cos(th5)
    r[15] = l12*dth9*sin(th9) + l13*dth5*sin(th5)
    
    r[16] = l6 + l13 - l11

    return r

def objective(x):
    r = residuals_array(x)
    return 0.5 * np.dot(r, r)

# ---------------------------
#   INEQUALITY CONSTRAINTS
# ---------------------------
cons = [
    # l10 < l11
    {'type':'ineq', 'fun': lambda x: l11 - x[8] - eps},

    # l5 > l7
    {'type': 'ineq', 'fun': lambda x: x[4] - x[6] - eps},

    # l7 > l6
    #{'type': 'ineq', 'fun': lambda x: x[6] - x[5] - eps},

     # (optional) l5 > l6
    {'type': 'ineq', 'fun': lambda x: x[4] - x[5] - eps},
    
    # l7 > l6  <=>  l13 > 0
    {'type': 'ineq', 'fun': lambda x: x[9] - eps},
    
    {'type': 'ineq', 'fun': lambda x: x[4] - l11 - eps},
]

# --------------- 
# reasonable bounds
# ---------------
lb = [(0.7, 20)]*8 + [(-np.pi, np.pi)]*7 + [(-5,5)]*4 + [(-10,10)]
lb[1] = (0.5, 3)   # set l2 lower bound = 0.5 mm
lb[8] = (1, 5)
lb[5] = (2.5, 5.5)

# -----------------------
# INITIAL GUESS (valid)
# -----------------------
x0 = np.zeros(20)

x0[0:8] = [
    6,   # l1
    2,  # l2
    3,  # l3
    3.09,  # l4
    11,  # l5 (increase so l7<l5)
    6,   # l6
    #9,   # l7
    #6,  # l8
    6.0,    # l10 < l11
    3 #l13
]

x0[8:15] = [1.6, 2, 3, 1.5, 2.3, 2.3, 2.5]
x0[15:19] = [0.05, 0.0066, 0.05, 0.05]
x0[19] = 0

# --------------------
# Perform solve
# --------------------
res = minimize(
    objective, x0, method='SLSQP', bounds=lb,
    constraints=cons, options={'maxiter':5000, 'ftol':1e-12, 'disp': True}
)

print("\nSUCCESS:", res.success)
print(res.message)

sol = res.x
print("\nSolution vector:")
print(sol)

print("\nResidual norm:", np.linalg.norm(residuals_array(sol)))
