import math
import numpy as np

def eval_delta(DC12,t, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap):
    time = t*1000000*365*86400
    DC13 = DC12*0.996

    for z_pos, z_value in enumerate(z):
        C12[z_pos] = (1-math.erf(z_value/(2*(DC12*time)**0.5)))*(C12_res-C12_cap)+C12_cap
        C13[z_pos] = (1-math.erf(z_value/(2*(DC13*time)**0.5)))*(C13_res-C13_cap)+C13_cap
    
    return ((C13/C12)/0.0112372-1)*1000

def execute_ML(Sim, delta_meas, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap):
    alpha_old = 100000000
    for _ in range(Sim):
        DC12 = np.random.uniform(low=1E-11, high=1E-10)
        t = np.random.uniform(low=0.1, high=8)
        delta = eval_delta(DC12, t, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
        alpha = (delta_meas-delta)@(delta_meas-delta)
        if alpha<alpha_old:
            D_ml = DC12
            t_ml = t
            alpha_old = alpha
    return D_ml, t_ml