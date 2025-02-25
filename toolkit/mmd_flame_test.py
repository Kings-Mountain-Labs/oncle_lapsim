import numpy as np
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from .common.constants import *
from .cars.car_configuration import Car
from .mmd import MMD
from .steady_state_solver.iterative import Iterative_Solver

v_avg = 30
max_beta = 30
max_delta = 30
size = 30
use_lin = True

## MMD Parameter Sweep
#
#samples = np.linspace(-16, 14, 10)
##samples = ["equal", "simple", "combined"]
##samples = [[325, 275], [295, 239], [234.7, 221.2]]
#
#mmds = []
#
#for val in samples:
#    print("Solving for:")
#    print(val)
#    car = Car(front_ackermann = "nonlinear", camber_type = "combined")
#    car.A = 1.05
#    
#    #car.k_f = val[0] * FTLB_TO_NM
#    #car.k_r = val[1] * FTLB_TO_NM
#    #car.k_f = 275 * FTLB_TO_NM
#    #car.k_r = val * 275 * FTLB_TO_NM
#     
#    car.set_lltd()
#    car.update_car()
#    
#    solver = Iterative_Solver()
#    mmd = MMD(car, solver = solver)
#    mmd.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=val)
#    #mmd.clear_high_sa(max_sa=15)
#    mmds.append(mmd)
#
## %%
#import copy
#
#for m in mmds:
#    temp = copy.deepcopy(m)
#    temp.clear_high_sa(max_sa=15)
#    temp.plot_mmd(pub=True, lat=3)

car1 = Car(front_ackermann = "nonlinear", toe_type="complexfast")
solver1 = Iterative_Solver()

mmd1 = MMD(car1, solver = solver1)
mmd1.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)
mmd1.clear_high_sa(max_sa=15)
# solver = Parachute()

car2 = Car(front_ackermann = "nonlinear", toe_type="simple")

#car2.front_toe_bump = [[-10, 0, 10], [-20, 0, -20]]
#car2.front_toe_roll = [[-10, 0, 10], [-20, 0, -20]]
#car2.rear_toe_bump = [[-10, 0, 10], [-20, 0, -20]]
#car2.rear_toe_roll = [[-10, 0, 10], [-20, 0, -20]]

solver2 = Iterative_Solver()

mmd2 = MMD(car2, solver = solver2)
mmd2.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)
mmd2.clear_high_sa(max_sa=15)

#mmd = MMD(car, solver=solver)
#mmd.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)

#mmd.clear_high_sa(max_sa=15)

mmd1.plot_mmd(pub=True, lat=3)
mmd2.plot_mmd(pub=True, lat=3)

#car_parallel = Car(front_ackermann = "parallel")
##car_parallel.front_ackermann = "parallel"
#car_parallel.update_car()
#
#mmd_parallel = MMD(car, solver=solver)
#mmd_parallel.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)
#mmd_parallel.clear_high_sa(max_sa=15)
#mmd_parallel.plot_mmd(pub=True)
#
## %%
#mmd.plot_mmd(show_bad=True)
#
## %%
#mmd.plot_ay()
#mmd.plot_ax()
#mmd.plot_yaw()
#mmd.plot_solve_iters()
#mmd.plot_valid()
#
## %%
#mmd.plot_stability()
#mmd.plot_control_moment()
#mmd.plot_understeer_gradient()
#
## %%
#mmd.plot_sa()
#
## %%
#mmd.error
#
#
