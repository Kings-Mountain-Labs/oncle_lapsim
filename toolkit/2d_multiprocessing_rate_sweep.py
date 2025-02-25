# 2D multiprocessing rate sweep
size = 30
v_avg = 15
long_g = 0
samples = np.linspace(0.95, 1.55, 3)

import numpy as np
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from .common.constants import *
from .cars.car_configuration import Car
from .mmd import MMD
from .steady_state_solver.ls_optimize import LS_Solver
from .steady_state_solver.iterative import Iterative_Solver
from .steady_state_solver.parachute import Parachute
from .common.maths import interpolate, sa_lut

from joblib import Parallel, delayed
from tqdm import tqdm

max_beta = 30
max_delta = 30
use_lin = True

# tool for graphing balance as a function of velocity. mmds must be sorted in ascending order by velocity
# def plot_accel_vs_vel(mmd_list, save_html = False, mmd_list2 = None):
#     accs = []
#     bals = []
#     saccs = []
#     sbals = []
#     vels = []
    
#     #mmds.sort(key=lambda x:x.v_avg)
#     # you must do this before calling the function
    
#     for m in mmd_list:
#         res = m.calculate_max_acc()
#         steady = res[0]
#         unsteady = res[1]
#         #steady, unsteady = m.calculate_max_acc()
#         accs.append(unsteady[0])
#         bals.append(unsteady[1])
#         saccs.append(steady[0])
#         sbals.append(steady[1])
#         vels.append(m.v_avg)
        
#     fig = go.Figure()
#     fig.update_xaxes(title_text="Velocity (m/s)", range=[min(vels), max(vels)])
#     fig.update_yaxes(title_text="Lateral Acceleration (G)", range=[-1, max(accs)+0.5])
#     fig.add_trace(
#         go.Scatter(
#             x = vels,
#             y = saccs,
#             mode ='lines',
#             marker = dict(color='blue'),
#             name = 'Steady-State'
#         ))
#     fig.add_trace(
#         go.Scatter(
#             x = vels,
#             y = accs,
#             mode ='lines',
#             marker = dict(color='LightSkyBlue'),
#             name = 'Absolute'
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x = vels,
#             y = bals,
#             mode = 'lines',
#             marker = dict(color='purple'),
#             name = 'Yaw Coefficient'
#         )
#     )
    
#     if mmd_list2 is not None:
#         print('Comparing to alt config...')
#         accs2 = []
#         bals2 = []
#         saccs2 = []
#         sbals2 = []
#         vels2 = []
#         for m in mmd_list2:
#             res = m.calculate_max_acc()
#             steady = res[0]
#             unsteady = res[1]
#             #steady, unsteady = m.calculate_max_acc()
#             accs2.append(unsteady[0])
#             bals2.append(unsteady[1])
#             saccs2.append(steady[0])
#             sbals2.append(steady[1])
#             vels2.append(m.v_avg)
#         fig.add_trace(
#             go.Scatter(
#                 x = vels2,
#                 y = saccs2,
#                 mode ='lines',
#                 marker = dict(color='red'),
#                 name = 'Steady-State 2'
#             ))
#         fig.add_trace(
#             go.Scatter(
#                 x = vels2,
#                 y = accs2,
#                 mode ='lines',
#                 marker = dict(color='pink'),
#                 name = 'Absolute 2'
#             ))
#         fig.add_trace(
#             go.Scatter(
#                 x = vels2,
#                 y = bals2,
#                 mode ='lines',
#                 marker = dict(color='DarkSlateGrey'),
#                 name = 'Balance 2'
#             ))
        
#     fig.update_layout(title_text = "Peak Grip and Balance vs Velocity")
#     fig.show()
    
#     if save_html:
#         # Ensure the directory exists
#         output_dir = r".\\MMDs"
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         import time
#         t = time.time()
#         fig.write_html(f"{output_dir}\PeakGripVsVel{t:.0f}.html")
        
# default car config
f_ack = "nonlinear"
c_type = "combined"
a_type = "complex"
t_type = "complexfast"

def gen_car(front_ackermann = f_ack, camber_type = c_type, aero_type = a_type, toe_type = t_type, front_track = 46.6*IN_TO_M, rear_track = 46*IN_TO_M, name = "", kf=1.0, kr=1.0):
    car = Car(front_ackermann = front_ackermann, camber_type = camber_type, aero_type = aero_type, toe_type = toe_type, front_track=front_track, rear_track=rear_track)
    if kf is not None:
        car.k_f_b = car.k_r_b * kf
        car.k_f = car.k_r * kf
    if kr is not None:
        car.k_r_b *= kr
        car.k_r *= kr
    car.set_lltd()
    car.update_car()
    solver = Iterative_Solver(tangent_effects = True)
    mmd = MMD(car, solver=solver, name = name)
    return car, solver, mmd
        

# Define the worker function
def process_combination(i, j, kf, kr, v, ul, mb, md, s, long):
    # print("Starting...")
    car, _, mmd = gen_car(name=f"KF: {kf}, KR: {kr}", kf=kf, kr=kr)
    # Perform operations on mmd
    mmd.mmd_sweep(v, lin_space=ul, max_beta=mb, max_delta=md, size=s, mu=0.65, long_g=long)
    mmd.clear_high_sa(sa_lut(mmd.v_avg))
    
    # Return the results (without control values calculation)
    return i, j, mmd

# Initialize arrays
mmds = np.full((len(samples), len(samples)), None)
accs = np.full((len(samples), len(samples)), None)
bals = np.full((len(samples), len(samples)), None)
stabs_s = np.full((len(samples), len(samples)), None)
stabs_d = np.full((len(samples), len(samples)), None)
ctrls_s = np.full((len(samples), len(samples)), None)
ctrls_d = np.full((len(samples), len(samples)), None)

# Prepare arguments for the worker function
args = []
for i, fb in enumerate(samples):
    for j, rb in enumerate(samples):
        args.append((i, j, fb, rb, v_avg, use_lin, max_beta, max_delta, size, long_g))

from multiprocessing import Pool
import os
pool = Pool(os.cpu_count() - 2)
print("Starting Multiprocess...")
results = pool.map(process_combination, args)
pool.close()
pool.join()
print("end")

# Run using joblib (parallel processing)
# results = Parallel(n_jobs=-2, backend="loky")(delayed(process_combination)(*arg) for arg in tqdm(args))

# Store the mmd results in the arrays (control calculations will happen later)
for result in results:
    i, j, mmd = result
    mmds[i][j] = mmd

# Calculate important control values sequentially (after parallel processing)
for i in range(len(samples)):
    for j in range(len(samples)):
        mmd = mmds[i][j]
        if mmd is not None:
            # Perform stability and control value calculations
            print(f"PROGRESS: {np.floor(100*(i*j+j)/(len(samples)**2))}%")
            print(f"{i}, {j}")
            # mmd.plot_mmd(pub=True)
            stab_s, stab_d, _, _ = mmd.calculate_important_control_values(mode="Stability")
            ctrl_s, ctrl_d, _, _ = mmd.calculate_important_control_values(mode="Control")
            a = mmd.calculate_max_acc()[1]
            print(stab_s)
            print(stab_d)
            print(ctrl_s)
            print(ctrl_d)
            print(a)
            
            # Store the calculated values in the arrays
            stabs_s[i][j] = stab_s
            stabs_d[i][j] = stab_d
            ctrls_s[i][j] = ctrl_s
            ctrls_d[i][j] = ctrl_d
            accs[i][j] = a[0]
            bals[i][j] = a[1]

print("Finished!")

X, Y = np.meshgrid(samples, samples)

res = [accs, bals, stabs_s, stabs_d, ctrls_s, ctrls_d]
titles = ["Peak Ay", "Balance", "Straightline Stability", "Cornering Stability", "Straightline Control", "Cornering Control"]

for i, axis in enumerate(res):
    print(f"Printing {titles[i]}")
    print(axis)
    heatmap = go.Figure(
        data = go.Heatmap(
            z=np.divide(axis, axis[len(axis)//2][len(axis)//2]),
            x=samples,
            y=samples,
            colorscale='Viridis'
        )
    )
    
    heatmap.update_layout(
        title=titles[i],
        xaxis_title='Front Bump Stiffness [lb/in]',
        yaxis_title='Rear Bump Stiffness [lb/in]'
    )
    
    heatmap.show()

# velocity sweep

# mmds = []

# samples = np.linspace(5, 30, 10)

# for val in samples:
#     car, _, mmd = gen_car(name = f"2D MMD at {val:.2f}")
    
#     mmd.mmd_sweep(val, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)
    
#     mmds.append(mmd)

# # %%
# # compare
# mmds2 = []
# # same samples
# for val in samples:
#     car, _, mmd = gen_car(k_r = 1.3)
#     # car, _, mmd = gen_car(front_track = 47.6*IN_TO_M, rear_track = 47*IN_TO_M)
#     mmd.mmd_sweep(val, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)
#     mmds2.append(mmd)

# # %%
# mmd_list = []
# # print(max(np.array(mmds[0].ay).flatten()))
# for m in mmds:
#     m.clear_high_sa(sa_lut(m.v_avg))
#     m.plot_mmd(pub=True, lat=2.5, use_name=True)
#     mmd_list.append(m)

# # %%
# mmd_list2 = []
# # print(max(np.array(mmds2[0].ay).flatten()))
# for m in mmds2:
#     m.clear_high_sa(sa_lut(m.v_avg))
#     m.plot_mmd(pub=True, lat=2.5)
#     mmd_list2.append(m)

# # %%
# #mmds.sort(key=lambda x:x.v_avg)
# plot_accel_vs_vel(mmd_list, mmd_list2 = mmd_list2)
# plot_accel_vs_vel(mmd_list2)

# # %%



