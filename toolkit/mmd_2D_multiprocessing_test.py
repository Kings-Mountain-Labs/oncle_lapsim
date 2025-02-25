# IMPORTANT: YOU MUST RUN .venv/Scripts/activate, THEN RUN THIS FILE FROM ./Sims/

import numpy as np
import sys
import os
import io
import PIL
import plotly.graph_objects as go
import plotly.express as px

from .common.constants import *
from .cars.car_configuration import Car
from .lap.gps import *
from .mmd import MMD
from .steady_state_solver.iterative import Iterative_Solver

from .common.maths import sa_lut, interpolate

import numbers
from copy import deepcopy

base = Car(front_ackermann='nonlinear', camber_type='combined', toe_type='complexfast', aero_type='complex')

long_g = 0
# long_g = 3.5
v_avg = 15
driver_skill = 0.75
resamples = 301

# To change values used in test cases, see generate_2d_param_array(), and modify the match/case block.
# To add a new parameter test case:
    # 1) add an entry to default_array below, and label its index for easy reference.
    # 2) add the test case name to name_array
        # If this parameter should simply be scaled by 5% as its test, stop here.
    # 3) If the test should add a specific value, or scale by a nonstandard amount, add a case in generate_2d_param_array which matches the index from step 1. 
        # case <index>:
        #   <modify temp[i] somehow here>
        #   test_array.append(temp)
        #   continue

# imports the default values from car_configuration.py 
# the first entry is used as v_avg in mmd generation.
# A comment is made specifying how the value is modified for its test, if unspecified, the parameter is scaled by 1.05.
default_array = [
    v_avg, #velocity, 0, add 1 m/s
    base.mass, # 1, add 1kg
    base.wb, # 2, add 0.5in
    base.front_track, # 3, add 0.5in
    base.rear_track, # 4, add 0.5in
    base.cg_height, # 5, add 0.5in
    base.A, # 6
    base.cl, # 7
    base.cd, # 8
    base.front_axle_downforce, # 9
    base.izz, # 10
    base.toe_front, # 11, add 0.5deg
    base.toe_rear, # 12, add 0.5deg
    base.i_a_f, # 13, add 0.5deg
    base.i_a_r, # 14, add 0.5deg
    base.k_c, # 15
    base.k_f, # 16
    base.k_r, # 17
    base.z_f, # 18
    base.z_r, # 19
    base.k_f_b, # 20
    base.k_r_b, # 21
    base.ackermann_curve, # 22, steer outside tire 0.5deg less BEING WEIRD FOR COME REASON
    base.front_camber_roll, # 23
    base.front_camber_bump, # 24 # ALSO BEING WEIRD, 23 & 25 SEEM TO BE INCORRECTLY APPLYING
    base.rear_camber_roll, # 25
    base.rear_camber_bump, # 26
    base.f_t_r, # 27
    base.f_t_b, # 28
    base.r_t_r, # 29
    base.r_t_b, # 30
    base.brake_bias # 31
]

name_array = [
    "Velocity + mod", # 0
    "Mass + mod", # 1
    "WB + .5mod", # 2
    "FT + .5mod", # 3
    "RT + .5mod", # 4
    "CGH + .5mod", # 5
    "A *s", # 6
    "Cl * s", # 7
    "Cd * s", # 8
    "Front DF *s", # 9
    "Izz *s", # 10
    "Toe F + .5mod", # 11, add 0.5deg
    "Toe R + .5mod", # 12, add 0.5deg
    "IA F + .5mod", # 13, add 0.5deg
    "IA R + .5mod", # 14, add 0.5deg
    "k_c *s", # 15
    "k_f *s", # 16
    "k_r *s", # 17
    "z_f *s", # 18
    "z_r *s", # 19
    "k_f_b *s", # 20
    "k_r_b *s", # 21
    "Outside Steer - mod", # 22, steer outside tire 0.5deg less
    "front_camber_roll + .5mod", # 23
    "front_camber_bump *s", # 24
    "rear_camber_roll + .5mod", # 25
    "rear_camber_bump *s", # 26
    "front_toe_roll *s", # 27
    "front_toe_bump *s", # 28
    "front_toe_roll *s", # 29
    "front_toe_bump *s", # 30
    "Brake Bias *s" # 31
]

def generate_2d_param_array(increase=True):
    
    mod = 1.0
    scale = 1.10
    # scale = 1.05
    
    if not increase:
        mod *= -1
        scale = 0.90
        # scale = 0.95
    test_array = [default_array]
    for i in range(len(default_array)):
        temp = deepcopy(default_array)
        match i:
            case 0 | 1:
                temp[i] += 1 * mod
                test_array.append(temp)
                continue
            case 2 | 3 | 4 | 5:
                temp[i] += 0.5 * IN_TO_M * mod
                test_array.append(temp)
                continue
            case 11 | 12 | 13 | 14 | 23 | 25:
                temp[i] += 0.5 * mod
                test_array.append(temp)
                continue
            case 22:
                ack = temp[i]
                ack[1][2] -= mod
                temp[i] = ack
                test_array.append(temp)
                continue
            # case 27 | 29:
            #     toe = temp[i]
            #     if toe[0][1] >= 0:
            #         toe[0][1] += 0.1*mod
            #     else:
            #         toe[0][1] -= 0.1*mod
            #     if toe[0][-1] >= 0:
            #         toe[0][-1] += 0.1*mod
            #     else:
            #         toe[0][-1] -= 0.1*mod
            #     temp[i] = toe
            #     test_array.append(temp)
            case 27 | 28 | 29 | 30:
                # Scale interpolate mod for toe curves
                # Assume curve is symmetric
                toe = temp[i]
                start_mod = 0.1 * mod * (abs(toe[0][1]) / toe[0][1]) # will set start_mod to negative if first toe is negative
                end_mod = 0.1 * mod * (abs(toe[0][-1]) / toe[0][-1]) # as above but with end_mod
                for j, pair in enumerate(toe):
                    add = interpolate([toe[0][0], 0, toe[-1][0]], [start_mod, 0, end_mod], pair[0])
                    toe[j][1] += add
                temp[i] = toe
                test_array.append(temp)
            case _:
                if not isinstance(temp[i], numbers.Number):
                    # if it's not a number, it's probably an array
                    temp2 = np.array(temp[i])
                    temp2 *= scale
                    temp[i] = temp2
                    test_array.append(temp)
                    continue
                else:
                    temp[i]*=scale
                    test_array.append(temp)
                    continue
    
    return test_array

def generate_and_plot(
    velocity=20, max_b=30, max_d=30, gridsize=30, use_linear=True, long_g=long_g,
    param_array=None,
    front_ackermann=base.front_ackermann,
    camber_type=base.camber_type,
    toe_type=base.toe_type,
    aero_type=base.aero_type,
    mass=base.mass,
    wb=base.wb, 
    front_track=base.front_track, 
    rear_track=base.rear_track,
    cg_height=base.cg_height,
    A=base.A,
    cl=base.cl,
    cd=base.cd,
    front_axle_downforce=base.front_axle_downforce,
    izz=base.izz,
    toe_front=base.toe_front,
    toe_rear=base.toe_rear,
    i_a_f=base.i_a_f,
    i_a_r=base.i_a_r,
    k_c=base.k_c,
    k_f=base.k_f,
    k_r=base.k_r,
    z_f=base.z_f,
    z_r=base.z_r,
    k_f_b=base.k_f_b,
    k_r_b=base.k_r_b,
    ackermann_curve=base.ackermann_curve,
    front_camber_roll=base.front_camber_roll,
    front_camber_bump=base.front_camber_bump,
    rear_camber_roll=base.rear_camber_roll,
    rear_camber_bump=base.rear_camber_bump,
    f_t_r=base.f_t_r,
    f_t_b=base.f_t_b,
    r_t_r=base.r_t_r,
    r_t_b=base.r_t_b,
    brake_bias=base.brake_bias
    ):
    
    print("Starting...")
    
    #v_avg = 20
    v_avg = velocity
    
    max_beta = max_b
    max_delta = max_d
    size = gridsize
    use_lin = use_linear
    
    car = Car(front_ackermann = "nonlinear", camber_type = "combined", aero_type="complex", toe_type="complexfast")
    
    #PARAMETER SETTING:
    
    if param_array is not None:
        v_avg = param_array[0]
        car.mass = param_array[1]
        car.wb = param_array[2]
        car.front_track = param_array[3]
        car.rear_track = param_array[4]
        car.cg_height = param_array[5]
        car.A = param_array[6]
        car.cl = param_array[7]
        car.cd = param_array[8]
        car.front_axle_downforce = param_array[9]
        car.izz = param_array[10]
        car.toe_front = param_array[11]
        car.toe_rear = param_array[12]
        car.i_a_f = param_array[13]
        car.i_a_r = param_array[14]
        car.k_c = param_array[15]
        car.k_f = param_array[16]
        car.k_r = param_array[17]
        car.z_f = param_array[18]
        car.z_r = param_array[19]
        car.k_f_b = param_array[20]
        car.k_r_b = param_array[21]
        car.ackermann_curve = param_array[22]
        car.front_camber_roll = param_array[23]
        car.front_camber_bump = param_array[24]
        car.rear_camber_roll = param_array[25]
        car.rear_camber_bump = param_array[26]
        car.f_t_r = param_array[27]
        car.f_t_b = param_array[28]
        car.r_t_r = param_array[29]
        car.r_t_b = param_array[30]
        car.brake_bias = param_array[31]
    
    else:
        car.front_ackermann=front_ackermann
        car.camber_type=camber_type
        car.toe_type=toe_type
        car.aero_type=aero_type
        car.mass=mass
        car.wb=wb
        car.front_track=front_track
        car.rear_track=rear_track
        car.cg_height=cg_height
        car.A=A
        car.cl=cl
        car.cd=cd
        car.front_axle_downforce=front_axle_downforce
        car.izz=izz
        car.toe_front=toe_front
        car.toe_rear=toe_rear
        car.i_a_f=i_a_f
        car.i_a_r=i_a_r
        car.k_c=k_c
        car.k_f=k_f
        car.k_r=k_r
        car.z_f=z_f
        car.z_r=z_r
        car.k_f_b=k_f_b
        car.k_r_b=k_r_b
        car.ackermann_curve=ackermann_curve
        car.front_camber_roll=front_camber_roll
        car.front_camber_bump=front_camber_bump
        car.rear_camber_roll=rear_camber_roll
        car.rear_camber_bump=rear_camber_bump
        car.f_t_r=f_t_r
        car.f_t_b=f_t_b
        car.r_t_r=r_t_r
        car.r_t_b=r_t_b
        car.brake_bias=brake_bias
    
    
    car.set_lltd()
    car.update_toe_geometry()
    car.update_car()
    solver = Iterative_Solver()
    
    
    #mmd = MMD(car, solver=solver, name=f"TW_F={params[0]:.2f} TW_R={params[1]:.2f}")
    mmd = MMD(car, solver=solver)
    
    
    mmd.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=long_g)
    mmd.clear_high_sa(sa_lut(v_avg))
    #fig, path = mmd.plot_mmd(pub=True, lat=3, use_name=True, save_html=True)
    #return fig, path
    fig = mmd.plot_mmd(pub=True, lat=3, use_name=True, save_html=False, show=False)
    return [mmd, fig]

def generate_and_plot_with_array(param_array):
    return generate_and_plot(param_array=param_array)

def extract_data(params):
        m = params[0]
        defaults = params[1]
        accs = m.calculate_max_acc()
        ctrls = m.calculate_important_control_values(resamples=resamples, driver_skill = driver_skill)
        stabs = m.calculate_important_control_values(resamples=resamples, driver_skill = driver_skill, mode = 'Stability')

        test_max = accs[1][0]
        test_cn = accs[1][1]
        test_ctrl_s = ctrls[0]
        test_ctrl_d = ctrls[1]
        test_stab_d = stabs[1]
        test_stab_s = stabs[0]
        
        default_max=defaults[0]
        default_cn=defaults[1]
        default_ctrl_s=defaults[2]
        default_ctrl_d=defaults[3]
        default_stab_s=defaults[4]
        default_stab_s=defaults[5]
        
        id_acc = ((test_max - default_max) / default_max * 100)
        id_cn = ((test_cn - default_cn) / default_cn * 100)
        id_ctrl_s = ((test_ctrl_s - default_ctrl_s) / default_ctrl_s * 100)
        id_ctrl_d = ((test_ctrl_d - default_ctrl_d) / default_ctrl_d * 100)
        id_stab_s = ((test_stab_s - default_stab_s) / default_stab_s * -100) # flip so bigger is better
        id_stab_d = ((test_stab_d - default_stab_d) / default_stab_d * -100)
        
        return [id_acc, id_cn, id_ctrl_s, id_ctrl_d, id_stab_s, id_stab_d]

from multiprocessing import Pool
import os

if __name__ == "__main__":
    
    #twf = np.linspace(45, 47, 4)
    #twr = np.linspace(45, 47, 4)
    #
    #samples = []
    #
    #for f in twf:
    #    for r in twr:
    #        samples.append([f, r])
            
    #samples = np.linspace(6, 30, 28)
    samples = generate_2d_param_array(increase=True)
    #for i, test in enumerate(samples):
    #    print(f"Test {i}")
    #    print(test)
    #exit()
            
    pool = Pool(os.cpu_count() - 2)
    print(f"Pool established with size {os.cpu_count() - 2}")
    
    import time
    start = time.time()
    
    result = pool.map(generate_and_plot_with_array, samples)
    print("closing pool...")
    pool.close()
    print("joining pool...")
    pool.join()
    print("end")
    
    end = time.time()
    print(f"Elapsed time: {end - start}sec")
    print(f"Average Execution Time: {(end-start)/len(samples)}sec")
    #print(end - start)
    
    mmds = []
    figs = []
    
    for test in result:
        
        mmd = test[0]
        fig = test[1]
        
        mmds.append(mmd)
        figs.append(fig)
        
        #print(test)
        #fig = test[0]
        #path = test[1]
        #fig.show()
        #fig.write_html(path)
    
    default_case = mmds[0]
    default_accs = default_case.calculate_max_acc()
    default_controls = default_case.calculate_important_control_values(resamples=resamples, driver_skill = driver_skill)
    default_stabilities = default_case.calculate_important_control_values(resamples=resamples, driver_skill = driver_skill, mode = 'Stability')
    
    default_max = default_accs[1][0]
    default_cn = default_accs[1][1]
    default_ctrl_s = default_controls[0]
    default_ctrl_d = default_controls[1]
    default_stab_s = default_stabilities[0]
    default_stab_d = default_stabilities[1]

    d_acc = []
    d_cn = []
    d_ctrl_s = []
    d_ctrl_d = []
    d_stab_s = []
    d_stab_d = []
    
    # DEBUGGING
    # print(len(mmds))
    
    for i, m in enumerate(mmds):
        if i > 0:
            print(f"Progress: {i}/{len(mmds)}")
            accs = m.calculate_max_acc()
            ctrls = m.calculate_important_control_values(resamples=resamples, driver_skill = driver_skill)
            stabs = m.calculate_important_control_values(resamples=resamples, driver_skill = driver_skill, mode = 'Stability')

            test_max = accs[1][0]
            test_cn = accs[1][1]
            test_ctrl_s = ctrls[0]
            test_ctrl_d = ctrls[1]
            test_stab_d = stabs[1]
            test_stab_s = stabs[0]
            
            d_acc.append((test_max - default_max) / default_max * 100)
            d_cn.append((test_cn - default_cn) / default_cn * 100)
            d_ctrl_s.append((test_ctrl_s - default_ctrl_s) / default_ctrl_s * 100)
            d_ctrl_d.append((test_ctrl_d - default_ctrl_d) / default_ctrl_d * 100)
            d_stab_s.append((test_stab_s - default_stab_s) / default_stab_s * -100) # flip so bigger is better
            d_stab_d.append((test_stab_d - default_stab_d) / default_stab_d * -100)
    
    print(default_stab_d)
    print(d_stab_d)
    
    # pool2 = Pool(os.cpu_count() - 2)
    # print(f"Pool established with size {os.cpu_count() - 2}")
    
    # start2 = time.time()
    
    # params = []
    # defs = [default_max, default_cn, default_ctrl_s, default_ctrl_d, default_stab_s, default_stab_d]
    # for i, m in enumerate(mmds):
    #     params.append([m, defs])
    # result2 = pool2.map(extract_data, params)
    # print("closing pool...")
    # pool2.close()
    # print("joining pool...")
    # pool2.join()
    # print("end")
    
    # end2 = time.time()
    # print(f"Elapsed time: {end2 - start2}sec")
    # print(f"Average Execution Time: {(end2-start2)/len(mmds)}sec")
    
    # for arr in result2:
    #     d_acc.append(arr[0])
    #     d_cn.append(arr[1])
    #     d_ctrl_s.append(arr[2])
    #     d_ctrl_d.append(arr[3])
    #     d_stab_s.append(arr[4])
    #     d_stab_d.append(arr[5])
    
    data = [] # data for bar chart
    cats = ['d Ay', 'd Cn', 'd Control (Straight)', 'd Control (Driver)', 'd Stability (Straight)', 'd Stability (Driver)'] # category names
    vals = [d_acc ,d_cn ,d_ctrl_s ,d_ctrl_d ,d_stab_s ,d_stab_d] # values
    for i, d in enumerate(vals):
        data.append(go.Bar(name=cats[i], x=name_array, y=d))
    
    # DEBUGGING
    # print(len(name_array))
    # print(len(data))
    # print(len(data[0]))

    fig = go.Figure(data)
    fig.update_layout(barmode='group', title_text=f'Sensitivity Analysis, {(long_g / 9.81):.2f} Gs, {(v_avg):.2f} m/s')
    
    # print(len(data[0]))
    
    fig.show()
    
    #print("Sorting list...")
    #mmds.sort(key=lambda x:x.v_avg)
    
    # GENERATE SLIDER GRAPH
    #print("Generating graphs...")
    #fig = go.Figure()
    #fig.update_xaxes(title_text='Lat Acc (G)')
    #fig.update_yaxes(title_text='Cn')
    #
    #vels = samples
    #obj_steps = [(0, 0)] * len(vels)
    #max_cn, max_ay = 15, 3
    #
    #for i, mmd in enumerate(mmds):
    #    obj_range = mmd.add_mmd(fig, f"{mmd.v_avg:.1f} m/s")
    #    obj_steps[i] = obj_range
    #    max_cn = max(max_cn, np.max(mmd.cn))
    #    max_ay = max(max_ay, np.max(mmd.ay)/G)
    #    
    #for ob in fig.data[obj_steps[0][0]:obj_steps[0][1]]: ob.visible = True
    #
    #steps = []
    #for i, m in enumerate(mmds):
    #    step = dict(
    #        method="update",
    #        args=[{"visible": [False] * len(fig.data)}],  # layout attribute
    #        label=f"{m.v_avg:.1f} m/s"
    #    )
    #    step["args"][0]["visible"][obj_steps[i][0]:obj_steps[i][1]] = [True] * (obj_steps[i][1] - obj_steps[i][0])
    #    steps.append(step)
#
    #sliders = [dict(active=10, currentvalue={"prefix": "Velocity: "}, pad={"t": 50}, steps=steps)]
    #
    #fig.update_layout(sliders=sliders, title_text=f"2D MMD Across Velocity", height=1024, width=1024) # , template="plotly_dark"
    #fig.update_xaxes(range=[-max_ay, max_ay])
    #fig.update_yaxes(range=[-max_cn, max_cn])
    #
    #print("Saving animation...")
    #start = time.time()
    
    #frames = []
    #for s, fr in enumerate(steps):
    #    # move slider to correct place
    #    fig.layout.sliders[0].update(active=s)
    #    for obj, viz in zip(fig.data, fr["args"][0]["visible"]):
    #        obj.visible = viz
    #    #print(s)
    #    #print(fr)
    #    # generate image of current state
    #    #img_bytes = fig.to_image(format="png", engine="kaleido")
    #    frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png", engine="kaleido"))))
    #
    #frames[0].save(
    #    "test.gif",
    #    save_all=True,
    #    append_images=frames[1:],
    #    optimize=True,
    #    duration=500,
    #    loop=0,
    #)
    
    end = time.time()
    
    #print(f"Animation saved in {end - start}sec")
    #print("Showing figure...")
    #fig.show()