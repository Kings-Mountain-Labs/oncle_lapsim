# IMPORTANT: YOU MUST RUN ../.venv/Scripts/activate, THEN RUN THIS FILE FROM Sims/

import numpy as np
import sys
import os
import io
import PIL
import plotly.graph_objects as go
import plotly.express as px


from .common.constants import *
from .cars.car_configuration import Car
from .mmd import MMD
from .steady_state_solver.iterative import Iterative_Solver

from .common.maths import sa_lut

def generate_and_plot(params):
    
    #return [1, 2]

    print("Starting...")
    
    #v_avg = 20
    v_avg = params
    
    max_beta = 30
    max_delta = 30
    size = 30
    use_lin = True
    
    car = Car(front_ackermann = "nonlinear", camber_type = "combined", aero_type="simple")
    
    #car.front_track = params[0] * IN_TO_M
    #car.rear_track = params[1] * IN_TO_M
    
    car.set_lltd()
    car.update_car()
    solver = Iterative_Solver(tangent_effects=True)
    
    
    #mmd = MMD(car, solver=solver, name=f"TW_F={params[0]:.2f} TW_R={params[1]:.2f}")
    mmd = MMD(car, solver=solver)
    
    
    mmd.mmd_sweep(v_avg, lin_space=use_lin, max_beta=max_beta, max_delta=max_delta, size=size, mu=0.65, long_g=0)
    mmd.clear_high_sa(sa_lut(v_avg))
    #fig, path = mmd.plot_mmd(pub=True, lat=3, use_name=True, save_html=True)
    #return fig, path
    fig = mmd.plot_mmd(pub=True, lat=3, use_name=True, save_html=False, show=False)
    return [mmd, fig]

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
            
    samples = np.linspace(6, 30, 28)
            
    # This shit is BROKEN and idk why
            
    pool = Pool(os.cpu_count() - 2)
    print(f"Pool established with size {os.cpu_count() - 2}")
    
    import time
    start = time.time()
    
    result = pool.map(generate_and_plot, samples)
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
    
    print("Sorting list...")
    mmds.sort(key=lambda x:x.v_avg)
    
    # GENERATE SLIDER GRAPH
    print("Generating graphs...")
    fig = go.Figure()
    fig.update_xaxes(title_text='Lat Acc (G)')
    fig.update_yaxes(title_text='Cn')
    
    vels = samples
    obj_steps = [(0, 0)] * len(vels)
    max_cn, max_ay = 15, 3
    
    for i, mmd in enumerate(mmds):
        obj_range = mmd.add_mmd(fig, f"{mmd.v_avg:.1f} m/s")
        obj_steps[i] = obj_range
        max_cn = max(max_cn, np.max(mmd.cn))
        max_ay = max(max_ay, np.max(mmd.ay)/G)
        
    for ob in fig.data[obj_steps[0][0]:obj_steps[0][1]]: ob.visible = True
    
    steps = []
    for i, m in enumerate(mmds):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label=f"{m.v_avg:.1f} m/s"
        )
        step["args"][0]["visible"][obj_steps[i][0]:obj_steps[i][1]] = [True] * (obj_steps[i][1] - obj_steps[i][0])
        steps.append(step)

    sliders = [dict(active=10, currentvalue={"prefix": "Velocity: "}, pad={"t": 50}, steps=steps)]
    
    fig.update_layout(sliders=sliders, title_text=f"2D MMD Across Velocity", height=1024, width=1024) # , template="plotly_dark"
    fig.update_xaxes(range=[-max_ay, max_ay])
    fig.update_yaxes(range=[-max_cn, max_cn])
    
    print("Saving animation...")
    start = time.time()
    
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
    
    print(f"Animation saved in {end - start}sec")
    print("Showing figure...")
    fig.show()