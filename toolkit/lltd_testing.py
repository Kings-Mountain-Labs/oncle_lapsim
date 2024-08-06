from sim_patton import sim_qss
from car_configuration import Car
from gps_importer import *
import numpy as np
from sim_ian import sim_qts
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
import plotly.express as px
from constants import *

if __name__ == '__main__':

    total_roll_stiffness = 800 * FTLB_TO_NM #np.linspace(400, 1400, 20)
    k_c_r = np.linspace(250, 2500, 30)
    rsd_range = np.linspace(0.2, 0.8, 30)
    rsd_v, k_c_v = np.meshgrid(rsd_range, k_c_r)
    inds = np.argwhere(np.full(rsd_v.shape, True))
    lltd = np.zeros(rsd_v.shape)
    errors = np.zeros(rsd_v.shape)
    tot_time = time.time()
    car = Car()
    for index in inds:
        runtime = time.time()
        trs = total_roll_stiffness
        rsd = rsd_v[index[0], index[1]]
        k_c = k_c_v[index[0], index[1]] * FTLB_TO_NM
        # print(f"{rsd}\t{k_c}")
        car.k_f = trs * rsd
        car.k_r = trs * (1-rsd)
        car.k_c = k_c
        car.set_lltd()
        lltd[index[0], index[1]] = car.LLTD
        errors[index[0], index[1]] = car.LLTD - car.set_lltd(chassis=False)
    fig1 = px.imshow(lltd, labels=dict(x="Roll Stiffness Distribution", y="Chassis Roll Stiffness (lbf/deg)", color="LLTD"), origin='lower', x=rsd_range, y=k_c_r, aspect="auto")
    fig1.show()
    fig2 = px.imshow(errors, labels=dict(x="Roll Stiffness Distribution", y="Chassis Roll Stiffness (lbf/deg)", color="error points"), origin='lower', x=rsd_range, y=k_c_r, aspect="auto")
    fig2.show()
    fig = go.Figure()
    for i, k_c in enumerate(k_c_r):
        fig.add_trace(go.Scatter(x=rsd_range, y=lltd[i, :], mode='lines', name=f"TR={k_c}"))
    fig.update_layout(template="plotly_dark", title_text="RSD vs LLTD", xaxis_title=f"RSD % Front", yaxis_title=f"LLTD % Front")
    fig.show()