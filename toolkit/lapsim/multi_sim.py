from .sim_qss import sim_qss
from toolkit.cars import Car
from toolkit.lap.track import *
import numpy as np
from .sim_qts import sim_qts
import plotly.graph_objects as go
import time, copy, os
from plotly.subplots import make_subplots
import plotly.express as px
from toolkit.common.constants import *
from multiprocessing import Pool
from tqdm.notebook import tqdm
from toolkit.las_solvers.las import LAS
from typing import List
import threading
  
def calculate_lap_times(car_tracks): # car_tracks is a tuple of (car, tracks) so that it can be used with pool.imap so that the progress bar works (it wont with starmap)
    car, tracks_raw, las_r, mu_corr, target, sim_type = car_tracks
    tracks = copy.deepcopy(tracks_raw)
    las: LAS = copy.deepcopy(las_r)
    las.generate_las(car, vel_bins=15, mu=mu_corr)
    lap_times = []
    for track in tracks:
        if sim_type == 'qts':
            lon, lat, omega, dt, long, vel, vel_init, ddt, critc, _, _, _, d_f, d_r, count, last_changed = sim_qts(car, track, las, target, silent=True)
        elif sim_type == 'qss':
            lon, lat, omega, dt, long, vel, vel_init, ddt, critc, _, _, _, d_f, d_r, count, last_changed = sim_qss(car, track, las, target, silent=True)
        else:
            raise ValueError(f"sim_type must be either 'qts' or 'qss', not {sim_type}")
        lap_time = 0
        tt = np.abs(np.cumsum(dt))
        if tt[-1] < 200 and np.argwhere(np.abs(omega) > 10).shape[0] ==0:
            lap_time = tt[-1]
        lap_times.append(lap_time)
    return lap_times, car, las

def calculate_skidpad(car_tracks): # car_tracks is a tuple of (car, tracks) so that it can be used with pool.imap so that the progress bar works (it wont with starmap)
    car, mu_corr, radius, sr_lim = car_tracks
    beta, delta, ay, yaw, ax, bruh, vel, lap_time = car.solve_skidpad_time(radius, mu_corr, sr_lim=sr_lim)
    return (beta, delta, ay, yaw, ax, bruh, vel, lap_time), car

class MultiSim:
    cars: List[Car]
    tracks: List[Track]
    def __init__(self, tracks, car_func, x, y, x_name, y_name):
        self.x_name, self.y_name = x_name, y_name
        self.tracks = tracks
        self.x, self.y = x, y
        self.x_v, self.y_v = np.meshgrid(x, y)
        self.inds = np.argwhere(np.full(self.x_v.shape, True))
        # inds = [[0, 0]]
        self.lap_times = []
        for i in range(len(self.tracks)):
            self.lap_times.append(np.zeros(self.x_v.shape))
        self.lltd = np.zeros(self.x_v.shape)
        self.lltd_diff = np.zeros(self.x_v.shape)
        self.lat_acc, self.yaw_acc, self.acc_acc, self.dec_acc, self.pos_name, self.pos_vel, self.pos_label, self.sa1, self.sa2 = [], [], [], [], [], [], [], [], []
        self.cars = []
        for index in self.inds:
            self.cars.append(car_func(self.x_v[index[0], index[1]], self.y_v[index[0], index[1]]))

    def run_lltd(self):
        if np.sum(self.lltd) != 0:
            return
        for i, index in enumerate(self.inds):
            car = self.cars[i]
            chassis = car.set_lltd(chassis=False)
            self.lltd[index[0], index[1]] = car.set_lltd(chassis=True)
            self.lltd_diff[index[0], index[1]] = car.LLTD - chassis

    def run_sim(self, las: LAS, mu=0.65, convergence_target=0.01, sim_type="qts"):
        self.run_lltd()
        tot_time = time.time()
        self.lap_times_raw = []
        self.new_cars = []
        self.new_las = []

        lap_time, car, las = calculate_lap_times((self.cars[0], self.tracks, las, mu, convergence_target, sim_type))

        print(f"Generating LAS & Simulating {len(self.tracks)} tracks with {len(self.cars)} cars:")
        with Pool(os.cpu_count() -2) as p: # -2 because I dont wanna freeze my computer
            for ret_val in tqdm(p.imap(calculate_lap_times, [(car, self.tracks, las, mu, convergence_target, sim_type) for car in self.cars]), total=len(self.cars)):
                lap_time, car, las = ret_val
                self.lap_times_raw.append(lap_time)
                self.new_cars.append(car)
                self.new_las.append(las)
        self.cars = self.new_cars
        
        for i, index in enumerate(self.inds):
            car = self.cars[i]
            las = self.new_las[i]
            self.lat_acc.append(las.aymax)
            self.yaw_acc.append(las.yawmax)
            self.acc_acc.append(las.longAcc_forward)
            self.dec_acc.append(las.longAcc_reverse)
            self.sa1.append(las.aymax_sa)
            self.sa2.append(las.yawmax_sa)
            self.pos_name.append([f"{car.description} vel:{x:.2f}" for x in las.vels])
            self.pos_vel.append(las.vels)
            self.pos_label.append(car.description)
            for j in range(len(self.tracks)):
                self.lap_times[j][index[0], index[1]] = self.lap_times_raw[i][j]

        print(f"Total time for sim: {(time.time() - tot_time):.3f}")

    def run_skidpad(self, mu=0.65, radius=(15.25 / 2 + 1), sr_lim=0.2):
        self.run_lltd()
        tot_time = time.time()
        self.skidpad_times_raw = []
        self.beta_raw, self.delta_raw, self.ay_raw, self.vel_raw = [], [], [], []
        self.bruh_raw = []
        self.new_cars = []

        dark_green = "\033[1;32;40m"
        bar_format = f"{dark_green}{{l_bar}}{{bar:50}} [{{elapsed}}]{{r_bar}}"


        print(f"Generating LAS & Simulating {len(self.tracks)} tracks with {len(self.cars)} cars:")
        with Pool(os.cpu_count() -2) as p: # -2 because I dont wanna freeze my computer
            for ret_val in tqdm(p.imap(calculate_skidpad, [(car, mu, radius, sr_lim) for car in self.cars]), total=len(self.cars), bar_format=bar_format):
                lap_info, car = ret_val
                beta, delta, ay, yaw, ax, bruh, vel, lap_time = lap_info
                self.skidpad_times_raw.append(lap_time)
                self.beta_raw.append(beta)
                self.delta_raw.append(delta)
                self.ay_raw.append(ay)
                self.vel_raw.append(vel)
                self.new_cars.append(car)
                self.bruh_raw.append(bruh)
        self.cars = self.new_cars
        self.skidpad_times = np.zeros(self.x_v.shape)
        self.skidpad_beta = np.zeros(self.x_v.shape)
        self.skidpad_delta = np.zeros(self.x_v.shape)
        self.skidpad_ay = np.zeros(self.x_v.shape)
        self.skidpad_vel = np.zeros(self.x_v.shape)
        self.skidpad_bruh = np.zeros(self.x_v.shape)
        for i, index in enumerate(self.inds):
            car = self.cars[i]
            self.skidpad_times[index[0], index[1]] = self.skidpad_times_raw[i]
            self.skidpad_beta[index[0], index[1]] = self.beta_raw[i]
            self.skidpad_delta[index[0], index[1]] = self.delta_raw[i]
            self.skidpad_ay[index[0], index[1]] = self.ay_raw[i]
            self.skidpad_vel[index[0], index[1]] = self.vel_raw[i]
            self.skidpad_bruh[index[0], index[1]] = self.bruh_raw[i]
        print(f"Total time for sim: {(time.time() - tot_time):.3f}")

    def plot_skidpad(self):
        fig = px.imshow(self.skidpad_times, labels=dict(x=self.x_name, y=self.y_name, color=f"Skidpad Time (s)"), origin='lower', x=self.x, y=self.y, aspect="auto")
        # put a point on the minimum
        min_index = np.unravel_index(np.argmin(self.skidpad_times, axis=None), self.skidpad_times.shape)
        fig.add_trace(go.Scatter(x=np.array([self.x_v[min_index]]), y=np.array([self.y_v[min_index]]), mode="markers", marker=dict(color="red", size=4)))
        fig.update_layout(template="plotly_dark")
        fig.show()
        fig = px.imshow(np.rad2deg(self.skidpad_beta), labels=dict(x=self.x_name, y=self.y_name, color=f"Skidpad Beta (deg)"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig.update_layout(template="plotly_dark")
        fig.show()
        fig = px.imshow(np.rad2deg(self.skidpad_delta), labels=dict(x=self.x_name, y=self.y_name, color=f"Skidpad Delta (deg)"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig.update_layout(template="plotly_dark")
        fig.show()
        fig = px.imshow(self.skidpad_ay, labels=dict(x=self.x_name, y=self.y_name, color=f"Skidpad Ay (m/s^2)"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig.update_layout(template="plotly_dark")
        fig.show()
        fig = px.imshow(self.skidpad_vel, labels=dict(x=self.x_name, y=self.y_name, color=f"Skidpad Vel (m/s)"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig.update_layout(template="plotly_dark")
        fig.show()
        fig = px.imshow(self.skidpad_bruh, labels=dict(x=self.x_name, y=self.y_name, color=f"Skidpad Bruh (m/s^2)"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig.update_layout(template="plotly_dark")
        fig.show()

    def plot_tracks(self):
        for j, track in enumerate(self.tracks):
            fig = px.imshow(self.lap_times[j], labels=dict(x=self.x_name, y=self.y_name, color=f"Lap {j} Time (s)"), origin='lower', x=self.x, y=self.y, aspect="auto")
            fig.update_layout(template="plotly_dark")
            fig.show()
        
        fig1 = px.imshow(self.lltd, labels=dict(x=self.x_name, y=self.y_name, color="LLTD"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig1.update_layout(template="plotly_dark")
        fig1.show()
        fig2 = px.imshow(self.lltd_diff * 100, labels=dict(x=self.x_name, y=self.y_name, color="LLTD error from stiff chassis in percent"), origin='lower', x=self.x, y=self.y, aspect="auto")
        fig2.update_layout(template="plotly_dark")
        fig2.show()

    def plot_LAS_corners(self):
        fig4 = make_subplots(rows=3, cols=2)
        for ind in range(len(self.lat_acc)):
            fig4.add_trace(go.Scatter(x=self.lat_acc[ind][:, 1], y=self.lat_acc[ind][:, 2], text=self.pos_name[ind], marker_color=self.pos_vel[ind], legendgroup=f"group{ind}", showlegend=False), row=1, col=1)
            fig4.add_trace(go.Scatter(x=self.yaw_acc[ind][:, 1], y=self.yaw_acc[ind][:, 2], text=self.pos_name[ind], marker_color=self.pos_vel[ind], legendgroup=f"group{ind}", showlegend=False), row=1, col=2)
            fig4.add_trace(go.Scatter(x=self.sa1[ind][:, 0], y=self.sa1[ind][:, 1], text=self.pos_name[ind], marker_color=self.pos_vel[ind], legendgroup=f"group{ind}", showlegend=False), row=2, col=1)
            fig4.add_trace(go.Scatter(x=self.sa2[ind][:, 0], y=self.sa2[ind][:, 1], text=self.pos_name[ind], marker_color=self.pos_vel[ind], legendgroup=f"group{ind}", showlegend=False), row=2, col=2)
            fig4.add_trace(go.Scatter(x=self.pos_vel[ind], y=self.acc_acc[ind][:, 0], text=self.pos_name[ind], marker_color=self.pos_vel[ind], legendgroup=f"group{ind}", showlegend=False), row=3, col=1)
            fig4.add_trace(go.Scatter(x=self.pos_vel[ind], y=self.dec_acc[ind][:, 0], text=self.pos_name[ind], marker_color=self.pos_vel[ind], legendgroup=f"group{ind}", showlegend=True, name=self.pos_label[ind]), row=3, col=2)
        fig4.update_yaxes(title_text='Yaw Acc (rad/s^2)', row=1, col=1)
        fig4.update_xaxes(title_text='Acc (m/s^2)', row=1, col=1)
        fig4.update_yaxes(title_text='Yaw Acc (rad/s^2)', row=1, col=2)
        fig4.update_xaxes(title_text='Acc (m/s^2)', row=1, col=2)
        fig4.update_xaxes(title_text='Delta (deg)', row=2, col=1)
        fig4.update_xaxes(title_text='Delta (deg)', row=2, col=2)
        fig4.update_yaxes(title_text='Beta (deg)', row=2, col=1)
        fig4.update_yaxes(title_text='Beta (deg)', row=2, col=2)
        fig4.update_yaxes(title_text='Acc (m/s^2)', row=3, col=1)
        fig4.update_yaxes(title_text='Acc (m/s^2)', row=3, col=2)
        fig4.update_xaxes(title_text='Vel (m/s)', row=3, col=1)
        fig4.update_xaxes(title_text='Vel (m/s)', row=3, col=2)
        fig4.update_layout(template="plotly_dark")
        fig4.show()

    def plot_LLTD(self):
        fig5 = make_subplots(rows=2, cols=2)
        for ind in range(len(self.pos_vel[0])):
            ay_max1 = np.array([self.lat_acc[i][ind, 1] for i in range(len(self.lat_acc))])
            yaw_max1 = np.array([self.lat_acc[i][ind, 2] for i in range(len(self.lat_acc))])
            ay_max2 = np.array([self.yaw_acc[i][ind, 1] for i in range(len(self.yaw_acc))])
            yaw_max2 = np.array([self.yaw_acc[i][ind, 2] for i in range(len(self.yaw_acc))])
            vels_list = np.array([self.pos_vel[i][ind] for i in range(len(self.pos_vel))])
            lltd_list = np.array([self.lltd[self.inds[i, 0], self.inds[i, 1]] for i in range(self.inds.shape[0])])
            inds_list = np.array([self.inds[i, 0] for i in range(self.inds.shape[0])])
            for i in range(self.x_v.shape[0]):
                tr_inds = (inds_list == i)
                name = f"{self.pos_vel[0][ind]:.2f} m/s {self.y_name}: {self.y[i]:.2f}"
                fig5.add_trace(go.Scatter(x=lltd_list[tr_inds], y=ay_max1[tr_inds], name=name, marker_color=vels_list, legendgroup=f"group{ind}_{i}", showlegend=False), row=1, col=1)
                fig5.add_trace(go.Scatter(x=lltd_list[tr_inds], y=yaw_max1[tr_inds], name=name, marker_color=vels_list, legendgroup=f"group{ind}_{i}", showlegend=False), row=1, col=2)
                fig5.add_trace(go.Scatter(x=lltd_list[tr_inds], y=ay_max2[tr_inds], name=name, marker_color=vels_list, legendgroup=f"group{ind}_{i}", showlegend=False), row=2, col=1)
                fig5.add_trace(go.Scatter(x=lltd_list[tr_inds], y=yaw_max2[tr_inds], name=name, marker_color=vels_list, legendgroup=f"group{ind}_{i}", showlegend=True), row=2, col=2)
        fig5.update_yaxes(title_text='Max Lateral Acc (m/s^2)', row=1, col=1)
        fig5.update_xaxes(title_text='LLTD', row=1, col=1)
        fig5.update_yaxes(title_text='Max Yaw Acc (rad/s^2)', row=1, col=2)
        fig5.update_xaxes(title_text='LLTD', row=1, col=2)
        fig5.update_yaxes(title_text='Max Lateral Acc (m/s^2)', row=2, col=1)
        fig5.update_xaxes(title_text='LLTD', row=2, col=1)
        fig5.update_yaxes(title_text='Max Yaw Acc (rad/s^2)', row=2, col=2)
        fig5.update_xaxes(title_text='LLTD', row=2, col=2)
        fig5.update_layout(template="plotly_dark")
        fig5.show()