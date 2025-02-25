from toolkit.lapsim.sim_qss import sim_qss
from toolkit.lapsim.sim_qts import sim_qts
from toolkit.cars.car_configuration import Car, MMD_3D_Graphs
from toolkit.lap.track import *
from toolkit.las_solvers.las import LAS
import numpy as np
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
import plotly.express as px
from toolkit.common.constants import *
from scipy.spatial import KDTree
from typing import List

def find_closest_kd(tree: KDTree, points: np.ndarray, ax: float, ay: float, yaw: float) -> tuple[float, float, float]:
    """
    Finds the closest beta and delta to the given ax, ay, yaw and vel
    :param LAS_points: List of LAS points
    :param ax: Longitudinal acceleration
    :param ay: Lateral acceleration
    :param yaw: Yaw rate
    :param vel: Velocity
    :return: The closest beta and delta
    """
    targets = np.array([ay, yaw, ax]).T
    # # do a weighted average of the closest 5 points
    dist, ind = tree.query(targets)
    return points[ind-1, 0], points[ind-1, 1], dist

def find_closest_delta_beta(LAS_points, ax, ay, yaw, vel, vels):
    """
    Finds the closest beta and delta to the given ax, ay, yaw and vel
    :param LAS_points: List of LAS points
    :param ax: Longitudinal acceleration
    :param ay: Lateral acceleration
    :param yaw: Yaw rate
    :param vel: Velocity
    :return: The closest beta and delta
    """
    trees: List[KDTree] = []
    points = []
    for i in range(len(vels)):
        (vel_ind,) = np.where(LAS_points[:, 5] == vels[i])
        _, good_ind = np.unique(LAS_points[vel_ind, 2:5], axis=0, return_index=True)
        points.append(LAS_points[vel_ind[good_ind], :])
        trees.append(KDTree(LAS_points[vel_ind[good_ind], 2:5]))
    delta, beta, dist = np.zeros(len(ax)), np.zeros(len(ax)), np.zeros(len(ax))
    for i, v in enumerate(vel):
        if len(np.argwhere(vels > v)) == 0:
            vel_up = np.argmin(vels)
        else:
            (vel_up,) = np.argwhere(vels > v)[np.argmin(vels[vels > v])]
        if len(np.argwhere(vels < v)) == 0:
            vel_down = np.argmax(vels)
        else:
            (vel_down,) = np.argwhere(vels < v)[np.argmax(vels[vels < v])]
        delta_up, beta_up, dist_up = find_closest_kd(trees[vel_up], points[vel_up], ax[i], ay[i], yaw[i])
        delta_down, beta_down, dist_down = find_closest_kd(trees[vel_down], points[vel_down], ax[i], ay[i], yaw[i])
        weights = np.array([vels[vel_up] - v, v - vels[vel_down]]) / (vels[vel_up] - vels[vel_down])
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        delta[i] = (weights[0] * delta_up + weights[1] * delta_down)
        beta[i] = (weights[0] * beta_up + weights[1] * beta_down)
        dist[i] = weights[0] * dist_up + weights[1] * dist_down
    return delta, beta, dist


class RunSim():
    def __init__(self, track: Track, car: Car, las: LAS, debug=False):
        """
        Runs a simulation of the car on the track and generates plots comparing the simulation to the real data
        :param track: Track object
        :param car: Car object
        :param convergence_target: Convergence target for the simulation
        :param mu: Friction scaling coefficient
        :param v_average: Average velocity of the car, only used for the patton simulation
        :param debug: If true, shows debug information
        :param sim_type: Which simulation to use, either 'ian' or 'patton'
        """
        self.track = track
        self.car: Car = car
        self.las: LAS = las
        self.debug = debug
        self.mmd3d_gen = False
        self.use_beta_init = False

    def simulate(self, sim_type='qts', convergence_target=0.001, mu=1.0, v_average=15, bins=10):
        if sim_type == 'qss':
            if self.las.set_las is None or self.las.set_las != (mu, bins):
                self.las.generate_las(self.car, vel_bins=bins, mu=mu)
            beta_init = np.zeros(self.track.k.shape[0])
            if self.use_beta_init:
                beta_init = self.track.real_beta
            self.lon, self.lat, self.omega_dot, self.dt, self.long, self.vel, self.vel_init, self.ddt, self.critc, self.iters, self.delta, self.beta, self.d_f, self.d_r, self.count, self.last_changed = sim_qss(self.car, self.track, self.las, convergence_target, true_beta=beta_init)
        elif sim_type == 'qts':
            if self.las.set_las is None or self.las.set_las != (mu, bins):
                self.las.generate_las(self.car, vel_bins=bins, mu=mu)
            self.lon, self.lat, self.omega_dot, self.dt, self.long, self.vel, self.vel_init, self.ddt, self.critc, self.iters, self.delta, self.beta, self.d_f, self.d_r, self.count, self.last_changed = sim_qts(self.car, self.track, self.las, convergence_target)
        else:
            raise ValueError(f"Sim type {sim_type} not recognized")
        
        self.mu = mu
        self.v_average = v_average
        self.convergence_target = convergence_target
        self.sim_type = sim_type
        self.bins = bins
        self.calc_channels()

    def calc_channels(self):
        self.tt = np.abs(np.cumsum(self.dt))
        self.calc_vel = np.cumsum(self.ddt * self.long)
        self.omega = self.vel * np.interp(self.track.u_crit, self.track.u, self.track.k)
        self.diff_time = np.cumsum(self.ddt) - np.interp(self.track.u_crit, self.track.smooth_gps.dist, self.track.smooth_gps.time)
        self.dt_dx = np.zeros(self.track.u_crit.shape[0])
        self.dt_dx[1:] = self.ddt[1:] - np.diff(np.interp(self.track.u_crit, self.track.smooth_gps.dist, self.track.smooth_gps.time))
        self.tt_t = np.interp(self.track.u, self.track.u_crit, self.tt)
        self.generate_power_curve()

    def generate_power_curve(self, regen=False):
        # this is currently a very lazy way of doing this, it does not account for drag or rolling resistance
        self.motor_power = np.zeros(self.vel.shape[0])
        power_draw = self.lon * self.car.mass * self.vel
        self.motor_power[power_draw > 0] = power_draw[power_draw > 0]
        self.energy = np.cumsum(self.motor_power * self.dt)

    def plot(self):
        self.plot_mmd2d()
        self.plot_vs(distance=True)
        self.plot_vs(distance=False)
        self.plot_vs_vel()
        self.plot_LAS_lims()

    def generate_MMD3D(self):
        if self.mmd3d_gen:
            return
        else:
            self.mmd3d_gen = True
            start = time.time()
            self.ay_it1, self.yaw_it1, _, self.ax_it1, self.delta_1, self.beta_1, self.kappa_1, self.valid_1, self.LAS_point_raw = self.las.MMD_3D_Long(mu_corr=self.mu)
            print(f"Time: {(time.time() - start):.2f}")

    def generate_LAS_points(self, use_mmd3d=False):
        if use_mmd3d:
            self.generate_MMD3D()
        if self.mmd3d_gen:
            self.LAS_points = np.array(self.LAS_point_raw)
            LAS_points = np.concatenate((np.array(self.las.vps), np.array(self.las.vps)))
            LAS_points[len(self.las.vps):, :4] *= -1
            self.LAS_points = np.concatenate((self.LAS_points, LAS_points), axis=0)
        else:
            self.LAS_points = np.concatenate((np.array(self.las.vps), np.array(self.las.vps)))
            self.LAS_points[len(self.las.vps):, :4] *= -1

    def generate_delta_beta(self, use_mmd3d=False):
        self.generate_LAS_points(use_mmd3d=use_mmd3d)
        # self.delta_est, self.beta_est, self.dist = find_closest_delta_beta(self.LAS_points, self.lon, self.lat, self.omega_dot, self.vel)
        self.delta_est, self.beta_est, self.dist = find_closest_delta_beta(self.LAS_points, self.lon, self.lat, self.omega_dot, self.vel, self.las.vels)

    def plot_mmd2d(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Lat_Acc v Long_Acc", "Long_Acc v Yaw_Acc", "Lat_Acc v Long_Acc", "Map with Vel (m/s)"))
        color1 = px.colors.qualitative.Light24[0]
        color2 = px.colors.qualitative.Light24[1]
        color3 = px.colors.qualitative.Light24[2]
        fig.add_trace(go.Scattergl(x=self.lat, y=self.lon, mode='lines', name="Sim", marker=dict(color=color1), legendgroup=f"group1", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=self.track.get_ch("__acc_y"), y=self.track.get_ch("__acc_x"), mode='lines', name="Real_Smoothed", marker=dict(color=color3), legendgroup=f"group3", showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text='Lat Acc (m/s^2)', row=1, col=1)
        fig.update_yaxes(title_text='Long Acc (m/s^2)', row=1, col=1)

        fig.add_trace(go.Scattergl(x=self.lon, y=self.omega_dot, mode='lines', name="Sim", marker=dict(color=color1), legendgroup=f"group1", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scattergl(x=self.track.get_ch("__acc_x"), y=self.track.get_ch("__yacc"), mode='lines', name="Real_Smoothed", marker=dict(color=color3), legendgroup=f"group3", showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text='Long Acc (m/s^2)', row=1, col=2)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=1, col=2)

        ay_it2, yaw_it2, _, delta_2, beta_2 = self.car.MMD(self.v_average, mu_corr=self.mu)
        fig.add_trace(go.Scattergl(x=self.lat, y=self.omega_dot, mode='lines', name="Sim", marker=dict(color=color1), legendgroup=f"group1", showlegend=True), row=2, col=1)
        fig.add_trace(go.Scattergl(x=self.track.get_ch("__acc_y"), y=self.track.get_ch("__yacc"), mode='lines', name="Real_Smoothed", marker=dict(color=color3), legendgroup=f"group3", showlegend=True), row=2, col=1)
        fig.update_xaxes(title_text='Lat Acc (m/s^2)', row=2, col=1)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=2, col=1)
        for i, betax in enumerate(beta_2):
            valid = ay_it2[i, :] != 0
            fig.add_trace(
                go.Scattergl(
                    x=ay_it2[i, valid],
                    y=yaw_it2[i, valid],
                    mode='lines',
                    hovertext="β={:.2f}°".format(np.rad2deg(betax)),
                    marker=dict(color='red'), legendgroup=f"group2", showlegend=False
                ),
                row=2, col=1)
        for j, deltax in enumerate(delta_2):
            valid = ay_it2[:, j] != 0
            fig.add_trace(
                go.Scattergl(
                    x=ay_it2[valid, j],
                    y=yaw_it2[valid, j],
                    mode='lines',
                    hovertext="{:.1f}° Delta".format(np.rad2deg(deltax)),
                    marker=dict(color='blue'), name="MMD", legendgroup=f"group2", showlegend=(j == 0)
                ),
                row=2, col=1)
        
        # fig.add_trace(go.Scattergl(x=np.interp(self.track.u_crit, self.track.u, self.track.x_ss), y=np.interp(self.track.u_crit, self.track.u, self.track.y_ss), mode='markers', name="Sim", marker=dict(color=self.vel/self.vel.max()), legendgroup=f"group1", showlegend=False), row=2, col=2)
        # fig.add_trace(go.Scattergl(x=self.track.x_out_raw, y=self.track.y_out_raw, mode='markers', name="Real", marker=dict(color=self.track.vel/self.vel.max()), legendgroup=f"group3", showlegend=False), row=2, col=2)
        # fig.add_trace(go.Scattergl(x=np.interp(self.track.u_crit, self.track.u, self.track.x_ss), y=np.interp(self.track.u_crit, self.track.u, self.track.y_ss), mode='markers', name="Time Diff", marker=dict(color=np.clip(self.ddt, -0.075, 0.075)), legendgroup=f"group4", showlegend=True), row=2, col=2)
        # fig.add_trace(go.Scattergl(x=self.track.x_ss, y=self.track.y_ss, mode='markers', name="Beta Angle", marker=dict(color=self.track.real_beta), legendgroup=f"group5", showlegend=True), row=2, col=2)
        fig.update_xaxes(title_text='East-West (m)', row=2, col=2)
        fig.update_yaxes(title_text='North-South (m)', row=2, col=2)

        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()

    def map_plot(self):
        px.set_mapbox_access_token(open(".mapbox_token").read())
        fig = px.scatter_mapbox(lat=self.track.lat_ss, lon=self.track.lon_ss, color=self.track.real_beta, color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
        fig.show()

    def plot_mmd2d_LAS_err(self):
        fig = make_subplots(rows=3, cols=2, subplot_titles=("Lat_Acc v Long_Acc", "Long_Acc v Yaw_Acc", "Lat_Acc v Long_Acc", "None"))
        f_ind = np.abs(self.d_f) > np.abs(self.d_f).mean()
        r_ind = np.abs(self.d_r) > np.abs(self.d_r).mean()
        fig.add_trace(go.Scattergl(x=self.lat[f_ind], y=self.lon[f_ind], mode='markers', name="Front", marker=dict(color=self.d_f[f_ind]), legendgroup=f"group1", showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text='Lat Acc (m/s^2)', row=1, col=1)
        fig.update_yaxes(title_text='Long Acc (m/s^2)', row=1, col=1)
        fig.add_trace(go.Scattergl(x=self.lon[f_ind], y=self.omega_dot[f_ind], mode='markers', name="Front", marker=dict(color=self.d_f[f_ind]), legendgroup=f"group1", showlegend=False), row=2, col=1)
        fig.update_xaxes(title_text='Long Acc (m/s^2)', row=2, col=1)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=2, col=1)
        fig.add_trace(go.Scattergl(x=self.lat[r_ind], y=self.lon[r_ind], mode='markers', name="Rear", marker=dict(color=self.d_r[r_ind]), legendgroup=f"group2", showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text='Lat Acc (m/s^2)', row=1, col=2)
        fig.update_yaxes(title_text='Long Acc (m/s^2)', row=1, col=2)
        fig.add_trace(go.Scattergl(x=self.lon[r_ind], y=self.omega_dot[r_ind], mode='markers', name="Rear", marker=dict(color=self.d_r[r_ind]), legendgroup=f"group2", showlegend=False), row=2, col=2)
        fig.update_xaxes(title_text='Long Acc (m/s^2)', row=2, col=2)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=2, col=2)

        fig.add_trace(go.Scattergl(x=self.lat[f_ind], y=self.omega_dot[f_ind], mode='markers', name="Front", marker=dict(color=self.d_f[f_ind]), legendgroup=f"group1", showlegend=True), row=3, col=1)
        fig.update_xaxes(title_text='Lat Acc (m/s^2)', row=3, col=1)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=3, col=1)
        fig.add_trace(go.Scattergl(x=self.lat[r_ind], y=self.omega_dot[r_ind], mode='markers', name="Rear", marker=dict(color=self.d_r[r_ind]), legendgroup=f"group2", showlegend=True), row=3, col=2)
        fig.update_xaxes(title_text='Lat Acc (m/s^2)', row=3, col=2)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=3, col=2)
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()

    def plot_vs(self, distance=True, debug=False, yaw_rate=True, yaw_acc=True, separate_acc=True, angles=True, separate_angles=False, fz=True, curvature=False, vel_limit=True, delta_beta_est=False, weight_transfer=False, fz_est=False, yaw_rate_real=False, power_draw=False):
        if distance:
            b, c = self.track.u_crit, self.track.u
            x_label = "Distance (m)"
        else:
            b, c = self.tt, self.tt_t
            x_label = "Time (s)"

        if delta_beta_est:
            self.generate_delta_beta()

        # Velocity + yaw rate + yaw acc + lateral acc + longitudinal acc + angles + separated angles + delta time + ddt + power_draw + curvature + fz
        rows = 1 + int(yaw_rate) + int(yaw_acc) + 1 + int(separate_acc) + int(angles) + int(angles) * int(separate_angles) + 1 + 1 + (1 * int(power_draw)) + int(curvature) + int(fz)
        row = 1
        fig = make_subplots(rows=rows, shared_xaxes=True, vertical_spacing=0.02)

        fig.add_trace(go.Scattergl(x=b, y=self.vel, mode='lines', name="Sim Velocity", legendgroup=f"group1", showlegend=True), row=row, col=1)
        fig.add_trace(self.track.get_channel_go("__gps_vel", distance, group=f"group2", legend=True), row=row, col=1)
        if debug:
            fig.add_trace(go.Scattergl(x=b, y=self.calc_vel, mode='lines', name="Calc Velocity", legendgroup=f"group3", showlegend=True), row=row, col=1)
        fig.add_trace(go.Scattergl(x=b, y=self.vel_init, mode='lines', name="Vel Limit", legendgroup=f"group4", showlegend=True), row=row, col=1)
        if vel_limit:
            fig.add_trace(go.Scattergl(x=c, y=self.track.vc, mode='lines', name="Vel Limit Curve", legendgroup=f"group5", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=c, y=self.track.vc_r, mode='lines', name="Vel Limit Rate", legendgroup=f"group6", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b[self.critc], y=self.vel[self.critc], mode='markers', name="Velocity Crit", legendgroup=f"group7", showlegend=True), row=row, col=1)
        if debug:
            fig.add_trace(go.Scattergl(x=b, y=(np.interp(b, a, self.track.vel) - self.vel), mode='lines', name="Velocity Diff", legendgroup=f"group8", showlegend=True), row=row, col=1)
        fig.update_yaxes(title_text='Velocity (m/s)', range=[-5, 30], row=row, col=1)

        if yaw_acc:
            row += 1
            fig.add_trace(go.Scattergl(x=b, y=self.omega_dot, mode='lines', name="Yaw Acc Sim", legendgroup=f"group1", showlegend=False), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__yacc", distance, group=f"group2", legend=False), row=row, col=1)
            fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', range=[-10, 10], row=row, col=1)

        if yaw_rate:
            row += 1
            fig.add_trace(go.Scattergl(x=b, y=self.omega, mode='lines', name="Yaw Rate Sim", legendgroup=f"group1", showlegend=False), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__gyro_z", distance, group=f"group2", legend=False), row=row, col=1)
            if yaw_rate_real:
                fig.add_trace(go.Scattergl(x=c, y=self.track.y_r_k, mode='lines', name="Yaw Rate Real Curvature", legendgroup=f"group111", showlegend=True), row=row, col=1)
                fig.add_trace(go.Scattergl(x=d, y=self.track.y_r_b, mode='lines', name="Yaw Rate Real Beta Contribution", legendgroup=f"group112", showlegend=True), row=row, col=1)
            if debug:
                fig.add_trace(go.Scattergl(x=b, y=(np.interp(b, d, self.track.y_r) - self.omega), mode='lines', name="Yaw Rate Diff", legendgroup=f"group8", showlegend=False), row=row, col=1)
            fig.update_yaxes(title_text='Yaw Rate (rad/sec^2)', range=[-4, 4], row=row, col=1)

        if angles:
            row += 1
            # fig.add_trace(go.Scattergl(x=c, y=self.track.real_beta, mode='lines', name="Beta Angle", legendgroup=f"group113", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b, y=self.beta, mode='lines', name="Beta Angle Sim", legendgroup=f"group118", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b, y=self.delta, mode='lines', name="Delta Angle Sim", legendgroup=f"group119", showlegend=True), row=row, col=1)
            if delta_beta_est:
                fig.add_trace(go.Scattergl(x=b, y=-self.beta_est, mode='lines', name="Beta Angle Est", legendgroup=f"group163", showlegend=True), row=row, col=1)
                fig.add_trace(go.Scattergl(x=b, y=-self.delta_est, mode='lines', name="Delta Angle Est", legendgroup=f"group173", showlegend=True), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__steering_angle", distance, group=f"group117", legend=True), row=row, col=1)
            if separate_angles:
                fig.update_yaxes(title_text='Angle (deg)', range=[-35, 35], row=row, col=1)
                row += 1
                fig.update_yaxes(title_text='Angle (deg)', range=[-200, 200], row=row, col=1)
            else:
                fig.update_yaxes(title_text='Angle (deg)', range=[-200, 200], row=row, col=1)
            fig.add_trace(go.Scattergl(x=c, y=self.track.angle, mode='lines', name="Track Angle", legendgroup=f"group153", showlegend=True), row=row, col=1)
            # fig.add_trace(go.Scattergl(x=d, y=np.rad2deg(self.track.real_angle), mode='lines', name="Real Angle", legendgroup=f"group115", showlegend=True), row=row, col=1)

        row += 1
        fig.add_trace(go.Scattergl(x=b, y=self.lat, mode='lines', name="Lat Acc Sim", legendgroup=f"group1", showlegend=False), row=row, col=1)
        fig.add_trace(self.track.get_channel_go("__acc_y", distance, group=f"group2", legend=False), row=row, col=1)
        if debug:
            fig.add_trace(go.Scattergl(x=b, y=(np.interp(b, d, self.track.lat_acc) - self.lat), mode='lines', name="Lat Acc Diff", legendgroup=f"group8", showlegend=False), row=row, col=1)
        if separate_acc:
            fig.update_yaxes(title_text='Acceleration Lat (m/s^2)', range=[-20, 20], row=row, col=1)
            row += 1
            fig.update_yaxes(title_text='Acceleration Long (m/s^2)', range=[-20, 20], row=row, col=1)
        else:
            fig.update_yaxes(title_text='Acceleration (m/s^2)', range=[-20, 20], row=row, col=1)

        fig.add_trace(go.Scattergl(x=b, y=self.lon, mode='lines', name="Long Acc Sim", legendgroup=f"group1", showlegend=False), row=row, col=1)
        if debug:
            fig.add_trace(go.Scattergl(x=b, y=self.long, mode='lines', name="Long Acc Sim Max Init", legendgroup=f"group1", showlegend=False), row=row, col=1)
        fig.add_trace(self.track.get_channel_go("__acc_x", distance, group=f"group2", legend=False), row=row, col=1)
        if debug:
            fig.add_trace(go.Scattergl(x=b, y=(np.interp(b, d, self.track.long_acc) - self.lon), mode='lines', name="Long Acc Diff", legendgroup=f"group8", showlegend=False), row=row, col=1)

        if power_draw:
            row += 1
            fig.add_trace(go.Scattergl(x=b, y=self.motor_power / 1000, mode='lines', name="Motor Power", legendgroup=f"group1", showlegend=False), row=row, col=1)
            if "MCM_DC_Bus_Current" in self.track.data_keys:
                dc_power = self.track.raw_track["MCM_DC_Bus_Current"]["Value"][0, 0][0, :] * self.track.raw_track["MCM_DC_Bus_Voltage"]["Value"][0, 0][0, :]
                power_times = self.track.raw_track["MCM_DC_Bus_Current"]["Time"][0, 0][0, :] - self.track.start_time
                fig.add_trace(go.Scattergl(x=np.interp(power_times, self.track.raw_time, a), y=dc_power/1000, mode='lines', name="Motor Power Real", legendgroup=f"group2", showlegend=False), row=row, col=1)
            fig.update_yaxes(title_text='Motor Power (kW)', range=[-10, 80], row=row, col=1)
            # row += 1
            # fig.add_trace(go.Scattergl(x=b, y=self.energy / 1000 / 360, mode='lines', name="Energy Use", legendgroup=f"group1", showlegend=False), row=row, col=1)
            # fig.add_trace(go.Scattergl(x=d, y=self.y_a, mode='lines', name="Yaw Acc Real", legendgroup=f"group2", showlegend=False), row=row, col=1)
            # fig.update_yaxes(title_text='Energy Use (kWh)', range=[-1, 3], row=row, col=1)

        row +=1
        fig.add_trace(go.Scattergl(x=b, y=self.diff_time, mode='lines', showlegend=False), row=row, col=1)
        fig.update_yaxes(title_text='Elapsed Time (s)', row=row, col=1)
        row +=1
        fig.add_trace(go.Scattergl(x=b, y=self.dt_dx * self.track.sc, mode='lines', showlegend=False), row=row, col=1)
        fig.update_yaxes(title_text='Delta Time (s)', range=[-0.075, 0.075], row=row, col=1)

        if curvature:
            row +=1
            fig.add_trace(go.Scattergl(x=c, y=self.track.k, mode='lines', name="Curvature (1/m)", showlegend=False), row=row, col=1)
            fig.add_trace(go.Scattergl(x=c, y=self.track.k_prime, mode='lines', name="dCurvature (1/(m^2))", showlegend=False), row=row, col=1)
            fig.update_yaxes(title_text='k and dK', row=row, col=1)

        if fz:
            row +=1
            fzfl, fzfr, fzrl, fzrr, wt_pitch, wt_roll = self.car.find_contact_patch_loads(long_g=self.lon, lat_g=-self.lat, vel=self.vel)
            if weight_transfer:
                fig.add_trace(go.Scattergl(x=b, y=wt_pitch, mode='lines', name="WT Pitch", showlegend=True), row=row, col=1)
                fig.add_trace(go.Scattergl(x=b, y=wt_roll, mode='lines', name="WT Roll", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b, y=fzfl, mode='lines', name="FZFL", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b, y=fzfr, mode='lines', name="FZFR", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b, y=fzrl, mode='lines', name="FZRL", showlegend=True), row=row, col=1)
            fig.add_trace(go.Scattergl(x=b, y=fzrr, mode='lines', name="FZRR", showlegend=True), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__nl_fl", distance), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__nl_fr", distance), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__nl_rl", distance), row=row, col=1)
            fig.add_trace(self.track.get_channel_go("__nl_rr", distance), row=row, col=1)
            if fz_est:
                fzfl_r, fzfr_r, fzrl_r, fzrr_r, wt_pitch_r, wt_roll_r = self.car.find_contact_patch_loads(long_g=self.long_acc, lat_g=self.lat_acc, vel=np.interp(self.track.spa_t, self.track.raw_time, self.track.vel))
                fig.add_trace(go.Scattergl(x=d, y=fzfl_r, mode='lines', name="FZFL_Accel", showlegend=True), row=row, col=1)
                fig.add_trace(go.Scattergl(x=d, y=fzfr_r, mode='lines', name="FZFR_Accel", showlegend=True), row=row, col=1)
                fig.add_trace(go.Scattergl(x=d, y=fzrl_r, mode='lines', name="FZRL_Accel", showlegend=True), row=row, col=1)
                fig.add_trace(go.Scattergl(x=d, y=fzrr_r, mode='lines', name="FZRR_Accel", showlegend=True), row=row, col=1)
            fig.update_yaxes(title_text='Contact Patch Load (N)', row=row, col=1)

        fig.update_xaxes(title_text=x_label, row=row, col=1)
        fig.update_layout(template="plotly_dark", height=1000)
        fig.show()

    def plot_vs_vel(self, include_yaw_lims=False):
        fig = make_subplots(rows=4, cols=1, subplot_titles=("Lat_Acc v Vel", "Long_Acc v Vel", "Yaw_Acc v Vel", "Yaw_Rate v Vel"))
        color1 = px.colors.qualitative.Light24[0]
        color2 = px.colors.qualitative.Light24[1]
        fig.add_trace(go.Scattergl(x=self.vel, y=self.lat, mode='markers', name="Sim", marker=dict(size=2, color=color1), legendgroup=f"group1", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=np.interp(self.track.spa, self.track.interp_dist, self.track.vel), y=self.track.lat_acc, mode='markers', name="Real", marker=dict(size=2, color=color2), legendgroup=f"group2", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=self.las.vels, y=self.las.aymax[:, 1], mode='markers', name="LAS Lim", marker_color=self.las.vels, legendgroup=f"group3", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=self.las.vels, y=-self.las.aymax[:, 1], mode='markers', name="LAS Lim", marker_color=self.las.vels, legendgroup=f"group3", showlegend=False), row=1, col=1)
        fig.update_yaxes(title_text='Lat Acc (m/s^2)', row=1, col=1)

        fig.add_trace(go.Scattergl(x=self.vel, y=self.lon, mode='markers', name="Sim", marker=dict(size=2, color=color1), legendgroup=f"group1", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=np.interp(self.track.spa, self.track.interp_dist, self.track.vel), y=self.track.long_acc, mode='markers', name="Real", marker=dict(size=2, color=color2), legendgroup=f"group2", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=self.las.vels, y=self.las.longAcc_forward[:, 0], mode='markers', name="LAS Lim", marker_color=self.las.vels, legendgroup=f"group3", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=self.las.vels, y=self.las.longAcc_reverse[:, 0], mode='markers', name="LAS Lim", marker_color=self.las.vels, legendgroup=f"group3", showlegend=False), row=2, col=1)
        fig.update_yaxes(title_text='Long Acc (m/s^2)', row=2, col=1)

        fig.add_trace(go.Scattergl(x=self.vel, y=self.omega_dot, mode='markers', name="Sim", marker=dict(size=2, color=color1), legendgroup=f"group1", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scattergl(x=np.interp(self.track.spa, self.track.interp_dist, self.track.vel), y=self.track.get_ch("__yacc"), mode='markers', name="Real", marker=dict(size=2, color=color2), legendgroup=f"group2", showlegend=False), row=3, col=1)
        if include_yaw_lims:
            fig.add_trace(go.Scattergl(x=self.las.vels, y=self.las.yawmax[:, 2], mode='markers', name="LAS Lim", marker_color=self.las.vels, legendgroup=f"group3", showlegend=False), row=3, col=1)
            fig.add_trace(go.Scattergl(x=self.las.vels, y=-self.las.yawmax[:, 2], mode='markers', name="LAS Lim", marker_color=self.las.vels, legendgroup=f"group3", showlegend=False), row=3, col=1)
        fig.update_yaxes(title_text='Yaw Acc (rad/sec^2)', row=3, col=1)

        fig.add_trace(go.Scattergl(x=self.vel, y=self.omega, mode='markers', name="Sim", marker=dict(size=2, color=color1), legendgroup=f"group1", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scattergl(x=np.interp(self.track.spa, self.track.interp_dist, self.track.vel), y=self.track.y_r, mode='markers', name="Real", marker=dict(size=2, color=color2), legendgroup=f"group2", showlegend=False), row=4, col=1)
        fig.update_yaxes(title_text='Yaw Rate (rad/sec)', row=4, col=1)

        fig.update_xaxes(title_text='Vel (m/s)', row=4, col=1)
        fig.update_layout(template="plotly_dark")
        fig.show()

    def print_time(self):
        print(f"Sim time: {(self.tt[-1]):.3f} Actual time: {(self.track.gps.time[-1]):.3f}")

    def plot_LAS_lims(self):
        pos_name = [f"vel:{x:.2f}" for x in self.las.vels]

        fig = make_subplots(rows=3, cols=2)
        fig.add_trace(go.Scattergl(x=self.las.aymax[:, 1], y=self.las.aymax[:, 2], text=pos_name, marker_color=self.las.vels, legendgroup=f"group1", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=self.las.yawmax[:, 1], y=self.las.yawmax[:, 2], text=pos_name, marker_color=self.las.vels, legendgroup=f"group1", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scattergl(x=self.las.vels, y=self.las.longAcc_forward[:, 0], text=pos_name, marker_color=self.las.vels, legendgroup=f"group1", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scattergl(x=self.las.vels, y=self.las.longAcc_reverse[:, 0], text=pos_name, marker_color=self.las.vels, legendgroup=f"group1", showlegend=True), row=3, col=2)
        fig.add_trace(go.Scattergl(x=self.las.aymax_sa[:, 0], y=self.las.aymax_sa[:, 1], text=pos_name, marker_color=self.las.vels, legendgroup=f"group1", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=self.las.yawmax_sa[:, 0], y=self.las.yawmax_sa[:, 1], text=pos_name, marker_color=self.las.vels, legendgroup=f"group1", showlegend=False), row=2, col=2)
        fig.update_yaxes(title_text='Yaw Acc (rad/s^2)', row=1, col=1)
        fig.update_xaxes(title_text='Acc (m/s^2)', row=1, col=1)
        fig.update_yaxes(title_text='Yaw Acc (rad/s^2)', row=1, col=2)
        fig.update_xaxes(title_text='Acc (m/s^2)', row=1, col=2)
        fig.update_yaxes(title_text='Acc (m/s^2)', row=3, col=1)
        fig.update_yaxes(title_text='Acc (m/s^2)', row=3, col=2)
        fig.update_xaxes(title_text='Vel (m/s)', row=3, col=1)
        fig.update_xaxes(title_text='Vel (m/s)', row=3, col=2)
        fig.update_xaxes(title_text='Delta (deg)', row=2, col=1)
        fig.update_xaxes(title_text='Delta (deg)', row=2, col=2)
        fig.update_yaxes(title_text='Beta (deg)', row=2, col=1)
        fig.update_yaxes(title_text='Beta (deg)', row=2, col=2)
        fig.update_layout(template="plotly_dark")
        fig.show()

    def print_error_points(self):
        print(np.argwhere(np.abs(self.omega_dot) == 0).shape[0])

    def plot_mmd_vel(self):
        fig = go.Figure(data=[
            go.Scatter3d(x=self.lat, y=self.omega_dot, z=self.vel, mode='lines'),
            go.Scatter3d(x=self.lat, y=self.omega_dot, z=self.vel, mode='markers', marker=dict(size=2, color=self.dt_dx, colorscale='Viridis')),
            go.Scatter3d(x=self.track.raw_track["G_Force_Lat"]["Value"][0, 0][0, :]*9.81, y=self.track.get_ch("__yacc"), z=np.interp(self.track.spa, self.track.interp_dist, self.track.vel), mode='lines'),
            go.Scatter3d(x=self.lat[self.critc], y=self.omega_dot[self.critc], z=self.vel[self.critc], mode='markers', marker=dict(size=2)),
            go.Scatter3d(x=self.track.lat_acc, y=self.track.get_ch("__yacc"), z=self.track.real_vel, mode='markers', marker=dict(size=2, color=self.track.real_vel, colorscale='Viridis')),
            
        ])
        fig.update_layout(scene = dict(
            xaxis_title='Lat Acc (m/s^2)',
            yaxis_title='Yaw Acc (rad/sec^2)',
            zaxis_title='Vel (m/s)'),
            margin=dict(r=20, b=10, l=10, t=10)
        )
        fig.update_layout(template="plotly_dark",
                        scene = dict(
        xaxis = dict(range=[-20, 20],),
                     yaxis = dict(range=[-12, 12],),
                     zaxis = dict(range=[0, 35],),),
                          title_text="Parameter Optimization")

        fig.show()

    def plot_gg_vel(self):
        fig = go.Figure(data=[
            go.Scatter3d(x=self.lat, y=self.lon, z=self.vel, mode='lines'),
            go.Scatter3d(x=self.lat, y=self.lon, z=self.vel, mode='markers', marker=dict(size=2, color=self.dt_dx, colorscale='Viridis')),
            go.Scatter3d(x=self.track.raw_track["G_Force_Lat"]["Value"][0, 0][0, :]*9.81, y=self.track.raw_track["G_Force_Long"]["Value"][0, 0][0, :]*9.81, z=np.interp(self.track.spa, self.track.interp_dist, self.track.vel), mode='lines'),
            go.Scatter3d(x=self.lat[self.critc], y=self.lon[self.critc], z=self.vel[self.critc], mode='markers', marker=dict(size=2)),
            go.Scatter3d(x=self.track.lat_acc, y=self.track.long_acc, z=self.track.real_vel, mode='markers', marker=dict(size=2, color=self.track.real_vel, colorscale='Viridis')),
            
        ])
        fig.update_layout(scene = dict(
            xaxis_title='Lat Acc (m/s^2)',
            yaxis_title='Lon Acc (m/s^2)',
            zaxis_title='Vel (m/s)'),
            margin=dict(r=20, b=10, l=10, t=10)
        )
        fig.update_layout(template="plotly_dark",
                        scene = dict(
        xaxis = dict(range=[-15, 15],),
                     yaxis = dict(range=[-15, 15],),
                     zaxis = dict(range=[0, 35],),),
                          title_text="Parameter Optimization")

        fig.show()

    def plot_mmd3d(self, use_mmd3d=False):
        if use_mmd3d:
            self.generate_MMD3D()

            fig7 = MMD_3D_Graphs(self.ay_it1, self.yaw_it1, self.ax_it1, self.las.vels, self.valid_1)

            fig7.add_trace(go.Scatter3d(x=self.lat, y=self.omega_dot, z=self.lon, mode='markers', marker=dict(size=2, color=self.dt_dx, colorscale='Viridis' ), name="Sim"))
            fig7.add_trace(go.Scatter3d(x=self.track.get_ch("__acc_y"), y=self.track.get_ch("__yacc"), z=self.track.get_ch("__acc_x"), mode='lines', name="Real"))

            fig7.show()

        fig6 = go.Figure(data=[
            go.Scatter3d(x=self.lat, y=self.omega_dot, z=self.lon, mode='lines'),
            go.Scatter3d(x=self.lat, y=self.omega_dot, z=self.lon, mode='markers', marker=dict(size=2, color=self.dt_dx, colorscale='Viridis')),
            go.Scatter3d(x=self.track.get_ch("__acc_y"), y=self.track.get_ch("__yacc"), z=self.track.get_ch("__acc_x"), mode='lines'),
            go.Scatter3d(x=self.lat[self.critc], y=self.omega_dot[self.critc], z=self.lon[self.critc], mode='markers', marker=dict(size=1)),
            # go.Scatter3d(x=self.track.get_ch("__acc_y"), y=self.track.get_ch("__yacc"), z=self.track.get_ch("__acc_x"), mode='markers', marker=dict(size=2, color=self.track.real_vel, colorscale='Viridis')),
            
        ])

        self.las.plot_las(fig6)

        fig6.update_layout(scene = dict(
            xaxis_title='Lat Acc',
            yaxis_title='Yaw Acc (rad/sec^2)',
            zaxis_title='Lon Acc'),
            margin=dict(r=20, b=10, l=10, t=10)
        )
        
        fig6.show()

    def plot_convergence(self):
        # plot how each of the values change through the iterations
        iters = np.array(self.iters)
        iterations = np.arange(iters.shape[0])
        fig = make_subplots(rows=6, cols=1, subplot_titles=("Normalized Error", "Error std", "Run Time", "Critical Points", "Points Touched", "Iteration Time"))
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 0], mode='lines', name="Max Error", legendgroup=f"group1", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 1], mode='lines', name="Min Error", legendgroup=f"group2", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 2], mode='lines', name="Mean Error", legendgroup=f"group3", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 3], mode='lines', name="Error Std", legendgroup=f"group4", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 4], mode='lines', name="Time", legendgroup=f"group5", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 5], mode='lines', name="Critical Points", legendgroup=f"group6", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 6], mode='lines', name="Points Touched", legendgroup=f"group7", showlegend=False), row=5, col=1)
        fig.add_trace(go.Scattergl(x=iterations, y=iters[:, 7], mode='lines', name="Iteration Time", legendgroup=f"group8", showlegend=False), row=6, col=1)
        fig.update_yaxes(title_text='Error', row=1, col=1)
        fig.update_yaxes(title_text='Error Std', row=2, col=1)
        fig.update_yaxes(title_text='Time (s)', row=3, col=1)
        fig.update_yaxes(title_text='Critical Points', row=4, col=1)
        fig.update_yaxes(title_text='Points Touched', row=5, col=1)
        fig.update_yaxes(title_text='Iteration Time (s)', row=6, col=1)
        fig.update_xaxes(title_text='Iteration', row=6, col=1)
        # update the range of the iteration time to be the max in the range of [1:]
        fig.update_yaxes(range=[0, np.max(iters[1:, 7])], row=6, col=1)
        fig.update_layout(template="plotly_dark")
        fig.show()