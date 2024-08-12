from toolkit.cars.car_configuration import Car
import numpy as np
import plotly.graph_objects as go
from .steady_state_solver.sss import Steady_State_Solver
from .steady_state_solver.iterative import Iterative_Solver
import plotly.express as px
from toolkit.common.constants import *

class MMD:
    def __init__(self, car: Car, solver: Steady_State_Solver = Iterative_Solver()):
        self.car = car
        self.solver: Steady_State_Solver = solver

    def mmd_sweep(self, v_avg, long_g = 0.0, lin_space = True, max_beta = 15, max_delta = 15, size = 31, mu = 1.0, seeded = False):
        self.v_avg = v_avg
        if lin_space:
            # delta and beta with linear spacing
            self.beta = np.linspace(-max_beta, max_beta, (2 * size + 1))
            delta = np.linspace(0, max_delta, (size + 1))
            self.delta_two = np.concatenate((-delta[::-1], delta[1:]))
        else:
            # delta and beta with log spacing
            beta = np.geomspace(0.1, max_beta, size)
            self.beta = np.concatenate((-beta[::-1], np.array([0]), beta))
            delta = np.geomspace(0.1, max_delta, size)
            self.delta_two = np.concatenate((-delta[::-1], np.array([0]), delta))
            delta = np.concatenate((np.array([0]), delta))

        # prepare for the sweep
        delta_v, beta_v = np.meshgrid(delta, self.beta)
        inds = np.argwhere(np.full(delta_v.shape, True))
        ay = np.zeros(delta_v.shape)
        ax = np.zeros(delta_v.shape)
        yaw = np.zeros(delta_v.shape)
        cn = np.zeros(delta_v.shape)
        bruh = np.zeros(delta_v.shape)
        error = np.zeros(delta_v.shape)

        for index in inds:
            ay_it = 0
            if seeded:
                if index[0] == 0  and index[1] != 0:
                    ay_it = ay[index[0], index[1] - 1]
            ay_it, cn_it, yaw_it, ax_it, bruh_it, fal = self.solver.solve_for_long(self.car, v_avg, long_g, delta_x=np.deg2rad(delta_v[index[0], index[1]]), beta_x=np.deg2rad(beta_v[index[0], index[1]]), zeros=False, ay_it=ay_it, mu_corr=mu)
            
            ay[index[0], index[1]] = ay_it
            ax[index[0], index[1]] = ax_it
            yaw[index[0], index[1]] = yaw_it
            cn[index[0], index[1]] = cn_it
            bruh[index[0], index[1]] = bruh_it
            error[index[0], index[1]] = fal

        # fill out the symmetric side
        self.ay = np.concatenate((-np.flip(ay[:, 1:], (0, 1)), ay), axis=1)
        self.ax = np.concatenate((np.flip(ax[:, 1:], (0, 1)), ax), axis=1)
        self.yaw = np.concatenate((-np.flip(yaw[:, 1:], (0, 1)), yaw), axis=1)
        self.cn = np.concatenate((-np.flip(cn[:, 1:], (0, 1)), cn), axis=1)
        self.bruh = np.concatenate((np.flip(bruh[:, 1:], (0, 1)), bruh), axis=1)
        self.error = np.concatenate((np.flip(error[:, 1:], (0, 1)), error), axis=1)
        self.delta_v = np.concatenate((-np.flip(delta_v[:, 1:], (0, 1)), delta_v), axis=1)
        self.beta_v = np.concatenate((-np.flip(beta_v[:, 1:], (0, 1)), beta_v), axis=1)
    
    def calc_stability(self):
        # calculate the stability index
        self.stability = np.zeros(self.error.shape)
        # the derivative of yaw accel with respect to beta * izz is the stability index
        self.stability = np.gradient(self.yaw, self.beta_v[:, 0], axis=0) * self.car.izz
        self.stability[self.error != 0] = 0

    def calc_control_moment(self):
        # calculate the control moment
        self.control_moment = np.zeros(self.error.shape)
        self.control_moment = np.gradient(self.yaw, self.delta_v[0, :], axis=1) * self.car.izz
        self.control_moment[self.error != 0] = 0

    def calc_understeer_gradient(self):
        # calculate the understeer gradient
        self.understeer_gradient = np.zeros(self.error.shape)
        self.understeer_gradient = np.gradient(self.ay, self.delta_v[0, :], axis=1)
        self.understeer_gradient[self.error != 0] = 0

    def plot_stability(self):
        self.calc_stability()
        # plot a line of the stability index for delta
        fig = go.Figure()
        fig.update_xaxes(title_text='Beta (deg)')
        fig.update_yaxes(title_text='Stability Index')
        for i in range(int(len(self.delta_two)/2)+1):
            i = int(len(self.delta_two)/2)+i
            deltax = self.delta_v[0, i]
            valid = self.error[:, i] == 0
            fig.add_trace(
                go.Scatter(
                    x=self.beta_v[valid, i],
                    y=self.stability[valid, i],
                    mode='lines',
                    hovertext="{:.1f}° Delta".format(deltax),
                    marker=dict(color='blue'), name="{:.1f}° Delta".format(deltax), legendgroup=f"group2{i}", showlegend=True
                ))
        fig.update_layout(template="plotly_dark", title_text=f"Stability Index at {self.v_avg:.2f} m/s")
        fig.show()

    def plot_control_moment(self):
        self.calc_control_moment()
        # plot a line of the control moment for delta
        fig = go.Figure()
        fig.update_xaxes(title_text='Delta (deg)')
        fig.update_yaxes(title_text='Control Moment')
        for i in range(int(len(self.beta)/2)+1):
            i = int(len(self.beta)/2)+i
            betax = self.beta_v[i, 0]
            valid = self.error[i, :] == 0
            fig.add_trace(
                go.Scatter(
                    x=self.delta_v[i, valid],
                    y=self.control_moment[i, valid],
                    mode='lines',
                    hovertext="{:.1f}° Beta".format(betax),
                    marker=dict(color='blue'), name="{:.1f}° Beta".format(betax), legendgroup=f"group{i}2", showlegend=True
                ))
        fig.update_layout(template="plotly_dark", title_text=f"Control Moment at {self.v_avg:.2f} m/s")
        fig.show()

    def plot_understeer_gradient(self):
        self.calc_understeer_gradient()
        # plot a line of the understeer gradient for delta
        fig = go.Figure()
        fig.update_xaxes(title_text='Delta (deg)')
        fig.update_yaxes(title_text='Understeer Gradient')
        for i in range(int(len(self.beta)/2)+1):
            i = int(len(self.beta)/2)+i
            betax = self.beta_v[i, 0]
            valid = self.error[i, :] == 0
            fig.add_trace(
                go.Scatter(
                    x=self.delta_v[i, valid],
                    y=self.understeer_gradient[i, valid],
                    mode='lines',
                    hovertext="{:.1f}° Beta".format(betax),
                    marker=dict(color='blue'), name="{:.1f}° Beta".format(betax), legendgroup=f"grou{i}p2", showlegend=True
                ))
        fig.update_layout(template="plotly_dark", title_text=f"Understeer Gradient at {self.v_avg:.2f} m/s")
        fig.show()

    def plot_ay(self):
        fig = px.imshow(self.ay, labels=dict(x="Delta", y="Beta", color="Ay (m/s^2)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()

    def plot_ax(self):
        fig = px.imshow(self.ax, labels=dict(x="Delta", y="Beta", color="Ax (m/s^2)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()
    
    def plot_yaw(self):
        fig = px.imshow(self.yaw, labels=dict(x="Delta", y="Beta", color="Yaw Accel (rad/s^2)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()

    def plot_valid(self):
        fig = px.imshow(self.error, labels=dict(x="Delta", y="Beta", color="Valid"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()
    
    def plot_solve_iters(self):
        fig = px.imshow(self.bruh, labels=dict(x="Delta", y="Beta", color="Solve Iterations"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()

    def clear_high_sa(self, max_sa=25):
        if not hasattr(self, 'sa_fl'):
            self.gen_sa()
        sa_lim = np.deg2rad(max_sa)
        self.error[((self.sa_fl > sa_lim) | (self.sa_fr > sa_lim) | (self.sa_rl > sa_lim) | (self.sa_rr > sa_lim) | (self.sa_fl < -sa_lim) | (self.sa_fr < -sa_lim) | (self.sa_rl < -sa_lim) | (self.sa_rr < -sa_lim))] = 1

    def gen_sa(self):
        delta_sp, beta_sp = np.meshgrid(np.deg2rad(self.delta_two), np.deg2rad(self.beta))
        delta_fl, delta_fr, delta_rl, delta_rr = self.car.calculate_tire_delta_angle(delta_sp.flatten(), 0.0, 0.0, 0.0)
        sa_fl, sa_fr, sa_rl, sa_rr = self.car.calculate_slip_angles(self.v_avg, self.ay.flatten()/self.v_avg, beta_sp.flatten(), delta_fl, delta_fr, delta_rl, delta_rr)
        self.sa_fl, self.sa_fr, self.sa_rl, self.sa_rr = sa_fl.reshape(beta_sp.shape), sa_fr.reshape(beta_sp.shape), sa_rl.reshape(beta_sp.shape), sa_rr.reshape(beta_sp.shape)

    def plot_sa(self):
        if not hasattr(self, 'sa_fl'):
            self.gen_sa()
        fig = px.imshow(np.rad2deg(self.sa_fl), labels=dict(x="Delta", y="Beta", color="Front Left SA (deg)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig.update_layout(template="plotly_dark", title_text="2D MMD")
        fig.show()
        fig2 = px.imshow(np.rad2deg(self.sa_fr), labels=dict(x="Delta", y="Beta", color="Front Right SA (deg)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig2.update_layout(template="plotly_dark", title_text="2D MMD")
        fig2.show()
        fig3 = px.imshow(np.rad2deg(self.sa_rl), labels=dict(x="Delta", y="Beta", color="Rear Left SA (deg)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig3.update_layout(template="plotly_dark", title_text="2D MMD")
        fig3.show()
        fig4 = px.imshow(np.rad2deg(self.sa_rr), labels=dict(x="Delta", y="Beta", color="Rear Right SA (deg)"), origin='lower', x=self.delta_two, y=self.beta, aspect="auto")
        fig4.update_layout(template="plotly_dark", title_text="2D MMD")
        fig4.show()

    def plot_mmd(self, show_bad=False, pub=False):
        # create a traditional MMD plot
        fig = go.Figure()
        if pub:
            fig.update_xaxes(title_text='Lat Acc (G)', range=[-2, 2])
            fig.update_yaxes(title_text='Cn', range=[-15, 15])
        else:
            fig.update_xaxes(title_text='Lat Acc (G)')
            fig.update_yaxes(title_text='Cn')
        for i, betax in enumerate(self.beta):
            valid = self.error[i, :] == 0
            if show_bad:
                valid = np.full(self.error[i, :].shape, True)
            fig.add_trace(
                go.Scatter(
                    x=self.ay[i, valid] / G,
                    y=self.cn[i, valid],
                    mode='lines',
                    hovertext="β={:.2f}°".format(betax),
                    marker=dict(color='red'), legendgroup=f"group2", showlegend=False
                ))
        for j, deltax in enumerate(self.delta_two):
            valid = self.error[:, j] == 0
            if show_bad:
                valid = np.full(self.error[:, j].shape, True)
            fig.add_trace(
                go.Scatter(
                    x=self.ay[valid, j] / G,
                    y=self.cn[valid, j],
                    mode='lines',
                    hovertext="{:.1f}° Delta".format(deltax),
                    marker=dict(color='blue'), name="MMD", legendgroup=f"group2", showlegend=(j == len(self.delta_two)-1)
                ))
        if pub:
            fig.update_layout(title_text=f"MMD at {self.v_avg:.2f} m/s", width=800, height=600)
        else:
            fig.update_layout(template="plotly_dark", title_text=f"MMD at {self.v_avg:.2f} m/s") #  and long. accel. of 0 m/s^2
        fig.show()

    def plot_ymd(self, show_bad=False, pub=False):
        # create a traditional YMD plot
        fig = go.Figure()
        if pub:
            fig.update_xaxes(title_text='Lat Acc (G)', range=[-2, 2])
            fig.update_yaxes(title_text='Yaw Moment (Nm)', range=[-5000, 5000])
        else:
            fig.update_xaxes(title_text='Lat Acc (G)')
            fig.update_yaxes(title_text='Yaw Moment (Nm)')
        for j, deltax in enumerate(self.delta_two):
            valid = self.error[:, j] == 0
            if show_bad:
                valid = np.full(self.error[:, j].shape, True)
            fig.add_trace(
                go.Scatter(
                    x=self.ay[valid, j] / G,
                    y=self.yaw[valid, j] * self.car.izz,
                    mode='lines',
                    hovertext="{:.1f}° Delta".format(deltax),
                    marker=dict(color='red', size=0.5), name="MMD", legendgroup=f"group2", showlegend=(j == len(self.delta_two)-1)
                ))
        for i, betax in enumerate(self.beta):
            valid = self.error[i, :] == 0
            if show_bad:
                valid = np.full(self.error[i, :].shape, True)
            fig.add_trace(
                go.Scatter(
                    x=self.ay[i, valid] / G,
                    y=self.yaw[i, valid] * self.car.izz,
                    mode='lines',
                    hovertext="β={:.2f}°".format(betax),
                    marker=dict(color='blue', size=0.5), legendgroup=f"group2", showlegend=False
                ))
        if pub:
            fig.update_layout(title_text=f"YMD at {self.v_avg:.2f} m/s", width=800, height=600)
        else:
            fig.update_layout(template="plotly_dark", title_text=f"YMD at {self.v_avg:.2f} m/s") #  and long. accel. of 0 m/s^2
        fig.show()

    def add_mmd(self, figure, name, visible=False, size=1):
        # create a traditional MMD plot
        initial_len = len(figure.data)
        for i, betax in enumerate(self.beta):
            valid = self.error[i, :] == 0
            figure.add_trace(
                go.Scatter(
                    x=self.ay[i, valid] / G,
                    y=self.cn[i, valid],
                    mode='lines',
                    visible=False,
                    hovertext="β={:.2f}°".format(betax),
                    marker=dict(color='red', size=size), name=name, legendgroup=f"group_{name}", showlegend=False
                ))
        for j, deltax in enumerate(self.delta_two):
            valid = self.error[:, j] == 0
            figure.add_trace(
                go.Scatter(
                    x=self.ay[valid, j] / G,
                    y=self.cn[valid, j],
                    mode='lines',
                    visible=False,
                    hovertext="{:.1f}° Delta".format(deltax),
                    marker=dict(color='blue', size=size), name=name, legendgroup=f"group_{name}", showlegend=False
                ))
        final_len = len(figure.data)
        if visible:
            for ob in figure.data[initial_len:final_len]: ob.visible = True
        return (initial_len, final_len)
    
    def enhance_line(self, cur_points, pts_to_add: int):
        new_pts = np.zeros((cur_points.shape[0], cur_points.shape[1] * (pts_to_add + 1) - 1))
        new_pts[:, ::pts_to_add + 1] = cur_points
        new_inds = np.arange(1, cur_points.shape[1] * (pts_to_add + 1) - 1, pts_to_add + 1)
        seed_weights = np.repeat(np.linspace(0, 1, pts_to_add + 2)[1:-1], cur_points.shape[0])
        for (ind, w) in zip(new_inds, seed_weights):
            pass



    def add_mmd_enhanced(self, figure, name, visible=False, size=1):
        # create a traditional MMD plot, but add additional points to refine the plot
        initial_len = len(figure.data)
        for i, betax in enumerate(self.beta):
            valid = self.error[i, :] == 0
            figure.add_trace(
                go.Scatter(
                    x=self.ay[i, valid] / G,
                    y=self.cn[i, valid],
                    mode='lines',
                    visible=False,
                    hovertext="β={:.2f}°".format(betax),
                    marker=dict(color='red', size=size), name=name, legendgroup=f"group_{name}", showlegend=False
                ))
        for j, deltax in enumerate(self.delta_two):
            valid = self.error[:, j] == 0
            figure.add_trace(
                go.Scatter(
                    x=self.ay[valid, j] / G,
                    y=self.cn[valid, j],
                    mode='lines',
                    visible=False,
                    hovertext="{:.1f}° Delta".format(deltax),
                    marker=dict(color='blue', size=size), name=name, legendgroup=f"group_{name}", showlegend=False
                ))
        final_len = len(figure.data)
        if visible:
            for ob in figure.data[initial_len:final_len]: ob.visible = True
        return (initial_len, final_len)