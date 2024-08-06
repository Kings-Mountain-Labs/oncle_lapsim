from dataclasses import dataclass
from .tire_model_pacejka_2010 import *
import numpy as np
from .ttc_loader import filter_eccentricity
from typing import List
from numpy.typing import ArrayLike
from .tire_fitting_masks import LABELS, NAMES
from .tire_model_utils import set_x
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


@dataclass
class Sweep:
    f_z: float = 0.0
    i_c: float = 0.0
    s_a: float = 0.0
    s_r: float = 0.0
    v: float = 0.0
    p: float = 0.0

def sweep_model(tire_model, fig_title, run_data=None, sr_plot=False, fx_plot=True, fy_plot=True, mz_plot=True, mx_plot=True, zero_sl=True, min_pts=100):
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Fy-SA", "Mz-SA", "Fx-SA", "Mx-SA"))
    if type(tire_model) != list:
        tire_model = [tire_model]
    color_num = 0
    run_dats: List[Sweep] = []
    for f_z in [230, 670, 1100, 1540, 1980]:
        for i_c in [0, 2, 4]:
            for v in [20/3.6, 40/3.6, 72/3.6]:
                for p in [55000, 69000, 82700, 96500]:
                    if sr_plot:
                        for sa in [6, 3, 0, -3, -6]:
                            run_dats.append(Sweep(f_z=f_z, i_c=i_c, v=v, p=p, s_a=sa))
                    else:
                        if zero_sl:
                            run_dats.append(Sweep(f_z=f_z, i_c=i_c, v=v, p=p, s_r=0.0))
                        else:
                            for sr in [0.1, 0.05, 0.025, 0.0, -0.025, -0.05, -0.1, -0.2, -0.3]:
                                run_dats.append(Sweep(f_z=f_z, i_c=i_c, v=v, p=p, s_r=sr))

    for run_set in run_dats:
        if run_data is not None:
            if sr_plot:
                add_filter = (np.abs(run_data.SA - run_set.s_a / 180 * np.pi) < 0.1 / 180 * np.pi)
            else:
                if zero_sl:
                    add_filter = not(run_data.SR.isna())
                else:
                    add_filter = (np.abs(run_data.SR - run_set.s_r) < 0.01)
            ds = run_data[add_filter & (np.abs(run_data.FZ - run_set.f_z) < run_set.f_z/10) & (np.abs(run_data.IA - run_set.i_c / 180 * np.pi) < 0.5 / 180 * np.pi) & (np.abs(run_data.P - run_set.p) < 6000) & (np.abs(run_data.V - run_set.v) < 1.0)]
            if ds.shape[0] > min_pts:
                title = f"{run_set.f_z}N, {run_set.i_c}deg, {run_set.v}kph, {run_set.p}kPa"
                sa_deg = ds.SA * 180 / np.pi

                color = px.colors.qualitative.Light24[color_num % 24]

                fig.add_trace(go.Scattergl(x=sa_deg, y=ds.FY, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=1, col=1)
                fig.add_trace(go.Scattergl(x=sa_deg, y=ds.MZ, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=1, col=2)
                fig.add_trace(go.Scattergl(x=sa_deg, y=ds.FX, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=2, col=1)
                fig.add_trace(go.Scattergl(x=sa_deg, y=ds.MX, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=2, col=2)

            if (ds.shape[0] > min_pts) or (run_data is None):
                plot_length = 100
                ru_o, ru_z = np.ones(plot_length), np.zeros(plot_length)
                if sr_plot:
                    upper_sr, lower_sr = 0.2, -0.3
                    slip_r = np.linspace(lower_sr, upper_sr, plot_length)
                    sr_range = slip_r
                    inputs = np.array([ru_o * run_set.f_z, sr_range, ru_o * run_set.s_a / 180 * np.pi, ru_o * run_set.i_c, ru_z, ru_o * run_set.v, ru_o * run_set.p, ru_z]).T
                else:
                    max_sa = 20
                    slip_r = np.linspace(-max_sa, max_sa, plot_length)
                    sa_range = slip_r * np.pi / 180
                    inputs = np.array([ru_o * run_set.f_z, ru_z, sa_range, ru_o * run_set.i_c, ru_z, ru_o * run_set.v, ru_o * run_set.p, ru_z]).T
                out = tire_model.fullSteadyState(inputs)
                fig.add_trace(go.Scattergl(x=slip_r, y=out[:, 1], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=1, col=1)
                fig.add_trace(go.Scattergl(x=slip_r, y=out[:, 5], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=1, col=2)
                fig.add_trace(go.Scattergl(x=slip_r, y=out[:, 0], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=2, col=1)
                fig.add_trace(go.Scattergl(x=slip_r, y=out[:, 3], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=True), row=2, col=2)
                color_num += 1

    if sr_plot:
        x_label = 'Slip Ratio'
    else:
        x_label = 'Slip Angle (deg)'

    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_xaxes(title_text=x_label, row=2, col=2)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=1, col=2)
    fig.update_yaxes(title_text='Longitudinal Force', row=2, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=2, col=2)
    v_mean, p_mean, sr_mean = run_data.V.mean(), run_data.P.mean(), run_data.SL.mean()
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title} {v_mean:.2f}kph {p_mean:.2f} kPa")
    fig.show()
    


    

def sweep_SA(tire_model, fig_title):
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Fy-SA", "Mz-SA", "Fx-SA", "Mx-SA"))
    if type(tire_model) != list:
        tire_model = [tire_model]
    color_num = 0
    run_dats: List[Sweep] = []
    for f_z in [230, 670, 1100, 1540, 1980]:
        for i_c in [0]: # , 2, 4
            for v in [20]: # , 40, 72
                for p in [55]: # 55, , 82.7, 96.5
                    for sr in [0.1, 0.05, 0.025, 0.0, -0.025, -0.05, -0.1, -0.2, -0.3]:
                        run_dats.append(Sweep(f_z=f_z, i_c=i_c, v=v, p=p, s_r=sr))

    for run_set in run_dats:
        raw_title = f"{run_set.f_z}N, {run_set.i_c}deg, {(run_set.v / 3.6):.2f}m/s, {run_set.p}kPa, {run_set.s_r}slip ratio"
        color = px.colors.qualitative.Light24[color_num % 24]
        sa_r = np.linspace(-20, 20, 100)
        inputs = np.array([np.ones(100) * run_set.f_z, np.ones(100) * run_set.s_r, np.deg2rad(sa_r), np.ones(100) * np.deg2rad(run_set.i_c), np.zeros(
            100), np.ones(100) * run_set.v / 3.6, np.ones(100) * run_set.p * 1000, np.ones(100) * run_set.f_z]).T
        for ind, tire in enumerate(tire_model):
            title = f"{ind}: {raw_title}"
            out = tire.fullSteadyState(inputs)
            fig.add_trace(go.Scatter(x=sa_r, y=out[:, 1],  mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sa_r, y=out[:, 5],  mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=sa_r, y=out[:, 0],  mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=sa_r, y=out[:, 3],  mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=2, col=2)
            fig.add_trace(go.Scatter(x=sa_r, y=out[:, 30], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=sa_r, y=out[:, 29], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=True), row=3, col=2)
        color_num += 1

    fig.update_xaxes(title_text='Slip Angle (deg)', row=1, col=1)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=1, col=2)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=2, col=1)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=2, col=2)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=3, col=1)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=3, col=2)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=1, col=2)
    fig.update_yaxes(title_text='Longitudinal Force', row=2, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=2, col=2)
    fig.update_yaxes(title_text='dFz', row=3, col=1)
    fig.update_yaxes(title_text='Kxk', row=3, col=2)
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title}")
    fig.show()

def sweep_SR(tire_model, fig_title):
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Fy-SR", "Mz-SR", "Fx-SR", "Mx-SR"))
    if type(tire_model) != list:
        tire_model = [tire_model]
    color_num = 0
    run_dats: List[Sweep] = []
    for f_z in [230, 670, 1100, 1540, 1980]:
        for i_c in [0]: # , 2, 4
            for v in [20 , 40, 72]: #
                for p in [55]: # 55, , 82.7, 96.5
                    for sa in [6, 3, 0]: # , -3, -6
                        run_dats.append(Sweep(f_z=f_z, i_c=i_c, v=v, p=p, s_a=sa))

    for run_set in run_dats:
        raw_title = f"{run_set.f_z}N, {run_set.i_c}deg, {(run_set.v / 3.6):.2f}m/s, {run_set.p}kPa, {run_set.s_a}slip angle"
        color = px.colors.qualitative.Light24[color_num % 24]
        sr_r = np.linspace(-0.4, 0.2, 100)
        inputs = np.array([np.ones(100) * run_set.f_z, sr_r, np.ones(100) * np.deg2rad(run_set.s_a), np.ones(100) * np.deg2rad(run_set.i_c), np.zeros(
            100), np.ones(100) * run_set.v/3.6, np.ones(100) * run_set.p * 1000, np.ones(100) * run_set.f_z]).T
        for ind, tire in enumerate(tire_model):
            title = f"{ind}: {raw_title}"
            out = tire.fullSteadyState(inputs)
            fig.add_trace(go.Scatter(x=sr_r, y=out[:, 1], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sr_r, y=out[:, 5], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=sr_r, y=out[:, 0], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=sr_r, y=out[:, 3], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=2, col=2)
            fig.add_trace(go.Scatter(x=sr_r, y=out[:, 30], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=sr_r, y=out[:, 29], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}_{ind}", showlegend=True), row=3, col=2)
        color_num += 1

    fig.update_xaxes(title_text='Slip Ratio', row=1, col=1)
    fig.update_xaxes(title_text='Slip Ratio', row=1, col=2)
    fig.update_xaxes(title_text='Slip Ratio', row=2, col=1)
    fig.update_xaxes(title_text='Slip Ratio', row=2, col=2)
    fig.update_xaxes(title_text='Slip Ratio', row=3, col=1)
    fig.update_xaxes(title_text='Slip Ratio', row=3, col=2)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=1, col=2)
    fig.update_yaxes(title_text='Longitudinal Force', row=2, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=2, col=2)
    fig.update_yaxes(title_text='dFz', row=3, col=1)
    fig.update_yaxes(title_text='Kxk', row=3, col=2)
    fig.update_layout(template="plotly_dark",
                      title_text=f"{fig_title}")
    fig.show()

def split_run_with_MF_SA(run_data, tire_model, fig_title):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Fy-SA", "Mz-SA", "Fx-SA", "Mx-SA"))
    color_num = 0
    run_dats: List[Sweep] = []
    for f_z in [140, 220, 440, 640, 860, 1100]:
        for i_c in [-4, -2, 0, 2, 4]:
            for v in [24, 40, 72]:
                for p in [55, 69, 82.7, 96.5]:
                    run_dats.append(Sweep(f_z=f_z, i_c=i_c, v=v, p=p))

    for run_set in run_dats:
        ds = run_data[(np.abs(run_data.FZ - run_set.f_z) < run_set.f_z/10) &
                      (np.abs(run_data.IA - run_set.i_c / 180 * np.pi) < 0.5 / 180 * np.pi) & (np.abs(run_data.P - run_set.p * 1000) < 6000) & (np.abs(run_data.V - run_set.v/3.6) < 2.0/3.6)]
        if ds.shape[0] > 100:
            title = f"{run_set.f_z}N, {run_set.i_c}deg, {run_set.v}kph, {run_set.p}kPa"
            sa_deg = ds.SA * 180 / np.pi

            color = px.colors.qualitative.Light24[color_num % 24]

            fig.add_trace(go.Scattergl(x=sa_deg, y=ds.FY, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scattergl(x=sa_deg, y=ds.MZ, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scattergl(x=sa_deg, y=ds.FX, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scattergl(x=sa_deg, y=ds.MX, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=2, col=2)

            sa_range = np.linspace(-20, 20, 100) * np.pi / 180
            sa_r = np.linspace(-20, 20, 100)
            inputs = np.array([np.ones(100) * ds.FZ.mean(), np.zeros(100), sa_range, np.ones(100) * ds.IA.mean(), np.zeros(
                100), np.ones(100) * ds.V.mean(), np.ones(100) * ds.P.mean(), np.ones(100) * ds.N.mean()]).T
            out = tire_model.fullSteadyState(inputs)
            fig.add_trace(go.Scattergl(x=sa_r, y=out[:, 1], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scattergl(x=sa_r, y=out[:, 5], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scattergl(x=sa_r, y=out[:, 0], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scattergl(x=sa_r, y=out[:, 3], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=True), row=2, col=2)
            color_num += 1

    fig.update_xaxes(title_text='Slip Angle (deg)', row=1, col=1)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=1, col=2)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=2, col=1)
    fig.update_xaxes(title_text='Slip Angle (deg)', row=2, col=2)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=1, col=2)
    fig.update_yaxes(title_text='Longitudinal Force', row=2, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=2, col=2)
    v_mean, p_mean, sr_mean = run_data.V.mean(), run_data.P.mean(), run_data.SL.mean()
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title} {v_mean:.2f}kph {p_mean:.2f} kPa")
    fig.show()


def split_run_with_MF_SR(run_data, tire_model, fig_title):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Fy-SR", "Mz-SR", "Fx-SR", "Mx-SR"))
    color_num = 0
    run_dats: List[Sweep] = []
    for f_z in [220, 440, 660, 880, 1100]:
        for i_c in [0, 2, 4]:
            for v in [24, 40, 72]:
                for p in [55, 69, 82.7, 96.5]:
                    for s_a in [0, 3, 6]:
                        run_dats.append(
                            Sweep(f_z=f_z, i_c=i_c, v=v, s_a=s_a, p=p))

    for run_set in run_dats:
        ds = run_data[(np.abs(run_data.SA - run_set.s_a / 180 * np.pi) < 0.1 / 180 * np.pi) & (
            np.abs(run_data.IA - run_set.i_c / 180 * np.pi) < 0.5 / 180 * np.pi) & (np.abs(run_data.V - run_set.v/3.6) < 2.0/3.6) & (np.abs(run_data.FZ - run_set.f_z) < run_set.f_z/10 + 20) & (np.abs(run_data.P - run_set.p * 1000) < 6000)]
        if ds.shape[0] > 100:
            title = f"{run_set.s_a}deg, {run_set.f_z}N, {run_set.i_c}deg, {run_set.v}kph, {run_set.p}kPa"
            sr_deg = ds.SL

            color = px.colors.qualitative.Light24[color_num % 24]

            fig.add_trace(go.Scattergl(x=sr_deg, y=ds.FY, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scattergl(x=sr_deg, y=ds.MZ, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scattergl(x=sr_deg, y=ds.FX, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scattergl(x=sr_deg, y=ds.MX, marker=dict(color=color, size=1), mode='markers', legendgroup=f"group{color_num}", showlegend=False), row=2, col=2)

            sr_range = np.linspace(-0.4, 0.2, 100)
            inputs = np.array([np.ones(100) * ds.FZ.mean(), sr_range, np.ones(100) * ds.SA.mean(), np.ones(100) * ds.IA.mean(
            ), np.zeros(100), np.ones(100) * ds.V.mean(), np.ones(100) * ds.P.mean(), np.ones(100) * ds.N.mean()]).T
            out = tire_model.fullSteadyState(inputs)
            fig.add_trace(go.Scattergl(x=sr_range, y=out[:, 1], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scattergl(x=sr_range, y=out[:, 5], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scattergl(x=sr_range, y=out[:, 0], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scattergl(x=sr_range, y=out[:, 3], mode='lines', name=title, marker=dict(color=color), legendgroup=f"group{color_num}", showlegend=True), row=2, col=2)
            color_num += 1

    fig.update_xaxes(title_text='Slip Ratio (%)', row=1, col=1)
    fig.update_xaxes(title_text='Slip Ratio (%)', row=1, col=2)
    fig.update_xaxes(title_text='Slip Ratio (%)', row=2, col=1)
    fig.update_xaxes(title_text='Slip Ratio (%)', row=2, col=2)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=1, col=2)
    fig.update_yaxes(title_text='Longitudinal Force', row=2, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=2, col=2)
    v_mean, p_mean = run_data.V.mean(), run_data.P.mean()
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title} {v_mean:.2f} kph {p_mean:.2f} kPa")
    fig.show()

def split_run_fit(run_data, tire_model, fig_title):
    fig = make_subplots(rows=9, cols=1, shared_xaxes=True) #,subplot_titles=("Fy-SA", "Mz-SA", "Fx-SA", "Mx-SA")

    color = px.colors.qualitative.Light24[0]
    color2 = px.colors.qualitative.Light24[1]

    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FY, marker=dict(color=color), mode='lines', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.MZ, marker=dict(color=color), mode='lines', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FX, marker=dict(color=color), mode='lines', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.MX, marker=dict(color=color), mode='lines', showlegend=True, name="Data"), row=4, col=1)

    inputs = np.array([run_data.FZ, run_data.SL, run_data.SA, run_data.IA, run_data.PHIT, run_data.V, run_data.P, run_data.N]).T
    out = tire_model.fullSteadyState(inputs)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 1], mode='lines', marker=dict(color=color2), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 5], mode='lines', marker=dict(color=color2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 0], mode='lines', marker=dict(color=color2), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 3], mode='lines', marker=dict(color=color2), showlegend=True, name="Model"), row=4, col=1)

    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.P/1000, mode='lines', showlegend=True, name="P"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.V, mode='lines', showlegend=True, name="Vel"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FZ, mode='lines', showlegend=True, name="FZ"), row=6, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.SA/3.14*180, mode='lines', showlegend=True, name="SA"), row=7, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.IA/3.14*180, mode='lines', showlegend=True, name="IA"), row=7, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.SR, mode='lines', showlegend=True, name="SR"), row=8, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.SL, mode='lines', showlegend=True, name="SL"), row=8, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.N, mode='lines', showlegend=True, name="N"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.TSTI, mode='lines', showlegend=True, name="TSTI"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.TSTC, mode='lines', showlegend=True, name="TSTC"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.TSTO, mode='lines', showlegend=True, name="TSTO"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.PHIT, mode='lines', showlegend=True, name="PHIT"), row=9, col=1)

    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=2, col=1)
    fig.update_yaxes(title_text='Longitudinal Force', row=3, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=4, col=1)
    v_mean, p_mean, sr_mean = run_data.V.mean(), run_data.P.mean(), run_data.SL.mean()
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title} {v_mean}kph {p_mean} kPa, {sr_mean}SR")
    fig.show()

def split_run_fit_f(run_data, run_data_two, fig_title):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True) #,subplot_titles=("Fy-SA", "Mz-SA", "Fx-SA", "Mx-SA")

    color = px.colors.qualitative.Light24[0]
    color2 = px.colors.qualitative.Light24[1]

    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FY, marker=dict(color=color, size=2), mode='markers', showlegend=True, name="FY 1"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.MZ, marker=dict(color=color, size=2), mode='markers', showlegend=True, name="MZ 1"), row=2, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FX, marker=dict(color=color, size=2), mode='markers', showlegend=True, name="FX 1"), row=3, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FZ, marker=dict(color=color, size=2), mode='markers', showlegend=True, name="FZ 1"), row=4, col=1)
    
    fig.add_trace(go.Scattergl(x=run_data_two.ET, y=run_data_two.FY, marker=dict(color=color2, size=2), mode='markers', showlegend=True, name="FY 2"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=run_data_two.ET, y=run_data_two.MZ, marker=dict(color=color2, size=2), mode='markers', showlegend=True, name="MZ 2"), row=2, col=1)
    fig.add_trace(go.Scattergl(x=run_data_two.ET, y=run_data_two.FX, marker=dict(color=color2, size=2), mode='markers', showlegend=True, name="FX 2"), row=3, col=1)
    fig.add_trace(go.Scattergl(x=run_data_two.ET, y=run_data_two.FZ, marker=dict(color=color2, size=2), mode='markers', showlegend=True, name="FZ 2"), row=4, col=1)


    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=2, col=1)
    fig.update_yaxes(title_text='Longitudinal Force', row=3, col=1)
    fig.update_yaxes(title_text='Normal Force (N)', row=4, col=1)
    v_mean, p_mean, sr_mean = run_data.V.mean(), run_data.P.mean(), run_data.SL.mean()
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title} {v_mean}kph {p_mean} kPa, {sr_mean}SR")
    fig.show()

def split_run_fit_ml(run_data, out, fig_title):
    fig = make_subplots(rows=9, cols=1, shared_xaxes=True) #,subplot_titles=("Fy-SA", "Mz-SA", "Fx-SA", "Mx-SA")

    color = px.colors.qualitative.Light24[0]
    color2 = px.colors.qualitative.Light24[1]

    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FY, marker=dict(color=color), mode='markers', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.MZ, marker=dict(color=color), mode='markers', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FX, marker=dict(color=color), mode='markers', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.MX, marker=dict(color=color), mode='markers', showlegend=True, name="Data"), row=4, col=1)

    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 1], mode='markers', marker=dict(color=color2), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 3], mode='markers', marker=dict(color=color2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 0], mode='markers', marker=dict(color=color2), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=out[:, 2], mode='markers', marker=dict(color=color2), showlegend=True, name="Model"), row=4, col=1)

    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.P/1000, mode='lines', showlegend=True, name="P"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.V, mode='lines', showlegend=True, name="Vel"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.FZ, mode='lines', showlegend=True, name="FZ"), row=6, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.SA/3.14*180, mode='lines', showlegend=True, name="SA"), row=7, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.IA/3.14*180, mode='lines', showlegend=True, name="IA"), row=7, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.SR, mode='lines', showlegend=True, name="SR"), row=8, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.SL, mode='lines', showlegend=True, name="SL"), row=8, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.N, mode='lines', showlegend=True, name="N"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.TSTI, mode='lines', showlegend=True, name="TSTI"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.TSTC, mode='lines', showlegend=True, name="TSTC"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.TSTO, mode='lines', showlegend=True, name="TSTO"), row=5, col=1)
    fig.add_trace(go.Scattergl(x=run_data.ET, y=run_data.PHIT, mode='lines', showlegend=True, name="PHIT"), row=9, col=1)

    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    fig.update_yaxes(title_text='Lateral Force (N)', row=1, col=1)
    fig.update_yaxes(title_text='Self aligning moment (Nm)', row=2, col=1)
    fig.update_yaxes(title_text='Longitudinal Force', row=3, col=1)
    fig.update_yaxes(title_text='Overturning moment (Nm)', row=4, col=1)
    v_mean, p_mean, sr_mean = run_data.V.mean(), run_data.P.mean(), run_data.SL.mean()
    fig.update_layout(template="plotly_dark", title_text=f"{fig_title} {v_mean}kph {p_mean} kPa, {sr_mean}SR")
    fig.show()

def parameter_estimation_function(x1, run, tk: TireMFModel, params_list, error_list, loss, mask: ArrayLike, ts: bool = False):

    x0 = tk.dump_params()
    x0[mask] = x1
    params_list.append(x0)

    set_x(x0, tk)

    loss_num: ArrayLike = None
    inputs = np.array([run.FZ, run.SL, run.SA, run.IA, run.PHIT, run.V, run.P, run.N]).T
    out = tk.fullSteadyState(inputs, use_turnslip=ts)
    loss_num = np.array(loss(run, out))
    error_list.append(np.abs(loss_num).sum())
    return loss_num

def fit_f_y_norm(df, out):
    return (df.FY - out[:, 1])/df.FZ

def fit_f_x_norm(df, out):
    return (df.FX - out[:, 0])/df.FZ

def fit_f_y(df, out):
    return (df.FY - out[:, 1])

def fit_f_x(df, out):
    return (df.FX - out[:, 0])

def fit_m_z(df, out):
    return (df.MZ - out[:, 5])

def fit_m_x(df, out):
    return (df.MX - out[:, 3])


def gen_param_optimization_graph(params_list, mask, error_list):
    if len(params_list) < 1:
        pass
    else:
        np_params = np.array(params_list)
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Params over iters", "Mz-SA", "Fx-SA", "Mx-SA"))
        for x in mask:
            fig.add_trace(go.Scattergl(x=np.arange(
                np_params.shape[0]), y=np_params[:, x], mode='lines', name=f"{LABELS[x]} {NAMES[x]} {x}"), row=1, col=1)
        fig.update_xaxes(title_text='Iters', row=1, col=1)
        fig.update_yaxes(title_text='Param Vals', row=1, col=1)  # , type="log"
        np_el = np.array(error_list)
        fig.add_trace(go.Scattergl(x=np.arange(
            np_el.shape[0]), y=np_el, mode='lines', name=f"Iter: {x}"), row=2, col=1)
        fig.update_yaxes(title_text='Param Vals', row=2, col=1, type="log")
        fig.update_layout(template="plotly_dark", title_text="Parameter Optimization")
        fig.show()

def dump_param(param):
    par_str = "["
    for par in param:
        par_str += f"{par}, "
    par_str = par_str[:-2] + "]"
    print(par_str)

def flip_and_merge(df_raw):
    df = df_raw.copy()
    df.FY = df.FY.multiply(-1)
    df.MZ = df.MZ.multiply(-1)
    df.MX = df.MX.multiply(-1)
    df.SA = df.SA.multiply(-1)
    df.IA = df.IA.multiply(-1)
    df.PHIT = df.PHIT.multiply(-1)
    df.ET = df.ET.add(df.ET.max()-df.ET.min())
    return pd.concat([df_raw, df])

def merge(df_new, df_main = None):
    if df_main is None:
        return df_new
    if df_main.ET.min() > 0:
        df_main.ET = df_main.ET.add(-df_main.ET.min())
    df = df_new.copy()
    dt = np.diff(df.ET).min()
    df.ET = df_new.ET.add(df_main.ET.max() + 2 * dt)
    return pd.concat([df_main, df])

def get_model_error(tire_model, data, ts=False, abs=True):
    inputs = np.array([data.FZ, data.SL, data.SA, data.IA, data.PHIT, data.V, data.P, data.N]).T
    out = tire_model.fullSteadyState(inputs, use_turnslip=ts)
    fx = out[:, 0]
    fy = out[:, 1]
    mz = out[:, 5]
    fx_error = fx - data.FX
    fy_error = fy - data.FY
    mz_error = mz - data.MZ
    if abs:
        fx_error = np.abs(fx_error)
        fy_error = np.abs(fy_error)
        mz_error = np.abs(mz_error)
    return fx_error, fy_error, mz_error

def run_fit(tire_model, data, loss_func, mask, ts=False, graph=False, long=False, param_graph=False, ftol=0.001):
    params_list = []
    error_list = []
    sol = least_squares(parameter_estimation_function, tire_model.dump_params()[mask], args=(data, tire_model, params_list, error_list, loss_func, mask,), method='trf', jac='2-point', verbose=2, ftol=ftol)
    out = parameter_estimation_function(sol.x, data, tire_model, params_list, error_list, loss_func, mask)
    if graph:
        if long:
            split_run_with_MF_SR(data, tire_model, f"After Fx Optimization")
        else:
            split_run_with_MF_SA(data, tire_model, f"After Fy Optimization")
    if param_graph:
        gen_param_optimization_graph(params_list, mask, error_list)

def load_runs(run_args, flip=False, smooth=False):
    # Get the ttc data for the tire you want to fit
    cornering_raw, drive_brake_raw, name = run_args
    # create all the boring lists and stuff

    combi_runs = None
    cornering = None
    drive_brake = None
    for df_raw in cornering_raw:
        if smooth:
            df_raw = filter_eccentricity(df_raw)
        if flip:
            df = flip_and_merge(df_raw)
        else:
            df = df_raw
        combi_runs = merge(df, combi_runs)
        cornering = merge(df, cornering)

    for df_raw in drive_brake_raw:
        if smooth:
            df_raw = filter_eccentricity(df_raw)
        if flip:
            df = flip_and_merge(df_raw)
        else:
            df = df_raw
        combi_runs = merge(df, combi_runs)
        drive_brake = merge(df, drive_brake)
    return combi_runs, cornering, drive_brake, name