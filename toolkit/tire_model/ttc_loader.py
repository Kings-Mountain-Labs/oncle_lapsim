from dataclasses import dataclass
import scipy.io as sio
from loading_util import make_path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from scipy.ndimage import uniform_filter1d

@dataclass
class DFNorms:
    f_y: float = 0.0
    f_x: float = 0.0
    m_x: float = 0.0
    m_z: float = 0.0
    num_pts: float = 0.0


def filter_sa(df):
    return df[(np.abs(df.SA) < 0.05)]

def filter_ia(df):
    return df[(np.abs(df.IA) < 0.02)]

def filter_phit(df):
    return df[(np.abs(df.PHIT) < 0.02)]

def filter_vel(df, vel=5.0, d_vel=None):
    if d_vel is None:
        return df[df.V > vel]
    return df[(np.abs(df.V - vel) < d_vel)]

def filter_press(df, press=82700, d_press=1000):
    return df[(np.abs(df.P - press) < d_press)]

def filter_sr(df):
    return df[(df.SL == 0)]

def filter_zero_sr(df):
    return df[(df.SL != 0)]

def filter_eccentricity(df):
    # find the edges of the each of the tests in data when the elapsed time (ET) jumps more than 50% more than the min diff time seconds
    diff_time = np.diff(df.ET)
    min_diff_time = max(diff_time.min(), 0.01) # make sure the min diff time is at least 0.01 seconds
    print(f"min diff time: {min_diff_time}")
    fs = 1 / min_diff_time
    edges = np.where(np.diff(df.ET) > (1.5 * min_diff_time))[0]
    # iterate through each section of data and apply the filter
    for i in range(len(edges)-1):
        f0 = (df.N[edges[i]:edges[i+1]]/np.pi/2).mean()
        window_size = int(fs / f0) + 1
        df.FZ[edges[i]:edges[i+1]] = uniform_filter1d(df.FZ[edges[i]:edges[i+1]], window_size)
        df.FX[edges[i]:edges[i+1]] = uniform_filter1d(df.FX[edges[i]:edges[i+1]], window_size)
        df.MZ[edges[i]:edges[i+1]] = uniform_filter1d(df.MZ[edges[i]:edges[i+1]], window_size)
        df.FY[edges[i]:edges[i+1]] = uniform_filter1d(df.FY[edges[i]:edges[i+1]], window_size)
    return df

def remove_time_gaps(df):
    # find the time gaps, where the time step is greater than 0.02 seconds
    time_diffs = df.ET.diff()
    time_diffs[time_diffs < 0.02] = 0
    shift = time_diffs.cumsum()
    # shift the time gaps to the start of the gap
    df.ET = df.ET - (shift + df.ET.min())
    return df

def load_ttc_from_mat(file_path: str):
    raw_ttc = sio.loadmat(file_path)
    df_dict = {'ET': raw_ttc['ET'][:, 0], 'V': raw_ttc['V'][:, 0], 'N': raw_ttc['N'][:, 0], 'SA': raw_ttc['SA'][:, 0],
        'IA': raw_ttc['IA'][:, 0], 'RL': raw_ttc['RL'][:, 0], 'RE': raw_ttc['RE'][:, 0], 'P': raw_ttc['P'][:, 0],
        'FX': raw_ttc['FX'][:, 0], 'FY': raw_ttc['FY'][:, 0], 'FZ': raw_ttc['FZ'][:, 0], 'MX': raw_ttc['MX'][:, 0],
        'MZ': raw_ttc['MZ'][:, 0], 'NFX': raw_ttc['NFX'][:, 0], 'NFY': raw_ttc['NFY'][:, 0], 'RST': raw_ttc['RST'][:, 0],
        'TSTI': raw_ttc['TSTI'][:, 0], 'TSTC': raw_ttc['TSTC'][:, 0], 'TSTO': raw_ttc['TSTO'][:, 0], 'AMBTMP': raw_ttc['AMBTMP'][:, 0],
        'SR': raw_ttc['SR'][:, 0], 'SL': raw_ttc['SL'][:, 0]}
    df = pd.DataFrame(df_dict)
    return df

def sae_to_iso(df_raw):
    df = df_raw.copy()
    df.FZ = df.FZ.multiply(-1)
    df.MZ = df.MZ.multiply(-1)
    df.NFY = df.NFY.multiply(-1)
    df.FY = df.FY.multiply(-1)
    df.SA = df.SA.multiply(-1)
    return df

def iso_to_sae(df_raw):
    df = sae_to_iso(df_raw.copy()) # it is a symmetric transformation
    return df

def iso_to_aiso(df_raw):
    df = df_raw.copy()
    df.SA = df.SA.multiply(-1)
    df.IA = df.IA.multiply(-1)
    return df

def iso_to_asae(df_raw):
    df = df_raw.copy()
    df.FZ = df.FZ.multiply(-1)
    df.MZ = df.MZ.multiply(-1)
    df.NFY = df.NFY.multiply(-1)
    df.FY = df.FY.multiply(-1)
    return df

def metric_to_si(df_raw):
    df = sae_to_iso(df_raw)
    df.SA = df.SA.multiply(np.pi / 180)
    df.IA = df.IA.multiply(np.pi / 180)
    df.P = df.P.multiply(1000)
    vel = df.V / 3.6
    vel[vel <= 0] = vel[vel > 0].min()
    df.V = vel
    # This is the only good PHIT explication I have been able to find
    # https://www.mathworks.com/help/sm/ref/magicformulatireforceandtorque.html
    u_phit = (np.gradient(df.SA) * 100) / df.V
    df["PHIT"] = u_phit
    # make phit 0 when the time step is greater than 0.02 seconds
    df.PHIT[df.ET.diff() > 0.01] = 0
    # df = df[np.abs(df.PHIT) < 0.1]
    u_slt = (np.gradient(df.SL) * 100) / df.V
    df["SLT"] = u_slt
    df.SLT[df.ET.diff() > 0.03] = 0
    df["T"] = df.RL * df.FX + df.MZ * np.sin(df.IA) # RCVD Figure 2.33 but missing the My * sin(IA) term
    df.N = df.N.multiply(2*np.pi/60)
    df.RE = df.RE.multiply(1/100)
    df.RL = df.RL.multiply(1/100)
    return df

def convert_ttc_to_metric(raw_ttc):
    raw_ttc.FX = raw_ttc.FX * 4.448
    raw_ttc.FY = raw_ttc.FY * 4.448
    raw_ttc.FZ = raw_ttc.FZ * -4.448
    raw_ttc.MX = raw_ttc.MX * 1.3558179
    raw_ttc.MZ = raw_ttc.MZ * 1.3558179
    raw_ttc.AMBTMP = (raw_ttc.AMBTMP - 32) * 5/9
    raw_ttc.RST = (raw_ttc.RST - 32) * 5/9
    raw_ttc.TSTI = (raw_ttc.TSTI - 32) * 5/9
    raw_ttc.TSTC = (raw_ttc.TSTC - 32) * 5/9
    raw_ttc.TSTO = (raw_ttc.TSTO - 32) * 5/9
    return raw_ttc

def plot_ttc_run(run_data):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1, sharex=True)
    ax1.plot(run_data.ET, run_data.FX, label="FX")
    ax1.plot(run_data.ET, run_data.FY, label="FY")
    ax1.plot(run_data.ET, run_data.FZ, label="FZ")
    ax1.set_ylabel('Force (N)')
    ax1.legend()

    ax2.plot(run_data.ET, run_data.SA, label="Slip Angle")
    ax2.plot(run_data.ET, run_data.IA, label="Inclination Angle")
    ax2.set_ylabel('Angle (deg)')
    ax2.legend()

    ax3.plot(run_data.ET, run_data.MX, label="Overturning Moment")
    ax3.plot(run_data.ET, run_data.MZ, label="Aligning Moment")
    ax3.set_ylabel('Torque (N m)')
    ax3.legend()

    ax4.plot(run_data.ET, run_data.SL, label="Slip Ratio SE")
    ax4.plot(run_data.ET, run_data.SR, label="Slip Ratio RL")
    ax4.set_ylabel('Slip Ratio')
    ax4.legend()

    ax5.plot(run_data.ET, run_data.AMBTMP, label="Ambient Temp")
    ax5.plot(run_data.ET, run_data.RST, label="Road Surface")
    ax5.plot(run_data.ET, run_data.TSTI, label="Tire Inside")
    ax5.plot(run_data.ET, run_data.TSTC, label="Tire Center")
    ax5.plot(run_data.ET, run_data.TSTO, label="Tire Outside")
    ax5.set_ylabel('Temperature (C)')
    ax5.legend()

    ax6.plot(run_data.ET, run_data.RE, label="Effective Radius")
    ax6.plot(run_data.ET, run_data.RL, label="Loaded Radius")
    ax6.set_ylabel('Tire Radius (cm)')
    ax6.legend()

    ax7.plot(run_data.ET, run_data.P, label="Pressure")
    ax7.set_ylabel('Tire Press (kPa)')
    ax7.legend()

    ax8.plot(run_data.ET, run_data.V, label="Road Speed")
    ax8.set_ylabel('Road Speed (kph')
    ax8.legend()
    ax8.set_xlabel('Time (s)')

def split_run(run_data, fig_title):
    fig2, ((ax6, ax7), (ax8, ax9)) = plt.subplots(2, 2, sharex=True)
    for f_z in [230, 670, 1100, 1540]:
        for i_c in [0, 2, 4]:
            ds = run_data[(np.abs(run_data.FZ - f_z) < 100) & (np.abs(run_data.IA - i_c) < 0.5)]
            title = f"{f_z}N, {i_c}deg"
            sa_deg = ds.SA * 180 / np.pi
            ax6.scatter(sa_deg, ds.FY, s = 0.2, label = title)
            ax7.scatter(sa_deg, ds.MZ, s = 0.2, label = title)
            ax8.scatter(sa_deg, ds.FY, s = 0.2, label = title)
            ax9.scatter(sa_deg, ds.MX, s = 0.2, label = title)

    ax6.set_title('Fy-SA')
    ax6.set_xlabel('Slip Angle (deg)')
    ax6.set_ylabel('Lateral Force (N)')
    ax6.legend()
    ax7.set_title('Mz-SA')
    ax7.set_xlabel('Slip Angle (deg)')
    ax7.set_ylabel('Self aligning moment (Nm)')
    ax7.legend()
    ax8.set_title('t-SA')
    ax8.set_xlabel('Slip Angle (deg)')
    ax8.set_ylabel('pneumatic trail (m)')
    ax8.legend()
    ax9.set_title('Mx-SA')
    ax9.set_xlabel('Slip Angle (deg)')
    ax9.set_ylabel('Overturning moment (Nm)')
    ax9.legend()
    fig2.canvas.set_window_title(fig_title)


# R25B 18x6 6rim
def get_R25B_18x6_6_runs():
    cornering: List = []
    drive_brake: List = []
    r_18 = metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run18.mat')))
    r_18 = r_18[np.logical_not(((r_18.ET > 523.0) & (r_18.ET < 540.0)) | ((r_18.ET > 679.0) & (r_18.ET < 685.0)) | ((r_18.ET > 958.0) & (r_18.ET < 975.0)))] # 12 psi
    r_18 = r_18[np.logical_not(((r_18.ET > 1348.0) & (r_18.ET < 1398.0)) | ((r_18.ET > 1468.0) & (r_18.ET < 1485.0)) | ((r_18.ET > 1526.0) & (r_18.ET < 1542.0)) | ((r_18.ET > 1816.0) & (r_18.ET < 1833.0)) | ((r_18.ET > 1961.0) & (r_18.ET < 1978.0)))] # 10 psi
    r_18 = r_18[np.logical_not(((r_18.ET > 2239.0) & (r_18.ET < 2256.0)) | ((r_18.ET > 2674.0) & (r_18.ET < 2691.0)))] # 14 psi
    cornering.append(r_18)
    r_19 = metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run19.mat')))
    r_19 = r_19[np.logical_not(((r_19.ET > 37.0) & (r_19.ET < 207.0)) | ((r_19.ET > 306.0) & (r_19.ET < 323.0)) | ((r_19.ET > 451.0) & (r_19.ET < 468.0)) | ((r_19.ET > 596.0) & (r_19.ET < 613.0)) | ((r_19.ET > 683.0) & (r_19.ET < 700.0)) | ((r_19.ET > 741.0) & (r_19.ET < 758.0)))] # 8 psi
    r_19 = r_19[np.logical_not(((r_19.ET > 1019.0) & (r_19.ET < 1036.0)) | ((r_19.ET > 1106.0) & (r_19.ET < 1123.0)) | ((r_19.ET > 1164.0) & (r_19.ET < 1181.0)) | ((r_19.ET > 1454.0) & (r_19.ET < 1471.0)))] # 12 psi
    cornering.append(r_19)
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run36.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run37.mat'))))
    return cornering, drive_brake, "Hoosier_R25B_18x6_6"

# R25B 18x6 7rim
def get_R25B_18x6_7_runs():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run20.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run21.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run39.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run40.mat'))))
    return cornering, drive_brake, "Hoosier_R25B_18x6_7"

# LC0 18x6 6rim
def get_LC0_18x6_6_runs():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run22.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run23.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run29.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run30.mat'))))
    return cornering, drive_brake, "Hoosier_LC0_18x6_6"

# LC0 18x6 7rim
def get_LC0_18x6_7_runs():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run24.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run25.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run33.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1464run34.mat'))))
    return cornering, drive_brake, "Hoosier_LC0_18x6_7"

# R25B 18x7.5 7rim
def get_R25B_18x75_7_runs():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run20.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run21.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run22.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run35.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run36.mat'))))
    return cornering, drive_brake, "Hoosier_R25B_18x7.5_7"

# R25B 18x7.5 8rim
def get_R25B_18x75_8_runs():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run23.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run24.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run25.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run38.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B1654run39.mat'))))
    return cornering, drive_brake, "Hoosier_R25B_18x7.5_8"

# R20 18x6 6" rim
def get_R20_18x6_6_runs():
    cornering: List = []
    drive_brake: List = []
    r_28 = metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run28.mat')))
    r_28 = r_28[np.logical_not(((r_28.ET > 885.0) & (r_28.ET < 899.0)) | ((r_28.ET > 1020.0) & (r_28.ET < 1034.0)))]
    cornering.append(r_28)
    r_29 = metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run29.mat')))
    r_29 = r_29[np.logical_not(((r_29.ET > 109.0) & (r_29.ET < 121.5)) | ((r_29.ET > 142.0) & (r_29.ET < 156.0)) | ((r_29.ET > 244.0) & (r_29.ET < 257.0)) | ((r_29.ET > 413.0) & (r_29.ET < 426.0)))]
    cornering.append(r_29)
    # cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run68.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run69.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run70.mat'))))
    return cornering, drive_brake, "Hoosier_R20_18x6_6"

# R20 18x6 7" rim
def get_R20_18x6_7_runs():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run31.mat'))))
    r_32 = metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run32.mat')))
    r_32 = r_32[np.logical_not(((r_32.ET > 93.0) & (r_32.ET < 121.5)) | ((r_32.ET > 42.0) & (r_32.ET < 156.0)) | ((r_32.ET > 244.0) & (r_32.ET < 257.0)))]
    cornering.append(r_32)
    # cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run71.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run72.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run73.mat'))))
    return cornering, drive_brake, "Hoosier_R20_18x6_7"

# R20 18x6 7" rim
def get_R20_18x6_7_runs_raw():
    cornering: List = []
    drive_brake: List = []
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run31.mat'))))
    cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run32.mat'))))
    # cornering.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run71.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run72.mat'))))
    drive_brake.append(metric_to_si(load_ttc_from_mat(make_path('./Data/TTCData/B2356run73.mat'))))
    return cornering, drive_brake, "Hoosier_R20_18x6_7"
