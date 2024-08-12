import scipy.io as sio
from loading_util import make_path
import numpy as np
import pymap3d as pm
from toolkit.lap.line_normals import linenormals2d
from toolkit.lap.line_curvature import linecurvature2D
from math_channels import add_contact_patch_load, add_corrected_acc, add_corrected_shockpots, add_lltd_chans, add_suspension_forces, get_data
from csaps import csaps
from scipy.ndimage import uniform_filter1d
from toolkit.common.maths import clean_interp
from toolkit.common.constants import *

class Track:
    def __init__(self, raw_track, x_track, y_track, z_track, sc, freq, spl_sm) -> None:
        self.gps_offset = 0.05 # 50ms offset to account for the time it takes for the GPS to update, which is half of the interval between samples
        self.raw_time = raw_track["GPS_Longitude"]["Time"][0, 0][0, :] - self.gps_offset
        self.start_time = self.raw_time[0]
        self.raw_time -= self.start_time
        self.track_x, self.track_y, self.track_z = x_track, y_track, z_track
        self.x_out_raw, self.y_out_raw, self.z_out_raw = interp_track(x_track, y_track, z_track, int(sc / freq), spl_sm=0.1)
        self.dist, ind, self.raw_dist = calc_distance(self.x_out_raw, self.y_out_raw)
        self.x_smooth, self.y_smooth, self.z_smooth = self.x_out_raw[ind], self.y_out_raw[ind], self.z_out_raw[ind]
        self.raw_track = raw_track
        self.u = np.linspace(0, max(self.dist), len(self.dist))
        self.time = clean_interp(np.linspace(0, len(self.raw_time), self.raw_dist.shape[0]), np.arange(len(self.raw_time)), self.raw_time)
        self.u_time = clean_interp(self.u, self.raw_dist, self.time)
        out = csaps(self.raw_dist, np.array([self.x_out_raw, self.y_out_raw, self.z_out_raw]), self.u, smooth=0.5)
        track = np.array([out[0, :], out[1, :]])
        self.k = linecurvature2D(track.T) * -1
        self.k[0] = 0
        self.track_normals = linenormals2d(track.T)
        self.angle = np.rad2deg(np.unwrap(np.arctan2(self.track_normals[:, 1], self.track_normals[:, 0]) - np.average(np.arctan2(self.track_normals[0:10, 1], self.track_normals[0:10, 0])))) * -1
        self.u_crit = self.u
        self.x_ss = out[0, :]
        self.y_ss = out[1, :]
        self.z_ss = out[2, :]
        self.make_k_prime()
        self.interp_gps()
        self.sc = int(sc / freq)
        self.samp = sc
        self.freq = freq
        self.vc = np.zeros(len(self.u))
        self.vc_r = np.zeros(len(self.u))
        self.import_car_data()
    

    def interp_gps(self):
        lat, lon, height = self.raw_track["GPS_Latitude"]["Value"][0, 0][0], self.raw_track["GPS_Longitude"]["Value"][0, 0][0], self.raw_track["GPS_Altitude"]["Value"][0, 0][0]
        lat_origin, lon_origin, height_origin = np.mean(lat), np.mean(lon), np.mean(height)
        self.lat_ss, self.lon_ss, self.height_ss = pm.enu2geodetic(self.x_ss, self.y_ss, self.z_ss, lat_origin, lon_origin, height_origin)

    def make_k_prime(self):
        # Derivative of Curvature
        # Forward and Centered Difference Equations (Patton, 65)
        # note: Time Changes Depending on which Numerical Method is Utilized
        self.K_prime = np.zeros(len(self.k))
        self.K_prime[1:-1] = (self.k[2:] - self.k[:-2]) / (2 * self.u[1])
    
    def import_car_data(self):
        self.vel = self.raw_track["GPS_Speed"]["Value"][0, 0][0, :-4]
        if self.raw_track["GPS_Speed"]["Units"] != "m/s":
            self.vel = self.vel * KMH_TO_MS
        self.average_vel = np.average(self.vel)
        self.ddt = np.diff(self.time)
        self.interp_dist = self.raw_dist[::self.sc]
        self.total_time = self.vel.shape[0] / self.freq
        self.ddt_cumsum = self.time[::self.sc]
        data_keys = list(self.raw_track.keys())
        self.data_keys = data_keys
        if "Wheel_Speed_FL" in data_keys:
            self.ws_fl = self.raw_track["Wheel_Speed_FL"]["Value"][0, 0][0, :]
            self.ws_fr = self.raw_track["Wheel_Speed_FR"]["Value"][0, 0][0, :]
            self.ws_rl = self.raw_track["Wheel_Speed_RL"]["Value"][0, 0][0, :]
            self.ws_rr = self.raw_track["Wheel_Speed_RR"]["Value"][0, 0][0, :]
        elif "VCU_WSS_FL_S" in data_keys:
            self.ws_fl = self.raw_track["VCU_WSS_FL_S"]["Value"][0, 0][0, :]
            self.ws_fr = self.raw_track["VCU_WSS_FR_S"]["Value"][0, 0][0, :]
            self.ws_rl = self.raw_track["VCU_WSS_RL_S"]["Value"][0, 0][0, :]
            self.ws_rr = self.raw_track["VCU_WSS_RR_S"]["Value"][0, 0][0, :]
        else:
            self.ws_fl = self.raw_track["Wheel_Speed_FL"]["Value"][0, 0][0, :]
            self.ws_fr = self.raw_track["Wheel_Speed_FR"]["Value"][0, 0][0, :]
            self.ws_rl = self.raw_track["Wheel_Speed_RL"]["Value"][0, 0][0, :]
            self.ws_rr = self.raw_track["Wheel_Speed_RR"]["Value"][0, 0][0, :]
        if "Steering_Angle" in data_keys:
            self.steer = self.raw_track["Steering_Angle"]["Value"][0, 0][0, :]
        elif "Steering_Wheel_Angle" in data_keys:
            self.steer = self.raw_track["Steering_Wheel_Angle"]["Value"][0, 0][0, :]
        elif "Driver_Steering" in data_keys:
            self.steer = self.raw_track["Driver_Steering"]["Value"][0, 0][0, :]
        else:
            self.steer = self.raw_track["Steering_Wheel_Angle"]["Value"][0, 0][0, :]
        mr = 1.5
        if "Normal_Load_FL" in data_keys:
            self.loadcell_fl = self.raw_track["Normal_Load_FL"]["Value"][0, 0][0, :]
            self.loadcell_fr = self.raw_track["Normal_Load_FR"]["Value"][0, 0][0, :]
            self.loadcell_rl = self.raw_track["Normal_Load_RL"]["Value"][0, 0][0, :]
            self.loadcell_rr = self.raw_track["Normal_Load_RR"]["Value"][0, 0][0, :]
            self.ls_time = self.raw_track["Normal_Load_FL"]["Time"][0, 0][0, :] - self.start_time
        elif "Load_Cell__Force_FL" in data_keys:
            self.loadcell_fl = self.raw_track["Load_Cell__Force_FL"]["Value"][0, 0][0, :] * LB_TO_KG * G / mr
            self.loadcell_fr = self.raw_track["Load_Cell_Force_FR"]["Value"][0, 0][0, :] * LB_TO_KG * G / mr
            self.loadcell_rl = self.raw_track["Load_Cell_Force_RL"]["Value"][0, 0][0, :] * LB_TO_KG * G * -1 / mr
            self.loadcell_rr = self.raw_track["Load_Cell_Force_RR"]["Value"][0, 0][0, :] * LB_TO_KG * G * -1 / mr
            self.ls_time = self.raw_track["Load_Cell__Force_FL"]["Time"][0, 0][0, :] - self.start_time
        else:
            self.loadcell_fl = np.zeros(self.vel.shape[0])
            self.loadcell_fr = np.zeros(self.vel.shape[0])
            self.loadcell_rl = np.zeros(self.vel.shape[0])
            self.loadcell_rr = np.zeros(self.vel.shape[0])
            self.ls_time = self.raw_time

        if "IMU_Acceleration_X" in data_keys:
            self.lat_acc = uniform_filter1d(self.raw_track["IMU_Acceleration_Y"]["Value"][0, 0][0, :]*9.81, 25)
            self.long_acc = uniform_filter1d(self.raw_track["IMU_Acceleration_X"]["Value"][0, 0][0, :]*9.81, 25)
            self.spa_t = self.raw_track["IMU_Acceleration_X"]["Time"][0, 0][0, :] - self.start_time
        elif "Aceinna_AccX" in data_keys:
            self.lat_acc = uniform_filter1d(self.raw_track["Aceinna_AccX"]["Value"][0, 0][0, :], 25) * -1
            self.long_acc = uniform_filter1d(self.raw_track["Aceinna_AccY"]["Value"][0, 0][0, :], 25)
            self.spa_t = self.raw_track["Aceinna_AccX"]["Time"][0, 0][0, :] - self.start_time
        elif "G_Force_Lat" in data_keys:
            self.lat_acc = uniform_filter1d(self.raw_track["G_Force_Lat"]["Value"][0, 0][0, :]*-9.81, 5)
            self.long_acc = uniform_filter1d(self.raw_track["G_Force_Long"]["Value"][0, 0][0, :]*9.81, 5)
            self.spa_t = self.raw_track["G_Force_Lat"]["Time"][0, 0][0, :] - self.start_time
        self.spa = clean_interp(self.spa_t, self.time, self.raw_dist)
        self.ls_dist = clean_interp(self.ls_time, self.u_time, self.u)
        self.ws_avg = (self.ws_fl + self.ws_fr + self.ws_rl + self.ws_rr) / 4

        if "Vehicle_Yaw_Rate" in self.data_keys:
            self.y_r = np.deg2rad(self.raw_track["Vehicle_Yaw_Rate"]["Value"][0, 0][0, :])
            self.yaw_freq = self.y_r.shape[0] / (self.raw_track["Vehicle_Yaw_Rate"]["Time"][0, 0][0, -1] - self.raw_track["Vehicle_Yaw_Rate"]["Time"][0, 0][0, 0])
        elif "IMU_Angular_Rate_Z" in self.raw_track.keys():
            self.y_r = np.deg2rad(self.raw_track["IMU_Angular_Rate_Z"]["Value"][0, 0][0, :])
            self.yaw_freq = self.y_r.shape[0] / (self.raw_track["IMU_Angular_Rate_Z"]["Time"][0, 0][0, -1] - self.raw_track["IMU_Angular_Rate_Z"]["Time"][0, 0][0, 0])
        elif "Aceinna_GyroZ" in self.raw_track.keys():
            self.y_r = uniform_filter1d(np.deg2rad(self.raw_track["Aceinna_GyroZ"]["Value"][0, 0][0, :]), 10)
            self.yaw_freq = self.y_r.shape[0] / (self.raw_track["Aceinna_GyroZ"]["Time"][0, 0][0, -1] - self.raw_track["Aceinna_GyroZ"]["Time"][0, 0][0, 0])
        else:
            self.y_r = np.zeros(self.spa_t.shape[0])
            self.yaw_freq = self.y_r.shape[0] / (self.spa_t[-1] - self.spa_t[0])
        self.y_a = np.zeros(self.y_r.shape)
        self.y_a[1:] = np.diff(uniform_filter1d(self.y_r, 20)) * self.yaw_freq
        self.real_vel = np.zeros(self.u.shape)
        self.real_vel[1:] = self.u[1:] / np.diff(self.u_time)
        self.real_vel[0] = self.u[0] / (self.u_time[1] - self.u_time[0])
        self.real_vel = uniform_filter1d(self.real_vel, 10)
        self.y_r_k = (self.k * self.real_vel)
        self.y_r_b = self.y_r - clean_interp(self.spa, self.u, self.y_r_k)
        self.real_angle_raw = np.cumsum(self.y_r) * 1/self.yaw_freq
        self.real_angle = self.real_angle_raw - uniform_filter1d(self.real_angle_raw - np.deg2rad(clean_interp(self.spa, self.u, self.angle)), 1000) # this should probably be a gaussian filter
        self.real_beta = clean_interp(self.u, self.spa, np.rad2deg(self.real_angle)) - self.angle

def interp_track(x, y, z, sc, spl_sm = 0.85):
    space = np.arange(len(x))
    val = np.linspace(0, len(x), len(x) * sc)
    out = csaps(space, np.array([x, y, z]), val, smooth=spl_sm)
    return out[0, :], out[1, :], out[2, :] # remove some nastyness at the end

def calc_distance(x_pts, y_pts):
    delta_dist = np.zeros(x_pts.shape)
    delta_dist[1:] = np.sqrt((x_pts[:-1] - x_pts[1:])**2 + (y_pts[:-1] - y_pts[1:])**2)
    jump_ind = np.where(delta_dist > 0.0001)
    dist = np.cumsum(delta_dist[jump_ind])
    return dist, jump_ind, np.cumsum(delta_dist)

def load_track_from_mat(file_path: str):
    return sio.loadmat(make_path(file_path))

def load_track_from_raw(raw_track, sc, freq, spl_sm = 0.85):
    # check to see if GPS Altitude was recorded, if not make it and fill it with zeros
    if "GPS_Altitude" not in raw_track.keys():
        raw_track["GPS_Altitude"] = {"Time": raw_track["GPS_Latitude"]["Time"].copy(), "Value": raw_track["GPS_Latitude"]["Time"].copy()}
        raw_track["GPS_Altitude"]["Value"][0, 0][0, :] = np.zeros(raw_track["GPS_Altitude"]["Time"][0, 0][0, :].shape)
    lat, lon, height = raw_track["GPS_Latitude"]["Value"][0, 0][0], raw_track["GPS_Longitude"]["Value"][0, 0][0], raw_track["GPS_Altitude"]["Value"][0, 0][0]
    return load_track_lat_lon(lat, lon, height, raw_track, sc, freq, spl_sm)

def load_track_lat_lon(lat, lon, height, raw_track, sc, freq, spl_sm):
    lat_origin, lon_origin, height_origin = np.mean(lat), np.mean(lon), np.mean(height)
    [x_track, y_track, z_track] = pm.geodetic2enu(lat, lon, height, lat_origin, lon_origin, height_origin)
    return Track(raw_track, x_track, y_track, z_track, sc, freq, spl_sm)

def get_MIS_2017_track(sc) -> Track:
    return load_track_from_mat('./Data/TrackMaps/2017_Michigan_Endurance.mat', sc, 10, spl_sm = 0.95)

def get_MIS_2017_End1_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Michigan_Endurance_First4Laps.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)
    
def get_MIS_2017_End2_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Michigan_Endurance_Second4Laps.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2017_AX_1_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Michigan_Skid_Accel_AutoX_Lap1.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2017_AX_2_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Michigan_Skid_Accel_AutoX_Lap2.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2017_AX_3_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Michigan_Skid_Accel_AutoX_Lap3.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2017_AX_4_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Michigan_Skid_Accel_AutoX_Lap4.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_Crows_2023(sc) -> Track: # Lap 2 of milos 2023-10-12 where it cut out
    raw_track = load_track_from_mat('./Data/TrackMaps/20231008-0910601-enduro1-milos.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_Crows_2022(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2022_Crows_Testlap.mat')
    a, b, c = raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"]
    a["Value"][0,0][0] =  a["Value"][0,0][0] - 1
    raw_track["G_Force_Vert"], raw_track["G_Force_Lat"], raw_track["G_Force_Long"] = c, a, b
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2019_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2019_MIS_ENDURANCE.mat')
    return load_track_from_raw(raw_track, sc, 1, spl_sm = 1.0)

def get_MIS_2018_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2018_MIS_ENDURANCE.mat')
    return load_track_from_raw(raw_track, sc, 1, spl_sm = 1.0)

def get_MIS_2021_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/SR12B_MIS2021_Autocross_3.mat')
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_Lincoln_2017_AX_track(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Lincoln_Autocross_Bobby_2.mat')
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_Lincoln_2017_AX_track_mixed(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2017_Lincoln_Autocross_Bobby_2_Mixed.mat')
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2021_track_mixed(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/2021_MIS_Endurance_2Laps.mat')
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_PNR_2022_11_6_Patton(sc) -> Track:
    raw_track = load_track_from_mat('./Data/TrackMaps/20221106-PNR-Clean_Lap.mat')
    return load_track_from_raw(raw_track, sc, 10, spl_sm = 0.95)

def get_MIS_2017_track_raw(sc) -> Track:
    dp = get_data(make_path('./Data/logs/2017_Michigan_Endurance.ld'), sc)
    print(f"{dp.keys()}")
    dp = add_corrected_acc(dp)
    dp = add_corrected_shockpots(dp)
    dp = add_suspension_forces(dp)
    dp = add_contact_patch_load(dp)
    dp = add_lltd_chans(dp)
    # dg = dp.loc[dp["Corr_Speed"] > 10]
    # dg = dp.loc[(dp["FLLTD_Real"]<1)&(dp["FLLTD_Real"]>0)&(dp["vehicle_yaw_rate"].abs()>5)&(dp["Corr_Speed"]>15&(dp["Corr_Yaw_Accel"].abs()<0.3))]
    lat, lon, height = dp["gps_latitude"].dropna(), dp["gps_longitude"].dropna(), dp["gps_altitude"].dropna()
    return load_track_lat_lon(lat, lon, height, dp, sc)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    track = get_MIS_2017_AX_3_track(50)

    fig = plt.figure()
    ax = fig.add_subplot()
    # ax.quiver(track.x_smooth, track.y_smooth, track.track_normals[:, 0] * track.k * 100, track.track_normals[:, 1] * track.k * 100)
    # ax.scatter(track.x_smooth, track.y_smooth, c=cm.cool(track.K_prime/track.K_prime.max()), s=0.5)
    ax.scatter(track.x_ss, track.y_ss, s=5) # , c=cm.cool(track.K_prime/track.K_prime.max())
    # ax.plot(track.x_out_raw, track.y_out_raw)
    ax.plot(track.track_x, track.track_y)
    ax.set_aspect('equal', 'box')
    

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    print(track.raw_track["G_Force_Vert"])
    ax2.plot(track.raw_track["G_Force_Vert"]["Time"][0, 0][0], track.raw_track["G_Force_Vert"]["Value"][0, 0][0], label="Vert")
    ax2.plot(track.raw_track["G_Force_Long"]["Time"][0, 0][0], track.raw_track["G_Force_Long"]["Value"][0, 0][0], label="Long")
    ax2.plot(track.raw_track["G_Force_Lat"]["Time"][0, 0][0], track.raw_track["G_Force_Lat"]["Value"][0, 0][0], label="Lat")
    ax2.legend()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.plot(track.spa_t, track.long_acc, label="Long")
    ax3.plot(track.spa_t, track.lat_acc, label="Lat")
    ax3.legend()

    plt.show()


    
