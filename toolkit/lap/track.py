import scipy.io as sio
from toolkit.loading_util import make_path
import numpy as np
import pymap3d as pm
from toolkit.lap.line_normals import linenormals2d
from toolkit.lap.line_curvature import linecurvature2D
from csaps import csaps
from scipy.ndimage import uniform_filter1d
from toolkit.common.maths import clean_interp
from toolkit.common.constants import *
from .channels import Channel
from .gps import GPS, smooth_gps

class Track:
    gps: GPS
    channels: dict[str, Channel]
    
    def __init__(self, gps: GPS, channels: dict[str, Channel], sc: float, freq: int, spl_sm: float) -> None:
        self.gps = gps
        self.channels = channels
        self.smooth_gps = smooth_gps(self.gps, sc, spl_sm)
        track = np.array([self.smooth_gps.x_track, self.smooth_gps.y_track])
        self.k = linecurvature2D(track.T) * -1
        self.k[0] = 0
        self.track_normals = linenormals2d(track.T)
        self.angle = np.rad2deg(np.unwrap(np.arctan2(self.track_normals[:, 1], self.track_normals[:, 0]) - np.average(np.arctan2(self.track_normals[0:10, 1], self.track_normals[0:10, 0])))) * -1
        self.u_crit = self.smooth_gps.dist
        self.make_k_prime()
        self.sc = int(sc / freq)
        self.vc = np.zeros(len(self.k))
        self.vc_r = np.zeros(len(self.k))
        self.import_car_data()
    
    @property
    def u(self):
        return self.smooth_gps.dist

    def make_k_prime(self):
        # Derivative of Curvature
        # Forward and Centered Difference Equations (Patton, 65)
        # note: Time Changes Depending on which Numerical Method is Utilized
        self.K_prime = np.zeros(len(self.k))
        self.K_prime[1:-1] = (self.k[2:] - self.k[:-2]) / (2 * self.smooth_gps.dist[1])
    
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
            self.nl_fl = self.raw_track["Normal_Load_FL"]["Value"][0, 0][0, :]
            self.nl_fr = self.raw_track["Normal_Load_FR"]["Value"][0, 0][0, :]
            self.nl_rl = self.raw_track["Normal_Load_RL"]["Value"][0, 0][0, :]
            self.nl_rr = self.raw_track["Normal_Load_RR"]["Value"][0, 0][0, :]
            self.ls_time = self.raw_track["Normal_Load_FL"]["Time"][0, 0][0, :] - self.start_time
        elif "Load_Cell__Force_FL" in data_keys:
            self.nl_fl = self.raw_track["Load_Cell__Force_FL"]["Value"][0, 0][0, :] * LB_TO_KG * G / mr
            self.nl_fr = self.raw_track["Load_Cell_Force_FR"]["Value"][0, 0][0, :] * LB_TO_KG * G / mr
            self.nl_rl = self.raw_track["Load_Cell_Force_RL"]["Value"][0, 0][0, :] * LB_TO_KG * G * -1 / mr
            self.nl_rr = self.raw_track["Load_Cell_Force_RR"]["Value"][0, 0][0, :] * LB_TO_KG * G * -1 / mr
            self.ls_time = self.raw_track["Load_Cell__Force_FL"]["Time"][0, 0][0, :] - self.start_time
        else:
            self.nl_fl = np.zeros(self.vel.shape[0])
            self.nl_fr = np.zeros(self.vel.shape[0])
            self.nl_rl = np.zeros(self.vel.shape[0])
            self.nl_rr = np.zeros(self.vel.shape[0])
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



    