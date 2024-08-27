import scipy.io as sio
from toolkit.loading_util import make_path
import numpy as np
import pymap3d as pm
from toolkit.lap.line_normals import linenormals2d
from toolkit.lap.line_curvature import linecurvature2d
from csaps import csaps
from scipy.ndimage import uniform_filter1d
from toolkit.common.maths import clean_interp
from toolkit.common.constants import *
from .channels import Channel, null_channel
from .gps import GPS, smooth_gps
import plotly.graph_objects as go

WELL_KNOWN_KEYS = ["__gps_vel", "__ws_fl", "__ws_fr", "__ws_rl", "__ws_rr", "__nl_fl", "__nl_fr", "__nl_rl", "__nl_rr", "__steering_angle", "__acc_x", "__acc_y", "__acc_z", "__gyro_x", "__gyro_y", "__gyro_z", "__beta_angle", "__delta_angle"]

"""
For the track there are a set of 'good' 'well known' channel names that are to be used for showing data in the graphs that are hard coded into the toolkit
All of these channels use a dunder as a prefix to the name to prevent collisions with names used for logging

The following channels are used in the toolkit:
- __gps_vel - GPS Speed in m/s
- __ws_fl - Wheel Speed in front left wheel in rpm
- __ws_fr - Wheel Speed in front right wheel in rpm
- __ws_rl - Wheel Speed in rear left wheel in rpm
- __ws_rr - Wheel Speed in rear right wheel in rpm
- __nl_fl - Contact patch normal force in front left
- __nl_fr - Contact patch normal force in front right
- __nl_rl - Contact patch normal force in rear left
- __nl_rr - Contact patch normal force in rear right
- __steering_angle - Steering Angle in degrees
- __acc_x - Acceleration in x direction in m/s^2
- __acc_y - Acceleration in y direction in m/s^2
- __acc_z - Acceleration in z direction in m/s^2
- __gyro_x - Gyro Rate in x direction in rad/s
- __gyro_y - Gyro Rate in y direction in rad/s
- __gyro_z - Gyro Rate in z direction in rad/s
- __beta_angle - Beta Angle in degrees
- __delta_angle - Delta Angle in degrees
"""

class Track:
    gps: GPS
    channels: dict[str, Channel]
    
    def __init__(self, gps: GPS, channels: dict[str, Channel], sc: float, spl_sm: float) -> None:
        self.gps = gps
        self.channels = channels
        self.smooth_gps = smooth_gps(self.gps, sc, spl_sm)
        track = np.array([self.smooth_gps.x_track, self.smooth_gps.y_track])
        self.k = linecurvature2d(track.T) * -1
        self.k[0] = 0
        self.track_normals = linenormals2d(track.T)
        self.angle = np.rad2deg(np.unwrap(np.arctan2(self.track_normals[:, 1], self.track_normals[:, 0]) - np.average(np.arctan2(self.track_normals[0:10, 1], self.track_normals[0:10, 0])))) * -1
        self.u_crit = self.smooth_gps.dist
        self.make_k_prime()
        self.sc = int(sc / gps.freq)
        # If they havent added one of the important channels, add it
        for key in WELL_KNOWN_KEYS:
            if key not in self.channels.keys():
                print(f"Channel {key} not found in track, adding it a null channel")
                self.channels[key] = null_channel(key, time_offset=self.gps.laptime_datum)
        self.vc = np.zeros(len(self.k))
        self.vc_r = np.zeros(len(self.k))
        # self.import_car_data()
    
    @property
    def u(self):
        return self.smooth_gps.dist
    
    @property
    def u_time(self):
        return self.smooth_gps.time

    def make_k_prime(self):
        # Derivative of Curvature
        # Forward and Centered Difference Equations (Patton, 65)
        # note: Time Changes Depending on which Numerical Method is Utilized
        self.k_prime = np.zeros(len(self.k))
        self.k_prime[1:-1] = (self.k[2:] - self.k[:-2]) / (2 * self.u[1])
    
    def get_channel(self, name, distance=False, datum=True) -> tuple[np.ndarray, np.ndarray]:
        if not name in self.channels.keys():
            print(f"Channel {name} not found in track")
            return np.array([0]), np.array([0])
        if distance:
            return np.interp(self.channels[name].time, self.smooth_gps.raw_time, self.u), self.channels[name].data
        else:
            if datum:
                return self.channels[name].time - self.gps.laptime_datum, self.channels[name].data
            else:
                return self.channels[name].time, self.channels[name].data

    def get_channel_go(self, name, distance=False) -> go.Scattergl:
        time, data = self.get_channel(name, distance)
        return go.Scattergl(x=time, y=data, mode='lines', name=name)

    # def import_car_data(self):
    #     self.vel = self.raw_track["GPS_Speed"]["Value"][0, 0][0, :-4]
    #     if self.raw_track["GPS_Speed"]["Units"] != "m/s":
    #         self.vel = self.vel * KMH_TO_MS
    #     self.ddt = np.diff(self.time)
    #     self.interp_dist = self.raw_dist[::self.sc]
    #     self.ddt_cumsum = self.time[::self.sc]
    #     if "Wheel_Speed_FL" in data_keys:
    #         self.ws_fl = self.raw_track["Wheel_Speed_FL"]["Value"][0, 0][0, :]
    #         self.ws_fr = self.raw_track["Wheel_Speed_FR"]["Value"][0, 0][0, :]
    #         self.ws_rl = self.raw_track["Wheel_Speed_RL"]["Value"][0, 0][0, :]
    #         self.ws_rr = self.raw_track["Wheel_Speed_RR"]["Value"][0, 0][0, :]
    #     elif "VCU_WSS_FL_S" in data_keys:
    #         self.ws_fl = self.raw_track["VCU_WSS_FL_S"]["Value"][0, 0][0, :]
    #         self.ws_fr = self.raw_track["VCU_WSS_FR_S"]["Value"][0, 0][0, :]
    #         self.ws_rl = self.raw_track["VCU_WSS_RL_S"]["Value"][0, 0][0, :]
    #         self.ws_rr = self.raw_track["VCU_WSS_RR_S"]["Value"][0, 0][0, :]
    #     else:
    #         self.ws_fl = self.raw_track["Wheel_Speed_FL"]["Value"][0, 0][0, :]
    #         self.ws_fr = self.raw_track["Wheel_Speed_FR"]["Value"][0, 0][0, :]
    #         self.ws_rl = self.raw_track["Wheel_Speed_RL"]["Value"][0, 0][0, :]
    #         self.ws_rr = self.raw_track["Wheel_Speed_RR"]["Value"][0, 0][0, :]
    #     if "Steering_Angle" in data_keys:
    #         self.steer = self.raw_track["Steering_Angle"]["Value"][0, 0][0, :]
    #     elif "Steering_Wheel_Angle" in data_keys:
    #         self.steer = self.raw_track["Steering_Wheel_Angle"]["Value"][0, 0][0, :]
    #     elif "Driver_Steering" in data_keys:
    #         self.steer = self.raw_track["Driver_Steering"]["Value"][0, 0][0, :]
    #     else:
    #         self.steer = self.raw_track["Steering_Wheel_Angle"]["Value"][0, 0][0, :]
    #     mr = 1.5
    #     if "Normal_Load_FL" in data_keys:
    #         self.nl_fl = self.raw_track["Normal_Load_FL"]["Value"][0, 0][0, :]
    #         self.nl_fr = self.raw_track["Normal_Load_FR"]["Value"][0, 0][0, :]
    #         self.nl_rl = self.raw_track["Normal_Load_RL"]["Value"][0, 0][0, :]
    #         self.nl_rr = self.raw_track["Normal_Load_RR"]["Value"][0, 0][0, :]
    #         self.ls_time = self.raw_track["Normal_Load_FL"]["Time"][0, 0][0, :] - self.start_time
    #     elif "Load_Cell__Force_FL" in data_keys:
    #         self.nl_fl = self.raw_track["Load_Cell__Force_FL"]["Value"][0, 0][0, :] * LB_TO_KG * G / mr
    #         self.nl_fr = self.raw_track["Load_Cell_Force_FR"]["Value"][0, 0][0, :] * LB_TO_KG * G / mr
    #         self.nl_rl = self.raw_track["Load_Cell_Force_RL"]["Value"][0, 0][0, :] * LB_TO_KG * G * -1 / mr
    #         self.nl_rr = self.raw_track["Load_Cell_Force_RR"]["Value"][0, 0][0, :] * LB_TO_KG * G * -1 / mr
    #         self.ls_time = self.raw_track["Load_Cell__Force_FL"]["Time"][0, 0][0, :] - self.start_time
    #     else:
    #         self.nl_fl = np.zeros(self.vel.shape[0])
    #         self.nl_fr = np.zeros(self.vel.shape[0])
    #         self.nl_rl = np.zeros(self.vel.shape[0])
    #         self.nl_rr = np.zeros(self.vel.shape[0])
    #         self.ls_time = self.raw_time

    #     if "IMU_Acceleration_X" in data_keys:
    #         self.lat_acc = uniform_filter1d(self.raw_track["IMU_Acceleration_Y"]["Value"][0, 0][0, :]*9.81, 25)
    #         self.long_acc = uniform_filter1d(self.raw_track["IMU_Acceleration_X"]["Value"][0, 0][0, :]*9.81, 25)
    #         self.spa_t = self.raw_track["IMU_Acceleration_X"]["Time"][0, 0][0, :] - self.start_time
    #     elif "Aceinna_AccX" in data_keys:
    #         self.lat_acc = uniform_filter1d(self.raw_track["Aceinna_AccX"]["Value"][0, 0][0, :], 25) * -1
    #         self.long_acc = uniform_filter1d(self.raw_track["Aceinna_AccY"]["Value"][0, 0][0, :], 25)
    #         self.spa_t = self.raw_track["Aceinna_AccX"]["Time"][0, 0][0, :] - self.start_time
    #     elif "G_Force_Lat" in data_keys:
    #         self.lat_acc = uniform_filter1d(self.raw_track["G_Force_Lat"]["Value"][0, 0][0, :]*-9.81, 5)
    #         self.long_acc = uniform_filter1d(self.raw_track["G_Force_Long"]["Value"][0, 0][0, :]*9.81, 5)
    #         self.spa_t = self.raw_track["G_Force_Lat"]["Time"][0, 0][0, :] - self.start_time
    #     self.spa = clean_interp(self.spa_t, self.time, self.raw_dist)
    #     self.ls_dist = clean_interp(self.ls_time, self.u_time, self.u)
    #     self.ws_avg = (self.ws_fl + self.ws_fr + self.ws_rl + self.ws_rr) / 4

    #     if "Vehicle_Yaw_Rate" in self.data_keys:
    #         self.y_r = np.deg2rad(self.raw_track["Vehicle_Yaw_Rate"]["Value"][0, 0][0, :])
    #         self.yaw_freq = self.y_r.shape[0] / (self.raw_track["Vehicle_Yaw_Rate"]["Time"][0, 0][0, -1] - self.raw_track["Vehicle_Yaw_Rate"]["Time"][0, 0][0, 0])
    #     elif "IMU_Angular_Rate_Z" in self.raw_track.keys():
    #         self.y_r = np.deg2rad(self.raw_track["IMU_Angular_Rate_Z"]["Value"][0, 0][0, :])
    #         self.yaw_freq = self.y_r.shape[0] / (self.raw_track["IMU_Angular_Rate_Z"]["Time"][0, 0][0, -1] - self.raw_track["IMU_Angular_Rate_Z"]["Time"][0, 0][0, 0])
    #     elif "Aceinna_GyroZ" in self.raw_track.keys():
    #         self.y_r = uniform_filter1d(np.deg2rad(self.raw_track["Aceinna_GyroZ"]["Value"][0, 0][0, :]), 10)
    #         self.yaw_freq = self.y_r.shape[0] / (self.raw_track["Aceinna_GyroZ"]["Time"][0, 0][0, -1] - self.raw_track["Aceinna_GyroZ"]["Time"][0, 0][0, 0])
    #     else:
    #         self.y_r = np.zeros(self.spa_t.shape[0])
    #         self.yaw_freq = self.y_r.shape[0] / (self.spa_t[-1] - self.spa_t[0])
    #     self.y_a = np.zeros(self.y_r.shape)
    #     self.y_a[1:] = np.diff(uniform_filter1d(self.y_r, 20)) * self.yaw_freq
    #     self.real_vel = np.zeros(self.u.shape)
    #     self.real_vel[1:] = self.u[1:] / np.diff(self.u_time)
    #     self.real_vel[0] = self.u[0] / (self.u_time[1] - self.u_time[0])
    #     self.real_vel = uniform_filter1d(self.real_vel, 10)
    #     self.y_r_k = (self.k * self.real_vel)
    #     self.y_r_b = self.y_r - clean_interp(self.spa, self.u, self.y_r_k)
    #     self.real_angle_raw = np.cumsum(self.y_r) * 1/self.yaw_freq
    #     self.real_angle = self.real_angle_raw - uniform_filter1d(self.real_angle_raw - np.deg2rad(clean_interp(self.spa, self.u, self.angle)), 1000) # this should probably be a gaussian filter
    #     self.real_beta = clean_interp(self.u, self.spa, np.rad2deg(self.real_angle)) - self.angle


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



    