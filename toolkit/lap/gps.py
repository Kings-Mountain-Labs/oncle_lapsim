import numpy as np
from .channels import Channel
import pymap3d as pm
from csaps import csaps
from toolkit.common.maths import clean_interp

class GPS:
    start_time: float  # this is the gps time at which the time series datum is
    freq: int
    time: np.ndarray
    raw_time: np.ndarray
    gps_time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    x_track: np.ndarray
    y_track: np.ndarray
    z_track: np.ndarray
    lat_origin: float
    lon_origin: float
    altitude_origin: float
    delta_dist: np.ndarray
    dist: np.ndarray

    def generate_enu(self):
        self.lat_origin, self.lon_origin, self.altitude_origin = (
            np.mean(self.lat),
            np.mean(self.lon),
            np.mean(self.alt),
        )
        [self.x_track, self.y_track, self.z_track] = pm.geodetic2enu(
            self.lat,
            self.lon,
            self.alt,
            self.lat_origin,
            self.lon_origin,
            self.altitude_origin,
        )

    def generate_dist(self):
        self.delta_dist = np.zeros(self.lat.shape)
        self.delta_dist[1:] = np.sqrt((self.x_track[:-1] - self.x_track[1:])**2 + (self.y_track[:-1] - self.y_track[1:])**2)
        self.dist = np.cumsum(self.delta_dist)

    def generate_gps(self):
        [self.lat, self.lon, self.alt] = pm.enu2geodetic(
            self.x_track,
            self.y_track,
            self.z_track,
            self.lat_origin,
            self.lon_origin,
            self.altitude_origin,
        )

    def get_2d_track(self, enu=False):
        if enu:
            return np.array([self.x_track, self.y_track]).T
        else:
            return np.array([self.lat, self.lon]).T


def gps_from_channels(lat: Channel, lon: Channel, alt: Channel = None, time: Channel = None, gps_offset: float = 0.05) -> GPS:
    gps = GPS()
    gps.lat = lat.data
    gps.lon = lon.data
    gps.time = lat.time - lat.time[0]
    gps.raw_time = lat.time - gps_offset # this is the phase shift of the gps logging
    gps.freq = lat.freq
    if alt is None:
        gps.alt = np.zeros(gps.lat.shape)
    else:
        gps.alt = alt.data

    if time is not None:
        gps.gps_time = time.data
        # then lets calculate the start time in gps time
        gps.start_time = time.data[0]
    else:
        gps.gps_time = np.zeros(gps.lat.shape)

    gps.generate_enu()
    gps.generate_dist()
    return gps

def smooth_gps(gps: GPS, sc: int, spl_sm: float = 0.85):
    new_gps = GPS()
    new_gps.freq = gps.freq
    new_gps.start_time = gps.start_time
    new_gps.lat_origin = gps.lat_origin
    new_gps.lon_origin = gps.lon_origin
    new_gps.altitude_origin = gps.altitude_origin
    jump_ind = np.where(gps.delta_dist > 0.0001)[0]
    space = np.arange(jump_ind.shape[0])
    val = np.linspace(0, jump_ind.shape[0], jump_ind.shape[0] * int(sc / gps.freq))
    out = csaps(space, np.array([gps.x_track[jump_ind], gps.y_track[jump_ind], gps.z_track[jump_ind]]), val, smooth=spl_sm)
    new_gps.x_track = out[0, :]
    new_gps.y_track = out[1, :]
    new_gps.z_track = out[2, :]
    new_gps.generate_gps()
    new_gps.generate_dist()
    new_gps.time = clean_interp(new_gps.dist, gps.dist, gps.time)
    new_gps.raw_time = clean_interp(new_gps.dist, gps.dist, gps.raw_time)
    new_gps.gps_time = clean_interp(new_gps.dist, gps.dist, gps.gps_time)
    return new_gps