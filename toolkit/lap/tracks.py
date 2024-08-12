from .track import *

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