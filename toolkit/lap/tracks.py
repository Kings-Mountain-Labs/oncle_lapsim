from .track import Track, load_track_from_mat
from .channels import parse_car_data_mat, unit_conversion, deg2rad_chan, derivative_chan, pitch_roll_yaw_transform
from toolkit.common import MS_TO_MPH, LB_TO_KG, G
from .gps import gps_from_channels
import numpy as np

def get_MIS_2017_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Endurance.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2017_End1_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Endurance_First4Laps.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

    
def get_MIS_2017_End2_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Endurance_Second4Laps.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2017_AX_1_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Skid_Accel_AutoX_Lap1.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2017_AX_2_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Skid_Accel_AutoX_Lap2.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2017_AX_3_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Skid_Accel_AutoX_Lap3.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2017_AX_4_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Michigan_Skid_Accel_AutoX_Lap4.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"] = unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration X", "acc_x")
    data_dict["__acc_y"] = unit_conversion(data_dict["G_Force_Vert"], "m/s^2", -G, "Acceleration Y", "acc_y")
    data_dict["__acc_y"].data += G
    data_dict["__acc_z"] = unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration Z", "acc_z")
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_Crows_2023(sc) -> Track: # Lap 2 of milos 2023-10-12 where it cut out
    raw_track = load_track_from_mat('./data/lap_data/20231008-0910601-enduro1-milos.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1/MS_TO_MPH)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1/MS_TO_MPH, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1/MS_TO_MPH, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1/MS_TO_MPH, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1/MS_TO_MPH, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Normal_Load_FL"], "N", 1, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Normal_Load_FR"], "N", 1, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Normal_Load_RL"], "N", 1, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Normal_Load_RR"], "N", 1, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Wheel_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), 0.0, np.deg2rad(-45), 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 100)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track


def get_Crows_2022(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2022_Crows_Testlap.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    wheel_circ = 0.26 * 3.14
    data_dict["__ws_fl"] = unit_conversion(data_dict["VCU_WSS_FL_S"], "m/s", wheel_circ/60, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["VCU_WSS_FR_S"], "m/s", wheel_circ/60, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["VCU_WSS_RL_S"], "m/s", wheel_circ/60, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["VCU_WSS_RR_S"], "m/s", wheel_circ/60, "Wheel Speed RR", "WS_RR")
    data_dict["IMU_Angular_Rate_X"].unit, data_dict["IMU_Angular_Rate_Y"].unit, data_dict["IMU_Angular_Rate_Z"].unit = "deg/s", "deg/s", "deg/s"
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Driver_Steering"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["IMU_Acceleration_X"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["IMU_Acceleration_Y"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["IMU_Acceleration_Z"], "m/s^2", G, "Acceleration Z", "acc_z"), 0.0, 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_x"], data_dict["__gyro_y"], data_dict["__gyro_z"] = pitch_roll_yaw_transform(deg2rad_chan(data_dict["IMU_Angular_Rate_X"]), deg2rad_chan(data_dict["IMU_Angular_Rate_Y"]), deg2rad_chan(data_dict["IMU_Angular_Rate_Z"]), 0.0, 0.0, 0.0)
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 100)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2019_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2019_MIS_ENDURANCE.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), np.deg2rad(3), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Gyro_Yaw_Velocity___Hi_Res"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track


def get_MIS_2018_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2018_MIS_ENDURANCE.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), np.deg2rad(3), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Gyro_Yaw_Velocity___Hi_Res"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2021_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/SR12B_MIS2021_Autocross_3.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__steering_angle"].data += -4.24
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), np.deg2rad(3), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_Lincoln_2017_AX_track(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Lincoln_Autocross_Bobby_2.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), np.deg2rad(-5), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)

def get_Lincoln_2017_AX_track_mixed(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2017_Lincoln_Autocross_Bobby_2_Mixed.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    mr = 1.5
    data_dict["__nl_fl"] = unit_conversion(data_dict["Load_Cell__Force_FL"], "N", LB_TO_KG * G / mr, "Normal Load Force FL", "NL_FL")
    data_dict["__nl_fr"] = unit_conversion(data_dict["Load_Cell_Force_FR"], "N", LB_TO_KG * G / mr, "Normal Load Force FR", "NL_FR")
    data_dict["__nl_rl"] = unit_conversion(data_dict["Load_Cell_Force_RL"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RL", "NL_RL")
    data_dict["__nl_rr"] = unit_conversion(data_dict["Load_Cell_Force_RR"], "N", LB_TO_KG * G * -1 / mr, "Normal Load Force RR", "NL_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), np.deg2rad(-5), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_MIS_2021_track_mixed(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/2021_MIS_Endurance_2Laps.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", G, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", G, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", G, "Acceleration Z", "acc_z"), np.deg2rad(3), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track

def get_PNR_2022_11_6_Patton(sc) -> Track:
    raw_track = load_track_from_mat('./data/lap_data/20221106-PNR-Clean_Lap.mat')
    data_dict = parse_car_data_mat(raw_track)
    track_gps = gps_from_channels(data_dict["GPS_Latitude"], data_dict["GPS_Longitude"], data_dict["GPS_Altitude"], data_dict["GPS_Time"])
    data_dict["__gps_vel"] = unit_conversion(data_dict["GPS_Speed"], "m/s", 1)
    data_dict["__ws_fl"] = unit_conversion(data_dict["Wheel_Speed_FL"], "m/s", 1, "Wheel Speed FL", "WS_FL")
    data_dict["__ws_fr"] = unit_conversion(data_dict["Wheel_Speed_FR"], "m/s", 1, "Wheel Speed FR", "WS_FR")
    data_dict["__ws_rl"] = unit_conversion(data_dict["Wheel_Speed_RL"], "m/s", 1, "Wheel Speed RL", "WS_RL")
    data_dict["__ws_rr"] = unit_conversion(data_dict["Wheel_Speed_RR"], "m/s", 1, "Wheel Speed RR", "WS_RR")
    data_dict["__steering_angle"] = deg2rad_chan(data_dict["Steering_Angle"])
    data_dict["__acc_x"], data_dict["__acc_y"], data_dict["__acc_z"] = pitch_roll_yaw_transform(unit_conversion(data_dict["G_Force_Long"], "m/s^2", 1, "Acceleration X", "acc_x"), unit_conversion(data_dict["G_Force_Lat"], "m/s^2", 1, "Acceleration Y", "acc_y"), unit_conversion(data_dict["G_Force_Vert"], "m/s^2", 1, "Acceleration Z", "acc_z"), np.deg2rad(3), 0.0, 0.0)
    data_dict["__acc_z"].data += -G
    data_dict["__gyro_z"] = deg2rad_chan(data_dict["Vehicle_Yaw_Rate"])
    data_dict["__yacc"] = derivative_chan(data_dict["__gyro_z"], "Yaw Acceleration", "__yacc", "rad/s^2", 20)
    track = Track(track_gps, data_dict, sc, 0.95)
    return track