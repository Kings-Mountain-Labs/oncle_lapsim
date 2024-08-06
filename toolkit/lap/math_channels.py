import pandas as pd
import numpy as np
from ldparser import ldData
from itertools import groupby
from loading_util import make_path



# static contact patch loads
m_fr = 72.6
m_fl = 63.0
m_rr = 54.9
m_rl = 88.45

shockpot_offset_fr = 13.75
shockpot_offset_fl = 14.3
shockpot_offset_rr = 12.8
shockpot_offset_rl = 11.75

i_zz = 0.0 
wb = 60.25/39.37 #wheelbase [m]
mass = m_fl+m_fl+m_rl+m_rr # mass of vehicle [kg] 620 lbs for SR-13 and SRE-6
front_axle_weight = (m_fl+m_fl)/mass # weight distribution toward front axle (#) 0.5 for SRE-6
a = mass/(1-front_axle_weight) # distance from CG to front axle [m]
b = mass/front_axle_weight # distance from CG to rear axle [m]
cg_height = 11.7/39.37 # CG height above ground [m]
mass_unsprung = 85/2.2 # 85 lbs
static_ride = 4 # static ride height [in]
spring_rate = 650 * 0.175126835 # springrate [N/mm]

spring_preload_fr = m_fr*9.81/spring_rate*0.698
spring_preload_fl = m_fl*9.81/spring_rate*0.698
spring_preload_rr = m_rr*9.81/spring_rate*0.706
spring_preload_rl = m_rl*9.81/spring_rate*0.706

k_c = 1400 * 1.3558 #Chassis stiffness in N-m/deg
k_f = 389 * 1.3558 #Front roll stiffness
k_r = 421 * 1.3558 #Rear roll stiffness
d_s = 11.4/39.37 #CG height minus roll axis height at CG
z_f = .126/39.37 #Front roll center height
z_r = 1.1/39.37 #Rear roll center height
hu_f = 8/39.37 #Unsprung mass CG height
hu_r = hu_f
ms_f = (mass-mass_unsprung)*front_axle_weight #Sprung mass on front axle
ms_r = (mass-mass_unsprung)-ms_f #Sprung mass on rear axle
mu_f = mass_unsprung/2 #Front unsprung mass
mu_r = mu_f

ds_f = cg_height-z_f
ds_r = cg_height-z_r

front_track = 48/39.37 # front track width [m]
rear_track = 47/39.37 # rear track width [m]

# belcrank angle equation params
rh_offset = 3.75 # [in]

ang_front_a = 0.144
ang_front_b = 1.07
ang_front_c = 44
ang_rear_a = 0.507
ang_rear_b = -1.41
ang_rear_c = 24.8

front_motion_ratio = 0.622
rear_motion_ratio = 0.67

def get_data(path):
    # just to check, read back the file

    l = ldData.fromfile(path)

    dprime = None

    for f, g in groupby(l.channs, lambda x:x.freq):
        df = pd.DataFrame({i.name.lower(): i.data for i in g})
        df['ms_step'] = df.index * 1/f
        df['Time'] = pd.to_timedelta(df['ms_step'], 's')
        df.set_index('Time')
        df = df.drop(columns = ['ms_step'])
        if dprime is not None:
            dprime = pd.merge_asof(dprime, df, on='Time', tolerance=pd.Timedelta(milliseconds=1))
        else:
            dprime = df

    # print(dprime.interpolate())
    # print(dprime)
    return dprime

def add_corrected_acc(dprime):
    dprime["Corr_Lat"] = dprime['g_force_vert']-1.08
    dprime["Corr_Lon"] = dprime['g_force_lat']+0.02
    dprime["Corr_Vert"] = dprime['g_force_long']
    dprime["Corr_Speed"] = (dprime['wheel_speed_rr'] + dprime['wheel_speed_rl'] + dprime['wheel_speed_fr'] + dprime['wheel_speed_fl'])/4
    return dprime

def add_corrected_shockpots(dprime):
    dprime["Corr_Damper_Pos_FR"] = (dprime['damper_pos_fr']-shockpot_offset_fr)
    dprime["Corr_Damper_Pos_FL"] = (dprime['damper_pos_fl']-shockpot_offset_fr)
    dprime["Corr_Damper_Pos_RR"] = (dprime['damper_pos_rr']-shockpot_offset_fr)
    dprime["Corr_Damper_Pos_RL"] = (dprime['damper_pos_rl']-shockpot_offset_fr)
    dprime["Corr_Damper_Rate_FR"] = np.gradient(dprime["Corr_Damper_Pos_FR"])
    dprime["Corr_Damper_Rate_FL"] = np.gradient(dprime["Corr_Damper_Pos_FL"])
    dprime["Corr_Damper_Rate_RR"] = np.gradient(dprime["Corr_Damper_Pos_RR"])
    dprime["Corr_Damper_Rate_RL"] = np.gradient(dprime["Corr_Damper_Pos_RL"])
    dprime["Corr_Ride_Height_FR"] = dprime["Corr_Damper_Pos_FR"] / (0.689 - 0.0148 * dprime["Corr_Damper_Pos_FR"])
    dprime["Corr_Ride_Height_FL"] = dprime["Corr_Damper_Pos_FL"] / (0.689 - 0.0148 * dprime["Corr_Damper_Pos_FL"])
    dprime["Corr_Ride_Height_RR"] = dprime["Corr_Damper_Pos_RR"] / 0.706
    dprime["Corr_Ride_Height_RL"] = dprime["Corr_Damper_Pos_RL"] / 0.706
    return dprime

def add_suspension_forces(dprime):
    dprime["Spring_Force_FR"] = (spring_preload_fr+dprime["Corr_Damper_Pos_FR"])*spring_rate
    dprime["Spring_Force_FL"] = (spring_preload_fr+dprime["Corr_Damper_Pos_FL"])*spring_rate
    dprime["Spring_Force_RR"] = (spring_preload_fr+dprime["Corr_Damper_Pos_RR"])*spring_rate
    dprime["Spring_Force_RL"] = (spring_preload_fr+dprime["Corr_Damper_Pos_RL"])*spring_rate
    
    return dprime

def add_contact_patch_load(dprime):
    dprime["FR_Loadcell_Angle"] = ang_front_c+ang_front_b*(dprime['Corr_Damper_Pos_FR']/25.4/front_motion_ratio+static_ride-rh_offset)+ang_front_a*((dprime['Corr_Damper_Pos_FR']/25.4/front_motion_ratio+static_ride-rh_offset)**2)
    dprime["FL_Loadcell_Angle"] = ang_front_c+ang_front_b*(dprime['Corr_Damper_Pos_FL']/25.4/front_motion_ratio+static_ride-rh_offset)+ang_front_a*((dprime['Corr_Damper_Pos_FL']/25.4/front_motion_ratio+static_ride-rh_offset)**2)
    dprime["RR_Loadcell_Angle"] = ang_rear_c+ang_rear_b*(dprime['Corr_Damper_Pos_RR']/25.4/rear_motion_ratio+static_ride-rh_offset)+ang_rear_a*((dprime['Corr_Damper_Pos_RR']/25.4/rear_motion_ratio+static_ride-rh_offset)**2)
    dprime["RL_Loadcell_Angle"] = ang_rear_c+ang_rear_b*(dprime['Corr_Damper_Pos_RL']/25.4/rear_motion_ratio+static_ride-rh_offset)+ang_rear_a*((dprime['Corr_Damper_Pos_RL']/25.4/rear_motion_ratio+static_ride-rh_offset)**2)
    dprime["FR_Contact_Patch"] = dprime['load_cell_force_fr']*np.cos(np.deg2rad(dprime["FR_Loadcell_Angle"]))
    dprime["FL_Contact_Patch"] = dprime['load_cell__force_fl']*np.cos(np.deg2rad(dprime["FL_Loadcell_Angle"]))
    dprime["RR_Contact_Patch"] = dprime['load_cell_force_rr'] * -1*np.cos(np.deg2rad(dprime["RR_Loadcell_Angle"]))
    dprime["RL_Contact_Patch"] = dprime['load_cell_force_rl'] * -1*np.cos(np.deg2rad(dprime["RL_Loadcell_Angle"]))
    dprime["Total_Normal"] = dprime['FL_Contact_Patch'] + dprime['FR_Contact_Patch'] + dprime['RR_Contact_Patch'] + dprime['RL_Contact_Patch']
    dprime["FTLT_Real"] = (dprime['FR_Contact_Patch'] - dprime['FL_Contact_Patch']-(m_fr-m_fl))*9.81
    dprime["RTLT_Real"] = (dprime['RR_Contact_Patch'] - dprime['RL_Contact_Patch']-(m_rr-m_rl))*9.81
    return dprime


# ((lambda(j)^2-(mu_c(i)+1)*lambda(j))./(lambda(j)^2-lambda(j)-mu_c(i)))*(ds_f/cg_height)*(ms_f/mass)-((mu_c(i).*lambda(j))/(lambda(j)^2-lambda(j)-mu_c(i)))*(ds_r/cg_height)*(ms_r/mass)+(z_f/cg_height)*(ms_f/mass)+(z_f*.1/cg_height)*(mu_f/mass)
def add_lltd_chans(dprime):
    dprime["FGLT"] = dprime['Corr_Lat'] * ms_f * z_f / front_track * 9.81 # Geometric load transfer
    dprime["RGLT"] = dprime['Corr_Lat'] * ms_r * z_r / rear_track * 9.81
    dprime["Roll_Angle"] = dprime['Corr_Lat'] * 9.81 * ((ms_r * ds_r + ms_f * ds_f)/(k_f+k_r))
    dprime["FELT"] = dprime['Corr_Lat'] * 9.81 * (ms_r * ds_r + ms_f * ds_f) / front_track * k_f/(k_f+k_r) # elastic load transfer
    dprime["RELT"] = dprime['Corr_Lat'] * 9.81 * (ms_r * ds_r + ms_f * ds_f) / rear_track * k_r/(k_f+k_r)

    fllts = ms_f * z_f / front_track + (ms_r * ds_r + ms_f * ds_f) / front_track * k_f/(k_f+k_r)
    rllts = ms_r * z_r / rear_track + (ms_r * ds_r + ms_f * ds_f) / rear_track * k_r/(k_f+k_r)
    print(f"Simple LLTD method: Front LLTD: {fllts/(fllts+rllts)}")
    rsd = k_f/(k_f+k_r)
    mu = k_c/(k_f+k_r)
    matlab_lltd = ((rsd*rsd-(mu+1)*rsd)/(rsd*rsd-rsd-mu)*ds_f*ms_f/cg_height/mass) - ((mu*rsd)/(rsd*rsd-rsd-mu)*ds_r*ms_r/cg_height/mass)+(z_f*ms_f/cg_height/mass)+(z_f*0.1*mu_f/cg_height/mass)
    mu = 69000000
    matlab_lltd_rigid = ((rsd*rsd-(mu+1)*rsd)/(rsd*rsd-rsd-mu)*ds_f*ms_f/cg_height/mass) - ((mu*rsd)/(rsd*rsd-rsd-mu)*ds_r*ms_r/cg_height/mass)+(z_f*ms_f/cg_height/mass)+(z_f*0.1*mu_f/cg_height/mass)
    print(f"LLTD from Arash's matlab script: {matlab_lltd} Rigid: {matlab_lltd_rigid}")

    dprime["FTLT"] = dprime['FGLT'] + dprime['FELT']  # total load transfer
    dprime["RTLT"] = dprime['RGLT'] + dprime['RELT']
    dprime["FLLTS_Real"] = dprime['FTLT_Real'] / dprime['Corr_Lat']  # total load transfer
    dprime["RLLTS_Real"] = dprime['RTLT_Real'] / dprime['Corr_Lat']
    dprime["FLLTD_Real"] = dprime['FLLTS_Real'] / (dprime['FLLTS_Real'] + dprime['RLLTS_Real'])  # total load transfer
    dprime["RLLTD_Real"] = dprime['RLLTS_Real'] / (dprime['FLLTS_Real'] + dprime['RLLTS_Real'])
    dprime["LLTD_Diff"] = abs(dprime['FTLT_Real']) - abs(dprime['RTLT_Real'])
    
    return dprime
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dp = get_data(make_path('./Data/logs/2017_Michigan_Endurance.ld'))
    dp = add_corrected_acc(dp)
    dp = add_corrected_shockpots(dp)
    dp = add_suspension_forces(dp)
    dp = add_contact_patch_load(dp)
    dp = add_lltd_chans(dp)
    # dg = dp.loc[dp["Corr_Speed"] > 10]
    dg = dp.loc[(dp["FLLTD_Real"]<1)&(dp["FLLTD_Real"]>0)&(dp["vehicle_yaw_rate"].abs()>5)&(dp["Corr_Speed"]>15&(dp["Corr_Yaw_Accel"].abs()<0.3))]
    # plt.hist(dg["FLLTD_Real"], 30)
    # plt.scatter(dg['gps latitude'], dg['gps longitude'])
    plt.plot(dp["Spring_Force_FR"])
    plt.plot(dp["Spring_Force_FL"])
    plt.plot(dp["Spring_Force_RR"])
    plt.plot(dp["Spring_Force_RL"])
    # plt.scatter(dg["Corr_Lat"], dg["vehicle yaw rate"])
    #pd.plotting.lag_plot(dg[["Corr_Lat", "vehicle yaw rate"]], lag=1)
    plt.show()
    # print(f"{dp.keys()}")
    # lines = dp.plot.line(x='Time', y=['Corr_Lat', 'FGLT', 'RGLT', 'FELT', 'RELT', 'FTLT', 'RTLT'])
    
