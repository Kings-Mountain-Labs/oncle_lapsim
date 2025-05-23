from toolkit.cars import Car
from toolkit.lap.track import Track
import numpy as np
import time
from numba import njit
from toolkit.common.constants import *
from toolkit.las_solvers import LAS
from scipy.signal import butter, filtfilt

@njit
def find_crits(vels):
    crit_diff = np.sign(vels[:-1] - vels[1:])
    critd_it = np.argwhere((((crit_diff[:-1] + crit_diff[1:]) == 0) | (vels[1:-1] == vels[2:])) & (crit_diff[:-1] > 0)).T[0] + 1
    critc = np.unique(np.concatenate((np.array([0]), critd_it)))
    return critc[np.argsort(vels[critc])]

# If the next step rolls you past the  point, that is the same as
# going to the start of the track, so we roll back to the first
# index. This prevents the code from breaking. If k = -ind+1 and
# you're doing ind+k, it becomes ind-ind+1=1 aka the start of
# the track.
# If the next step rolls you below the starting point, that is the same as
# going to the  of the track, so we roll back to the last
# index. This prevents the code from breaking.
@njit
def check_ind(j, k, ind, len_nd, asc=False):
    if asc:
        if ind >= len_nd or ind <= -len_nd:
            ind = 0
        if k+ind >= len_nd:
            k = -ind
        if j+ind >= len_nd:
            j = -ind
        if j == k:
            k += 1
    else:
        if ind >= len_nd or ind <= -len_nd:
            ind = 0
        if ind - k <= -len_nd:
            k = ind - len_nd + 1
        if ind - j <= -len_nd:
            j = ind - len_nd + 1
        if j == k:
            k -= 1
    return j, k, ind

def calc_accel(car: Car, las: LAS, j_ind: int, k_ind: int, dd: float, new_dist, v_it, latAcc_it, longAcc_it, omegadot_it, curvature, beta_dot_old, beta_old):
    ds = new_dist[j_ind] - new_dist[k_ind]
    if abs(ds) > abs(dd) * 3: # this should only happen for a flying lap
        ds = dd
    vv = (v_it[k_ind] + v_it[j_ind]) / 2
    vb_power = np.sqrt(vv**2 + 2 * (car.find_tractive_force(vv) / car.mass) * ds)
    return las.solve_point(vv, v_it[k_ind], v_it[j_ind], ds, curvature[k_ind], curvature[j_ind], beta_dot_old[k_ind], beta_dot_old[j_ind], beta_old[j_ind], latAcc_it[j_ind], longAcc_it[j_ind], omegadot_it[j_ind], True, vbp=vb_power)

def calc_decel(car: Car, las: LAS, j_ind: int, k_ind: int, dd: float, new_dist, v_it, latAcc_it, longAcc_it, omegadot_it, curvature, beta_dot_old, beta_old):
    ds = -(new_dist[k_ind] - new_dist[j_ind])
    if abs(ds) > abs(dd) * 3:
        ds = -dd
        
    vv = (v_it[k_ind] + v_it[j_ind]) / 2
    # vb_break = np.sqrt(v_it[ind - k]**2 + 2 * (car.find_tractive_force_braking(v_it[ind - k]) / car.mass) * ds)
    return las.solve_point(vv, v_it[k_ind], v_it[j_ind], ds, curvature[k_ind], curvature[j_ind], beta_dot_old[k_ind], beta_dot_old[j_ind], beta_old[j_ind], latAcc_it[j_ind], longAcc_it[j_ind], omegadot_it[j_ind], False)

def sim_qts(car: Car, track: Track, las: LAS, target, flying=False, silent=False):
    # Line Segment of LSBC (Bottom-Right of MMD) (Patton, 74)
    lim_ind = int(las.vel_bins/2)

    # Solving for initial velocity profile [m/s] (Patton, 75)
    # Checks for each segment along LSBC
    # This does not include beta double prime
    velocity = np.zeros([len(track.k)])
    velocity_curve = np.zeros([len(track.k)])
    velocity_curve_rate = np.zeros([len(track.k)])
    vel_begin = time.time()
    for i in range(1, len(track.k)):
        velocity_curve[i], velocity_curve_rate[i] = las.find_vel_limit(las.vels[lim_ind], (track.k[i] + track.k[i - 1]) / 2, (track.k_prime[i] + track.k_prime[i - 1]) / 2, (track.u[i] - track.u[i - 1]))
    velocity[1:] = velocity_curve[1:]#np.nanmin(np.array([velocity_curve_rate[1:], velocity_curve[1:]]), axis=0) # figure 104 Take the minimum of the velocity limit and the yaw rate minimum, this helps it converge faster velocity_curve[i] velocity_curve[i] # 
    if not silent:
        print(f"vel_crit_time: {(time.time()-vel_begin):.3f}")
    
    velocity[velocity <= 0] = velocity[velocity > 0].min()
    if not flying:
        velocity[0] = track.vel[0]
    track.vc_r = velocity_curve_rate
    track.vc = velocity_curve
    long_G = np.zeros([len(track.k)])
    if flying:
        long_G[0]  = (velocity[1] - velocity[-1]) / ((2 * track.u[1]) / ((velocity[1] + velocity[-1]) / 2)) / G # Connecting the long accel  of the track back to the beginning (flying lap)
    long_G[-1] = (velocity[-2] - velocity[-1]) / ((track.u[-2] - track.u[-1]) / ((velocity[-2] + velocity[-1]) / 2)) / G # Setting the long accel at the  so that it bls to the start of the track

    # Creating long accel map using delta V/delta X
    long_G[1:-1]  = (velocity[2:] - velocity[:-2]) / ((track.u[2:] - track.u[:-2]) / ((velocity[2:] + velocity[:-2]) / 2)) / G

    new_dist = track.u
    new_velocity = velocity
    new_curvature = track.k
    critc = find_crits(velocity)
    track.u_crit = new_dist

    dd = track.u[2] - track.u[0]

    # Initialize counters and other variables for iteration
    k, j, n, v_error = 0, 1, 0, 1

    len_new_dist = len(new_dist)

    # Initialize vectors for accelerations in iteration
    # Initialize D vector in iteration (Patton, 78)
    v_it = new_velocity.copy()
    latAcc_it, longAcc_it, omegadot_it, delta_it, beta_it, D_fcheck, D_rcheck = np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist])
    beta_old = np.zeros([len_new_dist])
    beta_dot_old = np.zeros([len_new_dist])
    butter_order = 2
    # the cutoff frequency should be approximately the distance the car travels between 3 cones in a tight slalom in units of 1/m
    # My best guess is that it in the range of 5-10m 
    butter_b, butter_a = butter(butter_order, 1/15.0, 'low', analog=False, fs=1/(dd/2))

    count = np.zeros([len_new_dist])
    last_changed = np.ones([len_new_dist]) * -1

    err = []
    begin = time.time()
    last = begin
    if not silent:
        print(f"Num_points: {v_it.shape[0]}")
    while v_error > target:
        v_it_old = v_it.copy()
        checks = 0
        # Limit velocities are calculated by starting from critical points and
        # identifying the next velocity based on acceleration limits (LAS of Octahedron)
        for i, ind in enumerate(critc):

            if last_changed[ind] == n:
                continue

            # Working from the front side

            j, k, ind = check_ind(1, 0, ind, len_new_dist, True)
            floor = False
            while v_it[ind + k] <= v_it[ind + j]:
                if (((ind + j + 2 * len_new_dist) % len_new_dist) - ((ind + k + 2 * len_new_dist) % len_new_dist)) != 1 and not flying:
                    break
                if floor:
                    break
                checks += 1
                v_it[ind+j], latAcc_it[ind+j], longAcc_it[ind+j], omegadot_it[ind+j], D_fcheck[ind+j], delta_it[ind+j], beta_it[ind+j], floor = calc_accel(car, las, ind+j, ind+k, dd, new_dist, v_it, latAcc_it, longAcc_it, omegadot_it, new_curvature, beta_dot_old, beta_old)
                count[ind+j] += 1
                last_changed[ind+j] = n
                k += 1
                j += 1
                j, k, ind = check_ind(j, k, ind, len_new_dist, True)
            
            j, k, ind = check_ind(1, 0, ind, len_new_dist, False)
            # Working from the back side
            floor = False
            while v_it[ind-k] <= v_it[ind-j]:
                if (((ind - j + 2 * len_new_dist) % len_new_dist) - ((ind - k + 2 * len_new_dist) % len_new_dist)) != -1 and not flying:
                    break
                if floor:
                    break
                checks += 1
                v_it[ind-j], latAcc_it[ind-j], longAcc_it[ind-j], omegadot_it[ind-j], D_rcheck[ind-j], delta_it[ind-j], beta_it[ind-j], floor = calc_decel(car, las, ind-j, ind-k, dd, new_dist, v_it, latAcc_it, longAcc_it, omegadot_it, new_curvature, beta_dot_old, beta_old)
                count[ind-j] += 1
                last_changed[ind-j] = n
                k += 1
                j += 1
                j, k, ind = check_ind(j, k, ind, len_new_dist, False)

            # Now solve for the critical point by solving the acceleration and deceleration separately and averaging them (this point should be near the equator of the LAS)
            if last_changed[ind] +2 < n:
                v_it_acc, latAcc_it_acc, longAcc_it_acc, omegadot_it_acc, _, delta_it_acc, beta_it_acc, floor = calc_accel(car, las, ind+1, ind-1, dd, new_dist, v_it, latAcc_it, longAcc_it, omegadot_it, new_curvature, beta_dot_old, beta_old)
                v_it_dec, latAcc_it_dec, longAcc_it_dec, omegadot_it_dec, _, delta_it_dec, beta_it_dec, floor = calc_decel(car, las, ind-1, ind+1, dd, new_dist, v_it, latAcc_it, longAcc_it, omegadot_it, new_curvature, beta_dot_old, beta_old)
                # print(f"Accelerating params: {v_it_acc:.2f}\t{latAcc_it_acc:.2f}\t{longAcc_it_acc:.2f}\t{omegadot_it_acc:.2f}\t{delta_it_acc:.2f}\t{beta_it_acc:.2f}")
                # print(f"Decelerating params: {v_it_dec:.2f}\t{latAcc_it_dec:.2f}\t{longAcc_it_dec:.2f}\t{omegadot_it_dec:.2f}\t{delta_it_dec:.2f}\t{beta_it_dec:.2f}")

                if (longAcc_it_acc < 0 and longAcc_it_dec > 0) or (longAcc_it_acc > 0 and longAcc_it_dec > 0) or (longAcc_it_dec < 0 and longAcc_it_acc < 0):
                    v_it[ind] = (v_it_acc + v_it_dec) / 2
                    latAcc_it[ind] = (latAcc_it_acc + latAcc_it_dec) / 2
                    longAcc_it[ind] = (longAcc_it_acc + longAcc_it_dec) / 2
                    omegadot_it[ind] = (omegadot_it_acc + omegadot_it_dec) / 2
                    delta_it[ind] = (delta_it_acc + delta_it_dec) / 2
                    beta_it[ind] = (beta_it_acc + beta_it_dec) / 2
                    count[ind] += 1
                    last_changed[ind] = n
                
        v_error_list = np.abs(v_it_old - v_it)/ v_it
        v_error = np.nanmax(v_error_list)

        # we need to take the current beta angles, apply a 2nd order butterworth filter with a 5m cutoff, and then take the derivative of that
        beta_old = filtfilt(butter_b, butter_a, np.deg2rad(beta_it))
        beta_dot_old = np.gradient(beta_old) / dd
        
        critc = find_crits(v_it)
        dt = np.zeros(len(v_it))
        dt[1:] = 2 * (new_dist[1:] - new_dist[:-1]) / (v_it[1:] + v_it[:-1])
        total_len = np.sum(dt)
        err.append([v_error, np.nanmin(v_error_list), np.nanmean(v_error_list), np.nanstd(v_error_list), total_len, len(critc), checks, (time.time()-last)])
        if not silent:
            print(f"{n}\ttime_el:{(time.time()-begin):.2f}\ttime_last:{(time.time()-last):.4f}\tv_err: {v_error:.6f}\tcc: {len(critc)}\titt:{checks}\ttime:{total_len:.2f}")

        last = time.time()
        checks = 0
        n = n + 1
        if n > 15:
            break
    if not silent:
        print(f"total time: {(time.time() - begin):.2f}")

    v_it = np.where(v_it < 0.1, 0.1, v_it)

    dt = np.zeros(len(v_it))
    dt[1:] = 2 * (new_dist[1:] - new_dist[:-1]) / (v_it[1:] + v_it[:-1])

    return longAcc_it, latAcc_it, omegadot_it, np.nan_to_num(dt), long_G, v_it, new_velocity, dt, critc, err, delta_it, beta_it, D_fcheck, D_rcheck, count, last_changed

