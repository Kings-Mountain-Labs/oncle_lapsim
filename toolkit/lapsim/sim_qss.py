from toolkit.cars import Car
from toolkit.lap.track import Track
import numpy as np
import time
from numba import njit
from toolkit.common.constants import *
from toolkit.las_solvers import LAS

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

def sim_qss(car: Car, track: Track, las: LAS, target, flying=False, silent=False, true_beta=None):
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
    # return
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
    k, j, n, v_error = 0, 1, 1, 1

    len_new_dist = len(new_dist)

    # Initialize vectors for accelerations in iteration
    # Initialize D vector in iteration (Patton, 78)
    v_it = new_velocity.copy()
    latAcc_it, longAcc_it, omegadot_it, delta_it, beta_it, D_fcheck, D_rcheck = np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist]), np.zeros([len_new_dist])
    if true_beta is None:
        true_beta = np.zeros([len_new_dist])
    beta_old = np.array(np.deg2rad(true_beta))
    beta_dot_old = np.gradient(beta_old) / dd
    count = np.zeros([len_new_dist])
    last_changed = np.zeros([len_new_dist])
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
            # Working from the front side
            if last_changed[ind] == n:
                continue

            j, k, ind = check_ind(1, 0, ind, len_new_dist, True)
            floor = False
            while v_it[ind + k] <= v_it[ind + j]:
                if (((ind + j + 2 * len_new_dist) % len_new_dist) - ((ind + k + 2 * len_new_dist) % len_new_dist)) != 1 and not flying:
                    break
                if floor:
                    break
                checks += 1
                ds = new_dist[ind + j] - new_dist[ind + k]
                if abs(ds) > abs(dd):
                    ds = dd
                vv = (v_it[ind + k] + v_it[ind + j]) / 2
                vb_power = np.sqrt(vv**2 + 2 * (car.find_tractive_force(vv) / car.mass) * ds)
                v_it[ind+j], latAcc_it[ind+j], longAcc_it[ind+j], omegadot_it[ind+j], D_fcheck[ind+j], delta_it[ind+j], beta_it[ind+j], floor = las.solve_point(vv, v_it[ind+k], v_it[ind+j], ds, new_curvature[ind + k], new_curvature[ind + j], beta_dot_old[ind + k], beta_dot_old[ind + j], beta_old[ind + j], latAcc_it[ind+j], longAcc_it[ind+j], omegadot_it[ind+j], True, vbp=vb_power)

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

                ds = -(new_dist[ind - k] - new_dist[ind - j])
                if abs(ds) > abs(dd):
                    ds = -dd
                    
                vv = (v_it[ind - k] + v_it[ind - j]) / 2
                # vb_break = np.sqrt(v_it[ind - k]**2 + 2 * (car.find_tractive_force_braking(v_it[ind - k]) / car.mass) * ds)
                v_it[ind-j], latAcc_it[ind-j], longAcc_it[ind-j], omegadot_it[ind-j], D_rcheck[ind-j], delta_it[ind-j], beta_it[ind-j], floor = las.solve_point(vv, v_it[ind-k], v_it[ind-j], ds, new_curvature[ind - k], new_curvature[ind - j], beta_dot_old[ind - k], beta_dot_old[ind - j], beta_old[ind - j], latAcc_it[ind-j], longAcc_it[ind-j], omegadot_it[ind-j], False) # , vbb=vb_break

                count[ind-j] += 1
                last_changed[ind-j] = n
                k += 1
                j += 1
                j, k, ind = check_ind(j, k, ind, len_new_dist, False)
                
        v_error_list = np.abs(v_it_old - v_it)/ v_it
        v_error = np.nanmax(v_error_list)
        
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
    if not silent:
        print(f"total time: {(time.time() - begin):.2f}")

    v_it = np.where(v_it < 0.1, 0.1, v_it)

    dt = np.zeros(len(v_it))
    dt[1:] = 2 * (new_dist[1:] - new_dist[:-1]) / (v_it[1:] + v_it[:-1])

    return longAcc_it, latAcc_it, omegadot_it, np.nan_to_num(dt), long_G, v_it, new_velocity, dt, critc, err, delta_it, beta_it, D_fcheck, D_rcheck, count, last_changed