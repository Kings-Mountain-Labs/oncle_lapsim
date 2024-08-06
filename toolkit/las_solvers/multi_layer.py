from .las import LAS
import numpy as np
from toolkit.cars.car_configuration import Car
from numba import njit
from toolkit.common.constants import *
import time
from typing import List
from .loss_funcs import lat_loss_func, yaw_loss_func
import plotly.graph_objs as go
from toolkit.common.maths import skew, is_point_in_triangle, db_for_point_in_triangle

@njit
def calc_vel(c_0, c_1, c_2, v_min = 0.1):
    # Solve the quadratic equation for the velocity because the numpy roots function is slow af
    v_1 = np.real((np.sqrt(c_1**2 - (4 * c_2 * c_0)) - c_1) / (2 * c_2))
    v_2 = np.real(((np.sqrt(c_1**2 - (4 * c_2 * c_0)) * -1) - c_1) / (2 * c_2))
    v_0 = np.maximum(v_1, v_2)
    if v_0 > v_min:
        return v_0, False
    return v_min, True

# @njit
def solve_point(aymax, yawmax, longAcc, facets, v_k, v_j, ds, k_k, k_j, bd_k, bd_j, beta_j, lat, long, yacc, vbp = 1000, vbb = 1000):
    corner_list = np.array([longAcc])
    #append the ay and yaw limits to the corner list
    corner_list = np.append(corner_list, aymax, axis=0)
    corner_list = np.append(corner_list, aymax, axis=0)
    corner_list = np.append(corner_list, yawmax, axis=0)
    corner_list[len(aymax) + 1, :] *= np.array([1, -1, -1, -1, -1])

    # Calculate the velocity limit based on the longitudinal acceleration limits
    max_vel = longAcc[0] * ds + v_k
    # If we are limited by max power output use that as the velocity limit
    if vbp < max_vel:
        max_vel = vbp
    if vbb < max_vel:
        max_vel = vbb
    # Check for different LAS for solution to roots
    signs = [np.array([1, 1, 1, 1, 1]), np.array([1, -1, -1, -1, -1])]
    min_v_it = np.inf
    for (a, b, c) in facets:
        for s in signs:
            point_a, point_b, point_c = corner_list[a] * s, corner_list[b] * s, corner_list[c] * s
            n_vector = np.dot(skew(point_a[:3] - (point_b[:3])), (point_c[:3]) - (point_b[:3])) # Normal vector to LAS
            c_0 = n_vector[0] * (point_a[0] + (v_k**2 / (2 * ds))) + (n_vector[1] * (point_a[1] - ((v_k**2 * (k_j + k_k)) / 8))) + n_vector[2] * (point_a[2] + (v_k**2 * (k_k - bd_k) / (2 * ds)))
            c_1 = v_k * (-n_vector[1] * ((k_k + k_j) / 4) + n_vector[2] * (((k_k - bd_k) - (k_j - bd_j)) / (2 * ds)))
            c_2 = -n_vector[0] / (2 * ds) - n_vector[1] * (k_k + k_j) / 8 - n_vector[2] * (k_j - bd_j) / (2 * ds)
            
            if c_2 == 0:
                # print("c_2 is zero")
                # print(c_0, c_1, c_2)
                # print(point_a, point_b, point_c)
                # print(a, b, c, len(corner_list), len(aymax), len(yawmax), v_k)
                continue
            v_save, nps = calc_vel(c_0, c_1, c_2)
            if nps or v_save > max_vel:
                v_save = max_vel
            v_it = min(v_save, v_j)
                
            longAcc_it = (v_it**2 - v_k**2) / (2 * ds) # equation 116
            latAcc_it = ((k_k + k_j) / 2) * (((v_k + v_it) / 2)**2) # equation 117
            omegadot_it = ((v_k + v_it) / (2 * ds)) * ((k_j * v_it) - (k_k * v_k)) # equation 118
            A_check = np.array([longAcc_it, latAcc_it, omegadot_it])
            min_v_it = min(min_v_it, v_it)
            # Check if the solution is inside the LAS
            if is_point_in_triangle(A_check*NVEC, point_c[:3]*NVEC, point_b[:3]*NVEC, point_a[:3]*NVEC) < 10e-10:
                delta_it, beta_it = db_for_point_in_triangle(A_check*NVEC, point_c*NVEC2, point_b*NVEC2, point_a*NVEC2)
                return v_it, latAcc_it, longAcc_it, omegadot_it, 0.0, delta_it, beta_it, False
    return min(max_vel, v_j, min_v_it), 0.0, 0.0, 0.0, 0.0, 0.0, beta_j, False

@njit
def interp_LAS_corner(vel, vels, point_arr):
    """
    Interpolate the LAS corner points to get the correct point for the current velocity
    """
    x = np.interp(vel, vels, point_arr[:, 0])
    y = np.interp(vel, vels, point_arr[:, 1])
    z = np.interp(vel, vels, point_arr[:, 2])
    d = np.interp(vel, vels, point_arr[:, 3])
    b = np.interp(vel, vels, point_arr[:, 4])
    return np.array([x, y, z, d, b])

@njit
def interp_LAS_surface(vel, vels, point_arr):
    """
    Interpolate the LAS corner points to get the correct point for the current velocity
    """
    # interpolate the 2d array of corners
    if vel < vels[0]:
        return point_arr[0]
    elif vel > vels[-1]:
        return point_arr[-1]
    else:
        # find the index of the velocity that is below the current velocity
        ind = np.where(vels <= vel)[0][-1]
        point_sp = ((vel - vels[ind]) / (vels[ind + 1] - vels[ind]))
        return point_arr[ind] * (1 - point_sp) + point_arr[ind + 1] * point_sp

# absolutely do not jit this function, it makes the other functions run slower for no reason I can understand
def find_vel_limit(vel, vels, aymax, yawmax, k, k_prime, u, itt=0):
    if np.isnan(vel):
        return vel, 0.0
    a_0 = interp_LAS_corner(vel, vels, aymax)
    a_1 = interp_LAS_corner(vel, vels, yawmax)
    if k == 0:
        return np.inf, 0 # if there is no curvature then return inf velocity limit
    chkCount = 0
    while True:
        if chkCount == 1:
            a_0 *= -1
        elif chkCount == 2:
            a_1 *= -1
        elif chkCount == 3:
            a_0 *= -1
        elif chkCount == 4:
            vel_limit = np.nan
        num = a_0[1] - ((a_1[1] * (a_0[0] * k - a_0[2])) / ((a_1[0] * k) - a_1[2]))
        denom = k + (a_1[1] * (k_prime) / ((a_1[0] * k) - a_1[2]))
        if (num / denom) < 0:
            chkCount += 1
            continue
        vel_limit = np.sqrt(num / denom)
        break
    if np.isnan(vel_limit):
        return np.inf, np.sqrt(abs((a_1[2] * u) / (k * 2)))
    if abs(vel_limit - vel) < 0.1 or itt > 10:
        return vel_limit, a_1[2]
    else:
        return find_vel_limit(vel_limit, vels, aymax, yawmax, k, k_prime, u, itt=itt+1)

def limit_surface(axmax_forward, axmax_brake, aymax, yawmax):
        """
        Defines the points and vertex indexes that make up the Quad-cross section LAS
        """
        # Make a list of points that define the LAS
        x: List = []
        y: List = []
        z: List = []
        twins = [1, -1]
        x.extend([0, 0])
        y.extend([0, 0])
        z.extend([axmax_forward[0], axmax_brake[0]])
        layers = aymax.shape[0]
        for l in range(layers):
            for a in twins:
                x.extend([yawmax[l, 1] * a, aymax[l, 1] * a])
                y.extend([yawmax[l, 2] * a, aymax[l, 2] * a])
                z.extend([yawmax[l, 0], aymax[l, 0]])
        # Make a list of vertex indices that define the LAS
        i: List = []
        j: List = []
        k: List = []
        # Generate the top and bottom pyramids
        for a in twins:
            for b in range(4):
                g = 2
                if a == 1:
                    j.append(0)
                    g = len(x) - 4
                else:
                    j.append(1)
                i.append(g + b)
                k.append(g + ((b + 1) % 4))
        # Generate the middle polyhedra
        for l in range(layers - 1):
            for b in range(4):
                for a in twins:
                    if a == 1:
                        i.append(2 + (l * 4) + ((b + 1) % 4))
                    else:
                        i.append(2 + (l * 4) + b + 4)
                    j.append(2 + (l * 4) + b)
                    k.append(2 + (l * 4) + ((b + 1) % 4) + 4)
        return x, y, z, i, j, k

class Multi_Layer_LAS(LAS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.long_acc_layers = None
        self.add_layers = None
        self.vel_bins = None

    def find_vel_limit(self, vel, k, k_prime, u) -> (np.float64, np.float64):
        return find_vel_limit(vel, self.vels, self.aymax, self.yawmax, k, k_prime, u)

    def solve_point(self, vv, v_k, v_j, ds, k_k, k_j, bd_k, bd_j, beta_j, lat, long, yacc, forward, vbp = 1000, vbb = 1000) -> (np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, bool):
        if forward:
            return solve_point(interp_LAS_surface(vv, self.vels, self.aymax_l[:, self.add_layers:]), interp_LAS_surface(vv, self.vels, self.yawmax_l[:, self.add_layers:]), interp_LAS_corner(vv, self.vels, self.longAcc_forward), self.facets, v_k, v_j, ds, k_k, k_j, bd_k, bd_j, beta_j, lat, long, yacc, vbp, vbb)
        else:
            return solve_point(interp_LAS_surface(vv, self.vels, self.aymax_l[:, :self.add_layers+1]), interp_LAS_surface(vv, self.vels, self.yawmax_l[:, :self.add_layers+1]), interp_LAS_corner(vv, self.vels, self.longAcc_reverse), self.facets, v_k, v_j, ds, k_k, k_j, bd_k, bd_j, beta_j, lat, long, yacc, vbp, vbb)


    def generate_las(self, car: Car, vel_bins=None, mu=1.0, use_drag=True, quiet=True, add_layers=None):
        self.set_las = (mu, vel_bins)
        ## Initialize some stuff
        car.max_velocity = np.power((car.power / (0.5 * 1.225 * car.cd * car.A)), 1/3)
        # print(f"Max velocity based on drag: {self.max_velocity} m/s")
        # Produce values for all accelerations based off of MMD at average velocity
        # MMD is based off of vehicle parameter inputs
        # LongForward, LongBrake, and Lat Accel [g's], Yaw Accel [rad/s^2]
        if vel_bins is not None: # if the user has explicitly set the number of velocity bins then we will use that
            self.vel_bins = vel_bins
        elif self.vel_bins is None: # if the user has not set the number ahead of time then we will use the default of 10
            self.vel_bins = 10
        
        if add_layers is not None:
            self.add_layers = add_layers
        elif self.add_layers is None:
            self.add_layers = 1
        self.layers = self.add_layers * 2 + 1
        self.vels = np.linspace(8, car.max_velocity * 0.8, self.vel_bins, endpoint=True) # first velocity is in m/s
        # print(self.vels)
        vel_begin = time.time()
        self.aymax, self.yawmax, self.longAcc_forward, self.longAcc_reverse, self.aymax_sa, self.yawmax_sa = np.zeros([self.vels.shape[0], 5]), np.zeros([self.vels.shape[0], 5]), np.zeros([self.vels.shape[0], 5]), np.zeros([self.vels.shape[0], 5]), np.zeros([self.vels.shape[0], 2]), np.zeros([self.vels.shape[0], 2])
        self.aymax_l, self.yawmax_l, self.aymax_sa_l, self.yawmax_sa_l = np.zeros([self.vels.shape[0], self.layers, 5]), np.zeros([self.vels.shape[0], self.layers, 5]), np.zeros([self.vels.shape[0], self.layers, 2]), np.zeros([self.vels.shape[0], self.layers, 2])
        self.long_acc_layers = np.zeros([self.vels.shape[0], self.layers])

        self.vps = []
        for ind, vel in enumerate(self.vels):
            self.MMD_lapsim(vel, car, ind, mu_corr=mu, use_drag=use_drag)

        self.gen_facets()
        if not quiet:
            print(f"vel_sweep_time: {(time.time()-vel_begin):.3f}")

    def gen_facets(self):
        n = self.add_layers + 1
        self.facets = []
        self.facets.append([0, 1, 2*n+1])
        self.facets.append([0, n+1, 2*n+1])
        for i in range(n-1):
            self.facets.append([i+1, i+2, 2*n+i+1])
            self.facets.append([i+2, 2*n+i+1, 2*n+i+2])
            self.facets.append([n+i+1, n+i+2, 2*n+i+1])
            self.facets.append([n+i+2, 2*n+i+1, 2*n+i+2])

    def MMD_lapsim(self, v_avg, car: Car, ind: int, mu_corr: float = 1.0, use_drag=True):
        max_acc_init, min_acc_init = 3 * G, -3 * G
        axmax_brake, axmax_forward = self.find_long_limits(car, v_avg, min_acc_init, mu_corr=mu_corr, use_drag=use_drag), self.find_long_limits(car, v_avg, max_acc_init, mu_corr=mu_corr, use_drag=use_drag)
        
        self.longAcc_forward[ind] = [axmax_forward, 0, 0, 0, 0] # long, lat, yaw, delta, beta
        self.longAcc_reverse[ind] = [axmax_brake, 0, 0, 0, 0] # long, lat, yaw, delta, beta
        # Find out what the layer heights should be
        for i in range(self.add_layers):
            self.long_acc_layers[ind, self.add_layers - i - 1] = (axmax_brake / (self.add_layers + 1)) * (i + 1)

        for i in range(self.add_layers):
            self.long_acc_layers[ind, i + self.add_layers + 1] = (axmax_forward / (self.add_layers + 1)) * (i + 1)
        # beta_ay, delta_ay, beta_yaw, delta_yaw = 0, 0, 0, 0
        for i, long_g in enumerate(self.long_acc_layers[ind]):
            beta_ay, delta_ay, beta_yaw, delta_yaw = 0, -5, 0, 0
            delta_ay, beta_ay, aymax_ay, yawmax_ay, ax_ay, itt_ay, vp_ay, _, _ = self.find_limit(car, v_avg, long_g, lat_loss_func, delta_lim=30.0, beta_lim=25.0, use_drag=use_drag, mu=mu_corr, b_guess=beta_ay, d_guess=delta_ay)
            delta_yaw, beta_yaw, aymax_yaw, yawmax_yaw, ax_yaw, itt_yaw, vp_yaw, _, _ = self.find_limit(car, v_avg, long_g, yaw_loss_func, delta_lim=30.0, beta_lim=25.0, use_drag=use_drag, mu=mu_corr, b_guess=beta_yaw, d_guess=delta_yaw)

        
            if car.debug:
                # fig = go.Figure(go.Scatter(x=delta_b, y=beta_b, fill="toself"))
                # fig.show()
                # fig = go.Figure(go.Scatter(x=delta_b, y=beta_b, fill="toself"))
                # fig.show()
                pass
            if aymax_ay == 0 or yawmax_yaw == 0:
                print(f"Unable to solve for a LAS with car: {car.description} at velocity: {v_avg:.2f} m/s")
            if aymax_ay == 0:
                print(f"AY\tVel:{v_avg:.2f}\tItt:{itt_ay}")
            if yawmax_yaw == 0:
                print(f"YAW\tVel:{v_avg:.2f}\tItt:{itt_yaw}")
            self.vps.extend(vp_ay)
            self.vps.extend(vp_yaw)
            aymax  = np.array([0, aymax_ay, yawmax_ay, delta_ay, beta_ay])
            yawmax = np.array([0, aymax_yaw, yawmax_yaw, delta_yaw, beta_yaw])
            if aymax_ay < 0:
                aymax = aymax * -1
            if yawmax_yaw < 0:
                yawmax = yawmax * -1

            aymax[0] = ax_ay
            yawmax[0] = ax_yaw
            
            # Initialize Acceleration Vectors
            self.aymax_l[ind, i] = aymax # long, lat, yaw, delta, beta
            self.aymax_sa_l[ind, i] = np.array([aymax[3], aymax[4]])
            self.yawmax_l[ind, i] = yawmax # long, lat, yaw, delta, beta
            self.yawmax_sa_l[ind, i] = np.array([yawmax[3], yawmax[4]])
        self.aymax[ind] = self.aymax_l[ind, self.add_layers] # long, lat, yaw, delta, beta
        self.aymax_sa[ind] = self.aymax_sa_l[ind, self.add_layers]
        self.yawmax[ind] = self.yawmax_l[ind, self.add_layers] # long, lat, yaw, delta, beta
        self.yawmax_sa[ind] = self.yawmax_sa_l[ind, self.add_layers]

    def plot_las(self, fig, vv:float = 15.0):
        x, y, z, i, j, k = limit_surface(interp_LAS_corner(vv, self.vels, self.longAcc_forward), interp_LAS_corner(vv, self.vels, self.longAcc_reverse), interp_LAS_surface(vv, self.vels, self.aymax_l), interp_LAS_surface(vv, self.vels, self.yawmax_l))
        # print(f"{x}\t{y}\t{z}\t{i}\t{j}\t{k}")
        fig.add_trace(go.Mesh3d(
                # 8 vertices of a cube
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color='lightpink',
                opacity=0.50,
                showscale=True
            ))
            
