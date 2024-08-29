import numpy as np
from toolkit.cars.car_configuration import Car
from toolkit.steady_state_solver import Steady_State_Solver, Iterative_Solver
from scipy.optimize import minimize
from toolkit.common.constants import G
from toolkit.common.maths import powspace
from abc import ABC, abstractmethod
import plotly.graph_objs as go

class LAS(ABC):
    solver: Steady_State_Solver
    set_las: dict[str, float] = None
    vels: np.ndarray
    aymax: np.ndarray
    yawmax: np.ndarray
    longAcc_forward: np.ndarray
    longAcc_reverse: np.ndarray
    aymax_sa: np.ndarray
    yawmax_sa: np.ndarray

    def __init__(self, solver: Steady_State_Solver = Iterative_Solver()):
        self.solver: Steady_State_Solver = solver

    @abstractmethod
    def find_vel_limit(self, vel, k, k_prime, u) -> tuple[np.float64, np.float64]:
        pass

    @abstractmethod
    def solve_point(self, v_k, v_j, ds, k_k, k_j, bd_k, bd_j, beta_j, lat, long, yacc, vbp = 1000, vbb = 1000) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, bool]:
        pass

    @abstractmethod
    def generate_las(self, car: Car, vel_bins: int=None, mu: float = 1.0):
        pass

    @abstractmethod
    def plot_las(self, fig: go.Figure):
        pass

    @abstractmethod
    def get_xyzijk(self, vv:float = 15.0):
        pass

    def find_limit(self, car: Car, v_avg, long_g, func, beta_lim = 7, delta_lim = 15, use_drag=True, mu = 1.0, b_guess = 0.0, d_guess = 0.0):
        """
        We find the corners of the MMD (peak lateral acceleration and peak yaw rate) using a bisection method
        this has lots of corner cases and is not very robust, but it is way faster than the actual gradient descent solvers i have tried
        this needs to be fixed, badly
        """
        itt: int = 0
        delta_bins, beta_bins = [], []
        valid_points = []

        res = minimize(func, [np.deg2rad(b_guess), np.deg2rad(d_guess)], args=(self, car, v_avg, long_g, mu, use_drag, valid_points,), bounds=((-np.deg2rad(beta_lim), np.deg2rad(beta_lim)), (-np.deg2rad(delta_lim), np.deg2rad(delta_lim))), options=dict(disp=False), method="Nelder-Mead")
        beta, delta = res.x
        itt += res.nit
        ay, _, yaw, ax, bruh, _ = self.solver.solve_for_long(car, v_avg, long_g, beta_x=beta, delta_x=delta, mu_corr=mu, use_drag=use_drag, zeros=True)
        valid_points.append([np.rad2deg(delta), np.rad2deg(beta), ay, yaw, ax, v_avg])
        return np.rad2deg(delta), np.rad2deg(beta), ay, yaw, ax, itt, valid_points, delta_bins, beta_bins
    
    def find_long_limits(self, car: Car, v_avg, lim, mu_corr: float = 1.0, use_drag=True, use_torque_lim=False):
        """
        We fine either the positive or negative long acceleration limit using a bisecting search
        we dont fully bisect because the solve_for_long function is not particularly reliable and sometimes fails to converge

        """
        lower, upper, itt = min(lim, 0), max(lim, 0), 0
        tol = 0.001
        while abs(lower - upper) > tol or bad:
            itt += 1
            long_acc = (lower + upper) / 2
            _, _, _, ax, _, bad = self.solver.solve_for_long(car, v_avg, long_acc, mu_corr=mu_corr, use_drag=use_drag, long_err=tol * 0.1, zeros=False, use_torque_lim=use_torque_lim)
            diff = (ax - long_acc)
            # print(f"long acc: {long_acc:.4f}, ax: {ax:.4f}, diff: {diff:.4f}, lower: {lower:.4f}, upper: {upper:.4f}, bad: {bad}, itt: {itt}")
            if abs(diff) < tol and not bad:
                if lim > 0:
                    lower = long_acc
                else:
                    upper = long_acc
            elif lim < 0:
                lower = long_acc
            else:
                upper = long_acc
            
        return ax

    def long_g_spread(self, car: Car, v_avg, layers: int = 11, use_torque_lim=False):
        max_acc_init, min_acc_init = 3 * G, -3 * G
        max_acc, min_acc = self.find_long_limits(car, v_avg, max_acc_init, use_torque_lim=use_torque_lim), self.find_long_limits(car, v_avg, min_acc_init, use_torque_lim=use_torque_lim)
        layers += (layers + 1) % 2 #makes it odd
        layers_in_each_direction = int((layers - 1) / 2) + 1
        long_g = np.unique(np.concatenate((np.linspace(min_acc, 0, layers_in_each_direction, endpoint=True), np.linspace(0, max_acc, layers_in_each_direction, endpoint=True))))
        return long_g
    
    def MMD_3D_Long(self, car: Car, mu_corr: float = 1.0, geom=False, max_sa=25):
        """
        Returns a set of isobeta and isodelta lines at a range of longitudinal accelerations
        """
        # Initialize beta and delta vectors to create isobeta, isodelta, and iso long_acc lines
        b_len, d_len, g_len = 10, 10, 20
        b_lens, d_lens = b_len * 2 + 1, d_len * 2 + 1
        b_ext, d_ext = 25, 15

        if geom:
            # beta = np.geomspace(0.01, b_ext, b_len)
            # beta = np.deg2rad(np.concatenate((-beta[::-1], np.array([0]), beta)))
            # delta = np.geomspace(0.01, d_ext, d_len)
            # delta = np.deg2rad(np.concatenate((-delta[::-1], np.array([0]), delta)))
            beta = powspace(0.0, b_ext, 2, b_len)
            beta = np.deg2rad(np.concatenate((-beta[::-1], np.array([0]), beta)))
            delta = powspace(0.0, d_ext, 2, d_len)
            delta = np.deg2rad(np.concatenate((-delta[::-1], np.array([0]), delta)))
        else:
            beta   = np.deg2rad(np.linspace(-b_ext, b_ext, b_lens, endpoint=True))
            delta  = np.deg2rad(np.linspace(-d_ext, d_ext, d_lens, endpoint=True))
        
        long_g_s = self.long_g_spread(car, 15, layers=g_len)
        long_g = np.zeros([len(self.vels), len(long_g_s)])
        l_len  = [b_lens, d_lens, len(long_g_s), len(self.vels)]
        ay_it, ax_it, yaw_it, cn_it, valid  = np.zeros(l_len), np.zeros(l_len), np.zeros(l_len), np.zeros(l_len), np.zeros(l_len)
        ud = False
        LAS_points = []
        for v, vel in enumerate(self.vels):
            print(v)
            long_g[v] = self.long_g_spread(car, vel, layers=g_len, use_torque_lim=ud)
            for k, long_gx in enumerate(long_g[v]):
                for j, deltax in enumerate(delta):
                    for i, betax in enumerate(beta):
                        # if the mirrored point has already been calculated and did not converge, then we can skip this point and mark it as invalid
                        if valid[-i-1, -j-1, k, v] == -1 or (abs(betax - deltax) > max_sa):
                            valid[i, j, k, v] = -1
                            continue
                        if valid[-i-1, -j-1, k, v] == 1: # and if the mirrored point has already been calculated and did converge, then we can skip this point and mark it as valid and copy the values
                            yaw_it[i, j, k, v] = yaw_it[-i-1, -j-1, k, v] * -1
                            cn_it[i, j, k, v]  = cn_it[-i-1, -j-1, k, v] * -1
                            ay_it[i, j, k, v]  = ay_it[-i-1, -j-1, k, v] * -1
                            ax_it[i, j, k, v]  = ax_it[-i-1, -j-1, k, v]
                            valid[i, j, k, v] = 1
                            LAS_points.append([np.rad2deg(deltax), np.rad2deg(betax), ay_it[-i-1, -j-1, k, v] * -1, yaw_it[-i-1, -j-1, k, v] * -1, ax_it[-i-1, -j-1, k, v], vel])
                            continue
                        ay_it[i, j, k, v], cn_it[i, j, k, v], yaw_it[i, j, k, v], ax_it[i, j, k, v], _, bad_solution = self.solver.solve_for_long(car, vel, long_gx, deltax, betax, use_drag=ud, mu_corr=mu_corr, use_torque_lim=True)
                        if bad_solution: # if the solution did not converge, then we mark it as invalid
                            valid[i, j, k, v] = -1
                        else:
                            valid[i, j, k, v] = 1
                            LAS_points.append([np.rad2deg(deltax), np.rad2deg(betax), ay_it[i, j, k, v], yaw_it[i, j, k, v], ax_it[i, j, k, v], vel])
        return ay_it / G, yaw_it, cn_it, ax_it / G, delta, beta, long_g / G, valid, LAS_points
