from Functions.py_functions.car_configuration import Car
from .las import LAS
import numpy as np

def lat_loss_func(bd, las: LAS, car: Car, vel, long_g, mu_corr, use_drag, vp):
    ay, _, yaw, ax, bruh, _ = las.solver.solve_for_long(car, vel, long_g, delta_x=bd[1], beta_x=bd[0], mu_corr=mu_corr, use_drag=use_drag, zeros=True)
    cost = np.arctan2(1.0 * np.sign(ay), ay)
    cost = cost * (ay > 0) + (np.pi / 2 + -1.0 * ay) * (ay <= 0)
    if ay != 0:
        vp.append([np.rad2deg(bd[1]), np.rad2deg(bd[0]), ay, yaw, ax, vel])
    # print(f"cost: {cost:.5f}\tay: {ay:.3f}\tyaw: {yaw:.3f}\tax: {ax:.3f}\tbeta: {np.rad2deg(bd[0]):.3f}\tdelta: {np.rad2deg(bd[1]):.3f}")
    return cost

def yaw_loss_func(bd, las: LAS, car: Car, vel, long_g, mu_corr, use_drag, vp):
    ay, _, yaw, ax, bruh, _ = las.solver.solve_for_long(car, vel, long_g, delta_x=bd[1], beta_x=bd[0], mu_corr=mu_corr, use_drag=use_drag, zeros=True)
    cost = np.arctan2(1.0 * np.sign(yaw), yaw)
    cost = cost * (yaw > 0) + (np.pi / 2 + -1.0 * yaw) * (yaw <= 0)
    if yaw != 0:
        vp.append([np.rad2deg(bd[1]), np.rad2deg(bd[0]), ay, yaw, ax, vel])
    # print(f"cost: {cost:.5f}\tay: {ay:.3f}\tyaw: {yaw:.3f}\tax: {ax:.3f}\tbeta: {np.rad2deg(bd[0]):.3f}\tdelta: {np.rad2deg(bd[1]):.3f}")
    return cost
