import numpy as np
from toolkit.cars.car_configuration import Car
from .sss import Steady_State_Solver
from toolkit.common.maths import to_vel_frame, clip, to_car_frame
from scipy.optimize import least_squares, minimize


@np.vectorize
def car_state_func(ay_targ, lfx, car: Car, v_avg, long_g, delta_x, beta_x, mu_corr, drag, max_f, max_r, max_tractive_force):
    # get ax initial by converting long_g to ax
    ax_i, ay_i = to_car_frame(long_g, ay_targ, beta_x)
    omega = ay_targ / v_avg #Initial yaw rate [rad/s]
    fzfl, fzfr, fzrl, fzrr, _, _ = car.find_contact_patch_loads(long_g=ax_i, lat_g=ay_i, vel=v_avg)
    delta_fl, delta_fr, delta_rl, delta_rr = car.calculate_tire_delta_angle(delta_x, 0.0, 0.0, 0.0)
    [safl, safr, sarl, sarr] = car.calculate_slip_angles(v_avg, omega, beta_x, delta_fl, delta_fr, delta_rl, delta_rr)
    v_fl, v_fr, v_rl, v_rr = car.calculate_vel_at_tire(v_avg, omega, beta_x)
    ia_fl, ia_fr, ia_rl, ia_rr = car.calculate_ia(ay_i, fzfl, fzfr, fzrl, fzrr)
    if long_g < 0:
        fx_r = max(lfx * (1 - car.effective_brake_bias), max_r * 2)
        fx_f = max(lfx * car.effective_brake_bias, max_f * 2)
    else:
        fx_r = min(lfx, max_tractive_force)
        fx_f = 0
        kappax_fl, kappax_fr = 0.0, 0.0

    if long_g < 0:
        kappax_fl, kappax_fr, bam_f = car.s_r_ind(fzfl, clip(safl), ia_fl, v_fl, fzfr, clip(safr), ia_fr, v_fr, fx_f, non_driven=True, rear=False, mu_corr=mu_corr)
    kappax_fl, kappax_fr = min(kappax_fl, 0.0), min(kappax_fr, 0.0)
    kappax_rl, kappax_rr, bam_r = car.s_r_ind(fzrl, clip(sarl), ia_rl, v_rl, fzrr, clip(sarr), ia_rr, v_rr, fx_r, mu_corr=mu_corr)
    
    fyfl, fxfl, mzfl = car.steady_state_mmd(fzfl, clip(safl), kappax_fl, v_fl, ia_fl, delta_fl, flip_s_a=True, mu=mu_corr)
    fyfr, fxfr, mzfr = car.steady_state_mmd(fzfr, clip(safr), kappax_fr, v_fr, ia_fr, delta_fr, mu=mu_corr)
    fyrl, fxrl, mzrl = car.steady_state_mmd(fzrl, clip(sarl), kappax_rl, v_rl, ia_rl, delta_rl, flip_s_a=True, mu=mu_corr)
    fyrr, fxrr, mzrr = car.steady_state_mmd(fzrr, clip(sarr), kappax_rr, v_rr, ia_rr, delta_rr, mu=mu_corr)
    # Normalized yaw moments created by the front axle and rear
    # axle about the CG
            
    # Looking from the top down, the positive yaw moment is clock wise
    CN_fl = fyfl * car.a + car.front_track * fxfl / 2 + mzfl
    CN_fr = fyfr * car.a - car.front_track * fxfr / 2 + mzfr
    CN_rl = -fyrl * car.b + car.rear_track * fxrl / 2 + mzrl # (-) to indicate opposing moment to front axle
    CN_rr = -fyrr * car.b - car.rear_track * fxrr / 2 + mzrr # (-) to indicate opposing moment to front axle
    CN_total = CN_fl + CN_fr + CN_rl + CN_rr
    yaw_it = CN_total / car.izz # yaw accel [rad/s^2]
    total_fy = fyfl + fyfr + fyrr + fyrl
    cn_it = car.izz * yaw_it / (car.mass * car.wb)
    ay_it = total_fy / car.mass

    total_fx = fxfl + fxfr + fxrr + fxrl - drag
    ax_it = total_fx / car.mass # long accel [g's]
    ax_v, ay_v = to_vel_frame(ax_it, ay_it, beta_x) # equation 51 and 52 in the patton paper
    omega = ay_v / v_avg
    long_error = abs(long_g - ax_v)
    ay_error = abs(ay_v - ay_targ)
    return ay_v, cn_it, yaw_it, ax_v, long_error, ay_error


def backup_loss_func(x, car: Car, ax_targ, v_avg, delta_x, beta_x, mu_corr, drag, max_f, max_r, max_tractive_force):
    ay_targ, lfx = x
    ay_v, cn_it, yaw_it, ax_v, long_error, ay_error = car_state_func(ay_targ, lfx, car, v_avg, ax_targ, delta_x, beta_x, mu_corr, drag, max_f, max_r, max_tractive_force)
    return [ay_error, long_error]

class LS_Solver(Steady_State_Solver):
    def __init__(self):
        super().__init__()
        self.set_solver = "Least_Squares"

    def solve_for_long(self, car: Car, v_avg, long_g, delta_x = 0, beta_x = 0, mu_corr: float = 1.0, ay_it = None, use_drag = False, long_err = 0.01, lat_err = 0.01, zeros = True, use_torque_lim=False, use_break_lim=True):
        yaw_it, cn_it = 0.0, 0.0
        drag = 0
        if use_break_lim:
            max_f, max_r = car.max_front_brake_torque / -car.mf_tire.UNLOADED_RADIUS, car.max_rear_brake_torque / -car.mf_tire.UNLOADED_RADIUS
        else:
            max_f, max_r = -1e10, -1e10

        if use_torque_lim: # at the moment i dont think there should be a torque limit for acceleration when creating the LAS
            max_tractive_force = car.find_tractive_force(vel=v_avg, use_aero=use_drag)
        else:
            max_tractive_force = 1e10
        if use_drag:
            drag = 0.5 * 1.225 * v_avg**2 * car.cd * car.A

        lfx = car.mass * long_g + drag
        lfx = min(max(lfx, 2*(max_f + max_r) - drag), max_tractive_force)
        if ay_it is None or True:
            # do a search to find the best start point
            ay_z = np.linspace(-20, 20, 100)
            _, _, _, _, _, ay_error = car_state_func(ay_z, lfx, car, v_avg, long_g, delta_x, beta_x, mu_corr, drag, max_f, max_r, max_tractive_force)
            ay_it = ay_z[np.argmin(ay_error)]
        ay_init = ay_it
        args = (car, long_g, v_avg, delta_x, beta_x, mu_corr, drag, max_f, max_r, max_tractive_force)
        res = least_squares(backup_loss_func, [ay_it, lfx], args=args, bounds=((-30, 2*(max_f + max_r) - drag),(30, max_tractive_force)), method="trf", max_nfev=30, ftol=1e-5, loss="cauchy", verbose=0)
        ay_it, lfx = res.x
        bruh = res.nfev
        ay_v, cn_it, yaw_it, ax_v, long_error, ay_error = car_state_func(ay_it, lfx, car, v_avg, long_g, delta_x, beta_x, mu_corr, drag, max_f, max_r, max_tractive_force)

        # if the initial guess is out of bounds we should print a warning
        # print(f"Good Val {bruh}\nay_it: {ay_it:.6f}\tay_it: {ay_v:.6f}\tay_error: {ay_error:.6f}\tbeta_x: {np.rad2deg(beta_x):.2f}\tdelta_x: {np.rad2deg(delta_x):.2f}\tlong_g: {long_g:.6f}\tlong_error: {long_error:.6f}")
        # if ay_error >= lat_err:
        #     print(f"Warning: initial guess for ay_it is out of bounds for constraint {bruh}\nay_it: {ay_it:.6f}\tay_init: {ay_init:.6f}\tay_it: {ay_v:.6f}\tay_error: {ay_error:.6f}\tbeta_x: {np.rad2deg(beta_x):.2f}\tdelta_x: {np.rad2deg(delta_x):.2f}\tlong_g: {long_g:.6f}\tlong_error: {long_error:.6f}")
        # if long_error >= lat_err:
        #     print(f"Ax error is big")

        if (long_error > long_err or ay_error > lat_err): # ay_error > lat_err:#
            if zeros:
                return 0.0, 0.0, 0.0, 0.0, bruh, True
            else:
                return ay_v, cn_it, yaw_it, ax_v, bruh, True
        
        return ay_v, cn_it, yaw_it, ax_v, bruh, False