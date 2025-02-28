import numpy as np
from toolkit.cars.car_configuration import Car
from .sss import Steady_State_Solver
from toolkit.common.maths import to_vel_frame, clip, to_car_frame

class Iterative_Solver(Steady_State_Solver):
    def __init__(self):
        super().__init__()
        self.set_solver = "Iterative"

    def solve_for_long(self, car: Car, v_avg, long_g, delta_x = 0, beta_x = 0, mu_corr: float = 1.0, ay_it = 0.0, use_drag = False, long_err = 0.005, lat_err = 0.001, zeros = True, use_torque_lim=False, use_break_lim=True):
        yaw_it, cn_it, ax_it = 0.0, 0.0, 0.0
        ay_v = 0.0
        omega = ay_it / v_avg #Initial yaw rate [rad/s]
        kappax_fl, kappax_fr, kappax_rl, kappax_rr = 0, 0, 0, 0
        bruh, long_error, total_fx = 0, 1, 0
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
        safl, safr, sarl, sarr = 0, 0, 0, 0
        ay_error = 1
        lfx = car.mass * long_g + drag
        ax_it = long_g
        # p = 0.8
        p = np.interp(v_avg, [5, 10, 20], [0.8, 0.5, 0.8])
        while bruh < 40 and (long_error > long_err or ay_error > lat_err): # this while loop is Figure 52 in the patton paper
            ay_targ = ay_it
            # Generate slip angles for two-track model as a function of
            # beta and delta along with included parameters for toe.
            ax_c, ay_c = to_car_frame(long_g, ay_v, beta_x)
            fzfl, fzfr, fzrl, fzrr, _, _ = car.find_contact_patch_loads(long_g=ax_c, lat_g=ay_c, vel=v_avg)
            delta_fl, delta_fr, delta_rl, delta_rr = car.calculate_tire_delta_angle(delta_x, 0.0, 0.0, 0.0)
            [safl, safr, sarl, sarr] = car.calculate_slip_angles(v_avg, omega, beta_x, delta_fl, delta_fr, delta_rl, delta_rr)
            v_fl, v_fr, v_rl, v_rr = car.calculate_vel_at_tire(v_avg, omega, beta_x)
            ia_fl, ia_fr, ia_rl, ia_rr = car.calculate_ia(ay_it, fzfl, fzfr, fzrl, fzrr)
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
            nay_it = total_fy / car.mass
            if bruh == 0: ay_it = nay_it
            ay_it = nay_it * (1 - p) + ay_it * p # lat accel [g's]

            total_fx = fxfl + fxfr + fxrr + fxrl - drag
            ax_it = total_fx / car.mass # long accel [g's]
            ax_v, ay_v = to_vel_frame(ax_it, ay_it, beta_x) # equation 51 and 52 in the patton paper
            long_error_old = long_error
            long_error = abs(long_g - ax_v)
            omega = ay_v / v_avg
            ay_error = abs(ay_it - ay_targ)
            bruh += 1
            lfx += car.mass * (long_g - ax_v) / 2
            if bam_r and abs(long_error - long_error_old) < 0.001 and ay_error < 0.001:
                break
            # print(f"too high\t{bruh}\tB:{np.rad2deg(beta_x):.2f}\tD:{np.rad2deg(delta_x):.2f}\t{omega:.3f}\tv:{v_avg:.3f}\tay:{ay_it:.3f}\tay_curr:{(total_fy / car.mass):.3f}\tyaw:{yaw_it:.4f}\tax:{ax_it:.3f}\tax_v:{ax_v:.3f}\tlfx{lfx:.2f}\tkappas:{kappax_rl:.4f}\t{kappax_rr:.4f}\tfz:{fzfl:.2f}\t{fzfr:.2f}\t{fzrl:.2f}\t{fzrr:.2f}\t{bam_r}\tfx:{fxfl:.2f}\t{fxfr:.2f}\t{fxrl:.2f}\t{fxrr:.2f}")
            # if bruh > 20 and abs(long_error) < 5: # long_error > long_err and bam_r and bam_f and 
            # print(f"too high\t{bruh}\tB:{np.rad2deg(beta_x):.2f}\tD:{np.rad2deg(delta_x):.2f}\t{omega:.3f}\tv:{v_avg:.3f}\tay:{ay_it:.3f}\tay_curr:{(total_fy / car.mass):.3f}\tax:{ax_it:.3f}\tax_v:{ax_v:.3f}\tlfx:{lfx:.2f}\tkappas:{kappax_fl:.4f}\t{kappax_fr:.4f}\t{kappax_rl:.4f}\t{kappax_rr:.4f}\tfz:{fzfl:.2f}\t{fzfr:.2f}\t{fzrl:.2f}\t{fzrr:.2f}\t{bam_r}\tfx:{fxfl:.2f}\t{fxfr:.2f}\t{fxrl:.2f}\t{fxrr:.2f}")
        if (long_error > long_err or ay_error > lat_err):
            if zeros:
                return 0.0, 0.0, 0.0, 0.0, bruh, True
            else:
                return ay_v, cn_it, yaw_it, ax_v, bruh, True
        
        return ay_v, cn_it, yaw_it, ax_v, bruh, False