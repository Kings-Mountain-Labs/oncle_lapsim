import numpy as np
from toolkit.cars.car_configuration import Car
from .sss import Steady_State_Solver
from toolkit.common.maths import to_vel_frame, clip

class Parachute(Steady_State_Solver):
    def __init__(self):
        super().__init__()
        self.set_solver = "Parachute"

    def solve_for_long(self, car: Car, v_avg, long_g, delta_x = 0, beta_x = 0, mu_corr: float = 1.0, ay_it = 0.0, use_drag = False, long_err = 0.01, lat_err = 0.001, zeros = True, use_torque_lim=False, use_break_lim=True) -> (float, float, float, float, int, bool):
        yaw_it, cn_it, ax_it = 0.0, 0.0, 0.0
        omega = ay_it / v_avg #Initial yaw rate [rad/s]
        kappax_fl, kappax_fr, kappax_rl, kappax_rr = 0, 0, 0, 0
        bruh, total_fx = 0, 0
        drag = 0
        if use_drag:
            drag = 0.5 * 1.225 * v_avg**2 * car.cd * car.A
        safl, safr, sarl, sarr = 0, 0, 0, 0
        ay_error = 1
        ax_it = long_g
        while bruh < 40 and ay_error > lat_err: # this while loop is Figure 52 in the patton paper
            ay_targ = ay_it
            # Generate slip angles for two-track model as a function of
            # beta and delta along with included parameters for toe.
            fzfl, fzfr, fzrl, fzrr, _, _ = car.find_contact_patch_loads(long_g=ax_it, lat_g=ay_it, vel=v_avg)
            delta_fl, delta_fr, delta_rl, delta_rr = car.calculate_tire_delta_angle(delta_x, 0.0, 0.0, 0.0)
            [safl, safr, sarl, sarr] = car.calculate_slip_angles(v_avg, omega, beta_x, delta_fl, delta_fr, delta_rl, delta_rr)
            v_fl, v_fr, v_rl, v_rr = car.calculate_vel_at_tire(v_avg, omega, beta_x)
            ia_fl, ia_fr, ia_rl, ia_rr = car.calculate_ia(ay_it, fzfl, fzfr, fzrl, fzrr)
            
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
            ay_error = abs(ay_it - ay_targ)
            bruh += 1
        if ay_error > lat_err:
            if zeros:
                return 0.0, 0.0, 0.0, 0.0, bruh, True
            else:
                return ay_v, cn_it, yaw_it, ax_v, bruh, True
        
        return ay_v, cn_it, yaw_it, ax_v, bruh, False