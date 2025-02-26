import numpy as np
from toolkit.tire_model.tire_model_utils import H_R20_18X6_7
from toolkit.tire_model.tire_model_pacejka_2010 import tire_model_from_arr
import plotly.graph_objs as go
import time
from toolkit.common.constants import *
from toolkit.tire_model.fast_pacejka import get_rs_pacejka
from scipy.optimize import minimize, OptimizeResult
from toolkit.common.maths import vel_at_tire, clip, to_vel_frame, to_car_frame

def loss_func_two(bd, car, ay_targ, vel, mu_corr, sr_lim):
    ay, yaw, ax, bruh = car.solve_for_yaw(ay_targ, vel, bd[0], bd[1], mu_corr, sr_lim=sr_lim)
    return (ay - ay_targ)**2 + (yaw/5)**2 + ax**2

def variable_sr(v_a, v_b, sr):
    reference_vel = (v_a + v_b) / 2
    ref_slip_speed = reference_vel * (sr + 1)
    return (ref_slip_speed / v_a) - 1, (ref_slip_speed / v_b) - 1

def sr_variable_lim(v_a, v_b, sr, upper, lower):
    reference_vel = (v_a + v_b) / 2
    
    # Derive bounds for sr based on sr_a
    lower_bound_a = (lower + 1) * v_a / reference_vel - 1
    upper_bound_a = (upper + 1) * v_a / reference_vel - 1

    # Derive bounds for sr based on sr_b
    lower_bound_b = (lower + 1) * v_b / reference_vel - 1
    upper_bound_b = (upper + 1) * v_b / reference_vel - 1

    # Determine the overlapping region between the two bounds
    final_lower_bound = max(lower_bound_a, lower_bound_b)
    final_upper_bound = min(upper_bound_a, upper_bound_b)

    # If current sr is within the bounds, return it. Otherwise, return a bound value.
    if final_lower_bound <= sr <= final_upper_bound:
        return sr
    else:
        # Return the bound that's closest to the original sr
        if abs(final_lower_bound - sr) < abs(final_upper_bound - sr):
            return final_lower_bound
        else:
            return final_upper_bound


class Car:
    def __init__(self, mass = 663 * LB_TO_KG, front_axle_weight = 0.49) -> None:
        # this needs to be redone so it is a sensible and easy to use constructor
        self.description = "This is for labeling in sweeps"
        self.debug = False
        self.mass = mass # mass of vehicle
        self.wb = 60.25 * IN_TO_M
        self.front_axle_weight = front_axle_weight # weight distribution toward front axle
        self.mass_unsprung = 74 * LB_TO_KG # mass of outboard
        self.front_track = 48 * IN_TO_M
        self.rear_track = 47 * IN_TO_M
        self.cg_height = 11.7 * IN_TO_M
        self.A = 1.0
        self.cd = 1.59 # frontal drag coefficient
        self.cl = 3.3 # frontal lift coefficient
        self.front_axle_downforce = .4 # CoP distribution toward front axle
        self.izz = 78.5 # moment of inertia of vehicle
        # Front and Rear toe angles are per side, negative is toe out confirmed
        self.toe_front = -0.5
        self.toe_rear = -0.5
        # Camber is in adapted ISO so a positive number is negative camber
        self.i_a_f = -1.25
        self.i_a_r = -1.25
        self.k_c = 1456 * FTLB_TO_NM #Chassis stiffness
        self.k_f = 325 * FTLB_TO_NM #Front roll stiffness
        self.k_r = 275 * FTLB_TO_NM #Rear roll stiffness
        self.z_f = .126 * IN_TO_M #Front roll center height
        self.z_r = 1.1 * IN_TO_M #Rear roll center height
        self.hu_f = 8 * IN_TO_M #Unsprung mass CG height
        self.hu_r = self.hu_f

        self.power = 80000
        self.max_torque = 165
        self.drive_ratio = 4.1 # cock and balls

        self.pedal_force = 75 * LB_TO_KG * G # force applied to pedal in N
        self.pedal_ratio = 4.5 # pedal ratio
        self.bias_bar = 0.5 # bias bar ratio
        self.front_master_cylinder_dia = 19.1
        self.rear_master_cylinder_dia = 23.8
        self.front_piston_dia = 31.75
        self.rear_piston_dia = 31.75
        self.front_rotor_er = 5.84 * 0.5 * IN_TO_M # m effective radius
        self.rear_rotor_er = 5.84 * 0.5 * IN_TO_M # m
        self.front_pad_mu = 1.0 # friction coefficient
        self.rear_pad_mu = 1.0 # friction coefficient
        self.number_of_brake_pistons = 2

        self.brake_bias = 2.5 # 3.3.8.1. Driving and Braking Constraints in the Patton paper, 1 is even, 2 is 2x in the front

        self.diff_model_front = "open" # open, locked, viscous, torsen, clutch
        self.diff_model_rear = "open" # open, locked, viscous, torsen, clutch

        self.set_tire(H_R20_18X6_7)

        ## Initialize some stuff
        self.vel_bins = None

        self.update_car()
        

    def set_tire(self, tire):
        self.mf_tire = tire_model_from_arr(tire)
        try:
            self.fast_mf = get_rs_pacejka(self.mf_tire)
        except:
            print("Failed to load fast pacejka, using slow pacejka, you should really compile the fast pacejka its like 3x faster")
            self.fast_mf = None

    def update_car(self):
        self.mu_f = self.mass_unsprung/2 #Front unsprung mass
        self.mu_r = self.mass_unsprung/2
        self.mass_sprung = self.mass - self.mass_unsprung
        self.ms_f = self.mass_sprung * self.front_axle_weight #Sprung mass on front axle
        self.ms_r = self.mass_sprung * (1 - self.front_axle_weight) #Sprung mass on rear axle
        self.a = self.wb * (1-self.front_axle_weight) # distance from CG to front axle [m]
        self.b = self.wb * (self.front_axle_weight) # distance from CG to rear axle [m]
        self.LLTD, self.df_f, self.df_r = self.calculate_lltd_chassis(self.ms_f, self.ms_r)
        front_pressure, rear_pressure = self.calculate_brake_pressure(self.pedal_force)
        self.max_front_brake_torque, self.max_rear_brake_torque = self.calculate_brake_torque(front_pressure, rear_pressure)
        self.effective_brake_bias = self.max_front_brake_torque / (self.max_front_brake_torque + self.max_rear_brake_torque)
        self.max_velocity = np.power((self.power / (0.5 * 1.225 * self.cd)), 1/3)

    def find_tractive_force(self, vel, use_aero = True):
        if use_aero:
            drag = 0.5 * 1.225 * vel**2 * self.cd # drag
        else:
            drag = 0.0
        omega = vel / self.mf_tire.UNLOADED_RADIUS * self.drive_ratio
        torque = min((self.power/((omega**3) / (self.power / self.max_torque)**2)), self.max_torque)
        return (torque / self.mf_tire.UNLOADED_RADIUS * self.drive_ratio) - drag

    # All the braking functions are based on the this google sheet that Ross made
    # https://docs.google.com/spreadsheets/d/1m3ZPXid02hijrBa735rZvVp145GBNQBnq6ns3Xg1l4A/edit#gid=0
    def calculate_brake_pressure(self, pedal_force):
        """
        Calculates the pressure in the front and rear master cylinders in N/mm^2 = 10 bar
        """
        # Front master cylinder pressure
        front_master_cylinder_pressure = pedal_force * self.pedal_ratio * self.bias_bar / ((self.front_master_cylinder_dia / 2)**2 * PI)
        # Rear master cylinder pressure
        rear_master_cylinder_pressure = pedal_force * self.pedal_ratio * (1 - self.bias_bar) / ((self.rear_master_cylinder_dia / 2)**2 * PI)
        return front_master_cylinder_pressure, rear_master_cylinder_pressure

    def calculate_brake_torque(self, front_pressure, rear_pressure):
        """
        Calculates the torque in the front and rear brake calipers in N x m
        takes in the pressure in the front and rear master cylinders in N/mm^2 = 10 bar
        """
        # Front brake torque
        front_brake_torque = front_pressure * (self.front_piston_dia / 2)**2 * PI * self.front_rotor_er * self.number_of_brake_pistons * self.front_pad_mu
        # Rear brake torque
        rear_brake_torque = rear_pressure * (self.rear_piston_dia / 2)**2 * PI * self.rear_rotor_er * self.number_of_brake_pistons * self.rear_pad_mu
        return front_brake_torque, rear_brake_torque

    def find_tractive_force_braking(self, vel, use_aero = True):
        """
        Calculates the maximum tractive force in N
        """
        max_front_brake_torque, max_rear_brake_torque = self.calculate_brake_pressure(self.pedal_force)
        front_brake_torque, rear_brake_torque = self.calculate_brake_torque(max_front_brake_torque, max_rear_brake_torque)
        if use_aero:
            drag = 0.5 * 1.225 * vel**2 * self.cd # drag
        else:
            drag = 0.0
        torque = front_brake_torque * 2 + rear_brake_torque * 2
        return (torque / self.mf_tire.UNLOADED_RADIUS * -1) - drag

    def find_contact_patch_loads(self, long_g = 0, lat_g = 0, vel = 0, used_aero = True):
        """
        Calculates the contact patch loads in N
        Takes in the longitudinal and lateral acceleration in G and the velocity in m/s
        To include downforce, set used_aero to True
        """
        if used_aero:
            front_df = 0.5 * 1.225 * vel**2 * self.cl * self.front_axle_downforce # front wing downforce [N]
            rear_df  = 0.5 * 1.225 * vel**2 * self.cl * (1 - self.front_axle_downforce) # rear wing downforce [N]
        else:
            front_df = 0.0
            rear_df  = 0.0
        fz_f = (self.ms_f + self.mu_f) * G + front_df
        fz_r = (self.ms_r + self.mu_r) * G + rear_df
        wt_pitch = np.clip(self.mass_sprung * self.cg_height * long_g / (self.a + self.b), -fz_r, fz_f)
        # body_roll = (self.mass_sprung * G * self.cg_height * np.sin(self.calculate_body_roll(lat_g))) # the calculated body roll is cg migration due to roll
        fz_f = fz_f - wt_pitch
        fz_r = fz_r + wt_pitch
        # https://kktse.github.io/jekyll/update/2021/05/12/simplied-lateral-load-transfer-analysis.html
        # https://www.waveydynamics.com/post/weight-transfer-rc
        # I derived the equations for the weight transfer from the above two links
        # You have to solve the spring equation to get these and its kinda wack, slack me if you want to know how - Ian
        # The equation for the amount of force exerted by spring 1 when it is in parallel with spring 2 as a function of the total force is:
        # F1 = Ft * (k1 / (k1 + k2))
        # The equation for the equivalent spring constant of springs 1 and 2 in series is:
        # Keq = (k1 * k2) / (k1 + k2)
        # So for the weight transferred through the front suspension would be equal to two springs in parallel
        # where one spring rate is the front spring rate and the other is the equivalent spring rate of the chassis and rear rate
        # so the equation for the weight transferred through the front suspension would be:
        # F1 = Ft * (Kf / (Kf + (Kr * Kc) / (Kr + Kc)))
        # and we can treat it the same even though its a torque
        # the actual equations for these constants are located in the calculate_lltd_chassis function

        df_f, df_r = self.df_f, self.df_r

        wt_roll_f = np.clip(df_f * lat_g, -fz_f / 2, fz_f / 2)
        wt_roll_r = np.clip(df_r * lat_g, -fz_r / 2, fz_r / 2)
        
        fzfl = fz_f / 2 - wt_roll_f
        fzfr = fz_f / 2 + wt_roll_f
        fzrl = fz_r / 2 - wt_roll_r
        fzrr = fz_r / 2 + wt_roll_r

        return fzfl, fzfr, fzrl, fzrr, wt_pitch, (wt_roll_f + wt_roll_r) / 2

    def calculate_ia(self, ay, fzfl, fzfr, fzrl, fzrr):
        """
        Calculates the inclination angle in radians
        """
        body_roll = self.calculate_body_roll(ay)
        return np.deg2rad(self.i_a_f) - body_roll, np.deg2rad(self.i_a_f) + body_roll, np.deg2rad(self.i_a_r) - body_roll, np.deg2rad(self.i_a_r) + body_roll

    def calculate_body_roll(self, ay):
        """
        Calculates the body roll angle in radians given the lateral acceleration in g's
        """
        return (self.ds_f * self.ms_f + self.ds_r * self.ms_r) * ay / np.rad2deg(self.k_f + self.k_r) # were converting from 1/deg to 1/rad so we use np.rad2deg because its the inverse of the inverse

    def set_lltd(self, chassis=True):
        if chassis:
            self.LLTD, self.df_f, self.df_r = self.calculate_lltd_chassis(self.ms_f, self.ms_r)
        else:
            self.LLTD, self.df_f, self.df_r = self.calculate_lltd(self.ms_f, self.ms_r)
        return self.LLTD

    def calculate_lltd(self, ms_f, ms_r):
        self.ds_f = self.cg_height - self.z_f
        self.ds_r = self.cg_height - self.z_r
        # https://kktse.github.io/jekyll/update/2021/05/12/simplied-lateral-load-transfer-analysis.html
        rsd = self.k_f/(self.k_f+self.k_r)
        a = self.ds_f * ms_f + self.ds_r * ms_r
        df_f = (ms_f * self.z_f + rsd * a) / self.front_track
        df_r = (ms_r * self.z_r + (1 - rsd) * a) / self.rear_track
        return df_f / (df_r + df_f), df_f, df_r

    def calculate_lltd_chassis(self, ms_f, ms_r):
        self.ds_f = self.cg_height - self.z_f
        self.ds_r = self.cg_height - self.z_r
        # see the comment in find_contact_patch_loads for the derivation of these equations
        a = self.ds_f * ms_f + self.ds_r * ms_r
        df_f = (ms_f * self.z_f + a * self.k_f / ((self.k_r * self.k_c) / (self.k_r + self.k_c) + self.k_f)) / self.front_track
        df_r = (ms_r * self.z_r + a * self.k_r / ((self.k_f * self.k_c) / (self.k_f + self.k_c) + self.k_r)) / self.rear_track
        return df_f / (df_r + df_f), df_f, df_r

    def calculate_tire_delta_angle(self, delta, heave, pitch, roll):
        # with positive delta the right tire will be turned clockwise, and the negative toe angle will be subtracted from the delta angle
        # so a negative toe angle will make the tire turn more, capiche?
        
        delta_FL = delta + np.deg2rad(self.toe_front)
        delta_FR = delta - np.deg2rad(self.toe_front)
        delta_RL = np.deg2rad(self.toe_rear)
        delta_RR = -np.deg2rad(self.toe_rear)
        return delta_FL, delta_FR, delta_RL, delta_RR

    def calculate_slip_angles(self, V, omega, beta, delta_FL, delta_FR, delta_RL, delta_RR):
        # The tire model uses SAE sign convention see above comment, but tldr alpha is the opposite sign of beta, delta, and toe
        # this makes a ton of sense in the car coordinate system, but in the tire coordinate system the velocity vector isn't our datum
        # So you need to negate the delta angle to get the correct sign for the SAE sign convention that the tire model uses
        # based on equation 1.3 in the third edition of tyre and vehicle dynamics by pacejka
        alpha_FL = np.arctan((V * np.sin(beta) + (omega * self.a)) / (V * np.cos(beta) + (omega * self.front_track / 2))) - delta_FL
        alpha_FR = np.arctan((V * np.sin(beta) + (omega * self.a)) / (V * np.cos(beta) - (omega * self.front_track / 2))) - delta_FR
        alpha_RL = np.arctan((V * np.sin(beta) - (omega * self.b)) / (V * np.cos(beta) + (omega * self.rear_track / 2))) - delta_RL
        alpha_RR = np.arctan((V * np.sin(beta) - (omega * self.b)) / (V * np.cos(beta) - (omega * self.rear_track / 2))) - delta_RR
        return alpha_FL, alpha_FR, alpha_RL, alpha_RR

    def calculate_vel_at_tire(self, v, omega, beta):
        # beta might have to be negated here, im pretty sure it is correct now but im not 100% sure
        # velocity of the car in the velocity reference frame
        # and adjust for the radius of the turn, and the velocity
        # velocity of the tire in the car reference frame
        # at the radius of gyration of the tire
        v_fl = vel_at_tire(v, omega, beta, self.a, self.front_track / 2)
        v_fr = vel_at_tire(v, omega, beta, self.a, -self.front_track / 2)
        v_rl = vel_at_tire(v, omega, beta, -self.b, self.rear_track / 2)
        v_rr = vel_at_tire(v, omega, beta, -self.b, -self.rear_track / 2)
        return v_fl, v_fr, v_rl, v_rr

    def s_r_ind_edif(self, f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper = 0.2, lower = -0.3, p: float = 82500, flip_s_a=False, non_driven=False, mu_corr: float = 1.0):
        kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_target / 2, non_driven=non_driven, upper=upper, lower=lower, flip_s_a=flip_s_a, mu_corr=mu_corr, p=p)
        fx_b_targ = fx_target
        if bam_a or fx_a == 0:
            fx_b_targ -= fx_a
        kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_b_targ, non_driven=non_driven, upper=upper, lower=lower, flip_s_a=(not flip_s_a), mu_corr=mu_corr, p=p)
        # print(f"fx_targ: {fx_target:.4f} fx_a: {fx_a:.4f} fx_b_targ: {fx_b_targ:.4f}, bam_a: {bam_a} kappa_a: {kappa_a:.4f} fx_b: {fx_b:.4f}, bam_b: {bam_b} kappa_b: {kappa_b:.4f}")
        return kappa_a, bam_a, fx_a, kappa_b, bam_b, fx_b
    
    def s_r_ind_locked(self, f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper = 0.2, lower = -0.3, p: float = 82500, non_driven=False, mu_corr: float = 1.0, og_upper = 0.2, og_lower = -0.3, kappa=0.0, prev_kappa=0.0, prev_fx=[0.0, 0.0], i=0):
        """
        Solves for the slip ratio of a locked differential, also determines if the tire is saturated with Fx
        To do this, we use a second order taylor series approximation to solve for the slip ratio
        """
        if (fx_target > 0 and non_driven): # If the tire is non driven (eg front wheels) and the target Fx is positive (acceleration), then the tire is wont be reacting any torque
            _, actual_fx_a, _ = self.steady_state_mmd(f_z_a, s_a_a, 0.0, v_a, i_a_a, 0.0, flip_s_a=True, mu=mu_corr, no_long_include=True, p=p)
            _, actual_fx_b, _ = self.steady_state_mmd(f_z_b, s_a_b, 0.0, v_b, i_a_b, 0.0, flip_s_a=False, mu=mu_corr, no_long_include=True, p=p)
            return 0.0, False, actual_fx_a, 0.0, False, actual_fx_b
        if f_z_a <= 0.0:
            kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=False, p=p)
            return 0.0, False, 0.0, kappa_b, bam_b, fx_b
        if f_z_b <= 0.0:
            kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=True, p=p)
            return kappa_a, bam_a, fx_a, 0.0, False, 0.0
        if i > 20:
            k_a, k_b = variable_sr(v_a, v_b, prev_kappa)
            return k_a, False, prev_fx[0], k_b, False, prev_fx[1]
        # first we solve for 3 points with a small offset of b from our slip ratio kappa to get the first and second derivatives
        # here is what is going on here https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
        b = 0.0001
        d_kappa = 0.001
        kappas = np.array([kappa - b, kappa, kappa + b])
        k_a, k_b = variable_sr(v_a, v_b, kappas)
        # With a locked diff there is low speed locked diff behavior and so the need to disable some of the error checking in the solver that is valid for a one wheel iterator
        if self.fast_mf == None:
            fx_a, _, _ = self.mf_tire.s_r_sweep(f_z_a, s_a_a, k_a, i_a=i_a_a, v=v_a, flip_s_a=True, mu_corr=mu_corr, p=p)
            fx_b, _, _ = self.mf_tire.s_r_sweep(f_z_b, s_a_b, k_b, i_a=i_a_b, v=v_b, flip_s_a=False, mu_corr=mu_corr, p=p)
        else:
            fx_a, _, _ = self.fast_mf.solve_sr_sweep(f_z_a, s_a_a, k_a, p, i_a_a, v_a, 0.0, 0.0, mu_corr, True)
            fx_b, _, _ = self.fast_mf.solve_sr_sweep(f_z_b, s_a_b, k_b, p, i_a_b, v_b, 0.0, 0.0, mu_corr, False)
        # now we use the first and second derivatives to solve for the slip ratio
        fx = fx_a + fx_b
        fx_1, fx_2, fx_3 = fx[0], fx[1], fx[2]
        d_fx = (fx_3 - fx_1) / (2 * b)
        dd_fx = (fx_3 - 2 * fx_2 + fx_1) / (b ** 2)
        max_fxy_mag = 3 * (f_z_a + f_z_b) # limit the maxima used in the quadratic equation, it can jump all the way off the other end of the curve if we dont
        delta_fx = np.clip(fx_target, -max_fxy_mag, max_fxy_mag) - fx_2
        if d_fx ** 2 - 4 * dd_fx * delta_fx < 0:
            # use linear approximation if the quadratic equation has no real roots
            new_kappa = kappa + delta_fx / d_fx
            kappa_1 = 0.0
            kappa_2 = 0.0
        else:
            kappa_1 = (-d_fx + np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            kappa_2 = (-d_fx - np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            new_kappa = kappa - kappa_1 if abs(kappa_1) < abs(kappa_2) else kappa - kappa_2
        if d_fx < 0:
            new_kappa = (prev_kappa + kappa) / 2
            if i == 0 and (fx_a[1] < fx_a[0] or fx_a[1] < fx_a[2] or fx_b[1] < fx_b[0] or fx_b[1] < fx_b[2]):
                print(f"{f_z_a:.1f} BAD TIRE MODEL: NEGATIVE FX-SL SLOPE AT SL=0 i:{i}")

        if abs(new_kappa - kappa) < 0.0001 or abs(fx_target - fx_2) < 0.1:
            km_a, km_b = variable_sr(v_a, v_b, new_kappa)
            maxima_a = (km_a > upper - d_kappa) or (km_a < lower + d_kappa) or ((np.sign(fx_a[1] - fx_a[2]) == np.sign(fx_a[1])) and (np.sign(fx_a[1] - fx_a[0]) == np.sign(fx_a[1])))
            maxima_b = (km_b > upper - d_kappa) or (km_b < lower + d_kappa) or ((np.sign(fx_b[1] - fx_b[2]) == np.sign(fx_b[1])) and (np.sign(fx_b[1] - fx_b[0]) == np.sign(fx_b[1])))
            # if maxima: print(f"{f_z:.1f} MAXIMA")
            return km_a, maxima_a, fx_a[1], km_b, maxima_b, fx_b[1]
        if (kappa == og_upper and new_kappa > og_upper) or (kappa == og_lower and new_kappa < og_lower):
            # print(f"{f_z:.1f} LIMS")
            return k_a[1], True, fx_a[1], k_b[1], True, fx_b[1]
        # this is a stupid way of doing this, but it will stay until this code is completely rewritten
        new_kappa = sr_variable_lim(v_a, v_b, new_kappa, upper, lower)
        if d_fx < 0:
            if kappa > 0:
                upper = kappa
            else: # if kappa < 0
                lower = kappa
            kappa = prev_kappa
            fx_2 = prev_fx[0] + prev_fx[1]
        return self.s_r_ind_locked(f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, i=(i+1), upper = upper, lower = lower, og_upper = og_upper, og_lower = og_lower, kappa=new_kappa, prev_kappa=kappa, prev_fx=[fx_a[1], fx_b[1]], p=p, non_driven=non_driven, mu_corr=mu_corr)

    def s_r_ind(self, f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper = 0.2, lower = -0.3, p: float = 82500, non_driven=False, rear=True, mu_corr: float = 1.0):
        """
        Solves for the slip ratio of two opposing tires, also determines if one of the tires are saturated with Fx
        If one of the tires is saturated, we know that andy more torque applied to the wheel will bring the tire beyond its peak grip
        While in reality there are some cases there may be more performant to apply more torque to the wheel beyond its peak Fx
        to bring the other wheel closer to its peak Fx and saturate the the tires as a pair, we make the simplifying assumption that this method gets close enough
        this is done because it makes applying torque limits per axle simpler
        """
        if rear: diff_model = self.diff_model_rear
        else: diff_model = self.diff_model_front

        if diff_model == "open":
            kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=True, p=p)
            kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=False, p=p)
            hysteresis = 10
            if fx_a - hysteresis > fx_b:
                kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_b, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=True, p=p)
            elif fx_b - hysteresis > fx_a:
                kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_a, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=False, p=p)
            # print(f"fx_a: {fx_a:.3f}, fx_b: {fx_b:.3f}, fx_target: {(fx_target/2):.3f} kappa_a: {kappa_a:.3f}, kappa_b: {kappa_b:.3f} bam_a: {bam_a}, bam_b: {bam_b} s_a_a: {np.rad2deg(s_a_a):.3f}, s_a_b: {np.rad2deg(s_a_b):.3f} i_a_a: {np.rad2deg(i_a_a):.3f}, i_a_b: {np.rad2deg(i_a_b):.3f} v_a: {v_a:.3f}, v_b: {v_b:.3f}")
    
        elif diff_model == "locked":
            kappa_a, bam_a, fx_a, kappa_b, bam_b, fx_b = self.s_r_ind_locked(f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, p=p)

        else:
            if f_z_a < f_z_b:
                # print("R Heavy")
                kappa_a, bam_a, fx_a, kappa_b, bam_b, fx_b = self.s_r_ind_edif(f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper=upper, lower=lower, non_driven=non_driven, flip_s_a=True, mu_corr=mu_corr, p=p)
            else:
                # print("L Heavy")
                kappa_b, bam_b, fx_b, kappa_a, bam_a, fx_a = self.s_r_ind_edif(f_z_b, s_a_b, i_a_b, v_b, f_z_a, s_a_a, i_a_a, v_a, fx_target, upper=upper, lower=lower, non_driven=non_driven, flip_s_a=False, mu_corr=mu_corr, p=p)
        
        return kappa_a, kappa_b, (bam_a or bam_b)
    
    def s_r_sel(self, f_z, s_a, i_a, v, fx_targ, flip_s_a=False, upper=0.2, lower=-0.3, p: float = 82500, non_driven=False, mu_corr: float = 1.0):
        if self.fast_mf == None:
            return self.s_r(f_z, s_a, v, fx_targ, i_a=i_a, non_driven=non_driven, upper=upper, lower=lower, og_lower=lower, og_upper=upper, flip_s_a=flip_s_a, mu_corr=mu_corr, p=p)
        else:
            # print(f"{f_z}, {s_a}, {upper:.2f}, {lower:.2f}, {upper:.2f}, {lower:.2f}, 0.0, 0.0, 0.0, {p}, {i_a}, {v}, 0.0, 0.0, {mu_corr}, {flip_s_a}, {non_driven}, {fx_targ}, 0")
            kappa_a, bam_a, fx_a = self.fast_mf.s_r(f_z, s_a, upper, lower, upper, lower, 0.0, 0.0, 0.0, p, i_a, v, 0.0, 0.0, mu_corr, flip_s_a, non_driven, fx_targ, 0)
            # print(f"kappa_a: {kappa_a:.3f}, bam_a: {bam_a}, fx_a: {fx_a:.3f}")
            return kappa_a, bam_a, fx_a

    def s_r(self, f_z, s_a, v_avg, fx_target, i_a = 0.0, upper = 0.2, lower = -0.3, og_upper = 0.2, og_lower = -0.3, kappa=0.0, prev_kappa=0.0, prev_fx=0.0, i=0, p: float = 82500, non_driven=False, flip_s_a=False, mu_corr: float = 1.0):
        """
        Solves for the slip ratio of a single tire, also determines if the tire is saturated with Fx
        To do this, we use a second order taylor quadratic approximation to solve for the slip ratio
        """
        if (fx_target > 0 and non_driven): # If the tire is non driven (eg front wheels) and the target Fx is positive (acceleration), then the tire is wont be reacting any torque
            _, actual_fx, _ = self.steady_state_mmd(f_z, s_a, 0.0, v_avg, i_a, 0.0, flip_s_a=flip_s_a, mu=mu_corr, no_long_include=True, p=p)
            return 0.0, False, actual_fx
        if f_z <= 0.0:
            return 0.0, False, 0.0
        if i > 20:
            return prev_kappa, False, prev_fx
        # first we solve for 3 points with a small offset of b from our slip ratio kappa to get the first and second derivatives
        # here is what is going on here https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
        b = 0.0001
        d_kappa = 0.001
        kappas = np.array([kappa - b, kappa, kappa + b])
        if self.fast_mf == None:
            fx, _, _ = self.mf_tire.s_r_sweep(f_z, s_a, kappas, i_a=i_a, v=v_avg, flip_s_a=flip_s_a, mu_corr=mu_corr, p=p)
        else:
            fx, _, _ = self.fast_mf.solve_sr_sweep(f_z, s_a, kappas, p, i_a, v_avg, 0.0, 0.0, mu_corr, flip_s_a)
        # now we use the first and second derivatives to solve for the slip ratio
        fx_1, fx_2, fx_3 = fx[0], fx[1], fx[2]
        d_fx = (fx_3 - fx_1) / (2 * b)
        dd_fx = (fx_3 - 2 * fx_2 + fx_1) / (b ** 2)
        max_fxy_mag = 3 * f_z # limit the maxima used in the quadratic equation, it can jump all the way off the other end of the curve if we dont
        delta_fx = np.clip(fx_target, -max_fxy_mag, max_fxy_mag) - fx_2
        if d_fx ** 2 - 4 * dd_fx * delta_fx < 0:
            # use linear approximation if the quadratic equation has no real roots
            new_kappa = kappa + delta_fx / d_fx
            kappa_1 = 0.0
            kappa_2 = 0.0
        else:
            kappa_1 = (-d_fx + np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            kappa_2 = (-d_fx - np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            new_kappa = kappa - kappa_1 if abs(kappa_1) < abs(kappa_2) else kappa - kappa_2
        if d_fx < 0:
            new_kappa = (prev_kappa + kappa) / 2
            if i == 0:
                print(f"{f_z:.1f} BAD TIRE MODEL: NEGATIVE FX-SL SLOPE AT SL=0")

        if abs(new_kappa - kappa) < 0.0001 or abs(fx_target - fx_2) < 0.1:
            maxima = (new_kappa > upper - d_kappa) or (new_kappa < lower + d_kappa) or ((np.sign(fx_2 - fx_3) == np.sign(fx_2)) and (np.sign(fx_2 - fx_1) == np.sign(fx_2)))
            # if maxima: print(f"{f_z:.1f} MAXIMA")
            return new_kappa, maxima, fx_2
        if (kappa == og_upper and new_kappa > og_upper) or (kappa == og_lower and new_kappa < og_lower):
            # print(f"{f_z:.1f} LIMS")
            return kappa, True, fx_2
        new_kappa = max(min(new_kappa, upper), lower)
        if d_fx < 0:
            if kappa > 0:
                upper = kappa
            else: # if kappa < 0
                lower = kappa
            kappa = prev_kappa
            fx_2 = prev_fx
        return self.s_r(f_z, s_a, v_avg, fx_target, i_a=i_a, i=(i+1), upper = upper, lower = lower, og_upper = og_upper, og_lower = og_lower, kappa=new_kappa, prev_kappa=kappa, prev_fx=fx_2, p=p, non_driven=non_driven, flip_s_a=flip_s_a, mu_corr=mu_corr)

    def steady_state_mmd(self, fz, sa, kappa, v_avg, i_a, alpha, p: float = 82500, flip_s_a=False, mu: float = 1.0, no_long_include=False):
        if self.fast_mf == None:
            fx, fy, mz = self.mf_tire.steady_state_mmd(fz, sa, kappa, v=v_avg, flip_s_a=flip_s_a, i_a=i_a, mu_corr=mu, p=p)
        else:
            fx, fy, mz = self.fast_mf.solve_steady_state(fz, sa, kappa, p, i_a, v_avg, 0.0, 0.0, mu, flip_s_a)
        # it is in adapted ISO so pos Fy is neg lat acc
        if no_long_include:
            return -fy * np.cos(-alpha), fx * np.cos(-alpha), -mz
        return -fy * np.cos(-alpha) - fx * np.sin(-alpha), fx * np.cos(-alpha) + -fy * np.sin(-alpha), -mz # per the z down orientation of the ttc data
   
    def solve_for_yaw(self, ay_targ, v_avg, beta_x, delta_x, mu_corr, use_drag=True, vecs=False, sr_lim=0.2):
        yaw_it = 0.0
        ax_c, ay_c = to_car_frame(0.0, ay_targ, beta_x)
        omega = ay_targ / v_avg # Initial yaw rate [rad/s]
        kappax_fl, kappax_fr, kappax_rl, kappax_rr = 0, 0, 0, 0
        bruh, long_error, total_fx = 0, 1, 0
        drag = 0
        if use_drag:
            drag = 0.5 * 1.225 * v_avg**2 * self.cd
        safl, safr, sarl, sarr = 0, 0, 0, 0
        long_err = 0.001
        lfx = self.mass * ax_c + drag
        while bruh < 25 and (long_error > long_err):
            fzfl, fzfr, fzrl, fzrr, _, _ = self.find_contact_patch_loads(long_g=ax_c, lat_g=ay_c, vel=v_avg)
            delta_fl, delta_fr, delta_rl, delta_rr = self.calculate_tire_delta_angle(delta_x, 0.0, 0.0, 0.0)
            [safl, safr, sarl, sarr] = self.calculate_slip_angles(v_avg, omega, beta_x, delta_fl, delta_fr, delta_rl, delta_rr)
            v_fl, v_fr, v_rl, v_rr = self.calculate_vel_at_tire(v_avg, omega, beta_x)
            ia_fl, ia_fr, ia_rl, ia_rr = self.calculate_ia(ay_c, fzfl, fzfr, fzrl, fzrr)
            kappax_rl, kappax_rr, _ = self.s_r_ind(fzrl, clip(sarl), ia_rl, v_rl, fzrr, clip(sarr), ia_rr, v_rr, lfx, mu_corr=mu_corr, upper=sr_lim)

            # Generate slip angles for two-track model as a function of
            # beta and delta along with included parameters for toe.
            fyfl, fxfl, mzfl = self.steady_state_mmd(fzfl, clip(safl), kappax_fl, v_fl, ia_fl, delta_fl, flip_s_a=True, mu=mu_corr)
            fyfr, fxfr, mzfr = self.steady_state_mmd(fzfr, clip(safr), kappax_fr, v_fr, ia_fr, delta_fr, mu=mu_corr)
            fyrl, fxrl, mzrl = self.steady_state_mmd(fzrl, clip(sarl), kappax_rl, v_rl, ia_rl, delta_rl, flip_s_a=True, mu=mu_corr)
            fyrr, fxrr, mzrr = self.steady_state_mmd(fzrr, clip(sarr), kappax_rr, v_rr, ia_rr, delta_rr, mu=mu_corr)
            # Normalized yaw moments created by the front axle and rear
            # axle about the CG
                    
            # Looking from the top down, the positive yaw moment is clock wise
            CN_fl = fyfl * self.a + self.front_track * fxfl / 2 + mzfl
            CN_fr = fyfr * self.a - self.front_track * fxfr / 2 + mzfr
            CN_rl = -fyrl * self.b + self.rear_track * fxrl / 2 + mzrl # (-) to indicate opposing moment to front axle
            CN_rr = -fyrr * self.b - self.rear_track * fxrr / 2 + mzrr # (-) to indicate opposing moment to front axle
            CN_total = CN_fl + CN_fr + CN_rl + CN_rr
            yaw_it = CN_total / self.izz # yaw accel [rad/s^2]
            total_fy = fyfl + fyfr + fyrr + fyrl
            ay_it = total_fy / self.mass # lat accel [m/s^2]
            total_fx = fxfl + fxfr + fxrr + fxrl - drag
            ax_it = total_fx / self.mass # lat accel [m/s^2]
            total_mz = mzfl + mzfr + mzrl + mzrr
            total_fz = fzfl + fzfr + fzrl + fzrr
            # Recalculate new weight transfer and tire vertical loads
            # due to new lat accel
            ax_v, ay_v = to_vel_frame(ax_it, ay_it, beta_x) # equation 51 and 52 in the patton paper
            long_error = abs(ax_c - ax_it)
            bruh += 1
            lfx += self.mass * (ax_c - ax_it)
        
        if vecs:
            return ay_v, yaw_it, ax_v, bruh, [fyfl, fyfr, fyrl, fyrr, total_fy], [fxfl, fxfr, fxrl, fxrr, total_fx], [mzfl, mzfr, mzrl, mzrr, total_mz], [fzfl, fzfr, fzrl, fzrr, total_fz], [safl, safr, sarl, sarr, 0.0], [delta_fl, delta_fr, delta_rl, delta_rr, 0.0], [kappax_fl, kappax_fr, kappax_rl, kappax_rr, 0.0]

        return ay_v, yaw_it, ax_v, bruh

    def find_skidpad_angles(self, ay_targ, vel, mu_corr, delta_x=0.0, beta_x=0.0, sr_lim=0.2):
        # iterate until the ay converges
        beta_max = np.deg2rad(25)
        delta_max = np.deg2rad(30)
        del_max, del_min = delta_max, -delta_max
        # del_min = -np.deg2rad(5)
        # res = least_squares(loss_func, [beta_x, delta_x], args=(self, ay_targ, vel, mu_corr, sr_lim,), bounds=((-beta_max, del_min), (beta_max, del_max)), verbose=0)
        res: OptimizeResult = minimize(loss_func_two, [beta_x, delta_x], args=(self, ay_targ, vel, mu_corr, sr_lim,), bounds=((-beta_max, beta_max), (del_min, del_max)), options=dict(disp=False), method="Nelder-Mead")
        beta, delta = res.x
        ay, yaw, ax, bruh = self.solve_for_yaw(ay_targ, vel, beta, delta, mu_corr, sr_lim=sr_lim)
        return beta, delta, ay, yaw, ax, bruh

    def solve_skidpad_time(self, radius, mu_corr, vel=5, sr_lim=0.2):
        last_beta, last_delta, last_ay, last_yaw, last_ax, last_bruh = 0, 0, 0.0, 0.0, 0.0, 0.0
        last_vel = vel
        vel_inc = 2
        while True:
            ay_targ = vel**2 / radius * -1 # ay is negative for right hand turn on skidpad in SAE sign convention
            beta, delta, ay, yaw, ax, bruh = self.find_skidpad_angles(ay_targ, vel, mu_corr, beta_x=last_beta, delta_x=last_delta, sr_lim=sr_lim)
            if ay == np.inf or abs(ay - ay_targ) > 0.01 or bruh == 25:
                if vel_inc < 0.001:
                    break
                vel_inc /= 2
                vel = last_vel + vel_inc
                continue
            last_beta, last_delta, last_ay, last_yaw, last_ax, last_bruh = beta, delta, ay, yaw, ax, bruh
            last_vel = vel
            vel += vel_inc

        lap_time = 2 * np.pi * radius / last_vel
        return last_beta, last_delta, last_ay, last_yaw, last_ax, last_bruh, last_vel, lap_time

def MMD_3D_Graphs(ay_it1, yaw_it1, ax_it1, vels, valid, mmd = True):
    fig = go.Figure()

    if mmd:
        for v, vel in enumerate(vels):
            mask = valid[:, :, :, v] == 1
            inds = np.argwhere(mask)
            inds = inds[inds[:, 0].argsort()] # First sort doesn't need to be stable.
            inds = inds[inds[:, 2].argsort(kind='mergesort')]
            inds = inds[inds[:, 1].argsort(kind='mergesort')]
            delim_locations = np.where((inds[:-1, 0] + 1 != inds[1:, 0]) | (inds[:-1, 1] != inds[1:, 1]) | (inds[:-1, 2] != inds[1:, 2]))
            delim_ind = [0]
            delim_ind.extend(delim_locations[0] + 1)
            ay_it2, yaw_it2, ax_it2 = [], [], []
            for i in range(len(delim_ind)-1):
                ay_it2.extend(ay_it1[inds[delim_ind[i]:delim_ind[i+1], 0], inds[delim_ind[i]:delim_ind[i+1], 1], inds[delim_ind[i]:delim_ind[i+1], 2], v] * G)
                ay_it2.append(None)
                yaw_it2.extend(yaw_it1[inds[delim_ind[i]:delim_ind[i+1], 0], inds[delim_ind[i]:delim_ind[i+1], 1], inds[delim_ind[i]:delim_ind[i+1], 2], v])
                yaw_it2.append(None)
                ax_it2.extend(ax_it1[inds[delim_ind[i]:delim_ind[i+1], 0], inds[delim_ind[i]:delim_ind[i+1], 1], inds[delim_ind[i]:delim_ind[i+1], 2], v] * G)
                ax_it2.append(None)
            fig.add_trace(
                go.Scatter3d(
                    x=ay_it2,
                    y=yaw_it2,
                    z=ax_it2,
                    mode='lines',
                    marker=dict(color='blue'), legendgroup=f"group{v}", name=f"Vel: {vel:.3f}", showlegend=True
                )
            )
            inds = np.argwhere(mask)
            inds = inds[inds[:, 1].argsort()] # First sort doesn't need to be stable.
            inds = inds[inds[:, 2].argsort(kind='mergesort')]
            inds = inds[inds[:, 0].argsort(kind='mergesort')]
            delim_locations = np.where((inds[:-1, 0] != inds[1:, 0]) | (inds[:-1, 1] + 1 != inds[1:, 1]) | (inds[:-1, 2] != inds[1:, 2]))
            delim_ind = [0]
            delim_ind.extend(delim_locations[0] + 1)
            ay_it3, yaw_it3, ax_it3 = [], [], []
            for i in range(len(delim_ind)-1):
                ay_it3.extend(ay_it1[inds[delim_ind[i]:delim_ind[i+1], 0], inds[delim_ind[i]:delim_ind[i+1], 1], inds[delim_ind[i]:delim_ind[i+1], 2], v] * G)
                ay_it3.append(None)
                yaw_it3.extend(yaw_it1[inds[delim_ind[i]:delim_ind[i+1], 0], inds[delim_ind[i]:delim_ind[i+1], 1], inds[delim_ind[i]:delim_ind[i+1], 2], v])
                yaw_it3.append(None)
                ax_it3.extend(ax_it1[inds[delim_ind[i]:delim_ind[i+1], 0], inds[delim_ind[i]:delim_ind[i+1], 1], inds[delim_ind[i]:delim_ind[i+1], 2], v] * G)
                ax_it3.append(None)
            fig.add_trace(
                go.Scatter3d(
                    x=ay_it3,
                    y=yaw_it3,
                    z=ax_it3,
                    mode='lines',
                    marker=dict(color='red'), legendgroup=f"group{v}", name=f"Vel: {vel:.3f}", showlegend=False
                )
            )

    ay_it_err = ay_it1[valid < 1].flatten()
    yaw_it_err = yaw_it1[valid < 1].flatten()
    ax_it_err = ax_it1[valid < 1].flatten()
    fig.add_trace(
        go.Scatter3d(
            x=ay_it_err * G,
            y=yaw_it_err,
            z=ax_it_err * G,
            mode='markers',
            marker=dict(size=2, color='orange'),
            name="Error"
        )
    )

    fig.update_layout(scene = dict(
        xaxis_title='Lat Acc. (m/s^2)',
        yaxis_title='Yaw Rate (rad/sec^2)',
        zaxis_title='Lon Acc. (m/s^2)')
    )
    fig.update_layout(template="plotly_dark",
                        scene = dict(
        xaxis = dict(range=[-30, 30],),
                     yaxis = dict(range=[-75, 75],),
                     zaxis = dict(range=[-25, 25],),),
                          title_text="Parameter Optimization")

    return fig


if __name__ == '__main__':
    car = Car()
    v_average, track_mu = 15, 1.0
    start = time.time()
    ay_it1, yaw_it1, cn1, ax_it1, delta, beta, kappa, valid, LAS_points = car.MMD_3D_Long()
    print(f"Time: {time.time() - start}")
    x, y, z, i, j, k = car.limit_surface(v_average, 0.5)

    fig3 = MMD_3D_Graphs(ay_it1, yaw_it1, ax_it1, car.vels, valid)#, mmd=False
    fig3.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightpink', opacity=0.50, showscale=True))# 8 vertices of a cube
    
    fig3.show()
    

    

