import numpy as np
from .transforms import calculate_real_power, calc_abc
from astropy import units as u

class Motor:
    name: str = ""
    max_rate: u.Quantity[u.PhysicalType("angular velocity")] = 6500 * u.rpm
    max_torque: u.Quantity[u.PhysicalType("torque")] = 231 * u.N * u.m
    max_phase_current: u.Quantity[u.PhysicalType("current")] = 340 * u.A
    Ld: u.Quantity[u.PhysicalType("inductance")] = 76 * u.uH
    Lq: u.Quantity[u.PhysicalType("inductance")] = 79 * u.uH
    Rs: u.Quantity[u.PhysicalType("resistance")] = 0.0071 * u.ohm
    poles: int = 10
    Fl: u.Quantity[u.PhysicalType("magnetic flux")] = 0.0355 * u.Wb
    rotor_inertia: u.Quantity[u.PhysicalType("moment of inertia")] = 0.0383 * u.kg * u.m**2
    mass: u.Quantity[u.PhysicalType("mass")] = 12.4 * u.kg

    def get_torque(self, I_d, I_q):
        return 3 / 2 * self.poles * I_q * (self.Fl - I_d * (self.Lq - self.Ld))

    def get_qd_currents(self, w, torque_req, voltage, use_mtpa=False):
        w_e = w * self.poles
        if w_e == 0:
            w_e += 1e-3
        v_max = (voltage / np.sqrt(3)) - (self.Rs * self.max_phase_current)
        # https://www.mathworks.com/help/mcb/ref/mtpacontrolreference.html
        I_m_ref = 2 * torque_req / (3 * self.poles * self.Fl)
        I_m = min(I_m_ref, self.max_phase_current)
        if use_mtpa:
            I_d_mtpa = (self.Fl / (4 * (self.Lq - self.Ld))) - np.sqrt(
                (self.Fl**2 / (16 * (self.Lq - self.Ld) ** 2)) + I_m**2 / 2
            )
            I_q_mtpa = np.sqrt(I_m**2 - I_d_mtpa**2)
        else:
            I_q_mtpa = I_m_ref
            I_d_mtpa = 0
        w_base = (1 / self.poles) * (
            v_max
            / np.sqrt((self.Lq * I_q_mtpa) ** 2 + (self.Fl + self.Ld * I_d_mtpa) ** 2)
        )

        # get the smallest root that is greater than 0
        # Binary search for max T_ref with real solutions
        T_ref_low = 0  # you may adjust this based on any given bounds
        T_ref_high = torque_req * 2.0  # this will make the middle value the first guess
        T_ref_mid = (T_ref_low + T_ref_high) / 2.0
        epsilon = 1e-5  # precision
        was_good = False
        while T_ref_high - T_ref_low > epsilon or not was_good:
            T_ref_mid = (T_ref_low + T_ref_high) / 2.0
            if any(np.isreal(self.calc_iq_fw_roots(w_e, v_max, T_ref_mid))):
                was_good = True
                if T_ref_low == 0 and T_ref_high == torque_req * 2.0:
                    break
                T_ref_low = T_ref_mid
            else:
                T_ref_high = T_ref_mid
                was_good = False

        roots = self.calc_iq_fw_roots(w_e, v_max, T_ref_mid)
        roots = roots[np.isreal(roots)]
        I_q_fw = min(max(max(roots.real), 0), self.I_qmax)
        I_d_fw = -self.Fl / self.Ld + (1 / self.Ld) * np.sqrt(
            (v_max**2 / (w_e**2)) - (self.Lq * I_q_fw) ** 2
        )
        k = 1
        if I_d_fw > 0:
            I_d_fw = 0
        if I_d_fw < -self.I_dmax:
            I_d_fw = -self.I_dmax
            k = 2

        # print(f"I_d_fw: {I_d_fw} I_q_fw: {I_q_fw} a: {self.calc_max_iq(w_e, voltage, I_d_fw)}")
        I_q_fw = min(
            np.sqrt(self.max_phase_current**2 - I_d_fw**2),
            I_q_fw,
            self.calc_max_iq(w_e, voltage, I_d_fw),
        )
        if np.isnan(I_q_fw):
            I_q_fw = 0
        if I_q_fw < 0.5:
            k = 3
        o = -1
        if w <= w_base:
            I_d = I_d_mtpa
            I_q = I_q_mtpa
            o = 0
        else:
            I_d = I_d_fw
            I_q = I_q_fw
            o = k

        achieved_torque = self.get_torque(I_d, I_q)
        return I_d, I_q, achieved_torque, w_base, v_max, o

    def calc_max_iq(self, w_e, v, I_d):
        U_ac = v / np.sqrt(3)
        denom = self.Ld**2 * w_e**2 + self.Rs**2
        first_term = -1 * self.Rs * w_e * (self.Fl - I_d * self.Ld + I_d * self.Lq)
        the_wall = (
            (-1 * self.Fl**2 * self.Ld**2 * w_e**4)
            + (-2 * self.Fl * I_d * self.Ld**2 * self.Lq * w_e**4)
            + (-2 * self.Fl * I_d * self.Ld * self.Rs**2 * w_e**2)
            + (-1 * I_d**2 * self.Ld**2 * self.Lq**2 * w_e**4)
            + (-2 * I_d**2 * self.Lq * self.Ld * self.Rs**2 * w_e**2)
            + (-1 * I_d**2 * self.Rs**4)
            + (self.Ld**2 * U_ac**2 * w_e**2)
            + (self.Rs**2 * U_ac**2)
        )
        if the_wall < 0:
            return 0.0
        # print(f"a: {(first_term + np.sqrt(the_wall)) / denom} b: {(first_term - np.sqrt(the_wall)) / denom}")
        return (first_term + np.sqrt(the_wall)) / denom

    def calc_iq_fw_roots(self, w_e, v_max, torque_req):
        coeffs = [
            9
            * self.poles**2
            * (self.Ld - self.Lq) ** 2
            * self.Lq**2
            * w_e**2,  # Coefficient of i_{q_fw}^4
            0,  # Coefficient of i_{q_fw}^3 (since it doesn't appear in the equation)
            9 * self.poles**2 * self.Fl**2 * self.Lq**2 * w_e**2
            - 9
            * self.poles**2
            * (self.Ld - self.Lq) ** 2
            * v_max**2,  # Coefficient of i_{q_fw}^2
            -12
            * torque_req
            * self.poles
            * self.Fl
            * self.Ld
            * self.Lq
            * w_e**2,  # Coefficient of i_{q_fw}
            4 * torque_req**2 * self.Ld**2 * w_e**2,  # Constant term
        ]

        # Calculate the roots using NumPy
        return np.roots(coeffs)

    @property
    def i_max(self):
        return self.max_phase_current * np.sqrt(2)

    def calc_mtpa(self, current):
        i_dmtpa = (
            self.Fl - np.sqrt(self.Fl**2 - 8 * ((self.Lq - self.Ld) ** 2) * current**2)
        ) / (4 * (self.Ld - self.Lq))
        i_qmtpa = np.sqrt(i_dmtpa**2 - ((self.Fl / (self.Lq - self.Ld)) * i_dmtpa))
        return i_dmtpa, i_qmtpa

    def calc_iq(self, torque, id):
        return torque / ((3 / 2) * self.poles * (self.Fl - ((self.Lq - self.Ld) * id)))

    def calc_within_emf(self, w, voltage, id, iq):
        u_ac = voltage / np.sqrt(2)
        v_d = self.Rs * id - w * self.Lq * iq
        v_q = self.Rs * iq + w * (self.Fl + self.Ld * id)
        v_ac = np.sqrt(v_d**2 + v_q**2)
        return v_ac <= u_ac

    def calc_emf_limit(self, w, voltage, points):
        w_e = self.poles * w
        u_ac = voltage / np.sqrt(2)
        # Calculate points along the profile where v_ac = u_ac, we will solve for points at different angles from the center of the limit profile
        # The center of the limit profile is at the point (0, -l_d_center)
        l_d_center = -self.Fl / self.Ld
        angles = np.linspace(-np.pi, np.pi, points)
        m = np.tan(angles)
        s = -l_d_center * m
        # Solve for the points where the limit profile intersects the solution lines
        # i_q = m * i_d + b
        a = (
            self.Rs**2 * (1 + m**2)
            + w_e**2 * (self.Ld**2 + (self.Lq**2 * m**2))
            + (2 * (self.Ld - self.Lq) * m * self.Rs * w_e)
        )
        b = (
            (2 * self.Rs * w_e * (((self.Ld - self.Lq) * s) + (self.Fl * m)))
            + (2 * w_e**2 * ((m * s * self.Lq**2) + (self.Fl * self.Ld)))
            + (2 * m * s * self.Rs**2)
        )
        c = (
            (w_e**2 * ((self.Lq**2 * s**2) + self.Fl**2))
            + (s**2 * self.Rs**2)
            + (2 * w_e * self.Fl * self.Rs * s)
            - u_ac**2
        )
        i_d_neg = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        i_d_pos = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        i_d = np.zeros(points)
        i_d = i_d_pos
        i_d[angles > np.pi / 2] = i_d_neg[angles > np.pi / 2]
        i_d[angles < -np.pi / 2] = i_d_neg[angles < -np.pi / 2]
        i_q = m * i_d + s
        return i_d, i_q

    def get_voltages(self, w, id, iq):
        w_e = self.poles * w
        v_d = self.Rs * id - w_e * self.Lq * iq
        v_q = self.Rs * iq + w_e * (self.Fl + self.Ld * id)
        return v_d, v_q

    def calc_current_limit(self, points):
        angles = np.linspace(0, 2 * np.pi, points)
        i_d = self.i_max * np.cos(angles)
        i_q = self.i_max * np.sin(angles)
        return i_d, i_q

    def calculate_power(self, w, id, iq):
        i_a, i_b, i_c = calc_abc(0, id, iq)
        v_d, v_q = self.get_voltages(w, id, iq)
        v_a, v_b, v_c = calc_abc(0, v_d, v_q)
        power, _ = calculate_real_power(v_a, v_b, v_c, i_a, i_b, i_c)
        return power * 2
