from .motor import Motor
import astropy.units as u


# These motors use the datasheet version 5.4 values, it is still unknown when the motor design changed
def get_emrax_228_mv_5_4():
    motor = Motor('EMRAX 228 MV V5.4')
    motor.Ld = 76 * u.uH
    motor.Lq = 79 * u.uH
    motor.Rs = 0.007 * u.ohm
    motor.max_phase_current = 340 * u.A
    motor.max_rate = 6500 * u.rpm
    motor.Fl = 0.0355 * u.Wb
    motor.poles = 10
    motor.rotor_inertia = 0.0383 * u.kg * u.m**2
    motor.mass = 12.4 * u.kg
    return motor

def get_emrax_228_hv_5_4():
    motor = Motor('EMRAX 228 HV V5.4')
    motor.Ld = 177 * u.uH
    motor.Lq = 183 * u.uH
    motor.Rs = 0.0167 * u.ohm
    motor.max_phase_current = 240 * u.A
    motor.max_rate = 6500 * u.rpm
    motor.Fl = 0.0542 * u.Wb
    motor.poles = 10
    motor.rotor_inertia = 0.0383 * u.kg * u.m**2
    motor.mass = 12.4 * u.kg
    return motor

def get_emrax_228_lv_5_4():
    motor = Motor('EMRAX 228 LV V5.4')
    motor.Ld = 10.3 * u.uH
    motor.Lq = 10.6 * u.uH
    motor.Rs = 0.0011 * u.ohm
    motor.max_phase_current = 900 * u.A
    motor.max_rate = 6500 * u.rpm
    motor.Fl = 0.0131 * u.Wb
    motor.poles = 10
    motor.rotor_inertia = 0.0383 * u.kg * u.m**2
    motor.mass = 12.4 * u.kg
    return motor