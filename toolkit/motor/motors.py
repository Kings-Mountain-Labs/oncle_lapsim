from .motor import Motor


# These motors use the datasheet version 5.4 values, it is still unknown when the motor design changed
def get_emrax_228_mv_5_4():
    motor = Motor('EMRAX 228 MV V5.4')
    motor.Ld = 0.000076
    motor.Lq = 0.000079
    motor.Rs = 0.007
    motor.max_phase_current = 340
    motor.max_rate = 6500
    motor.Fl = 0.0355
    motor.poles = 10
    motor.rotor_inertia = 0.0383
    motor.mass = 12.4
    return motor

def get_emrax_228_hv_5_4():
    motor = Motor('EMRAX 228 HV V5.4')
    motor.Ld = 0.000177
    motor.Lq = 0.000183
    motor.Rs = 0.0167
    motor.max_phase_current = 240
    motor.max_rate = 6500
    motor.Fl = 0.0542
    motor.poles = 10
    motor.rotor_inertia = 0.0383
    motor.mass = 12.4
    return motor

def get_emrax_228_lv_5_4():
    motor = Motor('EMRAX 228 LV V5.4')
    motor.Ld = 0.0000103
    motor.Lq = 0.0000106
    motor.Rs = 0.0011
    motor.max_phase_current = 900
    motor.max_rate = 6500
    motor.Fl = 0.0131
    motor.poles = 10
    motor.rotor_inertia = 0.0383
    motor.mass = 12.4
    return motor