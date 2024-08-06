

class Inverter:
    i_d_limit: float
    i_q_limit: float

    def __init__(self, i_d_limit, i_q_limit, max_phase_current):
        self.i_d_limit = i_d_limit
        self.i_q_limit = i_q_limit
        self.max_phase_current = max_phase_current