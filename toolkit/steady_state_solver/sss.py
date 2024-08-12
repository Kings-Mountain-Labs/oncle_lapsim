import numpy as np
from toolkit.cars.car_configuration import Car
from abc import ABC, abstractmethod

class Steady_State_Solver(ABC):
    set_solver: str

    @abstractmethod
    def solve_for_long(self, car: Car, v_avg, long_g, delta_x = 0, beta_x = 0, mu_corr: float = 1.0, ay_it = 0.0, use_drag = False, long_err = 0.01, lat_err = 0.001, zeros = True, use_torque_lim=False, use_break_lim=True) -> tuple[float, float, float, float, int, bool]:
        pass
        