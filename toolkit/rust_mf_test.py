from car_configuration import Car
import time
from constants import *
from tire_model.fast_pacejka import *

if __name__ == '__main__':
    start = time.time()
    car = Car()
    fast_mf = get_rs_pacejka(car.mf_tire)
    start = time.time()
    for i in range(1000):
        _ = fast_mf.solve_steady_state(1000, 0.1, 0.01, 0.0, 0.1, 15.0, 0.0, 0.0, 1.0, True)
    print(time.time() - start)
    start = time.time()
    for i in range(1000):
        _ = car.mf_tire.steady_state_mmd(1000, 0.1, 0.01, 0.0, 0.1, 15.0, 0.0, 1.0, True)
    print(time.time() - start)
    print(fast_mf.solve_steady_state(1000, 0.1, 0.01, 0.0, 0.1, 15.0, 0.0, 0.0, 1.0, True))
    print(car.mf_tire.steady_state_mmd(1000, 0.1, 0.01, 0.0, 0.1, 15.0, 0.0, 1.0, True))