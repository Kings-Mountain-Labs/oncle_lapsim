from car_configuration import Car
from gps_importer import *
import time
from constants import *
from lap_sims import RunSim
from previous_cars import sr_9

if __name__ == '__main__':
    # This python script is only here so that the the simulation function can be benchmarked and optimized
    start = time.time()
    track = get_MIS_2017_AX_3_track(50)
    # car = Car()
    car = sr_9()
    # car.debug = True
    v_average = 15
    mu = 0.65 # mu correction factor, a value of 1 means the road is sandpaper and the actual value should be something lower but im kinda just setting this value to overfit atm
    target = 0.0001
    sim = RunSim(track, car)
    sim.simulate(sim_type='ian', v_average=v_average, mu=mu, convergence_target=target)
    sim.plot()
    sim.plot_vs(distance=True, delta_beta_est=True, fz=False, seperate_angles=True, yaw_acc=False, seperate_acc=False, power_draw=True)