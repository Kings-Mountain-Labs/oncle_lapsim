from toolkit.common.constants import *
from toolkit.lapsim import MultiSim
from toolkit.cars import Car
from toolkit.lap import tracks
from toolkit.las_solvers import Octahedral_LAS, Multi_Layer_LAS, LAS
from toolkit.steady_state_solver import LS_Solver, Parachute, Iterative_Solver
import numpy as np

track_one = tracks.get_Lincoln_2017_AX_track_mixed(50)
track_two = tracks.get_MIS_2017_AX_1_track(50)
track_three = tracks.get_MIS_2017_AX_3_track(50)
tracks = [track_one, track_two, track_three]
# tracks = []
v_average = 15
mu_corr = 0.65 # mu correction factor, a value of 1 means the road is sandpaper and the actual value should be something lower but im kinda just setting this value to overfit atm
target = 0.001
# solver = LS_Solver()
solver = Iterative_Solver()
# las = Octahedral_LAS(solver=solver)
las = Multi_Layer_LAS(solver=solver)

# here we make a function that generates a car with a given roll stiffness distribution and chassis roll stiffness
# the first and second arguments take the values of the first and second sweep variables that you give to run_sim
def gen_car_lltd(rsd, k_c):
    total_roll_stiffness = 900 * FTLB_TO_NM
    car = Car()
    car.k_f = total_roll_stiffness * rsd
    car.k_r = total_roll_stiffness * (1-rsd)
    car.k_c = k_c * FTLB_TO_NM
    car.description = f"rsd:{rsd:.2f} k_c:{k_c:.2f}"
    return car
# do a sweep of roll stiffness distribution and chassis roll stiffness
rsd_range = np.linspace(0.25, 0.5, 5)
k_c_r = np.linspace(500, 2500, 5) # np.array([1000]) #
sim = MultiSim(tracks, gen_car_lltd, rsd_range, k_c_r, "Roll Stiffness Distribution", "Chassis Roll Stiffness (lbf/deg)")


sim.run_sim(las=las, convergence_target=target, mu=mu_corr, sim_type='qts')
sim.plot_tracks()
sim.plot_LAS_corners()
sim.plot_LLTD()