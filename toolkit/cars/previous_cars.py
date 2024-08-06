from car_configuration import Car
from constants import *

def sr_9():
    # https://docs.google.com/presentation/d/1IPxDEFjDicdDgofIkFA93hF2H0lZS8ht/edit#slide=id.p5
    car = Car(mass=570*LB_TO_KG, front_axle_weight=0.49)
    car.mass_unsprung = 100 * LB_TO_KG
    car.k_f = 271 * FTLB_TO_NM
    car.k_r = 359 * FTLB_TO_NM
    car.k_c = 630 * FTLB_TO_NM
    car.cg_height = 9.4 * IN_TO_M
    car.z_f = 1 * IN_TO_M
    car.z_r = 2 * IN_TO_M
    return car