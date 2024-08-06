from .inverter import Inverter

def get_pm100dx_with_228mv():
    return Inverter(221, 453)

def get_pm100dx_with_228hv():
    return Inverter(150, 339)

def get_pm100dx_with_208mv():
    return Inverter(425, 425)

def get_pm100dx_with_208hv():
    return Inverter(283, 150)

def get_pm100dx_with_188mv():
    return Inverter(424, 200)

def get_pm100dx_with_188hv():
    return Inverter(283, 150)