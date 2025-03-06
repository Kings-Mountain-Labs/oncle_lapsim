from typing import Dict, List
from dataclasses import dataclass
from numpy.typing import ArrayLike
import numpy as np
from .pacejka_coefficients import PacejkaModel
from .tire_fitting_masks import LABELS, NAMES
from toolkit.loading_util import make_path

MODEL_DEFAULTS: Dict = {"FILE_TYPE": 'default', "FILE_VERSION": 3.0, "FILE_FORMAT": 'default',
                        "LENGTH": 'default', "FORCE": 'default', "ANGLE": 'default', "MASS": 'default', "TIME": 'default',
                        "FITTYP": 62, "TYRESIDE": 'LEFT', "LONGVL": 16.7, "VXLOW": 1.0,
                        "ROAD_INCREMENT": 0.01, "ROAD_DIRECTION": 1.0,
                        "PROPERTY_FILE_FORMAT": 'default', "USER_SUB_ID": 815,
                        "N_TIRE_STATES": 4.0, "USE_MODE": 124, "HMAX_LOCAL": 2.5E-4,
                        "TIME_SWITCH_INTEG": 0.1,
                        "UNLOADED_RADIUS": 0.3135, "WIDTH": 0.205, "ASPECT_RATIO": 0.6, "RIM_RADIUS": 0.1905, "RIM_WIDTH": 0.152,
                        "INFLPRES": 220000.0, "NOMPRES": 220000.0,
                        "MASS1": 9.3,
                        "IXX": 0.391, "IYY": 0.736,
                        "BELT_MASS": 7.0, "BELT_IXX": 0.34, "BELT_IYY": 0.6,
                        "GRAVITY": -9.81,
                        "FNOMIN": 1500.0,
                        "VERTICAL_STIFFNESS": 200000.0, "VERTICAL_DAMPING": 50.0,
                        "MC_CONTOUR_A": 0.5, "MC_CONTOUR_B": 0.5,
                        "BREFF": 8.4, "DREFF": 0.27, "FREFF": 0.07,
                        "Q_RE0": 1.0,
                        "Q_V1": 0.0, "Q_V2": 0.0, "Q_FZ2": 1.0E-4, "Q_FCX": 0.0, "Q_FCY": 0.0, "Q_CAM": 0.0,
                        "PFZ1": 0.0,
                        "BOTTOM_OFFST": 0.01, "BOTTOM_STIFF": 2000000.0,
                        "LONGITUDINAL_STIFFNESS": 300000.0, "LATERAL_STIFFNESS": 100000.0, "YAW_STIFFNESS": 5000.0,
                        "FREQ_LONG": 80.0, "FREQ_LAT": 40.0, "FREQ_YAW": 50.0, "FREQ_WINDUP": 70.0,
                        "DAMP_LONG": 0.04, "DAMP_LAT": 0.04, "DAMP_YAW": 0.04, "DAMP_WINDUP": 0.04, "DAMP_RESIDUAL": 0.0020, "DAMP_VLOW": 0.0010,
                        "Q_BVX": 0.0, "Q_BVT": 0.0,
                        "PCFX1": 0.0, "PCFX2": 0.0, "PCFX3": 0.0,
                        "PCFY1": 0.0, "PCFY2": 0.0, "PCFY3": 0.0,
                        "PCMZ1": 0.0,
                        "Q_RA1": 0.5, "Q_RA2": 1.0, "Q_RB1": 1.0, "Q_RB2": -1.0,
                        "ELLIPS_SHIFT": 0.8, "ELLIPS_LENGTH": 1.0, "ELLIPS_HEIGHT": 1.0, "ELLIPS_ORDER": 1.8, "ELLIPS_MAX_STEP": 0.025, "ELLIPS_NWIDTH": 10.0, "ELLIPS_NLENGTH": 10.0,
                        "PRESMIN": 50000.0, "PRESMAX": 150000.0,
                        "FZMIN": 10.0, "FZMAX": 2000.0,
                        "KPUMIN": -1.5, "KPUMAX": 1.5,
                        "ALPMIN": -1.5, "ALPMAX": 1.5,
                        "CAMMIN": -0.175, "CAMMAX": 0.175,
                        "LFZO": 1.0, "LCX": 1.0, "LMUX": 1.0, "LEX": 1.0, "LKX": 1.0, "LHX": 1.0,
                        "LVX": 1.0, "LCY": 1.0, "LMUY": 1.0, "LEY": 1.0, "LKY": 1.0, "LHY": 1.0, "LVY": 1.0,
                        "LTR": 1.0, "LRES": 1.0, "LXAL": 1.0, "LYKA": 1.0, "LVYKA": 1.0,  "LS": 1.0,
                        "LKYC": 1.0, "LKZC": 1.0,
                        "LVMX": 1.0,
                        "LMX": 1.0, "LMY": 1.0, "LMP": 1.0,
                        "PCX1": 1.65,
                        "PDX1": 1.3, "PDX2": -0.15, "PDX3": 0.0,
                        "PEX1": 0.0, "PEX2": 0.0, "PEX3": 0.0, "PEX4": 0.0,
                        "PKX1": 20.0, "PKX2": 0.0, "PKX3": 0.0,
                        "PHX1": 0.0, "PHX2": 0.0,
                        "PVX1": 0.0, "PVX2": 0.0,
                        "PPX1": 0.0, "PPX2": 0.0, "PPX3": 0.0, "PPX4": 0.0,
                        "RBX1": 20.0, "RBX2": 15, "RBX3": 0.0,
                        "RCX1": 1.0,
                        "REX1": 0.0, "REX2": 0.0,
                        "RHX1": 0.0,
                        "QSX1": 0.0, "QSX2": 0.0, "QSX3": 0.0, "QSX4": 5.0, "QSX5": 1.0, "QSX6": 10, "QSX7": 0.0, "QSX8": 0.0, "QSX9": 0.4, "QSX10": 0.0, "QSX11": 5.0, "QSX12": 0.0, "QSX13": 0.0, "QSX14": 0.0,
                        "PPMX1": 0.0,
                        "PCY1": 1.3,
                        "PDY1": 1.1, "PDY2": -0.15, "PDY3": 0.0,
                        "PEY1": 0.0, "PEY2": 0.0, "PEY3": 0.0, "PEY4": 0.0, "PEY5": 0.0,
                        "PKY1": -20.0, "PKY2": 1.0, "PKY3": 0.0, "PKY4": 2.0, "PKY5": 0.0, "PKY6": -1.0, "PKY7": 0.0,
                        "PHY1": 0.0, "PHY2": 0.0,
                        "PVY1": 0.0, "PVY2": 0.0, "PVY3": 0.0, "PVY4": 0.0,
                        "PPY1": 0.0, "PPY2": 0.0, "PPY3": 0.0, "PPY4": 0.0, "PPY5": 0.0,
                        "RBY1": 10.0, "RBY2": 10.0, "RBY3": 0.0, "RBY4": 0.0,
                        "RCY1": 1.0,
                        "REY1": 0.0, "REY2": 0.0,
                        "RHY1": 0.0, "RHY2": 0.0,
                        "RVY1": 0.0, "RVY2": 0.0, "RVY3": 0.0, "RVY4": 20.0, "RVY5": 2.0, "RVY6": 10.0,
                        "QSY1": 0.01, "QSY2": 0.0, "QSY3": 4.0E-4, "QSY4": 4.0E-5, "QSY5": 0.0, "QSY6": 0.0, "QSY7": 0.85, "QSY8": -0.4,
                        "QBZ1": 10.0, "QBZ2": 0.0, "QBZ3": 0.0, "QBZ4": 0.0, "QBZ5": 0.0, "QBZ9": 10.0, "QBZ10": 0.0,
                        "QCZ1": 1.1,
                        "QDZ1": 0.12, "QDZ2": 0.0, "QDZ3": 0.0, "QDZ4": 0.0, "QDZ6": 0.0, "QDZ7": 0.0, "QDZ8": -0.05, "QDZ9": 0.0, "QDZ10": 0.0, "QDZ11": 0.0,
                        "QEZ1": 0.0, "QEZ2": 0.0, "QEZ3": 0.0, "QEZ4": 0.0, "QEZ5": 0.0,
                        "QHZ1": 0.0, "QHZ2": 0.0, "QHZ3": 0.0, "QHZ4": 0.0,
                        "PPZ1": 0.0, "PPZ2": 0.0,
                        "SSZ1": 0.0, "SSZ2": 0.0, "SSZ3": 0.0, "SSZ4": 0.0,
                        "PDXP1": 0.4, "PDXP2": 0.0, "PDXP3": 0.0,
                        "PKYP1": 1.0,
                        "PDYP1": 0.4, "PDYP2": 0.0, "PDYP3": 0.0, "PDYP4": 0.0,
                        "PHYP1": 1.0, "PHYP2": 0.15, "PHYP3": 0.0, "PHYP4": -4.0,
                        "PECP1": 0.5, "PECP2": 0.0,
                        "QDTP1": 10.0,
                        "QCRP1": 0.2, "QCRP2": 0.1,
                        "QBRP1": 0.1, "QDRP1": 1.0,
                        # Declare Magic Formula 6.2 parameter names for C code generation
                        "FUNCTION_NAME": 'default', "SWITCH_INTEG": 0.0,
                        "Q_FCY2": 0.0,
                        "Q_CAM1": 0.0, "Q_CAM2": 0.0, "Q_CAM3": 0.0,
                        "Q_FYS1": 0.0, "Q_FYS2": 0.0, "Q_FYS3": 0.0,
                        "ENV_C1": 0.0, "ENV_C2": 0.0,
                        # Declare Magic Formula 5.2 parameter names for C code generation
                        "Q_A1": 0.0, "Q_A2": 0.0,
                        "PHY3": 0.0,
                        "PTX1": 0.0, "PTX2": 0.0, "PTX3": 0.0,
                        "PTY1": 0.0, "PTY2": 0.0,
                        "LSGKP": 1.0,
                        "LIGAN": 1.0,
                        "LMUV": 0.0}

@dataclass
class PostProInputs:
    omega: ArrayLike
    phi: ArrayLike
    Vsx: ArrayLike
    uFz: ArrayLike
    ukappa: ArrayLike
    ukappaLow: ArrayLike
    ualpha: ArrayLike
    ugamma: ArrayLike
    uphit: ArrayLike
    uVcx: ArrayLike
    alpha: float = 0
    kappa: float = 0
    gamma: float = 0
    phit: float = 0
    Fz: float = 0
    p: float = 0
    nInputs: float = 0
    Fz_lowLimit: float = 0

@dataclass
class Mode:
    useLimitsCheck: bool
    useAlphaStar: bool
    useTurnSlip: bool
    isLowSpeed: ArrayLike
    isLowSpeedAlpha: ArrayLike
    userDynamics: ArrayLike

@dataclass
class Result:
    Cx: bool = False
    Dx: bool = False
    Ex: bool = False
    Cy: bool = False
    Ey: bool = False
    Bt: bool = False
    Ct: bool = False
    Et: bool = False
    Bxa: bool = False
    Exa: bool = False
    Gxa: bool = False
    Byk: bool = False
    Eyk: bool = False
    Gyk: bool = False
    Nan: bool = False


@dataclass
class RetValue:
    Cx: float = 0.0
    Dx: float = 0.0
    Ex: float = 0.0
    Cy: float = 0.0
    Ey: float = 0.0
    Bt: float = 0.0
    Ct: float = 0.0
    Et: float = 0.0
    Bxa: float = 0.0
    Exa: float = 0.0
    Gxa: float = 0.0
    Byk: float = 0.0
    Eyk: float = 0.0
    Gyk: float = 0.0
    Nan: float = 0.0


@dataclass
class InputRanges:
    dPi: ArrayLike
    dFz: ArrayLike
    Fz:  ArrayLike
    Pi:  ArrayLike
    SR:  ArrayLike
    SA:  ArrayLike
    IA:  ArrayLike

@dataclass
class ForceMoments:
    Fx: float = 0.0
    Fy: float = 0.0
    Fz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0
    dFz: float = 0.0

@dataclass
class VarInf:
    Kxk: float = 0.0
    mux: float = 0.0
    Kya: float = 0.0
    muy: float = 0.0
    t:   float = 0.0
    Mzr: float = 0.0

def set_x(x0, tk: PacejkaModel):
    tk.PCY1 = x0[0]         # Shape factor Cfy for lateral forces
    tk.PDY1 = x0[1]         # Lateral friction Muy
    tk.PDY2 = x0[2]         # Variation of friction Muy with load
    tk.PDY3 = x0[3]  	    # Variation of friction Muy with squared camber
    tk.PEY1 = x0[4]  	    # Lateral curvature Efy at Fznom
    tk.PEY2 = x0[5]   	    # Variation of curvature Efy with load
    tk.PEY3 = x0[6]   	    # Zero order camber dependency of curvature Efy
    tk.PEY4 = x0[7]  	    # Variation of curvature Efy with camber
    tk.PEY5 = x0[8]   	    # Variation of curvature Efy with camber squared
    tk.PKY1 = x0[9]	        # Maximum value of stiffness Kfy/Fznom
    tk.PKY2 = x0[10] 	    # Load at which Kfy reaches maximum value
    tk.PKY3 = x0[11]   	    # Variation of Kfy/Fznom with camber
    tk.PKY4 = x0[12]   	    # Curvature of stiffness Kfy
    tk.PKY5 = x0[13]   	    # Peak stiffness variation with camber squared
    tk.PKY6 = x0[14]   	    # Fy camber stiffness factor
    tk.PKY7 = x0[15]   	    # Vertical load dependency of camber stiffness
    tk.PHY1 = x0[16]  	    # Horizontal shift Shy at Fznom
    tk.PHY2 = x0[17]   	    # Variation of shift Shy with load
    tk.PVY1 = x0[18]  	    # Vertical shift in Svy/Fz at Fznom
    tk.PVY2 = x0[19]   	    # Variation of shift Svy/Fz with load
    tk.PVY3 = x0[20]   	    # Variation of shift Svy/Fz with camber
    tk.PVY4 = x0[21]  	    # Variation of shift Svy/Fz with camber and load
    # influence of inflation pressure on cornering stiffness
    tk.PPY1 = x0[22]
    # influence of inflation pressure on dependency of nominal tyre load on cornering stiffness
    tk.PPY2 = x0[23]
    # linear influence of inflation pressure on lateral peak friction
    tk.PPY3 = x0[24]
    # quadratic influence of inflation pressure on lateral peak friction
    tk.PPY4 = x0[25]
    tk.PPY5 = x0[26]   	    # Influence of inflation pressure on camber stiffness
    tk.QSX1 = x0[27]        # Vertical shift of overturning moment
    tk.QSX2 = x0[28]        # Camber induced overturning couple
    tk.QSX3 = x0[29]        # Fy induced overturning couple
    tk.QSX4 = x0[30]        # Mixed load lateral force and camber on Mx
    tk.QSX5 = x0[31]        # Load effect on Mx with lateral force and camber
    tk.QSX6 = x0[32]        # B-factor of load with Mx
    tk.QSX7 = x0[33]        # Camber with load on Mx
    tk.QSX8 = x0[34]        # Lateral force with load on Mx
    tk.QSX9 = x0[35]        # B-factor of lateral force with load on Mx
    tk.QSX10 = x0[36]       # Vertical force with camber on Mx
    tk.QSX11 = x0[37]       # B-factor of vertical force with camber on Mx
    # Influence of inflation pressure on overturning moment
    tk.PPMX1 = x0[38]
    tk.QBZ1 = x0[39]        # Trail slope factor for trail Bpt at Fznom
    tk.QBZ2 = x0[40]        # Variation of slope Bpt with load
    tk.QBZ3 = x0[41]        # Variation of slope Bpt with load squared
    tk.QBZ4 = x0[42]        # Variation of slope Bpt with camber
    tk.QBZ5 = x0[43]        # Variation of slope Bpt with absolute camber
    # Factor for scaling factors of slope factor Br of Mzr
    tk.QBZ9 = x0[44]
    # Factor for dimensionless cornering stiffness of Br of Mzr
    tk.QBZ10 = x0[45]
    tk.QCZ1 = x0[46]        # Shape factor Cpt for pneumatic trail
    tk.QDZ1 = x0[47]        # Peak trail Dpt = Dpt*(Fz/Fznom*R0)
    tk.QDZ2 = x0[48]        # Variation of peak Dpt" with load
    tk.QDZ3 = x0[49]        # Variation of peak Dpt" with camber
    tk.QDZ4 = x0[50]        # Variation of peak Dpt" with camber squared
    tk.QDZ6 = x0[51]        # Peak residual torque Dmr" = Dmr/(Fz*R0)
    tk.QDZ7 = x0[52]        # Variation of peak factor Dmr" with load
    tk.QDZ8 = x0[53]        # Variation of peak factor Dmr" with camber
    tk.QDZ9 = x0[54]        # Variation of peak factor Dmr" with camber and load
    tk.QDZ10 = x0[55]       # Variation of peak factor Dmr with camber squared
    tk.QDZ11 = x0[56]       # Variation of Dmr with camber squared and load
    tk.QEZ1 = x0[57]        # Trail curvature Ept at Fznom
    tk.QEZ2 = x0[58]        # Variation of curvature Ept with load
    tk.QEZ3 = x0[59]        # Variation of curvature Ept with load squared
    tk.QEZ4 = x0[60]        # Variation of curvature Ept with sign of Alpha-t
    tk.QEZ5 = x0[61]        # Variation of Ept with camber and sign Alpha-t
    tk.QHZ1 = x0[62]        # Trail horizontal shift Sht at Fznom
    tk.QHZ2 = x0[63]        # Variation of shift Sht with load
    tk.QHZ3 = x0[64]        # Variation of shift Sht with camber
    tk.QHZ4 = x0[65]        # Variation of shift Sht with camber and load
    # effect of inflation pressure on length of pneumatic trail
    tk.PPZ1 = x0[66]
    # Influence of inflation pressure on residual aligning torque
    tk.PPZ2 = x0[67]
    tk.SSZ1 = x0[68]        # Nominal value of s/R0: effect of Fx on Mz
    tk.SSZ2 = x0[69]        # Variation of distance s/R0 with Fy/Fznom
    tk.SSZ3 = x0[70]        # Variation of distance s/R0 with camber
    tk.SSZ4 = x0[71]        # Variation of distance s/R0 with load and camber
    tk.RBY1 = x0[72]        # Slope factor for combined Fy reduction
    tk.RBY2 = x0[73]        # Variation of slope Fy reduction with alpha
    tk.RBY3 = x0[74]        # Shift term for alpha in slope Fy reduction
    tk.RBY4 = x0[75]        # Influence of camber on stiffness of Fy combined
    tk.RCY1 = x0[76]        # Shape factor for combined Fy reduction
    tk.REY1 = x0[77]        # Curvature factor of combined Fy
    tk.REY2 = x0[78]        # Curvature factor of combined Fy with load
    tk.RHY1 = x0[79]        # Shift factor for combined Fy reduction
    tk.RHY2 = x0[80]        # Shift factor for combined Fy reduction with load
    tk.RVY1 = x0[81]        # Kappa induced side force Svyk/Muy*Fz at Fznom
    tk.RVY2 = x0[82]        # Variation of Svyk/Muy*Fz with load
    tk.RVY3 = x0[83]        # Variation of Svyk/Muy*Fz with camber
    tk.RVY4 = x0[84]        # Variation of Svyk/Muy*Fz with alpha
    tk.RVY5 = x0[85]        # Variation of Svyk/Muy*Fz with kappa
    tk.RVY6 = x0[86]        # Variation of Svyk/Muy*Fz with atan(kappa)
    tk.LFZO = x0[87]        # Scale factor of nominal (rated) load
    tk.LCX = x0[88]         # Scale factor of Fx shape factor
    tk.LMUX = x0[89]        # Scale factor of Fx peak friction coefficient
    tk.LEX = x0[90]         # Scale factor of Fx curvature factor
    tk.LKX = x0[91]         # Scale factor of Fx slip stiffness
    tk.LHX = x0[92]         # Scale factor of Fx horizontal shift
    tk.LVX = x0[93]         # Scale factor of Fx vertical shift
    tk.LCY = x0[94]         # Scale factor of Fy shape factor
    tk.LMUY = x0[95]        # Scale factor of Fy peak friction coefficient
    tk.LEY = x0[96]         # Scale factor of Fy curvature factor
    tk.LKY = x0[97]         # Scale factor of Fy cornering stiffness
    tk.LHY = x0[98]         # Scale factor of Fy horizontal shift
    tk.LVY = x0[99]         # Scale factor of Fy vertical shift
    tk.LTR = x0[100]        # Scale factor of peak of pneumatic trail
    tk.LRES = x0[101]       # Scale factor for offset of residual torque
    tk.LXAL = x0[102]       # Scale factor of alpha influence on Fx
    tk.LYKA = x0[103]       # Scale factor of kappa influence on Fy
    tk.LVYKA = x0[104]      # Scale factor of kappa induced Fy
    tk.LS = x0[105]         # Scale factor of moment arm of Fx
    tk.LKYC = x0[106]       # Scale factor of camber force stiffness
    tk.LKZC = x0[107]       # Scale factor of camber torque stiffness
    tk.LVMX = x0[108]       # Scale factor of Mx vertical shift
    tk.LMX = x0[109]        # Scale factor of overturning couple
    tk.LMY = x0[110]        # Scale factor of rolling resistance torque
    tk.LMP = x0[111]        # Scale factor of Parking Moment
    tk.Q_RA1 = x0[112]      # Square root term in contact length equation
    tk.Q_RA2 = x0[113]      # Linear term in contact length equation
    tk.Q_RB1 = x0[114]      # Root term in contact width equation
    tk.Q_RB2 = x0[115]      # Linear term in contact width equation
    # Load and speed influence on in-plane translation stiffness
    tk.Q_BVX = x0[116]
    # Load and speed influence on in-plane rotation stiffness
    tk.Q_BVT = x0[117]
    # Tyre overall longitudinal stiffness vertical deflection dependency linear term
    tk.PCFX1 = x0[118]
    # Tyre overall longitudinal stiffness vertical deflection dependency quadratic term
    tk.PCFX2 = x0[119]
    # Tyre overall longitudinal stiffness pressure dependency
    tk.PCFX3 = x0[120]
    # Tyre overall lateral stiffness vertical deflection dependency linear term
    tk.PCFY1 = x0[121]
    # Tyre overall lateral stiffness vertical deflection dependency quadratic term
    tk.PCFY2 = x0[122]
    # Tyre overall lateral stiffness pressure dependency
    tk.PCFY3 = x0[123]
    tk.PCMZ1 = x0[124]     # Tyre overall yaw stiffness pressure dependency
    tk.PCX1 = x0[125]      # Shape factor Cfx for longitudinal force
    tk.PDX1 = x0[126]      # Longitudinal friction Mux at Fznom
    tk.PDX2 = x0[127]      # Variation of friction Mux with load
    tk.PDX3 = x0[128]      # Variation of friction Mux with camber squared
    tk.PEX1 = x0[129]      # Longitudinal curvature Efx at Fznom
    tk.PEX2 = x0[130]      # Variation of curvature Efx with load
    tk.PEX3 = x0[131]      # Variation of curvature Efx with load squared
    tk.PEX4 = x0[132]      # Factor in curvature Efx while driving
    tk.PKX1 = x0[133]      # Longitudinal slip stiffness Kfx/Fz at Fznom
    tk.PKX2 = x0[134]      # Variation of slip stiffness Kfx/Fz with load
    tk.PKX3 = x0[135]      # Exponent in slip stiffness Kfx/Fz with load
    tk.PHX1 = x0[136]      # Horizontal shift Shx at Fznom
    tk.PHX2 = x0[137]      # Variation of shift Shx with load
    tk.PVX1 = x0[138]      # Vertical shift Svx/Fz at Fznom
    tk.PVX2 = x0[139]      # Variation of shift Svx/Fz with load
    # linear influence of inflation pressure on longitudinal slip stiffness
    tk.PPX1 = x0[140]
    # quadratic influence of inflation pressure on longitudinal slip stiffness
    tk.PPX2 = x0[141]
    # linear influence of inflation pressure on peak longitudinal friction
    tk.PPX3 = x0[142]
    # quadratic influence of inflation pressure on peak longitudinal friction
    tk.PPX4 = x0[143]
    tk.RBX1 = x0[144]      # Slope factor for combined slip Fx reduction
    tk.RBX2 = x0[145]      # Variation of slope Fx reduction with kappa
    tk.RBX3 = x0[146]      # Influence of camber on stiffness for Fx combined
    tk.RCX1 = x0[147]      # Shape factor for combined slip Fx reduction
    tk.REX1 = x0[148]      # Curvature factor of combined Fx
    tk.REX2 = x0[149]      # Curvature factor of combined Fx with load
    tk.RHX1 = x0[150]      # Shift factor for combined slip Fx reduction
    tk.LMUV = x0[151]      # Scale factor with slip speed Vs decaying friction
    tk.QSX12 = x0[152]    # Camber squared induced overturning moment
    tk.QSX13 = x0[153]    # Lateral force induced overturning moment
    tk.QSX14 = x0[154]    # Lateral force induced overturning moment with camber
    tk.QSY1  = x0[155]    # Rolling resistance torque coefficien
    tk.QSY2  = x0[156]    # Rolling resistance torque depending on Fx
    tk.QSY3  = x0[157]    # Rolling resistance torque depending on speed
    tk.QSY4  = x0[158]    # Rolling resistance torque depending on speed ^4
    tk.QSY5  = x0[159]    # Rolling resistance torque depending on camber squared
    tk.QSY6  = x0[160]    # Rolling resistance torque depending on load and camber squared
    tk.QSY7  = x0[161]    # Rolling resistance torque coefficient load dependency
    tk.QSY8  = x0[162]    # Rolling resistance torque coefficient pressure dependency
    tk.PDXP1 = x0[163]    # Peak Fx reduction due to spin parameter
    tk.PDXP2 = x0[164]    # Peak Fx reduction due to spin with varying load parameter
    tk.PDXP3 = x0[165]    # Peak Fx reduction due to spin with kappa parameter
    tk.PKYP1 = x0[166]    # Cornering stiffness reduction due to spin
    tk.PDYP1 = x0[167]    # Peak Fy reduction due to spin parameter
    tk.PDYP2 = x0[168]    # Peak Fy reduction due to spin with varying load parameter
    tk.PDYP3 = x0[169]    # Peak Fy reduction due to spin with alpha parameter
    tk.PDYP4 = x0[170]    # Peak Fy reduction due to square root of spin parameter
    tk.PHYP1 = x0[171]    # Fy-alpha curve lateral shift limitation
    tk.PHYP2 = x0[172]    # Fy-alpha curve maximum lateral shift parameter
    tk.PHYP3 = x0[173]    # Fy-alpha curve maximum lateral shift varying with load parameter
    tk.PHYP4 = x0[174]    # Fy-alpha curve maximum lateral shift parameter
    tk.PECP1 = x0[175]    # Camber w.r.t. spin reduction factor parameter in camber stiffness
    tk.PECP2 = x0[176]    # Camber w.r.t. spin reduction factor varying with load parameter in camber stiffness
    tk.QDTP1 = x0[177]    # Pneumatic trail reduction factor due to turn slip parameter
    tk.QCRP1 = x0[178]    # Turning moment at constant turning and zero forward speed parameter
    tk.QCRP2 = x0[179]    # Turn slip moment (at alpha = 90deg) parameter for increase with spin
    tk.QBRP1 = x0[180]    # Residual (spin) torque reduction factor parameter due to side slip
    tk.QDRP1 = x0[181]    # Turn slip moment peak magnitude parameter
    tk.BREFF = x0[182]    # Low load stiffness effective rolling radius
    tk.DREFF = x0[183]    # Peak value of effective rolling radius
    tk.FREFF = x0[184]    # High load stiffness effective rolling radius
    tk.Q_RE0 = x0[185]    # Ratio of free tyre radius with nominal tyre radius
    tk.Q_V1  = x0[186]    # Tyre radius increase with speed
    # Set nominal parameters of the model (DO NOT CHANGE AFTER)
    tk.UNLOADED_RADIUS = x0[187] # Unloaded tire radius
    tk.FNOMIN = x0[188]   # Nominal load THIS MUST BE SET TO AT LEAST 50% OF THE MAXIMUM LOAD OR YOU WILL GET FUCKING CRAZY HARD TO DIAGNOSE ERRORS
    tk.LONGVL = x0[189]   # Nominal reference speed
    tk.NOMPRES = x0[190]  # Nominal inflation pressure Pa
    tk.FZMIN = x0[191]    # Minimum load for which the model is valid

def dump_tk(tk: PacejkaModel):
    x0: List = [0] * 192
    x0[0] = tk.PCY1  
    x0[1] = tk.PDY1  
    x0[2] = tk.PDY2  
    x0[3] = tk.PDY3  
    x0[4] = tk.PEY1  
    x0[5] = tk.PEY2  
    x0[6] = tk.PEY3  
    x0[7] = tk.PEY4  
    x0[8] = tk.PEY5  
    x0[9] = tk.PKY1  
    x0[10] = tk.PKY2 
    x0[11] = tk.PKY3
    x0[12] = tk.PKY4
    x0[13] = tk.PKY5
    x0[14] = tk.PKY6
    x0[15] = tk.PKY7
    x0[16] = tk.PHY1 
    x0[17] = tk.PHY2
    x0[18] = tk.PVY1 
    x0[19] = tk.PVY2
    x0[20] = tk.PVY3
    x0[21] = tk.PVY4 
    x0[22] = tk.PPY1
    x0[23] = tk.PPY2
    x0[24] = tk.PPY3
    x0[25] = tk.PPY4
    x0[26] = tk.PPY5
    x0[27] = tk.QSX1
    x0[28] = tk.QSX2
    x0[29] = tk.QSX3
    x0[30] = tk.QSX4
    x0[31] = tk.QSX5
    x0[32] = tk.QSX6
    x0[33] = tk.QSX7
    x0[34] = tk.QSX8
    x0[35] = tk.QSX9
    x0[36] = tk.QSX10
    x0[37] = tk.QSX11
    x0[38] = tk.PPMX1
    x0[39] = tk.QBZ1
    x0[40] = tk.QBZ2
    x0[41] = tk.QBZ3
    x0[42] = tk.QBZ4
    x0[43] = tk.QBZ5
    x0[44] = tk.QBZ9
    x0[45] = tk.QBZ10
    x0[46] = tk.QCZ1
    x0[47] = tk.QDZ1
    x0[48] = tk.QDZ2
    x0[49] = tk.QDZ3
    x0[50] = tk.QDZ4
    x0[51] = tk.QDZ6
    x0[52] = tk.QDZ7
    x0[53] = tk.QDZ8
    x0[54] = tk.QDZ9
    x0[55] = tk.QDZ10
    x0[56] = tk.QDZ11
    x0[57] = tk.QEZ1
    x0[58] = tk.QEZ2
    x0[59] = tk.QEZ3
    x0[60] = tk.QEZ4
    x0[61] = tk.QEZ5
    x0[62] = tk.QHZ1
    x0[63] = tk.QHZ2
    x0[64] = tk.QHZ3
    x0[65] = tk.QHZ4
    x0[66] = tk.PPZ1
    x0[67] = tk.PPZ2
    x0[68] = tk.SSZ1
    x0[69] = tk.SSZ2
    x0[70] = tk.SSZ3
    x0[71] = tk.SSZ4
    x0[72] = tk.RBY1
    x0[73] = tk.RBY2
    x0[74] = tk.RBY3
    x0[75] = tk.RBY4
    x0[76] = tk.RCY1
    x0[77] = tk.REY1
    x0[78] = tk.REY2
    x0[79] = tk.RHY1
    x0[80] = tk.RHY2
    x0[81] = tk.RVY1
    x0[82] = tk.RVY2
    x0[83] = tk.RVY3
    x0[84] = tk.RVY4
    x0[85] = tk.RVY5
    x0[86] = tk.RVY6
    x0[87] = tk.LFZO
    x0[88] = tk.LCX
    x0[89] = tk.LMUX
    x0[90] = tk.LEX
    x0[91] = tk.LKX
    x0[92] = tk.LHX
    x0[93] = tk.LVX
    x0[94] = tk.LCY
    x0[95] = tk.LMUY
    x0[96] = tk.LEY
    x0[97] = tk.LKY
    x0[98] = tk.LHY
    x0[99] = tk.LVY
    x0[100] = tk.LTR
    x0[101] = tk.LRES
    x0[102] = tk.LXAL
    x0[103] = tk.LYKA
    x0[104] = tk.LVYKA
    x0[105] = tk.LS
    x0[106] = tk.LKYC
    x0[107] = tk.LKZC
    x0[108] = tk.LVMX
    x0[109] = tk.LMX
    x0[110] = tk.LMY
    x0[111] = tk.LMP
    x0[112] = tk.Q_RA1
    x0[113] = tk.Q_RA2
    x0[114] = tk.Q_RB1
    x0[115] = tk.Q_RB2
    x0[116] = tk.Q_BVX
    x0[117] = tk.Q_BVT
    x0[118] = tk.PCFX1
    x0[119] = tk.PCFX2
    x0[120] = tk.PCFX3
    x0[121] = tk.PCFY1
    x0[122] = tk.PCFY2
    x0[123] = tk.PCFY3
    x0[124] = tk.PCMZ1
    x0[125] = tk.PCX1
    x0[126] = tk.PDX1
    x0[127] = tk.PDX2
    x0[128] = tk.PDX3
    x0[129] = tk.PEX1
    x0[130] = tk.PEX2
    x0[131] = tk.PEX3
    x0[132] = tk.PEX4
    x0[133] = tk.PKX1
    x0[134] = tk.PKX2
    x0[135] = tk.PKX3
    x0[136] = tk.PHX1
    x0[137] = tk.PHX2
    x0[138] = tk.PVX1
    x0[139] = tk.PVX2
    x0[140] = tk.PPX1
    x0[141] = tk.PPX2
    x0[142] = tk.PPX3
    x0[143] = tk.PPX4
    x0[144] = tk.RBX1
    x0[145] = tk.RBX2
    x0[146] = tk.RBX3
    x0[147] = tk.RCX1
    x0[148] = tk.REX1
    x0[149] = tk.REX2
    x0[150] = tk.RHX1
    x0[151] = tk.LMUV
    x0[152] = tk.QSX12
    x0[153] = tk.QSX13
    x0[154] = tk.QSX14
    x0[155] = tk.QSY1 
    x0[156] = tk.QSY2 
    x0[157] = tk.QSY3 
    x0[158] = tk.QSY4 
    x0[159] = tk.QSY5 
    x0[160] = tk.QSY6 
    x0[161] = tk.QSY7 
    x0[162] = tk.QSY8 
    x0[163] = tk.PDXP1
    x0[164] = tk.PDXP2
    x0[165] = tk.PDXP3
    x0[166] = tk.PKYP1
    x0[167] = tk.PDYP1
    x0[168] = tk.PDYP2
    x0[169] = tk.PDYP3
    x0[170] = tk.PDYP4
    x0[171] = tk.PHYP1
    x0[172] = tk.PHYP2
    x0[173] = tk.PHYP3
    x0[174] = tk.PHYP4
    x0[175] = tk.PECP1
    x0[176] = tk.PECP2
    x0[177] = tk.QDTP1
    x0[178] = tk.QCRP1
    x0[179] = tk.QCRP2
    x0[180] = tk.QBRP1
    x0[181] = tk.QDRP1
    x0[182] = tk.BREFF
    x0[183] = tk.DREFF
    x0[184] = tk.FREFF
    x0[185] = tk.Q_RE0
    x0[186] = tk.Q_V1 
    x0[187] = tk.UNLOADED_RADIUS
    x0[188] = tk.FNOMIN
    x0[189] = tk.LONGVL
    x0[190] = tk.NOMPRES
    x0[191] = tk.FZMIN

    return np.array(x0)

def tire_model_from_arr(arr):
    tm = PacejkaModel(MODEL_DEFAULTS)
    set_x(arr, tm)
    return tm


EXAMP_ARR = [ 0.871635698042293, 3.835108280765198, 0.24286264125683876, 5.555324956431501, -0.16104158564808702, 1.684588594488567, -0.01983224771440499, -2.845372029447575, -0.29199969871955256, -45.028443943723325, 1.4607180866565455, 0.5896204513281181, 2.0655688891230795, 1.888278538275951, 1.3738047681642824, 0.5665685000065671, -0.0015568114405175445, 4.657950576518741e-05, -0.015340406484339094, -0.012325682167770025, -0.0024713368467329023, -0.3190886423450116, 0.666612264880553, 1.480043218337634, -0.1575914497466752, 0.03665491116227539, 0.3914206481685555, -0.02068520923648993, 9.99999992174444e-07, 0.061838604519491, 4.9958514999423755, 0.4115428284053738, 10.018751444377227, 1.0000000066329324e-06, 0.02447393260885965, 0.007583839456160414, 9.999999913227617e-07, 5.000000000000012, 1.0000000107650052e-06, 13.28131931646991, -0.4676944518764176, -0.5482633065341462, 3.477320840550834, -4.355283225300479, 6.826658902752458, 0.591059217072533, 5.765275810084571, -0.0764526980296856, -0.011187764218081327, 1.4360019774304003, 1.226937105672486, -0.02077367773152089, 0.03891995515008825, -1.0187367790441153, 1.3060151728765004, -0.5057077596473424, 0.11843516374676959, -1.8601200568404233, -1.1695844606375658, 0.8590505510117654, 1.0444725539599795, 0.7275689261998006, -0.3782122974657393, -0.06266641051206889, 0.055288390882440266, -0.4848091429731051, 1.8078020690756749, -0.5727142440782835, -0.052476179673127295, 0.07738831386886791, 9.999999772038116e-07, 1.0000000169786917e-06, 5.793651418497266, 2.164879811903075, 0.002063439957476809, -2.466316119343624e-14, 1.3382503325558195, 1.0057049967484786, -0.04220199710672606, 0.010942165832036464, 0.01303117832706997, -0.020176691233813868, -0.024655058125729844, -2.342286155714066e-14, 19.846895530265133, 1.7488732844842443, 9.884605353121495, 1.1870967637811571, 1.0072270235639238, 0.9485382203590609, 1.020946454023419, 0.5967023090064599, 1.0065571840344267, 0.7971766934294593, 0.3575972164828737, 1.8367706454640034, 1.0966330163016278, 0.7368908112886183, 1.0005117909149088, 0.9751746903873566, 0.9848196492543491, 0.9785819096454574, 2.050204220914901, 3.456094485078719, 0.9898653922289152, 1.0008207657819408, 2.9389088511264165, 1.1134163885054646, 0.9994283485170865, 0.9790187816412459, 1.0, 1.0000000000000147, 0.79, 0.35, 1.0, -1.0, 1e-06, 1e-06, 1e-06, 1e-06, 0.2, 1e-06, 1e-06, 0.5, 1e-06, 1.4927586610614192, 2.6339567034031863, -0.2059217515029416, 7.283240044732358, -0.15367448634010855, 0.6386373728676568, -0.2946878788735154, -2.8650880689438556, 64.12159136380396, -69.34503809845852, 1.2295299404266653, -0.0011391133561497902, -0.004251354496879074, 0.016323928518864494, 0.23326601061217062, -0.8175831510739994, -0.6850173623722056, -0.22748006559774014, 0.2657538636762964, 10.142353560936954, 11.579095632219536, 9.99999976181964e-07, 1.60006953853343, 1.0015770848041978, 1.9150458259309512e-05, -1.0796981659463147, 0.5063041498465363, 1.0000000010642583e-06, -0.009011161373194994, 1.0000000029785314e-06, 0.01, 1e-06, 0.0004, 4e-05, 1e-06, 1e-06, 0.85, -0.4, 0.4, 1e-06, 1e-06, 0.9973304614775792, 0.37703390493818273, -0.273277123939716, -4.496032907767676e-14, 0.07047190364778738, 1.2983989813566663, 1.911912346138584, 1.1985467923100641, -3.9496571166930856, 0.5, 1e-06, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.23, 1500, 11.1, 83000, 0.0]
H_R20_18X6_6 = [1.9509179762515743, 2.4502239583034426, -0.4151634587715227, 8.04240177371771, 4.112010295869466, 0.8947979239802039, -0.3056655000018777, -1.7420809666870634, 1.6434362979514936, -33.59883929325499, -0.3076069066632764, 1.526516044391027, -0.6628986566197681, 18.224439382729404, -2.851711324082719, 1.1133792162066587, 0.0, 0.0, 0.0, 0.0, 0.07716393633027603, 0.3269490901622526, 0.6579667964930671, 2.232268420592817, -0.18718717421046108, 0.24615463900202358, -0.6013232357232908, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 6.535894936344297, -0.6907610596751407, -0.8899505050310963, 74.19642417830539, -74.98375995475288, 10.276693788198926, 0.3306115246223776, 1.7622365593151101, 0.2144712396502534, -0.022368427279735124, -0.5038572082812348, -8.94849912983722, 0.011577684729656684, -0.03064897579961491, -0.38616175475254333, 1.0053622086213974, -24.26877678070708, 46.11892983239403, 0.5937335250057824, 0.24534560594197266, -1.4898048892540376, 0.07218086337801458, 0.42830032234281945, 0.007123761679890366, -0.0012441371625750797, -0.025886968311472733, -0.11830628492503871, 1.069522054245014, 1.998811168965796, -0.04393155295896065, -0.08168970135713388, 0.0, 0.0, 22.78465825521183, 19.28826482672524, 0.0, -0.7645504329757425, 0.9365277675363708, 0.0, 0.0, 0.02473795820094404, 0.034169142707505915, 0.07627716095237477, 0.38821720220764616, 0.6198287114792986, -0.08926979641366861, 0.26941450837202296, 10.644372894036133, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 2.2613497076503593, 2.3858143898797093, -0.247060879312696, 13.82037010282656, 1.14736960364782, 0.7790940274354764, 2.944037154455736, 0.06509691263093421, 42.07748649059425, -95.4360770843624, 1.694423114322486, 0.0026154892201700917, 0.010918592347475916, -0.07501275847296185, -0.21391333451242064, -0.9681110279995524, -2.0853442999350835, -0.23055922639048515, -0.3879925099100577, 10.973344424198102, 6.609414664294439, 26.574896206955977, 1.2306997510641935, 3.215471762304219, 0.8409945908741758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1100.0, 11.1, 83000.0, 100.0]
H_R20_18X6_7 = [1.3947134496015567, 2.5206144417506327, -0.25116696610448214, 11.36975246032364, 0.3428506796384224, -0.166153372598482, -0.06891681754678018, -24.93966133736291, -124.13902924113646, -39.689297158110705, 0.5302062495693125, 1.5716113687384623, 0.7978435638901301, -2.6212106831978432, -3.4604017638643167, -0.04419131198116542, 0.0, 0.0, 0.0, 0.0, -1.2728402994344594, 0.20181901631412993, 0.6343419810937693, 1.837556293518635, -0.22454462622398844, -0.006498743725166365, -1.0008335527715386, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 4.130441073037772, -0.8919875935327826, -0.06878994205835531, 8.450896457921544, -9.136260976989217, 9.597734203240309, 0.5928212038114667, 2.340266933609776, 0.17210657554893008, -0.0675918090013524, -1.1744402363532234, -11.717591710211638, 0.014387794966942433, -0.03509349595997413, -0.5633346310435156, 1.450560833091806, -18.76463678930552, 23.60631548304449, 1.352358366339977, 1.5858565523835253, 0.14015929858147683, -0.3444668783352184, 6.0100112109794965, 0.01200383772132669, 0.0026552838119828274, -0.23687911569939196, -0.33050856855932964, 0.9468071503994389, 2.321891297227055, -0.031082447444536934, -0.06411704799035721, 0.0, 0.0, 19.519206137596615, 17.901229601718075, 0.0, -88.54645765878779, 0.9530866893419222, 0.0, 0.0, 0.010800262826255767, 0.016859646800574994, 0.007080132738618617, 0.04201474058455998, -0.09633053219550698, -0.10498920599556547, 1.5854393481441522, 15.921199132485212, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 1.1840293694422783, 2.217076972323406, -0.22303786112540455, 8.24889458447967, -1.4339662165815024, 7.510731708426267, 18.498681508376546, -0.1062177597160216, 39.65198527797864, 7.756893148714715, -0.4133078837291841, -0.00011483137956625985, 0.001536945001382888, -0.020206242727348132, 0.058172064800581014, -1.0950148178848267, -2.590898317065856, -0.2144701742571981, -0.7062456019554589, 18.565426245087846, 14.516978048508129, 16.072848139050237, 0.9790829929708201, 0.3581333166993733, 1.6167823834516653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1100.0, 11.1, 83000.0, 100.0]
H_R25B_18X6_6 = [1.8213902618465099, 2.465232109377777, -0.3406045868534304, 10.69927988863678, 0.3935241931379088, -0.046381188206278776, -0.166107332074275, -3.8811127668914667, -97.5187016205349, -49.203387482525784, 2.6407283716259893, 1.2143880174349198, 2.764640131564042, 34.73094677903762, -3.8096054583624546, 0.9234243785110479, 0.0, 0.0, 0.0, 0.0, -1.1757446728284568, 0.28377898401228907, 0.5252062159349066, 0.786808293203161, -0.10197038365068002, 0.14771427906582482, 0.23448136177966705, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 6.702571693210816, 1.9796665811299807, -1.9642468583176902, -1.5125501357163547, 0.12388039235115092, 17.186416658362567, 4.1063585254525075, 2.0136110918516703, 0.12443204015413771, 0.004027353066696302, -0.5938230380819606, -11.338764128863628, -0.04090191897673515, 0.01417111085353673, -0.178954210957971, -0.0005758915097715759, -17.583832161544045, 4.943193808378885, 0.6786693073132396, 1.9170216172549357, -1.3315843451591673, 0.01594865176020244, -0.9147627595586224, 0.009078410515286961, -0.004755465744939925, 0.0212365908966577, -0.14531765603034155, 1.6206938465347913, 4.633246762174211, -0.005236442875209442, -0.04046712755001262, 0.0, 0.0, 17.678905804664915, 15.100398164071922, 0.0, -0.3266012494437915, 0.9708316511194602, 0.0, 0.0, 0.006544626437098933, 0.021148866781246264, 0.03950930801905979, 0.012646235137157888, 0.310359275124734, 0.04295752891314422, 1.8395056197225343, 10.885334834036202, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 1.750028567194664, 2.4205433380607206, -0.3689146875021161, 12.810582885573318, 2.1972472494573743, 0.7752707467066294, 2.111937882086009, 5.00016407788607e-12, 64.52055436896863, 0.014721077938553151, -0.28322553313217735, 0.0010407504784858848, -0.0014254040217803088, -0.020998371795279858, 0.020595216093420805, -0.792154203648786, -0.9837443755005021, -0.13921802985941684, 0.12002074020255368, 17.73410622077721, 21.455196481305627, -4.2159900846608185, 0.9672069297328788, -0.8577021732976933, 1.625729122196034, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 660.0, 11.1, 83000.0, 100.0]
H_R25B_18X6_7 = [1.7210094064577293, 2.9167843996435314, -0.6461955057992751, 11.514215410525104, -0.43265515535864957, -2.2828874633222727, -3.8101779473583604e-08, 3.181021228924469, 48.07227967706373, -44.992721897688874, -0.6308213127484736, 0.6470406013605645, -1.1593709994341697, -23.344627694792994, 6.357109629681489, 2.7545411755964357, 2.1262955524329874e-09, 5.181545915189982e-09, 5.103302413881819e-08, 1.369332761901892e-07, -0.09236540150214774, 2.1358382393495967, 0.7759939731798288, 1.8620225484276516, -0.17869931249329046, 0.12704202453763938, -0.7189127320409312, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 9.896790791355299, -0.4754574497296513, -0.048095282988550925, 8.534014006042037e-08, -0.6039926128824469, 10.078536260120563, -1.4208181658208745, 1.6185666070252398, 0.1971308425687492, -0.09938406028626266, 6.875877799692194e-08, -28.449063380458487, -6.520707656397982e-09, -1.4276205570608115e-08, 0.04921850771937449, 0.030242254942403, 4.944445317293855, -5.041597238422382, 0.5064943204435174, 0.23093723920915263, 0.17749868842213135, 1.8378136759838626e-10, -0.8734669539154954, -6.201663200367101e-10, -1.237367884247533e-09, -0.0030397998177777207, 0.9279418691935114, 1.5071633451954534, 5.9360720329613486, -9.208762001882641e-12, -0.03198728835971939, 0.0, 0.0, 5.481654033091774, 11.119795771381597, 3.1178999419552776e-10, -130.97848816355835, 1.091447614144719, 0.3256297137224199, -0.39761343617272804, 0.02055215522088823, 0.005943803685251592, -6.988526755148664e-10, -1.4104270917843102e-09, -0.5888680013784133, 19.999999999997282, 2.0000000022742226, 10.000000000360597, 1.0, 1.2022318595341135, 0.8845074916103176, 3.43086644111828, 0.40180344471642954, 1.002952414520195, 0.9917163380239865, 1.0, 0.7420338500884806, 1.2720361238893152, 0.6524617697068835, 1.000000342025778, 0.999999988168772, 1.0138638819449295, 1.0000001019123714, 1.7651699505296397, 2.986794339332601, 0.9999999997556139, 0.9649838810398967, 0.9325167127638804, 4.588198829000288, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 1.6287396010628716, 2.2911548288044927, -1.1688399895696147, 14.240601483744795, 1.1670794127104156, 2.7875170867479326, 2.0851419303748275, 0.23023177151062976, 117.29315556368897, 2.5667806997134215, -0.737010629777806, -6.548496238429896e-05, 0.0012471751890326007, -0.04109167719632327, -0.06417612725896749, -0.5256460214457709, -0.2384388497861595, -0.21757910549678502, 0.07571359085750909, 10.067857378123968, 16.950589803381455, -122.53765232447101, 1.0055502263070546, 0.8131962398724234, 2.6362129242077006, 2.8953770293483157e-11, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1500.0, 11.1, 83000.0, 100.0]
H_LC0_18X6_6 = [1.9158161188515426, 2.20934099894627, -0.5599683563541571, 9.847974886332725, 4.0997999099674045, 3.17131860704774, 1.4625984430133364e-14, 8.410161163651566e-15, 6.707442749139288e-14, -31.353017040548657, 0.19604815803173528, 1.2082478123130338, 0.428975032637431, 1.5213218706912182, -7.927691191933474, -6.355146945539183, 1.1312252849729834e-10, 2.404574485579085e-10, 1.4350087148473198e-09, 3.2367575426692817e-09, 3.423678669222367, 5.861932016345632, 1.0122695018581627, 2.362207092238878, -0.27941943179651, -0.46316610141320147, 0.4826251288503195, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 5.392689209616505, 2.624968304744954, 2.999785717508106, 1.0594310839042258e-07, 0.6675109800866962, 9.401620518831015, 6.856933122429071, 2.2275639130863802, 0.36044507100858697, 0.09839590519078996, 8.837709427567581e-08, -18.4539358269438, -5.340019721323479e-10, -1.2889609162426537e-09, 0.2671169865960418, 0.5027390884752557, -19.224258873841077, 10.114759677199114, 4.449337247414084, -2.8721492267350843, -2.9354867604567794, 3.798582507320196e-15, 9.49175698125536e-17, -1.4545289006022635e-10, -4.764322470844229e-10, -0.12077723442278039, -0.9570023140264676, 1.280981570589805, -5.980450285136019, 2.1546152275494187e-12, -0.051927204373091514, 0.0, 0.0, 8.604244852812943, 6.739200284989253, -6.982876237326898e-11, -48.62033946272015, 1.5275971137602844, 0.9960895434956092, -0.2512321475848452, 0.05376156189387161, 0.06022491825250874, -6.925534491686504e-11, -1.479186513454648e-10, -0.5751298088446415, 19.99999914028117, 10.37149302338414, 11.543814753900998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 1.5901496421214352, 1.8717745219518749, -1.178179755826228, 9.298429676305334, 2.4356096386224095, 12.163521666216567, 17.590455772027568, 0.007889947724486736, 31.19107793881317, -12.276139254513174, -0.797769501127017, 0.0031262654592725376, 0.0019610256122292178, -0.09351020815117841, 0.01699346019429906, -0.6151228765850991, -1.0895128660977107, -0.34984034647559575, -0.48675848205568684, 25.566558055255733, 17.265317548939752, -12.702963110357793, 0.8567477327826373, 1.3663270333673478, 3.4203799080176958, 1.1577506367655538e-10, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1500.0, 11.1, 83000.0, 100.0]
H_LC0_18X6_7 = [1.8006888851165435, 2.3340154936384043, -0.3200924216335851, 9.001373910440718, 3.510145678818565, 0.7551559124242612, 2.819486034944109e-13, 4.71621616617354e-13, -1.5297376922378588e-13, -36.70108897414165, -0.36962542112270796, 1.3065046202403054, -0.5503428965866294, -2.0021503736447532, -7.662909416997492, -4.780929992999271, -1.7286597645547625e-10, -4.131775220521767e-10, -2.3150995231411865e-09, -5.931098060953323e-09, 2.7604841293551274, 2.8853767243062713, 1.1557353451378114, 2.3463840325804406, -0.28353193271242955, -0.3880009063326802, 0.08591722047416557, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 4.331372437410898, -0.2354584110989884, 1.0144908166899242, -4.430330671378252e-08, 1.3168832267494976, 10.79450605607159, -10.112590077620233, 2.296758807725966, 0.24062044761028156, -0.06037768424215467, -2.7591719941455005e-08, -5.423062816004181, 7.680456591214601e-10, 1.7067863051364174e-09, 0.35325868259698434, 0.873127870152738, -27.548328854245938, 10.22057449649166, 5.252033416369881, 1.024535531489472, -2.125420335849696, -1.3845631933134587e-15, 1.0950178219252637e-15, 2.237210085447585e-10, 6.996288616410654e-10, 0.0639566377060436, -1.0592155193862673, 1.2776560830800492, -5.980961923349163, -5.620810273971284e-12, -0.05326685080921275, 0.0, 0.0, 8.097605444759223, 9.875791401370346, -5.013617044539917e-11, -125.58121624460226, 1.5991461504423794, 3.8921542842677304, -0.42480589767515003, 0.04342945366498728, 0.035869583960901584, -7.778995113701213e-11, -1.864924702105445e-10, -0.5003058384084661, 20.06843602138423, -2.9159091914388924, 2.79694845484596, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 2.101955828221889, 1.9102451276859023, -1.168277577681548, 8.090925064449195, 2.440388579572656, 4.100778702843739, 6.098892358705557, -0.5422066071648819, 33.06128359069537, 11.317960295355629, -1.4729457663292795, 0.004808192693812404, 0.004668786093539984, -0.25539793443438474, -0.21387545132547553, -0.7056365279916214, -1.9918478750851911, -0.3956187270322553, -0.691877372098673, 19.6625968170485, 17.14235209621459, -13.987051341493572, 0.8746069809222529, 1.2081899266616212, 4.7018959828738875, -9.757759570782537e-11, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1500.0, 11.1, 83000.0, 100.0]
H_R25B_18X75_7 = [1.7877370675639876, 2.95711873401945, -0.5128075639653776, 12.75907533765065, 2.7586554902472877, -0.9341082943735666, 3.8907277232283213e-13, -1.0290636155749008e-13, -2.139834944725509e-13, -44.99160973160116, -0.5183778508636198, -0.8245783873159265, -0.9402805511492536, -59.87166451492142, -9.977565709344459, -5.9696613404864625, 3.7724728212124036e-10, 8.392706199783554e-10, 5.0375669316524444e-09, 1.1337687944521124e-08, -0.7559145478991323, -7.458559724632965, 0.7512919932865336, 2.1473581444631558, -0.2544995132737721, -0.0971812917188021, -0.4789162643400262, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 8.78320913453345, 0.8285269369380858, -2.9267657827424554, -5.202621419998163e-08, 1.9741149206931825, 9.983527677437792, 0.2929765320920469, 1.700013609935922, 0.20917688814549812, -0.1081638712190159, -4.625630121823062e-08, -0.02160929273970723, -3.1527712739410732e-09, -6.956727053409665e-09, 0.5503144482526843, -0.7727254638713603, 1.3592059055998797, -0.5899741182072228, 1.2381107031539917, 2.5729260366575577, 0.5126396510929485, -9.07472209277275e-10, 0.3216786843313352, -1.9217544558780266e-09, -4.5344840816676485e-09, 0.0904588041788402, -0.004975547074258242, 0.9214374571953191, -2.3896636454912232, 2.1918858888139856e-11, 0.029450864198397052, 9.050236318532365e-30, 0.0, 5.793302013025013, 11.337221046300193, 5.984505819941676e-11, -172.20111843946086, 1.019195519188902, 0.4301982077360649, -0.31724706841679284, 0.012724733460844259, 0.014647485984329996, 1.4397420699931106e-11, 1.3050515627291018e-11, 0.1290584688282958, 19.999999999999986, 2.0000000001761276, 10.000000000032571, 1.0, 0.7142598948032157, 1.2441719402377815, 1.6524401649244325, 0.6959172438926858, 0.9999117750081479, 1.000705610587777, 1.0, 0.8253626454243366, 1.0, 0.650386047124958, 0.9999991805119519, 1.0000000098961666, 1.013735770500582, 0.9999998097020494, 2.0198637920630924, 3.553927052282548, 1.0000000003404852, 0.9085654984137196, -0.6036269779566537, 1.2439615003426585, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 1.3223772028870633, 1.500265433530999, -2.690692280494258, 21.891855569878988, -0.6137147188904437, -0.020383999696764077, 6.024243928000388, 0.10488514580373683, 52.42397738471967, -78.44592873689537, 0.25028568171693205, 1.0525789128555224e-06, 8.89772562246529e-05, -0.006836581336998106, 0.06223215274441585, -0.8789044922927745, -0.33562807541600115, -0.17010830684230946, 0.5967167522118404, 6.980630688037098, 15.655213090687996, 57.48202327774629, 1.286128909188702, 2.326225022113537, -0.360880261687522, -3.738094175885617e-11, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1500.0, 11.1, 83000.0, 100.0]
H_R25B_18X75_8 = [1.8979237675551766, 2.8895955780660305, -0.7634490787356585, 17.268535301625857, 2.057322278399794, -0.9393385948234826, -8.483016851683427e-13, -2.6893699175000975e-14, 6.011814340040535e-13, -44.99019861278104, -3.2773019657881863, -0.31189257241666785, -5.8182407821497275, -272.44458215085103, -7.567051289483717, -4.771203477170783, -9.434894932857827e-10, -2.340852361985595e-09, -1.4334988906553382e-08, -3.685045703049248e-08, 2.7570096223073457, -3.524772327585422, 0.09215845335228635, 0.8519284116171212, -0.21670556823395712, -0.11098985167985415, -0.735312218482175, 0.0, 0.0, 0.0, 5.0, 1.0, 10.0, 0.0, 0.0, 0.4, 0.0, 5.0, 0.0, 6.46156569842646, -4.6750351724053365, -7.041666036269811, 2.4926325427329444e-08, 1.9875059497885037, 9.975425255655002, 0.49188453565016543, 1.846444547824074, 0.16735850869795457, -0.16334650483644622, 4.819431779443927e-08, -0.11425558769138738, 7.265005207216432e-10, 1.4576803174216016e-09, -0.08671959351441594, -0.8925120655553098, 1.9124866421806037, -0.876065844159538, 0.8082327992325935, 0.33618680414595514, -1.6493188769353964, -6.583534466167139e-09, 0.10975085366962802, -6.481382562379684e-11, -2.4257494737264033e-10, -0.3885556734090896, -0.43598260827964874, 0.5575462337433931, -2.8302272960509605, -9.73173291702557e-12, 0.028898402846668163, 0.0, 0.0, 5.514619287539191, 16.241626117175304, -2.6736060827893676e-10, -325.7671964869419, 1.2944934156570578, 9.195934424229517, -2.631858674137749, 0.005466069819918737, -0.0006189172944302997, 3.728329894635246e-11, 2.3578887822920064e-10, 0.1912277683809036, 19.99999999999998, 2.0000000944336933, 10.000000028356107, 1.0, 0.5805758787068432, 1.5538022762544246, 1.2021459188478727, 0.7424351628312988, 0.9999975264130104, 0.9915539461395478, 1.0, 0.7808660968233778, 1.0, 0.6081824753482649, 0.9999998467440546, 1.0000027502305504, 1.0046189252311357, 1.0000002600513864, 1.586295055510443, 3.6187692879903532, 0.999999973113924, 1.371455055211531, -0.8528328744315327, 1.7434402289230517, 1.0, 1.0, 1.0, 1.0, 0.79, 0.35, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 1.2527620598361764, 1.2934622187875924, -2.978255938928979, 23.33412669072113, -0.4445818533752435, 0.12922745578301453, 5.942741588116976, -0.06788254402250947, 50.61664966578042, -79.60980256975046, 0.3059725936624027, -0.0011779069976050286, -0.0020849790386074617, 0.02793696349913063, 0.1119472162225622, -0.8837599365873179, -0.590155243874908, -0.0506626909911555, 0.24177452822122641, 8.380207372941058, 14.196287800021091, 86.91578251677883, 1.3236908090509143, 2.785274217965793, -0.6479904099273097, -3.31257574816826e-11, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0004, 4e-05, 0.0, 0.0, 0.85, -0.4, 0.4, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, -4.0, 0.5, 0.0, 10.0, 0.2, 0.1, 0.1, 1.0, 8.0, 0.24, 0.01, 1.0, 0.0, 0.223, 1500.0, 11.1, 83000.0, 100.0]

def read_tir(file_path):
    """
    Reads the *.tir file at the specified path and returns a TireMFModel for the given file
    """

    curr_tire_model = MODEL_DEFAULTS

    mass_switch = True  # there are two MASS vars and we need to throw away the first one
    with open(file_path, 'r') as filehandle:
        for line in filehandle:
            if line[0] != "$" and line[0] != "[":
                parts = line.split()
                if len(parts) > 2 and parts[0] in curr_tire_model.keys():
                    var_type = type(curr_tire_model[parts[0]])
                    if parts[0] == "MASS":
                        if mass_switch:
                            mass_switch = False
                            curr_tire_model[parts[0]] = parts[2]
                        else:
                            curr_tire_model["MASS1"] = float(parts[2])
                    elif var_type is int:
                        curr_tire_model[parts[0]] = int(parts[2])
                    elif var_type is float:
                        curr_tire_model[parts[0]] = float(parts[2])
                    else:
                        curr_tire_model[parts[0]] = parts[2]
    tm = PacejkaModel(curr_tire_model)
    # Set nominal parameters of the model (DO NOT CHANGE AFTER)
    tm.UNLOADED_RADIUS = 0.26  # Unloaded tire radius
    tm.FNOMIN = 1500  # Nominal load THIS MUST BE SET TO AT LEAST 50% OF THE MAXIMUM LOAD OR YOU WILL GET FUCKING CRAZY HARD TO DIAGNOSE ERRORS
    tm.LONGVL = 11.1  # Nominal reference speed
    tm.NOMPRES = 83000  # Nominal inflation pressure Pa
    tm.FZMIN = 100
    return tm

def write_tir(file_path, tm):
    """
    Writes the *.tir file at the specified path and returns a TireMFModel for the given file
    """
    additional_params = ["UNLOADED_RADIUS", "FNOMIN", "LONGVL", "NOMPRES", "FZMIN"]
    num_params = [tm.UNLOADED_RADIUS, tm.FNOMIN, tm.LONGVL, tm.NOMPRES, tm.FZMIN]
    lables_param = ["Free tyre radius", "Nominal wheel load", "Nominal speed", "Nominal tyre inflation pressure - NEVER MODIFY THIS PARAMETER", "Minimum allowed wheel load"]
    params = tm.dump_params()
    row = "{name:<25s}=    {val:<12g}   ${label}\n".format # This formating works, dont mess with it
    with open(file_path, 'w') as filehandle:
        with open(make_path('./Data/TTCData/TIR_Templates/FSAE_Defaults.tir'), 'r') as template:
            for line in template:
                if line[0] != "$" and line[0] != "[":
                    parts = line.split()
                    if len(parts) > 2:
                        if parts[0] in NAMES:
                            param_ind = NAMES.index(parts[0])
                            param = params[param_ind]
                            filehandle.write(row(name=parts[0], val=param, label=LABELS[param_ind]))
                            # filehandle.write(f"{parts[0]} = {param} ${LABLES[param_ind]}\n")
                        elif parts[0] in additional_params:
                            param_ind = additional_params.index(parts[0])
                            param = num_params[param_ind]
                            # filehandle.write(f"{parts[0]} = {param} ${lables_param[param_ind]}\n")
                            filehandle.write(row(name=parts[0], val=param, label=lables_param[param_ind]))
                        else:
                            filehandle.write(line)
                    else:
                        filehandle.write(line)
                else:
                    filehandle.write(line)
                        