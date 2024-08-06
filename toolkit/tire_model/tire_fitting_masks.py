import numpy as np

DEFAULT_MASK = np.arange(186)

# dont use 87, 151, 175, 176 in any of them, they are kinda global params

# surface specific scaling factors
# 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 111

LABELS = ["Shape factor Cfy for lateral forces", "Lateral friction Muy", "Variation of friction Muy with load", "Variation of friction Muy with squared camber", "Lateral curvature Efy at Fznom", "Variation of curvature Efy with load", "Zero order camber dependency of curvature Efy", "Variation of curvature Efy with camber", "Variation of curvature Efy with camber squared", "Maximum value of stiffness Kfy/Fznom", "Load at which Kfy reaches maximum value", "Variation of Kfy/Fznom with camber", "Curvature of stiffness Kfy", "Peak stiffness variation with camber squared", "Fy camber stiffness factor", "Vertical load dependency of camber stiffness", "Horizontal shift Shy at Fznom", "Variation of shift Shy with load", "Vertical shift in Svy/Fz at Fznom", "Variation of shift Svy/Fz with load", "Variation of shift Svy/Fz with camber", "Variation of shift Svy/Fz with camber and load", "influence of inflation pressure on cornering stiffness", "influence of inflation pressure on dependency of nominal tyre load on cornering stiffness", "linear influence of inflation pressure on lateral peak friction", "quadratic influence of inflation pressure on lateral peak friction", "Influence of inflation pressure on camber stiffness", "Vertical shift of overturning moment", "Camber induced overturning couple", "Fy induced overturning couple", "Mixed load lateral force and camber on Mx", "Load effect on Mx with lateral force and camber", "B-factor of load with Mx", "Camber with load on Mx", "Lateral force with load on Mx", "B-factor of lateral force with load on Mx", "Vertical force with camber on Mx", "B-factor of vertical force with camber on Mx", "Influence of inflation pressure on overturning moment", "Trail slope factor for trail Bpt at Fznom", "Variation of slope Bpt with load", "Variation of slope Bpt with load squared", "Variation of slope Bpt with camber", "Variation of slope Bpt with absolute camber", "Factor for scaling factors of slope factor Br of Mzr", "Factor for dimensionless cornering stiffness of Br of Mzr", "Shape factor Cpt for pneumatic trail",
          "Peak trail Dpt = Dpt*(Fz/Fznom*R0)", "Variation of peak Dpt with load", "Variation of peak Dpt with camber", "Variation of peak Dpt with camber squared", "Peak residual torque Dmr = Dmr/(Fz*R0)", "Variation of peak factor Dmr with load", "Variation of peak factor Dmr with camber", "Variation of peak factor Dmr with camber and load", "Variation of peak factor Dmr with camber squared", "Variation of Dmr with camber squared and load", "Trail curvature Ept at Fznom", "Variation of curvature Ept with load", "Variation of curvature Ept with load squared", "Variation of curvature Ept with sign of Alpha-t", "Variation of Ept with camber and sign Alpha-t", "Trail horizontal shift Sht at Fznom", "Variation of shift Sht with load", "Variation of shift Sht with camber", "Variation of shift Sht with camber and load", "effect of inflation pressure on length of pneumatic trail", "Influence of inflation pressure on residual aligning torque", "Nominal value of s/R0: effect of Fx on Mz", "Variation of distance s/R0 with Fy/Fznom", "Variation of distance s/R0 with camber", "Variation of distance s/R0 with load and camber", "Slope factor for combined Fy reduction", "Variation of slope Fy reduction with alpha", "Shift term for alpha in slope Fy reduction", "Influence of camber on stiffness of Fy combined", "Shape factor for combined Fy reduction", "Curvature factor of combined Fy", "Curvature factor of combined Fy with load", "Shift factor for combined Fy reduction", "Shift factor for combined Fy reduction with load", "Kappa induced side force Svyk/Muy*Fz at Fznom", "Variation of Svyk/Muy*Fz with load", "Variation of Svyk/Muy*Fz with camber", "Variation of Svyk/Muy*Fz with alpha", "Variation of Svyk/Muy*Fz with kappa", "Variation of Svyk/Muy*Fz with atan(kappa)", "Scale factor of nominal (rated) load", "Scale factor of Fx shape factor", "Scale factor of Fx peak friction coefficient", "Scale factor of Fx curvature factor", "Scale factor of Fx slip stiffness", "Scale factor of Fx horizontal shift", "Scale factor of Fx vertical shift", "Scale factor of Fy shape factor", "Scale factor of Fy peak friction coefficient", "Scale factor of Fy curvature factor", "Scale factor of Fy cornering stiffness", "Scale factor of Fy horizontal shift", "Scale factor of Fy vertical shift", "Scale factor of peak of pneumatic trail", "Scale factor for offset of residual torque", "Scale factor of alpha influence on Fx", "Scale factor of kappa influence on Fy", "Scale factor of kappa induced Fy", "Scale factor of moment arm of Fx", "Scale factor of camber force stiffness", "Scale factor of camber torque stiffness", "Scale factor of Mx vertical shift", "Scale factor of overturning couple", "Scale factor of rolling resistance torque", "Scale factor of Parking Moment", "Square root term in contact length equation", "Linear term in contact length equation", "Root term in contact width equation", "Linear term in contact width equation", "Load and speed influence on in-plane translation stiffness", "Load and speed influence on in-plane rotation stiffness", "Tyre overall longitudinal stiffness vertical deflection dependency linear term", "Tyre overall longitudinal stiffness vertical deflection dependency quadratic term", "Tyre overall longitudinal stiffness pressure dependency", "Tyre overall lateral stiffness vertical deflection dependency linear term", "Tyre overall lateral stiffness vertical deflection dependency quadratic term", "Tyre overall lateral stiffness pressure dependency", "Tyre overall yaw stiffness pressure dependency", "Shape factor Cfx for longitudinal force", "Longitudinal friction Mux at Fznom", "Variation of friction Mux with load", "Variation of friction Mux with camber squared", "Longitudinal curvature Efx at Fznom", "Variation of curvature Efx with load", "Variation of curvature Efx with load squared", "Factor in curvature Efx while driving", "Longitudinal slip stiffness Kfx/Fz at Fznom", "Variation of slip stiffness Kfx/Fz with load", "Exponent in slip stiffness Kfx/Fz with load", "Horizontal shift Shx at Fznom", "Variation of shift Shx with load", "Vertical shift Svx/Fz at Fznom", "Variation of shift Svx/Fz with load", "linear influence of inflation pressure on longitudinal slip stiffness", "quadratic influence of inflation pressure on longitudinal slip stiffness", "linear influence of inflation pressure on peak longitudinal friction", "quadratic influence of inflation pressure on peak longitudinal friction", "Slope factor for combined slip Fx reduction", "Variation of slope Fx reduction with kappa", "Influence of camber on stiffness for Fx combined", "Shape factor for combined slip Fx reduction", "Curvature factor of combined Fx", "Curvature factor of combined Fx with load", "Shift factor for combined slip Fx reduction", "Scale factor with slip speed Vs decaying friction", "Camber squared induced overturning moment", "Lateral force induced overturning moment", "Lateral force induced overturning moment with camber", "Rolling resistance torque coefficien", "Rolling resistance torque depending on Fx", "Rolling resistance torque depending on speed", "Rolling resistance torque depending on speed ^4", "Rolling resistance torque depending on camber squared", "Rolling resistance torque depending on load and camber squared", "Rolling resistance torque coefficient load dependency", "Rolling resistance torque coefficient pressure dependency", "Peak Fx reduction due to spin parameter", "Peak Fx reduction due to spin with varying load parameter", "Peak Fx reduction due to spin with kappa parameter", "Cornering stiffness reduction due to spin", "Peak Fy reduction due to spin parameter", "Peak Fy reduction due to spin with varying load parameter", "Peak Fy reduction due to spin with alpha parameter", "Peak Fy reduction due to square root of spin parameter", "Fy-alpha curve lateral shift limitation", "Fy-alpha curve maximum lateral shift parameter", "Fy-alpha curve maximum lateral shift varying with load parameter", "Fy-alpha curve maximum lateral shift parameter", "Camber w.r.t. spin reduction factor parameter in camber stiffness", "Camber w.r.t. spin reduction factor varying with load parameter in camber stiffness", "Pneumatic trail reduction factor due to turn slip parameter", "Turning moment at constant turning and zero forward speed parameter", "Turn slip moment (at alpha = 90deg) parameter for increase with spin", "Residual (spin) torque reduction factor parameter due to side slip", "Turn slip moment peak magnitude parameter",
          "Low load stiffness effective rolling radius", "Peak value of effective rolling radius", "High load stiffness effective rolling radius", "Ratio of free tyre radius with nominal tyre radius", "Tyre radius increase with speed", "UNLOADED Radius", "Force Nominal", "Refrence Velocity", "Nominal Pressure", "Fz Min"]

NAMES = ["PCY1", "PDY1", "PDY2", "PDY3", "PEY1", "PEY2", "PEY3", "PEY4", "PEY5", "PKY1", "PKY2", "PKY3", "PKY4", "PKY5", "PKY6", "PKY7", "PHY1", "PHY2", "PVY1", "PVY2", "PVY3", "PVY4", "PPY1", "PPY2", "PPY3", "PPY4", "PPY5", "QSX1", "QSX2", "QSX3", "QSX4", "QSX5", "QSX6", "QSX7", "QSX8", "QSX9", "QSX10", "QSX11", "PPMX1", "QBZ1", "QBZ2", "QBZ3", "QBZ4", "QBZ5", "QBZ9", "QBZ10", "QCZ1", "QDZ1", "QDZ2", "QDZ3", "QDZ4", "QDZ6", "QDZ7", "QDZ8", "QDZ9", "QDZ10", "QDZ11", "QEZ1", "QEZ2", "QEZ3", "QEZ4", "QEZ5", "QHZ1", "QHZ2", "QHZ3", "QHZ4", "PPZ1", "PPZ2", "SSZ1", "SSZ2", "SSZ3", "SSZ4", "RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2", "RHY1", "RHY2", "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6", "LFZO", "LCX", "LMUX", "LEX", "LKX", "LHX", "LVX", "LCY", "LMUY", "LEY", "LKY", "LHY", "LVY", "LTR", "LRES", "LXAL", "LYKA ", "LVYKA", "LS", "LKYC", "LKZC", "LVMX", "LMX", "LMY", "LMP", "Q_RA1", "Q_RA2", "Q_RB1", "Q_RB2", "Q_BVX", "Q_BVT", "PCFX1", "PCFX2", "PCFX3", "PCFY1", "PCFY2", "PCFY3", "PCMZ1", "PCX1", "PDX1", "PDX2", "PDX3", "PEX1", "PEX2", "PEX3", "PEX4", "PKX1", "PKX2", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2", "PPX1", "PPX2", "PPX3", "PPX4", "RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1", "LMUV", "QSX12", "QSX13", "QSX14", "QSY1", "QSY2", "QSY3", "QSY4", "QSY5", "QSY6", "QSY7", "QSY8", "PDXP1", "PDXP2", "PDXP3", "PKYP1", "PDYP1", "PDYP2", "PDYP3", "PDYP4", "PHYP1", "PHYP2", "PHYP3", "PHYP4", "PECP1", "PECP2", "QDTP1", "QCRP1", "QCRP2", "QBRP1", "QDRP1", "BREFF", "DREFF", "FREFF", "Q_RE0", "Q_V1", "UNLOADED Radius", "FNOMIN", "LONGVL", "NOMPRES", "FZMIN"]

def generate_mask(param_list):
    params = []
    for i, param in enumerate(NAMES):
        if param in param_list:
            params.append(i)
    return params

def get_param_list(mask):
    params = "[\""
    for i in mask:
        params += NAMES[i] + "\", \""
    params = params[:-4] + "\"]"
    return params

FY_MASK_IND = generate_mask(["PCY1", "PDY1", "PDY2", "PEY1", "PEY2", "PKY1", "PKY2", "PKY4", "PHY1", "PHY2", "PVY1", "PVY2"])

FY_MASK_COM = generate_mask(["RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2", "RHY1", "RHY2", "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6"])

FY_MASK_PRESS = generate_mask(["PPY1", "PPY2", "PPY3", "PPY4"])

FY_MASK_IA = generate_mask(["PDY3", "PEY3", "PEY4", "PEY5", "PKY3", "PKY5", "PKY6", "PKY7", "PVY3", "PVY4"])

FY_MASK_IA_PRESS = generate_mask(["PPY5"])

FY_TURNSLIP = generate_mask(["PKYP1", "PDYP1", "PDYP2", "PDYP3", "PDYP4", "PHYP1", "PHYP2", "PHYP3", "PHYP4"])

FY_LAMBDA = generate_mask(["LMY", "LKY", "LEY", "LKYC", "LCY", "LVY"])

FX_MASK_IND = generate_mask(["PCX1", "PDX1", "PDX2", "PEX1", "PEX2", "PEX3", "PEX4", "PKX1", "PKX2", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2"])

FX_MASK_COM = generate_mask(["RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1"])

FX_MASK_PRESS = generate_mask(["PPX1", "PPX2", "PPX3", "PPX4"])

FX_MASK_IA = generate_mask(["PDX3"])

FX_TURNSLIP = generate_mask(["PDXP1", "PDXP2", "PDXP3"])

MZ_MASK_IND = generate_mask(["QBZ1", "QBZ2", "QBZ3", "QBZ9", "QBZ10", "QCZ1", "QDZ1", "QDZ2", "QDZ6", "QDZ7", "QEZ1", "QEZ2", "QEZ3", "QEZ4", "QHZ1", "QHZ2"])

MZ_MASK_COM = generate_mask(["SSZ1", "SSZ2", "SSZ3", "SSZ4"])

MZ_MASK_PRESS = generate_mask(["PPZ1", "PPZ2"])

MZ_MASK_IA = generate_mask(["QBZ4", "QBZ5", "QDZ3", "QDZ4", "QDZ8", "QDZ9", "QDZ10", "QDZ11", "QEZ5", "QHZ3", "QHZ4"])

MZ_TURNSLIP = generate_mask(["QDTP1", "QCRP1", "QCRP2", "QBRP1", "QDRP1"])

MX_MASK_IND_PRESS = np.array([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 108, 109, 152, 153, 154], int)

RE_MASK_IND_PRESS = np.array([182, 183, 184, 185, 186], int)