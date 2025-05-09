import numpy as np
from .tire_model_utils import VarInf, PostProInputs, Mode, ForceMoments, Result, RetValue, InputRanges, dump_tk
from toolkit.common import safe_sign, interpolate
from .pacejka_coefficients import PacejkaModel

class MFModel(PacejkaModel):
    # TireMFModel Solver for Magic Formula 5.2, 6.1 and 6.2 Tyre Models
    def __init__(self, tire_coefficients: dict) -> None:
        # Parameters not specified in the TIR file
        # Used to avoid low speed singularity
        self.epsilon = 1e-6  # [Eqn (4.E6a) Page 178 - Book]
        super().__init__(tire_coefficients)

    def fullSteadyState(self, inputs, use_turnslip=False):

        userDynamics = 0  # Solve in steady state mode

        Fz = inputs[:, 0]       # vertical load         (N)
        kappa = inputs[:, 1]    # longitudinal slip     (-) (-1 = locked wheel)
        alpha = inputs[:, 2]    # side slip angle       (radians)
        gamma = inputs[:, 3]    # inclination angle     (radians)
        phit = inputs[:, 4]     # turn slip             (1/m)
        Vcx = inputs[:, 5]      # forward velocity      (m/s)
        p = inputs[:, 6]        # pressure              (Pa)
        omega = inputs[:, 7]    # wheel speed           (rad/s)

        # Call parseInputs
        postProInputs, reductionSmooth, modes = self.parseInputs(userDynamics, Fz, kappa, alpha, gamma, phit, Vcx, p, omega, useTurnSlip=use_turnslip)

        # Call doForcesAndMoments
        forces_and_moments, varinf = self.doForcesAndMoments(postProInputs, reductionSmooth, modes)

        # Call doExtras
        Re, omega_out, rho, Rl, a, b, Cx, Cy, Cz, sigmax, sigmay, instKya = self.doExtras(postProInputs, forces_and_moments, varinf)

        # Check the sign of the coefficient of friction
        # The calculation of Fy is not affected by the sign of muy
        if modes.useLimitsCheck:
            if np.any(varinf.muy < 0):
                print('Negative lateral coefficient of friction forced to be positive')
        varinf.muy = np.abs(varinf.muy)

        # Preallocate out variable
        out = np.zeros((inputs.shape[0], 32))

        # Pack all the outputs
        out[:, 0] = np.real(forces_and_moments.Fx)
        out[:, 1] = np.real(forces_and_moments.Fy)
        out[:, 2] = np.real(postProInputs.uFz)
        out[:, 3] = np.real(forces_and_moments.Mx)
        out[:, 4] = np.real(forces_and_moments.My)
        out[:, 5] = np.real(forces_and_moments.Mz)
        out[:, 6] = np.real(postProInputs.ukappaLow)
        out[:, 7] = np.real(postProInputs.ualpha)
        out[:, 8] = np.real(postProInputs.ugamma)
        out[:, 9] = np.real(postProInputs.phit)
        out[:, 10] = np.real(postProInputs.uVcx)
        out[:, 11] = np.real(postProInputs.p)
        out[:, 12] = np.real(Re)
        out[:, 13] = np.real(rho)
        out[:, 14] = np.real(2*a)
        out[:, 15] = np.real(varinf.t)
        out[:, 16] = np.real(varinf.mux)
        out[:, 17] = np.real(varinf.muy)
        out[:, 18] = np.real(omega_out)
        out[:, 19] = np.real(Rl)
        out[:, 20] = np.real(2*b)
        out[:, 21] = np.real(varinf.Mzr)
        out[:, 22] = np.real(Cx)
        out[:, 23] = np.real(Cy)
        out[:, 24] = np.real(Cz)
        out[:, 25] = np.real(varinf.Kya)
        out[:, 26] = np.real(sigmax)
        out[:, 27] = np.real(sigmay)
        out[:, 28] = np.real(instKya)
        out[:, 29] = np.real(varinf.Kxk)
        out[:, 30] = np.real(forces_and_moments.dFz)
        out[:, 31] = np.real(postProInputs.phi)

        return out

    def steady_state(self, Fz, kappa, alpha, gamma, phit, Vcx, p, omega):

        userDynamics = 0  # Solve in steady state mode

        postProInputs, reductionSmooth, modes = self.parseInputs(userDynamics, Fz, kappa, alpha, gamma, phit, Vcx, p, omega)

        forces_and_moments, varinf = self.doForcesAndMoments(postProInputs, reductionSmooth, modes)

        return np.real(forces_and_moments.Fx), np.real(forces_and_moments.Fy), np.real(forces_and_moments.Mx), np.real(forces_and_moments.Mz)

    def steady_state_mmd(self, f_z: float, s_a: float, s_r: float, p: float = 82500, i_a: float = 0, v: float = 15, omega: float = 0.0, mu_corr: float = 1.0, flip_s_a: bool = False):
        if flip_s_a:
            s_a = -s_a
        postProInputs, reductionSmooth, modes = self.fast_parse_inputs(0, f_z, s_r, s_a, i_a, 0.0, v, p, omega, ncolumns=1)

        Fx, Fy, Mz = self.do_forces_and_moments_fast(postProInputs, reductionSmooth, modes)
        if flip_s_a:
            return Fx.real * mu_corr, -Fy.real * mu_corr, -Mz.real * mu_corr
        return Fx.real * mu_corr, Fy.real * mu_corr, Mz.real * mu_corr

    def s_r_sweep(self, f_z: float, s_a: float, s_r, p: float = 82500, i_a: float = 0, v: float = 15, mu_corr: float = 1.0, flip_s_a: bool = False):
        exp_len = len(s_r)
        if flip_s_a:
            s_a = -s_a
        postProInputs, reductionSmooth, modes = self.fast_parse_inputs(0, f_z, s_r, s_a, i_a, 0.0, v, p, 0.0, ncolumns=exp_len)

        Fx, Fy, Mz = self.do_forces_and_moments_fast(postProInputs, reductionSmooth, modes)
        if flip_s_a:
            return Fx.real * mu_corr, -Fy.real * mu_corr, -Mz.real * mu_corr
        return Fx.real * mu_corr, Fy.real * mu_corr, Mz.real * mu_corr

    def s_a_sweep(self, f_z: float, s_a, s_r, p: float = 82500, i_a: float = 0, v: float = 15, mu_corr: float = 1.0, flip_s_a: bool = False):
        exp_len = len(s_r)
        if flip_s_a:
            s_a = -s_a
        postProInputs, reductionSmooth, modes = self.fast_parse_inputs(0, f_z, s_r, s_a, i_a, 0.0, v, p, 0.0, ncolumns=exp_len)

        Fx, Fy, Mz = self.do_forces_and_moments_fast(postProInputs, reductionSmooth, modes)
        if flip_s_a:
            return Fx.real * mu_corr, -Fy.real * mu_corr, -Mz.real * mu_corr
        return Fx.real * mu_corr, Fy.real * mu_corr, Mz.real * mu_corr

    def dump_params(self):
        return dump_tk(self)

    def parseInputs(self, userDynamics, Fz, kappa, alpha, gamma, phit, Vcx, p, omega, useLimitsCheck=False, useAlphaStar=False, useTurnSlip=False):
        ncolumns = len(Fz)

        # IMPORTANT NOTE: Vx = Vcx [Eqn (7.4) Page 331 - Book]
        # It is assumed that the difference between the wheel centre
        # longitudinal velocity Vx and the longitudinal velocity Vcx of
        # the contact centre is negligible

        Fz[Fz < 0] = 0  # If any Fz is negative set it to zero

        # Create a copy of the variables (u stands for unlimited)
        uFz = Fz
        ukappa = kappa
        ukappaLow = kappa
        ualpha = alpha
        ugamma = gamma
        uphit = phit
        uVcx = Vcx

        # Limits:
        # This section applies the appropriate limits to the input
        # values of the model based on the MF limits specified in
        # the TIR File

        Fz_lowLimit = Fz

        if useLimitsCheck:
            # Turn slip modifications
            phit = phit * np.cos(alpha)  # Empirically discovered

            # Low Speed Model:
            # Create a reduction factor for low speed and standstill
            isLowSpeed = np.where(np.abs(Vcx) <= self.VXLOW)

            # Create a vector with numbers between 0 and 1 to apply a
            # reduction factor with smooth transitions.
            Wvlow = 0.5 *  (1 + np.cos(np.pi() * (Vcx[isLowSpeed] / self.VXLOW)))
            reductionSmooth = 1-Wvlow

            # Create a vector with numbers between 0 and 1 to apply a
            # linear reduction toward zero
            reductionLinear = np.abs(Vcx[isLowSpeed] / self.VXLOW)

            # ukappaLow is equal to ukappa but with low speed
            # correction. This is only used to export kappa
            ukappaLow[isLowSpeed] = np.real(ukappaLow[isLowSpeed] * reductionLinear)

            # If Vcx is close to zero, use linear reduction
            phit[isLowSpeed] = phit[isLowSpeed] * reductionLinear

            # If the speed is negative, the turn slip is also negative
            isNegativeSpeed = np.where(Vcx < 0)
            phit[isNegativeSpeed] = -phit[isNegativeSpeed]

            # Sum the forward and lateral speeds
            Vsy = -np.tan(alpha) * np.abs(Vcx)
            speedSum = np.abs(Vcx) + np.abs(Vsy)

            # The slip angle also suffers a reduction when the sum of
            # Vx and Vy is less than self.VXLOW
            isLowSpeedAlpha = np.where(speedSum < self.VXLOW)

            # Create a vector with numbers between 0 and 1 to apply a
            # linear reduction toward zero for alpha
            reductionLinear_alpha = speedSum[isLowSpeedAlpha] / self.VXLOW

            kappa[isLowSpeed] = kappa[isLowSpeed] * reductionLinear

            alpha[isLowSpeedAlpha] = alpha[isLowSpeedAlpha] *  reductionLinear_alpha

            # Check Slip Angle limits
            isLowSlip = np.where(alpha < self.ALPMIN)
            alpha[isLowSlip] = self.ALPMIN

            isHighSlip = np.where(alpha > self.ALPMAX)
            alpha[isHighSlip] = self.ALPMAX

            # Check camber limits
            isLowCamber = np.where(gamma < self.CAMMIN)
            gamma[isLowCamber] = self.CAMMIN

            isHighCamber = np.where(gamma > self.CAMMAX)
            gamma[isHighCamber] = self.CAMMAX

            # Check Fz limits
            isHighLoad = np.where(Fz > self.FZMAX)
            Fz[isHighLoad] = self.FZMAX

            # Create a copy of Fz and apply the low limit.
            # This is only used in some Moments equations
            Fz_lowLimit = Fz
            isLowLoad = np.where(Fz < self.FZMIN)
            Fz_lowLimit[isLowLoad] = self.FZMIN

            # Check pressure limits
            isLowPressure = np.where(p < self.PRESMIN)
            p[isLowPressure] = self.PRESMIN

            isHighPressure = np.where(p > self.PRESMAX)
            p[isHighPressure] = self.PRESMAX

            # Check slip ratio limits
            isLowSlipR = np.where(kappa < self.KPUMIN)
            kappa[isLowSlipR] = self.KPUMIN

            isHighSlipR = np.where(kappa > self.KPUMAX)
            kappa[isHighSlipR] = self.KPUMAX

            # Flag if anything is out of range.
            if np.any(isLowSlip):
                print('Slip angle below the limit. Values have been saturated.')
            if np.any(isHighSlip):
                print('Slip angle above the limit. Values have been saturated.')
            if np.any(isLowCamber):
                print('Inclination angle below the limit. Values have been saturated.')
            if np.any(isHighCamber):
                print('Inclination angle above the limit. Values have been saturated.')
            if np.any(isLowLoad):
                print('Vertical load below the limit. Values have been saturated.')
            if np.any(isHighLoad):
                print('Vertical load above the limit. Values have been saturated.')
            if np.any(isLowPressure):
                print('Pressure below the limit. Values have been saturated.')
            if np.any(isHighPressure):
                print('Pressure above the limit. Values have been saturated.')
            if np.any(isLowSlipR):
                print('Slip ratio below the limit. Values have been saturated.')
            if np.any(isHighSlipR):
                print('Slip ratio above the limit. Values have been saturated.')
            if np.any(isLowSpeed):
                print('Speed input VX below the limit. Low speed mode activated.')

        else:
            isLowSpeed = np.zeros(len(Fz))
            reductionSmooth = np.ones(len(Fz))
            reductionLinear = reductionSmooth
            isLowSpeedAlpha = isLowSpeed
            reductionLinear_alpha = reductionSmooth

        modes: Mode = Mode(useLimitsCheck, useAlphaStar, useTurnSlip,
                           isLowSpeed, isLowSpeedAlpha, userDynamics)

        ualpha[uVcx == 0] = 0  # Zero speed (empirically discovered)

        post_pro_inputs = PostProInputs(omega, 0.0, 0.0, uFz, ukappa, ukappaLow, ualpha, ugamma, uphit, uVcx, alpha, kappa, gamma, phit, Fz, p, ncolumns, Fz_lowLimit)

        # [Eqn (4.E2b) Page 177 - Book]
        # dpi = (p - self.NOMPRES) / self.NOMPRES

        # Re, _, _ = self.calculateRe(post_pro_inputs, dpi)

        # Longitudinal slip velocity
        # it dosnt even use this
        # Vsx = Vcx - Re * omega  # [Eqn (2.3) Page 64 - Book]
        # post_pro_inputs.Vsx = Vsx

        epsilon = 0
        if useTurnSlip:
            Fz0_prime = self.LFZO * self.FNOMIN  # [Eqn (4.E1) Page 177 - Book]
            dfz = (Fz - Fz0_prime) / Fz0_prime  # [Eqn (4.E2a) Page 177 - Book]

            Vsy = -np.tan(alpha) * np.abs(Vcx)
            v_c = np.sqrt(Vcx**2 + Vsy**2)

            # [Eqn (4.90) Page 186 - Book] Camber reduction factor
            epsilon = self.PECP1 * (1 + self.PECP2 * dfz)

            # Speed limits (avoid zero speed)
            # Vc_prime = Vcx # From the book
            Vc_prime = v_c  # From the Equation Manual

            isLowSpeed = np.abs(Vcx) < self.VXLOW
            signVcx = np.sign(Vcx[isLowSpeed])
            signVcx[signVcx == 0] = 1
            # Singularity protected velocity, text on Page 184
            Vc_prime[isLowSpeed] = np.real(self.VXLOW * signVcx)

            # Rearrange [Eqn (4.75) Page 183 - Book]
            psi_dot = -phit * Vc_prime

            phi = (1 / Vc_prime) * (psi_dot - (1 - epsilon) * omega * np.sin(gamma))  # [Eqn (4.76) Page 184 - Book]

            # IMPORTANT NOTE: Eqn (4.76) has been modified
            # In chapter 2.2 "Definition of tire input quantities" in the Pacejka
            # book, it is assumed that the z-axis of the road coordinate system
            # "points downward normal to the road plane" (p. 62). Due to this
            # definition, Pacejka introduces the minus sign for the spin slip so
            # that a positive spin leads to a positive torque Mz (p. 68).But in
            # CarMaker (and other MBS software), the road coordinate system is
            # orientated differently. The z-axis points upward to the
            # road plane. Thus switching the signs is not needed here.

            post_pro_inputs.phi = np.real(phi)

        return post_pro_inputs, reductionSmooth, modes

    def fast_parse_inputs(self, userDynamics, Fz, kappa, alpha, gamma, phit, Vcx, p, omega, useLimitsCheck=False, useAlphaStar=False, useTurnSlip=False, ncolumns=None):
        if ncolumns is None:
            ncolumns = len(Fz)

        # IMPORTANT NOTE: Vx = Vcx [Eqn (7.4) Page 331 - Book]
        # It is assumed that the difference between the wheel centre
        # longitudinal velocity Vx and the longitudinal velocity Vcx of
        # the contact centre is negligible
        if type(Fz) is np.ndarray:
            Fz[Fz < 0] = 0  # If any Fz is negative set it to zero
        elif Fz < 0:
            Fz = 0

        # Create a copy of the variables (u stands for unlimited)
        ualpha = alpha

        isLowSpeed = 0.0
        reductionSmooth = 1.0

        modes: Mode = Mode(useLimitsCheck, useAlphaStar, useTurnSlip,
                           isLowSpeed, reductionSmooth, userDynamics)
        if type(ualpha) is np.ndarray:
            ualpha[Vcx == 0] = 0  # Zero speed (empirically discovered)

        post_pro_inputs = PostProInputs(omega, 0.0, 0.0, Fz, kappa, kappa, ualpha, gamma, phit, Vcx, alpha, kappa, gamma, phit, Fz, p, ncolumns, Fz)

        if useTurnSlip:
            Fz0_prime = self.LFZO * self.FNOMIN  # [Eqn (4.E1) Page 177 - Book]
            dfz = (Fz - Fz0_prime) / Fz0_prime  # [Eqn (4.E2a) Page 177 - Book]

            Vsy = -np.tan(alpha) * np.abs(Vcx)
            v_c = np.sqrt(Vcx**2 + Vsy**2)

            # [Eqn (4.90) Page 186 - Book] Camber reduction factor
            epsilon = self.PECP1 * (1 + self.PECP2 * dfz)

            # Speed limits (avoid zero speed)
            # Vc_prime = Vcx # From the book
            Vc_prime = v_c  # From the Equation Manual

            isLowSpeed = np.abs(Vcx) < self.VXLOW
            signVcx = np.sign(Vcx[isLowSpeed])
            signVcx[signVcx == 0] = 1
            # Singularity protected velocity, text on Page 184
            Vc_prime[isLowSpeed] = np.real(self.VXLOW * signVcx)

            # Rearrange [Eqn (4.75) Page 183 - Book]
            psi_dot = -phit * Vc_prime

            phi = (1 / Vc_prime) * (psi_dot - (1 - epsilon) * omega * np.sin(gamma))  # [Eqn (4.76) Page 184 - Book]

            # IMPORTANT NOTE: Eqn (4.76) has been modified
            # In chapter 2.2 "Definition of tire input quantities" in the Pacejka
            # book, it is assumed that the z-axis of the road coordinate system
            # "points downward normal to the road plane" (p. 62). Due to this
            # definition, Pacejka introduces the minus sign for the spin slip so
            # that a positive spin leads to a positive torque Mz (p. 68).But in
            # CarMaker (and other MBS software), the road coordinate system is
            # orientated differently. The z-axis points upward to the
            # road plane. Thus switching the signs is not needed here.

            post_pro_inputs.phi = np.real(phi)

        return post_pro_inputs, reductionSmooth, modes

    def calculateBasic(self, modes, postProInputs):
        # Velocities in point S (slip point)
        # [Eqn (4.E5) Page 181 - Book]
        Vsx = -postProInputs.kappa * np.abs(postProInputs.uVcx)
        # [Eqn (2.12) Page 67 - Book] and [(4.E3) Page 177 - Book]
        Vsy = np.tan(postProInputs.alpha) * np.abs(postProInputs.uVcx)

        # Important Note:
        # Due to the ISO sign convention, equation 2.12 does not need a
        # negative sign. The Pacejka book is written in adapted SAE.
        # [Eqn (3.39) Page 102 - Book] -> Slip velocity of the slip point S
        Vs = np.sqrt(Vsx**2 + Vsy**2)

        # Velocities in point C (contact)
        # Assumption from page 67 of the book, paragraph above Eqn (2.11)
        Vcy = Vsy
        # Velocity of the wheel contact centre C, Not described in the book but is the same as [Eqn (3.39) Page 102 - Book]
        Vc = np.sqrt(postProInputs.uVcx**2 + Vcy**2)

        # Effect of having a tire with a different nominal load
        Fz0_prime = self.LFZO * self.FNOMIN  # [Eqn (4.E1) Page 177 - Book]

        # Normalized change in vertical load
        dfz = (postProInputs.Fz - Fz0_prime) / Fz0_prime  # [Eqn (4.E2a) Page 177 - Book]
        # Normalized change in inflation pressure
        # [Eqn (4.E2b) Page 177 - Book]
        dpi = (postProInputs.p - self.NOMPRES) / self.NOMPRES

        # Use of star (*) definition. Only valid for the book
        # implementation. TNO MF-Tyre does not use this.
        if modes.useAlphaStar:
            # [Eqn (4.E3) Page 177 - Book]
            alpha_star = np.tan(postProInputs.alpha) * np.sign(postProInputs.uVcx)
            # [Eqn (4.E4) Page 177 - Book]
            gamma_star = np.sin(postProInputs.gamma)
        else:
            alpha_star = postProInputs.alpha
            gamma_star = postProInputs.gamma

        # For the aligning torque at high slip angles
        signVc = safe_sign(Vc)
        # [Eqn (4.E6a) Page 178 - Book] [sign(Vc) term explained on page 177]
        Vc_prime = Vc + self.epsilon * signVc

        # [Eqn (4.E6) Page 177 - Book]
        alpha_prime = np.arccos(postProInputs.uVcx / Vc_prime)

        # Slippery surface with friction decaying with increasing (slip) speed
        # [Eqn (4.E7) Page 179 - Book]
        LMUX_star = self.LMUX / (1 + self.LMUV * Vs / self.LONGVL)
        # [Eqn (4.E7) Page 179 - Book]
        LMUY_star = self.LMUY / (1 + self.LMUV * Vs / self.LONGVL)

        # Digressive friction factor
        # On Page 179 of the book is suggested Amu = 10, but after
        # comparing the use of the scaling factors against TNO, Amu = 1
        # was giving perfect match
        Amu = 1
        # [Eqn (4.E8) Page 179 - Book]
        LMUX_prime = Amu * LMUX_star / (1 + (Amu - 1) * LMUX_star) # this literally does nothing
        # [Eqn (4.E8) Page 179 - Book]
        LMUY_prime = Amu * LMUY_star / (1 + (Amu - 1) * LMUY_star)

        return alpha_star, gamma_star, LMUX_star, LMUY_star, Fz0_prime, alpha_prime, LMUX_prime, LMUY_prime, dfz, dpi

    def calculateFx0(self, postProInputs: PostProInputs, reductionSmooth, modes: Mode, LMUX_star, LMUX_prime, dfz, dpi):
        Fz, kappa, gamma, Vx = postProInputs.Fz, postProInputs.kappa, postProInputs.gamma, postProInputs.uVcx

        if modes.useTurnSlip:
            Bxp = self.PDXP1 * (1 + self.PDXP2 * dfz) * np.cos(np.arctan(self.PDXP3 * kappa))  # [Eqn (4.106) Page 188 - Book]
            # [Eqn (4.105) Page 188 - Book]
            zeta1 = np.cos(np.arctan(Bxp * self.UNLOADED_RADIUS * postProInputs.phi))
        else:
            zeta1 = 1.0

        Cx = self.PCX1 * self.LCX  # (> 0) (4.E11)
        mux = (self.PDX1 + self.PDX2 * dfz) * (1 + self.PPX3 * dpi + self.PPX4 * dpi**2) * (1 - self.PDX3 * gamma**2) * LMUX_star  # (4.E13)
        
        if type(mux) is np.ndarray: # Zero Fz correction
            mux[Fz == 0] = 0
        elif Fz == 0:
            mux = 0
        Dx = mux * Fz * zeta1  # (> 0) (4.E12)
        Kxk = Fz * (self.PKX1 + self.PKX2 * dfz) * np.exp(self.PKX3 * dfz) * (1 + self.PPX1 * dpi + self.PPX2 * dpi**2) * self.LKX  # (= BxCxDx = dFxo / dkx at kappax = 0) (= Cfk) (4.E15)

        # If [Dx = 0] then [sign(0) = 0]. This is done to avoid [Kxk / 0 = NaN] in Eqn 4.E16
        signDx = safe_sign(Dx)

        # (4.E16) [sign(Dx) term explained on page 177]
        Bx = Kxk / (Cx * Dx + self.epsilon * signDx)
        SHx = (self.PHX1 + self.PHX2 * dfz) * self.LHX  # (4.E17)
        SVx = Fz * (self.PVX1 + self.PVX2 * dfz) * self.LVX * LMUX_prime * zeta1  # (4.E18)

        if type(modes.isLowSpeed) is np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:  # Low speed model
            SVx[modes.isLowSpeed] = SVx[modes.isLowSpeed] * reductionSmooth
            SHx[modes.isLowSpeed] = SHx[modes.isLowSpeed] * reductionSmooth

        kappax = kappa + SHx  # (4.E10)

        if modes.userDynamics == 1:  # Only in Linear Transients mode
            kappax[Vx < 0] = -kappax[Vx < 0]

        Ex = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz**2) * (1 - self.PEX4 * np.sign(kappax)) * self.LEX  # (<=1) (4.E14)

        if modes.useLimitsCheck:
            if np.any[Ex > 1]:
                print('Ex over limit (>1), Eqn(4.E14)')
        
        if type(Ex) is np.ndarray: # Zero Fz correction
            Ex[Ex > 1] = 1
        elif Ex > 1:
            Ex = 1

        # Pure longitudinal force
        Fx0 = Dx * np.sin(Cx * np.arctan(Bx * kappax - Ex * (Bx * kappax - np.arctan(Bx * kappax)))) + SVx  # (4.E9)

        if modes.userDynamics != 2:  # Backward speed check
            if type(signDx) is np.ndarray:
                Fx0[Vx < 0] = -Fx0[Vx < 0]
            elif Vx < 0:
                Fx0 = -Fx0

        return Fx0, mux, Kxk

    def calculateFy0(self, postProInputs: PostProInputs, modes: Mode, reductionSmooth, alpha_star, gamma_star, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi):
        # Turn slip
        if modes.useTurnSlip:
            r_0 = self.UNLOADED_RADIUS  # Free tyre radius

            alpha = postProInputs.alpha
            phi = postProInputs.phi

            # [Eqn (4.79) Page 185 - Book]
            zeta3 = np.cos(np.arctan(self.PKYP1 * r_0**2 * phi**2))

            # [Eqn (4.78) Page 185 - Book]
            Byp = self.PDYP1 * (1 + self.PDYP2 * dfz) * np.cos(np.arctan(self.PDYP3 * np.tan(alpha)))

            # [Eqn (4.77) Page 184 - Book]
            zeta2 = np.cos(np.arctan(Byp * (r_0 * np.abs(phi) + self.PDYP4 * np.sqrt(r_0 * np.abs(phi)))))
        else:
            # No turn slip and small camber angles
            # First paragraph on page 178 of the book
            zeta2 = 1.0
            zeta3 = 1.0

        Kya = self.PKY1 * Fz0_prime * (1 + self.PPY1 * dpi) * (1 - self.PKY3 * np.abs(gamma_star)) * np.sin(self.PKY4 * np.arctan((postProInputs.Fz / Fz0_prime) / (
            (self.PKY2 + self.PKY5 * gamma_star**2) * (1 + self.PPY2 * dpi)))) * zeta3 * self.LKY  # (= ByCyDy = dFyo / dalphay at alphay = 0) (if gamma =0: =Kya0 = CFa) (PKY4=2)(4.E25)
        SVyg = postProInputs.Fz * (self.PVY3 + self.PVY4 * dfz) * gamma_star * self.LKYC * LMUY_prime * zeta2  # (4.E28)

        # MF6.1 and 6.2 equations
        # (=dFyo / dgamma at alpha = gamma = 0) (= CFgamma) (4.E30)
        Kyg0 = postProInputs.Fz * (self.PKY6 + self.PKY7 * dfz) * (1 + self.PPY5 * dpi) * self.LKYC

        # (4.E39) [sign(Kya) term explained on page 177]
        Kya_prime = Kya + self.epsilon * safe_sign(Kya)

        if modes.useTurnSlip:
            # this equation below seems very odd
            Kya0 = self.PKY1 * Fz0_prime * (1 + self.PPY1 * dpi) * np.sin(self.PKY4 * np.arctan(
                (postProInputs.Fz / Fz0_prime) / (self.PKY2 * (1 + self.PPY2 * dpi)))) * zeta3 * self.LKY

            # IMPORTANT NOTE: Explanation of the above equation, Kya0
            # Kya0 is the cornering stiffness when the camber angle is zero
            # (gamma=0) which is again the product of the coefficients By, Cy and
            # Dy at zero camber angle. Information from Kaustub Ragunathan, email:
            # carmaker-service-uk@ipg-automotive.com

            
            # epsilonk is a small factor added to avoid the singularity condition during zero velocity (equation 308, CarMaker reference Manual).
            Kyao_prime = Kya0 + self.epsilon * safe_sign(Kya0)

            CHyp = self.PHYP1  # (>0) [Eqn (4.85) Page 186 - Book]
            DHyp = (self.PHYP2 + self.PHYP3 * dfz) * np.sign(postProInputs.uVcx)  # [Eqn (4.86) Page 186 - Book]
            EHyp = self.PHYP4  # (<=1) [Eqn (4.87) Page 186 - Book]

            if modes.useLimitsCheck:
                if EHyp > 1:
                    print('EHyp over limit (>1), Eqn(4.87)')
            EHyp = min(EHyp, 1)
            KyRp0 = Kyg0 / (1-self.epsilon)  # Eqn (4.89)
            # [Eqn (4.88) Page 186 - Book]
            BHyp = KyRp0 / (CHyp * DHyp * Kyao_prime)
            phi_term = BHyp * self.UNLOADED_RADIUS * postProInputs.phi
            SHyp = DHyp * np.sin(CHyp * np.arctan(phi_term - EHyp * (phi_term - np.arctan(phi_term)))) * np.sign(postProInputs.uVcx)  # [Eqn (4.80) Page 185 - Book]

            zeta4 = 1 + SHyp - SVyg / Kya_prime  # [Eqn (4.84) Page 186 - Book]

            SHy = (self.PHY1 + self.PHY2 * dfz) * self.LHY + zeta4 - 1  # (4.E27) [sign(Kya) term explained on page 177]
        else:
            # No turn slip and small camber angles
            # First paragraph on page 178 of the book
            SHy = (self.PHY1 + self.PHY2 * dfz) * self.LHY + ((Kyg0 * gamma_star - SVyg) / Kya_prime)  # (4.E27) [sign(Kya) term explained on page 177]

        SVy = postProInputs.Fz * (self.PVY1 + self.PVY2 * dfz) * self.LVY * LMUY_prime * zeta2 + SVyg  # (4.E29)

        # Low speed model
        if type(modes.isLowSpeed) is np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:
            SVy[modes.isLowSpeed] = SVy[modes.isLowSpeed] * reductionSmooth
            SHy[modes.isLowSpeed] = SHy[modes.isLowSpeed] * reductionSmooth

        alphay = alpha_star + SHy  # (4.E20)
        Cy = self.PCY1 * self.LCY  # (> 0) (4.E21)
        muy = (self.PDY1 + self.PDY2 * dfz) * (1 + self.PPY3 * dpi + self.PPY4 * dpi**2) * (1 - self.PDY3 * gamma_star**2) * LMUY_star  # (4.E23)
        Dy = muy * postProInputs.Fz * zeta2  # (4.E22)
        Ey = (self.PEY1 + self.PEY2 * dfz) * (1 + self.PEY5 * gamma_star**2 - (self.PEY3 + self.PEY4 * gamma_star) * safe_sign(alphay)) * self.LEY  # (<=1)(4.E24)

        # Limits check
        if modes.useLimitsCheck:
            if np.any(Ey > 1):
                print('Ey over limit (>1), Eqn(4.E24)')
        
        if type(Ey) is np.ndarray: # Zero Fz correction
            Ey[Ey > 1] = 1
        elif Ey > 1:
            Ey = 1

        # (4.E26) [sign(Dy) term explained on page 177]
        By = Kya / (Cy * Dy + self.epsilon * safe_sign(Dy))

        Fy0 = Dy * np.sin(Cy * np.arctan(By * alphay - Ey * (By * alphay - np.arctan(By * alphay)))) + SVy  # (4.E19)

        # Backward speed check for alpha_star
        if modes.useAlphaStar:
            Fy0[postProInputs.uVcx < 0] = -Fy0[postProInputs.uVcx < 0]

        # Zero Fz correction
        if type(muy) is np.ndarray: # Zero Fz correction
            muy[postProInputs.Fz == 0] = 0
        elif postProInputs.Fz == 0:
            muy = 0

        return Fy0, muy, Kya, Kyg0, SHy, SVy, By, Cy, zeta2

    def calculateMz0(self, postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, Kya, SHy, SVy, By, Cy, zeta2):
        if modes.useLimitsCheck:
            Fz = postProInputs.Fz_lowLimit
            # Set Fz to zero if the input is negative
            Fz[postProInputs.Fz <= 0] = 0
        else:
            Fz = postProInputs.Fz

        r_0 = self.UNLOADED_RADIUS  # Free tyre radius

        SHt = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 + self.QHZ4 * dfz) * gamma_star  # (4.E35)

        signKya = safe_sign(Kya)

        # (4.E39) [sign(Kya) term explained on page 177]
        Kya_prime = Kya + self.epsilon * signKya
        SHf = SHy + SVy / Kya_prime  # (4.E38)
        alphar = alpha_star + SHf  # = alphaf (4.E37)
        alphat = alpha_star + SHt  # (4.E34)

        if modes.useTurnSlip:
            # [Eqn (4.91) Page 186 - Book]
            zeta5 = np.cos(np.arctan(self.QDTP1 * r_0 * postProInputs.phi))
        else:
            zeta5 = 1.0

        # Dt0 = Fz * (R0 / Fz0_prime) * (QDZ1 + QDZ2 * dfz) * (1 - PPZ1 * dpi) *  LTR * sign(Vcx) # (4.E42)
        # Dt = Dt0 * (1 + QDZ3 * abs(gamma_star) + QDZ4 * gamma_star**2) * zeta5 # (4.E43)
        #
        # IMPORTANT NOTE: The above original equation (4.E43) was not matching the
        # TNO solver. The coefficient Dt affects the pneumatic trail (t) and the
        # self aligning torque (Mz).
        # It was observed that when negative inclination angles where used as
        # inputs, there was a discrepancy between the TNO solver and mfeval.
        # This difference comes from the term QDZ3, that in the original equation
        # is multiplied by abs(gamma_star). But in the paper the equation is
        # different and the abs() term is not written. Equation (A60) from the
        # paper resulted into a perfect match with TNO.
        # Keep in mind that the equations from the paper don't include turn slip
        # effects. The term zeta5 has been added although it doesn't appear in the
        # paper.

        # Paper definition:
        Dt = (self.QDZ1 + self.QDZ2 * dfz) * (1 - self.PPZ1 * dpi) * (1 + self.QDZ3 * postProInputs.gamma + self.QDZ4 * postProInputs.gamma**2) * Fz * (r_0 / Fz0_prime) * self.LTR * zeta5  # (A60)

        # Bt = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz**2) * (1 + QBZ5 * abs(gamma_star) + QBZ6 * gamma_star**2) * LKY / LMUY_star #(> 0)(4.E40)
        #
        # IMPORTANT NOTE: In the above original equation (4.E40) it is used the
        # parameter QBZ6, which doesn't exist in the standard TIR files. Also note
        # that on page 190 and 615 of the book a full set of parameters is given
        # and QBZ6 doesn't appear.
        # The equation has been replaced with equation (A58) from the paper.

        # Paper definition:
        Bt = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz**2) * (1 + self.QBZ4 * postProInputs.gamma + self.QBZ5 * np.abs(postProInputs.gamma)) * self.LKY / LMUY_star  # (> 0) (A58)
        Ct = self.QCZ1  # (> 0) (4.E41)
        Et = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz**2) * (1 + (self.QEZ4 + self.QEZ5 * gamma_star) * (2 / np.pi) * np.arctan(Bt * Ct * alphat))  # (<=1) (4.E44)

        if modes.useLimitsCheck:  # Limits check
            if np.any(Et > 1):
                print('Et over limit (>1), Eqn(4.E44)')
        if type(Et) is np.ndarray: # Zero Fz correction
            Et[Et > 1] = 1
        elif Et > 1:
            Et = 1

        t0 = Dt * np.cos(Ct * np.arctan(Bt * alphat - Et * (Bt * alphat - np.arctan(Bt * alphat)))) * np.cos(alpha_prime)  # t(aplhat)(4.E33)

        # Evaluate Fy0 with gamma = 0 and phit = 0
        modes_sub0 = modes
        modes_sub0.useTurnSlip = False

        postProInputs_sub0 = postProInputs
        postProInputs_sub0.gamma = 0.0

        Fyo_sub0, _, _, _, _, _, _, _, _ = self.calculateFy0(postProInputs_sub0, modes_sub0, reductionSmooth, alpha_star, 0.0, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        Mzo_prime = -t0 * Fyo_sub0  # gamma=phi=0 (4.E32)

        if modes.useTurnSlip:
            zeta0 = 0.0

            # [Eqn (4.102) Page 188 - Book]
            zeta6 = np.cos(np.arctan(self.QBRP1 * r_0 * postProInputs.phi))

            Fy0, muy, _, _, _, _, _, _, _ = self.calculateFy0(postProInputs, modes, reductionSmooth, alpha_star, 0.0, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

            Mzp_inf = self.QCRP1 * np.abs(muy) * r_0 * Fz * np.sqrt(Fz / Fz0_prime) * self.LMP  # [Eqn (4.95) Page 187 - Book]

            Mzp_inf[Mzp_inf < 0] = 1e-6  # Mzp_inf should be always > 0

            CDrp = self.QDRP1  # (>0) [Eqn (4.96) Page 187 - Book]
            # [Eqn (4.94) Page 187 - Book]
            DDrp = Mzp_inf / np.sin(0.5 * np.pi * CDrp)
            Kzgr0 = Fz * r_0 * (self.QDZ8 * self.QDZ9 * dfz + (self.QDZ10 + self.QDZ11 * dfz * np.abs(postProInputs.gamma))) * self.LKZC  # [Eqn (4.99) Page 187 - Book]

            # Eqn from the manual
            BDrp = Kzgr0 / (CDrp * DDrp * (1 - self.epsilon))
            # Eqn from the manual
            Drp = DDrp * np.sin(CDrp * np.arctan(BDrp * r_0 * postProInputs.phit))

            _, Gyk, _ = self.calculateFy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0, muy, zeta2)

            Mzp90 = Mzp_inf * (2 / np.pi) * np.arctan(self.QCRP2 * r_0 * np.abs(postProInputs.phit)) * Gyk  # [Eqn (4.103) Page 188 - Book]

            zeta7 = (2 / np.pi) * np.arccos(Mzp90 / np.abs(Drp))  # Eqn from the manual
            zeta8 = 1 + Drp
        else:
            zeta0 = 1.0
            zeta6 = 1.0
            zeta7 = 1.0
            zeta8 = 1.0

        Dr = Fz * r_0 * ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES * zeta2 + ((self.QDZ8 + self.QDZ9 * dfz) * (1 + self.PPZ2 * dpi) + (self.QDZ10 +
                         self.QDZ11 * dfz) * np.abs(gamma_star)) * gamma_star * self.LKZC * zeta0) * LMUY_star * np.sign(postProInputs.uVcx) * np.cos(alpha_star) + zeta8 - 1  # (4.E47)
        Br = (self.QBZ9 * self.LKY / LMUY_star + self.QBZ10 * By * Cy) * zeta6  # preferred: qBz9 = 0 (4.E45)
        Cr = zeta7  # (4.E46)
        Mzr0 = Dr * np.cos(Cr * np.arctan(Br * alphar)) * np.cos(alpha_prime)  # =Mzr(alphar)(4.E36)
        Mz0 = Mzo_prime + Mzr0  # (4.E31)

        return Mz0, alphar, alphat, Dr, Cr, Br, Dt, Ct, Bt, Et, Kya_prime

    def calculateFx(self, postProInputs: PostProInputs, modes: Mode, alpha_star, gamma_star, dfz, Fx0):
        Cxa = self.RCX1  # (4.E55)
        Exa = self.REX1 + self.REX2 * dfz  # (<= 1) (4.E56)

        # Limits check
        if modes.useLimitsCheck:
            if np.any(Exa > 1):
                print('Exa over limit (>1), Eqn(4.E56)')
        if type(Exa) is np.ndarray: # Zero Fz correction
            Exa[Exa > 1] = 1
        elif Exa > 1:
            Exa = 1

        Bxa = (self.RBX1 + self.RBX3 * gamma_star**2) * np.cos(np.arctan(self.RBX2 * postProInputs.kappa)) * self.LXAL  # (> 0) (4.E54)

        alphas = alpha_star + self.RHX1  # (4.E53)

        Gxa0 = np.cos(Cxa * np.arctan(Bxa * self.RHX1 - Exa * (Bxa * self.RHX1 - np.arctan(Bxa * self.RHX1))))  # (4.E52)
        Gxa = np.cos(Cxa * np.arctan(Bxa * alphas - Exa * (Bxa * alphas - np.arctan(Bxa * alphas)))) / Gxa0  # (> 0)(4.E51

        Fx = Gxa * Fx0  # (4.E50)

        return Fx

    def calculateFy(self, postProInputs: PostProInputs, reductionSmooth, modes: Mode, alpha_star, gamma_star, dfz, Fy0, muy, zeta2):
        DVyk = muy * postProInputs.Fz * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma_star) * np.cos(np.arctan(self.RVY4 * alpha_star)) * zeta2  # (4.E67)
        SVyk = DVyk * np.sin(self.RVY5 * np.arctan(self.RVY6 * postProInputs.kappa)) * self.LVYKA  # (4.E66)
        SHyk = self.RHY1 + self.RHY2 * dfz  # (4.E65)
        Eyk = self.REY1 + self.REY2 * dfz  # (<=1) (4.E64)

        if modes.useLimitsCheck:  # Limits check
            if np.any(Eyk > 1):
                print('Eyk over limit (>1), Eqn(4.E64)')
        if type(Eyk) is np.ndarray: # Zero Fz correction
            Eyk[Eyk > 1] = 1
        elif Eyk > 1:
            Eyk = 1

        Cyk = self.RCY1  # (4.E63)
        Byk = (self.RBY1 + self.RBY4 * gamma_star**2) * np.cos(np.arctan(self.RBY2 * (alpha_star - self.RBY3))) * self.LYKA  # (> 0) (4.E62)
        kappas = postProInputs.kappa + SHyk  # (4.E61)

        Gyk0 = np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk))))  # (4.E60)
        Gyk = np.cos(Cyk * np.arctan(Byk * kappas - Eyk * (Byk * kappas - np.arctan(Byk * kappas)))) / Gyk0  # (> 0)(4.E59)

        if type(modes.isLowSpeed) is np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:  # If we are using the lowspeed mode and there are any lowspeed points we need to apply the reduction
            SVyk[modes.isLowSpeed] = SVyk[modes.isLowSpeed] * reductionSmooth

        Fy = Gyk * Fy0 + SVyk  # (4.E58)

        return Fy, Gyk, SVyk

    def calculateMx(self, postProInputs: PostProInputs, dpi, Fy):
        Fz = postProInputs.Fz
        gamma = postProInputs.gamma

        # Empirically discovered:
        # If Fz is below FzMin a reduction factor is applied:
        if self.FZMIN > 0:
            reduction_lowFz = Fz * (Fz / self.FZMIN)**2
            Fz[Fz < self.FZMIN] = np.real(reduction_lowFz[Fz < self.FZMIN])

        r_0 = self.UNLOADED_RADIUS  # Free tyre radius
        Mx = 0
        if (abs(self.QSX12) + abs(self.QSX13) + abs(self.QSX14)) != 0 and (abs(self.QSX1) + abs(self.QSX2) + abs(self.QSX3) + abs(self.QSX4) + abs(self.QSX5) + abs(self.QSX6) + abs(self.QSX7) + abs(self.QSX8) + abs(self.QSX9) + abs(self.QSX10) + abs(self.QSX11)) != 0:
            # Draft paper definition:
            Mx = r_0 * Fz * self.LMX * (self.QSX1 * self.LVMX - self.QSX2 * gamma * (1 + self.PPMX1 * dpi) - self.QSX12 * gamma * np.abs(gamma) + self.QSX3 * Fy / self.FNOMIN +
                                        self.QSX4 * np.cos(self.QSX5 * np.arctan((self.QSX6 * Fz / self.FNOMIN) ** 2)) * np.sin(self.QSX7 * gamma + self.QSX8 * np.arctan(self.QSX9 * Fy / self.FNOMIN)) +
                                        self.QSX10 * np.arctan(self.QSX11 * Fz / self.FNOMIN) * gamma) + r_0 * Fy * self.LMX * (self.QSX13 + self.QSX14 * np.abs(gamma))  # (49)
            # print('The parameter sets QSX1 to QSX11 and QSX12 to QSX14 cannot be both non-zero')

            # This is total bullshit and you need the second set of params to for Mx to be useful at all - Ian
            # QSX13 and 14 help define the static Mx from camber

            # IMPORTANT NOTE: Is not recommend to use both sets of
            # parameters (QSX1 to QSX11 and QSX12 to QSX14), so a warning
            # flag will appear.
            # However if this is the case, I found that equation (49)
            # described in the draft paper of Besselink (Not the official
            # paper) will match the TNO solver output. This draft can be
            # downloaded from:
            #
            # https://pure.tue.nl/ws/files/3139488/677330157969510.pdf
            # purl.tue.nl/677330157969510.pdf

        else:
            # # Book definition
            # Mx = R0 * Fz * (QSX1 * LVMX - QSX2 * gamma * (1 + PPMX1 * dpi) + QSX3 * ((Fy)/self.FNOMIN)...
            #     + QSX4 * np.cos(QSX5 * atan((QSX6 * (Fz / self.FNOMIN))**2)) * sin(QSX7 * gamma + QSX8 * atan...
            #     (QSX9 * ((Fy)/self.FNOMIN))) + QSX10 * atan(QSX11 * (Fz / self.FNOMIN)) * gamma) * LMX #(4.E69)
            # IMPORTANT NOTE: The above book equation (4.E69) is not used
            # because it does not contain the parameters QSX12 to QSX14.
            # Instead I have coded the equation from the TNO Equation
            # Manual to match the TNO results.

            # TNO Equation Manual definition
            Mx = r_0 * Fz * self.LMX * (self.QSX1 * self.LVMX - self.QSX2 * gamma * (1 + self.PPMX1 * dpi) + self.QSX3 * (Fy / self.FNOMIN)
                                        + self.QSX4 * np.cos(self.QSX5 * np.arctan((self.QSX6 * (Fz / self.FNOMIN))**2)) * np.sin(
                                            self.QSX7 * gamma + self.QSX8 * np.arctan(self.QSX9 * (Fy / self.FNOMIN)))
                                        + self.QSX10 * np.arctan(self.QSX11 * (Fz / self.FNOMIN)) * gamma) + r_0 * self.LMX * (Fy * (self.QSX13 + self.QSX14 * np.abs(gamma))-Fz * self.QSX12 * gamma * np.abs(gamma))

        return Mx

    def calculateMy(self, postProInputs: PostProInputs, Fx):
        Fz_unlimited, Vcx, kappa, gamma, p = postProInputs.uFz, postProInputs.uVcx, postProInputs.ukappa, postProInputs.gamma, postProInputs.p

        # Empirically discovered:
        # If Fz is below FzMin a reduction factor is applied:
        if self.FZMIN > 0:
            reduction_lowFz = Fz_unlimited * (Fz_unlimited / self.FZMIN)
            Fz_unlimited[Fz_unlimited < self.FZMIN] = np.real(reduction_lowFz[Fz_unlimited < self.FZMIN])


        # My = Fz.R0*(QSY1 + QSY2 * (Fx / self.FNOMIN) + QSY3 * abs(Vcx / self.LONGVL) + QSY4 * (Vcx / self.LONGVL)**4 ...
        #     +(QSY5 + QSY6 * (Fz / self.FNOMIN)) * gamma**2) * ((Fz / self.FNOMIN)**QSY7 * (p / pi0)**QSY8) * LMY. #(4.E70)
        #
        # IMPORTANT NOTE: The above equation from the book (4.E70) is not used
        # because is not matching the results of the official TNO solver.
        # This equation gives a positive output of rolling resistance, and in the
        # ISO coordinate system, My should be negative. Furthermore, the equation
        # from the book has an error, multiplying all the equation by Fz instead of
        # self.FNOMIN (first term).
        # Because of the previous issues it has been used the equation (A48) from
        # the paper.

        # Check MF version
        if self.FITTYP == 6 or self.FITTYP == 21:
            # MF5.2 equations
            My = -self.UNLOADED_RADIUS * Fz_unlimited * self.LMY * (self.QSY1 + self.QSY2 * (Fx / self.FNOMIN) + self.QSY3 * np.abs(
                Vcx / self.LONGVL) + self.QSY4 * (Vcx / self.LONGVL)**4)  # From the MF-Tyre equation manual
        else:
            # MF6.1 and MF6.2 equations
            # Paper definition:
            My = -self.UNLOADED_RADIUS * self.FNOMIN * self.LMY * (self.QSY1 + self.QSY2 * (Fx / self.FNOMIN) + self.QSY3 * np.abs(Vcx / self.LONGVL) + self.QSY4 * (Vcx / self.LONGVL)**4
                                                                   + (self.QSY5 + self.QSY6 * (Fz_unlimited / self.FNOMIN)) * gamma**2) * ((Fz_unlimited / self.FNOMIN)**self.QSY7 * (p / self.NOMPRES)**self.QSY8)  # (A48)

        # Backward speed check
        if type(My) is np.ndarray: # Zero Fz correction
            My[Vcx < 0] = -My[Vcx < 0]
        elif Vcx < 0:
            My = -My

        # Low speed model (Empirically discovered)
        highLimit = self.VXLOW / np.abs(Vcx) - 1
        lowLimit = -1 - self.VXLOW - highLimit
        idx = np.argwhere((kappa >= lowLimit) & (kappa <= highLimit))[:, 0]
        if idx.shape[0] > 0:
            # Points for the interpolation
            x = kappa[idx]
            x1 = highLimit[idx]
            y1 = np.ones(x.shape[0]) * np.pi/2
            x0 = -(np.ones(x.shape[0]))
            y0 = np.zeros(x.shape[0])
            # Call the interpolation function
            reduction = interpolate(x0, y0, x1, y1, x)
            # Reduce My values
            My[idx] = My[idx] * np.sin(reduction)

        # Negative SR check
        if type(My) is np.ndarray: # Zero Fz correction
            My[kappa < lowLimit] = -My[kappa < lowLimit]
        elif kappa < lowLimit:
            My = -My

        # Zero speed check
        if type(My) is np.ndarray: # Zero Fz correction
            My[Vcx == 0] = 0
        elif Vcx == 0:
            My = 0
        
        return My

    def calculateMz(self, postProInputs: PostProInputs, reductionSmooth, modes: Mode, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, alphar, alphat, Kxk, Kya_prime, Fy, Fx, Dr, Cr, Br, Dt, Ct, Bt, Et, SVyk, zeta2):
        kappa, gamma = postProInputs.kappa, postProInputs.gamma

        # alphar_eq = sqrt(alphar**2+(Kxk / Kya_prime)**2 * kappa**2) * sign(alphar) # (4.E78)
        # alphat_eq = sqrt(alphat**2+(Kxk / Kya_prime)**2 * kappa**2) * sign(alphat) # (4.E77)
        # s = R0 * (SSZ1 + SSZ2 * (Fy / Fz0_prime) + (SSZ3 + SSZ4 * dfz) * gamma_star) * LS # (4.E76)

        # IMPORTANT NOTE: The equations 4.E78 and 4.E77 are not used due to small
        # differences discovered at negative camber angles with the TNO solver.
        # Instead equations A54 and A55 from the paper are used.
        #
        # IMPORTANT NOTE: The coefficient "s" (Equation 4.E76) determines the
        # effect of Fx into Mz. The book uses "Fz0_prime" in the formulation,
        # but the paper uses "Fz0". The equation (A56) from the paper has a better
        # correlation with TNO.
        alphar_eq = np.arctan(np.sqrt(np.tan(alphar)**2 + (Kxk / Kya_prime)**2 * kappa**2)) * np.sign(alphar)  # (A54)
        alphat_eq = np.arctan(np.sqrt(np.tan(alphat)**2 + (Kxk / Kya_prime)**2 * kappa**2)) * np.sign(alphat)  # (A55)
        s = self.UNLOADED_RADIUS * (self.SSZ1 + self.SSZ2 * (Fy / self.FNOMIN) + (self.SSZ3 + self.SSZ4 * dfz) * gamma) * self.LS  # (A56)
        Mzr = Dr * np.cos(Cr * np.arctan(Br * alphar_eq))  # (4.E75)

        # Evaluate Fy and Fy0 with gamma = 0 and phit = 0
        postProInputs_sub0 = postProInputs
        postProInputs_sub0.gamma = 0.0

        # Evaluate Fy0 with gamma = 0 and phit  = 0
        Fy0_sub0, muy_sub0, _, _, _, _, _, _, _ = self.calculateFy0(postProInputs_sub0, modes, reductionSmooth, alpha_star, 0.0, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        # Evaluate Gyk with phit = 0 (Note: needs to take gamma into
        # account to match TNO)
        _, Gyk_sub0, _ = self.calculateFy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0_sub0, muy_sub0, zeta2)

        # Note: in the above equation starVar is used instead of
        # starVar_sub0 because it was found a better match with TNO

        Fy_prime = Gyk_sub0 * Fy0_sub0  # (4.E74)
        t = Dt * np.cos(Ct * np.arctan(Bt * alphat_eq - Et * (Bt * alphat_eq - np.arctan(Bt * alphat_eq)))) * np.cos(alpha_prime) * self.LFZO  # (4.E73)

        # IMPORTANT NOTE: the above equation does not contain LFZO in any written source, but "t"
        # is multiplied by LFZO in the TNO dteval function. This has been empirically discovered.

        if type(modes.isLowSpeed) is np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:
            t[modes.isLowSpeed] = t[modes.isLowSpeed] * reductionSmooth
            Mzr[modes.isLowSpeed] = Mzr[modes.isLowSpeed] * reductionSmooth

        if self.FITTYP == 6 or self.FITTYP == 21:  # Check MF version
            # MF5.2 equations
            # From the MF-Tyre equation manual
            Mz = -t * (Fy-SVyk) + Mzr + s * Fx
        else:
            # MF6.1 and 6.2 equations
            Mz = -t * Fy_prime + Mzr + s * Fx  # (4.E71) & (4.E72)

        return Mz, t, Mzr


    def doForcesAndMoments(self, postProInputs, reductionSmooth, modes):

        alpha_star, gamma_star, LMUX_star, LMUY_star, Fz0_prime, alpha_prime, LMUX_prime, LMUY_prime, dfz, dpi = self.calculateBasic(modes, postProInputs)

        Fx0, mux, Kxk = self.calculateFx0(postProInputs, reductionSmooth, modes, LMUX_star, LMUX_prime, dfz, dpi)

        Fy0, muy, Kya, _, SHy, SVy, By, Cy, zeta2 = self.calculateFy0(postProInputs, modes, reductionSmooth, alpha_star, gamma_star, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        _, alphar, alphat, Dr, Cr, Br, Dt, Ct, Bt, Et, Kya_prime = self.calculateMz0(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, Kya, SHy, SVy, By, Cy, zeta2)

        Fx = self.calculateFx(postProInputs, modes, alpha_star, gamma_star, dfz, Fx0)

        Fy, _, SVyk = self.calculateFy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0, muy, zeta2)

        Mx = self.calculateMx(postProInputs, dpi, Fy)

        My = self.calculateMy(postProInputs, Fx)

        Mz, t, Mzr = self.calculateMz(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, alphar, alphat, Kxk, Kya_prime, Fy, Fx, Dr, Cr, Br, Dt, Ct, Bt, Et, SVyk, zeta2)

        forces_and_moments = ForceMoments()
        forces_and_moments.Fx = Fx
        forces_and_moments.Fy = Fy
        forces_and_moments.Fz = postProInputs.Fz
        forces_and_moments.Mx = Mx
        forces_and_moments.My = My
        forces_and_moments.Mz = Mz
        forces_and_moments.dFz = dfz

        varinf = VarInf()
        varinf.Kxk = Kxk
        varinf.mux = mux
        varinf.Kya = Kya
        varinf.muy = muy
        varinf.t = t
        varinf.Mzr = Mzr

        return forces_and_moments, varinf

    def do_forces_and_moments_fast(self, postProInputs, reductionSmooth, modes):

        alpha_star, gamma_star, LMUX_star, LMUY_star, Fz0_prime, alpha_prime, LMUX_prime, LMUY_prime, dfz, dpi = self.calculateBasic(modes, postProInputs)

        Fx0, _, Kxk = self.calculateFx0(postProInputs, reductionSmooth, modes, LMUX_star, LMUX_prime, dfz, dpi)

        Fy0, muy, Kya, _, SHy, SVy, By, Cy, zeta2 = self.calculateFy0(postProInputs, modes, reductionSmooth, alpha_star, gamma_star, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        _, alphar, alphat, Dr, Cr, Br, Dt, Ct, Bt, Et, Kya_prime = self.calculateMz0(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, Kya, SHy, SVy, By, Cy, zeta2)

        Fx = self.calculateFx(postProInputs, modes, alpha_star, gamma_star, dfz, Fx0)

        Fy, _, SVyk = self.calculateFy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0, muy, zeta2)

        Mz, _, _ = self.calculateMz(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, alphar, alphat, Kxk, Kya_prime, Fy, Fx, Dr, Cr, Br, Dt, Ct, Bt, Et, SVyk, zeta2)

        return Fx, Fy, Mz


    def calculateRe(self, postProInputs: PostProInputs, dpi):
        Vcx = postProInputs.uVcx
        Fz_unlimited = postProInputs.uFz
        kappa_unlimited = postProInputs.ukappa

        # Rename the TIR file variables in the Pacejka style
        r_0 = self.UNLOADED_RADIUS  # Free tyre radius
        Cz0 = self.VERTICAL_STIFFNESS  # Vertical stiffness

        omega = postProInputs.omega  # rotational speed (rad/s)
        # [Eqn (1) Page 2 - Paper] - Centrifugal growth of the free tyre radius
        Romega = r_0 * (self.Q_RE0 + self.Q_V1 * ((omega * r_0) / self.LONGVL)**2)
        Re = (r_0*0.965)

        # Excerpt from OpenTIRE MF6.1 implementation
        # Date: 2016-12-01
        # Prepared for Marco Furlan/JLR
        # Questions: henning.olsson@calspan.com

        # Nominal stiffness (pressure corrected)
        # [Eqn (5) Page 2 - Paper] - Vertical stiffness adapted for tyre inflation pressure
        Cz = Cz0 * (1 + self.PFZ1 * dpi)

        # Check if omega is one of the inputs
        # If it is, use it to calculate Re, otherwise it can be approximated with a
        # short iteration
        if postProInputs.nInputs > 7 and np.sum(postProInputs.omega) != 0:
            # Omega is one of the inputs
            # rotational speed (rad/s) [eps is added to avoid Romega = 0]
            omega = postProInputs.omega + 1e-9

            # [Eqn (1) Page 2 - Paper] - Centrifugal growth of the free tyre radius
            Romega = r_0 * (self.Q_RE0 + self.Q_V1 * ((omega * r_0) / self.LONGVL)**2)

            Re = Romega - (self.FNOMIN / Cz) * (self.DREFF * np.arctan(self.BREFF * (Fz_unlimited / self.FNOMIN)) +
                                                self.FREFF * (Fz_unlimited / self.FNOMIN))  # Eff. Roll. Radius [Eqn (7) Page 2 - Paper]
        else:
            # Omega is not specified and is going to be approximated
            # Initial guess of Re based on something slightly less than R0
            Re_old = r_0
            while np.all(np.abs(Re_old - Re) > 1e-9):
                Re_old = Re
                # Use the most up to date Re to calculate an omega
                # omega = Vcx  /  Re # Old definition of Henning without kappa, not valid for brake and drive
                # [Eqn (2.5) Page 65 - Book]
                omega = np.real((kappa_unlimited * Vcx+Vcx) / Re) + 1e-9

                # Then we calculate free-spinning radius [Eqn (1) Page 2 - Paper] - Centrifugal growth of the free tyre radius
                Romega = r_0 * (self.Q_RE0 + self.Q_V1 * ((omega * r_0) / self.LONGVL)**2)

                Re = Romega - (self.FNOMIN / Cz) * (self.DREFF * np.arctan(self.BREFF * (Fz_unlimited / self.FNOMIN)) +
                                                    self.FREFF * (Fz_unlimited / self.FNOMIN))  # Effective Rolling Radius [Eqn (7) Page 2 - Paper]

        return Re, Romega, omega

    def calculateRhoRl(self, postProInputs: PostProInputs, forces_and_moments, dpi, omega, Romega):
        # Calculate the radius deping on the MF version
        if self.FITTYP == 62:
            rho, Rl, Cz = self.calculateRhoRl62(postProInputs, forces_and_moments, dpi, omega, Romega)  # MF6.2
        else:
            rho, Rl, Cz = self.calculateRhoRl61(postProInputs, forces_and_moments, dpi, omega, Romega)  # MF5.2 or MF6.1
        return rho, Rl, Cz

    def calculateRhoRl61(self, postProInputs: PostProInputs, forces_and_moments, dpi, omega, Romega):
        Fz = postProInputs.Fz_lowLimit
        Fx = forces_and_moments.Fx
        Fy = forces_and_moments.Fy

        # Model parameters as QFZ1 that normally aren't present in the TIR files
        # Rearranging [Eqn (4) Page 2 - Paper]
        Q_fsz = np.sqrt((self.VERTICAL_STIFFNESS * self.UNLOADED_RADIUS / self.FNOMIN)**2 - 4 * self.Q_FZ2)

        # Split Eqn (A3.3) Page 619 of the Book into different bits:
        speed_effect = self.Q_V2 * (self.UNLOADED_RADIUS / self.LONGVL) * np.abs(omega)
        fx_effect = (self.Q_FCX * Fx / self.FNOMIN)**2
        fy_effect = (self.Q_FCY * Fy / self.FNOMIN)**2
        pressure_effect = (1 + self.PFZ1 * dpi)

        # Joining all the effects except tyre deflection terms
        external_effects = (1 + speed_effect - fx_effect - fy_effect) * pressure_effect * self.FNOMIN

        # Equation (A3.3) can be written as:
        # Fz = (Q_FZ2*(rho/R0)^2 + Q_fsz*(rho/R0)) * external_effects

        # Rearranging all the terms we up with a quadratic equation as:
        # ax^2 + bx + c = 0
        # Q_FZ2*(rho/R0)^2 + Q_fsz*(rho/R0) -(Fz/(external_effects)) = 0

        # Note: use of capital letters to avoid confusion with contact patch
        # lengths "a" and "b"

        a = self.Q_FZ2
        if a == 0:
            a = 1e-6
        b = Q_fsz
        c = -(Fz / (external_effects))

        if np.all((b**2 - 4 * a * c) > 0):
            x = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        else:
            print('No positive solution for rho calculation')
            x = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # tyre deflection for a free rolling tyre
        rho_zfr = np.maximum(x * self.UNLOADED_RADIUS, 0)

        # The loaded radius is the free-spinning radius minus the deflection
        # Eqn A3.2 Page 619 - Book assuming positive rho at all the time
        Rl = np.maximum(Romega - rho_zfr, 0)

        rho_z = rho_zfr

        rho_z[rho_z == 0] = 1e-6  # Avoid division between zero, in the MFeval version this is [Fx == 0] which i believe to be wrong

        Cz = Fz / rho_z  # Vertical stiffness (Spring)

        return rho_z, Rl, Cz

    def calculateRhoRl62(self, postProInputs: PostProInputs, forces_and_moments, dpi, omega, Romega):
        Fz = postProInputs.uFz
        Fx = forces_and_moments.Fx
        Fy = forces_and_moments.Fy
        gamma = postProInputs.gamma

        # The loaded radius (Rl) cannot be calculated straight forward
        # when the vertical load (Fz) is an input.
        #
        # Here I have written the equation for the vertical load (Fz)
        # of the MF-Tyre 6.2 model that has the loaded radius (Rl) as
        # input. (see calculateFz62 function)
        #
        # The Rl is calculated in an iterative method
        # Note: Method developed by NK

        r_0 = np.ones(omega.shape[0]) * self.UNLOADED_RADIUS
        Fz_tol = self.FNOMIN * 1e-6

        # Solve with the secant method
        Rl = self.solve_secant(r_0 * 0.95, r_0, Fz_tol, gamma, omega, Romega, dpi, Fx, Fy, Fz)
        # Calculate the deflection
        _, rho_z = self.calculateFz62(gamma, omega, Romega, dpi, Rl, Fx, Fy)

        # Avoid division between zero
        rho_z[forces_and_moments.Fz == 0] = 1e-6

        # Vertical stiffness (Spring)
        Cz = forces_and_moments.Fz / rho_z

        return rho_z, Rl, Cz

    def calculateContactPatch(self, postProInputs: PostProInputs, dpi):
        Fz_unlimited = postProInputs.uFz

        # Rename the TIR file variables in the Pacejka style
        r_0 = self.UNLOADED_RADIUS  # Free tyre radius
        w = self.WIDTH       		# Nominal width of the tyre
        Cz0 = self.VERTICAL_STIFFNESS  # Vertical stiffness

        # Nominal stiffness (pressure corrected)
        # [Eqn (5) Page 2 - Paper] - Vertical stiffness adapted for tyre inflation pressure
        NCz = Cz0 * (1 + self.PFZ1 * dpi)

        Rrim = self.RIM_RADIUS  # Nominal rim radius
        Dbtm = self.BOTTOM_OFFST  # Distance to rim when bottoming starts to occur

        # Approximated loaded Radius
        Rl = r_0 - (Fz_unlimited / NCz)

        # Bottoming model (Empirically discovered):
        # Check if bottoming has happened
        isBottoming = Rl-(Rrim + Dbtm) < 0

        # Calculate the max Fz if bottoming happens to calculate the
        # contact patch
        maxFz = (r_0 - Rrim - Dbtm) * NCz

        # Substitute max Fz for the calculations
        Fz_unlimited[isBottoming] = np.real(maxFz[isBottoming])

        if self.FITTYP == 6 or self.FITTYP == 21:  # Check MF version
            # Set default values (Empirically discovered)
            y = np.log10(r_0 * (Cz0 / self.FNOMIN))
            Q_a1 = -0.0388 * y**3 + 0.2509 * y**2 + -0.6283 * y + 0.6279  # MF5.2 Square root load term in contact length
            Q_a2 = 1.693 * Q_a1**2  # MF5.2 Linear load term in contact length

            # From the MF-Tyre equation manual
            a = r_0 * (Q_a2 * (Fz_unlimited / self.FNOMIN) +
                       Q_a1 * np.sqrt(Fz_unlimited / self.FNOMIN))
            b = w/2  # From the MF-Tyre equation manual

        else:
            # MF6.1 and 6.2 equations
            a = r_0 * (self.Q_RA2 * (Fz_unlimited / (NCz * r_0)) + self.Q_RA1 * np.sqrt(Fz_unlimited / (NCz * r_0)))  # [Eqn (9) Page 3 - Paper]
            b = w * (self.Q_RB2 * (Fz_unlimited / (NCz * r_0)) + self.Q_RB1 * (Fz_unlimited / (NCz * r_0))**(1/3))  # [Eqn (10) Page 3 - Paper]

        return a, b, NCz

    def calculateRelax(self, postProInputs: PostProInputs, varinf):
        gamma = postProInputs.gamma
        Fz = postProInputs.Fz
        p = postProInputs.p
        Kxk = varinf.Kxk
        Kya = varinf.Kya

        r_0 = self.UNLOADED_RADIUS  # Free tyre radius
        pi0 = self.NOMPRES  # Reference pressure
        cx0 = self.LONGITUDINAL_STIFFNESS  # Tyre overall longitudinal stiffness
        cy0 = self.LATERAL_STIFFNESS  # Tyre overall lateral stiffness

        Fz0_prime = self.LFZO * self.FNOMIN  # [Eqn (4.E1) Page 177 - Book]
        dfz = (Fz - Fz0_prime) / Fz0_prime  # [Eqn (4.E2a) Page 177 - Book]
        dpi = (p - pi0) / pi0  # [Eqn (4.E2b) Page 177 - Book]

        # Overall longitudinal Cx and lateral stiffness Cy
        Cx = cx0 * (1 + self.PCFX1 * dfz + self.PCFX2 * dfz**2) * (1 + self.PCFX3 * dpi)  # (Eqn 17 - Paper)
        Cy = cy0 * (1 + self.PCFY1 * dfz + self.PCFY2 * dfz**2) * (1 + self.PCFY3 * dpi)  # (Eqn 18 - Paper)

        if self.FITTYP == 6 or self.FITTYP == 21:

            PTX1 = 0  # Relaxation length SigKap0/Fz at Fznom
            PTX2 = 0  # Variation of SigKap0/Fz with load
            PTX3 = 0  # Variation of SigKap0/Fz with exponent of load
            PTY1 = 0  # Peak value of relaxation length SigAlp0/R0
            PTY2 = 0  # Value of Fz/Fznom where SigAlp0 is extreme
            LSGKP = 0  # Scale factor of Relaxation length of Fx
            LSGAL = 0  # Scale factor of Relaxation length of Fy
            PKY3 = 0  # Variation of Kfy/Fznom with camber

            # MF 5.2 equation for the longitudinal relaxation length
            sigmax = (PTX1 + PTX2 * dfz) * np.exp(-PTX3 * dfz) * LSGKP * r_0 * Fz / self.FNOMIN  # From the MF-Tyre equation manual

            # MF 5.2 equations for the lateral relaxation length
            sigmayg = 1-PKY3 * abs(gamma)
            PTYfzn = PTY2 * Fz0_prime
            # From the MF-Tyre equation manual
            sigmay = PTY1 * np.sin(2 * np.arctan(Fz / PTYfzn)) * sigmayg * r_0 * self.LFZO * LSGAL
        else:
            # MF 6.1 and 6.2 equations for the relaxation lengths
            sigmax = abs(Kxk / Cx)  # (Eqn 19 - Paper)
            sigmay = abs(Kya / Cy)  # (Eqn 19 - Paper)

        return Cx, Cy, sigmax, sigmay

    def calculateInstantaneousKya(self, postProInputs: PostProInputs, forces_and_moments: ForceMoments):
        Fy = forces_and_moments.Fy
        s_a = postProInputs.alpha

        if np.size(Fy) > 1:
            # Derivative and appto get same number of elements
            diffFY = np.diff(Fy) / (np.diff(s_a) + 1E-10)
            instKya = np.concatenate((diffFY, [diffFY[-1]]))
        else:
            instKya = 0

        return instKya

    def calculateFz62(self, gamma, omega, Romega, dpi, Rl, Fx, Fy):
        # Calculate the vertical Force using the equations described in
        # the MF-Tyre/MF-Swift 6.2 equation manual Document revision:
        # 20130706

        # Model parameters as QFZ1 that normally aren't present in the TIR files
        # Rearranging [Eqn (4) Page 2 - Paper]
        Q_fsz = np.sqrt((self.VERTICAL_STIFFNESS * self.UNLOADED_RADIUS / self.FNOMIN)**2 - 4 * self.Q_FZ2)

        # Asymmetric effect for combinations of camber and lateral force
        Sfyg = (self.Q_FYS1 + self.Q_FYS2 * (Rl / Romega) + self.Q_FYS3 * (Rl / Romega)**2) * gamma

        # Tyre deflection for a free rolling tyre
        rho_zfr = Romega - Rl
        rho_zfr[rho_zfr < 0] = 0

        # Reference tread width
        rtw = (1.075 - 0.5 * self.ASPECT_RATIO) * self.WIDTH

        # Deflection caused by camber
        rho_zg = ((self.Q_CAM1 * Rl + self.Q_CAM2 * Rl**2) * gamma)**2 * (rtw/8) * np.abs(np.tan(gamma)) / \
            ((self.Q_CAM1 * Romega + self.Q_CAM2 * Romega**2) * gamma)**2 - (self.Q_CAM3 * rho_zfr * np.abs(gamma))

        # Change NaN to Zero
        rho_zg[np.isnan(rho_zg)] = 0

        # Vertical deflection
        rho_z = rho_zfr + rho_zg
        rho_z[rho_z <= 0] = 1e-12

        # Correction term
        fcorr = (1 + self.Q_V2 * (self.UNLOADED_RADIUS / self.LONGVL) * np.abs(omega) - ((self.Q_FCX * Fx) / self.FNOMIN) **
                 2 - ((rho_z / self.UNLOADED_RADIUS)**self.Q_FCY2 * (self.Q_FCY*(Fy - Sfyg) / self.FNOMIN))**2) * (1 + self.PFZ1 * dpi)

        # Vertical force
        Fz = fcorr * (Q_fsz * (rho_z / self.UNLOADED_RADIUS) + self.Q_FZ2 *
                      (rho_z / self.UNLOADED_RADIUS)**2) * self.FNOMIN

        return Fz, rho_z

    def doExtras(self, postProInputs: PostProInputs, forces_and_moments, varinf):

        dpi = (postProInputs.p - self.NOMPRES) / self.NOMPRES  # [Eqn (4.E2b) Page 177 - Book]

        Re, Romega, omega = self.calculateRe(postProInputs, dpi)

        rho, Rl, Cz = self.calculateRhoRl(postProInputs, forces_and_moments, dpi, omega, Romega)

        a, b, _ = self.calculateContactPatch(postProInputs, dpi)

        Cx, Cy, sigmax, sigmay = self.calculateRelax(postProInputs, varinf)

        instKya = self.calculateInstantaneousKya(postProInputs, forces_and_moments)

        return Re, omega, rho, Rl, a, b, Cx, Cy, Cz, sigmax, sigmay, instKya

    def solve_secant(self, x0, x1, ytol, gamma, omega, Romega, dpi, Fx, Fy, Fz):
        # NK: vectorized secant method solver
        # A simple and fast solver using the secant method.
        # Makes a minimum of 3 calls to the function to be zeroed.
        # Assume that the first two guesses aren't perfect
        y0 = self.delta_fz_func(gamma, omega, Romega, dpi, x0, Fx, Fy, Fz)
        y1 = self.delta_fz_func(gamma, omega, Romega, dpi, x1, Fx, Fy, Fz)

        # not converged flag
        nc = np.full(x0.shape, True)
        x = x1

        for calls in range(3, 20):  # to limit number of tries
            # interpolate (or extrapolate) to y == 0
            try_x = x1 - (x1 - x0) / (y1 - y0) * y1

            # only update non-converged x
            x[nc] = try_x[nc]

            # evaluate new guess and check convergence
            y = self.delta_fz_func(gamma, omega, Romega, dpi, x, Fx, Fy, Fz)
            nc = nc & (np.abs(y) > ytol)
            if not np.any(nc):
                # print(f"Secant method failed to converge in {calls} calls")
                return x

            # Keep the best in x0
            x1_best = np.where(np.abs(y1) < np.abs(y0))
            x0[x1_best] = x1[x1_best]
            y0[x1_best] = y1[x1_best]

            # replace x1,y1 for next iteration
            x1 = x
            y1 = y
        return x

    def delta_fz_func(self, gamma, omega, Romega, dpi, try_radius, Fx, Fy, Fz):
        try_Fz, _ = self.calculateFz62(gamma, omega, Romega, dpi, try_radius, Fx, Fy)
        return try_Fz - Fz

    def coefficientCheck(self, param_group=None):
        # COEFFICIENTCHECK Validate that model coefficients pass any restrictions
        # placed on them.
        #
        #   [res, c, vals] = mfeval.coefficientCheck(self)
        #   [res, c, vals] = mfeval.coefficientCheck(self, param_group)
        #
        #   Outputs:
        #
        #   res is a struct of logical results for each coefficient
        #   check where (0 = pass, 1 = fail)
        #
        #   c is a struct of values for which an optimizer must satisfy
        #   c <= 0 to make the coefficient check pass.
        #
        #   vals is a struct of the values for each coefficient check.
        #
        #   Inputs:
        #
        #   mfStruct is structure of Magic Formula parameters
        #
        #   paramGroup is a string defining the Magic Formula parameter
        #   group for which to conduct the coefficient checks for.
        #   Leaving blank will run all.

        if self.FITTYP != 61 or self.FITTYP != 62:
            print('coefficientCheck works only for Magic Formula 6.1 or 6.2 models. The provided tyre model is not compatible with this function')

        inputs = self.generateInputs()

        result = Result()
        values = RetValue()
        c = RetValue()

        # Switch calculation mode between 'all' or 'paramGroup' based on number of
        # inputs
        if param_group is None:
            # If one input, calculate all coefficients
            result.Cx, values.Cx, c.Cx = self.calcCx()
            result.Dx, values.Dx, c.Dx = self.calcDx(inputs)
            result.Ex, values.Ex, c.Ex = self.calcEx(inputs)
            result.Cy, values.Cy, c.Cy = self.calcCy()
            result.Ey, values.Ey, c.Ey = self.calcEy(inputs)
            result.Bt, values.Bt, c.Bt = self.calcBt(inputs)
            result.Ct, values.Ct, c.Ct = self.calcCt()
            result.Et, values.Et, c.Et = self.calcEt(inputs)
            # Note: Some models were not tested under combined
            # loading, therefore some of the combined coefficients were not
            # parametrized. When this is the case, do not evaluate this section
            if self.RBX1 != 0 or self.RBX3 != 0:
                result.Bxa, values.Bxa, c.Bxa = self.calcBxa(inputs)
                result.Exa, values.Exa, c.Exa = self.calcExa(inputs)
                result.Gxa, values.Gxa, c.Gxa = self.calcGxa(inputs)

            if self.RBY1 != 0 or self.RBY3 != 0:
                result.Byk, values.Byk, c.Byk = self.calcByk(inputs)
                result.Eyk, values.Eyk, c.Eyk = self.calcEyk(inputs)
                result.Gyk, values.Gyk, c.Gyk = self.calcGyk(inputs)

        else:
            if param_group == 'FyPure':
                #  Cy, Ey
                result.Cy, values.Cy, c.Cy = self.calcCy()
                result.Ey, values.Ey, c.Ey = self.calcEy(inputs)
            elif param_group == 'FxPure':
                # Cx, Dx, Ex
                result.Cx, values.Cx, c.Cx = self.calcCx()
                result.Dx, values.Dx, c.Dx = self.calcDx(inputs)
                result.Ex, values.Ex, c.Ex = self.calcEx(inputs)
            elif param_group == 'MzPure':
                # Ct, Et, Bt
                result.Bt, values.Bt, c.Bt = self.calcBt(inputs)
                result.Ct, values.Ct, c.Ct = self.calcCt()
                result.Et, values.Et, c.Et = self.calcEt(inputs)
            elif param_group == 'FyComb':
                # Gyk, Eyk, Byk
                result.Byk, values.Byk, c.Byk = self.calcByk(inputs)
                result.Eyk, values.Eyk, c.Eyk = self.calcEyk(inputs)
                result.Gyk, values.Gyk, c.Gyk = self.calcGyk(inputs)
            elif param_group == 'FxComb':
                # Gxa, Exa, Bxa
                result.Bxa, values.Bxa, c.Bxa = self.calcBxa(inputs)
                result.Exa, values.Exa, c.Exa = self.calcExa(inputs)
                result.Gxa, values.Gxa, c.Gxa = self.calcGxa(inputs)
            elif param_group == 'FxPureComb':
                # Cx, Dx, Ex
                result.Cx, values.Cx, c.Cx = self.calcCx()
                result.Dx, values.Dx, c.Dx = self.calcDx(inputs)
                result.Ex, values.Ex, c.Ex = self.calcEx(inputs)
                # Gxa, Exa, Bxa
                result.Bxa, values.Bxa, c.Bxa = self.calcBxa(inputs)
                result.Exa, values.Exa, c.Exa = self.calcExa(inputs)
                result.Gxa, values.Gxa, c.Gxa = self.calcGxa(inputs)
            else:
                result.Nan = False
                values.Nan = 0  # Replace with zero for rsim compatibility
                c.Nan = 0

        return result, values, c, inputs

    def generateInputs(self):
        # To prevent Dx from failing when self.FZMIN = 0, change FzMin to 1.
        # This occurs because Dx = Mux*Fz where Dx must be > 0.
        if self.FZMIN == 0:
            self.FZMIN = 1
        # Generate input parameters based on the tyre model limits
        inputs = InputRanges((np.linspace(self.PRESMIN, self.PRESMAX, 11) - self.NOMPRES) / self.NOMPRES, (np.linspace(self.FZMIN, self.FZMAX, 11) - self.FNOMIN) / self.FNOMIN, np.linspace(self.FZMIN, self.FZMAX, 11), np.linspace(self.PRESMIN, self.PRESMAX, 11), np.linspace(self.KPUMIN, self.KPUMAX, 11), np.linspace(self.ALPMIN, self.ALPMAX, 11), np.linspace(self.CAMMIN, self.CAMMAX, 11))

        return inputs

    def calcCx(self):
        # MF6.1 requirement is that Cx > 0
        # fmincon requirement is that c <= 0
        Cx = self.PCX1 * self.LCX  # EQN 35
        c = -Cx  # Convert Cx to c to satisfy conditions

        return (Cx <= 0), Cx, c

    def calcDx(self, inputs):
        # MF6.1 requirement is that Dx > 0
        # fmincon requirement is that c <= 0
        # Generate input matrix
        dFz, i_a, dPi = np.meshgrid(inputs.dFz, inputs.IA, inputs.dPi)
        Fz = np.tile(inputs.Fz.T, (len(inputs.Fz), len(inputs.Fz), 1))

        Mux = (self.PDX1 + self.PDX2 * dFz) * (1 - self.PDX3 * (i_a**2)) * (1 + self.PPX3 * dPi + self.PPX4 * (dPi**2)) * self.LMUX  # EQN 37
        Dx = Mux * Fz  # EQN 36
        c = -Dx  # Convert Dx to c to satisfy conditions

        return (Dx <= 0), Dx, c

    def calcEx(self, inputs):
        # MF6.1 requirement is that Ex <= 1
        # fmincon requirement is that c <= 0
        dFz, s_r = np.meshgrid(inputs.dFz, inputs.SR)  # Generate input matrix

        SHx = (self.PHX1 + self.PHX2 * dFz) * self.LHX  # EQN 41
        SRx = s_r + SHx  # EQN 34
        Ex = (self.PEX1 + self.PEX2 * dFz + self.PEX3 * (dFz**2)) * (1 - self.PEX4 * np.sign(SRx)) * self.LEX  # EQN 38
        c = Ex - 1  # Convert Dx to c to satisfy conditions

        return (Ex > 1), Ex, c

    def calcCy(self):
        # MF6.1 requirement is that Cy > 0
        # fmincon requirement is that c <= 0
        Cy = self.PCY1 * self.LCY  # EQN 54
        c = -Cy  # Convert Cx to c to satisfy conditions

        return (Cy <= 0), Cy, c

    def calcEy(self, inputs):
        # MF6.1 requirement is that Ey <= 1
        # fmincon requirement is that c <= 0

        # NOTE: EQN 57 takes the sign of the calculated SA, for which this is the
        # only use in this equation. To save computational speed, the sub
        # calculation of SAy is ignored, and a one positive and one negative value
        # are evaluated instead.

        # inputs.SAsgn = [-1, 1]  # Generate input matrix
        # SAsgn, dFz, i_a = np.meshgrid(inputs.SAsgn, inputs.dFz, inputs.IA)
        dFz, i_a = np.meshgrid(inputs.dFz, inputs.IA)
        Ey = (self.PEY1 + self.PEY2 * dFz) * (1 + self.PEY5 * (i_a**2) - (self.PEY3 + self.PEY4 * i_a) * self.LEY)  # EQN 57
        c = Ey - 1  # Convert Cx to c to satisfy conditions

        return (Ey > 1), Ey, c

    def calcBt(self, inputs):
        # MF6.1 requirement is that Bt > 0
        # fmincon requirement is that c <= 0
        dFz, i_a = np.meshgrid(inputs.dFz, inputs.IA)  # Generate input matrix
        Bt = (self.QBZ1 + self.QBZ2 * dFz + self.QBZ3 * (dFz**2)) * (2 + self.QBZ4 + self.QBZ5 * np.abs(i_a)) * (self.LKY / self.LMUY)  # EQN 84
        c = -Bt  # Convert Cx to c to satisfy conditions

        return (Bt <= 0), Bt, c

    def calcCt(self):
        # MF6.1 requirement is that Ct > 0
        # fmincon requirement is that c <= 0

        Ct = self.QCZ1  # EQN 54
        c = -Ct  # Convert Cx to c to satisfy conditions

        return (Ct <= 0), Ct, c

    def calcEt(self, inputs):
        # MF6.1 requirement is that Et <= 1
        # fmincon requirement is that c <= 0
        # Generate input matrix
        dFz, i_a, s_a = np.meshgrid(inputs.dFz, inputs.IA, inputs.SA)

        # Generate required sub calcs
        _, Bt, _ = self.calcBt(inputs)
        Bt = np.tile(Bt, (len(inputs.SA), 1, 1))
        _, Ct, _ = self.calcCt()

        SHt = self.QHZ1 + self.QHZ2 * dFz + (self.QHZ3 + self.QHZ4 * dFz) * i_a  # EQN 77
        SAt = s_a + SHt  # EQN 76
        Et = (self.QEZ1 + self.QEZ2 * dFz + self.QEZ3 * (dFz**2)) * (1 + (self.QEZ4 + self.QEZ5 * i_a) * (2 / np.pi) * (np.arctan(Bt * Ct * SAt)))  # EQN 87
        c = Et - 1  # Convert Dx to c to satisfy conditions

        return (Et > 1), Et, c

    def calcGxa(self, inputs):
        # MF6.1 requirement is that Gxa > 0
        # fmincon requirement is that c <= 0
        _, _, _, s_a = np.meshgrid(inputs.dFz, inputs.SR, inputs.IA, inputs.SA)  # Generate input matrix

        # Generate required sub calcs
        _, Bxa, _ = self.calcBxa(inputs)  # [SR, IA]
        _, Exa, _ = self.calcExa(inputs)  # [dFz]
        Bxa = np.reshape(Bxa, (1, 11, 11))
        Bxa = np.tile(Bxa, (len(inputs.dFz), len(inputs.SA), 1, 1))
        Exa = np.tile(Exa.T, (len(inputs.SA), len(inputs.SR), len(inputs.IA), 1))

        Cxa = self.RCX1  # EQN 46
        SHxa = self.RHX1  # EQN 48
        SAs = s_a + SHxa  # EQN 44
        Gxa = (np.cos(Cxa * np.arctan(Bxa * SAs - Exa * (Bxa * SAs - np.arctan(Bxa * SAs))))) / \
            (np.cos(Cxa * np.arctan(Bxa * SHxa - Exa * (Bxa * SHxa - np.arctan(Bxa * SHxa)))))  # EQN 43
        c = -Gxa  # Convert Dx to c to satisfy conditions

        return (Gxa <= 0), Gxa, c

    def calcBxa(self, inputs):
        # MF6.1 requirement is that Bxa > 0
        # fmincon requirement is that c <= 0
        s_r, i_a = np.meshgrid(inputs.SR, inputs.IA)  # Generate input matrix

        Bxa = (self.RBX1 + self.RBX3 * (i_a**2)) * np.cos(np.arctan(self.RBX2 * s_r)) * self.LXAL  # EQN 45
        c = -Bxa  # Convert Dx to c to satisfy conditions

        return (Bxa <= 0), Bxa, c

    def calcExa(self, inputs):
        # MF6.1 requirement is that Bxa <= 1
        # fmincon requirement is that c <= 0
        dFz = inputs.dFz  # Generate input matrix

        Exa = self.REX1 + self.REX2 * dFz  # EQN 47
        c = Exa - 1  # Convert Dx to c to satisfy conditions

        return (Exa > 1), Exa, c

    def calcGyk(self, inputs):
        # MF6.1 requirement is that Gyk > 0
        # fmincon requirement is that c <= 0
        # Generate input matrix
        dFz, s_r, _, _ = np.meshgrid(inputs.dFz, inputs.SR, inputs.IA, inputs.SA)

        # Generate required sub calcs
        _, Byk, _ = self.calcByk(inputs)  # [SA, IA]
        _, Eyk, _ = self.calcEyk(inputs)  # [dFz]
        Byk = Byk.transpose((1, 0))  # [IA, SA]
        Byk = np.reshape(Byk, (1, 1, len(inputs.IA), len(inputs.SA)))
        Byk = np.tile(Byk, (len(inputs.dFz), len(inputs.SR), 1, 1))
        Eyk = np.tile(Eyk.T, (len(inputs.SA), len(inputs.SR), len(inputs.IA), 1))
        Cyk = self.RCY1  # EQN 72
        SHyk = self.RHY1 + self.RHY2 * dFz  # EQN 74
        SRs = s_r + SHyk  # EQN 70
        Gyk = (np.cos(Cyk * np.arctan(Byk * SRs - Eyk * (Byk * SRs - np.arctan(Byk * SRs))))) / (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))  # EQN 69
        c = -Gyk  # Convert Dx to c to satisfy conditions

        return (Gyk <= 0), Gyk, c

    def calcByk(self, inputs):
        # MF6.1 requirement is that Bxa > 0
        # fmincon requirement is that c <= 0
        s_a, i_a = np.meshgrid(inputs.SA, inputs.IA)  # Generate input matrix
        Byk = (self.RBY1 + self.RBY4*(i_a**2)) * np.cos(np.arctan(self.RBY2 * (s_a - self.RBY3))) * self.LYKA  # EQN 45
        c = -Byk  # Convert Dx to c to satisfy conditions

        return (Byk <= 0), Byk, c

    def calcEyk(self, inputs):
        # MF6.1 requirement is that Eyk <= 1
        # fmincon requirement is that c <= 0
        dFz = inputs.dFz  # Generate input matrix
        Eyk = self.REY1 + self.REY2 * dFz  # EQN 47
        c = Eyk - 1  # Convert Dx to c to satisfy conditions

        return (Eyk > 1), Eyk, c
