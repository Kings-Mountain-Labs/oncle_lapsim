import numpy as np
from .tire_model_utils import MODEL_DEFAULTS, VarInf, PostProInputs, Mode, ForceMoments, set_x
from toolkit.common import safe_sign, interpolate
from .pacejka_coefficients import PacejkaModel
import torch
from torch import Tensor

def tire_model_from_arr(arr):
    tm = DiffMFModel(MODEL_DEFAULTS)
    set_x(arr, tm)
    return tm


class DiffMFModel(PacejkaModel, torch.nn.Module):
    # DiffMFModel Solver for Magic Formula 5.2, 6.1 and 6.2 Tyre Models
    def __init__(self, tire_coefficients: dict) -> None:
        # Parameters not specified in the TIR file
        # Used to avoid low speed singularity
        self.epsilon = 1e-6  # [Eqn (4.E6a) Page 178 - Book]
        super().__init__(tire_coefficients)
        torch.nn.Module.__init__(self)
    

    def steady_state_mmd(self, f_z: float, s_a: float, s_r: float, p: float = 82500, i_a: float = 0, phit: float = 0.0, v: float = 15, omega: float = 0.0, mu_corr: float = 1.0, flip_s_a: bool = False):
        if flip_s_a:
            s_a = -s_a
        postProInputs, reductionSmooth, modes = self.fast_parse_inputs(0, f_z, s_r, s_a, i_a, 0.0, v, p, omega, ncolumns=1)

        Fx, Fy, Mz = self.do_forces_and_moments_fast(postProInputs, reductionSmooth, modes)
        if flip_s_a:
            return Fx.real * mu_corr, -Fy.real * mu_corr, -Mz.real * mu_corr
        return Fx.real * mu_corr, Fy.real * mu_corr, Mz.real * mu_corr

    def fast_parse_inputs(self, userDynamics, Fz, kappa, alpha, gamma, phit, Vcx, p, omega, useLimitsCheck=False, useAlphaStar=False, useTurnSlip=False):

        # IMPORTANT NOTE: Vx = Vcx [Eqn (7.4) Page 331 - Book]
        # It is assumed that the difference between the wheel centre
        # longitudinal velocity Vx and the longitudinal velocity Vcx of
        # the contact centre is negligible
        Fz[Fz < 0] = 0  # If any Fz is negative set it to zero

        # Create a copy of the variables (u stands for unlimited)
        ualpha = alpha

        isLowSpeed = 0.0
        reductionSmooth = 1.0

        modes: Mode = Mode(useLimitsCheck, useAlphaStar, useTurnSlip,
                           isLowSpeed, reductionSmooth, userDynamics)
        ualpha[Vcx == 0] = 0  # Zero speed (empirically discovered)

        post_pro_inputs = PostProInputs(omega, 0.0, 0.0, Fz, kappa, kappa, ualpha, gamma, phit, Vcx, alpha, kappa, gamma, phit, Fz, p, 1, Fz)

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

    def calculate_basic(self, modes: Mode, postProInputs: PostProInputs):
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

    def calculate_fx0(self, modes: Mode, kappa: Tensor, gamma: Tensor, phi: Tensor, Fz: Tensor, is_low_speed: torch.BoolTensor, reductionSmooth: Tensor, LMUX_star: Tensor, LMUX_prime: Tensor, dfz: Tensor, dpi: Tensor):
        if modes.useTurnSlip:
            Bxp = self.PDXP1 * (1 + self.PDXP2 * dfz) * np.cos(np.arctan(self.PDXP3 * kappa))  # [Eqn (4.106) Page 188 - Book]
            # [Eqn (4.105) Page 188 - Book]
            zeta1 = np.cos(np.arctan(Bxp * self.UNLOADED_RADIUS * phi))
        else:
            zeta1 = 1.0

        Cx = self.PCX1 * self.LCX  # (> 0) (4.E11)
        mux = (self.PDX1 + self.PDX2 * dfz) * (1 + self.PPX3 * dpi + self.PPX4 * dpi**2) * (1 - self.PDX3 * gamma**2) * LMUX_star  # (4.E13)
        
        mux[Fz == 0] = 0 # Zero Fz correction

        Dx = mux * Fz * zeta1  # (> 0) (4.E12)
        Kxk = Fz * (self.PKX1 + self.PKX2 * dfz) * np.exp(self.PKX3 * dfz) * (1 + self.PPX1 * dpi + self.PPX2 * dpi**2) * self.LKX  # (= BxCxDx = dFxo / dkx at kappax = 0) (= Cfk) (4.E15)

        # If [Dx = 0] then [sign(0) = 0]. This is done to avoid [Kxk / 0 = NaN] in Eqn 4.E16
        signDx = safe_sign(Dx)

        # (4.E16) [sign(Dx) term explained on page 177]
        Bx = Kxk / (Cx * Dx + self.epsilon * signDx)
        SHx = (self.PHX1 + self.PHX2 * dfz) * self.LHX  # (4.E17)
        SVx = Fz * (self.PVX1 + self.PVX2 * dfz) * self.LVX * LMUX_prime * zeta1  # (4.E18)

        SVx[is_low_speed] = SVx[is_low_speed] * reductionSmooth
        SHx[is_low_speed] = SHx[is_low_speed] * reductionSmooth

        kappax = kappa + SHx  # (4.E10)

        Ex = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz**2) * (1 - self.PEX4 * np.sign(kappax)) * self.LEX  # (<=1) (4.E14)
        
        Ex[Ex > 1] = 1 # Zero Fz correction

        # Pure longitudinal force
        Fx0 = Dx * np.sin(Cx * np.arctan(Bx * kappax - Ex * (Bx * kappax - np.arctan(Bx * kappax)))) + SVx  # (4.E9)

        return Fx0, mux, Kxk

    def calculate_fy0(self, modes: Mode, alpha: Tensor, phi: Tensor, Fz: Tensor, uVcx: Tensor, is_low_speed: torch.BoolTensor, reductionSmooth: Tensor, alpha_star: Tensor, gamma_star: Tensor, LMUY_star: Tensor, Fz0_prime: Tensor, LMUY_prime: Tensor, dfz: Tensor, dpi: Tensor) -> Tensor:
        # Turn slip
        if modes.useTurnSlip:
            r_0 = self.UNLOADED_RADIUS  # Free tyre radius

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

        Kya = self.PKY1 * Fz0_prime * (1 + self.PPY1 * dpi) * (1 - self.PKY3 * np.abs(gamma_star)) * np.sin(self.PKY4 * np.arctan((Fz / Fz0_prime) / (
            (self.PKY2 + self.PKY5 * gamma_star**2) * (1 + self.PPY2 * dpi)))) * zeta3 * self.LKY  # (= ByCyDy = dFyo / dalphay at alphay = 0) (if gamma =0: =Kya0 = CFa) (PKY4=2)(4.E25)
        SVyg = Fz * (self.PVY3 + self.PVY4 * dfz) * gamma_star * self.LKYC * LMUY_prime * zeta2  # (4.E28)

        # MF6.1 and 6.2 equations
        # (=dFyo / dgamma at alpha = gamma = 0) (= CFgamma) (4.E30)
        Kyg0 = Fz * (self.PKY6 + self.PKY7 * dfz) * (1 + self.PPY5 * dpi) * self.LKYC

        # (4.E39) [sign(Kya) term explained on page 177]
        Kya_prime = Kya + self.epsilon * safe_sign(Kya)

        if modes.useTurnSlip:
            # this equation below seems very odd
            Kya0 = self.PKY1 * Fz0_prime * (1 + self.PPY1 * dpi) * np.sin(self.PKY4 * np.arctan(
                (Fz / Fz0_prime) / (self.PKY2 * (1 + self.PPY2 * dpi)))) * zeta3 * self.LKY

            # IMPORTANT NOTE: Explanation of the above equation, Kya0
            # Kya0 is the cornering stiffness when the camber angle is zero
            # (gamma=0) which is again the product of the coefficients By, Cy and
            # Dy at zero camber angle. Information from Kaustub Ragunathan, email:
            # carmaker-service-uk@ipg-automotive.com

            
            # epsilonk is a small factor added to avoid the singularity condition during zero velocity (equation 308, CarMaker reference Manual).
            Kyao_prime = Kya0 + self.epsilon * safe_sign(Kya0)

            CHyp = self.PHYP1  # (>0) [Eqn (4.85) Page 186 - Book]
            DHyp = (self.PHYP2 + self.PHYP3 * dfz) * np.sign(uVcx)  # [Eqn (4.86) Page 186 - Book]
            EHyp = self.PHYP4  # (<=1) [Eqn (4.87) Page 186 - Book]

            EHyp = min(EHyp, 1)
            KyRp0 = Kyg0 / (1-self.epsilon)  # Eqn (4.89)
            # [Eqn (4.88) Page 186 - Book]
            BHyp = KyRp0 / (CHyp * DHyp * Kyao_prime)
            phi_term = BHyp * self.UNLOADED_RADIUS * phi
            SHyp = DHyp * np.sin(CHyp * np.arctan(phi_term - EHyp * (phi_term - np.arctan(phi_term)))) * np.sign(uVcx)  # [Eqn (4.80) Page 185 - Book]

            zeta4 = 1 + SHyp - SVyg / Kya_prime  # [Eqn (4.84) Page 186 - Book]

            SHy = (self.PHY1 + self.PHY2 * dfz) * self.LHY + zeta4 - 1  # (4.E27) [sign(Kya) term explained on page 177]
        else:
            # No turn slip and small camber angles
            # First paragraph on page 178 of the book
            SHy = (self.PHY1 + self.PHY2 * dfz) * self.LHY + ((Kyg0 * gamma_star - SVyg) / Kya_prime)  # (4.E27) [sign(Kya) term explained on page 177]

        SVy = Fz * (self.PVY1 + self.PVY2 * dfz) * self.LVY * LMUY_prime * zeta2 + SVyg  # (4.E29)

        # Low speed model
        SVy[is_low_speed] = SVy[is_low_speed] * reductionSmooth
        SHy[is_low_speed] = SHy[is_low_speed] * reductionSmooth

        alphay = alpha_star + SHy  # (4.E20)
        Cy = self.PCY1 * self.LCY  # (> 0) (4.E21)
        muy = (self.PDY1 + self.PDY2 * dfz) * (1 + self.PPY3 * dpi + self.PPY4 * dpi**2) * (1 - self.PDY3 * gamma_star**2) * LMUY_star  # (4.E23)
        Dy = muy * Fz * zeta2  # (4.E22)
        Ey = (self.PEY1 + self.PEY2 * dfz) * (1 + self.PEY5 * gamma_star**2 - (self.PEY3 + self.PEY4 * gamma_star) * safe_sign(alphay)) * self.LEY  # (<=1)(4.E24)
        
        Ey[Ey > 1] = 1 # Zero Fz correction

        # (4.E26) [sign(Dy) term explained on page 177]
        By = Kya / (Cy * Dy + self.epsilon * safe_sign(Dy))

        Fy0 = Dy * np.sin(Cy * np.arctan(By * alphay - Ey * (By * alphay - np.arctan(By * alphay)))) + SVy  # (4.E19)

        # Backward speed check for alpha_star
        if modes.useAlphaStar:
            Fy0[uVcx < 0] = -Fy0[uVcx < 0]

        muy[Fz == 0] = 0 # Zero Fz correction

        return Fy0, muy, Kya, Kyg0, SHy, SVy, By, Cy, zeta2

    def calculate_mz0(self, modes: Mode, postProInputs: PostProInputs, reductionSmooth, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, Kya, SHy, SVy, By, Cy, zeta2):
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
        Et[Et > 1] = 1 # Zero Fz correction

        t0 = Dt * np.cos(Ct * np.arctan(Bt * alphat - Et * (Bt * alphat - np.arctan(Bt * alphat)))) * np.cos(alpha_prime)  # t(aplhat)(4.E33)

        # Evaluate Fy0 with gamma = 0 and phit = 0
        modes_sub0 = modes
        modes_sub0.useTurnSlip = False

        postProInputs_sub0 = postProInputs
        postProInputs_sub0.gamma = 0.0

        Fyo_sub0, _, _, _, _, _, _, _, _ = self.calculate_fy0(postProInputs_sub0, modes_sub0, reductionSmooth, alpha_star, 0.0, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        Mzo_prime = -t0 * Fyo_sub0  # gamma=phi=0 (4.E32)

        if modes.useTurnSlip:
            zeta0 = 0.0

            # [Eqn (4.102) Page 188 - Book]
            zeta6 = np.cos(np.arctan(self.QBRP1 * r_0 * postProInputs.phi))

            Fy0, muy, _, _, _, _, _, _, _ = self.calculate_fy0(postProInputs, modes, reductionSmooth, alpha_star, 0.0, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

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

            _, Gyk, _ = self.calculate_fy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0, muy, zeta2)

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

    def calculate_fx(self, kappa: Tensor, modes: Mode, alpha_star: Tensor, gamma_star: Tensor, dfz: Tensor, Fx0: Tensor) -> Tensor:
        Cxa = self.RCX1  # (4.E55)
        Exa = self.REX1 + self.REX2 * dfz  # (<= 1) (4.E56)

        # Limits check
        if modes.useLimitsCheck:
            if np.any(Exa > 1):
                print('Exa over limit (>1), Eqn(4.E56)')
        Exa[Exa > 1] = 1 # Zero Fz correction

        Bxa = (self.RBX1 + self.RBX3 * gamma_star**2) * np.cos(np.arctan(self.RBX2 * kappa)) * self.LXAL  # (> 0) (4.E54)

        alphas = alpha_star + self.RHX1  # (4.E53)

        Gxa0 = np.cos(Cxa * np.arctan(Bxa * self.RHX1 - Exa * (Bxa * self.RHX1 - np.arctan(Bxa * self.RHX1))))  # (4.E52)
        Gxa = np.cos(Cxa * np.arctan(Bxa * alphas - Exa * (Bxa * alphas - np.arctan(Bxa * alphas)))) / Gxa0  # (> 0)(4.E51

        Fx = Gxa * Fx0  # (4.E50)

        return Fx

    def calculate_fy(self, postProInputs: PostProInputs, reductionSmooth, modes: Mode, alpha_star, gamma_star, dfz, Fy0, muy, zeta2):
        DVyk = muy * postProInputs.Fz * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma_star) * np.cos(np.arctan(self.RVY4 * alpha_star)) * zeta2  # (4.E67)
        SVyk = DVyk * np.sin(self.RVY5 * np.arctan(self.RVY6 * postProInputs.kappa)) * self.LVYKA  # (4.E66)
        SHyk = self.RHY1 + self.RHY2 * dfz  # (4.E65)
        Eyk = self.REY1 + self.REY2 * dfz  # (<=1) (4.E64)

        if modes.useLimitsCheck:  # Limits check
            if np.any(Eyk > 1):
                print('Eyk over limit (>1), Eqn(4.E64)')
        Eyk[Eyk > 1] = 1 # Zero Fz correction

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

    def calculate_mz(self, postProInputs: PostProInputs, reductionSmooth, modes: Mode, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, alphar, alphat, Kxk, Kya_prime, Fy, Fx, Dr, Cr, Br, Dt, Ct, Bt, Et, SVyk, zeta2):
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
            # Evaluate Fy0 with gamma = 0 and phit  = 0
            Fy0_sub0, muy_sub0, _, _, _, _, _, _, _ = self.calculate_fy0(postProInputs_sub0, modes, reductionSmooth, alpha_star, 0.0, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

            # Evaluate Gyk with phit = 0 (Note: needs to take gamma into
            # account to match TNO)
            _, Gyk_sub0, _ = self.calculate_fy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0_sub0, muy_sub0, zeta2)

            # Note: in the above equation starVar is used instead of
            # starVar_sub0 because it was found a better match with TNO

            Fy_prime = Gyk_sub0 * Fy0_sub0  # (4.E74)
            Mz = -t * Fy_prime + Mzr + s * Fx  # (4.E71) & (4.E72)

        return Mz, t, Mzr


    def doForcesAndMoments(self, postProInputs: PostProInputs, reductionSmooth: Tensor, modes: Mode):

        alpha_star, gamma_star, LMUX_star, LMUY_star, Fz0_prime, alpha_prime, LMUX_prime, LMUY_prime, dfz, dpi = self.calculate_basic(modes, postProInputs)

        Fx0, mux, Kxk = self.calculate_fx0(postProInputs, reductionSmooth, modes, LMUX_star, LMUX_prime, dfz, dpi)

        Fy0, muy, Kya, _, SHy, SVy, By, Cy, zeta2 = self.calculate_fy0(postProInputs, modes, reductionSmooth, alpha_star, gamma_star, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        _, alphar, alphat, Dr, Cr, Br, Dt, Ct, Bt, Et, Kya_prime = self.calculate_mz0(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, Kya, SHy, SVy, By, Cy, zeta2)

        Fx = self.calculate_fx(postProInputs, modes, alpha_star, gamma_star, dfz, Fx0)

        Fy, _, SVyk = self.calculate_fy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0, muy, zeta2)

        Mx = self.calculateMx(postProInputs, dpi, Fy)

        Mz, t, Mzr = self.calculate_mz(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, alphar, alphat, Kxk, Kya_prime, Fy, Fx, Dr, Cr, Br, Dt, Ct, Bt, Et, SVyk, zeta2)

        forces_and_moments = ForceMoments()
        forces_and_moments.Fx = Fx
        forces_and_moments.Fy = Fy
        forces_and_moments.Fz = postProInputs.Fz
        forces_and_moments.Mx = Mx
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

        alpha_star, gamma_star, LMUX_star, LMUY_star, Fz0_prime, alpha_prime, LMUX_prime, LMUY_prime, dfz, dpi = self.calculate_basic(modes, postProInputs)

        Fx0, _, Kxk = self.calculate_fx0(postProInputs, reductionSmooth, modes, LMUX_star, LMUX_prime, dfz, dpi)

        Fy0, muy, Kya, _, SHy, SVy, By, Cy, zeta2 = self.calculate_fy0(postProInputs, modes, reductionSmooth, alpha_star, gamma_star, LMUY_star, Fz0_prime, LMUY_prime, dfz, dpi)

        _, alphar, alphat, Dr, Cr, Br, Dt, Ct, Bt, Et, Kya_prime = self.calculate_mz0(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, Kya, SHy, SVy, By, Cy, zeta2)

        Fx = self.calculate_fx(postProInputs, modes, alpha_star, gamma_star, dfz, Fx0)

        Fy, _, SVyk = self.calculate_fy(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, dfz, Fy0, muy, zeta2)

        Mz, _, _ = self.calculate_mz(postProInputs, reductionSmooth, modes, alpha_star, gamma_star, LMUY_star, alpha_prime, Fz0_prime, LMUY_prime, dfz, dpi, alphar, alphat, Kxk, Kya_prime, Fy, Fx, Dr, Cr, Br, Dt, Ct, Bt, Et, SVyk, zeta2)

        return Fx, Fy, Mz


    def calculate_re(self, postProInputs: PostProInputs, dpi):
        Fz_unlimited = postProInputs.uFz

        # Rename the TIR file variables in the Pacejka style
        r_0 = self.UNLOADED_RADIUS  # Free tyre radius
        Cz0 = self.VERTICAL_STIFFNESS  # Vertical stiffness

        omega = postProInputs.omega  # rotational speed (rad/s)
        # [Eqn (1) Page 2 - Paper] - Centrifugal growth of the free tyre radius
        Romega = r_0 * (self.Q_RE0 + self.Q_V1 * ((omega * r_0) / self.LONGVL)**2)

        # Excerpt from OpenTIRE MF6.1 implementation
        # Date: 2016-12-01
        # Prepared for Marco Furlan/JLR
        # Questions: henning.olsson@calspan.com

        # Nominal stiffness (pressure corrected)
        # [Eqn (5) Page 2 - Paper] - Vertical stiffness adapted for tyre inflation pressure
        Cz = Cz0 * (1 + self.PFZ1 * dpi)

        
        # Omega is one of the inputs
        # rotational speed (rad/s) [eps is added to avoid Romega = 0]
        omega = postProInputs.omega + 1e-9

        # [Eqn (1) Page 2 - Paper] - Centrifugal growth of the free tyre radius
        Romega = r_0 * (self.Q_RE0 + self.Q_V1 * ((omega * r_0) / self.LONGVL)**2)

        Re = Romega - (self.FNOMIN / Cz) * (self.DREFF * np.arctan(self.BREFF * (Fz_unlimited / self.FNOMIN)) +
                                            self.FREFF * (Fz_unlimited / self.FNOMIN))  # Eff. Roll. Radius [Eqn (7) Page 2 - Paper]

        return Re, Romega, omega


    def doExtras(self, postProInputs: PostProInputs, forces_and_moments, varinf):

        dpi = (postProInputs.p - self.NOMPRES) / self.NOMPRES  # [Eqn (4.E2b) Page 177 - Book]

        Re, Romega, omega = self.calculate_re(postProInputs, dpi)

        return Re, omega
