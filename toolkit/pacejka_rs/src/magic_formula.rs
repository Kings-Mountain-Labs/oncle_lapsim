use crate::structs::*;
use std::f64::consts::PI;

fn safe_sign(num: f64) -> f64 {
    if num >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

pub fn forces_and_moments(mf: &PacejkaParameters, options: &MFOptions, input: &Inputs, mu: &f64, flip_sa: &bool) -> (f64, f64, f64) {
    let red_sm = 1.0;
    let mut inputs = input.clone();
    if *flip_sa {
        inputs.alpha *= -1.0;
    }
    let (alpha_star, gamma_star, lmux_star, lmuy_star, fz0_prime, alpha_prime, lmux_prime, lmuy_prime, dfz, dpi) = calculate_basic(mf, &inputs, options);

    let (fx0, _, kxk) = calculate_fx0(mf, &inputs, options, &red_sm, &lmux_star, &lmux_prime, &dfz, &dpi);

    let (fy0, muy, kya, _, shy, svy, by, cy, zeta2,) = calculate_fy0(mf, &inputs, options, &red_sm, &alpha_star, &gamma_star, &lmuy_star, &fz0_prime, &lmuy_prime, &dfz, &dpi);

    let (_, alphar, alphat, dr, cr, br, dt, ct, bt, et, kya_prime,) = calculate_mz0(mf, &inputs, options, &red_sm, &alpha_star, &gamma_star, &lmuy_star, &alpha_prime, &fz0_prime, &lmuy_prime, &dfz, &dpi, &kya, &shy, &svy, &by, &cy, &zeta2);

    let fx = calculate_fx(mf, &inputs, options, &alpha_star, &gamma_star, &dfz, &fx0);

    let (fy, _, svyk) = calculate_fy(mf, &inputs, options, &red_sm, &alpha_star, &gamma_star, &dfz, &fy0, &muy, &zeta2);

    let (mz, _, _) = calculate_mz(mf, &inputs, options, &red_sm, &alpha_star, &gamma_star, &lmuy_star, &alpha_prime, &fz0_prime, &lmuy_prime, &dfz, &dpi, &alphar, &alphat, &kxk, &kya_prime, &fy, &fx, &dr, &cr, &br, &dt, &ct, &bt, &et, &svyk, &zeta2);

    if *flip_sa {
        (fx * mu, -fy * mu, -mz * mu)
    } else {
        (fx * mu, fy * mu, mz * mu)
    }
}

pub fn calculate_basic(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    // Velocities in point S (slip point)
    // [Eqn (4.E5) Page 181 - Book]
    let vsx = -inputs.kappa * inputs.u_vcx.abs();
    // [Eqn (2.12) Page 67 - Book] and [(4.E3) Page 177 - Book]
    let vsy = inputs.alpha.tan() * inputs.u_vcx.abs();

    // Important Note:
    // Due to the ISO sign convention, equation 2.12 does not need a
    // negative sign. The Pacejka book is written in adapted SAE.
    // [Eqn (3.39) Page 102 - Book] -> Slip velocity of the slip point S
    let vs = (vsx * vsx + vsy * vsy).sqrt();

    // Velocities in point C (contact)
    // Assumption from page 67 of the book, paragraph above Eqn (2.11)
    let vcy = vsy;
    // Velocity of the wheel contact centre C, Not described in the book but is the same as [Eqn (3.39) Page 102 - Book]
    let vc = (inputs.u_vcx * inputs.u_vcx + vcy * vcy).sqrt();

    // Effect of having a tire with a different nominal load
    let fz0_prime = mf.lfzo * mf.fnomin; // [Eqn (4.E1) Page 177 - Book]

    // Normalized change in vertical load
    let dfz = (inputs.fz - fz0_prime) / fz0_prime; // [Eqn (4.E2a) Page 177 - Book]
    // Normalized change in inflation pressure
    // [Eqn (4.E2b) Page 177 - Book]
    let dpi = (inputs.p - mf.nompres) / mf.nompres;

    // Use of star (*) definition. Only valid for the book
    // implementation. TNO MF-Tyre does not use this.
    let (alpha_star, gamma_star) = if options.use_alpha_star {
        // [Eqn (4.E3) Page 177 - Book]
        let alpha_star = inputs.alpha.tan() * inputs.u_vcx.signum();
        // [Eqn (4.E4) Page 177 - Book]
        let gamma_star = inputs.gamma.sin();
        (alpha_star, gamma_star)
    } else {
        (inputs.alpha, inputs.gamma)
    };

    // For the aligning torque at high slip angles
    let sign_vc = vc.signum();
    // [Eqn (4.E6a) Page 178 - Book] [sign(Vc) term explained on page 177]
    // [sign(Vc) term explained on page 177]
    let vc_prime = vc + mf.epsilon * sign_vc;

    // [Eqn (4.E6) Page 177 - Book]
    let alpha_prime = (inputs.u_vcx / vc_prime).acos();

    // Slippery surface with friction decaying with increasing (slip) speed
    // [Eqn (4.E7) Page 179 - Book]
    let lmux_star = mf.lmux / (1.0 + mf.lmuv * vs / mf.longvl);
    // [Eqn (4.E7) Page 179 - Book]
    let lmuy_star = mf.lmuy / (1.0 + mf.lmuv * vs / mf.longvl);

    // Digressive friction factor
    // On Page 179 of the book is suggested Amu = 10, but after
    // comparing the use of the scaling factors against TNO, Amu = 1
    // was giving perfect match
    let amu = 1.0;
    // [Eqn (4.E8) Page 179 - Book]
    let lmux_prime = amu * lmux_star / (1.0 + (amu - 1.0) * lmux_star);
    // [Eqn (4.E8) Page 179 - Book]
    let lmuy_prime = amu * lmuy_star / (1.0 + (amu - 1.0) * lmuy_star);

    return (alpha_star, gamma_star, lmux_star, lmuy_star, fz0_prime, alpha_prime, lmux_prime, lmuy_prime, dfz, dpi)
}

fn calculate_fx0(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions, red_sm: &f64, lmux_star: &f64, lmux_prime: &f64, dfz: &f64, dpi: &f64) -> (f64, f64, f64) {

    let zeta1 = if options.use_turn_slip {
        let bxp = mf.pdxp1 * (1.0 + mf.pdxp2 * dfz) * f64::cos(f64::atan(mf.pdxp3 * inputs.kappa)); // [Eqn (4.106) Page 188 - Book]
        // [Eqn (4.105) Page 188 - Book]
        f64::cos(f64::atan(bxp * mf.unloaded_radius * inputs.phi))
    } else {
        1.0
    };

    let cx = mf.pcx1 * mf.lcx;
    let mux = (mf.pdx1 + mf.pdx2 * dfz) * (1.0 + mf.ppx3 * dpi + mf.ppx4 * dpi.powi(2)) * (1.0 - mf.pdx3 * inputs.gamma.powi(2)) * lmux_star;

    let mut dx = mux * inputs.fz * zeta1;
    if inputs.fz == 0.0 {
        dx = 0.0;
    }

    let kxk = inputs.fz * (mf.pkx1 + mf.pkx2 * dfz) * f64::exp(mf.pkx3 * dfz) * (1.0 + mf.ppx1 * dpi + mf.ppx2 * dpi.powi(2)) * mf.lkx;

    let sign_dx = (dx as f64).signum();
    let bx = kxk / (cx * dx + mf.epsilon * sign_dx);
    let shx = (mf.phx1 + mf.phx2 * dfz) * mf.lhx;
    let svx = inputs.fz * (mf.pvx1 + mf.pvx2 * dfz) * mf.lvx * lmux_prime * zeta1;

    // if let Some(is_low_speed) = options.is_low_speed {
    //     if is_low_speed {
    //         svx *= reduction_smooth;
    //         shx *= reduction_smooth;
    //     }
    // }

    let mut kappax = inputs.kappa + shx;
    if options.use_dynamics == UseMode::LinearTransience {
        if inputs.u_vcx < 0.0 {
            kappax *= -1.0;
        }
    }
    
    let mut ex = (mf.pex1 + mf.pex2 * dfz + mf.pex3 * dfz * dfz) * (1.0 - mf.pex4 * kappax.signum()) * mf.lex;  // (<=1) (4.E14)

    if ex > 1.0 {
        ex = 1.0; // Ex[Ex > 1] = 1
    }

    // Pure longitudinal force
    let mut fx0 = dx * f64::sin(cx * f64::atan(bx * kappax - ex * (bx * kappax - f64::atan(bx * kappax)))) + svx;  // (4.E9)

    if options.use_dynamics != UseMode::Nonlinear {  // Backward speed check
        if inputs.u_vcx < 0.0 {
            fx0 *= -1.0;
        }
    }

    (fx0, mux, kxk)
}


fn calculate_fy0(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions, red_sm: &f64, alpha_star: &f64, gamma_star: &f64, lmuy_star: &f64, fz0_prime: &f64, lmuy_prime: &f64, dfz: &f64, dpi: &f64) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    // Turn slip
    let mut zeta2 = 1.0;
    let mut zeta3 = 1.0;
    let mut shy = 1.0;
    if options.use_turn_slip {
        let r_0 = mf.unloaded_radius;  // Free tyre radius

        let alpha = inputs.alpha;
        let phi = inputs.phi;

        // [Eqn (4.79) Page 185 - Book]
        zeta3 = f64::cos(f64::atan(mf.pkyp1 * r_0.powi(2) * phi.powi(2)));

        // [Eqn (4.78) Page 185 - Book]
        let byp = mf.pdyp1 * (1.0 + mf.pdyp2 * dfz) * f64::cos(f64::atan(mf.pdyp3 * f64::tan(alpha)));

        // [Eqn (4.77) Page 184 - Book]
        zeta2 = f64::cos(f64::atan(byp * (r_0 * f64::abs(phi) + mf.pdyp4 * f64::sqrt(r_0 * f64::abs(phi)))));
    }

    let kya = mf.pky1 * fz0_prime * (1.0 + mf.ppy1 * dpi) * (1.0 - mf.pky3 * gamma_star.abs()) * f64::sin(mf.pky4 * f64::atan((inputs.fz / fz0_prime) / ((mf.pky2 + mf.pky5 * gamma_star.powi(2)) * (1.0 + mf.ppy2 * dpi)))) * zeta3 * mf.lky;  // (= ByCyDy = dFyo / dalphay at alphay = 0) (if gamma =0: =Kya0 = CFa) (PKY4=2)(4.E25)
    let svyg = inputs.fz * (mf.pvy3 + mf.pvy4 * dfz) * gamma_star * mf.lkyc * lmuy_prime * zeta2;  // (4.E28)

    // MF6.1 and 6.2 equations
    // (=dFyo / dgamma at alpha = gamma = 0) (= CFgamma) (4.E30)
    let kyg0 = inputs.fz * (mf.pky6 + mf.pky7 * dfz) * (1.0 + mf.ppy5 * dpi) * mf.lkyc;

    if options.use_turn_slip {
        // this equation below seems very odd
        let kya0 = mf.pky1 * fz0_prime * (1.0 + mf.ppy1 * dpi) * f64::sin(mf.pky4 * f64::atan((inputs.fz / fz0_prime) / (mf.pky2 * (1.0 + mf.ppy2 * dpi)))) * zeta3 * mf.lky;

        // IMPORTANT NOTE: Explanation of the above equation, Kya0
        // Kya0 is the cornering stiffness when the camber angle is zero
        // (gamma=0) which is again the product of the coefficients By, Cy and
        // Dy at zero camber angle. Information from Kaustub Ragunathan, email:
        // carmaker-service-uk@ipg-automotive.com

        // (4.E39) [sign(Kya) term explained on page 177]
        let kya_prime = kya + mf.epsilon * safe_sign(kya);
        // epsilonk is a small factor added to avoid the singularity condition during zero velocity (equation 308, CarMaker reference Manual).
        let kyao_prime = kya0 + mf.epsilon * safe_sign(kya0);

        let chyp = mf.phyp1;  // (>0) (4.E40)
        let dhyp = (mf.phyp2 + mf.phyp3 * dfz) *   inputs.u_vcx.signum(); // [Eqn (4.86) Page 186 - Book]
        let mut ehyp = mf.phyp4;

        if options.use_limits_check {
            if ehyp > 1.0 {
                ehyp = 1.0;
            }
        }

        let kyrp0 = kyg0 / (1.0 - mf.epsilon); // Eqn (4.89)
        // [Eqn (4.88) Page 186 - Book]
        let bhyp = kyrp0 / (chyp * dhyp * kyao_prime);
        let shyp = dhyp * f64::sin(chyp * f64::atan(bhyp * mf.unloaded_radius * inputs.phi - ehyp * (bhyp * mf.unloaded_radius * inputs.phi - f64::atan(bhyp * mf.unloaded_radius * inputs.phi)))) * inputs.u_vcx.signum();  // [Eqn (4.80) Page 185 - Book]

        let zeta4 = 1.0 + shyp - svyg / kya_prime;  // [Eqn (4.84) Page 186 - Book]

        shy = (mf.phy1 + mf.phy2 * dfz) * mf.lhy + zeta4 - 1.0;  // (4.E27) [sign(Kya) term explained on page 177]
    } else {
        // No turn slip and small camber angles
        // First paragraph on page 178 of the book
        shy = (mf.phy1 + mf.phy2 * dfz) * mf.lhy + ((kyg0 * gamma_star - svyg) / (kya + mf.epsilon * safe_sign(kya)))  // (4.E27) [sign(Kya) term explained on page 177]
    }

    let svy = inputs.fz * (mf.pvy1 + mf.pvy2 * dfz) * mf.lvy * lmuy_prime * zeta2 + svyg;  // (4.E29)

    // Low speed model
    // if type(modes.isLowSpeed) == np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:
    //     SVy[modes.isLowSpeed] = SVy[modes.isLowSpeed] * red_sm
    //     SHy[modes.isLowSpeed] = SHy[modes.isLowSpeed] * red_sm

    let alphay = alpha_star + shy;  // (4.E20)
    let cy = mf.pcy1 * mf.lcy;  // (> 0) (4.E21)
    let mut muy = (mf.pdy1 + mf.pdy2 * dfz) * (1.0 + mf.ppy3 * dpi + mf.ppy4 * dpi * dpi) * (1.0 - mf.pdy3 * gamma_star * gamma_star) * lmuy_star;  // (4.E23)
    let dy = muy * inputs.fz * zeta2;  // (4.E22)
    let mut ey = (mf.pey1 + mf.pey2 * dfz) * (1.0 + mf.pey5 * gamma_star * gamma_star - (mf.pey3 + mf.pey4 * gamma_star) * safe_sign(alphay)) * mf.ley;  // (<=1)(4.E24)

    // Limits check
    if ey > 1.0 {
        ey = 1.0;
    }
    // (4.E26) [sign(Dy) term explained on page 177]
    let by = kya / (cy * dy + mf.epsilon * safe_sign(dy));

    let mut fy0 = dy * f64::sin(cy * f64::atan(by * alphay - ey * (by * alphay - f64::atan(by * alphay)))) + svy;  // (4.E19)

    // Backward speed check for alpha_star
    if options.use_alpha_star {
        if inputs.u_vcx < 0.0 {
            fy0 *= -1.0;
        }
    }

    // Zero fz correction
    if fy0 == 0.0 {
        muy = 0.0;
    }
 
    (fy0, muy, kya, kyg0, shy, svy, by, cy, zeta2)
}

fn calculate_mz0(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions, red_sm: &f64, alpha_star: &f64, gamma_star: &f64, lmuy_star: &f64, alpha_prime: &f64, fz0_prime: &f64, lmuy_prime: &f64, dfz: &f64, dpi: &f64, kya: &f64, shy: &f64, svy: &f64, by: &f64, cy: &f64, zeta2: &f64) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let sht = mf.qhz1 + mf.qhz2 * dfz + (mf.qhz3 + mf.qhz4 * dfz) * gamma_star;  // (4.E35)

    let signkya = safe_sign(kya * 1.0);

    // (4.E39) [sign(Kya) term explained on page 177]
    let kya_prime = kya + mf.epsilon * signkya;
    let shf = shy + svy / kya_prime;  // (4.E38)
    let alphar = alpha_star + shf;  // = alphaf (4.E37)
    let alphat = alpha_star + sht;  // (4.E34)

    let mut zeta5 = 1.0;
    if options.use_turn_slip {
        // [Eqn (4.91) Page 186 - Book]
        zeta5 = f64::cos(f64::atan(mf.qdtp1 * mf.unloaded_radius * inputs.phi));
    }

    // Dt0 = fz * (R0 / fz0_prime) * (QDZ1 + QDZ2 * dfz) * (1 - PPZ1 * dpi) *  LTR * sign(Vcx) // (4.E42)
    // Dt = Dt0 * (1 + QDZ3 * abs(gamma_star) + QDZ4 * gamma_star**2) * zeta5 // (4.E43)
        
    // IMPORTANT NOTE: The above original equation (4.E43) was not matching the
    // TNO solver. The coefficient Dt affects the pneumatic trail (t) and the
    // mf aligning torque (Mz).
    // It was observed that when negative inclination angles where used as
    // inputs, there was a discrepancy between the TNO solver and mfeval.
    // This difference comes from the term QDZ3, that in the original equation
    // is multiplied by abs(gamma_star). But in the paper the equation is
    // different and the abs() term is not written. Equation (A60) from the
    // paper resulted into a perfect match with TNO.
    // Keep in mind that the equations from the paper don't include turn slip
    // effects. The term zeta5 has been added although it doesn't appear in the
    // paper.

    // Paper definition:
    let dt = (mf.qdz1 + mf.qdz2 * dfz) * (1.0 - mf.ppz1 * dpi) * (1.0 + mf.qdz3 * inputs.gamma + mf.qdz4 * inputs.gamma * inputs.gamma) * inputs.fz * (mf.unloaded_radius / fz0_prime) * mf.ltr * zeta5;  // (A60)

    // Bt = (QBZ1 + QBZ2 * dfz + 
            // print('Ex over limit (>1), Eqn(4.E14)')QBZ3 * dfz**2) * (1 + QBZ5 * abs(gamma_star) + QBZ6 * gamma_star**2) * LKY / LMUY_star #(> 0)(4.E40)
        
    // IMPORTANT NOTE: In the above original equation (4.E40) it is used the
    // parameter QBZ6, which doesn't exist in the standard TIR files. Also note
    // that on page 190 and 615 of the book a full set of parameters is given
    // and QBZ6 doesn't appear.
    // The equation has been replaced with equation (A58) from the paper.

    // Paper definition:
    let bt = (mf.qbz1 + mf.qbz2 * dfz + mf.qbz3 * dfz * dfz) * (1.0 + mf.qbz4 * inputs.gamma + mf.qbz5 * inputs.gamma.abs()) * mf.lky / lmuy_star;  // (> 0) (A58)
    let ct = mf.qcz1;  // (> 0) (4.E41)
    let mut et = (mf.qez1 + mf.qez2 * dfz + mf.qez3 * dfz * dfz) * (1.0 + (mf.qez4 + mf.qez5 * gamma_star) * (2.0 / PI) * f64::atan(bt * ct * alphat));  // (<=1) (4.E44)

    // Limits check
    if et > 1.0 {
        et = 1.0;
    }

    let t0 = dt * f64::cos(ct * f64::atan(bt * alphat - et * (bt * alphat - f64::atan(bt * alphat)))) * f64::cos(alpha_prime * 1.0);  // t(aplhat)(4.E33)

        // Evaluate Fy0 with gamma = 0 and phit = 0
    let mut options_sub0 = options.clone();
    options_sub0.use_turn_slip = false;

    let mut inputs_sub0 = inputs.clone();
    inputs_sub0.gamma = 0.0;

    let (fyo_sub0, _, _, _, _, _, _, _, _,) = calculate_fy0(mf, &inputs_sub0, &options_sub0, red_sm, alpha_star, &0.0, lmuy_star, fz0_prime, lmuy_prime, dfz, dpi);

    let mzo_prime = -t0 * fyo_sub0;  // gamma=phi=0 (4.E32)

    let mut zeta0 = 1.0;
    let mut zeta6 = 1.0;
    let mut zeta7 = 1.0;
    let mut zeta8 = 1.0;
    if options.use_turn_slip {
        zeta0 = 0.0;

        // [Eqn (4.102) Page 188 - Book]
        zeta6 = f64::cos(f64::atan(mf.qbrp1 * mf.unloaded_radius * inputs.phi));

        let (fy0, muy, _, _, _, _, _, _, _,) = calculate_fy0(mf, inputs, options, red_sm, alpha_star, &0.0, lmuy_star, fz0_prime, lmuy_prime, dfz, dpi);

        let mut mzp_inf = mf.qcrp1 * muy.abs() * mf.unloaded_radius * inputs.fz * f64::sqrt(inputs.fz / fz0_prime) * mf.lmp;  // [Eqn (4.95) Page 187 - Book]

        if mzp_inf < 0.0 {
            mzp_inf = 1e-6;  // Mzp_inf should be always > 0
        }

        let cdrp = mf.qdrp1;  // (>0) [Eqn (4.96) Page 187 - Book]
        // [Eqn (4.94) Page 187 - Book]
        let ddrp = mzp_inf / f64::sin(0.5 * PI * cdrp);
        let kzgr0 = inputs.fz * mf.unloaded_radius * (mf.qdz8 * mf.qdz9 * dfz + (mf.qdz10 + mf.qdz11 * dfz * inputs.gamma.abs())) * mf.lkzc;  // [Eqn (4.99) Page 187 - Book]

        // Eqn from the manual
        let bdrp = kzgr0 / (cdrp * ddrp * (1.0 - mf.epsilon));
        // Eqn from the manual
        let drp = ddrp * f64::sin(cdrp * f64::atan(bdrp * mf.unloaded_radius * inputs.phi));

        let (_, gyk, _) = calculate_fy(mf, inputs, options, &red_sm, &alpha_star, &gamma_star, &dfz, &fy0, &muy, &zeta2);

        let mzp90 = mzp_inf * (2.0 / PI) * f64::atan(mf.qcrp2 * mf.unloaded_radius * inputs.phi.abs()) * gyk;  // [Eqn (4.103) Page 188 - Book]

        zeta7 = (2.0 / PI) * f64::acos(mzp90 / drp.abs());  // Eqn from the manual
        zeta8 = 1.0 + drp;
    }
    let dr = inputs.fz * mf.unloaded_radius * ((mf.qdz6 + mf.qdz7 * dfz) * mf.lres * zeta2 + ((mf.qdz8 + mf.qdz9 * dfz) * (1.0 + mf.ppz2 * dpi) + (mf.qdz10 + mf.qdz11 * dfz) * gamma_star.abs()) * gamma_star * mf.lkzc * zeta0) * lmuy_star * inputs.u_vcx.signum();// * f64::cos(alpha_star * 1.0) + zeta8 - 1.0;  // (4.E47)
    let br = (mf.qbz9 * mf.lky / lmuy_star + mf.qbz10 * by * cy) * zeta6;  // preferred: qBz9 = 0 (4.E45)
    let cr = zeta7;  // (4.E46)
    let mzr0 = dr * f64::cos(cr * f64::atan(br * alphar)) * f64::cos(alpha_prime * 1.0);  // =Mzr(alphar)(4.E36)
    let mz0 = mzo_prime + mzr0;  // (4.E31)

    (mz0, alphar, alphat, dr, cr, br, dt, ct, bt, et, kya_prime)
}

fn calculate_fx(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions, alpha_star: &f64, gamma_star: &f64, dfz: &f64, fx0: &f64) -> f64 {
    let cxa = mf.rcx1;  // (4.E55)
    let mut exa = mf.rex1 + mf.rex2 * dfz;  // (<= 1) (4.E56)

    // Limits check
    if exa > 1.0 {
        exa = 1.0;
    }
    
    let bxa = (mf.rbx1 + mf.rbx3 * gamma_star * gamma_star) * f64::cos(f64::atan(mf.rbx2 * inputs.kappa)) * mf.lxal;  // (> 0) (4.E54)

    let alphas = alpha_star + mf.rhx1;  // (4.E53)

    let gxa0 = f64::cos(cxa * f64::atan(bxa * mf.rhx1 - exa * (bxa * mf.rhx1 - f64::atan(bxa * mf.rhx1))));  // (4.E52)
    let gxa = f64::cos(cxa * f64::atan(bxa * alphas - exa * (bxa * alphas - f64::atan(bxa * alphas)))) / gxa0;  // (> 0)(4.E51

    let fx = gxa * fx0;  // (4.E50)

    fx
}

fn calculate_fy(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions, red_sm: &f64, alpha_star: &f64, gamma_star: &f64, dfz: &f64, fy0: &f64, muy: &f64, zeta2: &f64) -> (f64, f64, f64) {
    let dvyk = muy * inputs.fz * (mf.rvy1 + mf.rvy2 * dfz + mf.rvy3 * gamma_star) * f64::cos(f64::atan(mf.rvy4 * alpha_star)) * zeta2;  // (4.E67)
    let svyk = dvyk * f64::sin(mf.rvy5 * f64::atan(mf.rvy6 * inputs.kappa)) * mf.lvyka;  // (4.E66)
    let shyk = mf.rhy1 + mf.rhy2 * dfz;  // (4.E65)
    let mut eyk = mf.rey1 + mf.rey2 * dfz;  // (<=1) (4.E64)

    // Limits check
    if eyk > 1.0 {
        eyk = 1.0;
    }

    let cyk = mf.rcy1;  // (4.E63)
    let byk = (mf.rby1 + mf.rby4 * gamma_star * gamma_star) * f64::cos(f64::atan(mf.rby2 * (alpha_star - mf.rby3))) * mf.lyka;  // (> 0) (4.E62)
    let kappas = inputs.kappa + shyk;  // (4.E61)

    let gyk0 = f64::cos(cyk * f64::atan(byk * shyk - eyk * (byk * shyk - f64::atan(byk * shyk))));  // (4.E60)
    let gyk = f64::cos(cyk * f64::atan(byk * kappas - eyk * (byk * kappas - f64::atan(byk * kappas)))) / gyk0;  // (> 0)(4.E59)

        // if type(modes.isLowSpeed) == np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:  // If we are using the lowspeed mode and there are any lowspeed points we need to apply the reduction
            // SVyk[modes.isLowSpeed] = SVyk[modes.isLowSpeed] * reductionSmooth

    let fy = gyk * fy0 + svyk;  // (4.E58)

    (fy, gyk, svyk)
}

fn calculate_mz(mf: &PacejkaParameters, inputs: &Inputs, options: &MFOptions, red_sm: &f64, alpha_star: &f64, gamma_star: &f64, lmuy_star: &f64, alpha_prime: &f64, fz0_prime: &f64, lmuy_prime: &f64, dfz: &f64, dpi: &f64, alphar: &f64, alphat: &f64, kxk: &f64, kya_prime: &f64, fy: &f64, fx: &f64, dr: &f64, cr: &f64, br: &f64, dt: &f64, ct: &f64, bt: &f64, et: &f64, svyk: &f64, zeta2: &f64) -> (f64, f64, f64) {
    // alphar_eq = sqrt(alphar**2+(Kxk / Kya_prime)**2 * kappa**2) * sign(alphar) // (4.E78)
    // alphat_eq = sqrt(alphat**2+(Kxk / Kya_prime)**2 * kappa**2) * sign(alphat) // (4.E77)
    // s = R0 * (SSZ1 + SSZ2 * (Fy / fz0_prime) + (SSZ3 + SSZ4 * dfz) * gamma_star) * LS // (4.E76)

    // IMPORTANT NOTE: The equations 4.E78 and 4.E77 are not used due to small
    // differences discovered at negative camber angles with the TNO solver.
    // Instead equations A54 and A55 from the paper are used.
        
    // IMPORTANT NOTE: The coefficient "s" (Equation 4.E76) determines the
    // effect of Fx into Mz. The book uses "fz0_prime" in the formulation,
    // but the paper uses "fz0". The equation (A56) from the paper has a better
    // correlation with TNO.
    let alphar_eq = f64::atan(f64::sqrt(f64::tan(alphar * 1.0).powi(2) + (kxk / kya_prime).powi(2) * inputs.kappa * inputs.kappa)) * alphar.signum();  // (A54)
    let alphat_eq = f64::atan(f64::sqrt(f64::tan(alphat * 1.0).powi(2) + (kxk / kya_prime).powi(2) * inputs.kappa * inputs.kappa)) * alphat.signum();  // (A55)
    let s = mf.unloaded_radius * (mf.ssz1 + mf.ssz2 * (fy / mf.fnomin) + (mf.ssz3 + mf.ssz4 * dfz) * inputs.gamma) * mf.ls;  // (A56)
    let mzr = dr * f64::cos(cr * f64::atan(br * alphar_eq));  // (4.E75)

    // Evaluate Fy and Fy0 with gamma = 0 and phit = 0
    let mut inputs_sub0 = inputs.clone();
    inputs_sub0.gamma = 0.0;

    // Evaluate Fy0 with gamma = 0 and phit  = 0
    let (fy0_sub0, muy_sub0, _, _, _, _, _, _, _,) = calculate_fy0(mf, &inputs_sub0, options, red_sm, alpha_star, &0.0, lmuy_star, fz0_prime, lmuy_prime, dfz, dpi);

    // Evaluate Gyk with phit = 0 (Note: needs to take gamma into
    // account to match TNO)
    let (_, gyk_sub0, _) = calculate_fy(mf, inputs, options, red_sm, alpha_star, gamma_star, dfz, &fy0_sub0, &muy_sub0, zeta2);

    // Note: in the above equation starVar is used instead of
    // starVar_sub0 because it was found a better match with TNO

    let fy_prime = gyk_sub0 * fy0_sub0;  // (4.E74)
    let t = dt * f64::cos(ct * f64::atan(bt * alphat_eq - et * (bt * alphat_eq - f64::atan(bt * alphat_eq)))) * f64::cos(alpha_prime * 1.0) * mf.lfzo;  // (4.E73)

    // IMPORTANT NOTE: the above equation does not contain LFZO in any written source, but "t"
    // is multiplied by LFZO in the TNO dteval function. This has been empirically discovered.

    // if type(modes.isLowSpeed) == np.ndarray and np.count_nonzero(modes.isLowSpeed) > 0:
        // t[modes.isLowSpeed] = t[modes.isLowSpeed] * reductionSmooth
        // Mzr[modes.isLowSpeed] = Mzr[modes.isLowSpeed] * reductionSmooth

    // MF6.1 and 6.2 equations
    let mut mz = -1.0 * t * fy_prime + mzr + s * fx;  // (4.E71) & (4.E72)
    if mf.fittyp == MFVersion::MF52 {  // Check MF version
        // MF5.2 equations
        // From the MF-Tyre equation manual
        mz = -1.0 * t * (fy - svyk) + mzr + s * fx;
    }

    (mz, t, mzr)
}