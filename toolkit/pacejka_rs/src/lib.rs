use pyo3::prelude::*;
mod magic_formula;
mod structs;
use magic_formula::*;
use numpy::ndarray::{Array, Array1};
use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArrayDyn;
use pyo3::types::PyBytes;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use structs::*;

#[pyclass(module = "pacejka_rs")]
pub struct PyPacejka {
    pacejka: PacejkaParameters,
    #[pyo3(get, set)]
    upper_s_r: f64,
    #[pyo3(get, set)]
    lower_s_r: f64,
    x: Array1<f64>,
}

#[pymethods]
impl PyPacejka {
    #[new]
    pub fn new(fittyp: i32, x: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let fittyp = match fittyp {
            52 => MFVersion::MF52,
            6 => MFVersion::MF52,
            21 => MFVersion::MF52,
            61 => MFVersion::MF61,
            62 => MFVersion::MF62,
            0 => MFVersion::MF62,
            _ => panic!("Invalid fittyp"),
        };
        let pacejka = PacejkaParameters::new(fittyp, x.as_array())?;
        Ok(PyPacejka {
            pacejka,
            upper_s_r: 0.2,
            lower_s_r: -0.3,
            x: x.as_array().to_owned(),
        })
    }

    #[pyo3(name = "solve_steady_state")]
    pub fn solve_steady_state(
        &self,
        fz: f64,
        alpha: f64,
        kappa: f64,
        p: f64,
        gamma: f64,
        u_vcx: f64,
        omega: f64,
        phi: f64,
        mu: f64,
        flip_sa: bool,
    ) -> (f64, f64, f64) {
        let input = Inputs {
            fz,
            alpha,
            gamma,
            kappa,
            p,
            u_vcx,
            omega,
            phi,
        };
        let options = MFOptions {
            use_limits_check: false,
            use_alpha_star: false,
            use_turn_slip: false,
            use_dynamics: UseMode::SteadyState,
        };
        forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa)
    }

    #[pyo3(name = "s_r_check_maxima")]
    pub fn s_r_check_maxima(
        &self,
        fz: f64,
        alpha: f64,
        kappa: f64,
        d_kappa: f64,
        p: f64,
        gamma: f64,
        u_vcx: f64,
        omega: f64,
        phi: f64,
        mu: f64,
        flip_sa: bool,
    ) -> (f64, bool, f64) {
        let mut input = Inputs {
            fz,
            alpha,
            gamma,
            kappa,
            p,
            u_vcx,
            omega,
            phi,
        };
        let options = MFOptions {
            use_limits_check: false,
            use_alpha_star: false,
            use_turn_slip: false,
            use_dynamics: UseMode::SteadyState,
        };
        let (fx, _, _) = forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
        let fx_sign = f64::signum(fx);
        input.kappa += d_kappa;
        let (fx_up, _, _) = forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
        input.kappa -= 2.0 * d_kappa;
        let (fx_down, _, _) = forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
        if ((fx_sign * fx_up) < (fx_sign * fx) && (fx_sign * fx_down) < (fx_sign * fx))
            || (kappa < self.lower_s_r + d_kappa)
            || (kappa > self.upper_s_r - d_kappa)
        {
            (input.kappa, true, fx)
        } else {
            (input.kappa, false, fx)
        }
    }

    #[pyo3(name = "s_r")]
    pub fn s_r<'py>( &self, py: Python<'py>, fz: f64, alpha: f64, upper: f64, lower: f64, og_upper: f64, og_lower: f64, kappa: f64, prev_kappa: f64, prev_fx: f64, p: f64, gamma: f64, u_vcx: f64, omega: f64, phi: f64, mu: f64, flip_sa: bool, non_driven: bool, fx_target: f64, i: i64,
    ) -> (f64, bool, f64) {
        if (fx_target > 0.0) && (non_driven) {
            let kappa = 0.0;
            let (fx, _, _) =
                self.solve_steady_state(fz, alpha, kappa, p, gamma, u_vcx, omega, phi, mu, flip_sa);
            return (kappa, false, fx);
        }
        if fz <= 0.0 {
            let kappa = 0.0;
            let fx = 0.0;
            return (kappa, false, fx);
        }
        if i > 10 {
            return (kappa, false, prev_fx);
        }
        let d_kappa = 1e-6;
        let b_kappa = 1e-7;
        let mut input = Inputs { fz, alpha, gamma, kappa, p, u_vcx, omega, phi,
        };
        let options = MFOptions { use_limits_check: false, use_alpha_star: false, use_turn_slip: false, use_dynamics: UseMode::SteadyState,
        };
        let (fx, _, _) = forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
        input.kappa += b_kappa;
        let (fx_up, _, _) = forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
        input.kappa -= 2.0 * b_kappa;
        let (fx_down, _, _) = forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
        let max_fxy_mag = 3.0 * fz;
        let dfx = fx_target.max(-max_fxy_mag).min(max_fxy_mag) - fx;
        let d_fx = (fx_up - fx_down) / (2.0 * b_kappa);
        let dd_fx = (fx_up - 2.0 * fx + fx_down) / (b_kappa * b_kappa);
        let mut new_kappa = kappa;
        if (d_fx * d_fx - 4.0 * dfx * dd_fx) < 0.0 {
            new_kappa += dfx / d_fx;
        } else {
            let kappa_one = (-d_fx + f64::sqrt(d_fx * d_fx - 4.0 * dfx * dd_fx)) / (2.0 * dd_fx);
            let kappa_two = (-d_fx - f64::sqrt(d_fx * d_fx - 4.0 * dfx * dd_fx)) / (2.0 * dd_fx);
            if kappa_one.abs() < kappa_two.abs() {
                new_kappa -= kappa_one;
            } else {
                new_kappa -= kappa_two;
            }
        }
        if d_fx < 0.0 {
            new_kappa = (kappa + prev_kappa) / 2.0;
        }
        if ((new_kappa - kappa).abs() < b_kappa) | ((fx_target - fx).abs() < d_kappa) {
            let mid = ((fx - fx_up).signum() == fx.signum()) & ((fx - fx_down).signum() == fx.signum());
            let (new_fx, _, _) = self.solve_steady_state(fz, alpha, new_kappa, p, gamma, u_vcx, omega, phi, mu, flip_sa);
            return (new_kappa, ((new_kappa > upper - d_kappa) | (new_kappa < lower + d_kappa) | mid), new_fx)
        }
        if ((kappa == og_upper) & (new_kappa > og_upper)) | ((kappa == og_lower) & (new_kappa < og_lower)) {
            return (kappa, true, fx);
        }
        new_kappa = new_kappa.max(lower).min(upper);
        let mut ret_kappa = kappa;
        let mut ret_fx = fx;
        let mut ret_upper = upper;
        let mut ret_lower = lower;
        if d_fx < 0.0 {
            if kappa > 0.0 {
                ret_upper = kappa;
            } else {
                ret_lower = kappa;
            }
            ret_kappa = prev_kappa;
            ret_fx = prev_fx;
        }
        self.s_r(
                py, fz, alpha, ret_upper, ret_lower, og_upper, og_lower, new_kappa, ret_kappa, ret_fx, p, gamma, u_vcx, omega, phi, mu, flip_sa, non_driven, fx_target, i + 1,
            )
    }

    pub fn sr_sweep<'py>(
        &self,
        fz: f64,
        alpha: f64,
        kappa: Vec<f64>,
        p: f64,
        gamma: f64,
        u_vcx: f64,
        omega: f64,
        phi: f64,
        mu: f64,
        flip_sa: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let options = MFOptions {
            use_limits_check: false,
            use_alpha_star: false,
            use_turn_slip: false,
            use_dynamics: UseMode::SteadyState,
        };
        let mut fx = Vec::new();
        let mut fy = Vec::new();
        let mut mz = Vec::new();
        for i in 0..kappa.len() {
            let input = Inputs {
                fz,
                alpha,
                gamma,
                kappa: kappa[i],
                p,
                u_vcx,
                omega,
                phi,
            };
            let (fx_new, fy_new, mz_new) =
                forces_and_moments(&self.pacejka, &options, &input, &mu, &flip_sa);
            fx.push(fx_new);
            fy.push(fy_new);
            mz.push(mz_new);
        }
        (fx, fy, mz)
    }

    #[pyo3(name = "solve_sr_sweep")]
    pub fn solve_sr_sweep<'py>(
        &self,
        py: Python<'py>,
        fz: f64,
        alpha: f64,
        kappa: PyReadonlyArray1<'_, f64>,
        p: f64,
        gamma: f64,
        u_vcx: f64,
        omega: f64,
        phi: f64,
        mu: f64,
        flip_sa: bool,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) { // this is really the wrong way to do this but idc
        let (fx, fy, mz) = self.sr_sweep(
            fz,
            alpha,
            kappa.as_array().to_vec(),
            p,
            gamma,
            u_vcx,
            omega,
            phi,
            mu,
            flip_sa,
        );
        (
            Array::from_vec(fx).into_pyarray(py),
            Array::from_vec(fy).into_pyarray(py),
            Array::from_vec(mz).into_pyarray(py),
        )
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pacejka_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPacejka>()?;
    Ok(())
}
